import os
import json
import httpx
import base64
import asyncio
import logging
import re
from copy import deepcopy
from pydantic import BaseModel
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from structai import str2dict


def color_text(text, color="default", style="normal"):
    """
    Return ANSI-colored text for terminal output.

    Args:
        color: One of red/green/yellow/blue/magenta/cyan/white/default (or single letter).
        style: One of normal/bold/underline.
    """
    colors = {
        "default": 39,
        "black": 30, "k": 30,
        "red": 31, "r": 31,
        "green": 32, "g": 32,
        "yellow": 33, "y": 33,
        "blue": 34, "b": 34,
        "magenta": 35, "m": 35,
        "cyan": 36, "c": 36,
        "white": 97, "w": 97,
    }

    styles = {
        "normal": 0,
        "bold": 1,
        "underline": 4,
    }

    color_code = colors.get(color, 39)
    style_code = styles.get(style, 0)
    return f"\033[{style_code};{color_code}m{text}\033[0m"


def encode_file_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def pydantic_model_to_example(model: BaseModel, schema: dict = None, depth: int = 0) -> dict:
    """Convert a Pydantic model to an example dict for use in prompt instructions."""
    if schema is None:
        schema = model.model_json_schema()

    # Handle $defs for nested models
    defs = schema.get('$defs', {})

    def resolve_ref(ref_path: str):
        """Resolve a $ref to its definition."""
        if ref_path.startswith('#/$defs/'):
            def_name = ref_path.split('/')[-1]
            return defs.get(def_name, {})
        return {}

    def build_example(properties: dict, current_depth: int = 0):
        """Recursively build example from properties."""
        example = {}

        for field_name, field_info in properties.items():
            field_type = field_info.get('type')
            description = field_info.get('description', '')

            # Handle $ref (nested Pydantic models)
            if '$ref' in field_info:
                ref_schema = resolve_ref(field_info['$ref'])
                if ref_schema and current_depth < 3:  # Prevent infinite recursion
                    nested_props = ref_schema.get('properties', {})
                    example[field_name] = build_example(nested_props, current_depth + 1)
                else:
                    example[field_name] = {"key": "value"}
            elif field_type == 'string':
                example[field_name] = f"<{description if description else field_name}>"
            elif field_type == 'integer':
                example[field_name] = 0
            elif field_type == 'number':
                example[field_name] = 0.0
            elif field_type == 'boolean':
                example[field_name] = True
            elif field_type == 'array':
                items = field_info.get('items', {})
                if '$ref' in items:
                    # Handle array of nested models
                    ref_schema = resolve_ref(items['$ref'])
                    if ref_schema and current_depth < 3:
                        nested_props = ref_schema.get('properties', {})
                        example[field_name] = [build_example(nested_props, current_depth + 1)]
                    else:
                        example[field_name] = [{"key": "value"}]
                elif items.get('type') == 'string':
                    example[field_name] = ["item1", "item2"]
                elif items.get('type') == 'integer':
                    example[field_name] = [1, 2]
                elif items.get('type') == 'number':
                    example[field_name] = [1.0, 2.0]
                elif items.get('type') == 'object':
                    example[field_name] = [{"key": "value"}]
                else:
                    example[field_name] = []
            elif field_type == 'object':
                example[field_name] = {"key": "value"}
            else:
                example[field_name] = None

        return example

    properties = schema.get('properties', {})
    return build_example(properties, depth)


class Agent:
    def __init__(
        self,
        name: str = "LLMAgent",
        model_settings: dict = {
            "model": "gpt-4.1-mini",
            "model_provider": "openai",
        },
        api_key: str = None,
        base_url: str = None,
        system_prompt: str = "",
        tools: list = [],
        tool_runtime_args: dict = {},
        response_format: BaseModel = None,
        http_proxy: str = None,
        logger: logging.Logger = None,
        verbose: bool = False,
        log_input: bool = False,
        max_retries: int = 3,
        max_agent_iterations: int = 20,
        max_tool_iterations: int = 15,
        tool_iter_exceed_handler: str = "error_message",  # "raise" or "error_message"
        handle_tool_calls: bool = True,
        response_timeout: float = 180,
    ):
        """
        Initialize an LLM-based agent.

        Args:
            name: Display name for this agent instance.
            model_settings: Dict passed to LangChain's init_chat_model (must include 'model').
            api_key: API key. Defaults to the LLM_API_KEY environment variable.
            base_url: Base URL of the OpenAI-compatible API. Defaults to LLM_BASE_URL
                      env var or the official OpenAI URL.
            system_prompt: System message prepended to every conversation.
            tools: List of LangChain tools to bind to the model.
            tool_runtime_args: Extra kwargs injected into specific tool calls at runtime,
                               keyed by tool name.
            response_format: Optional Pydantic model for structured JSON output.
            http_proxy: Optional HTTP proxy URL.
            logger: Optional Python logger; if set, verbose is forced to True.
            verbose: Print agent events to stdout when True.
            log_input: Also log the user's input message when verbose.
            max_retries: Number of LLM call retries on transient errors.
            max_agent_iterations: Hard limit on total agentic turns before raising.
            max_tool_iterations: Soft limit on tool calls before switching to
                                 error_message or raising (per tool_iter_exceed_handler).
            tool_iter_exceed_handler: "error_message" (default) sends a stop message
                                      to the model; "raise" raises RuntimeError.
            handle_tool_calls: Whether to automatically handle tool calls in chat().
            response_timeout: Seconds to wait for a single LLM response (None = no limit).
        """
        self.name = name
        self.model_settings = model_settings
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.system_prompt = system_prompt
        self.tools = tools
        self.response_format = response_format
        self.http_proxy = http_proxy
        self.logger = logger

        if logger is not None:
            verbose = True
        self.verbose = verbose
        self.log_input = log_input

        if tool_iter_exceed_handler not in ["raise", "error_message"]:
            raise ValueError(f"tool_iter_exceed_handler must be 'raise' or 'error_message'.")

        if response_format is not None:
            if not issubclass(response_format, BaseModel):
                raise ValueError(f"response_format must be a Pydantic BaseModel subclass.")

        self.max_retries = max_retries
        self.max_agent_iterations = max_agent_iterations
        self.max_tool_iterations = max_tool_iterations
        self.max_parse_retries = 3
        self.tool_iter_exceed_handler = tool_iter_exceed_handler
        self.handle_tool_calls = handle_tool_calls
        self.response_timeout = response_timeout

        self.binded_tools = False
        self.llm = self.initialize_llm()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_runtime_args = tool_runtime_args

        self.messages = []
        if self.system_prompt:
            self.messages.append(SystemMessage(content=self.system_prompt))

        self.event_counter = 0
        self.msg_mark_idx = None

    def initialize_llm(self):
        params = self.model_settings.copy()
        params["api_key"] = self.api_key
        params["base_url"] = self.base_url
        params["model_provider"] = params.get("model_provider", "openai")

        if self.http_proxy:
            params["http_client"] = httpx.Client(proxy=self.http_proxy)
            params["http_async_client"] = httpx.AsyncClient(proxy=self.http_proxy)

        llm: BaseChatModel = init_chat_model(**params)

        if self.tools:
            llm = llm.bind_tools(self.tools)
            self.binded_tools = True

        return llm

    def log(self, message: str, color: str = None, style: str = None, flush: bool = False, verbose: bool = False, count: bool = True):

        if count:
            self.event_counter += 1

        verbose = verbose or self.verbose
        if not verbose:
            return

        if color or style:
            message = color_text(message, color=color or "default", style=style or "normal")

        if self.logger:
            self.logger.info(message)
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()
        else:
            print(message, flush=flush)

    async def _get_response(self):
        cur_try = 0

        while True:
            try:
                if self.response_timeout is not None:
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(self.messages),
                        timeout=self.response_timeout
                    )
                else:
                    response = await self.llm.ainvoke(self.messages)

                return response

            except Exception as e:
                cur_try += 1

                if cur_try >= self.max_retries:
                    raise RuntimeError(f"{self.name}: Max retries exceeded when getting response. Error: {e}")

                self.log(f"{self.name}: Error when getting response [{cur_try}/{self.max_retries}]: {e}",
                         color='r', flush=True, verbose=True, count=False)
                await asyncio.sleep(3)

    async def _invoke_tool(self, tool_call, iter_exceeded: bool = False):
        name = tool_call['name']
        args = tool_call['args']
        tool_log = f"Invoking `{name}` with `{args}`"
        self.log(
            f"({color_text(self.name, 'g')}) {self.event_counter}: Agent took action:\n{color_text(tool_log, 'y')}", flush=True
        )

        if iter_exceeded and self.tool_iter_exceed_handler == "error_message":
            tool_response = f"Error: The maximum number of tool calls exceeded. DO NOT call any more tools. Provide final answer based on the information you have."

        else:
            args = args.copy()
            args.update(self.tool_runtime_args.get(name, {}))

            tool = self.tools_by_name.get(name)
            tool_response = await tool.ainvoke(args)

        self.log(
            f"({color_text(self.name, 'g')}) {self.event_counter}: Tool `{name}` returned:\n{color_text(tool_response, 'y')}\n", flush=True
        )

        return tool_call['id'], tool_response

    async def chat(self,
                   input_msg: str | HumanMessage,
                   handle_tool_calls: bool = None,
                   input_images: str | list[str] = None,
                   input_files: str | list[str] = None,
                   log_input: bool = False) -> AIMessage:

        # Add response format instructions to input message if needed
        if self.response_format is not None:
            format_instruction = self._get_format_instruction()
            if isinstance(input_msg, str):
                input_msg = input_msg + "\n\n" + format_instruction
            elif isinstance(input_msg, HumanMessage):
                if isinstance(input_msg.content, str):
                    input_msg.content = input_msg.content + "\n\n" + format_instruction
                else:
                    raise ValueError(f"When response_format is set, input_msg content must be a string.")

        if log_input or self.log_input:
            input_text = input_msg if isinstance(input_msg, str) else input_msg.content
            if input_images is not None:
                input_text += f"\n[Input Images: {input_images}]"
            if input_files is not None:
                input_text += f"\n[Input Files: {input_files}]"
            self.log(f"({color_text(self.name, 'g')}) {self.event_counter}: Agent started with input:\n{color_text(input_text, 'b')}\n", flush=True)
        else:
            self.log(f"({color_text(self.name, 'g')}) {self.event_counter}: Agent started", flush=True)

        input_images_and_files = []
        if input_images is not None:
            img_type_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png'}

            input_images = [input_images] if isinstance(input_images, str) else input_images
            input_images_and_files.extend([
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": f"image/{img_type_map[img_path.split('.')[-1].lower()]}",
                    "data": encode_file_base64(img_path),
                }
                for img_path in input_images
            ])
        if input_files is not None:
            input_files = [input_files] if isinstance(input_files, str) else input_files

            # Only PDF files are currently supported
            all_pdf = all([file_path.split('.')[-1].lower() == 'pdf' for file_path in input_files])
            if not all_pdf:
                raise ValueError(f"Only PDF files are supported for input_files.")

            input_images_and_files.extend([
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "application/pdf",
                    "data": encode_file_base64(file_path),
                    "filename": os.path.basename(file_path),
                }
                for file_path in input_files
            ])

        if isinstance(input_msg, str):
            if input_images_and_files:
                input_msg = HumanMessage(
                    input_images_and_files + [{
                        "type": "text",
                        "text": input_msg
                    }]
                )
            else:
                input_msg = HumanMessage(content=input_msg)
        elif isinstance(input_msg, HumanMessage):
            if input_images_and_files:
                if not isinstance(input_msg.content, list):
                    raise ValueError(f"When input_images or input_files are provided, input_msg must be a string or a HumanMessage with content as a list.")
                input_msg.content = input_msg.content + input_images_and_files
        else:
            raise ValueError(f"input_msg must be a string or a HumanMessage.")

        self.messages.append(input_msg)

        cur_iteration = 0
        parse_retry_count = 0
        if handle_tool_calls is None:
            handle_tool_calls = self.handle_tool_calls

        while True:
            response = await self._get_response()
            self.messages.append(response)

            if response.content:
                self.log(
                    f"({color_text(self.name, 'g')}) {self.event_counter}: Agent response:\n{color_text(response.content, 'm')}\n", flush=True
                )

            if not response.tool_calls or not handle_tool_calls:
                # Parse structured response if response_format is set
                if self.response_format:
                    try:
                        parsed_response = self._parse_structured_response(response.content)
                        return parsed_response
                    except Exception as e:
                        parse_retry_count += 1
                        self.log(
                            f"{self.name}: Failed to parse structured response (attempt {parse_retry_count}/{self.max_parse_retries}): {e}",
                            color='r', flush=True, verbose=True, count=False
                        )

                        if parse_retry_count >= self.max_parse_retries:
                            raise ValueError(
                                f"Failed to parse response into {self.response_format.__name__} after {self.max_parse_retries} attempts. "
                                f"Last error: {e}\nLast response: {response.content}"
                            )

                        # Ask the LLM to regenerate with explicit format instructions
                        format_instruction = self._get_format_instruction()
                        retry_msg = HumanMessage(
                            content=f"Your previous response could not be parsed into the required JSON format. "
                                    f"Error: {e}\n\n"
                                    f"Please regenerate your response.\n\n"
                                    f"{format_instruction}"
                        )
                        self.messages.append(retry_msg)
                        self.log(
                            f"({color_text(self.name, 'g')}) {self.event_counter}: Requesting LLM to regenerate response",
                            flush=True
                        )
                        continue
                else:
                    return response.content

            tasks = []
            for tool_call in response.tool_calls:
                tasks.append(
                    asyncio.create_task(self._invoke_tool(tool_call, cur_iteration >= self.max_tool_iterations))
                )
            tool_responses = await asyncio.gather(*tasks)
            for tool_call_id, tool_response in tool_responses:
                self.messages.append(
                    ToolMessage(content=tool_response, tool_call_id=tool_call_id)
                )

            cur_iteration += 1
            if cur_iteration >= self.max_agent_iterations:
                raise RuntimeError(f"{self.name}: Max agent iterations ({self.max_agent_iterations}) reached.")
            if cur_iteration >= self.max_tool_iterations and self.tool_iter_exceed_handler == "raise":
                raise RuntimeError(f"{self.name}: Max tool iterations ({self.max_tool_iterations}) reached.")

    def _get_format_instruction(self) -> str:
        """Generate format instruction based on response_format."""
        if self.response_format is None:
            return ""

        example = pydantic_model_to_example(self.response_format)
        schema = self.response_format.model_json_schema()

        instruction = "\n" + "="*60 + "\n"
        instruction += "IMPORTANT: Output Format Requirement\n"
        instruction += "="*60 + "\n"
        instruction += f"Your response must be a valid JSON object matching this structure:\n\n"
        instruction += json.dumps(example, indent=2, ensure_ascii=False) + "\n\n"
        instruction += "Field descriptions:\n"

        properties = schema.get('properties', {})
        for field_name, field_info in properties.items():
            description = field_info.get('description', 'N/A')
            field_type = field_info.get('type', 'unknown')
            instruction += f"- {field_name} ({field_type}): {description}\n"

        instruction += "\nPlease provide ONLY the JSON object in your response, without any additional explanation or markdown formatting.\n"
        instruction += "="*60 + "\n"

        return instruction

    def _parse_structured_response(self, response_text: str) -> BaseModel:
        """Parse response text into the configured Pydantic model."""
        # Attempt 1: direct JSON validation
        try:
            result = self.response_format.model_validate_json(response_text)
            return result
        except Exception as first_error:
            # Attempt 2: extract from markdown code block, then validate
            try:
                match = re.search(
                    rf"```\s*json\s*\n(.*?)\n```",
                    response_text,
                    flags=re.IGNORECASE | re.DOTALL
                )
                cleaned_text = match.group(1).strip()

                parsed_dict = str2dict(cleaned_text)

                result = self.response_format.model_validate(parsed_dict)
                return result

            except Exception as second_error:
                raise ValueError(
                    f"Failed to parse response into {self.response_format.__name__}. "
                    f"Direct validation error: {first_error}. "
                    f"Validation error after extracting markdown JSON block: {second_error}."
                )


if __name__ == "__main__":
    import asyncio

    async def test_agent():
        agent = Agent(
            model_settings={"model": "gpt-4.1-mini"},
            verbose=True,
            log_input=True,
        )

        await agent.chat("how are you")

    asyncio.run(test_agent())
