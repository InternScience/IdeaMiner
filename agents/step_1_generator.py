import sys
import os
import argparse
from structai import LLMAgent, load_file, save_file

# Initialize LLM Agent
llm_generator = LLMAgent(model_version="gemini-3-pro-preview", temperature=1.0)


def generate_scientific_questions(field, keywords, research_type, granularity_level, use_innovative_vocab=True):
    """
    Generate scientific research questions using an LLM agent.

    Args:
        field (str): The scientific field (e.g., "Life Sciences").
        keywords (list): List of keywords to focus the questions on.
        research_type (str): Type of research (e.g., "Theory", "Experiment").
        granularity_level (str): Level of granularity (e.g., "Microscopic").
        use_innovative_vocab (bool): Whether to require "Verb + Noun" innovative
                                     vocabulary patterns in the questions.

    Returns:
        list: A list of dicts, each containing the generated question and metadata.
              Returns an empty list if generation fails.
    """
    prompt = f"""
You are a visionary senior scientist in the field of {field}.
Your task is to propose 30 high-quality, novel, and feasible scientific research questions based on the following research context.

Research Context:
- Field: {field}
- Keywords: {', '.join(keywords)}
- Research Type: {research_type}
- Granularity Level: {granularity_level}

Requirements for the questions:
1. **High Diversity**: The questions must be significantly different from each other in terms of focus and hypothesis. Avoid repetition.
2. **Deep Integration**: Do not simply stitch concepts together (e.g., A+B). The questions should reflect a deep understanding of the underlying mechanisms and logic.
3. **Novelty**: The questions must be innovative, exploring uncharted territories, new perspectives, or challenging existing paradigms.
4. **Feasibility**: The questions must be scientifically feasible to investigate with current or near-future technologies and methodologies.
5. **Language**: All content in the output values must be in English.
"""

    if use_innovative_vocab:
        prompt += '\nConstraint: The questions must contain innovative "Verb + Noun" vocabulary constructions (e.g., similar style to \'clicking chemistry\') to describe processes or phenomena.\n'

    prompt += """
Output Format:
Please output the result strictly as a Python list of dictionaries.

Each dictionary must contain the following keys:
- "background": A brief background introduction.
- "question": The core research question in one sentence.
- "significance": A brief statement of the question's significance and value.
- "methodology": A list of strings, where each string is a step in the proposed methodology, starting with a number (e.g., "1. ...").
- "rationale": A brief explanation of the novelty and logic behind the question.
- "key_concepts": A list of 2-3 key concepts involved.

The list must contain exactly 30 dictionaries.
"""

    # Example structure used for response validation (values are English placeholders)
    return_example = [
        {
            "background": "<brief background>",
            "question": "<core research question>",
            "significance": "<significance and value>",
            "methodology": ["1. <step one>", "2. <step two>"],
            "rationale": "<novelty rationale>",
            "key_concepts": ["<concept 1>", "<concept 2>"]
        }
    ]

    print("Generating questions via LLM...")
    response = llm_generator(
        query=prompt,
        return_example=return_example,
        list_len=30
    )

    if response is None:
        print("Failed to generate valid questions after retries.")
        return []

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate scientific questions based on a config file.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--use_innovative_vocab', action='store_true',
                        help='Require innovative "Verb + Noun" vocabulary in questions')

    args = parser.parse_args()

    config_path = args.config_path
    use_innovative_vocab = args.use_innovative_vocab
    print(f"{config_path=},{use_innovative_vocab=}")

    output_dir = './data/raw_questions'

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from {config_path}...")
    config_content = load_file(config_path)

    questions = generate_scientific_questions(
        field=config_content.get("field", "Unknown Field"),
        keywords=config_content.get("keywords", []),
        research_type=config_content.get("research_type", "General"),
        granularity_level=config_content.get("granularity_level", "Macroscopic"),
        use_innovative_vocab=use_innovative_vocab
    )

    if questions:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        config_filename = os.path.basename(config_path)
        output_path = os.path.join(output_dir, config_filename)
        if not use_innovative_vocab:
            output_path = output_path.replace(".json", "_without_innovative_vocab.json")

        if os.path.exists(output_path):
            print("Existing file found. Appending new questions to it.")
            old_questions = load_file(output_path)
            questions = old_questions + questions

        save_file(questions, output_path)
        print(f"Successfully saved {len(questions)} questions to {output_path}")
    else:
        print("No questions were generated.")
