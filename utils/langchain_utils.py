import io
import os
import contextlib
from pydantic import Field
from typing import Any, Union, Iterable
from langchain_openai import OpenAIEmbeddings


class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI Embeddings with tokenizer initialization."""

    tokenizer: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        tokenizer_name = self.tiktoken_model_name or self.model
        if not self.tiktoken_enabled:
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO())
                ):
                    from transformers import AutoTokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed for OpenAIEmbeddings to work without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_name
            )
        else:
            try:
                import tiktoken
                tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            except KeyError:
                tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

    def _tokenize(
        self, texts: list[str], chunk_size: int
    ) -> tuple[Iterable[int], list[list[int] | str], list[int], list[int]]:
        """Tokenize and batch input texts.

        Splits texts based on `embedding_ctx_length` and groups them into batches
        of size `chunk_size`.

        Args:
            texts: The list of texts to tokenize.
            chunk_size: The maximum number of texts to include in a single batch.

        Returns:
            A tuple containing:
                1. An iterable of starting indices in the token list for each batch.
                2. A list of tokenized texts (token arrays for tiktoken, strings for
                    HuggingFace).
                3. An iterable mapping each token array to the index of the original
                    text. Same length as the token list.
                4. A list of token counts for each tokenized text.
        """
        tokens: list[list[int] | str] = []
        indices: list[int] = []
        token_counts: list[int] = []
        
        tokenizer = self.tokenizer

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:

            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized: list[int] = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk: list[int] = tokenized[
                        j : j + self.embedding_ctx_length
                    ]

                    # Convert token IDs back to a string
                    chunk_text: str = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
                    token_counts.append(len(token_chunk))
        else:
            encoder_kwargs: dict[str, Any] = {
                k: v
                for k, v in {
                    "allowed_special": self.allowed_special,
                    "disallowed_special": self.disallowed_special,
                }.items()
                if v is not None
            }
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                if encoder_kwargs:
                    token = tokenizer.encode(text, **encoder_kwargs)
                else:
                    token = tokenizer.encode_ordinary(text)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)
                    token_counts.append(len(token[j : j + self.embedding_ctx_length]))

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(tokens), chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), chunk_size)
        else:
            _iter = range(0, len(tokens), chunk_size)
        return _iter, tokens, indices, token_counts