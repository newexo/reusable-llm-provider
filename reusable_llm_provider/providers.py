"""Provider abstraction layer for LLM APIs.

This module provides a unified interface for different LLM providers (Anthropic,
OpenAI, Vertex AI, Ollama) while wrapping their specific SDKs.

Structured output uses langchain's ``with_structured_output`` with a
caller-supplied Pydantic schema. This gives us schema-enforced decoding where
the provider supports it (OpenAI json_schema, Anthropic tool-use, Vertex
json_schema, Ollama json_schema) and returns the raw ``AIMessage`` alongside
parsing errors via ``include_raw=True`` so diagnostics are preserved.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Protocol, Type

from anthropic import Anthropic
from google import genai
from google.genai import types
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

from .config import LLMConfig, LLMProviderType


class LLMGenerationError(Exception):
    """Exception raised when LLM generation fails.

    When the error occurred while parsing structured output, ``raw`` holds
    the provider's raw response (typically an ``AIMessage``) so callers can
    inspect exactly what came back.
    """

    def __init__(self, provider: str, original_error: Exception, raw=None):
        self.provider = provider
        self.original_error = original_error
        self.raw = raw
        error_type = type(original_error).__name__
        error_msg = str(original_error)
        message = f"LLM generation failed for {provider}: {error_type}: {error_msg}"
        if raw is not None:
            message += f"\nRaw response: {raw!r}"
        super().__init__(message)


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    def invoke(self, prompt: str) -> str:
        """Generate text from a prompt.

        Raises:
            LLMGenerationError: If generation fails.
        """
        ...

    def invoke_json(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        """Generate structured output conforming to ``schema``.

        Uses the provider's strongest available structured-output mechanism
        (native JSON-schema decoding, tool-use, or equivalent). Returns an
        instance of ``schema``.

        Raises:
            LLMGenerationError: If generation or parsing fails. When the
                provider returned text but the parse failed, the raw
                ``AIMessage`` is attached to the original error for
                diagnostics.
        """
        ...


class BaseLLMProvider(ABC):
    """Template-method base class for LLM providers.

    Subclasses supply two primitives: ``_invoke_raw_text(prompt)`` which
    returns a raw text string, and ``_structured_model()`` which returns a
    langchain chat model suitable for ``with_structured_output``. This
    base handles the shared concerns: wrapping every exception in
    ``LLMGenerationError`` with the subclass's ``NAME`` and applying
    schema-enforced structured decoding with ``include_raw=True`` so
    parsing errors preserve the raw provider response.
    """

    NAME: str = ""
    STRUCTURED_OUTPUT_METHOD: str | None = None

    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature

    @contextmanager
    def _wrap_errors(self):
        """Context manager that rewraps any exception as LLMGenerationError."""
        try:
            yield
        except Exception as e:
            raise LLMGenerationError(self.NAME, e) from e

    def invoke(self, prompt: str) -> str:
        with self._wrap_errors():
            return self._invoke_raw_text(prompt)

    def invoke_json(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        with self._wrap_errors():
            chat_model = self._structured_model()
            kwargs = {"include_raw": True}
            if self.STRUCTURED_OUTPUT_METHOD is not None:
                kwargs["method"] = self.STRUCTURED_OUTPUT_METHOD
            structured = chat_model.with_structured_output(schema, **kwargs)
            result = structured.invoke(prompt)
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")
            if parsed is None or parsing_error is not None:
                error = parsing_error or ValueError(
                    "Structured output returned no parsed value"
                )
                raise LLMGenerationError(self.NAME, error, raw=result.get("raw"))
            return parsed

    @abstractmethod
    def _invoke_raw_text(self, prompt: str) -> str:
        """Return the provider's raw text output for ``prompt``."""

    @abstractmethod
    def _structured_model(self):
        """Return a langchain chat model supporting ``with_structured_output``."""


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic's Claude API."""

    NAME = "anthropic"
    STRUCTURED_OUTPUT_METHOD = "function_calling"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.chat_model = ChatAnthropic(
            model=self.model,
            api_key=config.anthropic_api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _invoke_raw_text(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _structured_model(self):
        return self.chat_model


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI's API."""

    NAME = "openai"
    STRUCTURED_OUTPUT_METHOD = "json_schema"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.openai_api_key,
            organization=config.openai_organization,
        )
        self.chat_model = ChatOpenAI(
            model=self.model,
            api_key=config.openai_api_key,
            organization=config.openai_organization,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _invoke_raw_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _structured_model(self):
        return self.chat_model


class VertexAIProvider(BaseLLMProvider):
    """Provider for Google's Vertex AI (Gemini).

    Uses the native ``google.genai`` SDK for free-form text generation and
    ``langchain_google_vertexai.ChatVertexAI`` for structured output so
    ``with_structured_output`` can apply Vertex's native JSON-schema
    decoding.
    """

    NAME = "vertex"
    STRUCTURED_OUTPUT_METHOD = "json_schema"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = genai.Client(
            vertexai=True,
            project=config.vertex_project_id,
            location=config.vertex_location,
            http_options=types.HttpOptions(api_version="v1"),
        )
        self.chat_model = ChatVertexAI(
            model=self.model,
            project=config.vertex_project_id,
            location=config.vertex_location,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

    def _invoke_raw_text(self, prompt: str) -> str:
        request_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=request_config,
        )
        return response.text

    def _structured_model(self):
        return self.chat_model


class OllamaProvider(BaseLLMProvider):
    """Provider for local Ollama instances.

    Uses ``OllamaLLM`` for free-form text and ``ChatOllama`` for structured
    output so ``with_structured_output`` can apply Ollama's native JSON-
    schema constrained decoding (stricter than ``format="json"``).
    """

    NAME = "ollama"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Temperature fixed at construction time; per-call mutation was the
        # shape of the sticky-``format`` bug we previously fixed.
        self.llm = OllamaLLM(model=config.model, temperature=self.temperature)
        self.chat_model = ChatOllama(model=config.model, temperature=self.temperature)

    def _invoke_raw_text(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    def _structured_model(self):
        return self.chat_model


_PROVIDER_MAP = {
    LLMProviderType.ANTHROPIC: AnthropicProvider,
    LLMProviderType.OPENAI: OpenAIProvider,
    LLMProviderType.VERTEX: VertexAIProvider,
    LLMProviderType.OLLAMA: OllamaProvider,
}


def create_provider(config: LLMConfig):
    """Create a provider instance based on configuration.

    Raises:
        ValueError: If ``config.provider`` is not a known provider type.
    """
    try:
        provider_cls = _PROVIDER_MAP[config.provider]
    except KeyError as e:
        raise ValueError(f"Unsupported provider: {config.provider}") from e
    return provider_cls(config)
