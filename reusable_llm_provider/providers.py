"""Provider abstraction layer for LLM APIs.

This module provides a unified interface for different LLM providers
(Anthropic, OpenAI, Vertex AI, Ollama). The public contract is intentionally
small and framework-neutral:

- ``invoke(prompt) -> str`` for free-form text generation.
- ``invoke_structured(prompt, output_model) -> BaseModel`` for typed output.

Callers supply a Pydantic model class as ``output_model`` and receive a
validated instance of that exact class. How a given backend produces the
candidate value (native JSON-schema decoding, tool calling, constrained
decoding, text extraction, or framework mediation such as LangChain) is an
internal implementation detail captured by ``StructuredOutputStrategy``.

Local Pydantic validation is always applied to the candidate before returning,
regardless of any vendor guarantees. This keeps the public contract grounded
in local type safety rather than provider promises.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Protocol, Type

from anthropic import Anthropic
from google import genai
from google.genai import types
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import LLMConfig, LLMProviderType


class StructuredOutputStrategy(str, Enum):
    """How a provider attempts to produce structured output.

    These are interchangeable internal strategies. The public
    ``invoke_structured`` contract is identical regardless of which
    strategy a provider uses; the enum exists so maintainers can reason
    about provider quality and migration risk as vendor ecosystems
    evolve.
    """

    NATIVE_JSON_SCHEMA = "native_json_schema"
    TOOL_CALLING = "tool_calling"
    CONSTRAINED_DECODING = "constrained_decoding"
    TEXT_EXTRACTION = "text_extraction"
    LANGCHAIN_MEDIATED = "langchain_mediated"


class LLMGenerationError(Exception):
    """Base class for failures in LLM generation."""

    def __init__(self, provider: str, original_error: Exception, raw: Any = None):
        self.provider = provider
        self.original_error = original_error
        self.raw = raw
        error_type = type(original_error).__name__
        error_msg = str(original_error)
        message = f"LLM generation failed for {provider}: {error_type}: {error_msg}"
        if raw is not None:
            message += f"\nRaw response: {raw!r}"
        super().__init__(message)


class LLMTransportError(LLMGenerationError):
    """The provider could not be reached or the request itself failed.

    Covers network errors, authentication failures, rate-limit responses,
    and other transport-layer faults where no usable content was produced
    by the model. SDK exception classification is best-effort; ambiguous
    failures fall back to ``LLMGenerationError``.
    """


class LLMProviderGenerationError(LLMGenerationError):
    """The provider responded but indicated a generation-side failure.

    Covers content-filter refusals, explicit error responses from the
    model, or empty completions where the provider did not raise but
    returned no usable output.
    """


class StructuredOutputValidationError(LLMGenerationError):
    """The provider returned content that did not validate against the
    requested Pydantic ``output_model``.

    Attributes:
        provider: Name of the provider that produced the content.
        output_model: The Pydantic model class the caller requested.
        strategy: Which ``StructuredOutputStrategy`` was used.
        validation_error: The underlying ``pydantic.ValidationError`` (or
            other parse error) describing why validation failed.
        raw: The raw provider response, when available.
    """

    def __init__(
        self,
        provider: str,
        output_model: Type[BaseModel],
        strategy: StructuredOutputStrategy,
        validation_error: Exception,
        raw: Any = None,
    ):
        self.output_model = output_model
        self.strategy = strategy
        super().__init__(provider, validation_error, raw=raw)
        self.args = (
            f"Structured output for {provider} did not validate against "
            f"{output_model.__name__} (strategy={strategy.value}): "
            f"{type(validation_error).__name__}: {validation_error}",
        )

    @property
    def validation_error(self) -> Exception:
        return self.original_error


class LLMProvider(Protocol):
    """Public protocol for LLM providers."""

    def invoke(self, prompt: str) -> str:
        """Generate free-form text from ``prompt``.

        Raises:
            LLMGenerationError: If generation fails.
        """
        ...

    def invoke_structured(
        self, prompt: str, output_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a typed result conforming to ``output_model``.

        The return value is always an instance of ``output_model``,
        validated locally via Pydantic regardless of any provider-side
        guarantees.

        Raises:
            StructuredOutputValidationError: If the provider returned
                content but it did not validate against ``output_model``.
            LLMTransportError: If the request failed before any content
                was produced.
            LLMProviderGenerationError: If the provider explicitly
                signaled a generation failure or refusal.
            LLMGenerationError: For any other failure not classified
                above.
        """
        ...


class BaseLLMProvider(ABC):
    """Template-method base class for LLM providers.

    Subclasses supply two primitives:

    - ``_invoke_raw_text(prompt)`` — return free-form text.
    - ``_invoke_structured_candidate(prompt, output_model)`` — return a
      candidate value (a ``BaseModel`` instance, a dict, or a JSON
      string) that this base will validate locally.

    The base handles error wrapping and the mandatory local Pydantic
    validation step.
    """

    NAME: str = ""
    STRATEGY: StructuredOutputStrategy = StructuredOutputStrategy.LANGCHAIN_MEDIATED

    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature

    @contextmanager
    def _wrap_errors(self):
        try:
            yield
        except LLMGenerationError:
            raise
        except Exception as e:
            raise LLMGenerationError(self.NAME, e) from e

    def invoke(self, prompt: str) -> str:
        with self._wrap_errors():
            return self._invoke_raw_text(prompt)

    def invoke_structured(
        self, prompt: str, output_model: Type[BaseModel]
    ) -> BaseModel:
        with self._wrap_errors():
            candidate, raw = self._invoke_structured_candidate(prompt, output_model)
            return self._validate_candidate(candidate, output_model, raw)

    def _validate_candidate(
        self,
        candidate: Any,
        output_model: Type[BaseModel],
        raw: Any,
    ) -> BaseModel:
        """Run mandatory local Pydantic validation on ``candidate``.

        Even when a strategy claims native schema enforcement, the
        candidate is re-validated against ``output_model`` so the public
        contract rests on local type safety, not vendor promises.
        """
        if candidate is None:
            raise StructuredOutputValidationError(
                provider=self.NAME,
                output_model=output_model,
                strategy=self.STRATEGY,
                validation_error=ValueError(
                    "Structured output strategy returned no candidate value"
                ),
                raw=raw,
            )
        try:
            if isinstance(candidate, BaseModel):
                payload = candidate.model_dump()
            elif isinstance(candidate, dict):
                payload = candidate
            elif isinstance(candidate, str):
                return output_model.model_validate_json(candidate)
            else:
                raise TypeError(
                    f"Unsupported candidate type from strategy "
                    f"{self.STRATEGY.value}: {type(candidate).__name__}"
                )
            return output_model.model_validate(payload)
        except (ValidationError, ValueError, TypeError) as e:
            raise StructuredOutputValidationError(
                provider=self.NAME,
                output_model=output_model,
                strategy=self.STRATEGY,
                validation_error=e,
                raw=raw,
            ) from e

    @abstractmethod
    def _invoke_raw_text(self, prompt: str) -> str:
        """Return the provider's raw text output for ``prompt``."""

    @abstractmethod
    def _invoke_structured_candidate(
        self, prompt: str, output_model: Type[BaseModel]
    ) -> tuple[Any, Any]:
        """Return ``(candidate, raw)`` from the provider.

        ``candidate`` may be a ``BaseModel`` instance, a dict, or a JSON
        string. ``raw`` is the unparsed provider response when
        available, used to enrich diagnostics on validation failure.
        """


class _LangChainStructuredMixin:
    """Internal helper: produce a structured candidate via LangChain.

    LangChain is one of several possible strategies and is treated as a
    private implementation detail. Its specific result shape
    (``include_raw=True`` returning ``{"parsed", "raw", "parsing_error"}``)
    must not leak past this mixin.
    """

    _LANGCHAIN_METHOD: str | None = None

    def _structured_model(self):
        raise NotImplementedError

    def _invoke_structured_candidate(
        self, prompt: str, output_model: Type[BaseModel]
    ) -> tuple[Any, Any]:
        chat_model = self._structured_model()
        kwargs = {"include_raw": True}
        if self._LANGCHAIN_METHOD is not None:
            kwargs["method"] = self._LANGCHAIN_METHOD
        structured = chat_model.with_structured_output(output_model, **kwargs)
        result = structured.invoke(prompt)
        return result.get("parsed"), result.get("raw")


class AnthropicProvider(_LangChainStructuredMixin, BaseLLMProvider):
    """Provider for Anthropic's Claude API."""

    NAME = "anthropic"
    STRATEGY = StructuredOutputStrategy.LANGCHAIN_MEDIATED
    _LANGCHAIN_METHOD = "function_calling"

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


class OpenAIProvider(_LangChainStructuredMixin, BaseLLMProvider):
    """Provider for OpenAI's API."""

    NAME = "openai"
    STRATEGY = StructuredOutputStrategy.LANGCHAIN_MEDIATED
    _LANGCHAIN_METHOD = "json_schema"

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


class VertexAIProvider(_LangChainStructuredMixin, BaseLLMProvider):
    """Provider for Google's Vertex AI (Gemini).

    Uses the native ``google.genai`` SDK for free-form text generation
    and the LangChain Vertex chat model for structured output.
    """

    NAME = "vertex"
    STRATEGY = StructuredOutputStrategy.LANGCHAIN_MEDIATED
    _LANGCHAIN_METHOD = "json_schema"

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


class OllamaProvider(_LangChainStructuredMixin, BaseLLMProvider):
    """Provider for local Ollama instances."""

    NAME = "ollama"
    STRATEGY = StructuredOutputStrategy.LANGCHAIN_MEDIATED

    def __init__(self, config: LLMConfig):
        super().__init__(config)
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
