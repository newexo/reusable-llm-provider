"""Provider abstraction layer for LLM APIs.

This module provides a unified interface for different LLM providers (Anthropic,
OpenAI, Vertex AI, Ollama) while wrapping their specific SDKs.
"""

from abc import ABC, abstractmethod
from typing import Protocol

from .config import LLMConfig, LLMProviderType

import json

from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from openai import BadRequestError
from google import genai
from google.genai import types
from langchain_ollama import OllamaLLM


class LLMGenerationError(Exception):
    """Exception raised when LLM generation fails."""

    def __init__(self, provider: str, original_error: Exception):
        self.provider = provider
        self.original_error = original_error
        error_type = type(original_error).__name__
        error_msg = str(original_error)
        super().__init__(
            f"LLM generation failed for {provider}: {error_type}: {error_msg}"
        )


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    def invoke(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt text

        Returns:
            Generated text response

        Raises:
            LLMGenerationError: If generation fails
        """
        ...

    def invoke_json(self, prompt: str) -> dict:
        """Generate JSON output from a prompt.

        Args:
            prompt: The input prompt text

        Returns:
            Parsed JSON response as dictionary

        Raises:
            LLMGenerationError: If generation fails
        """
        ...


class BaseLLMProvider(ABC):
    """Base class for LLM providers with common properties."""

    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def invoke_json(self, prompt: str) -> dict:
        pass


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic's Claude API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.chat_model = ChatAnthropic(
            model=self.model,
            api_key=config.anthropic_api_key,
            max_tokens=self.max_tokens,
        )

    def invoke(self, prompt: str) -> str:
        """Generate text using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise LLMGenerationError("anthropic", e) from e

    def invoke_json(self, prompt: str) -> dict:
        """Generate JSON output using Anthropic API via JsonOutputParser."""

        parser = JsonOutputParser()

        try:
            llm = self.chat_model.bind(temperature=self.temperature)
            chain = llm | parser
            return chain.invoke(prompt + "\n\n" + parser.get_format_instructions())
        except Exception as e:
            raise LLMGenerationError("anthropic", e) from e


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI's API."""

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
        )

    def invoke(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMGenerationError("openai", e) from e

    def invoke_json(self, prompt: str) -> dict:
        """Generate JSON output using OpenAI API.

        Uses JSON mode if the model supports it, falls back to
        JsonOutputParser for older models (e.g. gpt-4).
        """

        parser = JsonOutputParser()

        try:
            try:
                llm = self.chat_model.bind(
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                result = llm.invoke(prompt)
                return json.loads(result.content)
            except BadRequestError:
                # Model doesn't support response_format
                llm = self.chat_model.bind(temperature=self.temperature)
                chain = llm | parser
                return chain.invoke(prompt + "\n\n" + parser.get_format_instructions())
        except Exception as e:
            raise LLMGenerationError("openai", e) from e


class VertexAIProvider(BaseLLMProvider):
    """Provider for Google's Vertex AI (Gemini) via the unified google.genai SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        self.client = genai.Client(
            vertexai=True,
            project=config.vertex_project_id,
            location=config.vertex_location,
            http_options=types.HttpOptions(api_version="v1"),
        )

    def invoke(self, prompt: str) -> str:
        """Generate text using Vertex AI."""
        try:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=config,
            )
            return response.text
        except Exception as e:
            raise LLMGenerationError("vertex", e) from e

    def invoke_json(self, prompt: str) -> dict:
        """Generate JSON output using Vertex AI with JSON mode."""
        try:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=config,
            )
            return json.loads(response.text)
        except Exception as e:
            raise LLMGenerationError("vertex", e) from e


class OllamaProvider(BaseLLMProvider):
    """Provider for local Ollama instances."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        self.llm = OllamaLLM(model=config.model)

    def invoke(self, prompt: str) -> str:
        """Generate text using Ollama."""
        try:
            self.llm.temperature = self.temperature
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            raise LLMGenerationError("ollama", e) from e

    def invoke_json(self, prompt: str) -> dict:
        """Generate JSON output using Ollama with JSON format.

        Passes ``format="json"`` as a per-call keyword argument rather than
        mutating ``self.llm.format``; otherwise the JSON mode would leak
        into subsequent plain ``invoke()`` calls on the same provider.
        """
        try:
            self.llm.temperature = self.temperature
            response = self.llm.invoke(prompt, format="json")
            return json.loads(response)
        except Exception as e:
            raise LLMGenerationError("ollama", e) from e


def create_provider(config: LLMConfig):
    """Create a provider instance based on configuration.

    Args:
        config: LLMConfig instance with provider settings

    Returns:
        Provider instance (AnthropicProvider, OpenAIProvider, VertexAIProvider,
        or OllamaProvider)

    Raises:
        ValueError: If provider is invalid
    """
    if config.provider == LLMProviderType.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == LLMProviderType.OPENAI:
        return OpenAIProvider(config)
    elif config.provider == LLMProviderType.VERTEX:
        return VertexAIProvider(config)
    elif config.provider == LLMProviderType.OLLAMA:
        return OllamaProvider(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
