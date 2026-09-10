"""Configuration management for LLM providers.

This module handles provider configuration and credential injection.
Credentials can be passed explicitly via LLMConfig or read from
environment variables via the create_*_config convenience functions.
"""

import os
from enum import Enum
from typing import Optional, Union

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.4-nano",
    "vertex": "gemini-2.5-flash",
    "ollama": "gemma2",
}


class LLMProviderType(Enum):
    """Enum for supported LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VERTEX = "vertex"
    OLLAMA = "ollama"


class LLMConfig:
    """LLM configuration with explicit credential injection.

    Credentials can be passed directly or read from environment variables
    via the create_*_config convenience functions.
    """

    def __init__(
        self,
        provider: LLMProviderType,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking: Optional[Union[str, int]] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_organization: Optional[str] = None,
        vertex_project_id: Optional[str] = None,
        vertex_location: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        # None means "do not send temperature at all", and is the default.
        #
        # The sampling parameter is being withdrawn across the industry, and it
        # fails three different ways: Anthropic's Claude 5 family and OpenAI's
        # GPT-5 reasoning models reject the request outright, while Gemini 3.x
        # accepts and silently ignores it. Ollama still honours it. Sending a
        # value by default therefore made the newest models of two providers
        # unusable. Callers who want deterministic sampling, and are on a model
        # that still supports it, pass temperature explicitly.
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else 1000

        # None means "do not send a thinking parameter at all", and is the
        # default. It keeps existing callers byte-identical and lets each
        # provider apply its own default.
        #
        # Disabling by default was considered and rejected: the off switch is
        # itself refused by some models. reasoning_effort="none" is a 400 on
        # gpt-4o and gpt-4o-mini, so sending it unconditionally would break
        # those callers the same way sending temperature once broke the
        # newest ones.
        #
        # "auto" is accepted and currently behaves exactly like None, because
        # no provider needs an explicit "you decide" signal. It exists so a
        # caller can record the intent, and is deliberately NOT mapped to
        # Anthropic's "adaptive", which claude-haiku-4-5 rejects.
        self.thinking = self._validate_thinking(thinking)

        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.openai_organization = openai_organization
        self.vertex_project_id = vertex_project_id
        self.vertex_location = vertex_location

    @staticmethod
    def _validate_thinking(value):
        """Accept None, "off", "auto", or a positive token budget."""
        if value is None or value in ("off", "auto"):
            return value
        # bool is a subclass of int; True is not a token budget.
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        raise ValueError(
            f"Invalid thinking value {value!r}. Expected None, 'off', 'auto', "
            "or a positive integer token budget."
        )


def create_anthropic_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[Union[str, int]] = None,
) -> LLMConfig:
    """Create Anthropic configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["anthropic"]

    return LLMConfig(
        provider=LLMProviderType.ANTHROPIC,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        model=model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def create_openai_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[Union[str, int]] = None,
) -> LLMConfig:
    """Create OpenAI configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["openai"]

    return LLMConfig(
        provider=LLMProviderType.OPENAI,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_ORGANIZATION"),
    )


def create_vertex_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[Union[str, int]] = None,
) -> LLMConfig:
    """Create Vertex AI configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["vertex"]

    return LLMConfig(
        provider=LLMProviderType.VERTEX,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        model=model,
        vertex_project_id=os.getenv("VERTEX_PROJECT_ID"),
        vertex_location=os.getenv("VERTEX_LOCATION"),
    )


def create_ollama_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[Union[str, int]] = None,
) -> LLMConfig:
    """Create Ollama configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["ollama"]

    return LLMConfig(
        provider=LLMProviderType.OLLAMA,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        model=model,
    )
