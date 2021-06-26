"""Configuration management for LLM providers.

This module handles provider configuration and credential injection.
Credentials can be passed explicitly via LLMConfig or read from
environment variables via the create_*_config convenience functions.
"""

import os
from enum import Enum
from typing import Optional

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
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
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_organization: Optional[str] = None,
        vertex_project_id: Optional[str] = None,
        vertex_location: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature if temperature is not None else 0.0
        self.max_tokens = max_tokens if max_tokens is not None else 1000

        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.openai_organization = openai_organization
        self.vertex_project_id = vertex_project_id
        self.vertex_location = vertex_location


def create_anthropic_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLMConfig:
    """Create Anthropic configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["anthropic"]

    return LLMConfig(
        provider=LLMProviderType.ANTHROPIC,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def create_openai_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLMConfig:
    """Create OpenAI configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["openai"]

    return LLMConfig(
        provider=LLMProviderType.OPENAI,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_ORGANIZATION"),
    )


def create_vertex_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLMConfig:
    """Create Vertex AI configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["vertex"]

    return LLMConfig(
        provider=LLMProviderType.VERTEX,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        vertex_project_id=os.getenv("VERTEX_PROJECT_ID"),
        vertex_location=os.getenv("VERTEX_LOCATION"),
    )


def create_ollama_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLMConfig:
    """Create Ollama configuration from environment variables."""
    if model is None:
        model = DEFAULT_MODELS["ollama"]

    return LLMConfig(
        provider=LLMProviderType.OLLAMA,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )
