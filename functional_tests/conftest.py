"""Shared pytest fixtures for functional tests.

These tests make real calls to LLM providers. Credentials are loaded from
secrets/.env at collection time.
"""

import pytest

from reusable_llm_provider.config import (
    create_anthropic_config,
    create_openai_config,
    create_vertex_config,
    create_ollama_config,
)
from reusable_llm_provider.providers import create_provider
from reusable_llm_provider.env import load_reusable_llm_provider_env

load_reusable_llm_provider_env()


@pytest.fixture
def ollama_provider():
    """Create Ollama provider. Fails if server is not running."""
    return create_provider(create_ollama_config())


@pytest.fixture
def openai_provider():
    """Create OpenAI provider. Fails if OPENAI_API_KEY is missing."""
    return create_provider(create_openai_config())


@pytest.fixture
def anthropic_provider():
    """Create Anthropic provider. Fails if ANTHROPIC_API_KEY is missing."""
    return create_provider(create_anthropic_config())


@pytest.fixture
def vertex_provider():
    """Create Vertex AI provider. Fails if VERTEX_PROJECT_ID is missing.

    Requires gcloud authentication.
    """
    return create_provider(create_vertex_config())
