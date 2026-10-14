"""Shared pytest fixtures for functional tests.

These tests make real calls to LLM providers. Credentials are loaded from
secrets/.env at collection time.
"""

import pytest

from reusable_llm_provider.config import (
    create_anthropic_config,
    create_ollama_config,
    create_openai_config,
    create_vertex_config,
)
from reusable_llm_provider.providers import create_provider

from .env import load_env

load_env()


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


# Gemini counts thinking tokens against max_output_tokens, and 2.5 Flash sizes
# its thinking dynamically. Measured over six runs at the library default of
# 1000 tokens, thinking took 841-957 of them and four runs ended in MAX_TOKENS
# with 39-113 tokens of answer -- a truncated stub that failed the keyword
# assertion in test_provider_generates_text one run in three. Capping the
# budget leaves room for the answer: the same six runs then used ~200 thinking
# tokens and every one finished with STOP.
#
# Capped rather than disabled, so these tests keep exercising a real thinking
# pass, and the integer-budget branch of VertexProvider._thinking() gets the
# only live coverage it has.
VERTEX_THINKING_BUDGET = 256


@pytest.fixture
def vertex_provider():
    """Create Vertex AI provider. Fails if VERTEX_PROJECT_ID is missing.

    Requires gcloud authentication.
    """
    return create_provider(create_vertex_config(thinking=VERTEX_THINKING_BUDGET))
