"""Functional tests for provider text and JSON generation.

These tests verify that each provider can successfully generate text and
JSON output when given simple prompts. Tests fail if credentials are
missing or incorrect, or if the local Ollama server is not running.
"""

import pytest
from pydantic import BaseModel, Field

SIMPLE_PROMPT = "Explain why the sky is blue in one paragraph to a six year old."
JSON_PROMPT = "Explain in one sentence why the sky is blue."


class SkyExplanation(BaseModel):
    """Schema for structured-output tests."""

    explanation: str = Field(
        description="A one-sentence explanation of why the sky is blue."
    )


PROVIDER_FIXTURES = [
    "ollama_provider",
    "openai_provider",
    "anthropic_provider",
    "vertex_provider",
]


@pytest.mark.parametrize("provider_fixture", PROVIDER_FIXTURES)
def test_provider_generates_text(provider_fixture, request):
    """Provider can generate prose text."""
    provider = request.getfixturevalue(provider_fixture)
    result = provider.invoke(SIMPLE_PROMPT)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "blue" in result.lower() or "sky" in result.lower()


@pytest.mark.parametrize("provider_fixture", PROVIDER_FIXTURES)
def test_provider_generates_json(provider_fixture, request):
    """Provider can generate schema-conforming structured output."""
    provider = request.getfixturevalue(provider_fixture)
    result = provider.invoke_json(JSON_PROMPT, SkyExplanation)

    assert isinstance(result, SkyExplanation)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0


@pytest.mark.parametrize("provider_fixture", PROVIDER_FIXTURES)
def test_invoke_after_invoke_json_returns_plain_text(provider_fixture, request):
    """Regression: a plain invoke following invoke_json must still return prose.

    Guards against providers that mutate shared client state in invoke_json
    (e.g. setting format='json' on the underlying client) and fail to reset
    it before the next plain invoke.
    """
    provider = request.getfixturevalue(provider_fixture)

    _ = provider.invoke_json(JSON_PROMPT, SkyExplanation)
    result = provider.invoke(SIMPLE_PROMPT)

    assert isinstance(result, str)
    assert len(result) > 0
    stripped = result.strip()
    assert not (stripped.startswith("{") and stripped.endswith("}")), (
        f"Expected prose output after invoke_json; got JSON-shaped string: {stripped[:200]}"
    )
