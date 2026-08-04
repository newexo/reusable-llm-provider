"""Functional tests for provider text and structured generation.

These tests verify that each provider can successfully generate free-form
text and validated typed output when given simple prompts. Tests fail if
credentials are missing or incorrect, or if the local Ollama server is
not running.
"""

import pytest
from pydantic import BaseModel, Field

SIMPLE_PROMPT = "Explain why the sky is blue in one paragraph to a six year old."
STRUCTURED_PROMPT = "Explain in one sentence why the sky is blue."


class SkyExplanation(BaseModel):
    """Output model for structured-generation tests."""

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
def test_provider_generates_structured_output(provider_fixture, request):
    """Provider returns a validated instance of the requested output model."""
    provider = request.getfixturevalue(provider_fixture)
    result = provider.invoke_structured(STRUCTURED_PROMPT, SkyExplanation)

    assert isinstance(result, SkyExplanation)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0


@pytest.mark.parametrize("provider_fixture", PROVIDER_FIXTURES)
def test_invoke_after_invoke_structured_returns_plain_text(provider_fixture, request):
    """Regression: a plain invoke following invoke_structured must still return prose.

    Guards against providers that mutate shared client state during
    structured generation (e.g. setting format='json' on the underlying
    client) and fail to reset it before the next plain invoke.
    """
    provider = request.getfixturevalue(provider_fixture)

    _ = provider.invoke_structured(STRUCTURED_PROMPT, SkyExplanation)
    result = provider.invoke(SIMPLE_PROMPT)

    assert isinstance(result, str)
    assert len(result) > 0
    stripped = result.strip()
    assert not (stripped.startswith("{") and stripped.endswith("}")), (
        "Expected prose output after invoke_structured; got JSON-shaped "
        f"string: {stripped[:200]}"
    )


# Models that reject the `temperature` parameter outright. The library's
# default model, claude-haiku-4-5, still accepts it, so every test above would
# have passed before temperature became opt-in — none of them exercise the
# case the change exists for. These do.
TEMPERATURE_REJECTING_MODELS = ["claude-sonnet-5", "claude-opus-5"]


@pytest.mark.parametrize("model", TEMPERATURE_REJECTING_MODELS)
def test_models_that_reject_temperature_are_usable(model):
    """Newer models must work when temperature is left unset.

    Sending `temperature` to these returns 400 invalid_request_error. Before
    temperature became opt-in the library sent it unconditionally, so these
    models could not be driven at all.
    """
    from reusable_llm_provider.config import create_anthropic_config
    from reusable_llm_provider.providers import create_provider

    provider = create_provider(create_anthropic_config(model=model))
    result = provider.invoke(STRUCTURED_PROMPT)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", TEMPERATURE_REJECTING_MODELS)
def test_explicit_temperature_still_rejected_by_these_models(model):
    """The escape hatch remains closed where the provider says it is closed.

    Passing temperature explicitly must still reach the API and still fail.
    The library omits the parameter by default; it does not silently discard
    a value the caller asked for.
    """
    from reusable_llm_provider.config import create_anthropic_config
    from reusable_llm_provider.providers import LLMGenerationError, create_provider

    provider = create_provider(create_anthropic_config(model=model, temperature=0.0))
    with pytest.raises(LLMGenerationError):
        provider.invoke(STRUCTURED_PROMPT)
