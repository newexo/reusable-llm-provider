"""Tests for LLM provider implementations."""

import pytest
from unittest.mock import Mock, patch
from reusable_llm_provider.config import LLMConfig, LLMProviderType
from pydantic import BaseModel

from reusable_llm_provider.providers import (
    LLMGenerationError,
    StructuredOutputStrategy,
    StructuredOutputValidationError,
    AnthropicProvider,
    OpenAIProvider,
    VertexAIProvider,
    OllamaProvider,
    create_provider,
)


class _Echo(BaseModel):
    name: str


class _StubProvider(AnthropicProvider):
    """AnthropicProvider with the network primitives stubbed out for unit
    tests of the base class's local-validation contract."""

    def __init__(self, candidate, raw=None):
        # Skip parent __init__ to avoid constructing real SDK clients.
        self.NAME = "stub"
        self.STRATEGY = StructuredOutputStrategy.LANGCHAIN_MEDIATED
        self.model = "stub"
        self.max_tokens = 1
        self.temperature = 0.0
        self._candidate = candidate
        self._raw = raw

    def _invoke_structured_candidate(self, prompt, output_model):
        return self._candidate, self._raw


class TestLocalValidation:
    """The base class must always validate the candidate locally."""

    def test_basemodel_candidate_is_revalidated(self):
        provider = _StubProvider(candidate=_Echo(name="ok"))
        result = provider.invoke_structured("p", _Echo)
        assert isinstance(result, _Echo)
        assert result.name == "ok"

    def test_dict_candidate_is_validated(self):
        provider = _StubProvider(candidate={"name": "ok"})
        result = provider.invoke_structured("p", _Echo)
        assert result.name == "ok"

    def test_json_string_candidate_is_validated(self):
        provider = _StubProvider(candidate='{"name": "ok"}')
        result = provider.invoke_structured("p", _Echo)
        assert result.name == "ok"

    def test_invalid_candidate_raises_structured_output_validation_error(self):
        provider = _StubProvider(candidate={"wrong_key": "ok"}, raw="raw-blob")
        with pytest.raises(StructuredOutputValidationError) as exc_info:
            provider.invoke_structured("p", _Echo)
        err = exc_info.value
        assert err.output_model is _Echo
        assert err.strategy is StructuredOutputStrategy.LANGCHAIN_MEDIATED
        assert err.raw == "raw-blob"

    def test_none_candidate_raises_structured_output_validation_error(self):
        provider = _StubProvider(candidate=None, raw=None)
        with pytest.raises(StructuredOutputValidationError):
            provider.invoke_structured("p", _Echo)


class TestLLMGenerationError:
    """Tests for LLMGenerationError exception."""

    def test_error_message_format(self):
        """Test error message includes provider and original error details."""
        original_error = ValueError("API key invalid")
        error = LLMGenerationError("anthropic", original_error)

        assert error.provider == "anthropic"
        assert error.original_error is original_error
        assert "anthropic" in str(error)
        assert "ValueError" in str(error)
        assert "API key invalid" in str(error)

    def test_error_with_different_exception_types(self):
        """Test error handling with different exception types."""
        runtime_error = RuntimeError("Connection timeout")
        error = LLMGenerationError("openai", runtime_error)

        assert "RuntimeError" in str(error)
        assert "Connection timeout" in str(error)


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-haiku",
            anthropic_api_key="test-key",
        )
        provider = create_provider(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-haiku"

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = LLMConfig(
            provider=LLMProviderType.OPENAI,
            model="gpt-4o-mini",
            openai_api_key="test-key",
        )
        provider = create_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_create_vertex_provider(self):
        """Test creating Vertex AI provider."""
        config = LLMConfig(
            provider=LLMProviderType.VERTEX,
            model="gemini-2.5-flash",
            vertex_project_id="my-project",
            vertex_location="us-central1",
        )
        provider = create_provider(config)
        assert isinstance(provider, VertexAIProvider)

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        config = LLMConfig(
            provider=LLMProviderType.OLLAMA,
            model="llama2",
        )
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)

    def test_create_provider_with_invalid_provider(self):
        """Test that invalid provider type raises ValueError."""
        config = LLMConfig(
            provider=None,
            model="some-model",
        )
        # Manually set invalid provider to bypass enum constraint
        config.provider = "invalid"

        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider(config)


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_base_provider_initialization(self):
        """Test that base provider sets up config correctly."""
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-haiku",
            temperature=0.5,
            max_tokens=500,
            anthropic_api_key="test-key",
        )
        # Use Anthropic as concrete implementation
        provider = AnthropicProvider(config)

        assert provider.model == "claude-3-haiku"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 500

    def test_provider_inherits_config_values(self):
        """Test that provider properly inherits config values."""
        config = LLMConfig(
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            openai_api_key="test-key",
        )
        provider = OpenAIProvider(config)

        assert provider.temperature == 0.7
        assert provider.max_tokens == 2000


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initializes with config."""
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-haiku",
            anthropic_api_key="sk-ant-test",
        )
        # Just test that it initializes without error
        provider = AnthropicProvider(config)
        assert provider.model == "claude-3-haiku"

    @patch("reusable_llm_provider.providers.Anthropic")
    def test_anthropic_invoke_wraps_api_errors(self, mock_anthropic):
        """Test that invoke wraps API errors properly."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = RuntimeError("API Error")
        mock_anthropic.return_value = mock_client

        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-haiku",
            anthropic_api_key="test-key",
        )
        provider = AnthropicProvider(config)

        with pytest.raises(LLMGenerationError) as exc_info:
            provider.invoke("test prompt")

        assert exc_info.value.provider == "anthropic"
        assert isinstance(exc_info.value.original_error, RuntimeError)

    @staticmethod
    def _block(block_type: str, text: str | None = None):
        """A stand-in for an SDK content block.

        A thinking block genuinely has no ``text`` attribute, so the stub
        deletes it rather than leaving a Mock auto-attribute in its place —
        otherwise the bug under test cannot reproduce.
        """
        block = Mock()
        block.type = block_type
        if text is None:
            del block.text
        else:
            block.text = text
        return block

    @patch("reusable_llm_provider.providers.Anthropic")
    def test_anthropic_invoke_skips_thinking_blocks(self, mock_anthropic):
        """Extended thinking puts a non-text block first; invoke must skip it.

        Indexing content[0] blindly raises AttributeError on any model that
        returns a thinking block, which _wrap_errors then reports as a
        generation failure rather than the local bug it is.
        """
        response = Mock()
        response.content = [
            self._block("thinking"),
            self._block("text", "The sky is blue."),
        ]
        mock_client = Mock()
        mock_client.messages.create.return_value = response
        mock_anthropic.return_value = mock_client

        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-opus-5",
            anthropic_api_key="test-key",
        )

        assert AnthropicProvider(config).invoke("test prompt") == "The sky is blue."

    @patch("reusable_llm_provider.providers.Anthropic")
    def test_anthropic_invoke_joins_multiple_text_blocks(self, mock_anthropic):
        """All text blocks are returned, not just the first.

        Tool-using turns emit intermediate prose ("let me look that up")
        as its own block; returning only the first drops the real answer.
        """
        response = Mock()
        response.content = [
            self._block("text", "Let me look that up."),
            self._block("tool_use"),
            self._block("text", "The sky is blue."),
        ]
        mock_client = Mock()
        mock_client.messages.create.return_value = response
        mock_anthropic.return_value = mock_client

        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-opus-5",
            anthropic_api_key="test-key",
        )

        result = AnthropicProvider(config).invoke("test prompt")
        assert result == "Let me look that up.\n\nThe sky is blue."


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_openai_provider_initialization(self):
        """Test OpenAI provider initializes with config."""
        config = LLMConfig(
            provider=LLMProviderType.OPENAI,
            model="gpt-4o-mini",
            openai_api_key="sk-test",
            openai_organization="org-123",
        )
        provider = OpenAIProvider(config)
        assert provider.model == "gpt-4o-mini"

    @patch("reusable_llm_provider.providers.OpenAI")
    def test_openai_invoke_wraps_api_errors(self, mock_openai):
        """Test that invoke wraps API errors properly."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API Error")
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProviderType.OPENAI,
            model="gpt-4o-mini",
            openai_api_key="test-key",
        )
        provider = OpenAIProvider(config)

        with pytest.raises(LLMGenerationError) as exc_info:
            provider.invoke("test prompt")

        assert exc_info.value.provider == "openai"


class TestVertexAIProvider:
    """Tests for VertexAIProvider."""

    def test_vertex_provider_initialization(self):
        """Test Vertex AI provider initializes with config."""
        config = LLMConfig(
            provider=LLMProviderType.VERTEX,
            model="gemini-2.5-flash",
            vertex_project_id="my-project",
            vertex_location="us-central1",
        )
        provider = VertexAIProvider(config)
        assert provider.model == "gemini-2.5-flash"


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_ollama_provider_initialization(self):
        """Test Ollama provider initializes with config."""
        config = LLMConfig(
            provider=LLMProviderType.OLLAMA,
            model="llama2",
        )
        provider = OllamaProvider(config)
        assert provider.model == "llama2"


class TestSamplingOmission:
    """The temperature parameter must be absent, not null, when unset.

    Anthropic's Claude 5 family and OpenAI's GPT-5 reasoning models reject the
    field outright; Gemini 3.x accepts and ignores it. Sending null is not
    equivalent to omitting it, so this asserts on the kwargs dict itself
    rather than on behaviour further downstream.
    """

    def _provider(self, temperature):
        from reusable_llm_provider.config import LLMConfig, LLMProviderType
        from reusable_llm_provider.providers import BaseLLMProvider

        class _Bare(BaseLLMProvider):
            NAME = "bare"

            def _invoke_raw_text(self, prompt):  # pragma: no cover
                return ""

            def _invoke_structured_candidate(self, prompt, output_model):
                return None  # pragma: no cover

        return _Bare(
            LLMConfig(
                provider=LLMProviderType.ANTHROPIC,
                model="m",
                temperature=temperature,
            )
        )

    def test_unset_temperature_yields_no_kwargs(self):
        assert self._provider(None)._sampling() == {}

    def test_explicit_zero_is_still_sent(self):
        """Zero is a meaningful value and must not be confused with unset."""
        assert self._provider(0.0)._sampling() == {"temperature": 0.0}

    def test_explicit_value_is_sent(self):
        assert self._provider(0.7)._sampling() == {"temperature": 0.7}

    def test_key_can_be_renamed_for_providers_that_differ(self):
        assert self._provider(0.5)._sampling("temp") == {"temp": 0.5}
