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
