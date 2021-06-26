"""Tests for configuration management."""

from reusable_llm_provider.config import (
    LLMConfig,
    LLMProviderType,
    create_anthropic_config,
    create_openai_config,
    create_vertex_config,
    create_ollama_config,
    DEFAULT_MODELS,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_create_config_with_defaults(self):
        """Test creating config with default temperature and max_tokens."""
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-sonnet",
        )
        assert config.provider == LLMProviderType.ANTHROPIC
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000

    def test_create_config_with_custom_values(self):
        """Test creating config with custom temperature and max_tokens."""
        config = LLMConfig(
            provider=LLMProviderType.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            openai_api_key="test-key",
        )
        assert config.provider == LLMProviderType.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.openai_api_key == "test-key"

    def test_config_with_explicit_credentials(self):
        """Test config accepts explicit credential injection."""
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-haiku",
            anthropic_api_key="sk-ant-test",
        )
        assert config.anthropic_api_key == "sk-ant-test"

    def test_vertex_config_credentials(self):
        """Test Vertex AI credentials in config."""
        config = LLMConfig(
            provider=LLMProviderType.VERTEX,
            model="gemini-pro",
            vertex_project_id="my-project",
            vertex_location="us-central1",
        )
        assert config.vertex_project_id == "my-project"
        assert config.vertex_location == "us-central1"


class TestConfigFactory:
    """Tests for config factory functions."""

    def test_create_anthropic_config_with_env_var(self, monkeypatch):
        """Test creating Anthropic config from environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        config = create_anthropic_config()
        assert config.provider == LLMProviderType.ANTHROPIC
        assert config.anthropic_api_key == "test-key-123"
        assert config.model == DEFAULT_MODELS["anthropic"]

    def test_create_anthropic_config_custom_model(self, monkeypatch):
        """Test creating Anthropic config with custom model."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = create_anthropic_config(model="claude-3-opus")
        assert config.model == "claude-3-opus"

    def test_create_anthropic_config_custom_temperature(self, monkeypatch):
        """Test creating Anthropic config with custom temperature."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = create_anthropic_config(temperature=0.5, max_tokens=500)
        assert config.temperature == 0.5
        assert config.max_tokens == 500

    def test_create_openai_config_with_env_vars(self, monkeypatch):
        """Test creating OpenAI config from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_ORGANIZATION", "org-123")
        config = create_openai_config()
        assert config.provider == LLMProviderType.OPENAI
        assert config.openai_api_key == "sk-test-key"
        assert config.openai_organization == "org-123"
        assert config.model == DEFAULT_MODELS["openai"]

    def test_create_vertex_config_with_env_vars(self, monkeypatch):
        """Test creating Vertex AI config from environment variables."""
        monkeypatch.setenv("VERTEX_PROJECT_ID", "my-gcp-project")
        monkeypatch.setenv("VERTEX_LOCATION", "us-east1")
        config = create_vertex_config()
        assert config.provider == LLMProviderType.VERTEX
        assert config.vertex_project_id == "my-gcp-project"
        assert config.vertex_location == "us-east1"
        assert config.model == DEFAULT_MODELS["vertex"]

    def test_create_ollama_config(self):
        """Test creating Ollama config (no environment variables needed)."""
        config = create_ollama_config()
        assert config.provider == LLMProviderType.OLLAMA
        assert config.model == DEFAULT_MODELS["ollama"]

    def test_create_ollama_config_custom_model(self):
        """Test creating Ollama config with custom model."""
        config = create_ollama_config(model="llama2")
        assert config.model == "llama2"

    def test_create_config_with_missing_env_var(self, monkeypatch):
        """Test that config gracefully handles missing environment variables."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = create_anthropic_config()
        assert config.anthropic_api_key is None


class TestProviderType:
    """Tests for LLMProviderType enum."""

    def test_provider_type_values(self):
        """Test that all provider types have correct values."""
        assert LLMProviderType.ANTHROPIC.value == "anthropic"
        assert LLMProviderType.OPENAI.value == "openai"
        assert LLMProviderType.VERTEX.value == "vertex"
        assert LLMProviderType.OLLAMA.value == "ollama"

    def test_provider_type_comparison(self):
        """Test provider type enum comparison."""
        config = LLMConfig(
            provider=LLMProviderType.ANTHROPIC,
            model="claude-3-haiku",
        )
        assert config.provider == LLMProviderType.ANTHROPIC
        assert config.provider != LLMProviderType.OPENAI
