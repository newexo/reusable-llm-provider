"""Functional tests for environment setup and configuration.

Verify that secrets/.env loads and contains the keys required for the
configured provider. These tests only run meaningfully when credentials
are present; otherwise they assert loading does not raise.
"""

import os

from reusable_llm_provider.directories import secrets
from reusable_llm_provider.env import load_reusable_llm_provider_env


class TestEnvironmentSetup:
    def test_env_file_exists(self):
        env_path = secrets(".env")
        assert os.path.exists(env_path), f".env file not found at {env_path}"

    def test_env_loads_without_error(self):
        load_reusable_llm_provider_env()

    def test_anthropic_keys_present_if_configured(self):
        load_reusable_llm_provider_env()
        if os.getenv("LLM_PROVIDER") == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            assert api_key, "ANTHROPIC_API_KEY required but not set"

    def test_openai_keys_present_if_configured(self):
        load_reusable_llm_provider_env()
        if os.getenv("LLM_PROVIDER") == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key, "OPENAI_API_KEY required but not set"

    def test_vertex_keys_present_if_configured(self):
        load_reusable_llm_provider_env()
        if os.getenv("LLM_PROVIDER") == "vertex":
            project_id = os.getenv("VERTEX_PROJECT_ID")
            assert project_id, "VERTEX_PROJECT_ID required but not set"

    def test_temperature_setting(self):
        load_reusable_llm_provider_env()
        temp = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        assert 0.0 <= temp <= 2.0, f"Temperature {temp} outside valid range"

    def test_max_tokens_setting(self):
        load_reusable_llm_provider_env()
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1000"))
        assert max_tokens > 0
