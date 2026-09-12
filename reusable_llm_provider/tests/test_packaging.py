"""What must hold for an installed package with no backend at all.

A bare install is the supported install: every vendor SDK is an extra, so the
package has to import, name all four providers, and refuse the ones that are
missing with a usable error — none of which requires a backend to be present.

`make test-wheel` runs exactly this file (plus the config, directories and
version tests) against a bare wheel, having deselected the `needs_backends`
suite. Keep it free of vendor imports and of anything that constructs a
provider, or the wheel test stops working.
"""

import importlib.util

import pytest

from reusable_llm_provider.config import DEFAULT_MODELS, LLMConfig, LLMProviderType
from reusable_llm_provider.providers import (
    _PROVIDER_MAP,
    MissingBackendError,
    create_provider,
)

EXTRA_MODULES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "vertex": "google.genai",
    "ollama": "langchain_ollama",
}


def _installed(extra):
    """Whether this extra's SDK can be imported, without importing it."""
    try:
        return importlib.util.find_spec(EXTRA_MODULES[extra]) is not None
    except (ImportError, ValueError):
        return False


class TestBareInstall:
    def test_importing_providers_pulls_in_no_vendor_sdk(self):
        import sys

        import reusable_llm_provider.providers  # noqa: F401

        loaded = {name.split(".")[0] for name in sys.modules}
        assert not loaded & {
            "anthropic",
            "openai",
            "langchain_anthropic",
            "langchain_google_genai",
            "langchain_ollama",
            "langchain_openai",
        }

    def test_every_provider_is_defined_without_its_backend(self):
        """Defining a provider class must not require its SDK.

        This is what keeps `_PROVIDER_MAP` intact on a bare install, so the
        error a caller gets is the actionable one rather than a KeyError.
        """
        assert {cls.NAME for cls in _PROVIDER_MAP.values()} == {
            "anthropic",
            "openai",
            "vertex",
            "ollama",
        }

    def test_missing_backend_error_is_an_import_error(self):
        assert issubclass(MissingBackendError, ImportError)

    @pytest.mark.parametrize("provider_type", list(LLMProviderType))
    def test_absent_backend_names_its_extra(self, provider_type):
        extra = _PROVIDER_MAP[provider_type].NAME
        if _installed(extra):
            pytest.skip(f"the {extra} extra is installed here")

        config = LLMConfig(
            provider=provider_type,
            model=DEFAULT_MODELS[extra],
            anthropic_api_key="not-a-real-key",
            openai_api_key="not-a-real-key",
            vertex_project_id="not-a-real-project",
            vertex_location="us-central1",
        )
        with pytest.raises(MissingBackendError) as exc_info:
            create_provider(config)
        assert f"[{extra}]" in str(exc_info.value)
