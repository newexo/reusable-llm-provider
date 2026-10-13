"""What must hold for an installed package with no backend at all.

A bare install is the supported install: every vendor SDK is an extra, so the
package has to import, name all four providers, and refuse the ones that are
missing with a usable error — none of which requires a backend to be present.

`make test-wheel` runs exactly this file (plus the config and version tests)
against a bare wheel, having deselected the `needs_backends` suite. Keep it free
of vendor imports and of anything that constructs a provider, or the wheel test
stops working.

Two things carry the weight here, and neither is an assertion:

- **Collection is the regression test.** In the bare wheel, a vendor import that
  moved back to module scope makes importing `providers` raise, so this file
  fails to import and the wheel job goes red.
- **`test_absent_backend_names_its_extra` only executes where a backend really is
  missing**, so it skips in development and runs for real in the wheel job. It is
  the one place the `MissingBackendError` path meets a genuinely absent SDK
  rather than a mock.

Do not add an assertion over `sys.modules` here. Observing what an import pulls
in requires a process that has not already imported it, which no test in this
file can be; the previous attempt passed only because this filename sorts before
`test_providers.py`, whose mocks import the vendor SDKs. That check belongs in a
fresh process and lives in `scripts/check_extras.py`, which CI runs per extra and
with none.
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
    def test_all_four_providers_are_registered(self):
        """Defining a provider class must not require its SDK.

        Trivially true where every extra is installed; in the wheel job it is
        the assertion that `_PROVIDER_MAP` survives a bare install, so a caller
        asking for a missing backend gets `MissingBackendError` and not a
        `KeyError`.
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
