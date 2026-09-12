"""Assert that a backend is usable if and only if its extra is installed.

Usage::

    python scripts/check_extras.py                # expect no backends
    python scripts/check_extras.py anthropic      # expect only Anthropic

This cannot be an ordinary unit test. It asserts what is *absent* from the
environment, and the development environment has every backend installed, so a
test living there could only simulate absence -- which is the failure mode this
whole change is trying to avoid. It therefore runs in CI, in environments built
with exactly one extra (or none).

Exits non-zero on the first mismatch, describing it.
"""

import sys

from reusable_llm_provider.config import DEFAULT_MODELS, LLMConfig, LLMProviderType
from reusable_llm_provider.providers import (
    _PROVIDER_MAP,
    MissingBackendError,
    create_provider,
)

# Importing the module must never pull in a vendor SDK: that is the regression
# this guards. Checked before anything constructs a provider.
VENDOR_MODULES = {
    "anthropic",
    "google",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_ollama",
    "langchain_openai",
    "openai",
}


def _config(provider: LLMProviderType) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        model=DEFAULT_MODELS[provider.value],
        anthropic_api_key="not-a-real-key",
        openai_api_key="not-a-real-key",
        vertex_project_id="not-a-real-project",
        vertex_location="us-central1",
    )


def main(installed: set[str]) -> int:
    failures = []

    leaked = sorted(VENDOR_MODULES & {name.split(".")[0] for name in sys.modules})
    if leaked:
        failures.append(
            "importing reusable_llm_provider.providers loaded vendor modules "
            f"{leaked}; a vendor import has moved back to module scope"
        )

    for provider_type, provider_cls in _PROVIDER_MAP.items():
        extra = provider_cls.NAME
        try:
            create_provider(_config(provider_type))
        except MissingBackendError as exc:
            if extra in installed:
                failures.append(f"{extra}: extra is installed but raised {exc}")
            elif f"[{extra}]" not in str(exc):
                failures.append(f"{extra}: error names the wrong extra: {exc}")
        except Exception as exc:
            # Anything else means the vendor imports succeeded and the SDK
            # objected to the fake credentials, which is what we want to see.
            if extra not in installed:
                failures.append(
                    f"{extra}: extra is not installed but construction got past "
                    f"the imports, raising {type(exc).__name__}: {exc}"
                )
        else:
            if extra not in installed:
                failures.append(
                    f"{extra}: extra is not installed but the provider "
                    "constructed successfully"
                )

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        return 1

    print(f"ok: extras installed = {sorted(installed) or ['none']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:])))
