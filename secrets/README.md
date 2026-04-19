# Secrets directory

This directory holds secrets used only by the functional test suite (API
keys for live provider calls). The library itself does not read from this
directory at import time — consumers are expected to construct their own
`LLMConfig` objects.

If a `.env` file exists here, `reusable_llm_provider.env.load_reusable_llm_provider_env()`
will load it. The `.env` file is gitignored.
