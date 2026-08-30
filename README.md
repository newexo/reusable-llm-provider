# reusable-llm-provider

A reusable provider abstraction layer for large language model APIs. It exposes a uniform interface over multiple backends so application code can stay agnostic to the specific SDK, authentication mechanism, or response format of each provider.

## Public Contract

Every provider implements the same `LLMProvider` protocol with two methods:

- `invoke(prompt) -> str` — free-form text generation.
- `invoke_structured(prompt, output_model) -> BaseModel` — typed output. The caller supplies a Pydantic model class as `output_model` and receives a validated instance of that exact class.

The Pydantic model is the public contract. Whether a backend reaches that result via native JSON-schema decoding, tool calling, constrained decoding, text extraction, or framework mediation is an internal implementation detail. The package always re-validates the candidate against the requested `output_model` locally before returning, so the guarantee rests on local type safety rather than vendor promises.

## Supported Providers

- **Anthropic** — Claude models via the `anthropic` SDK.
- **OpenAI** — GPT models via the `openai` SDK.
- **Vertex AI** — Google Gemini models via the unified `google-genai` SDK with Vertex AI endpoints.
- **Ollama** — Local models via `langchain_ollama`, suitable for offline or self-hosted inference.

## Installation

Install from the GitHub repository using Poetry:

```toml
[tool.poetry.dependencies]
reusable-llm-provider = {git = "https://github.com/newexo/reusable-llm-provider.git", branch = "main"}
```

Or with pip:

```bash
pip install git+https://github.com/newexo/reusable-llm-provider.git
```

## Configuration

The package follows an **explicit credential injection** pattern. `LLMConfig` accepts credentials directly, so calling code is in full control of where those credentials originate — environment variables, secret managers, configuration files, or test fixtures.

For the common case of reading credentials from environment variables, convenience factories are provided:

```python
from reusable_llm_provider.config import (
    create_anthropic_config,
    create_openai_config,
    create_vertex_config,
    create_ollama_config,
)

# Reads ANTHROPIC_API_KEY from the environment
config = create_anthropic_config()

# Reads OPENAI_API_KEY and OPENAI_ORGANIZATION from the environment
config = create_openai_config(model="gpt-5.4-nano", temperature=0.2)

# Reads VERTEX_PROJECT_ID and VERTEX_LOCATION from the environment
config = create_vertex_config()

# Ollama requires no credentials
config = create_ollama_config(model="llama2")
```

The package does **not** load `.env` files or read environment variables implicitly. Callers that wish to use `.env` files should invoke `python-dotenv` in their own application code before constructing a config.

### Direct Construction

For full control over credential sources, `LLMConfig` may be instantiated directly:

```python
from reusable_llm_provider.config import LLMConfig, LLMProviderType

config = LLMConfig(
    provider=LLMProviderType.ANTHROPIC,
    model="claude-haiku-4-5-20251001",
    temperature=0.0,
    max_tokens=1000,
    anthropic_api_key="sk-ant-...",
)
```

## Usage

```python
from pydantic import BaseModel, Field

from reusable_llm_provider.config import create_anthropic_config
from reusable_llm_provider.providers import create_provider


class Town(BaseModel):
    name: str = Field(description="The town's proper name.")
    population: int = Field(description="Approximate number of residents.")


config = create_anthropic_config()
provider = create_provider(config)

text = provider.invoke("Describe a market town in two sentences.")
print(text)

town = provider.invoke_structured(
    "Invent a small fictional market town.",
    output_model=Town,
)
print(town.name, town.population)
```

The same pattern applies to every supported provider; only the factory function changes.

### Error Handling

All provider methods raise `LLMGenerationError` (or a subclass) on failure:

| Exception                          | Meaning                                                                 |
|------------------------------------|-------------------------------------------------------------------------|
| `LLMTransportError`                | Request failed before content was produced (network, auth, rate limit). |
| `LLMProviderGenerationError`       | Provider responded but indicated a generation-side failure or refusal.  |
| `StructuredOutputValidationError`  | Content was returned but did not validate against the requested model.  |
| `LLMGenerationError`               | Base class; raised for any failure not classified above.                |

`StructuredOutputValidationError` carries the provider name, the requested `output_model`, the `StructuredOutputStrategy` that was used, the underlying validation error, and the raw provider response when available:

```python
from reusable_llm_provider.providers import (
    LLMGenerationError,
    StructuredOutputValidationError,
)

try:
    town = provider.invoke_structured(prompt, output_model=Town)
except StructuredOutputValidationError as exc:
    print(f"{exc.provider} produced unparseable output for {exc.output_model.__name__}")
    print(f"  strategy: {exc.strategy.value}")
    print(f"  validation error: {exc.validation_error}")
    print(f"  raw response: {exc.raw!r}")
except LLMGenerationError as exc:
    print(f"Provider {exc.provider} failed: {exc.original_error}")
```

## Default Models

The factory functions use sensible default models when none is specified:

| Provider  | Default Model                  |
|-----------|--------------------------------|
| Anthropic | `claude-haiku-4-5-20251001`    |
| OpenAI    | `gpt-5.4-nano`                 |
| Vertex AI | `gemini-2.5-flash`             |
| Ollama    | `gemma2`                       |

Any model may be specified explicitly via the `model` parameter to either the factory or the `LLMConfig` constructor.

## Development

This project uses Poetry for dependency management.

### Environment Setup

```bash
poetry install --with dev
```

### Commands

| Command                | Description                                    |
|------------------------|------------------------------------------------|
| `make test`            | Run the unit test suite.                       |
| `make functional-test` | Run live tests against real LLM providers.     |
| `make format`          | Format the code with Ruff.                     |
| `make lint`            | Run Ruff lint checks.                          |
| `make check`           | Run formatting, linting, and tests.            |
| `make coverage`        | Run tests with coverage enforcement.           |
| `make coverage-html`   | Create an HTML coverage report.                |

### Functional Tests

Functional tests under `functional_tests/` make live calls to real LLM providers and are excluded from the default `make test` target and from CI. Run them locally with:

```bash
make functional-test
```

Requirements:

- A `.env` file at `secrets/.env` containing the API keys for the providers being exercised. `load_reusable_llm_provider_env()` (called from `functional_tests/conftest.py`) loads this file at collection time. Recognized keys include `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_ORGANIZATION`, `VERTEX_PROJECT_ID`, and `VERTEX_LOCATION`.
- For Vertex AI, a valid `gcloud` authentication (`gcloud auth application-default login`).
- For the Ollama tests, a running local Ollama server with the default model pulled (`ollama pull gemma2`).

The `secrets/` directory is gitignored apart from its README; the `.env` file never leaves your machine. Tests for providers whose credentials are absent will fail fast rather than skip — this is intentional, so that a partial configuration is visible rather than silently ignored.

## Project Structure

```
reusable-llm-provider/
    reusable_llm_provider/
        __init__.py
        _version.py
        config.py          # LLMConfig and factory functions
        providers.py       # Provider implementations and create_provider
        tests/
            test_config.py
            test_providers.py
            test_version.py
    pyproject.toml
    README.md
```

## Design Notes

- **Public contract is Pydantic-typed, not JSON.** The caller supplies a model class and receives a validated instance. JSON Schema may be derived internally from the model when a backend requires it, but is not part of the public surface.
- **Local validation is mandatory.** Even when a strategy claims native schema enforcement, the candidate value is always re-validated against `output_model` before being returned. The shim's contract is grounded in local type safety, not vendor promises.
- **Strategies are interchangeable internal details.** Different providers may use different mechanisms to produce structured output: native JSON-schema decoding, tool calling, constrained decoding, text extraction, or framework mediation. These are modeled by `StructuredOutputStrategy` so maintainers can reason about and swap them as provider ecosystems evolve. Today every provider uses the `LANGCHAIN_MEDIATED` strategy via `with_structured_output`; a future migration to native paths will not change the public contract.
- **No framework leakage.** LangChain, when used internally, is treated as one possible adapter. Its result shapes (`{"parsed", "raw", "parsing_error"}`, parser objects, method-specific naming) do not appear in the public interface.
- **Protocol-based interface.** `LLMProvider` is a `typing.Protocol`, so adapters to additional backends do not need to inherit from a shared class.
- **No implicit global state.** The package does not read environment variables, configuration files, or secret stores of its own accord.
- **Thin wrapper.** The abstraction is intentionally minimal. It unifies construction and invocation, but does not attempt to normalize provider-specific features such as streaming or multimodal inputs.
