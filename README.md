# reusable-llm-provider

A reusable provider abstraction layer for large language model APIs. It exposes a uniform interface over multiple backends so that application code can remain agnostic to the specific SDK, authentication mechanism or response format of each provider.

## Supported Providers

- **Anthropic** — Claude models via the `anthropic` SDK, with `langchain_anthropic` used for structured (JSON) output.
- **OpenAI** — GPT models via the `openai` SDK, with native JSON mode and a graceful fallback for older models.
- **Vertex AI** — Google Gemini models via the unified `google-genai` SDK with Vertex AI endpoints.
- **Ollama** — Local models via `langchain_ollama`, suitable for offline or self-hosted inference.

All four providers implement the same `LLMProvider` protocol, with two methods: `invoke(prompt)` for free-form text and `invoke_json(prompt)` for structured JSON output.

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

The package follows an **explicit credential injection** pattern. `LLMConfig` accepts credentials directly, so calling code is in full control of where those credentials originate — environment variables, secret managers, configuration files or test fixtures.

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
config = create_openai_config(model="gpt-4o-mini", temperature=0.2)

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

Once a config is available, construct a provider with `create_provider` and invoke it:

```python
from reusable_llm_provider.config import create_anthropic_config
from reusable_llm_provider.providers import create_provider

config = create_anthropic_config()
provider = create_provider(config)

text = provider.invoke("Describe a market town in two sentences.")
print(text)

data = provider.invoke_json(
    "Return a JSON object with keys 'name' and 'population' for a fictional town."
)
print(data["name"], data["population"])
```

The same pattern applies to every supported provider; only the factory function changes.

### Error Handling

All provider methods raise `LLMGenerationError` on failure. The exception preserves the original error as `original_error` and identifies the failing backend by name:

```python
from reusable_llm_provider.providers import LLMGenerationError

try:
    text = provider.invoke(prompt)
except LLMGenerationError as exc:
    print(f"Provider {exc.provider} failed: {exc.original_error}")
```

## Default Models

The factory functions use sensible default models when none is specified:

| Provider  | Default Model                  |
|-----------|--------------------------------|
| Anthropic | `claude-haiku-4-5-20251001`    |
| OpenAI    | `gpt-4o-mini`                  |
| Vertex AI | `gemini-2.5-flash`             |
| Ollama    | `gemma2`                       |

Any model may be specified explicitly via the `model` parameter to either the factory or the `LLMConfig` constructor.

## Development

This project uses Poetry for dependency management.

### Environment Setup

Install dependencies, including the development group:

```bash
poetry install --with dev
```

### Commands

| Command              | Description                          |
|----------------------|--------------------------------------|
| `make test`          | Run the test suite.                  |
| `make format`        | Format the code with Black.          |
| `make lint`          | Run Flake8 checks.                   |
| `make check`         | Run formatting, linting and tests.   |
| `make coverage`      | Run tests with coverage enforcement. |
| `make coverage-html` | Create an HTML coverage report.      |

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

- **Protocol-based interface**: `LLMProvider` is a `typing.Protocol` rather than a required base class, so adapters to additional backends do not need to inherit from a shared class.
- **No implicit global state**: The package does not read environment variables, configuration files or secret stores of its own accord. All such concerns are the caller's responsibility.
- **Thin wrapper**: The abstraction is intentionally minimal. It unifies construction and invocation, but does not attempt to normalize provider-specific features such as tool use, streaming or multimodal inputs.
