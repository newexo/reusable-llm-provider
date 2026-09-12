"""Pytest configuration that travels with the shipped tests.

The marker is registered here rather than in `pyproject.toml` because these
tests also run from an installed wheel, where `make test-wheel` invokes pytest
from a temporary directory and no project config is found. Registering it there
only would leave the wheel run emitting PytestUnknownMarkWarning.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_backends: requires the vendor SDKs to be importable (the test "
        "patches a vendor symbol, or constructs a provider). Deselected by "
        "`make test-wheel`, which installs a bare wheel.",
    )
