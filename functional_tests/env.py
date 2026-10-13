"""Credentials for the live provider tests.

This lives beside the tests rather than in the package because it addresses
`secrets/` in this checkout — a gitignored directory that is not in the wheel
and cannot be. Shipping a loader for it gave every consumer a function that
silently found nothing, and, under an editable install, one that would have
loaded this project's own credentials into theirs.

Paths are anchored to this file rather than to the package's `directories`
helpers, so the tests go on addressing the checkout even when the package under
test is installed from a wheel.
"""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def secrets(filename=None):
    """The repository's gitignored `secrets/` directory."""
    directory = REPO_ROOT / "secrets"
    if filename is None:
        return directory
    return directory / filename


def find_env():
    """Path to `secrets/.env`, or `''` if it is not there."""
    return find_dotenv(secrets(".env"))


def load_env():
    """Load `secrets/.env` into the environment.

    Returns whether anything was loaded, which the caller may want to assert on
    -- the old package-level wrapper discarded this and left "loaded" and
    "found nothing" indistinguishable.
    """
    return load_dotenv(find_env())
