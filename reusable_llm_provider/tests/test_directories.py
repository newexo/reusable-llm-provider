from pathlib import Path

import pytest

from reusable_llm_provider import directories


def _is_source_checkout():
    return directories.base("pyproject.toml").exists()


class TestDirectories:
    """`code()` and `tests()` address the package and hold anywhere it is
    installed. `base()`, `data()` and `secrets()` address the repository, so
    assertions about them are skipped when the tests run from a wheel.
    """

    def test_package_directories_exist(self):
        assert directories.code().is_dir()
        assert directories.tests().is_dir()

    def test_package_filenames(self):
        assert directories.code("__init__.py").exists()
        assert directories.tests("__init__.py").exists()

    @pytest.mark.skipif(
        not _is_source_checkout(),
        reason="base() and secrets() address the repository, not an installed package",
    )
    def test_source_checkout_layout(self):
        assert directories.base().is_dir()
        assert directories.secrets().is_dir()
        assert directories.base("README.md").exists()
        assert directories.secrets("README.md").exists()

    def test_data_helpers_are_addressed_correctly(self):
        """Neither data directory exists yet, so only the shape is asserted.

        `data()` addresses repository data that must never be packaged;
        `package_data()` addresses data inside the package, readable from an
        installed wheel. Keeping both means the distinction is available before
        anything needs it.
        """
        assert directories.data("corpus.csv") == directories.base() / "data/corpus.csv"
        assert (
            directories.package_data("models.json")
            == directories.code() / "data/models.json"
        )
        assert directories.test_data("fixture.json") == (
            directories.tests() / "test_data/fixture.json"
        )

    def test_qualifyname_without_filename_returns_the_directory(self):
        assert directories.qualifyname("/tmp") == Path("/tmp")
        assert directories.qualifyname("/tmp", "x.txt") == Path("/tmp/x.txt")
