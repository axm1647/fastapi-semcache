"""Regression tests for the ``fastapi_semcache`` install-name stub module."""

import importlib

import pytest


def test_fastapi_semcache_stub_raises_actionable_import_error() -> None:
    """Importing ``fastapi_semcache`` fails with a hint to use ``semanticcache``."""
    with pytest.raises(
        ImportError,
        match=(
            r"The install name is fastapi-semcache but the import name is "
            r"semanticcache\. Use import semanticcache instead\."
        ),
    ):
        importlib.import_module("fastapi_semcache")
