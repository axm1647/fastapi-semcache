"""Unit tests for ``OpenAIEmbedder`` helpers (mocked optional deps)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from semanticcache.embedders import openai as openai_mod
from semanticcache.embedders.openai import OpenAIEmbedder


def test_dimensions_param_omitted_when_send_dimensions_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``send_dimensions_to_api=False`` excludes ``dimensions`` from API kwargs."""
    fake_openai = MagicMock()
    fake_tiktoken = MagicMock()
    enc = MagicMock()
    fake_tiktoken.encoding_for_model = MagicMock(return_value=enc)
    monkeypatch.setattr(
        openai_mod,
        "_require_openai",
        lambda: (fake_openai, fake_tiktoken),
    )

    emb = OpenAIEmbedder(
        model_name="text-embedding-ada-002",
        dimensions=1536,
        send_dimensions_to_api=False,
    )
    assert emb._dimensions_param() == {}


def test_dimensions_param_included_when_send_dimensions_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default behavior forwards declared width as API ``dimensions``."""
    fake_openai = MagicMock()
    fake_tiktoken = MagicMock()
    enc = MagicMock()
    fake_tiktoken.encoding_for_model = MagicMock(return_value=enc)
    monkeypatch.setattr(
        openai_mod,
        "_require_openai",
        lambda: (fake_openai, fake_tiktoken),
    )

    emb = OpenAIEmbedder(dimensions=512, send_dimensions_to_api=True)
    assert emb._dimensions_param() == {"dimensions": 512}


def test_require_openai_import_error_has_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing packages surface an install extra hint."""
    import builtins

    real_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("simulated missing openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ImportError, match=r"fastapi-semcache\[embed-openai\]"):
        OpenAIEmbedder()
