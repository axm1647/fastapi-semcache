"""Tests for ``CacheSettings`` validation unrelated to embedders."""

from __future__ import annotations

import warnings

import pytest

from semanticcache.config import CacheSettings


def test_rejection_threshold_below_primary_raises() -> None:
    """Reject rejection_threshold strictly below threshold."""
    with pytest.raises(ValueError, match="rejection_threshold must be >="):
        CacheSettings(threshold=0.9, rejection_threshold=0.89)


def test_rejection_threshold_equals_primary_warns() -> None:
    """Equality allows validation but warns that stage two has no effect."""
    with pytest.warns(UserWarning, match="rejection_threshold equals threshold"):
        CacheSettings(threshold=0.9, rejection_threshold=0.9)


def test_rejection_threshold_above_primary_no_equality_warning() -> None:
    """Strictly greater rejection threshold does not emit the equality warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        CacheSettings(threshold=0.8, rejection_threshold=0.9)
    equality_warnings = [
        w
        for w in record
        if "rejection_threshold equals threshold" in str(w.message)
    ]
    assert not equality_warnings
