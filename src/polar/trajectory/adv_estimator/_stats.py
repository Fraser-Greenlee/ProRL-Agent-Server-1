"""Shared stats helpers for advantage estimators."""

from __future__ import annotations

import statistics

_EPS = 1e-8


def standardize(values: list[float]) -> list[float]:
    """(x - mean) / std, returns zeros when std ~ 0 or len <= 1."""
    n = len(values)
    if n <= 1:
        return [0.0] * n
    mu = statistics.mean(values)
    sigma = statistics.pstdev(values)
    if sigma < _EPS:
        return [0.0] * n
    return [(v - mu) / sigma for v in values]
