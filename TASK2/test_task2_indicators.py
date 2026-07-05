#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for TASK2 indicator calculations."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_task2 import add_indicators, bollinger_bands, kdj, macd, rsi


def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
    assert abs(actual - expected) <= tol, f"{actual=} {expected=}"


def test_bollinger_bands_uses_rolling_mean_and_two_std() -> None:
    close = pd.Series([10, 12, 14, 16, 18], dtype="float64")
    middle, upper, lower = bollinger_bands(close, window=3, num_std=2)

    assert middle.iloc[:2].isna().all()
    assert_close(middle.iloc[-1], 16.0)
    assert_close(upper.iloc[-1], 16.0 + 2 * close.iloc[-3:].std(ddof=0))
    assert_close(lower.iloc[-1], 16.0 - 2 * close.iloc[-3:].std(ddof=0))


def test_macd_returns_dif_dea_and_chinese_style_histogram() -> None:
    close = pd.Series([10, 11, 12, 13, 14, 15], dtype="float64")
    dif, dea, hist = macd(close, fast=3, slow=5, signal=2)
    expected_dif = close.ewm(span=3, adjust=False).mean() - close.ewm(span=5, adjust=False).mean()
    expected_dea = expected_dif.ewm(span=2, adjust=False).mean()

    assert_close(dif.iloc[-1], expected_dif.iloc[-1])
    assert_close(dea.iloc[-1], expected_dea.iloc[-1])
    assert_close(hist.iloc[-1], 2 * (expected_dif.iloc[-1] - expected_dea.iloc[-1]))


def test_rsi_is_high_after_sustained_gains() -> None:
    close = pd.Series(range(1, 21), dtype="float64")
    values = rsi(close, period=14)

    assert values.iloc[:14].isna().all()
    assert_close(values.iloc[-1], 100.0)


def test_kdj_initializes_at_50_and_responds_to_price_position() -> None:
    high = pd.Series([10, 11, 12, 13, 14, 15], dtype="float64")
    low = pd.Series([8, 8, 9, 10, 11, 12], dtype="float64")
    close = pd.Series([9, 10, 11, 12, 13, 14], dtype="float64")
    k, d, j = kdj(high, low, close, window=3)

    assert_close(k.iloc[0], 50.0)
    assert_close(d.iloc[0], 50.0)
    assert k.iloc[-1] > d.iloc[-1]
    assert_close(j.iloc[-1], 3 * k.iloc[-1] - 2 * d.iloc[-1])


def test_add_indicators_preserves_rows_and_adds_columns() -> None:
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=30),
            "open": range(30),
            "high": range(2, 32),
            "low": range(30),
            "close": range(1, 31),
            "vol": range(100, 130),
        }
    )
    result = add_indicators(df)

    assert len(result) == len(df)
    for column in ["rsi14", "macd_dif", "macd_dea", "macd_hist", "bb_middle", "bb_upper", "bb_lower", "kdj_k", "kdj_d", "kdj_j"]:
        assert column in result.columns


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("TASK2 indicator tests passed")
