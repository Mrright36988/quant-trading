#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for TASK3 dual moving-average strategy logic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_task3 import add_ma_signals, extract_trades, perf_metrics, run_backtest


def make_frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=len(closes))
    return pd.DataFrame({"trade_date": dates, "close": closes})


def test_golden_and_death_cross_detected() -> None:
    # Down leg pushes MA2 below MA3, the rebound crosses it back above, then a
    # second slump crosses below again.
    closes = [10, 9, 8, 7, 6, 7.5, 9, 10, 11, 10, 8, 6.5, 5.5]
    df = add_ma_signals(make_frame(closes), short=2, long=3)
    signals = df["signal"].tolist()
    assert signals.count(1) == 1, signals
    assert signals.count(-1) >= 1, signals
    golden_idx = signals.index(1)
    death_idx = signals.index(-1, golden_idx)
    assert golden_idx < death_idx


def test_position_effective_next_day() -> None:
    closes = [10, 9, 8, 7, 6, 7.5, 9, 10, 11, 10, 8, 6.5, 5.5]
    df = add_ma_signals(make_frame(closes), short=2, long=3)
    golden_idx = df.index[df["signal"] == 1][0]
    assert df.loc[golden_idx, "position"] == 0.0
    assert df.loc[golden_idx + 1, "position"] == 1.0


def test_backtest_flat_before_first_signal() -> None:
    closes = [10, 9, 8, 7, 6, 7.5, 9, 10, 11, 10, 8, 6.5, 5.5]
    bt = run_backtest(add_ma_signals(make_frame(closes), short=2, long=3))
    golden_idx = bt.index[bt["signal"] == 1][0]
    assert (bt.loc[:golden_idx, "strategy_ret"] == 0).all()
    assert bt["strategy_nav"].iloc[0] == 1.0


def test_extract_trades_pairs_and_open_position() -> None:
    closes = [10, 9, 8, 7, 6, 7.5, 9, 10, 11, 10, 8, 6.5, 5.5]
    df = add_ma_signals(make_frame(closes), short=2, long=3)
    trades, open_trade = extract_trades(df)
    assert len(trades) == 1
    assert open_trade is None
    trade = trades[0]
    assert math.isclose(trade["ret"], trade["sell_price"] / trade["buy_price"] - 1)


def test_perf_metrics_max_drawdown() -> None:
    nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3])
    returns = nav.pct_change().fillna(0)
    metrics = perf_metrics(returns, nav)
    assert math.isclose(metrics["mdd"], 1 - 0.9 / 1.2)
    assert math.isclose(metrics["total"], 0.3)


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
