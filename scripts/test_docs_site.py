#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for the GitHub Pages portfolio site."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_portfolio_files_exist() -> None:
    for path in [
        DOCS / "index.html",
        DOCS / "assets" / "styles.css",
        DOCS / "assets" / "app.js",
        DOCS / "assets" / "task1_close.png",
        DOCS / "assets" / "task2_rsi.png",
        DOCS / "assets" / "task2_macd.png",
        DOCS / "assets" / "task2_bollinger.png",
        DOCS / "assets" / "task2_kdj.png",
        DOCS / "assets" / "task3_ma_signal.png",
        DOCS / "assets" / "task3_nav.png",
        DOCS / "assets" / "task3_drawdown.png",
        DOCS / "assets" / "task4_channel.png",
        DOCS / "assets" / "task4_atr.png",
        DOCS / "assets" / "task4_equity.png",
        DOCS / "assets" / "task4_params.png",
        DOCS / "assets" / "task5_confusion.png",
        DOCS / "assets" / "task5_roc.png",
        DOCS / "assets" / "task5_metrics.png",
        DOCS / "assets" / "task5_importance.png",
        DOCS / "assets" / "task6_nav.png",
        DOCS / "assets" / "task6_excess.png",
        DOCS / "assets" / "task6_decile.png",
        DOCS / "assets" / "task6_importance.png",
        DOCS / "data" / "task2_indicators.csv",
    ]:
        assert path.exists(), f"missing {path.relative_to(ROOT)}"


def test_index_has_navigation_and_no_private_pdf_links() -> None:
    html = read(DOCS / "index.html")

    for expected in [
        'href="#overview"',
        'href="#task1"',
        'href="#task2"',
        'href="#task3"',
        'href="#task4"',
        'href="#task5"',
        'href="#task6"',
        'href="#dashboard"',
        'id="task3"',
        'id="task4"',
        'id="task5"',
        'id="task6"',
        'id="price-chart"',
        'id="rsi-chart"',
        'id="macd-chart"',
        'id="bollinger-chart"',
        'id="kdj-chart"',
    ]:
        assert expected in html

    blocked = ["private_submissions", ".pdf", "祁彦龙TASK2.pdf", "祁彦龙TASK1.pdf"]
    for text in blocked:
        assert text not in html


def test_public_indicator_data_has_required_columns() -> None:
    header = read(DOCS / "data" / "task2_indicators.csv").splitlines()[0].split(",")

    for column in [
        "trade_date",
        "close",
        "rsi14",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "kdj_k",
        "kdj_d",
        "kdj_j",
    ]:
        assert column in header


if __name__ == "__main__":
    tests = [
        test_portfolio_files_exist,
        test_index_has_navigation_and_no_private_pdf_links,
        test_public_indicator_data_has_required_columns,
    ]
    for test in tests:
        test()
    print("docs site smoke tests passed")
