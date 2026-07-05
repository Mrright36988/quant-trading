#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch one year of A-share daily data from Tushare and draw close-price chart.

Usage:
    export TUSHARE_TOKEN="your token"
    python TASK1/fetch_tushare_stock.py --ts-code 000776.SZ --name 广发证券

The token is read from the environment and is never written to output files.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts-code", default="000776.SZ", help="Tushare code, e.g. 000776.SZ")
    parser.add_argument("--name", default="广发证券", help="Stock name used in chart title")
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--days", type=int, default=365)
    return parser.parse_args()


def _patch_legacy_tushare_for_modern_pandas() -> None:
    if hasattr(pd.DataFrame, "append"):
        return

    def _append(self, other, ignore_index: bool = False, **_: object):
        frame = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, frame], ignore_index=ignore_index)

    pd.DataFrame.append = _append  # type: ignore[attr-defined]


def _fetch_daily_legacy(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    _patch_legacy_tushare_for_modern_pandas()
    code = ts_code.split(".", 1)[0]
    start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    df = ts.get_k_data(code, start=start, end=end)
    if df.empty:
        raise RuntimeError(f"No legacy daily data returned for {ts_code} from {start_date} to {end_date}")
    return (
        df.rename(columns={"date": "trade_date", "volume": "vol"})
        .assign(
            ts_code=ts_code,
            trade_date=lambda x: x["trade_date"].str.replace("-", "", regex=False),
            pre_close=lambda x: x["close"].shift(1),
            change=lambda x: x["close"] - x["pre_close"],
            pct_chg=lambda x: x["change"] / x["pre_close"] * 100,
            amount=pd.NA,
            data_source="tushare_legacy_get_k_data",
        )
        [[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "data_source",
        ]]
        .reset_index(drop=True)
    )


def fetch_daily(ts_code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not set")
    pro = ts.pro_api(token)
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            raise RuntimeError(f"No Pro daily data returned for {ts_code} from {start_date} to {end_date}")
        return (
            df.sort_values("trade_date").assign(data_source="tushare_pro_daily").reset_index(drop=True),
            "tushare_pro_daily",
        )
    except Exception as exc:
        message = str(exc)
        permission_denied = "没有接口(daily)访问权限" in message or "权限" in message
        if not permission_denied:
            raise
        return _fetch_daily_legacy(ts_code, start_date, end_date), "tushare_legacy_get_k_data"


def draw_close_chart(df: pd.DataFrame, title: str, output: Path) -> None:
    width, height = 1200, 720
    margin_l, margin_r, margin_t, margin_b = 95, 45, 82, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(30)
    font = load_font(20)
    font_small = load_font(16)

    closes = df["close"].astype(float).tolist()
    dates = df["trade_date"].astype(str).tolist()
    lo, hi = min(closes), max(closes)
    pad = (hi - lo) * 0.08 or 1.0
    y_min, y_max = lo - pad, hi + pad

    def x_at(i: int) -> float:
        return margin_l + (plot_w * i / max(len(closes) - 1, 1))

    def y_at(v: float) -> float:
        return margin_t + plot_h * (1 - (v - y_min) / (y_max - y_min))

    draw.text((width // 2, 32), title, font=font_title, fill="#111111", anchor="mm")
    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], outline="#333333", width=2)

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_at(value)
        draw.line([(margin_l, y), (width - margin_r, y)], fill="#e6e6e6", width=1)
        draw.text((margin_l - 12, y), f"{value:.2f}", font=font_small, fill="#333333", anchor="rm")

    label_count = 6
    for tick in range(label_count):
        i = round((len(dates) - 1) * tick / (label_count - 1))
        x = x_at(i)
        label = f"{dates[i][:4]}-{dates[i][4:6]}-{dates[i][6:]}"
        draw.line([(x, height - margin_b), (x, height - margin_b + 6)], fill="#333333", width=1)
        draw.text((x, height - margin_b + 13), label, font=font_small, fill="#333333", anchor="ma")

    points = [(x_at(i), y_at(v)) for i, v in enumerate(closes)]
    if len(points) >= 2:
        draw.line(points, fill="#1f77b4", width=4)
    for point in [points[0], points[-1]]:
        x, y = point
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#d62728")

    start_close, end_close = closes[0], closes[-1]
    change_pct = (end_close / start_close - 1) * 100
    summary = (
        f"样本：{len(df)} 个交易日    "
        f"起始收盘价：{start_close:.2f}    "
        f"期末收盘价：{end_close:.2f}    "
        f"区间涨跌幅：{change_pct:.2f}%"
    )
    draw.text((margin_l, height - 38), summary, font=font, fill="#111111")
    image.save(output)


def main() -> None:
    args = parse_args()
    end = pd.to_datetime(args.end_date).date()
    start = end - timedelta(days=args.days)
    start_date = start.strftime("%Y%m%d")

    df, source = fetch_daily(args.ts_code, start_date, args.end_date)
    csv_path = OUT_DIR / f"{args.ts_code.replace('.', '_')}_daily.csv"
    chart_path = OUT_DIR / f"{args.ts_code.replace('.', '_')}_close.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    draw_close_chart(
        df,
        f"图1 {args.name}（{args.ts_code}）过去一年每日收盘价曲线",
        chart_path,
    )
    print(f"rows={len(df)}")
    print(f"source={source}")
    print(f"date_range={df['trade_date'].iloc[0]}..{df['trade_date'].iloc[-1]}")
    print(f"csv={csv_path}")
    print(f"chart={chart_path}")


if __name__ == "__main__":
    main()
