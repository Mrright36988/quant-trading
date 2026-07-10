#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TASK3 dual moving-average strategy assets and PDF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN_DATA_PATH = ROOT / "TASK1" / "000776_SZ_daily.csv"
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
FONT_PATHS = [
    SONGTI,
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

TRADING_DAYS = 252
SHORT_WINDOW = 5
LONG_WINDOW = 15
PARAM_GRID = [(5, 15), (5, 20), (10, 30), (20, 60)]
STOCKS = [
    ("000776.SZ", "广发证券"),
    ("600036.SH", "招商银行"),
    ("600519.SH", "贵州茅台"),
]


# ---------------------------------------------------------------- data loading

def _patch_legacy_tushare_for_modern_pandas() -> None:
    if hasattr(pd.DataFrame, "append"):
        return

    def _append(self, other, ignore_index: bool = False, **_: object):
        frame = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, frame], ignore_index=ignore_index)

    pd.DataFrame.append = _append  # type: ignore[attr-defined]


def fetch_daily_legacy(ts_code: str, start: str, end: str) -> pd.DataFrame:
    import tushare as ts

    _patch_legacy_tushare_for_modern_pandas()
    code = ts_code.split(".", 1)[0]
    df = ts.get_k_data(code, start=start, end=end)
    if df.empty:
        raise RuntimeError(f"No legacy daily data returned for {ts_code} from {start} to {end}")
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
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount", "data_source",
        ]]
        .reset_index(drop=True)
    )


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"].astype(str), format="%Y%m%d")
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("trade_date").reset_index(drop=True)


def load_price_data(ts_code: str) -> pd.DataFrame:
    """Load stored daily data; fetch and cache it under TASK3/ when absent."""
    if ts_code == "000776.SZ":
        path = MAIN_DATA_PATH
    else:
        path = HERE / f"{ts_code.replace('.', '_')}_daily.csv"
    if not path.exists():
        main = pd.read_csv(MAIN_DATA_PATH, encoding="utf-8-sig")
        dates = pd.to_datetime(main["trade_date"].astype(str), format="%Y%m%d")
        start = dates.min().strftime("%Y-%m-%d")
        end = dates.max().strftime("%Y-%m-%d")
        fetched = fetch_daily_legacy(ts_code, start, end)
        fetched.to_csv(path, index=False, encoding="utf-8-sig")
    return normalize_price_frame(pd.read_csv(path, encoding="utf-8-sig"))


# ---------------------------------------------------------- strategy & backtest

def add_ma_signals(df: pd.DataFrame, short: int = SHORT_WINDOW, long: int = LONG_WINDOW) -> pd.DataFrame:
    """Add MA columns, golden/death cross signals and next-day-effective position."""
    result = df.copy()
    close = result["close"].astype(float)
    result["ma_short"] = close.rolling(window=short, min_periods=short).mean()
    result["ma_long"] = close.rolling(window=long, min_periods=long).mean()

    above = result["ma_short"] > result["ma_long"]
    prev_above = above.shift(1)
    valid = (
        result["ma_short"].notna()
        & result["ma_long"].notna()
        & result["ma_short"].shift(1).notna()
        & result["ma_long"].shift(1).notna()
    )
    result["signal"] = 0
    result.loc[valid & above & (prev_above == False), "signal"] = 1  # noqa: E712 golden cross
    result.loc[valid & ~above & (prev_above == True), "signal"] = -1  # noqa: E712 death cross

    state = pd.Series(float("nan"), index=result.index, dtype="float64")
    state[result["signal"] == 1] = 1.0
    state[result["signal"] == -1] = 0.0
    # Signal fires at day-t close, so the position earns returns from day t+1.
    result["position"] = state.ffill().fillna(0.0).shift(1).fillna(0.0)
    return result


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    daily_ret = result["close"].astype(float).pct_change().fillna(0.0)
    result["daily_ret"] = daily_ret
    result["strategy_ret"] = result["position"] * daily_ret
    result["strategy_nav"] = (1 + result["strategy_ret"]).cumprod()
    result["bh_nav"] = (1 + daily_ret).cumprod()
    result["strategy_dd"] = 1 - result["strategy_nav"] / result["strategy_nav"].cummax()
    result["bh_dd"] = 1 - result["bh_nav"] / result["bh_nav"].cummax()
    return result


def extract_trades(df: pd.DataFrame) -> tuple[list[dict], dict | None]:
    """Pair golden-cross buys with death-cross sells at signal-day close prices."""
    trades: list[dict] = []
    entry: dict | None = None
    for row in df.itertuples():
        if row.signal == 1 and entry is None:
            entry = {"buy_date": row.trade_date, "buy_price": float(row.close)}
        elif row.signal == -1 and entry is not None:
            trades.append({
                **entry,
                "sell_date": row.trade_date,
                "sell_price": float(row.close),
                "ret": float(row.close) / entry["buy_price"] - 1,
            })
            entry = None
    open_trade = None
    if entry is not None:
        last = df.iloc[-1]
        open_trade = {
            **entry,
            "sell_date": last["trade_date"],
            "sell_price": float(last["close"]),
            "ret": float(last["close"]) / entry["buy_price"] - 1,
        }
    return trades, open_trade


def perf_metrics(returns: pd.Series, nav: pd.Series) -> dict[str, float]:
    n = len(returns)
    total = float(nav.iloc[-1]) - 1
    annual = float(nav.iloc[-1]) ** (TRADING_DAYS / n) - 1 if n else math.nan
    vol = float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    sharpe = float(returns.mean()) * TRADING_DAYS / vol if vol > 0 else math.nan
    mdd = float((1 - nav / nav.cummax()).max())
    return {"total": total, "annual": annual, "vol": vol, "sharpe": sharpe, "mdd": mdd}


def evaluate(df: pd.DataFrame, short: int, long: int) -> dict:
    bt = run_backtest(add_ma_signals(df, short, long))
    trades, open_trade = extract_trades(bt)
    closed = trades + ([open_trade] if open_trade else [])
    wins = sum(1 for t in closed if t["ret"] > 0)
    return {
        "bt": bt,
        "trades": trades,
        "open_trade": open_trade,
        "n_trades": len(closed),
        "win_rate": wins / len(closed) if closed else math.nan,
        "strategy": perf_metrics(bt["strategy_ret"], bt["strategy_nav"]),
        "buy_hold": perf_metrics(bt["daily_ret"], bt["bh_nav"]),
    }


# ------------------------------------------------------------------- charting

def load_image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fmt_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def draw_axes(
    draw: ImageDraw.ImageDraw,
    dates: Sequence[pd.Timestamp],
    title: str,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    margins: tuple[int, int, int, int],
    y_fmt: str = "{:.2f}",
) -> tuple[callable, callable]:
    margin_l, margin_r, margin_t, margin_b = margins
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    font_title = load_image_font(30)
    font_small = load_image_font(17)
    draw.text((width // 2, 34), title, font=font_title, fill="#111111", anchor="mm")
    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], outline="#333333", width=2)

    def x_at(i: int) -> float:
        return margin_l + plot_w * i / max(len(dates) - 1, 1)

    def y_at(v: float) -> float:
        return margin_t + plot_h * (1 - (v - y_min) / (y_max - y_min))

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_at(value)
        draw.line([(margin_l, y), (width - margin_r, y)], fill="#e8e8e8", width=1)
        draw.text((margin_l - 12, y), y_fmt.format(value), font=font_small, fill="#333333", anchor="rm")

    for tick in range(6):
        i = round((len(dates) - 1) * tick / 5)
        x = x_at(i)
        draw.line([(x, height - margin_b), (x, height - margin_b + 7)], fill="#333333", width=1)
        draw.text((x, height - margin_b + 16), fmt_date(dates[i]), font=font_small, fill="#333333", anchor="ma")
    return x_at, y_at


def draw_legend(draw: ImageDraw.ImageDraw, entries: Sequence[tuple[str, str]], x: int, y: int) -> None:
    font = load_image_font(18)
    cursor = x
    for label, color in entries:
        draw.line([(cursor, y + 9), (cursor + 34, y + 9)], fill=color, width=4)
        draw.text((cursor + 42, y + 9), label, font=font, fill="#111111", anchor="lm")
        cursor += 42 + len(label) * 18 + 24


def _plot_series(draw: ImageDraw.ImageDraw, series: pd.Series, color: str, x_at, y_at, width: int = 3) -> None:
    last: tuple[float, float] | None = None
    for i, value in enumerate(series):
        if pd.isna(value):
            last = None
            continue
        point = (x_at(i), y_at(float(value)))
        if last:
            draw.line([last, point], fill=color, width=width)
        last = point


def draw_signal_chart(df: pd.DataFrame, title: str, short: int, long: int, output: Path) -> None:
    width, height = 1300, 760
    margins = (95, 45, 88, 90)
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = [
        float(v)
        for column in ("close", "ma_short", "ma_long")
        for v in df[column].dropna()
    ]
    y_min, y_max = min(values), max(values)
    pad = (y_max - y_min) * 0.12 or 1.0
    y_min, y_max = y_min - pad, y_max + pad
    x_at, y_at = draw_axes(draw, df["trade_date"].tolist(), title, y_min, y_max, width, height, margins)

    _plot_series(draw, df["close"].astype(float), "#111111", x_at, y_at)
    _plot_series(draw, df["ma_short"], "#1f77b4", x_at, y_at)
    _plot_series(draw, df["ma_long"], "#ff7f0e", x_at, y_at)

    marker = (y_max - y_min) * 0.035
    font_marker = load_image_font(17)
    for i, row in enumerate(df.itertuples()):
        if row.signal == 0:
            continue
        x = x_at(i)
        close = float(row.close)
        if row.signal == 1:
            top = y_at(close) + 14
            draw.polygon([(x, top), (x - 9, top + 16), (x + 9, top + 16)], fill="#c33c2e")
            draw.text((x, top + 20), "买", font=font_marker, fill="#c33c2e", anchor="ma")
        else:
            bottom = y_at(close) - 14
            draw.polygon([(x, bottom), (x - 9, bottom - 16), (x + 9, bottom - 16)], fill="#2f8f4e")
            draw.text((x, bottom - 22), "卖", font=font_marker, fill="#2f8f4e", anchor="ms")

    draw_legend(
        draw,
        [
            ("收盘价", "#111111"),
            (f"MA{short}", "#1f77b4"),
            (f"MA{long}", "#ff7f0e"),
            ("金叉买入", "#c33c2e"),
            ("死叉卖出", "#2f8f4e"),
        ],
        margins[0],
        64,
    )
    image.save(output)


def draw_nav_chart(df: pd.DataFrame, title: str, output: Path) -> None:
    width, height = 1300, 760
    margins = (95, 45, 88, 90)
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = [float(v) for column in ("strategy_nav", "bh_nav") for v in df[column].dropna()]
    values.append(1.0)
    y_min, y_max = min(values), max(values)
    pad = (y_max - y_min) * 0.08 or 0.1
    x_at, y_at = draw_axes(draw, df["trade_date"].tolist(), title, y_min - pad, y_max + pad, width, height, margins)

    one_y = y_at(1.0)
    draw.line([(margins[0], one_y), (width - margins[1], one_y)], fill="#999999", width=1)
    _plot_series(draw, df["bh_nav"], "#8a8a8a", x_at, y_at)
    _plot_series(draw, df["strategy_nav"], "#c33c2e", x_at, y_at)
    draw_legend(draw, [("双均线策略净值", "#c33c2e"), ("买入持有净值", "#8a8a8a")], margins[0], 64)
    image.save(output)


def draw_drawdown_chart(df: pd.DataFrame, title: str, output: Path) -> None:
    width, height = 1300, 640
    margins = (95, 45, 88, 90)
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    dd = -df["strategy_dd"].astype(float) * 100
    y_min = float(dd.min()) * 1.15 or -1.0
    x_at, y_at = draw_axes(
        draw, df["trade_date"].tolist(), title, y_min, 0.5, width, height, margins, y_fmt="{:.1f}%"
    )
    zero_y = y_at(0)
    for i, value in enumerate(dd):
        x = x_at(i)
        draw.line([(x, zero_y), (x, y_at(float(value)))], fill="#e08a80", width=2)
    _plot_series(draw, dd, "#c33c2e", x_at, y_at, width=2)

    trough = int(dd.idxmin())
    draw.ellipse(
        [x_at(trough) - 6, y_at(float(dd.loc[trough])) - 6, x_at(trough) + 6, y_at(float(dd.loc[trough])) + 6],
        outline="#7a1f14",
        width=3,
    )
    draw_legend(draw, [("策略回撤幅度", "#c33c2e")], margins[0], 64)
    image.save(output)


# ------------------------------------------------------------------ pdf tables

def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def main_metrics_table_data(result: dict) -> list[list[str]]:
    strat, bh = result["strategy"], result["buy_hold"]
    return [
        ["指标", "双均线策略(5/15)", "买入持有基准"],
        ["累计回报", pct(strat["total"]), pct(bh["total"])],
        ["年化收益率", pct(strat["annual"]), pct(bh["annual"])],
        ["年化波动率", pct(strat["vol"]), pct(bh["vol"])],
        ["夏普比率(rf=0)", f"{strat['sharpe']:.2f}", f"{bh['sharpe']:.2f}"],
        ["最大回撤", pct(strat["mdd"]), pct(bh["mdd"])],
        ["交易次数", str(result["n_trades"]), "1(期初买入)"],
        ["胜率", pct(result["win_rate"], 1), "—"],
    ]


def sweep_table_data(sweep: list[dict]) -> list[list[str]]:
    rows = [["股票", "短/长均线", "策略累计回报", "策略年化", "最大回撤", "夏普", "交易次数", "买入持有回报"]]
    for item in sweep:
        strat = item["result"]["strategy"]
        rows.append([
            item["name"],
            f"{item['short']}/{item['long']}",
            pct(strat["total"]),
            pct(strat["annual"]),
            pct(strat["mdd"]),
            f"{strat['sharpe']:.2f}",
            str(item["result"]["n_trades"]),
            pct(item["result"]["buy_hold"]["total"]),
        ])
    return rows


def build_table(data: list[list[str]], font: str, col_widths: Sequence[float], font_size: float = 8.7) -> Table:
    table = Table(data, colWidths=list(col_widths), repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size * 1.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def register_pdf_font() -> str:
    if Path(SONGTI).exists():
        pdfmetrics.registerFont(TTFont("Songti", SONGTI))
        return "Songti"
    return "Helvetica"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


# ------------------------------------------------------------------- pdf body

def compare_word(a: float, b: float) -> str:
    return "高于" if a > b else "低于"


def build_pdf(
    student_name: str,
    main_result: dict,
    results_515: dict[str, dict],
    sweep: list[dict],
    charts: dict[str, Path],
    output: Path,
) -> None:
    font = register_pdf_font()
    body = ParagraphStyle(
        "body",
        fontName=font,
        fontSize=10.5,
        leading=15.75,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        spaceBefore=0,
        spaceAfter=0,
        firstLineIndent=21,
    )
    heading = ParagraphStyle("heading", parent=body, firstLineIndent=0)
    title = ParagraphStyle("title", parent=body, fontSize=14, leading=21, alignment=TA_CENTER, firstLineIndent=0)
    caption = ParagraphStyle("caption", parent=body, alignment=TA_CENTER, firstLineIndent=0)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        title=f"{student_name}TASK3",
        author=student_name,
    )

    bt = main_result["bt"]
    strat, bh = main_result["strategy"], main_result["buy_hold"]
    rows = len(bt)
    start_date = fmt_date(bt["trade_date"].iloc[0])
    end_date = fmt_date(bt["trade_date"].iloc[-1])
    n_golden = int((bt["signal"] == 1).sum())
    n_death = int((bt["signal"] == -1).sum())
    dd_trough = bt.loc[bt["strategy_dd"].idxmax()]
    cmb = results_515["600036.SH"]
    mt = results_515["600519.SH"]

    closed = main_result["trades"] + ([main_result["open_trade"]] if main_result["open_trade"] else [])
    best_trade = max(closed, key=lambda t: t["ret"])
    n_loss = sum(1 for t in closed if t["ret"] <= 0)
    best_trade_desc = (
        f" {fmt_date(best_trade['buy_date'])} 建仓、期末仍持有的一笔浮盈 {pct(best_trade['ret'])} "
        if best_trade is main_result["open_trade"]
        else f" {fmt_date(best_trade['buy_date'])} 至 {fmt_date(best_trade['sell_date'])} 的一笔 {pct(best_trade['ret'])} "
    )
    cmb_long = next(
        item["result"] for item in sweep if item["name"] == "招商银行" and (item["short"], item["long"]) == (20, 60)
    )

    story = [
        p(f"{student_name}TASK3", title),
        Spacer(1, 0.24 * cm),
        p("一、双均线策略与金叉、死叉", heading),
        p(
            "双均线策略（Dual Moving Average Crossover）是最经典的趋势跟踪策略之一。它使用两条不同周期的移动平均线："
            "短期均线（如 5 日线 MA5）反应灵敏，贴近最新价格；长期均线（如 15 日线 MA15）平滑滞后，代表中期趋势方向。"
            "两条均线的相对位置和交叉行为被用来判断趋势的启动与结束，并据此产生买卖信号。",
            body,
        ),
        p(
            "金叉（Golden Cross）指短期均线自下而上穿过长期均线。它说明近期价格的平均水平已经超过了中期平均水平，"
            "短期动能转强，市场可能进入上升趋势，因此被视为买入信号。死叉（Death Cross）与之相反，"
            "指短期均线自上而下跌破长期均线，说明短期动能衰竭、价格回落到中期均值之下，趋势可能转弱，因此被视为卖出信号。"
            "本策略的完整交易规则是：出现金叉时按当日收盘价买入并持有，出现死叉时按当日收盘价卖出并空仓，"
            "不做杠杆、不做卖空，信号在次一交易日开始计入收益，以避免使用未来数据。",
            body,
        ),
        p(
            "双均线策略的本质是用均线交叉把“趋势”这一模糊概念转化为可执行的量化规则：趋势一旦形成就顺势持有，"
            "趋势破坏就立即离场。它的优点是规则简单、纪律性强、能吃到大级别趋势的主要部分；"
            "缺点是均线天然滞后，买在趋势启动之后、卖在趋势结束之后，且在无趋势的震荡市中会因反复交叉而频繁出现假信号。",
            body,
        ),
        Spacer(1, 0.16 * cm),
        p("二、策略评价的基础指标", heading),
        p(
            "累计回报（Cumulative Return）是回测期内策略净值的总涨跌幅，计算公式为 期末净值/期初净值-1。"
            "它回答“这段时间总共赚了多少”，是最直观的收益指标；为了在不同长度的回测期之间比较，"
            "通常还会换算成年化收益率，即把累计回报按每年约 252 个交易日折算到一年的复利水平。",
            body,
        ),
        p(
            "最大回撤（Maximum Drawdown，MDD）是回测期内净值从任一历史最高点回落到其后最低点的最大跌幅，"
            "计算方法是对每个时点求 1-当前净值/历史最高净值，取全期最大值。它刻画策略“最深的坑”，"
            "衡量最坏情况下投资者需要承受的账面亏损，是最重要的风险指标之一：回撤越深，恢复所需的涨幅越大"
            "（例如回撤 50% 需要上涨 100% 才能回本），投资者中途放弃的可能性也越大。",
            body,
        ),
        p(
            "夏普比率（Sharpe Ratio）衡量每承担一单位波动风险所获得的超额收益，计算公式为 "
            "(策略年化收益率-无风险利率)/年化波动率，其中年化波动率为日收益率标准差乘以√252。"
            "本报告为简化取无风险利率为 0。夏普比率把收益和风险放在同一把尺子上：两个策略收益相同时，"
            "波动更小的策略夏普更高、质量更好。一般认为夏普比率大于 1 的策略具有较好的风险调整后收益。"
            "此外，本报告还统计交易次数与胜率（盈利交易占全部交易的比例），用于观察策略信号的频率与可靠程度。",
            body,
        ),
        PageBreak(),
        p("三、Python 实现与回测结果", heading),
        p(
            f"程序首先加载 TASK1 已存储的广发证券（000776.SZ）日线数据 TASK1/000776_SZ_daily.csv，"
            f"共 {rows} 个交易日，区间为 {start_date} 至 {end_date}。然后用收盘价的滚动均值计算 MA{SHORT_WINDOW} 与 "
            f"MA{LONG_WINDOW} 两条均线，并按“前一日短均线在长均线之下、当日转到之上”识别金叉（买入信号），"
            f"反向识别死叉（卖出信号）。样本期内共出现 {n_golden} 次金叉和 {n_death} 次死叉。"
            f"信号、均线与仓位的完整结果保存为 TASK3/000776_SZ_ma_signals.csv。",
            body,
        ),
        KeepTogether([
            Image(str(charts["signal_main"]), width=15.5 * cm, height=9.05 * cm),
            p(f"图1 广发证券收盘价、MA{SHORT_WINDOW}/MA{LONG_WINDOW} 均线与金叉死叉信号", caption),
        ]),
        p(
            f"图1中黑线为收盘价，蓝线为 MA{SHORT_WINDOW}，橙线为 MA{LONG_WINDOW}；红色三角标记金叉买入点，"
            f"绿色三角标记死叉卖出点。可以看到：在几段明显的上涨行情中，金叉出现在趋势启动初期，策略得以持仓吃到主升段；"
            f"而在价格横盘或反复震荡的阶段，两条均线纠缠在一起，金叉死叉交替出现，形成多次快进快出的假信号，"
            f"这正是双均线策略最典型的两面性。",
            body,
        ),
        p(
            f"回测采用“信号次日生效”的规则：金叉当日收盘价买入后，从下一交易日起将标的日收益计入策略；"
            f"死叉后空仓，收益记为 0。回测不考虑交易成本与滑点，不加杠杆。策略净值与买入持有基准的对比见图2，"
            f"策略逐日回撤见图3，主要评价指标汇总于表1。",
            body,
        ),
        KeepTogether([
            Image(str(charts["nav_main"]), width=15.5 * cm, height=9.05 * cm),
            p("图2 广发证券双均线策略净值与买入持有净值对比", caption),
        ]),
        p(
            f"图2中红线为双均线策略净值，灰线为期初买入并一直持有的基准净值。样本期广发证券整体上行，"
            f"买入持有累计回报为 {pct(bh['total'])}；双均线策略累计回报为 {pct(strat['total'])}，"
            f"{compare_word(strat['total'], bh['total'])}基准。策略净值曲线在下跌阶段明显更平缓，"
            f"因为死叉信号让策略在部分回调期间空仓躲过了下跌；代价有两个：一是样本开头 7 月的快速上涨"
            f"发生在均线尚未形成、策略等待首次金叉的空仓期（首次买入迟至 {fmt_date(closed[0]['buy_date'])}），"
            f"这段涨幅被完全踏空；二是每次趋势重新启动时都要等金叉确认后才重新进场，"
            f"少赚趋势初段的涨幅，震荡期的假信号还造成了多次小额亏损。",
            body,
        ),
        PageBreak(),
        KeepTogether([
            Image(str(charts["dd_main"]), width=15.5 * cm, height=7.6 * cm),
            p("图3 广发证券双均线策略逐日回撤曲线", caption),
        ]),
        p(
            f"图3展示策略净值相对历史最高点的回落幅度。策略最大回撤为 {pct(strat['mdd'])}，"
            f"出现在 {fmt_date(dd_trough['trade_date'])}，明显小于买入持有基准的最大回撤 {pct(bh['mdd'])}，"
            f"说明趋势跟踪规则确实起到了控制下行风险的作用。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        build_table(main_metrics_table_data(main_result), font, [4.2 * cm, 4.6 * cm, 4.6 * cm]),
        p("表1 广发证券双均线策略（5/15）与买入持有基准的回测指标对比。", caption),
        p(
            f"从表1看，策略在风险端全面占优：年化波动率 {pct(strat['vol'])} 低于基准的 {pct(bh['vol'])}，"
            f"最大回撤约为基准的一半；但收益端明显吃亏，累计回报 {pct(strat['total'])} 和夏普比率 {strat['sharpe']:.2f} "
            f"都{compare_word(strat['total'], bh['total'])}基准（{pct(bh['total'])}、{bh['sharpe']:.2f}），"
            f"原因就是期初踏空和震荡期假信号的双重拖累。全期共 {main_result['n_trades']} 笔交易，"
            f"其中 {n_loss} 笔为震荡期的小额亏损，利润主要由{best_trade_desc}的趋势行情贡献，胜率仅 "
            f"{pct(main_result['win_rate'], 1)}。这是趋势策略的典型形态：不靠高胜率取胜，"
            f"而是依靠“截断亏损、让利润奔跑”，用少数大趋势覆盖多数小止损。",
            body,
        ),
        PageBreak(),
        p("四、不同股票与不同均线周期的对比", heading),
        p(
            f"为观察策略在不同市场环境下的表现，另取招商银行（600036.SH）与贵州茅台（600519.SH）同期日线数据，"
            f"并对每只股票分别测试 {('、'.join(f'{s}/{l}' for s, l in PARAM_GRID))} 四组均线周期，结果见表2。"
            f"样本期内三只股票的走势差异明显：广发证券震荡上行，买入持有回报 {pct(bh['total'])}；"
            f"招商银行趋势性下跌，买入持有回报 {pct(cmb['buy_hold']['total'])}；"
            f"贵州茅台震荡下行，买入持有回报 {pct(mt['buy_hold']['total'])}。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        build_table(
            sweep_table_data(sweep),
            font,
            [2.2 * cm, 1.7 * cm, 2.3 * cm, 1.9 * cm, 1.9 * cm, 1.4 * cm, 1.7 * cm, 2.3 * cm],
            font_size=8.1,
        ),
        p("表2 三只股票在四组均线周期下的双均线策略回测指标对比。", caption),
        PageBreak(),
        KeepTogether([
            Image(str(charts["signal_cmb"]), width=15.5 * cm, height=9.05 * cm),
            p(f"图4 招商银行收盘价、MA{SHORT_WINDOW}/MA{LONG_WINDOW} 均线与金叉死叉信号", caption),
        ]),
        p(
            f"图4给出招商银行的价格与信号分布。在这类趋势性下跌的行情中，双均线策略的价值体现为“避险”："
            f"死叉让策略在大部分下跌时段保持空仓。5/15 参数下策略累计回报 {pct(cmb['strategy']['total'])}、"
            f"最大回撤 {pct(cmb['strategy']['mdd'])}，好于买入持有的 {pct(cmb['buy_hold']['total'])} 和 "
            f"{pct(cmb['buy_hold']['mdd'])}，但优势有限——下跌途中的反弹不断触发金叉，事后多被证明是陷阱，"
            f"{cmb['n_trades']} 笔交易胜率仅 {pct(cmb['win_rate'], 1)}，空仓躲过的跌幅有相当一部分又被假信号亏了回去。"
            f"把均线放长到 20/60 后，交易次数降到 {cmb_long['n_trades']} 笔，累计回报收窄至 "
            f"{pct(cmb_long['strategy']['total'])}、最大回撤 {pct(cmb_long['strategy']['mdd'])}，避险效果显著改善，"
            f"说明下跌市中更长的均线周期能过滤掉大部分反弹噪声。另外本策略只做多，"
            f"下跌市中最多做到少亏，无法把下跌本身变成利润。",
            body,
        ),
        PageBreak(),
        p("五、适用场景总结与应用心得", heading),
        p(
            f"综合表1、表2的对比可以得到几点规律。第一，双均线策略的表现高度依赖行情形态。"
            f"它最适合趋势持续时间长、回调幅度浅的单边行情：上涨趋势中能持仓吃到主升段，"
            f"下跌趋势中能空仓规避大部分跌幅。而在震荡或涨跌反复的行情中，均线纠缠导致假信号密集，"
            f"本次三只股票在短周期参数下的胜率普遍只有一到四成，多数交易以小亏收场。"
            f"还要注意“有趋势”不等于“策略必赢”：广发证券全年上涨 {pct(bh['total'])}，"
            f"但由于期初踏空和中段震荡，策略仍明显跑输买入持有，说明趋势跟踪赚的是趋势中段的钱，"
            f"起点和拐点附近的收益注定要让渡。",
            body,
        ),
        p(
            "第二，均线周期决定了灵敏度与稳定性之间的取舍，且不存在普适最优参数。短周期组合（如 5/15、5/20）"
            "信号多、反应快，但假信号也多；长周期组合（如 20/60）过滤了大部分噪声，但进出场明显滞后。"
            f"表2显示同一组参数在不同股票上表现差异巨大：5/20 在广发证券上是四组参数中最好的"
            f"（累计回报 {pct(next(i['result']['strategy']['total'] for i in sweep if i['name'] == '广发证券' and i['long'] == 20))}），"
            f"在招商银行上却是最差的；招商银行反而以 20/60 最优。事后挑选的“最优参数”很可能只是拟合了历史噪声，"
            "实际使用前应在更长的历史区间上滚动检验参数的稳定性。",
            body,
        ),
        p(
            f"第三，评估策略不能只看累计回报，要结合最大回撤和夏普比率看风险调整后的质量。"
            f"本次回测中双均线策略最可靠的贡献是控制回撤：5/15 参数下三只股票的策略最大回撤"
            f"（{pct(strat['mdd'])}、{pct(cmb['strategy']['mdd'])}、{pct(mt['strategy']['mdd'])}）"
            f"都明显低于对应的买入持有回撤（{pct(bh['mdd'])}、{pct(cmb['buy_hold']['mdd'])}、{pct(mt['buy_hold']['mdd'])}）。"
            f"但降低波动不等于夏普必然更高——广发证券样本中基准夏普反而高于策略，"
            f"因为回撤的压缩是用收益的让渡换来的，两者要放在一起权衡。",
            body,
        ),
        p(
            "第四，本报告的回测忽略了佣金、印花税和滑点，而双均线策略在震荡市中交易频繁，"
            "真实成本会进一步侵蚀收益；同时单一年度、三只股票的样本有限，结论不能直接外推。"
            "后续可以从三个方向改进：引入交易成本和止损规则，让回测更接近实盘；"
            "增加趋势过滤条件（如成交量、波动率或长期均线方向过滤），减少震荡市的假信号；"
            "在更多标的和更长历史区间上做滚动回测，检验策略与参数的稳健性。",
            body,
        ),
    ]

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ------------------------------------------------------------------ entrypoint

def generate_assets(student_name: str, output: Path) -> dict[str, Path]:
    HERE.mkdir(exist_ok=True)
    frames = {code: load_price_data(code) for code, _ in STOCKS}

    results_515 = {code: evaluate(frames[code], SHORT_WINDOW, LONG_WINDOW) for code, _ in STOCKS}
    main_result = results_515["000776.SZ"]

    signal_csv = HERE / "000776_SZ_ma_signals.csv"
    export = main_result["bt"].copy()
    export["trade_date"] = export["trade_date"].dt.strftime("%Y%m%d")
    export.to_csv(signal_csv, index=False, encoding="utf-8-sig")

    sweep = [
        {
            "name": name,
            "short": short,
            "long": long,
            "result": results_515[code] if (short, long) == (SHORT_WINDOW, LONG_WINDOW) else evaluate(frames[code], short, long),
        }
        for code, name in STOCKS
        for short, long in PARAM_GRID
    ]

    charts = {
        "signal_main": HERE / "000776_SZ_ma_signal.png",
        "nav_main": HERE / "000776_SZ_nav.png",
        "dd_main": HERE / "000776_SZ_drawdown.png",
        "signal_cmb": HERE / "600036_SH_ma_signal.png",
    }
    draw_signal_chart(
        main_result["bt"],
        f"图1 广发证券收盘价与 MA{SHORT_WINDOW}/MA{LONG_WINDOW} 交叉信号",
        SHORT_WINDOW,
        LONG_WINDOW,
        charts["signal_main"],
    )
    draw_nav_chart(main_result["bt"], "图2 广发证券双均线策略与买入持有净值", charts["nav_main"])
    draw_drawdown_chart(main_result["bt"], "图3 广发证券双均线策略逐日回撤", charts["dd_main"])
    draw_signal_chart(
        results_515["600036.SH"]["bt"],
        f"图4 招商银行收盘价与 MA{SHORT_WINDOW}/MA{LONG_WINDOW} 交叉信号",
        SHORT_WINDOW,
        LONG_WINDOW,
        charts["signal_cmb"],
    )

    build_pdf(student_name, main_result, results_515, sweep, charts, output)
    return {"pdf": output, "signal_csv": signal_csv, **charts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="姓名")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else HERE / f"{args.student_name}TASK3.pdf"
    assets = generate_assets(args.student_name, output)
    for name, path in assets.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
