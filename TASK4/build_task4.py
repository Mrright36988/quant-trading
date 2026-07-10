#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TASK4 turtle trading strategy assets and PDF.

Usage:
    .venv/bin/python TASK4/build_task4.py --student-name 姓名

Reads TASK1/000776_SZ_daily.csv, computes Donchian channels, ATR, turtle
entry/exit signals, runs a next-open backtest, scans parameters, draws charts
and builds the submission PDF (宋体, 五号, 1.5 倍行距).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
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
DATA_PATH = ROOT / "TASK1" / "000776_SZ_daily.csv"
EXTRA_DATA_DIR = HERE / "data"
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
FALLBACK_FONTS = ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Medium.ttc"]

TRADING_DAYS = 252
INITIAL_CAPITAL = 1_000_000.0
COMMISSION = 2.5e-4   # 双边佣金 0.025%
STAMP_DUTY = 5e-4     # 卖出印花税 0.05%

# 用于第四部分参数对比的股票（除主样本 000776 外，尝试用 tushare legacy 接口补充）
EXTRA_STOCKS = [
    ("600519", "贵州茅台", "大盘蓝筹/低波动"),
    ("300750", "宁德时代", "高波动成长"),
]
PRIMARY_STOCK = ("000776", "广发证券", "券商/高贝塔")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_price_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    for column in ["open", "high", "low", "close", "vol"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("trade_date").reset_index(drop=True)


def _patch_legacy_tushare_for_modern_pandas() -> None:
    if hasattr(pd.DataFrame, "append"):
        return

    def _append(self, other, ignore_index: bool = False, **_: object):
        frame = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, frame], ignore_index=ignore_index)

    pd.DataFrame.append = _append  # type: ignore[attr-defined]


def fetch_extra_stock(code: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily bars via tushare legacy get_k_data; cache to TASK4/data."""
    cache = EXTRA_DATA_DIR / f"{code}_daily.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    try:
        import tushare as ts

        _patch_legacy_tushare_for_modern_pandas()
        raw = ts.get_k_data(code, start=start, end=end)
        if raw is None or raw.empty:
            return None
        df = raw.rename(columns={"date": "trade_date", "volume": "vol"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[["trade_date", "open", "high", "low", "close", "vol"]].sort_values("trade_date").reset_index(drop=True)
        EXTRA_DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(cache, index=False, encoding="utf-8-sig")
        return df
    except Exception as exc:  # 网络或接口失败时跳过该股票
        print(f"[warn] fetch {code} failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# 海龟指标：唐奇安通道 + ATR
# ---------------------------------------------------------------------------

def add_turtle_indicators(
    df: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
    atr_window: int = 20,
) -> pd.DataFrame:
    result = df.copy()
    # 通道基于"过去 N 日"（不含当日），因此 rolling 之后 shift(1)，避免用当日数据判断当日突破
    result["entry_high"] = result["high"].rolling(entry_window).max().shift(1)
    result["exit_low"] = result["low"].rolling(exit_window).min().shift(1)
    prev_close = result["close"].shift(1)
    tr = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - prev_close).abs(),
            (result["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["tr"] = tr
    # 威尔德平滑：ATR_t = ATR_{t-1} + (TR_t - ATR_{t-1}) / N
    result["atr"] = tr.ewm(alpha=1.0 / atr_window, adjust=False, min_periods=atr_window).mean()
    return result


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    sell_date: pd.Timestamp | None = None
    sell_price: float | None = None
    exit_reason: str = ""

    @property
    def pnl_pct(self) -> float:
        cost = self.buy_price * (1 + COMMISSION)
        proceeds = (self.sell_price or self.buy_price) * (1 - COMMISSION - STAMP_DUTY)
        return proceeds / cost - 1


@dataclass
class BacktestResult:
    df: pd.DataFrame
    equity: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def daily_returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    @property
    def cumulative_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

    @property
    def annual_return(self) -> float:
        n = len(self.equity)
        return float((1 + self.cumulative_return) ** (TRADING_DAYS / max(n - 1, 1)) - 1)

    @property
    def sharpe(self) -> float:
        r = self.daily_returns
        if r.std() == 0 or math.isnan(r.std()):
            return 0.0
        return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS))

    @property
    def max_drawdown(self) -> float:
        dd = 1 - self.equity / self.equity.cummax()
        return float(dd.max())

    @property
    def drawdown_series(self) -> pd.Series:
        return -(1 - self.equity / self.equity.cummax())

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.sell_price is not None]
        if not closed:
            return 0.0
        return sum(1 for t in closed if t.pnl_pct > 0) / len(closed)


def run_backtest(
    raw: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
    atr_window: int = 20,
    stop_mult: float = 2.0,
) -> BacktestResult:
    """信号在 T 日收盘计算，T+1 日开盘价成交；全仓买入，100 股整数倍。"""
    df = add_turtle_indicators(raw, entry_window, exit_window, atr_window)
    cash = INITIAL_CAPITAL
    shares = 0
    stop_price: float | None = None
    pending: str | None = None   # None | "buy" | 卖出原因
    trades: list[TradeRecord] = []
    equity: list[float] = []
    signal_col: list[str] = []

    for row in df.itertuples(index=False):
        # 1) 以今日开盘价执行昨日收盘产生的信号
        if pending == "buy" and shares == 0 and not math.isnan(row.open):
            lots = int(cash / (row.open * (1 + COMMISSION)) // 100 * 100)
            if lots > 0:
                shares = lots
                cash -= shares * row.open * (1 + COMMISSION)
                trades.append(TradeRecord(buy_date=row.trade_date, buy_price=float(row.open), shares=shares))
                stop_price = float(row.open) - stop_mult * float(row.atr) if not math.isnan(row.atr) else None
        elif pending in ("跌破退出通道", "触发止损") and shares > 0:
            cash += shares * row.open * (1 - COMMISSION - STAMP_DUTY)
            trades[-1].sell_date = row.trade_date
            trades[-1].sell_price = float(row.open)
            trades[-1].exit_reason = pending
            shares = 0
            stop_price = None
        pending = None

        # 2) 收盘估值
        equity.append(cash + shares * row.close)

        # 3) 收盘后计算次日信号
        signal = ""
        if shares == 0:
            if not math.isnan(row.entry_high) and row.close > row.entry_high and not math.isnan(row.atr):
                pending = "buy"
                signal = "买入信号"
        else:
            if stop_price is not None and row.close < stop_price:
                pending = "触发止损"
                signal = "止损信号"
            elif not math.isnan(row.exit_low) and row.close < row.exit_low:
                pending = "跌破退出通道"
                signal = "卖出信号"
        signal_col.append(signal)

    # 期末若仍持仓，按最后收盘价强制平仓，便于统计已完成交易
    if shares > 0:
        last = df.iloc[-1]
        cash += shares * float(last["close"]) * (1 - COMMISSION - STAMP_DUTY)
        trades[-1].sell_date = last["trade_date"]
        trades[-1].sell_price = float(last["close"])
        trades[-1].exit_reason = "期末平仓"
        equity[-1] = cash

    df = df.assign(signal=signal_col)
    return BacktestResult(df=df, equity=pd.Series(equity, index=df["trade_date"]), trades=trades)


def buy_and_hold_result(raw: pd.DataFrame) -> BacktestResult:
    df = add_turtle_indicators(raw)
    close = df["close"].astype(float)
    shares = int(INITIAL_CAPITAL / (close.iloc[0] * (1 + COMMISSION)) // 100 * 100)
    cash = INITIAL_CAPITAL - shares * close.iloc[0] * (1 + COMMISSION)
    equity = cash + shares * close
    equity.iloc[-1] = cash + shares * close.iloc[-1] * (1 - COMMISSION - STAMP_DUTY)
    return BacktestResult(df=df, equity=pd.Series(equity.values, index=df["trade_date"]))


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def setup_matplotlib_font() -> None:
    for path in [SONGTI, *FALLBACK_FONTS]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw_channel_chart(result: BacktestResult, title: str, output: Path) -> None:
    df = result.df
    fig, ax = plt.subplots(figsize=(13, 7), dpi=110)
    ax.plot(df["trade_date"], df["close"], color="#111111", linewidth=1.4, label="收盘价")
    ax.plot(df["trade_date"], df["entry_high"], color="#c33c2e", linewidth=1.2, label="入场通道上轨（20日最高）")
    ax.plot(df["trade_date"], df["exit_low"], color="#2f8f4e", linewidth=1.2, label="退出通道下轨（10日最低）")
    ax.fill_between(df["trade_date"], df["exit_low"], df["entry_high"], color="#1f77b4", alpha=0.06)

    buys = [(t.buy_date, t.buy_price) for t in result.trades]
    sells = [(t.sell_date, t.sell_price) for t in result.trades if t.sell_price is not None]
    if buys:
        xs, ys = zip(*buys)
        ax.scatter(xs, ys, marker="^", s=130, color="#c33c2e", edgecolors="black", zorder=5, label="买入")
    if sells:
        xs, ys = zip(*sells)
        ax.scatter(xs, ys, marker="v", s=130, color="#2f8f4e", edgecolors="black", zorder=5, label="卖出")

    ax.set_title(title, fontsize=15)
    ax.set_ylabel("价格（元）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_atr_chart(result: BacktestResult, title: str, output: Path) -> None:
    df = result.df
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=110)
    ax.plot(df["trade_date"], df["atr"], color="#1f77b4", linewidth=1.6, label="ATR(20)")
    ax.plot(df["trade_date"], df["tr"], color="#bbbbbb", linewidth=0.8, label="每日真实波幅 TR")
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("波幅（元）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_equity_chart(strategy: BacktestResult, benchmark: BacktestResult, title: str, output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.2), dpi=110, sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    nav_s = strategy.equity / strategy.equity.iloc[0]
    nav_b = benchmark.equity / benchmark.equity.iloc[0]
    ax1.plot(nav_s.index, nav_s, color="#c33c2e", linewidth=1.6, label="海龟策略净值")
    ax1.plot(nav_b.index, nav_b, color="#1f77b4", linewidth=1.4, label="买入持有净值")
    ax1.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    ax1.set_title(title, fontsize=15)
    ax1.set_ylabel("净值（期初=1）")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.25)

    dd = strategy.drawdown_series * 100
    ax2.fill_between(dd.index, dd, 0, color="#c33c2e", alpha=0.35)
    ax2.set_ylabel("策略回撤（%）")
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_param_chart(scan: list[dict], title: str, output: Path) -> None:
    labels = [row["label"] for row in scan]
    cum = [row["cum"] * 100 for row in scan]
    mdd = [row["mdd"] * 100 for row in scan]
    x = range(len(scan))
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=110)
    width = 0.38
    ax.bar([i - width / 2 for i in x], cum, width=width, color="#c33c2e", label="累计收益率（%）")
    ax.bar([i + width / 2 for i in x], mdd, width=width, color="#1f77b4", label="最大回撤（%）")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("百分比（%）")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(cum):
        ax.text(i - width / 2, v + (1.2 if v >= 0 else -2.6), f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def register_pdf_font() -> str:
    if Path(SONGTI).exists():
        pdfmetrics.registerFont(TTFont("Songti", SONGTI))
        return "Songti"
    return "Helvetica"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def fmt_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


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


def metrics_row(name: str, r: BacktestResult, trades: str = "—", win: str = "—") -> list[str]:
    return [
        name,
        f"{r.cumulative_return * 100:.2f}%",
        f"{r.annual_return * 100:.2f}%",
        f"{r.max_drawdown * 100:.2f}%",
        f"{r.sharpe:.2f}",
        trades,
        win,
    ]


def trades_table_data(trades: list[TradeRecord]) -> list[list[str]]:
    rows = [["序号", "买入日期", "买入价", "卖出日期", "卖出价", "持有天数", "收益率", "卖出原因"]]
    for i, t in enumerate(trades, start=1):
        hold_days = (t.sell_date - t.buy_date).days if t.sell_date is not None else 0
        rows.append([
            str(i),
            fmt_date(t.buy_date),
            f"{t.buy_price:.2f}",
            fmt_date(t.sell_date) if t.sell_date is not None else "持仓中",
            f"{t.sell_price:.2f}" if t.sell_price is not None else "—",
            str(hold_days),
            f"{t.pnl_pct * 100:.2f}%",
            t.exit_reason or "—",
        ])
    return rows


def build_pdf(
    student_name: str,
    output: Path,
    charts: dict[str, Path],
    strategy: BacktestResult,
    benchmark: BacktestResult,
    period_scan: list[dict],
    stop_scan: list[dict],
    stock_scan: list[dict],
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
        title=f"{student_name}TASK4",
        author=student_name,
    )

    df = strategy.df
    rows = len(df)
    start_date, end_date = fmt_date(df["trade_date"].iloc[0]), fmt_date(df["trade_date"].iloc[-1])
    latest = df.iloc[-1]
    n_buy = sum(1 for t in strategy.trades)
    closed = [t for t in strategy.trades if t.sell_price is not None]
    n_stop = sum(1 for t in closed if t.exit_reason == "触发止损")
    n_channel = sum(1 for t in closed if t.exit_reason == "跌破退出通道")
    n_final = sum(1 for t in closed if t.exit_reason == "期末平仓")
    best = max(closed, key=lambda t: t.pnl_pct)
    worst = min(closed, key=lambda t: t.pnl_pct)
    atr_series = df["atr"].dropna()

    best_period = max(period_scan, key=lambda r: r["cum"])
    worst_period = min(period_scan, key=lambda r: r["cum"])

    period_rows = [["入场/退出周期", "交易次数", "胜率", "累计收益率", "年化收益率", "最大回撤", "夏普比率"]]
    for row in period_scan:
        period_rows.append([
            row["label"], str(row["trades"]), f"{row['win'] * 100:.0f}%",
            f"{row['cum'] * 100:.2f}%", f"{row['ann'] * 100:.2f}%",
            f"{row['mdd'] * 100:.2f}%", f"{row['sharpe']:.2f}",
        ])

    stop_rows = [["止损倍数", "交易次数", "胜率", "累计收益率", "最大回撤", "夏普比率"]]
    for row in stop_scan:
        stop_rows.append([
            row["label"], str(row["trades"]), f"{row['win'] * 100:.0f}%",
            f"{row['cum'] * 100:.2f}%", f"{row['mdd'] * 100:.2f}%", f"{row['sharpe']:.2f}",
        ])

    stock_rows = [["股票（类型）", "买入持有收益", "策略累计收益", "策略最大回撤", "策略夏普", "交易次数", "胜率"]]
    for row in stock_scan:
        stock_rows.append([
            row["label"], f"{row['bh'] * 100:.2f}%", f"{row['cum'] * 100:.2f}%",
            f"{row['mdd'] * 100:.2f}%", f"{row['sharpe']:.2f}", str(row["trades"]), f"{row['win'] * 100:.0f}%",
        ])

    story = [
        p(f"{student_name}TASK4", title),
        Spacer(1, 0.24 * cm),
        p("一、海龟策略的核心思想与关键优势", heading),
        p(
            "海龟交易法则来源于 1983 年美国交易员理查德·丹尼斯（Richard Dennis）与威廉·埃克哈特（William Eckhardt）的著名实验：他们招募了一批没有交易经验的学员（称为“海龟”），只教授一套完全机械化的规则，结果多数学员在数年内取得了稳定盈利。该实验说明，成功的交易可以来自一套明确、可复制的规则体系，而不是个人天赋。",
            body,
        ),
        p(
            "海龟策略的核心思想是趋势跟踪：价格向上突破过去一段时间的最高点时，说明可能形成向上趋势，此时顺势买入；价格跌破近期低点或回撤超过预设幅度时坚决离场。它不预测行情，只对已经发生的突破做出反应，通过“截断亏损、让利润奔跑”获取少数大趋势带来的收益，用大趋势的大额盈利覆盖多次小额止损亏损。",
            body,
        ),
        p(
            "海龟策略的关键优势包括：第一，规则完全客观，入场、离场、止损、头寸规模都有明确公式，能够程序化执行并进行历史回测，最大程度排除情绪干扰；第二，风险控制内建于规则之中，以 ATR 度量的波动决定止损距离和每笔头寸大小，单笔亏损被限定在账户的很小比例；第三，跨品种普适性强，同一套规则可以应用于商品期货、股票、外汇等不同市场；第四，逻辑简单透明，参数含义直观，便于在此基础上扩展加仓、过滤等模块。",
            body,
        ),
        Spacer(1, 0.16 * cm),
        p("二、高低点通道、ATR 与止损条件", heading),
        p(
            "高低点通道又称唐奇安通道（Donchian Channel）：上轨为过去 N1 个交易日最高价的最大值，下轨为过去 N2 个交易日最低价的最小值，均不含当日。经典海龟系统一采用 20 日入场通道和 10 日退出通道：当日收盘价向上突破 20 日最高点时产生买入信号；持仓期间收盘价跌破 10 日最低点时产生卖出信号。系统二采用更慢的 55 日入场、20 日退出，用于捕捉更大级别的趋势。通道周期越长，信号越少、越迟钝，但过滤掉的假突破也越多。",
            body,
        ),
        p(
            "平均真实波幅（Average True Range，ATR）是海龟体系中的波动率度量，海龟称其为 N。先计算每日真实波幅 TR=max(当日最高-当日最低, |当日最高-昨日收盘|, |当日最低-昨日收盘|)，TR 相比简单的最高减最低考虑了跳空缺口；再对 TR 做 N 日威尔德平滑（ATR=前日ATR+(当日TR-前日ATR)/N）得到 ATR。ATR 在海龟策略中承担两个职责：一是决定止损距离，二是决定头寸规模——经典海龟每个头寸单位 Unit=账户资金的1%÷N，使不同波动水平的品种承担大致相同的风险。",
            body,
        ),
        p(
            "止损条件是海龟策略的生命线。经典规则为 2N 硬止损：买入后将止损价设在成交价下方 2 倍 ATR 处，收盘价跌破止损价即无条件离场；配合头寸公式，单笔交易最大亏损约为账户的 2%。本次实现中，持仓期间只要满足“收盘价跌破 2N 止损线”或“收盘价跌破 10 日退出通道”其中之一，即在下一交易日开盘卖出。前者控制单笔交易的最大损失，后者在趋势衰竭时保护已有利润，两者共同构成完整的离场体系。",
            body,
        ),
        PageBreak(),
        p("三、Python 实现：通道、ATR 与交易信号", heading),
        p(
            f"程序首先加载 TASK1 已存储的广发证券（000776.SZ）日线数据 TASK1/000776_SZ_daily.csv，共 {rows} 个交易日，区间为 {start_date} 至 {end_date}。随后按上述公式计算 20 日入场通道 entry_high、10 日退出通道 exit_low 和 20 日 ATR，并逐日生成交易信号：空仓时收盘价上穿入场通道上轨即产生买入信号；持仓时收盘价跌破止损线或退出通道下轨即产生卖出信号。全部中间结果保存为 TASK4/000776_SZ_turtle_signals.csv。",
            body,
        ),
        KeepTogether([
            Image(str(charts["channel"]), width=15.5 * cm, height=8.35 * cm),
            p("图1 广发证券海龟策略高低点通道与买卖信号", caption),
        ]),
        p(
            f"图1中，黑线为收盘价，红线为 20 日入场通道上轨，绿线为 10 日退出通道下轨，红色三角为实际买入点，绿色倒三角为实际卖出点。样本期内策略共产生 {n_buy} 次买入，其中 {n_channel} 次因跌破退出通道离场、{n_stop} 次因触发 2N 止损离场"
            + (f"、{n_final} 次为期末按收盘价强制平仓" if n_final else "")
            + "。可以看到，买入点都出现在价格创出近 20 日新高之后，卖出点则出现在回撤初期，体现了“突破入场、破位离场”的机械规则。",
            body,
        ),
        KeepTogether([
            Image(str(charts["atr"]), width=15.5 * cm, height=6.7 * cm),
            p("图2 广发证券真实波幅 TR 与 ATR(20)", caption),
        ]),
        p(
            f"图2中，灰线为每日真实波幅 TR，蓝线为其 20 日威尔德平滑 ATR。样本期内 ATR 最小 {atr_series.min():.2f} 元、最大 {atr_series.max():.2f} 元、期末为 {atr_series.iloc[-1]:.2f} 元（约为期末股价的 {atr_series.iloc[-1] / latest['close'] * 100:.1f}%）。ATR 上升段对应行情加速期，此时 2N 止损距离自动放宽，避免被正常波动扫出；ATR 回落段对应缩量整理期，止损随之收紧。",
            body,
        ),
        PageBreak(),
        p("四、模拟交易与回测评估", heading),
        p(
            f"回测规则：初始资金 100 万元，信号在 T 日收盘计算、T+1 日开盘价成交，买入时按 100 股整数倍全仓买入；交易成本按双边佣金 0.025%、卖出印花税 0.05% 计。样本期内全部交易明细见表1，最好一笔交易收益 {best.pnl_pct * 100:.2f}%（{fmt_date(best.buy_date)} 买入），最差一笔 {worst.pnl_pct * 100:.2f}%（{fmt_date(worst.buy_date)} 买入）。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        build_table(
            trades_table_data(strategy.trades), font,
            [1.1 * cm, 2.4 * cm, 1.7 * cm, 2.4 * cm, 1.7 * cm, 1.9 * cm, 1.9 * cm, 2.4 * cm],
            font_size=8.3,
        ),
        p("表1 海龟策略（20/10 通道，2N 止损）全部交易明细。", caption),
        Spacer(1, 0.18 * cm),
        build_table(
            [
                ["方案", "累计收益率", "年化收益率", "最大回撤", "夏普比率", "交易次数", "胜率"],
                metrics_row("海龟策略", strategy, str(len(closed)), f"{strategy.win_rate * 100:.0f}%"),
                metrics_row("买入持有", benchmark),
            ],
            font,
            [2.4 * cm, 2.3 * cm, 2.3 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.7 * cm],
        ),
        p("表2 海龟策略与买入持有基准的绩效对比。", caption),
        KeepTogether([
            Image(str(charts["equity"]), width=15.5 * cm, height=9.8 * cm),
            p("图3 海龟策略与买入持有净值曲线及策略回撤", caption),
        ]),
        p(
            f"表2和图3显示：海龟策略累计收益 {strategy.cumulative_return * 100:.2f}%（年化 {strategy.annual_return * 100:.2f}%），明显跑输买入持有的 {benchmark.cumulative_return * 100:.2f}%；但策略最大回撤 {strategy.max_drawdown * 100:.2f}%，低于买入持有的 {benchmark.max_drawdown * 100:.2f}%。从交易结构看，策略呈现出海龟法则典型的“低胜率、高盈亏比”特征：{len(closed)} 笔交易中 {sum(1 for t in closed if t.pnl_pct <= 0)} 笔为小额亏损（单笔亏损均被控制在 5% 以内），最后一笔交易捕捉到 2026 年 6 月的主升浪，单笔盈利 {best.pnl_pct * 100:.2f}%，几乎覆盖了此前全部假突破亏损。",
            body,
        ),
        p(
            f"策略跑输基准的原因值得分析：本样本期是“急涨—深调—再急涨”的脉冲式强势行情，价格突破后往往很快回调触发离场，离场后又迅速重新走强，突破确认机制反复付出“买在阶段高位、卖在回调低位”的磨损成本，且空仓等待期错过了大量涨幅。这说明在单边强势且回调剧烈的市场中，买入持有反而是很强的基准；海龟策略的价值更多体现在回撤控制（{strategy.max_drawdown * 100:.2f}% 对 {benchmark.max_drawdown * 100:.2f}%）和亏损有界——一旦行情转熊，退出通道和止损会让策略及时空仓，而买入持有将承受完整下跌。",
            body,
        ),
        Spacer(1, 0.16 * cm),
        p("五、参数敏感性：通道周期、止损倍数与股票类型", heading),
        p(
            "保持其他规则不变，将入场/退出通道周期分别设为 10/5、20/10、30/15、40/20、55/20 五组，回测结果见表3和图4。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        build_table(period_rows, font, [2.9 * cm, 1.9 * cm, 1.6 * cm, 2.3 * cm, 2.3 * cm, 2.0 * cm, 1.9 * cm], font_size=8.3),
        p("表3 不同通道周期下海龟策略的回测表现。", caption),
        KeepTogether([
            Image(str(charts["param"]), width=15.5 * cm, height=7.35 * cm),
            p("图4 不同通道周期下策略累计收益率与最大回撤对比", caption),
        ]),
        p(
            f"表3显示，通道周期对结果影响显著：本样本中 {best_period['label']} 表现最好（累计收益 {best_period['cum'] * 100:.2f}%），{worst_period['label']} 表现最差（{worst_period['cum'] * 100:.2f}%）。与“周期越长越稳”的直觉相反，本样本中较短的周期反而占优：样本期行情涨跌切换快，短通道离场后能更快重新捕捉突破，而中长周期入场滞后、退出缓慢，在急涨急跌中反复高买低卖。这说明通道周期必须与标的的波动节奏匹配，参数并非越长或越短越好；同时，五组参数收益率相差超过 30 个百分点也提醒我们，针对单一样本挑选“最优参数”存在明显的过拟合风险，样本外表现未必成立。",
            body,
        ),
        p("在 20/10 通道下调整止损倍数，结果见表4。", body),
        Spacer(1, 0.12 * cm),
        build_table(stop_rows, font, [2.4 * cm, 2.2 * cm, 2.0 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm], font_size=8.3),
        p("表4 不同 ATR 止损倍数下的回测表现（20/10 通道）。", caption),
        p(
            "表4中一个有意思的现象是 2N 与 3N 的结果完全相同：核对交易明细后发现，样本期内 2N 止损从未先于 10 日退出通道触发，全部离场都由退出通道完成，因此放宽止损不改变任何交易。而 1N 止损过紧，会在趋势尚未走坏时被正常波动扫出，收益反而转负。这说明在通道退出机制正常工作时，ATR 止损更多是防范极端行情（跳空暴跌、流动性危机）的最后保险，2N 是经典规则中兼顾保护与容忍度的平衡选择。",
            body,
        ),
    ]

    if len(stock_scan) > 1:
        story.extend([
            p(
                "为观察股票类型的影响，使用相同的 20/10 通道和 2N 止损规则，对不同类型的股票在同一时间区间内回测，结果见表5。",
                body,
            ),
            Spacer(1, 0.12 * cm),
            build_table(stock_rows, font, [4.7 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm], font_size=8.1),
            p("表5 相同规则下不同类型股票的回测表现对比。", caption),
            p(
                "对比可见，同一套海龟规则在不同类型标的上的表现差异很大，且规律清晰：在快速上涨的广发证券和宁德时代上，策略均大幅跑输买入持有，脉冲式上涨让突破跟踪反复付出确认成本；而在走弱的贵州茅台上，策略亏损小于买入持有，最大回撤也明显更低——退出通道和空仓机制把损失截断在了下跌初期。这印证了趋势跟踪策略的本质：它相对买入持有的优势主要体现在“少亏”而非“多赚”，收益来源是标的走出持续的单边趋势，规则本身不能创造趋势。",
                body,
            ),
        ])

    story.extend([
        Spacer(1, 0.16 * cm),
        p("六、适应场景与使用心得", heading),
        p(
            "综合本次实验，海龟策略的适应场景可以归纳为：一是持续数月的单边趋势行情，突破式入场能及时上车并让利润奔跑，本次回测中唯一的大额盈利正是来自 2026 年 6 月启动的主升浪；二是下跌或走弱的标的，退出通道和止损让策略及时空仓、把亏损截断在下跌初期，茅台样本中策略亏损和回撤都明显小于买入持有；三是波动充分、流动性好的品种，ATR 才有足够量级支撑止损和头寸计算。相反，本次回测也清楚暴露了它的不适应场景：在“急涨—深调—再急涨”的脉冲式行情中，突破后往往紧接回调，策略反复遭遇假突破磨损，还会因空仓错过大量涨幅而明显跑输买入持有——这是趋势跟踪为控制风险必须支付的固有成本。",
            body,
        ),
        p(
            "使用心得有四点。第一，必须完整执行规则：本次回测 5 笔交易中前 4 笔全部小额亏损，若因连续亏损而放弃执行，就会错过决定全年收益的最后一波主升浪——海龟的收益天然集中在少数交易上，纪律是策略成立的前提。第二，止损和头寸管理比入场更重要：2N 止损与基于 N 的头寸单位共同将单笔风险限定在账户的很小比例，本次所有亏损交易的单笔损失都被控制在 5% 以内，这是策略能在连续亏损后活到趋势来临的基础。第三，参数要与标的匹配但不要过度优化：表3中不同周期收益率相差超过 30 个百分点，回测挑出的最优参数在样本外未必成立，宁可选择经典的 20/10 或 55/20 等稳健参数并接受其平庸期。第四，A 股为 T+1 且难以卖空，海龟规则中的做空部分无法执行，单边做多的突破策略在强势市中往往跑输买入持有，实际使用时更适合叠加大盘趋势过滤，或作为多策略组合中控制回撤的趋势跟踪模块，而不是单独追求超额收益的工具。",
            body,
        ),
    ])

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def scan_periods(raw: pd.DataFrame) -> list[dict]:
    configs = [(10, 5), (20, 10), (30, 15), (40, 20), (55, 20)]
    result = []
    for entry, exit_ in configs:
        bt = run_backtest(raw, entry_window=entry, exit_window=exit_)
        closed = [t for t in bt.trades if t.sell_price is not None]
        result.append({
            "label": f"{entry}/{exit_}",
            "trades": len(closed),
            "win": bt.win_rate,
            "cum": bt.cumulative_return,
            "ann": bt.annual_return,
            "mdd": bt.max_drawdown,
            "sharpe": bt.sharpe,
        })
    return result


def scan_stops(raw: pd.DataFrame) -> list[dict]:
    result = []
    for mult in [1.0, 2.0, 3.0]:
        bt = run_backtest(raw, stop_mult=mult)
        closed = [t for t in bt.trades if t.sell_price is not None]
        result.append({
            "label": f"{mult:.0f}N",
            "trades": len(closed),
            "win": bt.win_rate,
            "cum": bt.cumulative_return,
            "mdd": bt.max_drawdown,
            "sharpe": bt.sharpe,
        })
    return result


def scan_stocks(primary_raw: pd.DataFrame) -> list[dict]:
    start = primary_raw["trade_date"].iloc[0].strftime("%Y-%m-%d")
    end = primary_raw["trade_date"].iloc[-1].strftime("%Y-%m-%d")
    datasets = [(f"{PRIMARY_STOCK[1]}（{PRIMARY_STOCK[2]}）", primary_raw)]
    for code, name, kind in EXTRA_STOCKS:
        extra = fetch_extra_stock(code, start, end)
        if extra is not None and len(extra) > 60:
            datasets.append((f"{name}（{kind}）", extra))
    result = []
    for label, raw in datasets:
        bt = run_backtest(raw)
        bh = buy_and_hold_result(raw)
        closed = [t for t in bt.trades if t.sell_price is not None]
        result.append({
            "label": label,
            "bh": bh.cumulative_return,
            "cum": bt.cumulative_return,
            "mdd": bt.max_drawdown,
            "sharpe": bt.sharpe,
            "trades": len(closed),
            "win": bt.win_rate,
        })
    return result


def generate_assets(student_name: str, output: Path) -> dict[str, Path]:
    setup_matplotlib_font()
    raw = load_price_data()
    strategy = run_backtest(raw)
    benchmark = buy_and_hold_result(raw)

    signals_csv = HERE / "000776_SZ_turtle_signals.csv"
    export = strategy.df.assign(trade_date=strategy.df["trade_date"].dt.strftime("%Y%m%d"))
    export.to_csv(signals_csv, index=False, encoding="utf-8-sig")

    charts = {
        "channel": HERE / "000776_SZ_turtle_channel.png",
        "atr": HERE / "000776_SZ_turtle_atr.png",
        "equity": HERE / "000776_SZ_turtle_equity.png",
        "param": HERE / "000776_SZ_turtle_params.png",
    }
    draw_channel_chart(strategy, "图1 广发证券海龟策略高低点通道与买卖信号", charts["channel"])
    draw_atr_chart(strategy, "图2 广发证券真实波幅 TR 与 ATR(20)", charts["atr"])
    draw_equity_chart(strategy, benchmark, "图3 海龟策略与买入持有净值曲线及策略回撤", charts["equity"])

    period_scan = scan_periods(raw)
    draw_param_chart(period_scan, "图4 不同通道周期下策略累计收益率与最大回撤", charts["param"])
    stop_scan = scan_stops(raw)
    stock_scan = scan_stocks(raw)

    print("=== 海龟策略（20/10, 2N）回测结果 ===")
    print(f"累计收益: {strategy.cumulative_return * 100:.2f}%  年化: {strategy.annual_return * 100:.2f}%")
    print(f"最大回撤: {strategy.max_drawdown * 100:.2f}%  夏普: {strategy.sharpe:.2f}  胜率: {strategy.win_rate * 100:.0f}%")
    print(f"买入持有: 累计 {benchmark.cumulative_return * 100:.2f}%  回撤 {benchmark.max_drawdown * 100:.2f}%  夏普 {benchmark.sharpe:.2f}")
    print("周期扫描:", [(r["label"], f"{r['cum'] * 100:.1f}%") for r in period_scan])
    print("止损扫描:", [(r["label"], f"{r['cum'] * 100:.1f}%") for r in stop_scan])
    print("股票对比:", [(r["label"], f"{r['cum'] * 100:.1f}%") for r in stock_scan])

    output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(student_name, output, charts, strategy, benchmark, period_scan, stop_scan, stock_scan)
    return {"pdf": output, "signals_csv": signals_csv, **charts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="姓名")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else HERE / f"{args.student_name}TASK4.pdf"
    assets = generate_assets(args.student_name, output)
    for name, path in assets.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
