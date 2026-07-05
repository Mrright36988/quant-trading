#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TASK2 indicator analysis assets and PDF."""

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
DATA_PATH = ROOT / "TASK1" / "000776_SZ_daily.csv"
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
FONT_PATHS = [
    SONGTI,
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def register_pdf_font() -> str:
    if Path(SONGTI).exists():
        pdfmetrics.registerFont(TTFont("Songti", SONGTI))
        return "Songti"
    return "Helvetica"


def load_image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def fmt_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def load_price_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("trade_date").reset_index(drop=True)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    result = pd.Series(math.nan, index=close.index, dtype="float64")
    if len(close) <= period:
        return result

    avg_gain = gain.iloc[1 : period + 1].mean()
    avg_loss = loss.iloc[1 : period + 1].mean()
    result.iloc[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
        result.iloc[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.astype(float).ewm(span=fast, adjust=False).mean()
    ema_slow = close.astype(float).ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif, dea, hist


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.astype(float).rolling(window=window, min_periods=window).mean()
    std = close.astype(float).rolling(window=window, min_periods=window).std(ddof=0)
    return middle, middle + num_std * std, middle - num_std * std


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 9,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    lowest = low.astype(float).rolling(window=window, min_periods=1).min()
    highest = high.astype(float).rolling(window=window, min_periods=1).max()
    denom = highest - lowest
    rsv = ((close.astype(float) - lowest) / denom * 100).where(denom != 0, 50)

    k_values: list[float] = []
    d_values: list[float] = []
    k_prev = d_prev = 50.0
    for value in rsv.fillna(50):
        k_prev = (smooth - 1) / smooth * k_prev + value / smooth
        d_prev = (smooth - 1) / smooth * d_prev + k_prev / smooth
        k_values.append(k_prev)
        d_values.append(d_prev)
    k_series = pd.Series(k_values, index=close.index, dtype="float64")
    d_series = pd.Series(d_values, index=close.index, dtype="float64")
    return k_series, d_series, 3 * k_series - 2 * d_series


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["rsi14"] = rsi(result["close"])
    result["macd_dif"], result["macd_dea"], result["macd_hist"] = macd(result["close"])
    result["bb_middle"], result["bb_upper"], result["bb_lower"] = bollinger_bands(result["close"])
    result["kdj_k"], result["kdj_d"], result["kdj_j"] = kdj(result["high"], result["low"], result["close"])
    return result


def finite_values(series_list: Sequence[pd.Series]) -> list[float]:
    values: list[float] = []
    for series in series_list:
        values.extend(float(v) for v in series.dropna() if math.isfinite(float(v)))
    return values


def draw_axes(
    draw: ImageDraw.ImageDraw,
    dates: Sequence[pd.Timestamp],
    title: str,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    margins: tuple[int, int, int, int],
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
        draw.text((margin_l - 12, y), f"{value:.2f}", font=font_small, fill="#333333", anchor="rm")

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


def draw_line_chart(
    df: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[str],
    line_colors: Sequence[str],
    title: str,
    output: Path,
    hlines: Sequence[tuple[float, str, str]] = (),
    y_bounds: tuple[float, float] | None = None,
) -> None:
    width, height = 1300, 760
    margins = (95, 45, 88, 90)
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    series_list = [df[column].astype(float) for column in columns]
    values = finite_values([*series_list, *[pd.Series([value]) for value, _, _ in hlines]])
    y_min, y_max = y_bounds or (min(values), max(values))
    pad = (y_max - y_min) * 0.08 or 1.0
    if y_bounds is None:
        y_min, y_max = y_min - pad, y_max + pad
    x_at, y_at = draw_axes(draw, df["trade_date"].tolist(), title, y_min, y_max, width, height, margins)

    for value, label, color in hlines:
        y = y_at(value)
        draw.line([(margins[0], y), (width - margins[1], y)], fill=color, width=2)
        draw.text((width - margins[1] - 8, y - 4), label, font=load_image_font(16), fill=color, anchor="rs")

    for series, color in zip(series_list, line_colors, strict=True):
        last: tuple[float, float] | None = None
        for i, value in enumerate(series):
            if pd.isna(value):
                last = None
                continue
            point = (x_at(i), y_at(float(value)))
            if last:
                draw.line([last, point], fill=color, width=3)
            last = point

    draw_legend(draw, list(zip(labels, line_colors, strict=True)), margins[0], 64)
    image.save(output)


def draw_macd_chart(df: pd.DataFrame, output: Path) -> None:
    width, height = 1300, 760
    margins = (95, 45, 88, 90)
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = finite_values([df["macd_dif"], df["macd_dea"], df["macd_hist"], pd.Series([0.0])])
    y_min, y_max = min(values), max(values)
    pad = (y_max - y_min) * 0.12 or 1.0
    x_at, y_at = draw_axes(
        draw,
        df["trade_date"].tolist(),
        "图2 广发证券 MACD 指标",
        y_min - pad,
        y_max + pad,
        width,
        height,
        margins,
    )
    zero_y = y_at(0)
    draw.line([(margins[0], zero_y), (width - margins[1], zero_y)], fill="#555555", width=2)
    bar_w = max(2, int((width - margins[0] - margins[1]) / len(df) * 0.62))
    for i, value in enumerate(df["macd_hist"].astype(float)):
        if pd.isna(value):
            continue
        x = x_at(i)
        y = y_at(value)
        color = "#c33c2e" if value >= 0 else "#2f8f4e"
        draw.rectangle([x - bar_w / 2, min(y, zero_y), x + bar_w / 2, max(y, zero_y)], fill=color)

    for column, color in [("macd_dif", "#1f77b4"), ("macd_dea", "#ff7f0e")]:
        last: tuple[float, float] | None = None
        for i, value in enumerate(df[column].astype(float)):
            point = (x_at(i), y_at(float(value)))
            if last:
                draw.line([last, point], fill=color, width=3)
            last = point
    draw_legend(draw, [("DIF", "#1f77b4"), ("DEA", "#ff7f0e"), ("MACD柱", "#c33c2e")], margins[0], 64)
    image.save(output)


def missing_table_data(df: pd.DataFrame) -> list[list[str]]:
    rows = [["字段", "缺失数", "缺失率"]]
    for column, missing in df.isna().sum().items():
        rows.append([column, str(int(missing)), f"{missing / len(df) * 100:.2f}%"])
    return rows


def describe_table_data(df: pd.DataFrame) -> list[list[str]]:
    columns = ["open", "high", "low", "close", "pct_chg", "vol"]
    names = ["开盘价", "最高价", "最低价", "收盘价", "涨跌幅(%)", "成交量"]
    rows = [["指标", "计数", "均值", "标准差", "最小值", "中位数", "最大值"]]
    for column, name in zip(columns, names, strict=True):
        series = df[column].dropna().astype(float)
        rows.append([
            name,
            f"{len(series)}",
            f"{series.mean():.2f}",
            f"{series.std():.2f}",
            f"{series.min():.2f}",
            f"{series.median():.2f}",
            f"{series.max():.2f}",
        ])
    return rows


def other_indicator_table_data() -> list[list[str]]:
    return [
        ["指标", "主要作用", "常见使用方式"],
        ["MA/EMA", "观察价格趋势方向和平滑短期噪声", "短期均线上穿长期均线表示趋势可能转强"],
        ["ATR", "衡量真实波动幅度和风险水平", "用于止损距离、仓位控制和波动率过滤"],
        ["OBV", "结合成交量判断资金流向", "价格创新高但 OBV 不配合时警惕背离"],
        ["CCI", "衡量价格偏离统计均值的程度", "常用 +100 与 -100 观察强弱变化"],
        ["W&R", "判断近期收盘价在高低区间中的位置", "接近 0 表示偏强，接近 -100 表示偏弱"],
        ["BIAS", "衡量价格相对均线的乖离程度", "乖离过大时关注均值回归风险"],
    ]


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


def latest_signal_text(df: pd.DataFrame) -> dict[str, str]:
    latest = df.iloc[-1]
    close = latest["close"]
    rsi_value = latest["rsi14"]
    rsi_signal = "偏强，已进入常见超买观察区间" if rsi_value >= 70 else "偏弱，接近常见超卖观察区间" if rsi_value <= 30 else "处于中性震荡区间"
    macd_signal = "DIF 高于 DEA 且柱值为正，短期动能偏强" if latest["macd_hist"] > 0 else "DIF 低于 DEA 且柱值为负，短期动能偏弱"
    if close > latest["bb_upper"]:
        bb_signal = "收盘价突破布林带上轨，价格短线偏强但需警惕回落"
    elif close < latest["bb_lower"]:
        bb_signal = "收盘价跌破布林带下轨，价格短线偏弱但可能存在反弹"
    else:
        bb_signal = "收盘价位于布林带上下轨之间，仍处于带内波动"
    kdj_signal = "K 线高于 D 线，短期位置偏强" if latest["kdj_k"] > latest["kdj_d"] else "K 线低于 D 线，短期位置偏弱"
    return {
        "rsi": f"截至 {fmt_date(latest['trade_date'])}，RSI(14) 为 {rsi_value:.2f}，{rsi_signal}。",
        "macd": f"截至 {fmt_date(latest['trade_date'])}，DIF={latest['macd_dif']:.3f}，DEA={latest['macd_dea']:.3f}，MACD柱={latest['macd_hist']:.3f}，{macd_signal}。",
        "bb": f"截至 {fmt_date(latest['trade_date'])}，收盘价 {close:.2f} 元，上轨 {latest['bb_upper']:.2f} 元，中轨 {latest['bb_middle']:.2f} 元，下轨 {latest['bb_lower']:.2f} 元，{bb_signal}。",
        "kdj": f"截至 {fmt_date(latest['trade_date'])}，K={latest['kdj_k']:.2f}，D={latest['kdj_d']:.2f}，J={latest['kdj_j']:.2f}，{kdj_signal}。",
    }


def build_pdf(student_name: str, raw_df: pd.DataFrame, df: pd.DataFrame, output: Path, charts: dict[str, Path]) -> None:
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
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=13.5, firstLineIndent=0)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        title=f"{student_name}TASK2",
        author=student_name,
    )

    latest_text = latest_signal_text(df)
    rows = len(df)
    start_date = fmt_date(df["trade_date"].iloc[0])
    end_date = fmt_date(df["trade_date"].iloc[-1])
    close = df["close"].astype(float)
    pct = (close.iloc[-1] / close.iloc[0] - 1) * 100
    story = [
        p(f"{student_name}TASK2", title),
        Spacer(1, 0.24 * cm),
        p("一、数据基础诊断分析", heading),
        p(
            f"本次继续使用 TASK1 中已存储的广发证券（000776.SZ）日线数据，文件为 TASK1/000776_SZ_daily.csv。样本共 {rows} 个交易日，时间范围为 {start_date} 至 {end_date}。样本期内起始收盘价为 {close.iloc[0]:.2f} 元，期末收盘价为 {close.iloc[-1]:.2f} 元，区间涨跌幅为 {pct:.2f}%。",
            body,
        ),
        p(
            "缺失值检查显示，开盘价、最高价、最低价、收盘价和成交量没有缺失，可以满足本次技术指标计算需要。pre_close、change、pct_chg 的首行缺失来自第一条记录没有上一交易日数据；amount 字段全部缺失，原因是 TASK1 降级使用的 Tushare legacy 行情接口没有返回成交额。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        build_table(missing_table_data(raw_df), font, [2.8 * cm, 2.5 * cm, 2.5 * cm], font_size=8.3),
        p("表1 数据字段缺失值统计。", caption),
        Spacer(1, 0.18 * cm),
        build_table(describe_table_data(raw_df), font, [2.25 * cm, 1.65 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm, 1.85 * cm]),
        p("表2 主要价格与成交量字段描述性统计。", caption),
        p(
            f"从描述性统计看，样本期收盘价均值为 {close.mean():.2f} 元，中位数为 {close.median():.2f} 元，最低收盘价为 {close.min():.2f} 元，最高收盘价为 {close.max():.2f} 元，说明这一年价格主要在 {close.min():.2f} 元至 {close.max():.2f} 元区间内震荡上行。",
            body,
        ),
        PageBreak(),
        p("二、四个技术指标的计算方法与作用", heading),
        p(
            "RSI（Relative Strength Index，相对强弱指标）用于衡量一段时间内上涨幅度和下跌幅度的相对强弱。常用周期为 14 日，先计算每日涨跌差，再分别得到上涨均值 AvgGain 和下跌均值 AvgLoss，RS=AvgGain/AvgLoss，RSI=100-100/(1+RS)。作用上，RSI 主要用于识别短线强弱和超买超卖状态：通常 RSI 高于 70 表示价格短期偏强并可能进入超买观察区，低于 30 表示价格短期偏弱并可能进入超卖观察区。",
            body,
        ),
        p(
            "MACD（Moving Average Convergence Divergence，指数平滑异同移动平均线）由快慢两条指数移动平均线构成。常见参数为 12、26、9，DIF=EMA(12)-EMA(26)，DEA=EMA(DIF,9)，本报告采用国内行情软件常见写法 MACD柱=2*(DIF-DEA)。作用上，MACD 主要用于观察趋势方向和趋势动能：当 DIF 上穿 DEA 且柱值转正时，常被视为动能改善；DIF 下穿 DEA 且柱值转负时，说明短期动能转弱。",
            body,
        ),
        p(
            "布林带（Bollinger Bands）由中轨、上轨和下轨组成，常用中轨为 20 日收盘价移动平均线，标准差为同一窗口内收盘价标准差，上轨=中轨+2*标准差，下轨=中轨-2*标准差。作用上，布林带可以反映价格相对均值的位置和波动率变化：带宽扩大通常说明波动加大，价格接近或突破上下轨时需要结合趋势和成交量判断是否延续。",
            body,
        ),
        p(
            "KDJ 指标是在随机指标基础上扩展得到的短线动量指标，常用来判断收盘价在最近 N 日高低区间中的相对位置。先计算 RSV=(收盘价-N日最低价)/(N日最高价-N日最低价)*100，再递推计算 K=2/3*昨日K+1/3*RSV，D=2/3*昨日D+1/3*K，J=3*K-2*D。作用上，KDJ 对短期拐点较敏感，K 上穿 D 常被视为短线动能改善，K 下穿 D 则表示短线转弱；J 线反应更快，但噪声也更大。",
            body,
        ),
        p("资料核对主要参考 StockCharts ChartSchool 关于 RSI、MACD、Bollinger Bands 与 Stochastic Oscillator 的说明。", small),
        PageBreak(),
        p("三、Python 计算结果与可视化", heading),
        p(
            "Python 程序读取已保存的 CSV 数据后，统一按交易日期升序排列，并新增 rsi14、macd_dif、macd_dea、macd_hist、bb_middle、bb_upper、bb_lower、kdj_k、kdj_d、kdj_j 等字段，完整结果保存为 TASK2/000776_SZ_indicators.csv。",
            body,
        ),
        KeepTogether([
            Image(str(charts["rsi"]), width=15.5 * cm, height=9.05 * cm),
            p("图1 广发证券 RSI(14) 指标", caption),
            p(
                "图1中，RSI 位于 0 至 100 之间，虚线标出 30 和 70 两个常见观察线。RSI 接近 70 以上时代表上涨动能较强，但也需要警惕短线交易过热；接近 30 以下时代表下跌动能较强，后续可能出现修复。"
                + latest_text["rsi"],
                body,
            ),
        ]),
        PageBreak(),
        KeepTogether([
            Image(str(charts["macd"]), width=15.5 * cm, height=9.05 * cm),
            p("图2 广发证券 MACD 指标", caption),
            p(
                "图2中，DIF 与 DEA 的相对位置反映趋势动能变化，红绿柱反映 DIF 与 DEA 的差距。柱值从负转正通常表示短期动能改善，柱值持续放大说明动能增强；柱值收缩则说明趋势力度减弱。"
                + latest_text["macd"],
                body,
            ),
        ]),
        Spacer(1, 0.1 * cm),
        KeepTogether([
            Image(str(charts["bollinger"]), width=15.5 * cm, height=9.05 * cm),
            p("图3 广发证券布林带指标", caption),
            p(
                "图3中，中轨代表 20 日均线，上下轨代表均值附近约两个标准差的波动范围。价格在上轨附近运行时说明短期强势较明显，价格在下轨附近运行时说明短期承压较明显；上下轨距离扩大说明波动率上升。"
                + latest_text["bb"],
                body,
            ),
        ]),
        PageBreak(),
        p("四、扩展指标：KDJ", heading),
        p(
            "作业要求在 RSI、MACD、布林带之外再选取一个典型指标进行介绍、计算和图形展示，因此本次选择 KDJ 作为扩展指标。本次计算使用常见 9 日窗口，K 与 D 初始值取 50。",
            body,
        ),
        KeepTogether([
            Image(str(charts["kdj"]), width=15.5 * cm, height=9.05 * cm),
            p("图4 广发证券 KDJ 指标", caption),
            p(
                "图4中，K、D、J 三条线反映短期价格相对位置。K 上穿 D 常被视为短线动能改善，K 下穿 D 则表示短线转弱；J 线波动更大，对价格拐点更敏感，但也更容易产生噪声。"
                + latest_text["kdj"],
                body,
            ),
        ]),
        p(
            "综合来看，RSI、MACD、布林带和 KDJ 都只能描述历史价格与成交行为，不应单独作为买卖依据。实际交易中还需要结合基本面、成交量、风险控制和回测结果，避免只根据单一指标作出判断。",
            body,
        ),
        Spacer(1, 0.16 * cm),
        p("五、其他典型技术指标补充", heading),
        p(
            "除本次已经计算并绘图的四个指标外，技术分析中还常见以下指标。它们可以从趋势、波动率、成交量和均值偏离等角度补充观察，但同样需要结合回测和风险控制使用。",
            body,
        ),
        build_table(other_indicator_table_data(), font, [2.2 * cm, 6.6 * cm, 6.2 * cm], font_size=8.1),
        p("表3 其他典型技术指标及作用。", caption),
    ]

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def generate_assets(student_name: str, output: Path) -> dict[str, Path]:
    HERE.mkdir(exist_ok=True)
    raw_df = load_price_data()
    df = add_indicators(raw_df)
    indicator_csv = HERE / "000776_SZ_indicators.csv"
    df.assign(trade_date=df["trade_date"].dt.strftime("%Y%m%d")).to_csv(indicator_csv, index=False, encoding="utf-8-sig")
    charts = {
        "rsi": HERE / "000776_SZ_rsi.png",
        "macd": HERE / "000776_SZ_macd.png",
        "bollinger": HERE / "000776_SZ_bollinger.png",
        "kdj": HERE / "000776_SZ_kdj.png",
    }
    draw_line_chart(
        df,
        ["rsi14"],
        ["RSI(14)"],
        ["#1f77b4"],
        "图1 广发证券 RSI(14) 指标",
        charts["rsi"],
        hlines=[(70, "70", "#c33c2e"), (30, "30", "#2f8f4e")],
        y_bounds=(0, 100),
    )
    draw_macd_chart(df, charts["macd"])
    draw_line_chart(
        df,
        ["close", "bb_middle", "bb_upper", "bb_lower"],
        ["收盘价", "中轨", "上轨", "下轨"],
        ["#111111", "#1f77b4", "#c33c2e", "#2f8f4e"],
        "图3 广发证券布林带指标",
        charts["bollinger"],
    )
    draw_line_chart(
        df,
        ["kdj_k", "kdj_d", "kdj_j"],
        ["K", "D", "J"],
        ["#1f77b4", "#ff7f0e", "#7f3fbf"],
        "图4 广发证券 KDJ 指标",
        charts["kdj"],
        hlines=[(80, "80", "#c33c2e"), (20, "20", "#2f8f4e")],
    )
    build_pdf(student_name, raw_df, df, output, charts)
    return {"pdf": output, "indicator_csv": indicator_csv, **charts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="姓名")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else HERE / f"{args.student_name}TASK2.pdf"
    assets = generate_assets(args.student_name, output)
    for name, path in assets.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
