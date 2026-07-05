#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TASK1 PDF from the generated CSV and chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"


def register_fonts() -> str:
    if Path(SONGTI).exists():
        pdfmetrics.registerFont(TTFont("Songti", SONGTI))
        return "Songti"
    return "Helvetica"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def fmt_date(value: int | str) -> str:
    s = str(value)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def build_pdf(student_name: str, output: Path) -> None:
    font = register_fonts()
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
    heading = ParagraphStyle(
        "heading",
        parent=body,
        firstLineIndent=0,
        textColor=colors.black,
    )
    title = ParagraphStyle(
        "title",
        parent=body,
        fontSize=14,
        leading=21,
        alignment=TA_CENTER,
        firstLineIndent=0,
    )
    caption = ParagraphStyle(
        "caption",
        parent=body,
        fontSize=10.5,
        leading=15.75,
        alignment=TA_CENTER,
        firstLineIndent=0,
    )
    code_style = ParagraphStyle(
        "code",
        fontName=font,
        fontSize=8.5,
        leading=12.75,
        leftIndent=0,
        firstLineIndent=0,
        spaceBefore=0,
        spaceAfter=0,
        borderColor=colors.lightgrey,
        borderWidth=0.5,
        borderPadding=6,
        backColor=colors.whitesmoke,
    )

    csv_path = HERE / "000776_SZ_daily.csv"
    chart_path = HERE / "000776_SZ_close.png"
    script_path = HERE / "fetch_tushare_stock.py"
    df = pd.read_csv(csv_path)
    rows = len(df)
    start_date = fmt_date(df["trade_date"].iloc[0])
    end_date = fmt_date(df["trade_date"].iloc[-1])
    start_close = float(df["close"].iloc[0])
    end_close = float(df["close"].iloc[-1])
    high_close = float(df["close"].max())
    low_close = float(df["close"].min())
    change_pct = (end_close / start_close - 1) * 100
    source = str(df["data_source"].iloc[-1])

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        title=f"{student_name}TASK1",
        author=student_name,
    )

    story = [
        p(f"{student_name}TASK1", title),
        Spacer(1, 0.3 * cm),
        p("一、相较于传统手工操作交易的方法，量化交易有哪些优势？", heading),
        p(
            "量化交易把交易思想转化为可以被计算机执行的规则，核心优势是纪律性强、处理数据速度快、可回测验证、可同时监控多市场多品种，并且便于复盘和持续改进。传统手工交易更依赖个人经验和临场判断，容易受到情绪、注意力和执行速度限制；量化交易可以在事前明确买入、卖出、止损、仓位等条件，减少冲动交易和临时改变计划的情况。",
            body,
        ),
        p(
            "量化交易还可以利用历史数据评估策略表现，例如收益率、最大回撤、胜率和交易频率等指标，从而在真实投入资金前先发现规则缺陷。对需要快速响应的行情，程序能够自动读取数据、计算指标并发出交易信号，执行效率通常高于手工操作。不过量化交易并不等于稳赚，它仍然受到数据质量、模型失效、参数过拟合和市场结构变化等风险影响。",
            body,
        ),
        Spacer(1, 0.2 * cm),
        p("二、基本概念解释：K 线、基本面、技术面", heading),
        p(
            "K 线是用一个交易周期内的开盘价、收盘价、最高价和最低价表示价格变化的图形。日 K 线表示一天的价格变化，周 K 线表示一周的价格变化。通常实体部分反映开盘价和收盘价之间的差距，上下影线反映最高价和最低价。K 线可以直观展示价格波动、趋势强弱和买卖力量变化。",
            body,
        ),
        p(
            "基本面是指影响公司长期价值的经营和财务因素，例如营业收入、净利润、现金流、资产负债率、行业景气度、竞争格局、管理能力和政策环境等。基本面分析关注的是一家公司是否具有持续盈利能力，以及当前市场价格是否合理。",
            body,
        ),
        p(
            "技术面是指基于价格、成交量和市场交易行为进行分析的方法，常见内容包括均线、成交量、支撑位、压力位、趋势线、MACD、RSI 等指标。技术面更关注市场价格已经发生的变化和交易者行为，常用于判断买卖时机、趋势状态和短期风险。",
            body,
        ),
        PageBreak(),
        p("三、Python 获取股票交易数据、绘图并保存 CSV", heading),
        p(
            "本次选取本人持仓中的广发证券（000776.SZ）作为样本股票，时间区间为过去一年。程序读取 Tushare token 后，优先调用 Tushare Pro 的 daily 接口获取日线数据；本次实际运行时，当前 token 未开通 daily 接口权限，因此程序按预设降级到 Tushare 包的 legacy get_k_data 行情接口，仍然通过 Tushare 平台相关工具获取数据。输出文件包括 000776_SZ_daily.csv 和 000776_SZ_close.png。",
            body,
        ),
        p(
            f"本次运行结果：共获取 {rows} 个交易日数据，日期范围为 {start_date} 至 {end_date}；起始收盘价为 {start_close:.2f} 元，期末收盘价为 {end_close:.2f} 元，区间涨跌幅为 {change_pct:.2f}%；区间最高收盘价为 {high_close:.2f} 元，最低收盘价为 {low_close:.2f} 元。数据来源字段记录为 {source}。",
            body,
        ),
        Spacer(1, 0.2 * cm),
        KeepTogether([
            Image(str(chart_path), width=15.5 * cm, height=9.3 * cm),
            p("图1 广发证券（000776.SZ）过去一年每日收盘价曲线", caption),
            p(
                "图1显示，广发证券在样本期内总体呈现震荡上行走势。2025 年下半年股价先上升后回调，2026 年一季度附近出现阶段性下行，随后在 2026 年二季度重新走强，并在 2026 年 7 月初接近样本期较高位置。该走势说明单只股票价格会在趋势与回撤之间反复波动，后续若用于量化策略，还需要结合成交量、风险控制和买卖规则进一步分析。",
                body,
            ),
        ]),
        PageBreak(),
        p("四、主要程序代码", heading),
    ]

    code = script_path.read_text(encoding="utf-8")
    lines = code.splitlines()
    snippet = "\n".join(lines[:150])
    story.append(Preformatted(snippet, code_style))
    story.extend([
        Spacer(1, 0.2 * cm),
        p("五、CSV 数据样例", heading),
    ])

    sample = pd.concat([df.head(3), df.tail(3)])
    table_data = [["交易日期", "开盘价", "最高价", "最低价", "收盘价", "成交量"]] + [
        [
            fmt_date(row.trade_date),
            f"{row.open:.2f}",
            f"{row.high:.2f}",
            f"{row.low:.2f}",
            f"{row.close:.2f}",
            f"{row.vol:.0f}",
        ]
        for row in sample.itertuples(index=False)
    ]
    table = Table(table_data, colWidths=[2.7 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.6 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 13.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(p("表1 CSV 文件首尾交易日数据节选。完整数据已保存为 TASK1/000776_SZ_daily.csv，可供后续任务继续读取和分析。", caption))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="姓名")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else HERE / f"{args.student_name}TASK1.pdf"
    build_pdf(args.student_name, output)
    print(output)


if __name__ == "__main__":
    main()
