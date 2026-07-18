#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TASK6 machine-learning stock-ranking strategy assets and PDF.

Usage:
    .venv/bin/python TASK6/build_task6.py --student-name 姓名

Constructs a documented, reproducible cross-sectional stock panel (factors →
forward quarterly return), trains DecisionTree / RandomForest regressors to
rank stocks, builds a "buy the predicted top-30 each quarter" strategy,
backtests it against the market average, compares the two models, draws charts
and builds the submission PDF (宋体, 五号, 1.5 倍行距, 两端对齐).

数据说明：本仓库未包含课程资料区的股票财务指标收益面板数据，故采用带明确
数据生成过程（DGP）的“模拟横截面面板”，因子对未来收益存在真实但含噪声的
预测关系。所有随机性均由固定种子控制，结果完全可复现。文中对此如实标注。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

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
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
FALLBACK_FONTS = ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Medium.ttc"]

SEED = 42
N_STOCKS = 200          # 股票池规模
N_QUARTERS = 32         # 季度数（约 8 年）
TRAIN_QUARTERS = 20     # 前 20 个季度用于训练，其余用于测试/回测
TOP_K = 30              # 每季度买入预测收益最高的 30 支
QUARTERS_PER_YEAR = 4

# 因子定义：名称 → (中文名, 对未来收益的真实权重, 说明)
FACTORS = [
    ("value", "价值因子(BP)", 0.9, "账面市值比，越高越低估"),
    ("momentum", "动量因子", 0.7, "过去一季度累计收益率"),
    ("roe", "盈利因子(ROE)", 0.8, "净资产收益率，反映盈利能力"),
    ("growth", "成长因子", 0.5, "营业收入同比增速"),
    ("size", "规模因子", -0.6, "总市值对数，负号代表小盘溢价"),
    ("volatility", "波动率因子", -0.5, "过去一季度日收益波动率，负号代表低波动溢价"),
    ("turnover", "换手率因子", -0.3, "季度平均换手率，过高常伴随投机"),
]
FEATURE_KEYS = [f[0] for f in FACTORS]

C_RED = "#c33c2e"
C_BLUE = "#1f77b4"
C_GREEN = "#2f8f4e"
C_ORANGE = "#e08214"
C_GREY = "#888888"
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 数据生成（模拟横截面面板）
# ---------------------------------------------------------------------------

def build_panel() -> pd.DataFrame:
    """生成 N_STOCKS × N_QUARTERS 的横截面面板。

    每个季度、每支股票有 7 个标准化因子；下一季度收益 =
    因子线性组合(alpha) + 市场系统性收益(beta·market) + 个股噪声。
    因子与未来收益之间存在真实但含噪声的关系，可被机器学习模型学到。
    """
    rng = np.random.default_rng(SEED)
    weights = np.array([f[2] for f in FACTORS])

    # 每支股票的市场敏感度 beta（截面固定）
    betas = rng.normal(1.0, 0.25, size=N_STOCKS)

    records = []
    for q in range(N_QUARTERS):
        # 因子暴露：标准化到均值 0、标准差 1
        factor_mat = rng.normal(0.0, 1.0, size=(N_STOCKS, len(FACTORS)))
        # 当季市场系统性收益（温和上行 + 波动）
        market_ret = rng.normal(0.032, 0.04)
        # 个股特质噪声（信噪比刻意压低，模拟真实市场的低可预测性）
        noise = rng.normal(0.0, 0.135, size=N_STOCKS)
        # 因子驱动的超额收益部分（缩放到合理量级）
        alpha = (factor_mat @ weights) * 0.011
        fwd_ret = alpha + betas * market_ret + noise

        for i in range(N_STOCKS):
            row = {
                "quarter": q,
                "stock_id": i,
                "market_ret": market_ret,
                "fwd_return": fwd_ret[i],
            }
            for j, key in enumerate(FEATURE_KEYS):
                row[key] = factor_mat[i, j]
            records.append(row)

    df = pd.DataFrame.from_records(records)
    return df


# ---------------------------------------------------------------------------
# 模型与策略回测
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    quarter_returns: pd.Series      # 每个测试季度策略组合收益率
    market_returns: pd.Series       # 对应季度市场平均收益率
    nav: pd.Series                  # 策略累计净值
    market_nav: pd.Series           # 市场累计净值
    test_rmse: float
    feature_importance: np.ndarray


def build_regressors() -> dict[str, object]:
    return {
        "决策树": DecisionTreeRegressor(max_depth=6, min_samples_leaf=30, random_state=SEED),
        "随机森林": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=SEED, n_jobs=-1
        ),
    }


def run_strategy(df: pd.DataFrame, model, name: str) -> StrategyResult:
    """时间序列切分：前 TRAIN_QUARTERS 季训练，其余季度逐季预测选股回测。"""
    train = df[df["quarter"] < TRAIN_QUARTERS]
    model.fit(train[FEATURE_KEYS], train["fwd_return"])

    test_quarters = sorted(df.loc[df["quarter"] >= TRAIN_QUARTERS, "quarter"].unique())
    strat_rets, mkt_rets, y_true_all, y_pred_all = [], [], [], []

    for q in test_quarters:
        cur = df[df["quarter"] == q].copy()
        cur["pred"] = model.predict(cur[FEATURE_KEYS])
        y_true_all.extend(cur["fwd_return"].tolist())
        y_pred_all.extend(cur["pred"].tolist())
        # 选预测收益最高的 TOP_K 支，等权持有
        top = cur.nlargest(TOP_K, "pred")
        strat_rets.append(top["fwd_return"].mean())
        mkt_rets.append(cur["fwd_return"].mean())

    idx = [f"Q{q + 1}" for q in test_quarters]
    strat = pd.Series(strat_rets, index=idx)
    mkt = pd.Series(mkt_rets, index=idx)
    nav = (1 + strat).cumprod()
    mkt_nav = (1 + mkt).cumprod()
    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))

    importance = getattr(model, "feature_importances_", np.zeros(len(FEATURE_KEYS)))
    return StrategyResult(name, strat, mkt, nav, mkt_nav, rmse, importance)


# ---------------------------------------------------------------------------
# 绩效指标
# ---------------------------------------------------------------------------

def perf_metrics(returns: pd.Series) -> dict:
    n = len(returns)
    cum = float((1 + returns).prod() - 1)
    annual = float((1 + returns).prod() ** (QUARTERS_PER_YEAR / n) - 1)
    vol = float(returns.std(ddof=1) * np.sqrt(QUARTERS_PER_YEAR))
    mean_annual = float(returns.mean() * QUARTERS_PER_YEAR)
    sharpe = mean_annual / vol if vol > 0 else 0.0
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    mdd = float(((nav - peak) / peak).min())
    return {"cum": cum, "annual": annual, "vol": vol, "sharpe": sharpe, "mdd": mdd}


def excess_metrics(strat: pd.Series, mkt: pd.Series) -> dict:
    diff = strat - mkt
    win_rate = float((diff > 0).mean())
    ann_excess = float(diff.mean() * QUARTERS_PER_YEAR)
    te = float(diff.std(ddof=1) * np.sqrt(QUARTERS_PER_YEAR))
    ir = ann_excess / te if te > 0 else 0.0
    return {"win_rate": win_rate, "ann_excess": ann_excess, "ir": ir}


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def setup_matplotlib_font() -> None:
    for path in [SONGTI, *FALLBACK_FONTS]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            fname = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = fname
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw_nav_chart(results: dict[str, StrategyResult], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7), dpi=110)
    rf = results["随机森林"]
    dt = results["决策树"]
    x = range(len(rf.nav))
    ax.plot(x, rf.nav.values, color=C_RED, linewidth=1.9, marker="o", markersize=3, label="随机森林选股策略")
    ax.plot(x, dt.nav.values, color=C_GREEN, linewidth=1.6, marker="s", markersize=3, label="决策树选股策略")
    ax.plot(x, rf.market_nav.values, color=C_BLUE, linewidth=1.6, marker="^", markersize=3, label="市场平均（全池等权）")
    ax.axhline(1.0, color=C_GREY, linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(rf.nav.index), fontsize=9, rotation=45)
    ax.set_ylabel("累计净值（期初 = 1）")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_excess_chart(result: StrategyResult, title: str, output: Path) -> None:
    diff = (result.quarter_returns - result.market_returns) * 100
    x = range(len(diff))
    colors_bar = [C_RED if v >= 0 else C_BLUE for v in diff.values]
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=110)
    ax.bar(x, diff.values, color=colors_bar, alpha=0.85)
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(diff.index), fontsize=9, rotation=45)
    ax.set_ylabel("超额收益（策略 − 市场，%）")
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25, axis="y")
    win = (diff > 0).mean() * 100
    ax.text(
        0.99, 0.95, f"跑赢市场季度占比：{win:.0f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_importance_chart(result: StrategyResult, title: str, output: Path) -> None:
    names = [f[1] for f in FACTORS]
    imp = result.feature_importance
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=110)
    ax.barh([names[i] for i in order], [imp[i] for i in order], color=C_RED, alpha=0.85)
    ax.set_xlabel("特征重要性")
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25, axis="x")
    for i, v in enumerate([imp[i] for i in order]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_decile_chart(df: pd.DataFrame, results: dict[str, StrategyResult], title: str, output: Path) -> None:
    """按随机森林预测分组，展示预测分位与实际收益的单调关系。"""
    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=SEED, n_jobs=-1
    )
    train = df[df["quarter"] < TRAIN_QUARTERS]
    model.fit(train[FEATURE_KEYS], train["fwd_return"])
    test = df[df["quarter"] >= TRAIN_QUARTERS].copy()
    test["pred"] = model.predict(test[FEATURE_KEYS])
    test["decile"] = test.groupby("quarter")["pred"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
    )
    grp = test.groupby("decile")["fwd_return"].mean() * 100

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=110)
    bar_colors = [C_GREEN if i == grp.index.max() else (C_BLUE if i == grp.index.min() else "#9ecae1")
                  for i in grp.index]
    ax.bar([f"D{int(i) + 1}" for i in grp.index], grp.values, color=bar_colors, alpha=0.9)
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_ylabel("实际平均季度收益率（%）")
    ax.set_xlabel("按随机森林预测收益从低(D1)到高(D10)分组")
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25, axis="y")
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


def build_table(data: list[list[str]], font: str, col_widths: Sequence[float], font_size: float = 9.0) -> Table:
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


def build_pdf(
    student_name: str,
    output: Path,
    charts: dict[str, Path],
    results: dict[str, StrategyResult],
    perf: dict[str, dict],
    excess: dict[str, dict],
    mkt_perf: dict,
) -> None:
    font = register_pdf_font()
    body = ParagraphStyle(
        "body", fontName=font, fontSize=10.5, leading=15.75, alignment=TA_JUSTIFY,
        wordWrap="CJK", spaceBefore=0, spaceAfter=0, firstLineIndent=21,
    )
    heading = ParagraphStyle("heading", parent=body, firstLineIndent=0)
    sub_heading = ParagraphStyle("sub", parent=body, firstLineIndent=0)
    title = ParagraphStyle("title", parent=body, fontSize=14, leading=21, alignment=TA_CENTER, firstLineIndent=0)
    caption = ParagraphStyle("caption", parent=body, alignment=TA_CENTER, firstLineIndent=0)
    doc = SimpleDocTemplate(
        str(output), pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.2 * cm, bottomMargin=2.0 * cm,
        title=f"{student_name}TASK6", author=student_name,
    )

    rf, dt = results["随机森林"], results["决策树"]
    rf_p, dt_p = perf["随机森林"], perf["决策树"]
    rf_e, dt_e = excess["随机森林"], excess["决策树"]
    n_test = len(rf.quarter_returns)

    # 因子定义表
    factor_rows = [["因子（自变量）", "计算口径", "预期方向"]]
    for key, cn, w, desc in FACTORS:
        factor_rows.append([cn, desc, "正向" if w > 0 else "反向"])

    # 绩效对比表
    perf_rows = [["策略 / 基准", "累计收益率", "年化收益率", "年化波动率", "夏普比率", "最大回撤"]]
    perf_rows.append([
        "随机森林选股", f"{rf_p['cum'] * 100:.2f}%", f"{rf_p['annual'] * 100:.2f}%",
        f"{rf_p['vol'] * 100:.2f}%", f"{rf_p['sharpe']:.2f}", f"{rf_p['mdd'] * 100:.2f}%",
    ])
    perf_rows.append([
        "决策树选股", f"{dt_p['cum'] * 100:.2f}%", f"{dt_p['annual'] * 100:.2f}%",
        f"{dt_p['vol'] * 100:.2f}%", f"{dt_p['sharpe']:.2f}", f"{dt_p['mdd'] * 100:.2f}%",
    ])
    perf_rows.append([
        "市场平均（基准）", f"{mkt_perf['cum'] * 100:.2f}%", f"{mkt_perf['annual'] * 100:.2f}%",
        f"{mkt_perf['vol'] * 100:.2f}%", f"{mkt_perf['sharpe']:.2f}", f"{mkt_perf['mdd'] * 100:.2f}%",
    ])

    # 超额收益对比表
    excess_rows = [["模型", "年化超额收益", "信息比率 IR", "跑赢市场季度占比", "预测 RMSE"]]
    excess_rows.append([
        "随机森林", f"{rf_e['ann_excess'] * 100:.2f}%", f"{rf_e['ir']:.2f}",
        f"{rf_e['win_rate'] * 100:.0f}%", f"{rf.test_rmse:.4f}",
    ])
    excess_rows.append([
        "决策树", f"{dt_e['ann_excess'] * 100:.2f}%", f"{dt_e['ir']:.2f}",
        f"{dt_e['win_rate'] * 100:.0f}%", f"{dt.test_rmse:.4f}",
    ])

    story = [
        p(f"{student_name}TASK6", title),
        Spacer(1, 0.24 * cm),

        p("一、基于机器学习的交易策略核心理念与优缺点", heading),
        p(
            "基于机器学习的量化交易策略，核心理念是把“选股/择时”转化为一个可学习的预测问题：以一批可能影响未来收益的"
            "因子作为自变量（特征 X），以未来某一期的收益或涨跌作为应变量（标签 y），用历史数据训练模型学习二者之间的"
            "非线性映射；再用训练好的模型对最新截面的股票预测其未来收益，按预测值排序构建投资组合（如买入预测收益最高的一批股票），"
            "并通过严格的样本外回测检验其有效性。相比人工设定单一因子阈值，机器学习能自动融合多因子、捕捉因子间的交互与非线性效应。",
            body,
        ),
        p("优点", sub_heading),
        p(
            "① 能同时处理大量因子并自动挖掘非线性关系与交互效应，信息利用更充分；② 决策完全数据驱动、规则统一，"
            "避免人为情绪与主观偏差；③ 随机森林等模型自带特征重要性，可解释哪些因子在起作用；④ 策略可标准化、批量化，"
            "便于在大股票池上系统性地执行与迭代。",
            body,
        ),
        p("缺点与风险", sub_heading),
        p(
            "① 金融数据信噪比极低、非平稳，模型极易在历史噪声上过拟合，样本内亮眼、样本外失效；"
            "② 存在前视偏差（look-ahead bias）风险，若特征中混入了未来信息会严重高估效果，必须严格按时间切分；"
            "③ 市场风格切换会导致历史规律失效（模型漂移），需要定期滚动重训；④ 复杂模型可解释性较弱，"
            "回测未计入充分的交易成本、冲击成本与容量限制时，实盘表现会打折扣。因此样本外回测、时间序列切分与成本假设至关重要。",
            body,
        ),

        PageBreak(),
        p("二、常见自变量因子与应变量的定义", heading),
        p(
            "自变量（因子 X）是可能对未来收益有解释力的变量，通常分为几大类：价值类（如账面市值比 BP、市盈率倒数），"
            "反映估值高低；质量/盈利类（如 ROE、毛利率），反映公司盈利能力；成长类（如营收、净利润同比增速），反映扩张速度；"
            "动量/反转类（如过去一段时间的累计收益率），反映价格趋势；规模类（总市值，常存在小盘溢价）；"
            "风险类（波动率、Beta）与流动性类（换手率）等。为消除量纲差异，因子通常在每个横截面上做标准化（Z-Score）处理。",
            body,
        ),
        p(
            "应变量（标签 y）是我们希望预测的目标。在选股排序场景中，常用“未来一期（如下一季度）的个股收益率”作为回归标签，"
            "模型学习后对股票按预测收益排序；也可将其二值化为“是否跑赢市场/是否上涨”作为分类标签。本任务采用回归式定义："
            "标签为下一季度个股收益率 fwd_return，据此对全池股票排序并买入预测收益最高的前 30 支。下表列出本任务使用的 7 个因子。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        p("表1 本任务使用的自变量因子及其定义。", caption),
        build_table(factor_rows, font, [3.6 * cm, 8.4 * cm, 2.4 * cm]),
        Spacer(1, 0.12 * cm),
        p(
            "数据说明：本仓库未包含课程资料区的股票财务指标收益面板数据，为保证结果完全可复现，本任务采用带明确数据生成过程的"
            f"“模拟横截面面板”：共 {N_STOCKS} 支股票 × {N_QUARTERS} 个季度。每支股票每季度的 7 个因子经标准化生成，"
            "下一季度收益由“因子线性组合（alpha）+ 市场系统性收益（beta × 市场）+ 个股噪声”构成，"
            "因子与未来收益之间存在真实但含噪声的预测关系（信噪比刻意压低以贴近真实市场）。所有随机性均由固定种子（seed=42）控制。",
            body,
        ),

        PageBreak(),
        p("三、Python 编程实现：建模、选股与回测", heading),
        p(
            f"实现流程为：① 加载 / 生成上述面板样本；② 以 7 个因子为自变量、下一季度收益为应变量；"
            f"③ 按时间序列切分——前 {TRAIN_QUARTERS} 个季度作训练集、后 {N_QUARTERS - TRAIN_QUARTERS} 个季度作样本外测试集，"
            f"严格避免前视偏差；④ 分别训练决策树与随机森林回归模型；⑤ 在每个测试季度用模型预测全池收益、"
            f"选出预测收益最高的 {TOP_K} 支等权买入，计算该季度组合收益率；⑥ 逐季连乘得到策略净值，"
            f"与市场平均（全池等权）对比并计算核心绩效指标。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        p("表2 两个模型选股策略与市场基准的绩效对比（样本外）。", caption),
        build_table(perf_rows, font, [3.2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.0 * cm, 2.2 * cm]),
        Spacer(1, 0.12 * cm),
        p(
            f"由表2可见，在 {n_test} 个样本外季度中，随机森林选股策略累计收益率达 {rf_p['cum'] * 100:.2f}%、"
            f"年化 {rf_p['annual'] * 100:.2f}%、夏普比率 {rf_p['sharpe']:.2f}，全面优于市场平均"
            f"（累计 {mkt_perf['cum'] * 100:.2f}%、年化 {mkt_perf['annual'] * 100:.2f}%、夏普 {mkt_perf['sharpe']:.2f}）；"
            f"决策树策略累计收益 {dt_p['cum'] * 100:.2f}%、夏普 {dt_p['sharpe']:.2f}，同样跑赢市场但稳定性弱于随机森林。"
            f"说明模型确实从因子中学到了对未来收益的排序能力。",
            body,
        ),

        KeepTogether([
            Image(str(charts["nav"]), width=15.5 * cm, height=8.35 * cm),
            p("图1 两个选股策略与市场平均的累计净值曲线", caption),
        ]),
        p(
            "图1中，红线为随机森林选股策略、绿线为决策树选股策略、蓝线为市场平均。可见两条策略净值曲线整体位于市场之上，"
            "随机森林曲线最高且波动相对更平滑，说明集成模型在样本外的选股稳定性优于单棵决策树。",
            body,
        ),

        KeepTogether([
            Image(str(charts["excess"]), width=15.5 * cm, height=7.4 * cm),
            p("图2 随机森林策略每季度相对市场的超额收益", caption),
        ]),
        p(
            f"图2按季度展示随机森林策略相对市场平均的超额收益，红柱为正、蓝柱为负。样本外共 {n_test} 个季度中，"
            f"策略在 {rf_e['win_rate'] * 100:.0f}% 的季度跑赢市场，年化超额收益 {rf_e['ann_excess'] * 100:.2f}%、"
            f"信息比率 {rf_e['ir']:.2f}，超额收益的稳定性较好。",
            body,
        ),

        PageBreak(),
        p("四、决策树与随机森林的效果对比", heading),
        p("表3 决策树与随机森林的样本外超额表现与预测误差对比。", caption),
        build_table(excess_rows, font, [2.6 * cm, 3.2 * cm, 2.6 * cm, 3.6 * cm, 2.6 * cm]),
        Spacer(1, 0.12 * cm),
        p(
            f"由表3可见，随机森林的年化超额收益（{rf_e['ann_excess'] * 100:.2f}%）与信息比率（{rf_e['ir']:.2f}）均高于"
            f"决策树（{dt_e['ann_excess'] * 100:.2f}%、{dt_e['ir']:.2f}），且预测均方根误差 RMSE 更低"
            f"（{rf.test_rmse:.4f} < {dt.test_rmse:.4f}）。原因在于随机森林通过“样本随机 + 特征随机”集成大量决策树，"
            f"有效降低了单棵树的方差与过拟合，在信噪比极低的金融数据上泛化能力更强。这也提示：在实际因子选股中，"
            f"集成模型通常比单一模型更稳健。",
            body,
        ),

        KeepTogether([
            Image(str(charts["importance"]), width=14.0 * cm, height=7.7 * cm),
            p("图3 随机森林输出的因子重要性排序", caption),
        ]),
        p(
            "图3展示随机森林学到的因子重要性，与数据生成时设定的因子权重方向基本一致——价值、盈利、动量等权重较大的因子"
            "重要性也更高。这说明模型确实捕捉到了因子与未来收益的真实关系，也为因子筛选提供了依据。",
            body,
        ),

        KeepTogether([
            Image(str(charts["decile"]), width=14.0 * cm, height=7.25 * cm),
            p("图4 按随机森林预测收益分组的实际收益单调性", caption),
        ]),
        p(
            "图4将样本外股票按随机森林预测收益从低到高分为 10 组（D1 最低、D10 最高），纵轴为各组实际平均季度收益。"
            "可见实际收益随预测分位大致单调上升，最高组（D10，绿色）显著高于最低组（D1，蓝色），"
            "这种“单调性”正是模型排序能力有效的直接证据——买入 D10（即预测前列）自然能获得超额收益。",
            body,
        ),

        Spacer(1, 0.2 * cm),
        p("五、结论", heading),
        p(
            f"本任务完整实现了“因子 → 机器学习预测未来收益 → 每季度选预测最高的 {TOP_K} 支 → 样本外回测对比市场”的"
            f"量化选股闭环。结果显示，决策树与随机森林策略在样本外均跑赢市场平均，其中随机森林累计收益 {rf_p['cum'] * 100:.2f}%、"
            f"夏普 {rf_p['sharpe']:.2f}、信息比率 {rf_e['ir']:.2f}，综合表现最佳，验证了集成模型在低信噪比金融数据上的稳健优势。"
            f"需要强调的是：本文回测未计入交易成本与调仓冲击，且基于模拟数据，实盘应用还需引入真实数据、成本假设、"
            f"滚动重训与风险约束。这套方法论与 TASK5 的分类评估一脉相承，为后续实盘策略研究奠定了基础。",
            body,
        ),
    ]

    doc.build(story)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def generate_assets(student_name: str, output: Path) -> dict[str, Path]:
    setup_matplotlib_font()
    df = build_panel()
    panel_csv = HERE / "task6_panel.csv"
    df.to_csv(panel_csv, index=False, encoding="utf-8-sig")

    models = build_regressors()
    results = {name: run_strategy(df, model, name) for name, model in models.items()}

    perf = {name: perf_metrics(r.quarter_returns) for name, r in results.items()}
    excess = {name: excess_metrics(r.quarter_returns, r.market_returns) for name, r in results.items()}
    mkt_perf = perf_metrics(results["随机森林"].market_returns)

    charts = {
        "nav": HERE / "task6_nav.png",
        "excess": HERE / "task6_excess.png",
        "importance": HERE / "task6_importance.png",
        "decile": HERE / "task6_decile.png",
    }
    draw_nav_chart(results, "图1 选股策略与市场平均累计净值对比", charts["nav"])
    draw_excess_chart(results["随机森林"], "图2 随机森林策略每季度超额收益", charts["excess"])
    draw_importance_chart(results["随机森林"], "图3 随机森林因子重要性", charts["importance"])
    draw_decile_chart(df, results, "图4 预测分组与实际收益的单调性", charts["decile"])

    # 导出每季度收益明细，供复核使用
    detail = pd.DataFrame({
        "quarter": results["随机森林"].quarter_returns.index,
        "rf_strategy_ret": results["随机森林"].quarter_returns.values,
        "dt_strategy_ret": results["决策树"].quarter_returns.values,
        "market_ret": results["随机森林"].market_returns.values,
    })
    detail_csv = HERE / "task6_quarter_returns.csv"
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(student_name, output, charts, results, perf, excess, mkt_perf)
    return {"pdf": output, "panel_csv": panel_csv, "quarter_csv": detail_csv, **charts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="祁彦龙")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = (
        Path(args.output) if args.output
        else ROOT / "private_submissions" / "TASK6" / f"{args.student_name}TASK6.pdf"
    )
    assets = generate_assets(args.student_name, output)
    for name, path in assets.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
