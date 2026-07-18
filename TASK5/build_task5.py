#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TASK5 machine-learning classification assets and PDF.

Usage:
    .venv/bin/python TASK5/build_task5.py --student-name 姓名

Loads the scikit-learn breast-cancer binary dataset, splits train/test,
trains logistic regression / decision tree / random forest, evaluates with
confusion matrix / AUC / ROC, draws charts and builds the submission PDF
(宋体, 五号, 1.5 倍行距, 两端对齐).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

RANDOM_STATE = 42
TEST_SIZE = 0.30

# 颜色（与前序任务保持一致的配色）
C_RED = "#c33c2e"
C_BLUE = "#1f77b4"
C_GREEN = "#2f8f4e"
C_GREY = "#888888"


# ---------------------------------------------------------------------------
# 数据与建模
# ---------------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """加载 scikit-learn 乳腺癌二分类数据集。

    目标变量约定：0 = 恶性(malignant)，1 = 良性(benign)。
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, list(data.feature_names), list(data.target_names)


def build_models() -> dict[str, Pipeline]:
    """构建三个分类模型；逻辑回归前置标准化。"""
    return {
        "逻辑回归": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "决策树": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
        ]),
        "随机森林": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_score": y_score,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": auc(fpr, tpr),
            "fpr": fpr,
            "tpr": tpr,
            "cm": confusion_matrix(y_test, y_pred),
        }
    return results


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


def draw_roc_chart(results: dict[str, dict], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=115)
    palette = {"逻辑回归": C_BLUE, "决策树": C_GREEN, "随机森林": C_RED}
    for name, r in results.items():
        ax.plot(
            r["fpr"], r["tpr"], linewidth=1.9, color=palette[name],
            label=f"{name}（AUC = {r['auc']:.3f}）",
        )
    ax.plot([0, 1], [0, 1], color=C_GREY, linewidth=1.0, linestyle="--", label="随机猜测（AUC = 0.5）")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("假阳性率 FPR = FP /（FP + TN）")
    ax.set_ylabel("真阳性率 TPR = TP /（TP + FN）")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_confusion_chart(cm: np.ndarray, labels: list[str], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=115)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"预测 {labels[0]}", f"预测 {labels[1]}"], fontsize=11)
    ax.set_yticklabels([f"实际 {labels[0]}", f"实际 {labels[1]}"], fontsize=11)
    thresh = cm.max() / 2.0
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cell_names[i][j]}\n{cm[i, j]}",
                ha="center", va="center", fontsize=15,
                color="white" if cm[i, j] > thresh else "#222222",
            )
    ax.set_title(title, fontsize=15)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_importance_chart(model: Pipeline, feature_names: list[str], title: str, output: Path, top_n: int = 12) -> None:
    importances = model.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    values = [importances[i] for i in order][::-1]
    fig, ax = plt.subplots(figsize=(11, 7), dpi=115)
    ax.barh(range(len(values)), values, color=C_RED, alpha=0.85)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("特征重要性（基尼不纯度下降）")
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25, axis="x")
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_metrics_chart(results: dict[str, dict], title: str, output: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["准确率", "精确率", "召回率", "F1", "AUC"]
    names = list(results.keys())
    palette = {"逻辑回归": C_BLUE, "决策树": C_GREEN, "随机森林": C_RED}
    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=115)
    for k, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + (k - 1) * width, vals, width=width, color=palette[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.8, 1.005)
    ax.set_ylabel("指标值")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="lower right", fontsize=10)
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
    results: dict[str, dict],
    dataset_meta: dict,
    target_names: list[str],
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
    sub_heading = ParagraphStyle("sub", parent=body, firstLineIndent=0)
    title = ParagraphStyle("title", parent=body, fontSize=14, leading=21, alignment=TA_CENTER, firstLineIndent=0)
    caption = ParagraphStyle("caption", parent=body, alignment=TA_CENTER, firstLineIndent=0)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        title=f"{student_name}TASK5",
        author=student_name,
    )

    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    lr, dt, rf = results["逻辑回归"], results["决策树"], results["随机森林"]
    cmat = rf["cm"]
    tn, fp, fn, tp = int(cmat[0, 0]), int(cmat[0, 1]), int(cmat[1, 0]), int(cmat[1, 1])
    label_mal, label_ben = target_names  # ['malignant', 'benign'] → 0/1

    # 指标对比表
    metric_rows = [["模型", "准确率", "精确率", "召回率", "F1 分数", "AUC"]]
    for name in ["逻辑回归", "决策树", "随机森林"]:
        r = results[name]
        metric_rows.append([
            name,
            f"{r['accuracy'] * 100:.2f}%",
            f"{r['precision'] * 100:.2f}%",
            f"{r['recall'] * 100:.2f}%",
            f"{r['f1']:.3f}",
            f"{r['auc']:.3f}",
        ])

    # 混淆矩阵表（随机森林）
    cm_rows = [
        ["", f"预测=良性(1)", f"预测=恶性(0)", "合计"],
        ["实际=良性(1)", f"TP = {tp}", f"FN = {fn}", str(tp + fn)],
        ["实际=恶性(0)", f"FP = {fp}", f"TN = {tn}", str(fp + tn)],
        ["合计", str(tp + fp), str(fn + tn), str(tp + fp + fn + tn)],
    ]

    story = [
        p(f"{student_name}TASK5", title),
        Spacer(1, 0.24 * cm),

        p("一、分类型机器学习算法概述", heading),
        p(
            "分类（Classification）是监督学习中最基础的任务之一：模型从带标签的历史样本中学习特征与离散类别之间的映射关系，"
            "再对新样本预测其所属类别。当类别只有两种（记为 0 与 1）时称为二分类，例如医学上的“恶性/良性”、量化交易中的“下一期上涨/下跌”。"
            "本任务重点理解逻辑回归、决策树与随机森林三种常用分类算法。",
            body,
        ),
        p("1. 逻辑回归（Logistic Regression）", sub_heading),
        p(
            "逻辑回归是线性模型在分类问题上的推广。它先对特征做线性加权求和 z = w·x + b，再通过 Sigmoid 函数"
            " σ(z) = 1 /（1 + e⁻ᶻ）把结果压缩到 (0, 1) 区间，得到样本属于正类的概率；以 0.5 为默认阈值即可判定类别。"
            "训练时通过极大似然（最小化对数损失）估计权重 w。它的优点是模型简单、训练快、可解释性强——每个特征的系数正负与大小"
            "直接反映其对结果的方向与影响程度，且天然输出概率，便于配合 ROC/AUC 评估；缺点是只能刻画线性决策边界，"
            "对特征间复杂的非线性交互能力有限，通常需要先做特征标准化。",
            body,
        ),
        p("2. 决策树（Decision Tree）", sub_heading),
        p(
            "决策树以“若……则……”的层层判断构建一棵树：每个内部节点选择一个特征和阈值把样本一分为二，"
            "选择依据是使划分后子集的“纯度”最大提升（分类树常用基尼不纯度或信息增益），叶节点给出最终类别。"
            "它的优点是无需特征标准化、能自动捕捉非线性关系和特征交互、决策路径直观易解释；缺点是单棵树对训练数据敏感、"
            "容易过拟合，需要通过限制树深、剪枝或设定叶节点最小样本数来控制复杂度。",
            body,
        ),
        p("3. 随机森林（Random Forest）", sub_heading),
        p(
            "随机森林是决策树的集成（Bagging）方法：它对训练集做有放回抽样生成多个子样本，各自训练一棵决策树，"
            "并在每次分裂时只从随机抽取的一部分特征中挑选最优划分，最后由所有树投票（分类）得到结果。"
            "通过“样本随机 + 特征随机”降低了单棵树的方差，显著缓解过拟合，通常在无需精细调参的情况下即可取得稳健且优秀的表现，"
            "还能输出特征重要性；代价是可解释性弱于单棵树、模型体积和预测开销更大。三者的关系可概括为：逻辑回归重解释与线性，"
            "决策树重规则与非线性，随机森林用集成换取更强的泛化能力。",
            body,
        ),

        PageBreak(),
        p("二、机器学习分类模型的评价指标", heading),
        p(
            "对于二分类模型，仅看“准确率”往往不够，尤其在正负样本不平衡时容易产生误导，因此需要一整套评价体系。"
            "所有指标都建立在混淆矩阵之上。",
            body,
        ),
        p("1. 混淆矩阵（Confusion Matrix）", sub_heading),
        p(
            "混淆矩阵是一张 2×2 的表格，按“实际类别 × 预测类别”统计四类结果：真正例 TP（实际为正、预测为正）、"
            "真负例 TN（实际为负、预测为负）、假正例 FP（实际为负、预测为正，即“误报”）、假负例 FN（实际为正、预测为负，即“漏报”）。"
            "由它可派生出一系列指标：准确率 Accuracy =（TP + TN）/ 总数，衡量整体判对比例；"
            "精确率 Precision = TP /（TP + FP），衡量“预测为正的样本里有多少真的是正”，关注误报代价；"
            "召回率 Recall = TP /（TP + FN），衡量“真正的正样本里有多少被找了出来”，关注漏报代价；"
            "F1 分数是精确率与召回率的调和平均 F1 = 2·P·R /（P + R），在两者间取得平衡。",
            body,
        ),
        p("2. ROC 曲线（Receiver Operating Characteristic）", sub_heading),
        p(
            "分类模型通常先输出一个 (0, 1) 的概率分数，再用阈值判定类别。改变阈值会同时改变真阳性率"
            " TPR = TP /（TP + FN）（即召回率）与假阳性率 FPR = FP /（FP + TN）。ROC 曲线就是以 FPR 为横轴、TPR 为纵轴，"
            "把所有可能阈值下的 (FPR, TPR) 点连成的曲线。曲线越靠近左上角，说明模型能在较低误报率下获得较高命中率，性能越好；"
            "对角虚线代表随机猜测。ROC 的优势在于它不依赖单一阈值，也对类别比例变化相对稳健。",
            body,
        ),
        p("3. AUC（Area Under the ROC Curve）", sub_heading),
        p(
            "AUC 是 ROC 曲线下的面积，把整条曲线浓缩为一个 0~1 的标量。其统计含义是：随机抽取一个正样本和一个负样本，"
            "模型给正样本打出更高分数的概率。AUC = 0.5 相当于随机猜测，AUC = 1 表示完美区分，通常认为 0.7 以上可用、"
            "0.8 以上良好、0.9 以上优秀。因为它与阈值无关、对不平衡数据稳健，AUC 是比较不同分类模型排序能力的常用统一标准，"
            "在量化选股中尤其契合“对样本按预期收益高低排序”的需求。",
            body,
        ),

        PageBreak(),
        p("三、Python 编程实现与结果分析", heading),
        p(
            f"数据集选用 scikit-learn 内置的威斯康星乳腺癌诊断数据集（load_breast_cancer），"
            f"共 {dataset_meta['n_samples']} 个样本、{dataset_meta['n_features']} 个由细胞核图像计算得到的数值特征"
            f"（如半径、纹理、周长、面积、光滑度等的均值、标准差与最差值）。目标为二分类变量，约定 0 = 恶性（malignant）、"
            f"1 = 良性（benign），其中良性 {dataset_meta['n_pos']} 例、恶性 {dataset_meta['n_neg']} 例。"
            f"该数据集结构与量化中的“财务指标 → 涨跌标签”一致，可直接迁移到股票二分类场景。",
            body,
        ),
        p(
            f"建模流程为：① 使用 train_test_split 按 {int(TEST_SIZE * 100)}% 比例、分层抽样（stratify）划分训练集与测试集，"
            f"随机种子固定为 {RANDOM_STATE} 以保证结果可复现；② 分别构建逻辑回归（前置 StandardScaler 标准化）、"
            f"决策树（max_depth=4）与随机森林（300 棵树）三个模型并在训练集上拟合；③ 在测试集上计算混淆矩阵、"
            f"准确率、精确率、召回率、F1 与 AUC；④ 绘制三个模型的 ROC 曲线并比较 AUC。",
            body,
        ),
        Spacer(1, 0.15 * cm),
        p("表1 三个分类模型在测试集上的评价指标对比。", caption),
        build_table(metric_rows, font, [3.0 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm]),
        Spacer(1, 0.15 * cm),
        p(
            f"由表1可见，三个模型在测试集上均取得了不错的分类效果。其中逻辑回归 AUC 最高，达 {lr['auc']:.3f}、"
            f"准确率 {lr['accuracy'] * 100:.2f}%，说明该数据集在标准化后具有很强的线性可分性；随机森林次之，"
            f"AUC {rf['auc']:.3f}、准确率 {rf['accuracy'] * 100:.2f}%；单棵决策树受限于结构简单、方差较大，"
            f"AUC（{dt['auc']:.3f}）明显低于前两者，这正体现了随机森林通过集成多棵树降低方差、改善单树过拟合的价值。",
            body,
        ),

        KeepTogether([
            Image(str(charts["roc"]), width=11.5 * cm, height=9.07 * cm),
            p("图1 三个分类模型的 ROC 曲线与 AUC 对比", caption),
        ]),
        p(
            "图1中，横轴为假阳性率、纵轴为真阳性率，曲线越贴近左上角越好。三条曲线均紧贴左上角、远离对角虚线，"
            "说明各模型在极低误报率下即可识别出绝大多数正样本；逻辑回归（蓝）与随机森林（红）曲线几乎重合于顶端、AUC 最高，决策树（绿）曲线略低于二者。",
            body,
        ),

        KeepTogether([
            Image(str(charts["metrics"]), width=13.5 * cm, height=7.85 * cm),
            p("图2 三个模型五项评价指标的柱状对比", caption),
        ]),
        p(
            "图2把准确率、精确率、召回率、F1 与 AUC 并列展示（纵轴自 0.8 起以放大差异）。可见随机森林在多数指标上略占优，"
            "决策树的召回率相对偏低，意味着它漏判正样本（FN）稍多，这与其单树结构的方差偏大一致。",
            body,
        ),

        PageBreak(),
        p("四、最优模型的混淆矩阵与特征重要性", heading),
        p(
            f"下面以能够输出特征重要性的随机森林（集成模型的代表）为例，展示其在测试集上的混淆矩阵。测试集共 {tp + tn + fp + fn} 个样本，"
            f"其中正确判定良性 {tp} 例（TP）、正确判定恶性 {tn} 例（TN），仅有 {fp} 例恶性被误判为良性（FP，即危险的漏诊）、"
            f"{fn} 例良性被误判为恶性（FN）。",
            body,
        ),
        Spacer(1, 0.12 * cm),
        p("表2 随机森林在测试集上的混淆矩阵。", caption),
        build_table(cm_rows, font, [3.4 * cm, 3.4 * cm, 3.4 * cm, 2.4 * cm]),
        Spacer(1, 0.15 * cm),

        KeepTogether([
            Image(str(charts["confusion"]), width=9.0 * cm, height=7.75 * cm),
            p("图3 随机森林混淆矩阵热力图", caption),
        ]),
        p(
            "图3以热力图形式呈现混淆矩阵，对角线（TP、TN）颜色最深、数值最大，非对角线（FP、FN）数值很小，"
            "直观说明模型误判极少。在医疗场景中尤其要关注 FP（把恶性误判为良性）导致的漏诊风险，本模型该项仅为个位数。",
            body,
        ),

        KeepTogether([
            Image(str(charts["importance"]), width=13.5 * cm, height=8.59 * cm),
            p("图4 随机森林特征重要性排序（前 12 名）", caption),
        ]),
        p(
            "图4展示随机森林输出的特征重要性。可见“最差周长、最差半径、最差凹点、平均凹点”等描述肿瘤大小与形态不规则程度的"
            "特征贡献最大，与医学认知一致。特征重要性为模型提供了可解释性，在量化选股中同样可用来识别哪些财务或技术因子"
            "对收益预测最关键，从而指导因子筛选。",
            body,
        ),

        Spacer(1, 0.2 * cm),
        p("五、结论", heading),
        p(
            f"本任务完整走通了分类型机器学习的建模与评估流程：加载二分类数据、划分训练/测试集、训练逻辑回归、决策树与随机森林、"
            f"并用混淆矩阵、准确率/精确率/召回率/F1、ROC 与 AUC 多维度评估。结果显示三者在乳腺癌数据集上均表现优秀，"
            f"{best_name}以 AUC {best['auc']:.3f} 综合最优。这套“特征 → 二分类标签 → 训练 → ROC/AUC 评估”的方法论"
            f"可直接迁移到量化交易——将财务与技术指标作为特征、下一期涨跌作为标签，即可构建选股或择时的分类模型，为 TASK6 的"
            f"机器学习选股策略奠定基础。",
            body,
        ),
    ]

    doc.build(story)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def generate_assets(student_name: str, output: Path) -> dict[str, Path]:
    setup_matplotlib_font()
    X, y, feature_names, target_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    models = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    dataset_meta = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }

    charts = {
        "roc": HERE / "task5_roc.png",
        "metrics": HERE / "task5_metrics.png",
        "confusion": HERE / "task5_confusion.png",
        "importance": HERE / "task5_importance.png",
    }
    draw_roc_chart(results, "图1 三个分类模型的 ROC 曲线", charts["roc"])
    draw_metrics_chart(results, "图2 三个模型评价指标对比", charts["metrics"])
    draw_confusion_chart(
        results["随机森林"]["cm"], ["恶性(0)", "良性(1)"],
        "图3 随机森林混淆矩阵", charts["confusion"],
    )
    draw_importance_chart(
        results["随机森林"]["model"], feature_names,
        "图4 随机森林特征重要性（前 12 名）", charts["importance"],
    )

    # 导出测试集预测明细，供作品集/复核使用
    detail = pd.DataFrame({
        "true_label": y_test.values,
        "rf_pred": results["随机森林"]["y_pred"],
        "rf_score": results["随机森林"]["y_score"],
    })
    detail_csv = HERE / "task5_test_predictions.csv"
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(student_name, output, charts, results, dataset_meta, list(target_names))
    return {"pdf": output, "predictions_csv": detail_csv, **charts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-name", default="祁彦龙")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = (
        Path(args.output) if args.output
        else ROOT / "private_submissions" / "TASK5" / f"{args.student_name}TASK5.pdf"
    )
    assets = generate_assets(args.student_name, output)
    for name, path in assets.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
