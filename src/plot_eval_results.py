"""
评估结果可视化脚本

读取:
- results/llm_eval_by_row.csv
- results/traditional_baseline_by_row.csv

输出:
- results/visualizations/*.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _plot_scatter(ax, x, y, title, xlabel, ylabel):
    import matplotlib.pyplot as plt

    ax.scatter(x, y, s=40, alpha=0.85)
    # y=x 参考线（便于观察误差方向）
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6)


def _plot_confusion(ax, y_true, y_pred, labels, title):
    import matplotlib.pyplot as plt

    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    # 标注数字
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    vis_dir = results_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    llm_by_row_path = results_dir / "llm_eval_by_row.csv"
    base_by_row_path = results_dir / "traditional_baseline_by_row.csv"
    if not llm_by_row_path.exists():
        raise FileNotFoundError(f"未找到: {llm_by_row_path}")
    if not base_by_row_path.exists():
        raise FileNotFoundError(f"未找到: {base_by_row_path}")

    llm_df = pd.read_csv(llm_by_row_path)
    base_df = pd.read_csv(base_by_row_path)

    # ========== 散点图 ==========
    import matplotlib.pyplot as plt

    # 兼容中文显示（Windows 常见字体）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    _plot_scatter(
        ax1,
        llm_df["gt_valence_mean"].astype(float),
        llm_df["pred_valence"].astype(float),
        "LLM: Valence GT vs Pred",
        "GT Valence",
        "Pred Valence",
    )
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_scatter(
        ax2,
        llm_df["gt_arousal_mean"].astype(float),
        llm_df["pred_arousal"].astype(float),
        "LLM: Arousal GT vs Pred",
        "GT Arousal",
        "Pred Arousal",
    )
    fig.tight_layout()
    fig.savefig(vis_dir / "llm_scatter_valence_arousal.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    _plot_scatter(
        ax1,
        base_df["gt_valence_mean"].astype(float),
        base_df["pred_valence"].astype(float),
        "Traditional: Valence GT vs Pred",
        "GT Valence",
        "Pred Valence",
    )
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_scatter(
        ax2,
        base_df["gt_arousal_mean"].astype(float),
        base_df["pred_arousal"].astype(float),
        "Traditional: Arousal GT vs Pred",
        "GT Arousal",
        "Pred Arousal",
    )
    fig.tight_layout()
    fig.savefig(vis_dir / "traditional_scatter_valence_arousal.png", dpi=160)
    plt.close(fig)

    # ========== 混淆矩阵 ==========
    llm_labels = sorted(set(llm_df["gt_mapped_label"].astype(str).tolist()))
    # baseline 混淆矩阵用分类器输出 pred_label_from_clf
    base_labels = sorted(set(base_df["gt_label_mapped"].astype(str).tolist()))
    labels = sorted(set(llm_labels + base_labels))
    if not labels:
        labels = ["其他"]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    _plot_confusion(
        ax1,
        llm_df["gt_mapped_label"].astype(str).tolist(),
        llm_df["pred_label"].astype(str).tolist(),
        labels,
        "LLM: Confusion Matrix (pred_label)",
    )
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_confusion(
        ax2,
        base_df["gt_label_mapped"].astype(str).tolist(),
        base_df["pred_label_from_clf"].astype(str).tolist(),
        labels,
        "Traditional: Confusion Matrix (pred_label_from_clf)",
    )
    fig.tight_layout()
    fig.savefig(vis_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    print(f"[INFO] 图表已生成到: {vis_dir}")


if __name__ == "__main__":
    main()

