"""
评估结果可视化脚本

读取（LLM 侧，按提示词版本）:
- results/prompt_runs/{llm_version}/llm_eval_by_row.csv
  （若不存在，可回退 results/llm_eval_by_row.csv 以兼容旧目录）

基线:
- results/traditional_baseline_by_row.csv

输出:
- results/prompt_runs/{llm_version}/visualizations/*.png
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_llm_version_from_config(root: Path) -> str:
    for p in (root / "config.yaml", Path("config.yaml")):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            v = (cfg.get("prompts") or {}).get("version", "v1")
            return str(v).strip() or "v1"
    return "v1"

# DEAM 连续标注常用 1～9 量表；横纵统一便于 LLM 与传统基线散点图对比
# 若出现越界点（如 Ridge 无界预测），仍绘制在图内，坐标轴不随单点被拉到 0～14
DEAM_VALENCE_AROUSAL_AXIS = (1.0, 9.0)


def _plot_scatter(ax, x, y, title, xlabel, ylabel, *, axis_lim=DEAM_VALENCE_AROUSAL_AXIS):
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = float(axis_lim[0]), float(axis_lim[1])
    oob = (x < lo) | (x > hi) | (y < lo) | (y > hi)
    if np.any(oob):
        n_oob = int(np.sum(oob))
        print(
            f"[WARN] {title}: 有 {n_oob} 个点超出坐标范围 [{lo},{hi}]"
            f"（多为无界回归预测），图中不显示在框外；完整数值见 *_by_row.csv"
        )
    ax.scatter(x, y, s=40, alpha=0.85, clip_on=False)
    # y=x 与坐标范围一致（与 DEAM 量表一致，避免“传统/LLM 两图横纵单位看起来不同”）
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
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


def main(llm_version: Optional[str] = None) -> None:
    root = _project_root()
    results_dir = root / "results"
    ver = (llm_version or _load_llm_version_from_config(root)).strip() or "v1"

    llm_by_row_path = results_dir / "prompt_runs" / ver / "llm_eval_by_row.csv"
    if not llm_by_row_path.exists():
        legacy = results_dir / "llm_eval_by_row.csv"
        if legacy.exists():
            print(
                f"[WARN] 未找到分版本 LLM 评测文件，回退使用 {legacy.name}。"
                f" 建议先运行: python src/eval_llm_predictions.py（将写入 results/prompt_runs/{ver}/）"
            )
            llm_by_row_path = legacy
        else:
            raise FileNotFoundError(
                f"未找到 LLM 逐行评测: {llm_by_row_path}。请先运行: python src/eval_llm_predictions.py"
            )

    vis_dir = results_dir / "prompt_runs" / ver / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    base_by_row_path = results_dir / "traditional_baseline_by_row.csv"
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

    print(f"[INFO] 提示词版本: {ver}，图表已生成到: {vis_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM+基线评测可视化，LLM 图按 results/prompt_runs/{version}/ 分目录输出")
    ap.add_argument(
        "--llm-version",
        default=None,
        help="与 config.yaml 中 prompts.version 及 eval 输出一致，如 v1/v2/v3",
    )
    args = ap.parse_args()
    main(llm_version=args.llm_version)

