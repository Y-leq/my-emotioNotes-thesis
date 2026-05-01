"""
评估结果可视化脚本

默认读取（与 config 中 prompts.version 一致）:
- results/prompt_runs/{version}/llm_eval_by_row.csv
- results/traditional_baseline_by_row.csv（基线与提示词版本无关，各版本图可共用）

默认输出:
- results/visualizations/prompt_runs/{version}/*.png

可用 --llm-by-row / --out-vis 覆盖。传统基线散点/混淆会写入同一 out-vis 目录，便于每版自洽打包。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM+基线评测可视化，输出按 prompts.version 分目录。")
    ap.add_argument("--config", type=Path, default=None, help="config.yaml 路径，默认项目根")
    ap.add_argument(
        "--llm-by-row",
        type=Path,
        default=None,
        help="覆盖：LLM 逐行评测表；默认 results/prompt_runs/{version}/llm_eval_by_row.csv",
    )
    ap.add_argument(
        "--out-vis",
        type=Path,
        default=None,
        help="覆盖：PNG 输出目录；默认 results/visualizations/prompt_runs/{version}/",
    )
    ap.add_argument(
        "--baseline-by-row",
        type=Path,
        default=None,
        help="传统基线逐行表，默认 results/traditional_baseline_by_row.csv",
    )
    ap.add_argument(
        "--version",
        type=str,
        default=None,
        metavar="V",
        help="如 v1：与 llm_inference/eval 同版，选择 prompt_runs 与作图子目录",
    )
    args = ap.parse_args()

    from prompt_artifact_paths import (
        apply_prompt_profile,
        llm_results_run_dir,
        llm_visualizations_run_dir,
    )

    root = Path(__file__).resolve().parent.parent
    cfg_path = args.config or (root / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.version:
        apply_prompt_profile(cfg, args.version.strip())

    results_dir = root / "results"
    if args.llm_by_row is not None:
        llm_by_row_path = args.llm_by_row
    else:
        llm_by_row_path = llm_results_run_dir(root, cfg) / "llm_eval_by_row.csv"
    if args.baseline_by_row is not None:
        base_by_row_path = args.baseline_by_row
    else:
        base_by_row_path = results_dir / "traditional_baseline_by_row.csv"
    if args.out_vis is not None:
        vis_dir = args.out_vis
    else:
        vis_dir = llm_visualizations_run_dir(root, cfg)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not llm_by_row_path.exists():
        raise FileNotFoundError(
            f"未找到: {llm_by_row_path}（请先对当前 prompts.version 运行 python src/eval_llm_predictions.py）"
        )
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

