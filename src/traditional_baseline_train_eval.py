"""
传统声学特征基线训练与评估

目的：
- 读取 features/features/{segment_id}.csv 作为传统声学特征；
- 读取 data/descriptions/test_descriptions_real.csv（优先）或 dummy 作为评估集（segment_id + gt valence/arousal）；
- 使用训练集（data/splits/train_segments.csv）训练：
  - 回归器：预测 valence_mean / arousal_mean
  - 分类器：预测离散情感类别（快乐/悲伤/平静/激昂/紧张/放松/其他）
- 输出：
  - results/traditional_baseline_eval_summary.json
  - results/traditional_baseline_by_row.csv

与现有绘图脚本 results/plot_eval_results.py 的字段保持一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def map_valence_arousal_to_label(valence: float, arousal: float) -> str:
    """
    将连续 Valence/Arousal 映射到离散类别。
    采用固定阈值（与当前你已有结果的风格一致，便于论文解释）。
    """
    val_low = 4.5
    val_high = 6.0
    ar_low = 4.5
    ar_high = 6.0

    is_val_low = valence <= val_low
    is_val_high = valence >= val_high
    is_ar_low = arousal <= ar_low
    is_ar_high = arousal >= ar_high

    # 高唤醒（arousal high）
    if is_ar_high:
        if is_val_high:
            return "快乐"
        if is_val_low:
            return "紧张"
        return "激昂"

    # 低唤醒（arousal low）
    if is_ar_low:
        if is_val_high:
            return "放松"
        if is_val_low:
            return "悲伤"
        return "平静"

    # 中唤醒（arousal mid）
    if is_val_high:
        return "放松"
    if is_val_low:
        return "紧张"
    return "平静"


def _safe_mean_series_from_feature_csv(
    feature_path: Path, *, expected_cols: Optional[List[str]] = None
) -> Tuple[pd.Series, List[str]]:
    """
    从 features/{id}.csv 读取特征，并对“时间维度行”取均值得到一个片段级向量。
    """
    df = pd.read_csv(feature_path, sep=";")

    # 取数值列并求均值；若存在非数值列，自动剔除
    num_df = df.select_dtypes(include=[np.number])
    mean_series = num_df.mean(axis=0)

    cols = mean_series.index.tolist()
    if expected_cols is None:
        return mean_series, cols

    # 对齐列：缺失置 0，多余忽略
    aligned = mean_series.reindex(expected_cols).fillna(0.0)
    return aligned, expected_cols


def load_features_for_segment_ids(
    segment_ids: List[int],
    feature_dir: Path,
    *,
    expected_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    返回：
    - X: (n, d)
    - used_ids: 实际读取到特征的 ids
    - cols: 特征列顺序（长度 d）
    """
    used_ids: List[int] = []
    vectors: List[pd.Series] = []
    cols: Optional[List[str]] = expected_cols

    for sid in segment_ids:
        fpath = feature_dir / f"{sid}.csv"
        if not fpath.exists():
            continue

        if cols is None:
            mean_series, cols = _safe_mean_series_from_feature_csv(fpath, expected_cols=None)
        else:
            mean_series, _ = _safe_mean_series_from_feature_csv(fpath, expected_cols=cols)

        used_ids.append(int(sid))
        vectors.append(mean_series)

    if cols is None:
        raise RuntimeError(f"没有成功读取到任何特征文件：feature_dir={feature_dir}")

    X = np.vstack([v.to_numpy(dtype=float) for v in vectors])
    return X, used_ids, cols


def main(
    *,
    feature_dir: Optional[str] = None,
    n_train_used: int = 300,
    random_seed: int = 42,
) -> None:
    root = _project_root()
    feature_dir_path = Path(feature_dir) if feature_dir is not None else (root / "features" / "features")

    # 评估集：尽量与 llm_inference 使用的描述文件一致（small samples）
    desc_real = root / "data" / "descriptions" / "test_descriptions_real.csv"
    desc_dummy = root / "data" / "descriptions" / "test_descriptions_dummy.csv"
    if desc_real.exists():
        eval_desc_path = desc_real
    elif desc_dummy.exists():
        eval_desc_path = desc_dummy
    else:
        raise FileNotFoundError(f"未找到评估描述文件：{desc_real} 或 {desc_dummy}")

    eval_df = pd.read_csv(eval_desc_path)
    if not {"segment_id", "gt_valence_mean", "gt_arousal_mean", "valence_mean", "arousal_mean"}.intersection(eval_df.columns):
        # 你的描述文件是 valence_mean/arousal_mean
        pass

    required_cols = {"segment_id", "valence_mean", "arousal_mean"}
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise RuntimeError(f"评估描述文件缺少必要列：{missing}")

    eval_segment_ids = eval_df["segment_id"].astype(int).tolist()
    eval_y_valence = eval_df.set_index("segment_id")["valence_mean"].to_dict()
    eval_y_arousal = eval_df.set_index("segment_id")["arousal_mean"].to_dict()
    eval_gt_labels = {int(sid): map_valence_arousal_to_label(float(eval_y_valence[sid]), float(eval_y_arousal[sid])) for sid in eval_segment_ids if sid in eval_y_valence and sid in eval_y_arousal}

    # 训练集标签：来自 train_segments.csv
    train_path = root / "data" / "splits" / "train_segments.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"未找到训练集划分文件：{train_path}")
    train_df = pd.read_csv(train_path)
    if not {"segment_id", "valence_mean", "arousal_mean"}.issubset(set(train_df.columns)):
        raise RuntimeError("train_segments.csv 缺少必要列 segment_id/valence_mean/arousal_mean")

    train_segment_ids_all = train_df["segment_id"].astype(int).tolist()
    train_y_valence_map = train_df.set_index("segment_id")["valence_mean"].to_dict()
    train_y_arousal_map = train_df.set_index("segment_id")["arousal_mean"].to_dict()
    train_y_labels_map = {
        int(sid): map_valence_arousal_to_label(float(train_y_valence_map[sid]), float(train_y_arousal_map[sid]))
        for sid in train_segment_ids_all
        if sid in train_y_valence_map and sid in train_y_arousal_map
    }

    rng = np.random.default_rng(random_seed)
    if n_train_used is not None and len(train_segment_ids_all) > n_train_used:
        train_segment_ids_all = rng.choice(train_segment_ids_all, size=n_train_used, replace=False).astype(int).tolist()

    # ========= 读取特征并构建 X =========
    # 为了保证列一致：先读取训练特征决定 expected_cols，再读取评估特征对齐
    X_train, used_train_ids, cols = load_features_for_segment_ids(
        train_segment_ids_all, feature_dir_path, expected_cols=None
    )

    X_eval, used_eval_ids, _cols2 = load_features_for_segment_ids(
        eval_segment_ids, feature_dir_path, expected_cols=cols
    )

    # 如果某些 eval 片段缺失特征，则丢弃
    used_eval_y_valence = np.array([eval_y_valence[sid] for sid in used_eval_ids], dtype=float)
    used_eval_y_arousal = np.array([eval_y_arousal[sid] for sid in used_eval_ids], dtype=float)
    used_eval_y_labels = [eval_gt_labels[sid] for sid in used_eval_ids]

    # ========= 模型训练 =========
    # 回归：valence / arousal
    valence_reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=random_seed))])
    arousal_reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=random_seed))])
    valence_reg.fit(X_train, np.array([train_y_valence_map[sid] for sid in used_train_ids], dtype=float))
    arousal_reg.fit(X_train, np.array([train_y_arousal_map[sid] for sid in used_train_ids], dtype=float))

    # 分类：离散情感标签
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="auto", random_state=random_seed)),
        ]
    )
    clf.fit(X_train, [train_y_labels_map[sid] for sid in used_train_ids])

    # ========= 预测 =========
    pred_valence = valence_reg.predict(X_eval).astype(float)
    pred_arousal = arousal_reg.predict(X_eval).astype(float)
    pred_labels_from_reg = [map_valence_arousal_to_label(v, a) for v, a in zip(pred_valence, pred_arousal)]
    pred_labels_from_clf = clf.predict(X_eval).astype(str).tolist()

    # ========= 指标计算 =========
    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def pearson(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) < 2:
            return 0.0
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    valence_mae = mae(used_eval_y_valence, pred_valence)
    valence_rmse = rmse(used_eval_y_valence, pred_valence)
    valence_pearson = pearson(used_eval_y_valence, pred_valence)

    arousal_mae = mae(used_eval_y_arousal, pred_arousal)
    arousal_rmse = rmse(used_eval_y_arousal, pred_arousal)
    arousal_pearson = pearson(used_eval_y_arousal, pred_arousal)

    label_acc_reg = float(np.mean(np.array(pred_labels_from_reg, dtype=str) == np.array(used_eval_y_labels, dtype=str)))
    label_acc_clf = float(np.mean(np.array(pred_labels_from_clf, dtype=str) == np.array(used_eval_y_labels, dtype=str)))

    # ========= 写结果 =========
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_eval": int(len(used_eval_ids)),
        "valence_mae": valence_mae,
        "valence_rmse": valence_rmse,
        "valence_pearson": valence_pearson,
        "arousal_mae": arousal_mae,
        "arousal_rmse": arousal_rmse,
        "arousal_pearson": arousal_pearson,
        "label_accuracy_from_reg": label_acc_reg,
        "label_accuracy_from_clf": label_acc_clf,
        "n_train_used": int(len(used_train_ids)),
        "feature_dir": str(feature_dir_path),
    }

    (results_dir / "traditional_baseline_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for sid, gv, ga, pv, pa, grl, pcl in zip(
        used_eval_ids,
        used_eval_y_valence,
        used_eval_y_arousal,
        pred_valence,
        pred_arousal,
        pred_labels_from_reg,
        pred_labels_from_clf,
    ):
        gt_label_mapped = eval_gt_labels[int(sid)]
        rows.append(
            {
                "segment_id": int(sid),
                "gt_valence_mean": float(gv),
                "gt_arousal_mean": float(ga),
                "pred_valence": float(pv),
                "pred_arousal": float(pa),
                "gt_label_mapped": gt_label_mapped,
                "pred_label_from_reg": grl,
                "pred_label_from_clf": str(pcl),
                "valence_abs_error": float(abs(gv - pv)),
                "arousal_abs_error": float(abs(ga - pa)),
                "label_match_reg": int(grl == gt_label_mapped),
                "label_match_clf": int(str(pcl) == gt_label_mapped),
            }
        )

    by_row_df = pd.DataFrame(rows).sort_values(by="segment_id")
    by_row_df.to_csv(results_dir / "traditional_baseline_by_row.csv", index=False, encoding="utf-8-sig")

    print("[INFO] 传统基线评估完成：")
    print(f"  - {results_dir / 'traditional_baseline_eval_summary.json'}")
    print(f"  - {results_dir / 'traditional_baseline_by_row.csv'}")


if __name__ == "__main__":
    # 默认参数：与你历史 results 的风格一致
    main(n_train_used=300, random_seed=42)

