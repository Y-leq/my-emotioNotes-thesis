from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotion_label_mapping import map_valence_arousal_to_label


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def safe_pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    val = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(val):
        return 0.0
    return float(val)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_segment_feature_vector(
    feature_path: Path,
    *,
    expected_cols: Optional[List[str]] = None,
    sep: str = ";",
) -> Tuple[pd.Series, List[str]]:
    """
    将单个 segment 的逐帧特征 csv 聚合为片段级统计特征向量。
    使用 mean/std/min/max/median 五种统计量。
    """
    df = pd.read_csv(feature_path, sep=sep)
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        raise ValueError(f"文件中没有数值特征列: {feature_path}")

    feats: Dict[str, float] = {}

    for col in num_df.columns:
        arr = num_df[col].to_numpy(dtype=float)
        if arr.size == 0:
            continue

        feats[f"{col}__mean"] = float(np.mean(arr))
        feats[f"{col}__std"] = float(np.std(arr))
        feats[f"{col}__min"] = float(np.min(arr))
        feats[f"{col}__max"] = float(np.max(arr))
        feats[f"{col}__median"] = float(np.median(arr))

    series = pd.Series(feats, dtype=float)
    cols = series.index.tolist()

    if expected_cols is None:
        return series, cols

    aligned = series.reindex(expected_cols).fillna(0.0)
    return aligned, expected_cols


def load_features_for_segment_ids(
    segment_ids: List[int],
    feature_dir: Path,
    *,
    expected_cols: Optional[List[str]] = None,
    sep: str = ";",
) -> Tuple[np.ndarray, List[int], List[str], List[int]]:
    """
    返回:
    - X: 特征矩阵
    - used_ids: 成功读取特征的 segment_id
    - cols: 特征列名顺序
    - missing_ids: 缺失特征文件的 segment_id
    """
    used_ids: List[int] = []
    missing_ids: List[int] = []
    vectors: List[pd.Series] = []
    cols: Optional[List[str]] = expected_cols

    for sid in segment_ids:
        fpath = feature_dir / f"{sid}.csv"
        if not fpath.exists():
            missing_ids.append(int(sid))
            continue

        try:
            if cols is None:
                feat_series, cols = build_segment_feature_vector(
                    fpath,
                    expected_cols=None,
                    sep=sep,
                )
            else:
                feat_series, _ = build_segment_feature_vector(
                    fpath,
                    expected_cols=cols,
                    sep=sep,
                )
        except Exception as e:
            print(f"[WARN] 读取特征失败 sid={sid}: {e}")
            missing_ids.append(int(sid))
            continue

        used_ids.append(int(sid))
        vectors.append(feat_series)

    if cols is None or len(vectors) == 0:
        raise RuntimeError(f"没有成功读取到任何特征文件: {feature_dir}")

    X = np.vstack([v.to_numpy(dtype=float) for v in vectors])
    return X, used_ids, cols, missing_ids


def make_regressor(name: str, random_seed: int):
    name = name.lower()
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0))
        ])
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"不支持的 regressor: {name}")


def make_classifier(name: str, random_seed: int):
    name = name.lower()
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_seed
            ))
        ])
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"不支持的 classifier: {name}")


def main(
    *,
    feature_dir: Optional[str] = None,
    eval_desc_file: Optional[str] = None,
    n_train_used: Optional[int] = None,   # 默认全量
    random_seed: int = 42,
    regressor_name: str = "ridge",        # ridge / rf
    classifier_name: str = "logreg",      # logreg / rf
    feature_sep: str = ";",
) -> None:
    root = _project_root()
    feature_dir_path = Path(feature_dir) if feature_dir is not None else (root / "features" / "features")

    # ===== 读取评估集 =====
    if eval_desc_file is not None:
        eval_desc_path = Path(eval_desc_file)
    else:
        desc_real = root / "data" / "descriptions" / "test_descriptions_real.csv"
        desc_dummy = root / "data" / "descriptions" / "test_descriptions_dummy.csv"
        if desc_real.exists():
            eval_desc_path = desc_real
        elif desc_dummy.exists():
            eval_desc_path = desc_dummy
        else:
            raise FileNotFoundError(f"未找到评估描述文件: {desc_real} 或 {desc_dummy}")

    eval_df = pd.read_csv(eval_desc_path)

    required_eval_cols = {"segment_id", "valence_mean", "arousal_mean"}
    missing_eval = required_eval_cols - set(eval_df.columns)
    if missing_eval:
        raise RuntimeError(f"评估文件缺少必要列: {missing_eval}")

    eval_df = eval_df.copy()
    eval_df["segment_id"] = eval_df["segment_id"].astype(int)

    eval_segment_ids = eval_df["segment_id"].tolist()
    eval_y_valence_map = eval_df.set_index("segment_id")["valence_mean"].to_dict()
    eval_y_arousal_map = eval_df.set_index("segment_id")["arousal_mean"].to_dict()

    eval_gt_labels_map = {
        int(sid): map_valence_arousal_to_label(
            float(eval_y_valence_map[sid]),
            float(eval_y_arousal_map[sid]),
        )
        for sid in eval_segment_ids
    }

    # ===== 读取训练集 =====
    train_path = root / "data" / "splits" / "train_segments.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"未找到训练集划分文件: {train_path}")

    train_df = pd.read_csv(train_path)
    required_train_cols = {"segment_id", "valence_mean", "arousal_mean"}
    missing_train = required_train_cols - set(train_df.columns)
    if missing_train:
        raise RuntimeError(f"train_segments.csv 缺少必要列: {missing_train}")

    train_df = train_df.copy()
    train_df["segment_id"] = train_df["segment_id"].astype(int)

    # ===== 无泄漏检查 =====
    train_ids_all = set(train_df["segment_id"].tolist())
    eval_ids_all = set(eval_segment_ids)
    overlap = train_ids_all & eval_ids_all
    if len(overlap) > 0:
        raise RuntimeError(
            f"发现训练集与评估集 segment_id 重叠，存在数据泄漏风险，重叠数量={len(overlap)}，部分样例={list(sorted(overlap))[:10]}"
        )

    train_segment_ids_all = train_df["segment_id"].tolist()
    train_y_valence_map = train_df.set_index("segment_id")["valence_mean"].to_dict()
    train_y_arousal_map = train_df.set_index("segment_id")["arousal_mean"].to_dict()
    train_y_labels_map = {
        int(sid): map_valence_arousal_to_label(
            float(train_y_valence_map[sid]),
            float(train_y_arousal_map[sid]),
        )
        for sid in train_segment_ids_all
    }

    # ===== 可选抽样 =====
    rng = np.random.default_rng(random_seed)
    if n_train_used is not None and len(train_segment_ids_all) > n_train_used:
        train_segment_ids_all = rng.choice(
            train_segment_ids_all,
            size=n_train_used,
            replace=False
        ).astype(int).tolist()

    # ===== 读取训练特征 =====
    X_train, used_train_ids, feat_cols, train_missing_ids = load_features_for_segment_ids(
        train_segment_ids_all,
        feature_dir_path,
        expected_cols=None,
        sep=feature_sep,
    )

    # ===== 读取评估特征（列对齐到训练特征） =====
    X_eval, used_eval_ids, _, eval_missing_ids = load_features_for_segment_ids(
        eval_segment_ids,
        feature_dir_path,
        expected_cols=feat_cols,
        sep=feature_sep,
    )

    if len(used_eval_ids) == 0:
        raise RuntimeError("评估集没有任何可用特征样本，无法评估。")

    # ===== 构建 y =====
    y_train_valence = np.array([train_y_valence_map[sid] for sid in used_train_ids], dtype=float)
    y_train_arousal = np.array([train_y_arousal_map[sid] for sid in used_train_ids], dtype=float)
    y_train_label = [train_y_labels_map[sid] for sid in used_train_ids]

    y_eval_valence = np.array([eval_y_valence_map[sid] for sid in used_eval_ids], dtype=float)
    y_eval_arousal = np.array([eval_y_arousal_map[sid] for sid in used_eval_ids], dtype=float)
    y_eval_label = [eval_gt_labels_map[sid] for sid in used_eval_ids]

    # ===== 模型训练 =====
    valence_reg = make_regressor(regressor_name, random_seed)
    arousal_reg = make_regressor(regressor_name, random_seed)
    clf = make_classifier(classifier_name, random_seed)

    valence_reg.fit(X_train, y_train_valence)
    arousal_reg.fit(X_train, y_train_arousal)
    clf.fit(X_train, y_train_label)

    # ===== 预测 =====
    pred_valence = np.asarray(valence_reg.predict(X_eval), dtype=float)
    pred_arousal = np.asarray(arousal_reg.predict(X_eval), dtype=float)

    # clip 到 DEAM 合法范围 [1, 9]
    pred_valence = np.clip(pred_valence, 1.0, 9.0)
    pred_arousal = np.clip(pred_arousal, 1.0, 9.0)

    pred_labels_from_reg = [
        map_valence_arousal_to_label(v, a)
        for v, a in zip(pred_valence, pred_arousal)
    ]
    pred_labels_from_clf = clf.predict(X_eval).astype(str).tolist()

    # ===== 指标 =====
    valence_mae = mae(y_eval_valence, pred_valence)
    valence_rmse = rmse(y_eval_valence, pred_valence)
    valence_pearson = safe_pearson(y_eval_valence, pred_valence)

    arousal_mae = mae(y_eval_arousal, pred_arousal)
    arousal_rmse = rmse(y_eval_arousal, pred_arousal)
    arousal_pearson = safe_pearson(y_eval_arousal, pred_arousal)

    label_acc_reg = float(np.mean(
        np.array(pred_labels_from_reg, dtype=str) == np.array(y_eval_label, dtype=str)
    ))
    label_acc_clf = float(np.mean(
        np.array(pred_labels_from_clf, dtype=str) == np.array(y_eval_label, dtype=str)
    ))

    # ===== 类别分布诊断 =====
    train_label_dist = pd.Series(y_train_label).value_counts().to_dict()
    eval_label_dist = pd.Series(y_eval_label).value_counts().to_dict()

    # ===== 输出 =====
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_train_requested": None if n_train_used is None else int(n_train_used),
        "n_train_used": int(len(used_train_ids)),
        "n_eval_requested": int(len(eval_segment_ids)),
        "n_eval_used": int(len(used_eval_ids)),
        "n_train_missing_features": int(len(train_missing_ids)),
        "n_eval_missing_features": int(len(eval_missing_ids)),
        "feature_dim": int(X_train.shape[1]),
        "feature_dir": str(feature_dir_path),
        "eval_desc_file": str(eval_desc_path),
        "regressor_name": regressor_name,
        "classifier_name": classifier_name,
        "random_seed": int(random_seed),
        "valence_mae": valence_mae,
        "valence_rmse": valence_rmse,
        "valence_pearson": valence_pearson,
        "arousal_mae": arousal_mae,
        "arousal_rmse": arousal_rmse,
        "arousal_pearson": arousal_pearson,
        "label_accuracy_from_reg": label_acc_reg,
        "label_accuracy_from_clf": label_acc_clf,
        "train_label_distribution": train_label_dist,
        "eval_label_distribution": eval_label_dist,
    }

    summary_path = results_dir / "traditional_baseline_eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for sid, gv, ga, pv, pa, pl_reg, pl_clf in zip(
        used_eval_ids,
        y_eval_valence,
        y_eval_arousal,
        pred_valence,
        pred_arousal,
        pred_labels_from_reg,
        pred_labels_from_clf,
    ):
        gt_label = eval_gt_labels_map[int(sid)]
        rows.append(
            {
                "segment_id": int(sid),
                "gt_valence_mean": float(gv),
                "gt_arousal_mean": float(ga),
                "pred_valence": float(pv),
                "pred_arousal": float(pa),
                "gt_label_mapped": gt_label,
                "pred_label_from_reg": str(pl_reg),
                "pred_label_from_clf": str(pl_clf),
                "valence_abs_error": float(abs(gv - pv)),
                "arousal_abs_error": float(abs(ga - pa)),
                "label_match_reg": int(str(pl_reg) == gt_label),
                "label_match_clf": int(str(pl_clf) == gt_label),
            }
        )

    by_row_df = pd.DataFrame(rows).sort_values(by="segment_id")
    by_row_path = results_dir / "traditional_baseline_by_row.csv"
    by_row_df.to_csv(by_row_path, index=False, encoding="utf-8-sig")

    # 额外导出本次真实参与评估的 segment_id，便于和你的 pipeline 对齐
    used_eval_ids_path = results_dir / "traditional_baseline_used_eval_ids.csv"
    pd.DataFrame({"segment_id": used_eval_ids}).to_csv(
        used_eval_ids_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("[INFO] 传统声学特征基线评估完成")
    print(f"[INFO] summary: {summary_path}")
    print(f"[INFO] by_row:   {by_row_path}")
    print(f"[INFO] used_ids: {used_eval_ids_path}")
    print(f"[INFO] n_train_used={len(used_train_ids)}, n_eval_used={len(used_eval_ids)}, feature_dim={X_train.shape[1]}")
    print(f"[INFO] valence: MAE={valence_mae:.4f}, RMSE={valence_rmse:.4f}, Pearson={valence_pearson:.4f}")
    print(f"[INFO] arousal: MAE={arousal_mae:.4f}, RMSE={arousal_rmse:.4f}, Pearson={arousal_pearson:.4f}")
    print(f"[INFO] label acc (from reg)={label_acc_reg:.4f}, label acc (from clf)={label_acc_clf:.4f}")


if __name__ == "__main__":
    main(
        n_train_used=None,        # 建议先全量训练
        random_seed=42,
        regressor_name="ridge",   # 可改为 "rf"
        classifier_name="logreg", # 可改为 "rf"
    )