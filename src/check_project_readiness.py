"""
毕设流水线就绪检查：数据划分、描述、LLM 预测、声学特征文件是否对齐。

无需 API。在项目根目录执行: python src/check_project_readiness.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd
import yaml


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    p = _root() / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_ids(feature_dir: Path) -> Set[int]:
    if not feature_dir.is_dir():
        return set()
    ids: Set[int] = set()
    for f in feature_dir.glob("*.csv"):
        try:
            ids.add(int(f.stem))
        except ValueError:
            pass
    return ids


def _rel(root: Path, rel_path: str) -> Path:
    return root / str(rel_path).lstrip("./").replace("\\", "/")


def main() -> None:
    root = _root()
    cfg = _load_config()
    out = cfg.get("output", {})
    splits_dir = _rel(root, str(out.get("splits_dir", "./data/splits")))
    desc_dir = _rel(root, str(out.get("descriptions_dir", "./data/descriptions")))
    emotions_dir = _rel(root, str(out.get("emotions_dir", "./data/emotions")))
    dataset_cfg = cfg.get("dataset", {})
    feature_root = _rel(root, str(dataset_cfg.get("deam_features_dir", "./features")))
    # 与 traditional_baseline_train_eval 默认一致: features/features/{id}.csv
    feature_nested = feature_root / "features"

    lines = []
    lines.append(f"项目根目录: {root}")

    test_csv = splits_dir / "test_segments.csv"
    train_csv = splits_dir / "train_segments.csv"
    meta_csv = splits_dir / "segments_meta.csv"

    for label, path in [
        ("segments_meta", meta_csv),
        ("train_segments", train_csv),
        ("test_segments", test_csv),
    ]:
        if path.exists():
            n = len(pd.read_csv(path))
            lines.append(f"[OK] {label}: {path.name} ({n} 行)")
        else:
            lines.append(f"[缺] {label}: 未找到 {path}")

    desc_real = desc_dir / "test_descriptions_real.csv"
    desc_dummy = desc_dir / "test_descriptions_dummy.csv"
    if desc_real.exists():
        n = len(pd.read_csv(desc_real))
        lines.append(f"[OK] 描述: test_descriptions_real.csv ({n} 行)")
    elif desc_dummy.exists():
        n = len(pd.read_csv(desc_dummy))
        lines.append(f"[OK] 描述: test_descriptions_dummy.csv ({n} 行)")
    else:
        lines.append("[缺] 描述: test_descriptions_real.csv / dummy 均不存在")

    pver = str((cfg.get("prompts") or {}).get("version", "v1")).strip() or "v1"
    pred = emotions_dir / f"llm_predictions_{pver}.csv"
    pred_legacy = emotions_dir / "llm_predictions_test.csv"
    if pred.exists():
        n = len(pd.read_csv(pred))
        lines.append(f"[OK] LLM 预测: {pred.name}（prompts.version={pver}，{n} 行）")
    elif pred_legacy.exists():
        n = len(pd.read_csv(pred_legacy))
        lines.append(f"[OK] LLM 预测: {pred_legacy.name}（旧名，建议改为 {pred.name}）({n} 行)")
    else:
        lines.append(f"[缺] LLM 预测: {pred.name}（需运行 llm_inference.py）")

    feat_dir = feature_nested if feature_nested.is_dir() else feature_root
    feat_ids = _feature_ids(feat_dir)
    lines.append(f"[INFO] 特征目录: {feat_dir}（共 {len(feat_ids)} 个 *.csv）")

    if test_csv.exists() and feat_ids:
        tdf = pd.read_csv(test_csv)
        if "segment_id" in tdf.columns:
            test_ids = set(tdf["segment_id"].astype(int).tolist())
            missing = sorted(test_ids - feat_ids)
            lines.append(
                f"[INFO] 测试集 segment_id: {len(test_ids)}；缺失特征文件: {len(missing)}"
            )
            if missing and len(missing) <= 15:
                lines.append(f"       缺失 id 示例: {missing}")
            elif missing:
                lines.append(f"       缺失 id 示例（前 10）: {missing[:10]}")

    llm_eval = root / "results" / "prompt_runs" / pver / "llm_eval_by_row.csv"
    llm_eval_legacy = root / "results" / "llm_eval_by_row.csv"
    if llm_eval.exists():
        lines.append(f"[OK] LLM 评测: results/prompt_runs/{pver}/llm_eval_by_row.csv")
    elif llm_eval_legacy.exists():
        lines.append("[OK] results/llm_eval_by_row.csv（旧位置，建议运行 eval 生成分版本目录）")
    else:
        lines.append(
            f"[缺] results/prompt_runs/{pver}/llm_eval_by_row.csv（运行: python src/eval_llm_predictions.py）"
        )
    base_eval = root / "results" / "traditional_baseline_by_row.csv"
    lines.append(
        f"{'[OK]' if base_eval.exists() else '[缺]'} results/traditional_baseline_by_row.csv"
    )

    text = "\n".join(lines)
    print(text)


if __name__ == "__main__":
    main()
