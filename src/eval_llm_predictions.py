"""
从分版本的 LLM 预测 CSV 生成论文/作图用评测表。

输入（默认）: data/emotions/llm_predictions_{prompts.version}.csv
  （与 config.yaml 中 prompts.version 一致；若不存在可回退 llm_predictions_test.csv）
输出（默认）:
- results/prompt_runs/{version}/llm_eval_by_row.csv
- results/prompt_runs/{version}/llm_eval_summary.json

仅将「pred_valence / pred_arousal 可解析为有限数值」的行纳入回归指标与输出行，
避免散点图 astype(float) 失败；统计信息中记录跳过条数。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from emotion_label_mapping import map_valence_arousal_to_label


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_prompt_version_from_config(root: Path) -> str:
    for p in (root / "config.yaml", Path("config.yaml")):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            v = (cfg.get("prompts") or {}).get("version", "v1")
            return str(v).strip() or "v1"
    return "v1"


def _version_from_predictions_filename(path: Path) -> Optional[str]:
    m = re.match(r"^llm_predictions_(.+)\.csv$", path.name, re.I)
    return m.group(1) if m else None


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return v if np.isfinite(v) else None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _norm_pred_label(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run(
    *,
    predictions_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    prompt_version: Optional[str] = None,
) -> None:
    root = _project_root()
    cfg_ver = _load_prompt_version_from_config(root)
    ver = (prompt_version or cfg_ver).strip() or "v1"

    if predictions_path is not None:
        pred_path = Path(predictions_path).resolve()
        inferred = _version_from_predictions_filename(pred_path)
        if inferred is not None:
            ver = inferred
    else:
        pred_path = root / "data" / "emotions" / f"llm_predictions_{ver}.csv"
        if not pred_path.exists():
            leg = root / "data" / "emotions" / "llm_predictions_test.csv"
            if leg.exists():
                print(
                    f"[WARN] 未找到 {pred_path.name}，回退使用 {leg.name}。"
                    f" 建议重命名为 llm_predictions_{ver}.csv 与 prompts.version 一致。"
                )
                pred_path = leg
                ver = cfg_ver
            else:
                raise FileNotFoundError(
                    f"未找到 LLM 预测文件: {pred_path}，请先运行 llm_inference.py（将生成按版本命名文件）。"
                )

    if results_dir is not None:
        out_dir = Path(results_dir).resolve()
    else:
        out_dir = (root / "results" / "prompt_runs" / ver).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "llm_eval_by_row.csv"
    out_json = out_dir / "llm_eval_summary.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"未找到 LLM 预测文件: {pred_path}，请先运行 llm_inference.py")

    df = pd.read_csv(pred_path)
    required = {"segment_id", "gt_valence_mean", "gt_arousal_mean"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"预测文件缺少列: {missing}")

    n_input = len(df)
    rows_out = []
    skipped = 0

    for _, row in df.iterrows():
        gv = _try_float(row.get("gt_valence_mean"))
        ga = _try_float(row.get("gt_arousal_mean"))
        pv = _try_float(row.get("pred_valence"))
        pa = _try_float(row.get("pred_arousal"))

        if gv is None or ga is None or pv is None or pa is None:
            skipped += 1
            continue

        gt_lab = map_valence_arousal_to_label(gv, ga)
        pred_lab = _norm_pred_label(row.get("pred_label"))
        match = int(pred_lab == gt_lab) if pred_lab else 0

        def _col(name: str, default: Any = "") -> Any:
            return row[name] if name in row.index else default

        row_out = {
            "segment_id": int(row["segment_id"]),
            "song_id": int(_col("song_id", row["segment_id"])),
            "audio_path": _col("audio_path", ""),
            "description_raw": _col("description_raw", ""),
            "gt_valence_mean": gv,
            "gt_arousal_mean": ga,
            "pred_valence": pv,
            "pred_arousal": pa,
            "pred_label": pred_lab if pred_lab else np.nan,
            "raw_output": _col("raw_output", ""),
            "latency_ms": _col("latency_ms", ""),
            "llm_model": _col("llm_model", ""),
            "prompt_version": _col("prompt_version", ""),
            "gt_mapped_label": gt_lab,
            "label_match": match,
            "valence_abs_error": float(abs(gv - pv)),
            "arousal_abs_error": float(abs(ga - pa)),
        }
        pll = _col("pred_label_llm", "")
        if pll not in ("", None) and str(pll).strip().lower() not in ("nan", "none"):
            row_out["pred_label_llm"] = str(pll).strip()
        rows_out.append(row_out)

    if not rows_out:
        raise RuntimeError(
            f"没有可评测的有效样本（需同时有 GT 与 pred的数值 V/A）。"
            f"输入 {n_input} 行，跳过 {skipped} 行。"
        )

    out_df = pd.DataFrame(rows_out).sort_values(by="segment_id")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    yv = out_df["gt_valence_mean"].to_numpy(dtype=float)
    ypv = out_df["pred_valence"].to_numpy(dtype=float)
    ya = out_df["gt_arousal_mean"].to_numpy(dtype=float)
    ypa = out_df["pred_arousal"].to_numpy(dtype=float)

    # 标签准确率：仅统计 pred_label 非空的行，与旧 summary 语义一致
    pl = out_df["pred_label"].astype(str)
    nonempty = pl.notna() & (pl.str.strip() != "") & (pl.str.lower() != "nan")
    if nonempty.any():
        acc = float(np.mean(out_df.loc[nonempty, "label_match"].to_numpy(dtype=int)))
        n_label = int(nonempty.sum())
    else:
        acc = 0.0
        n_label = 0

    prompt_ver_meta = ""
    if "prompt_version" in df.columns and len(df) > 0:
        prompt_ver_meta = str(df["prompt_version"].dropna().iloc[0]) if df["prompt_version"].notna().any() else ""

    summary = {
        "n_input_rows": n_input,
        "n_eval": int(len(out_df)),
        "n_skipped_invalid_pred": skipped,
        "n_label_scored": n_label,
        "valence_mae": _mae(yv, ypv),
        "valence_rmse": _rmse(yv, ypv),
        "valence_pearson": _pearson(yv, ypv),
        "arousal_mae": _mae(ya, ypa),
        "arousal_rmse": _rmse(ya, ypa),
        "arousal_pearson": _pearson(ya, ypa),
        "label_accuracy": acc,
        "predictions_path": str(pred_path.as_posix()),
        "prompt_version": prompt_ver_meta,
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] 评测样本: {len(out_df)} / 输入 {n_input}（跳过无效预测 {skipped}）")
    print(f"[INFO] 已写入: {out_csv}")
    print(f"[INFO] 已写入: {out_json}")
    print(f"[INFO] 提示词版本目录: {out_dir}（作图: python src/plot_eval_results.py --llm-version {ver}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM 预测评测，输出写入 results/prompt_runs/{version}/")
    ap.add_argument(
        "--version",
        dest="prompt_version",
        default=None,
        help="提示词版本，如 v1/v2/v3；默认从 config.yaml 的 prompts.version 读取",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="覆盖预测 CSV 路径（若指定，输出目录仍由 --version 或文件名推断）",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="覆盖输出目录（默认 results/prompt_runs/{version}/）",
    )
    args = ap.parse_args()
    run(
        predictions_path=args.predictions,
        results_dir=args.results_dir,
        prompt_version=args.prompt_version,
    )
