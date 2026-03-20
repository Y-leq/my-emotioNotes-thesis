"""
数据集划分脚本：按 7:2:1 分层抽样划分 train/val/test

- 基于 valence_mean、arousal_mean 分层，保证各子集情感分布一致
"""
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_config(config_path=None) -> dict:
    if config_path is None:
        for p in [Path("config.yaml"), Path(__file__).parent.parent / "config.yaml"]:
            if p.exists():
                config_path = str(p)
                break
        config_path = config_path or "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_stratify_label(df: pd.DataFrame) -> pd.Series:
    """将 valence/arousal 分箱得到分层标签"""
    v = pd.qcut(df["valence_mean"], q=5, labels=False, duplicates="drop")
    a = pd.qcut(df["arousal_mean"], q=5, labels=False, duplicates="drop")
    return v.astype(str) + "_" + a.astype(str)


def run(config_path=None):
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    split_cfg = config["splitting"]
    output_cfg = config["output"]
    splits_dir = project_root / output_cfg["splits_dir"].lstrip("./")

    meta_path = splits_dir / "segments_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"请先运行 preprocess_audio.py 生成 {meta_path}")

    meta = pd.read_csv(meta_path)
    meta["stratify_label"] = make_stratify_label(meta)

    train_ratio = split_cfg.get("train_ratio", 0.7)
    val_ratio = split_cfg.get("val_ratio", 0.2)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    seed = split_cfg.get("random_seed", 42)

    train_val, test = train_test_split(
        meta,
        test_size=test_ratio,
        stratify=meta["stratify_label"],
        random_state=seed,
    )
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_ratio_adj,
        stratify=train_val["stratify_label"],
        random_state=seed,
    )

    train_df = train_df.drop(columns=["stratify_label"])
    val_df = val_df.drop(columns=["stratify_label"])
    test_df = test.drop(columns=["stratify_label"])

    train_path = splits_dir / "train_segments.csv"
    val_path = splits_dir / "val_segments.csv"
    test_path = splits_dir / "test_segments.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 训练集: {len(train_df)} -> {train_path}")
    print(f"[INFO] 验证集: {len(val_df)} -> {val_path}")
    print(f"[INFO] 测试集: {len(test_df)} -> {test_path}")


if __name__ == "__main__":
    run()
