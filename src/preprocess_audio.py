"""
DEAM 音频预处理脚本（方案A：直接使用 45s 原生片段）

- 统一格式为 WAV，采样率 44.1kHz，16bit
- 可选首尾静音裁剪
- 音量归一化
- 输出 segments_meta.csv（segment_id = song_id，一对一）
"""
import yaml
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_static_annotations(annotations_dir: Path) -> "pd.DataFrame":
    import pandas as pd
    static_dir = annotations_dir / "annotations averaged per song" / "song_level"
    files = [
        static_dir / "static_annotations_averaged_songs_1_2000.csv",
        static_dir / "static_annotations_averaged_songs_2000_2058.csv",
    ]
    dfs = []
    for f in files:
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        cols = [c for c in ["song_id", "valence_mean", "arousal_mean"] if c in df.columns]
        dfs.append(df[cols])
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["song_id"])
    return merged


def find_audio_files(audio_root: Path) -> list[Path]:
    """递归搜索所有 mp3 文件"""
    root = Path(audio_root)
    return list(root.rglob("*.mp3"))


def preprocess_one(
    in_path: Path,
    out_path: Path,
    sr: int = 44100,
    mono: bool = True,
    trim_db=30,
    normalize: bool = True,
) -> bool:
    try:
        y, orig_sr = librosa.load(in_path, sr=None, mono=mono)
    except Exception as e:
        print(f"[WARN] 无法加载 {in_path}: {e}")
        return False

    if mono and y.ndim > 1:
        y = np.mean(y, axis=0)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    if trim_db is not None and trim_db > 0:
        y_trimmed, _ = librosa.effects.trim(y, top_db=trim_db)
        if len(y_trimmed) < sr * 5:
            y = y
        else:
            y = y_trimmed

    if normalize:
        peak = np.abs(y).max()
        if peak > 1e-8:
            y = y / peak * 0.95

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr, subtype="PCM_16")
    return True


def run(config_path=None):
    if config_path is None:
        for p in [Path("config.yaml"), Path(__file__).parent.parent / "config.yaml"]:
            if p.exists():
                config_path = str(p)
                break
        config_path = config_path or "config.yaml"
    config = load_config(config_path)
    project_root = Path(config_path).resolve().parent
    dataset_cfg = config["dataset"]
    prep_cfg = config["preprocessing"]
    output_cfg = config["output"]

    annotations_dir = project_root / dataset_cfg["deam_annotations_dir"].lstrip("./")
    audio_root = project_root / dataset_cfg["deam_audio_dir"].lstrip("./")
    out_audio_dir = project_root / output_cfg["processed_audio_dir"].lstrip("./")
    splits_dir = project_root / output_cfg["splits_dir"].lstrip("./")

    print("[1] 加载静态标注...")
    static = load_static_annotations(annotations_dir)
    annotated_ids = set(static["song_id"].astype(int))

    print("[2] 搜索音频文件...")
    audio_files = find_audio_files(audio_root)
    id_to_path = {}
    for p in audio_files:
        try:
            sid = int(p.stem)
            id_to_path[sid] = p
        except ValueError:
            pass

    valid_ids = annotated_ids & set(id_to_path.keys())
    missing_audio = annotated_ids - set(id_to_path.keys())
    if missing_audio:
        print(f"[WARN] 缺失音频的 song_id 数量: {len(missing_audio)}（前 5 个: {list(missing_audio)[:5]}）")
    print(f"[INFO] 可处理样本数: {len(valid_ids)}")

    sr = prep_cfg.get("target_sr", 44100)
    mono = prep_cfg.get("mono", True)
    trim_db = prep_cfg.get("trim_db") if prep_cfg.get("trim_db") else None

    print("[3] 预处理音频...")
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in tqdm(sorted(valid_ids)):
        in_p = id_to_path[sid]
        out_p = out_audio_dir / f"{sid}.wav"
        ok = preprocess_one(in_p, out_p, sr=sr, mono=mono, trim_db=trim_db)
        if not ok:
            continue
        row = static[static["song_id"] == sid].iloc[0]
        rel_path = out_p.relative_to(project_root).as_posix()
        record = {
            "segment_id": sid,
            "song_id": sid,
            "audio_path": rel_path,
        }
        if "valence_mean" in row.index:
            record["valence_mean"] = float(row["valence_mean"])
        if "arousal_mean" in row.index:
            record["arousal_mean"] = float(row["arousal_mean"])
        if "valence_std" in row.index:
            record["valence_std"] = float(row["valence_std"])
        if "arousal_std" in row.index:
            record["arousal_std"] = float(row["arousal_std"])
        rows.append(record)

    import pandas as pd
    meta = pd.DataFrame(rows)
    splits_dir.mkdir(parents=True, exist_ok=True)
    meta_path = splits_dir / "segments_meta.csv"
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"[4] 已保存 {len(meta)} 条元数据 -> {meta_path}")
    return meta_path


if __name__ == "__main__":
    run()
