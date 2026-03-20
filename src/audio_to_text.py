"""
音频 -> 文本描述生成（占位版）

- 从 data/splits/test_segments.csv 读取测试集片段
- 为每个片段生成“假描述”（包含旋律/节奏/乐器/曲风/听感五个维度的中文文本）
- 保存到 data/descriptions/test_descriptions_dummy.csv

后续接入 MERT/CLAP 时，只需要替换 AudioToTextModel.generate_description 内部实现。
"""
import random
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path=None):
    if config_path is None:
        # 优先使用项目根目录下的 config.yaml
        here = Path(__file__).resolve().parent
        candidates = [here.parent / "config.yaml", Path("config.yaml")]
        for p in candidates:
            if p.exists():
                config_path = str(p)
                break
        if config_path is None:
            raise FileNotFoundError("未找到 config.yaml，请确认项目根目录下存在该文件。")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), Path(config_path).resolve().parent


class AudioToTextModel:
    """
    占位模型：根据随机模板生成五维度描述。

    后续接入真实模型时，可以保持接口不变，只改内部实现。
    """

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device

        # 一些简单的描述模板，用于构造占位文本
        self.melodies = [
            "旋律平缓起伏，线条流畅",
            "旋律跳跃明显，充满张力",
            "旋律多在中高音区徘徊，略显紧张",
            "旋律以长音为主，情绪较为舒缓",
        ]
        self.tempos = [
            "节奏轻快，拍点清晰稳定",
            "中速节奏，律动感适中",
            "偏慢板，节奏松弛舒缓",
            "节奏切分较多，略显复杂",
        ]
        self.instruments = [
            "以钢琴为主，偶尔加入弦乐铺垫",
            "吉他与鼓点共同构成主要织体",
            "合成器铺底，夹杂电子鼓与贝斯",
            "小提琴独奏为主，背景有弱弦乐和声",
        ]
        self.styles = [
            "偏向流行抒情风格",
            "带有摇滚色彩的编配",
            "接近古典交响乐的管弦配器",
            "偏电子舞曲风格，合成器色彩明显",
        ]
        self.impressions = [
            "整体听感温暖治愈，情绪偏积极",
            "整体略显压抑，带有一定紧张感",
            "情绪起伏较大，在平静与激昂之间来回切换",
            "氛围轻松愉悦，适合作为背景音乐",
        ]

    def generate_description(self, audio_path, row=None):
        """
        生成占位描述文本。
        row: 可选，对应 test_segments 中该条的元数据（包含 valence/arousal 等）。
        """
        parts = [
            random.choice(self.melodies),
            random.choice(self.tempos),
            random.choice(self.instruments),
            random.choice(self.styles),
            random.choice(self.impressions),
        ]
        return "；".join(parts) + "。"


def run(config_path=None):
    config, project_root = load_config(config_path)
    output_cfg = config["output"]

    splits_dir = project_root / output_cfg["splits_dir"].lstrip("./")
    descriptions_dir = project_root / output_cfg["descriptions_dir"].lstrip("./")
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    test_segments_path = splits_dir / "test_segments.csv"
    if not test_segments_path.exists():
        raise FileNotFoundError(f"未找到测试集划分文件: {test_segments_path}，请先运行 split_dataset.py")

    df = pd.read_csv(test_segments_path)

    model_cfg = config.get("audio_to_text", {})
    model_name = model_cfg.get("model_name", "dummy-model")
    device = model_cfg.get("device", "cpu")
    model = AudioToTextModel(model_name=model_name, device=device)

    records = []
    print(f"[INFO] 开始为测试集 {len(df)} 条样本生成占位描述...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        desc = model.generate_description(row["audio_path"], row=row)
        records.append(
            {
                "segment_id": row["segment_id"],
                "song_id": row["song_id"],
                "audio_path": row["audio_path"],
                "valence_mean": row.get("valence_mean"),
                "arousal_mean": row.get("arousal_mean"),
                "description_raw": desc,
            }
        )

    out_path = descriptions_dir / "test_descriptions_dummy.csv"
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已为测试集生成 {len(records)} 条占位描述 -> {out_path}")


if __name__ == "__main__":
    run()

