"""
DEAM 数据验证与探索脚本
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path


class DEAMDataExplorer:
    def __init__(self, project_root):
        """
        :param project_root: 项目根目录（DEAM_Annotations 与 DEAM_audio 所在目录）
        """
        self.project_root = Path(project_root)
        self.annotations_path = self.project_root / "DEAM_Annotations" / "annotations"

    def load_static_annotations(self):
        """
        加载每首歌静态标注
        """
        static_dir = self.annotations_path / "annotations averaged per song" / "song_level"
        static_files = [
            static_dir / "static_annotations_averaged_songs_1_2000.csv",
            static_dir / "static_annotations_averaged_songs_2000_2058.csv",
        ]
        dfs = []
        for f in static_files:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        static_annotations = pd.concat(dfs, ignore_index=True)
        static_annotations = static_annotations.drop_duplicates(subset=["song_id"], keep="first")

        print(f"[INFO] 加载了 {len(static_annotations)} 首歌的静态情感标注.")
        return static_annotations

    def load_dynamic_annotations(self):
        """
        加载每首歌逐秒标注（Valence 和 Arousal 曲线）
        """
        valence_path = (
            self.annotations_path / "dynamic (per second annotations)" / "valence.csv"
        )
        arousal_path = (
            self.annotations_path / "dynamic (per second annotations)" / "arousal.csv"
        )

        valence = pd.read_csv(valence_path, header=None)  # 无列名
        arousal = pd.read_csv(arousal_path, header=None)

        print(
            f"[INFO] 加载了 {valence.shape[0]} 首歌（每秒 Valence + Arousal）的逐秒标注."
        )
        return valence, arousal

    def validate_audio_files(self, audio_dir):
        """
        验证音频文件与标注的匹配性（递归搜索 mp3）
        """
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.rglob("*.mp3"))
        audio_ids = set()
        for f in audio_files:
            try:
                audio_ids.add(int(f.stem))
            except ValueError:
                pass

        print(f"[INFO] 音频文件数: {len(audio_files)}，有效 ID 数: {len(audio_ids)}")

        # 动态标注检查
        static_annotations = self.load_static_annotations()
        annotated_ids = set(static_annotations["song_id"])
        missing_ids = annotated_ids - audio_ids

        if missing_ids:
            print(f"[WARNING] 缺失音频的标注 ID: {missing_ids}")
        else:
            print("[INFO] 所有标注 ID 都有对应音频文件。")

    def visualize_annotations(self):
        """
        可视化 Valence 和 Arousal 的分布
        """
        print("\n[INFO] 开始可视化标注分布...")
        static_annotations = self.load_static_annotations()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        static_annotations["valence_mean"].hist(ax=axes[0], bins=20, color="skyblue")
        axes[0].set_title("Valence 分布")
        axes[0].set_xlabel("Valence")
        axes[0].set_ylabel("歌曲数量")
        static_annotations["arousal_mean"].hist(
            ax=axes[1], bins=20, color="lightcoral"
        )
        axes[1].set_title("Arousal 分布")
        axes[1].set_xlabel("Arousal")
        axes[1].set_ylabel("歌曲数量")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    audio_dir = project_root / "DEAM_audio"

    explorer = DEAMDataExplorer(project_root)

    print("\n[Step 1] 验证静态标注...")
    static_annotations = explorer.load_static_annotations()

    print("\n[Step 2] 验证动态标注...")
    valence, arousal = explorer.load_dynamic_annotations()

    print("\n[Step 3] 验证音频文件存在与标注对齐...")
    explorer.validate_audio_files(audio_dir)

    print("\n[Step 4] 可视化标注分布...")
    explorer.visualize_annotations()