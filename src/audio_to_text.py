"""
音频 -> 文本描述生成（云端真实模型版）

- 从 data/splits/test_segments.csv 读取测试集片段
- 调用阿里云百炼 DashScope 音频理解接口（qwen2-audio-instruct）生成描述
- 保存到 data/descriptions/test_descriptions_real.csv
"""
import base64
import json
import os
from pathlib import Path

import librosa
import pandas as pd
import requests
import soundfile as sf
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
    云端音频描述模型：调用 Qwen Audio 生成音乐文本描述。
    支持 provider:
    - dashscope: 阿里云百炼 DashScope 音频理解 API
    """

    def __init__(
        self,
        model_name,
        provider,
        api_key,
        endpoint,
        temperature=0.2,
        max_tokens=220,
        request_timeout=180,
        clip_seconds=30,
    ):
        self.model_name = model_name
        self.provider = provider
        self.api_key = (api_key or "").strip()
        self.endpoint = (endpoint or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.clip_seconds = int(clip_seconds)

    def _audio_to_data_url(self, audio_abs_path):
        """
        将本地音频转为 DashScope 支持的 data URL(base64)。
        同时将音频裁剪到前 clip_seconds 秒，满足接口时长限制。
        """
        y, sr = librosa.load(audio_abs_path, sr=16000, mono=True)
        max_len = self.clip_seconds * sr
        if len(y) > max_len:
            y = y[:max_len]

        tmp_wav = Path(audio_abs_path).with_suffix(".tmp_16k_mono.wav")
        sf.write(str(tmp_wav), y, sr)
        try:
            with open(tmp_wav, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            if tmp_wav.exists():
                tmp_wav.unlink()

        return f"data:;base64,{audio_b64}"

    def _call_dashscope(self, audio_abs_path):
        audio_data_url = self._audio_to_data_url(audio_abs_path)
        prompt = (
            "你是音乐内容分析助手。请仅基于音频内容生成中文描述，避免臆测。"
            "描述必须覆盖5个维度：旋律、节奏、乐器、曲风、整体听感。"
            "输出要求："
            "1) 只输出一段话（80-140字）；"
            "2) 用分号“；”分成5个短句，每个短句对应一个维度；"
            "3) 不使用列表、标题、JSON或markdown；"
            "4) 不确定的信息请用“可能/偏向/疑似”等保守措辞。"
        )
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"audio": audio_data_url},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "result_format": "message",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }
        try:
            resp = requests.post(
                self.endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.request_timeout,
            )
        except requests.RequestException as e:
            return f"[ERROR] Request failed: {e}"

        text = resp.text or ""
        if resp.status_code != 200:
            return f"[ERROR] HTTP {resp.status_code}: {text[:800]}"
        if not text.strip():
            return "[ERROR] Empty response body"

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            return f"[ERROR] JSONDecodeError: {e}. Body preview: {text[:300]}"

        # DashScope 返回结构：output.choices[0].message.content[0].text
        try:
            content = data["output"]["choices"][0]["message"]["content"]
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict)]
                merged = "".join(t for t in texts if t).strip()
                if merged:
                    return merged
            if isinstance(content, str):
                return content.strip()
        except Exception:
            pass

        return f"[ERROR] Unexpected response format: {str(data)[:500]}"

    def generate_description(self, audio_abs_path):
        if self.provider == "dashscope":
            return self._call_dashscope(audio_abs_path)
        return "[ERROR] Unsupported provider in audio_to_text config"


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
    llm_cfg = config.get("llm", {})
    model_name = model_cfg.get("model_name", "qwen2-audio-instruct")
    provider = str(model_cfg.get("provider", "dashscope")).lower()
    endpoint = model_cfg.get("endpoint") or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    api_key = model_cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY") or llm_cfg.get("api_key") or ""
    if not api_key:
        raise RuntimeError("未配置音频描述模型 API Key。请在 config.yaml 的 audio_to_text.api_key 或 llm.api_key 填写。")
    if not endpoint:
        raise RuntimeError("未配置音频描述模型 endpoint。请在 config.yaml 的 audio_to_text.endpoint 填写。")
    temperature = float(model_cfg.get("temperature", 0.2))
    max_tokens = int(model_cfg.get("max_tokens", 220))
    request_timeout = int(model_cfg.get("request_timeout", 180))
    clip_seconds = int(model_cfg.get("clip_seconds", 30))
    max_samples = model_cfg.get("max_samples")
    if max_samples is not None:
        df = df.head(int(max_samples))

    model = AudioToTextModel(
        model_name=model_name,
        provider=provider,
        api_key=api_key,
        endpoint=endpoint,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        clip_seconds=clip_seconds,
    )

    records = []
    print(f"[INFO] 开始为测试集 {len(df)} 条样本生成真实描述...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_rel = str(row["audio_path"])
        audio_abs = project_root / audio_rel
        desc = model.generate_description(str(audio_abs))
        records.append(
            {
                "segment_id": row["segment_id"],
                "song_id": row["song_id"],
                "audio_path": audio_rel,
                "valence_mean": row.get("valence_mean"),
                "arousal_mean": row.get("arousal_mean"),
                "description_raw": desc,
            }
        )

    out_path = descriptions_dir / "test_descriptions_real.csv"
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已为测试集生成 {len(records)} 条真实描述 -> {out_path}")


if __name__ == "__main__":
    run()

