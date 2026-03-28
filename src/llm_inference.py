"""
LLM 情感推理（Demo）

输入：data/descriptions/test_descriptions_dummy.csv
输出：data/emotions/llm_predictions_test.csv

说明：
- 先用占位描述跑通全链路，后续接入真实音频模型生成的描述时，保持输入 CSV 字段不变即可复用。
- provider=aliyun 时使用通义千问 DashScope OpenAI 兼容接口；
  provider=gitcode 时使用 api-ai.gitcode.com；provider=atomgit 时使用 AtomGit。
- API Key 通过环境变量（如 ALIYUN_API_KEY/GITCODE_API_KEY/ATOMGIT_API_KEY）
  或 config.yaml 的 llm.api_key 配置。
"""

import json
import os
import re
import time
import http.client
from pathlib import Path

from openai import OpenAI
import pandas as pd
import requests
import yaml
from tqdm import tqdm


def load_config(config_path=None):
    if config_path is None:
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


def extract_first_json_obj(text):
    """从文本中尽量提取第一个 JSON 对象。"""
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def build_prompt(description):
    system = (
        "你是音乐情感分析专家。你将收到一段对音乐片段的文字描述，"
        "请根据描述推断该片段的情感维度：Valence(愉悦度) 与 Arousal(唤醒度)。"
        "DEAM 常用 1~9 标度：1=很低，9=很高。"
        "请只输出 JSON，不要输出其他文字。"
    )
    user = (
        "请基于以下音乐描述推断情感：\n\n"
        f"描述：{description}\n\n"
        "输出 JSON 格式如下（数值保留 2 位小数）：\n"
        '{\n  "valence": 0.00,\n  "arousal": 0.00,\n  "label": "快乐/悲伤/平静/激昂/紧张/放松/其他"\n}\n'
        "其中 valence/arousal 取值范围为 1~9。"
    )
    return system, user


def call_aliyun_api(system_msg, user_msg, api_key, model, temperature, max_tokens, base_url=None):
    """
    使用通义千问 DashScope OpenAI 兼容接口调用 LLM。
    对应 test.py 中的用法：OpenAI(base_url=..., api_key=...).
    """
    client = OpenAI(
        base_url=(base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=api_key,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

    try:
        # openai>=1 返回的 content 一般是字符串
        content = resp.choices[0].message.content
        return content if isinstance(content, str) else str(content)
    except Exception as e:
        return f"[ERROR] Unexpected response format: {e}. Raw: {resp}"


def call_gitcode_api(system_msg, user_msg, api_key, model, temperature, max_tokens, base_url=None):
    """
    使用 GitCode API（api-ai.gitcode.com）调用 LLM。
    非流式请求；若响应体为空或非 JSON，不抛异常，返回 [ERROR] 信息便于排查。
    """
    url = (base_url or "https://api-ai.gitcode.com/v1").rstrip("/") + "/chat/completions"
    token = api_key.strip()
    if not token.startswith("Bearer "):
        token = f"Bearer {token}"
    headers = {"Authorization": token}

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        return f"[ERROR] Request failed: {e}"

    if resp.status_code != 200:
        body = (resp.text or "")[:500]
        return f"[ERROR] HTTP {resp.status_code}: {body}"

    text = resp.text or ""
    if not text.strip():
        return "[ERROR] Empty response body"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return f"[ERROR] JSONDecodeError: {e}. Body preview: {text[:300]}"

    if "choices" in data and len(data["choices"]) > 0:
        msg = data["choices"][0].get("message", {})
        return msg.get("content", "") or ""
    if "error" in data:
        err = data["error"]
        return f"[ERROR] API error: {err.get('message', err)}"
    return f"[ERROR] Unexpected response format: {list(data.keys())}"


def call_atomgit_api(system_msg, user_msg, api_key, model, temperature, max_tokens):
    """
    使用 AtomGit API 调用 LLM。
    请求格式与 AtomGit 文档示例一致：Authorization 为裸 token，body 使用 camelCase（maxTokens 等）。
    """
    conn = http.client.HTTPSConnection("api.atomgit.com")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 与你可连接的示例一致：camelCase maxTokens，并包含 top_k / frequency_penalty / top_p
    payload = json.dumps({
        "temperature": temperature,
        "top_k": 0,
        "top_p": 0,
        "frequency_penalty": 0,
        "messages": messages,
        "model": model,
        "maxTokens": max_tokens,
    })

    # AtomGit 当前接口使用裸 token，不加 Bearer 前缀（与你提供的可连接示例一致）
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": api_key.strip(),
    }

    try:
        conn.request("POST", "/api/v5/chat/completions", body=payload, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8", errors="replace")
        status = res.status
    finally:
        conn.close()

    if status != 200:
        return f"[ERROR] HTTP {status}: {data[:500] if data else '(empty body)'}"

    if not data or not data.strip():
        return "[ERROR] Empty response body"

    try:
        response_data = json.loads(data)
    except json.JSONDecodeError as e:
        return f"[ERROR] JSONDecodeError: {e}. Body preview: {data[:300]}"

    if "choices" in response_data and len(response_data["choices"]) > 0:
        msg = response_data["choices"][0].get("message", {})
        return msg.get("content", "") or ""
    if "error" in response_data:
        err = response_data["error"]
        return f"[ERROR] API error: {err.get('message', err)}"
    return f"[ERROR] Unexpected response format: {list(response_data.keys())}"


def get_api_key(llm_cfg):
    provider = (llm_cfg or {}).get("provider", "")
    if provider == "atomgit":
        env_key = os.environ.get("ATOMGIT_API_KEY")
    elif provider == "gitcode":
        env_key = os.environ.get("GITCODE_API_KEY")
    elif provider == "aliyun":
        env_key = os.environ.get("ALIYUN_API_KEY")
    else:
        env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    cfg_key = (llm_cfg or {}).get("api_key")
    if cfg_key and str(cfg_key).strip() and str(cfg_key) != "your-api-key-here":
        return cfg_key
    return None


def run(config_path=None):
    config, project_root = load_config(config_path)

    llm_cfg = config.get("llm", {})
    api_key = get_api_key(llm_cfg)
    if not api_key:
        raise RuntimeError(
            "未配置 API Key。请设置环境变量（如 ALIYUN_API_KEY）或在 config.yaml 的 llm.api_key 中填写。"
        )

    model = llm_cfg.get("model_name", "qwen-plus")
    temperature = float(llm_cfg.get("temperature", 0.7))
    max_tokens = int(llm_cfg.get("max_tokens", 200))
    base_url = (llm_cfg.get("base_url") or "").strip() or None
    provider = (llm_cfg.get("provider") or "").strip().lower()

    output_cfg = config["output"]
    descriptions_dir = project_root / output_cfg["descriptions_dir"].lstrip("./")
    emotions_dir = project_root / output_cfg["emotions_dir"].lstrip("./")
    emotions_dir.mkdir(parents=True, exist_ok=True)

    real_path = descriptions_dir / "test_descriptions_real.csv"
    dummy_path = descriptions_dir / "test_descriptions_dummy.csv"
    if real_path.exists():
        in_path = real_path
    elif dummy_path.exists():
        in_path = dummy_path
    else:
        raise FileNotFoundError(
            f"未找到输入描述文件：{real_path} 或 {dummy_path}，请先运行 audio_to_text.py"
        )

    out_path = emotions_dir / "llm_predictions_test.csv"

    df = pd.read_csv(in_path)
    required_cols = {"segment_id", "song_id", "audio_path", "description_raw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"输入文件缺少必要列：{missing}")

    # 为了控制 token 消耗，先只在前 N 条样本上测试
    MAX_SAMPLES = 20  # 可以根据需要调大或改为 None 跑全量
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)

    # 断点续跑：若输出存在，则跳过已完成的 segment_id
    done_ids = set()
    if out_path.exists():
        try:
            done_df = pd.read_csv(out_path)
            if "segment_id" in done_df.columns:
                done_ids = set(done_df["segment_id"].astype(int).tolist())
        except Exception:
            done_ids = set()

    buffer_records = []
    print(f"[INFO] 开始对测试集 {len(df)} 条样本进行 LLM 情感推理...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seg_id = int(row["segment_id"])
        if seg_id in done_ids:
            continue

        description = str(row["description_raw"])
        system_msg, user_msg = build_prompt(description)

        t0 = time.time()
        if provider == "gitcode":
            raw = call_gitcode_api(
                system_msg, user_msg, api_key, model, temperature, max_tokens, base_url=base_url
            )
        elif provider == "atomgit":
            raw = call_atomgit_api(system_msg, user_msg, api_key, model, temperature, max_tokens)
        else:  # 默认使用阿里云通义千问（OpenAI 兼容接口）
            raw = call_aliyun_api(
                system_msg, user_msg, api_key, model, temperature, max_tokens, base_url=base_url
            )
        latency_ms = int((time.time() - t0) * 1000)

        parsed = extract_first_json_obj(raw)
        pred_valence = None
        pred_arousal = None
        pred_label = None
        if isinstance(parsed, dict):
            pred_valence = parsed.get("valence")
            pred_arousal = parsed.get("arousal")
            pred_label = parsed.get("label")

        buffer_records.append(
            {
                "segment_id": seg_id,
                "song_id": int(row["song_id"]),
                "audio_path": row["audio_path"],
                "description_raw": description,
                "gt_valence_mean": row.get("valence_mean"),
                "gt_arousal_mean": row.get("arousal_mean"),
                "pred_valence": pred_valence,
                "pred_arousal": pred_arousal,
                "pred_label": pred_label,
                "raw_output": raw,
                "latency_ms": latency_ms,
                "llm_model": model,
            }
        )

        # 轻量限速，避免瞬间打爆（可按需调整）
        time.sleep(0.2)

        # 每 10 条落盘一次，防止中断丢数据
        if len(buffer_records) >= 10:
            new_df = pd.DataFrame(buffer_records)
            if out_path.exists():
                existing = pd.read_csv(out_path)
                merged = pd.concat([existing, new_df], ignore_index=True)
            else:
                merged = new_df
            merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            buffer_records = []

    # 最后落盘
    if buffer_records:
        new_df = pd.DataFrame(buffer_records)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 推理完成 -> {out_path}")


if __name__ == "__main__":
    run()