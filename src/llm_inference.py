"""
LLM 情感推理（Demo）

输入：data/descriptions/test_descriptions_real_{audio_version}.csv（或回退无后缀旧文件）
输出：data/emotions/llm_predictions_test_{prompts.version}.csv（与 config 中 prompts.version 对齐，便于多版本对照）

说明：
- 先用占位描述跑通全链路，后续接入真实音频模型生成的描述时，保持输入 CSV 字段不变即可复用。
- provider=openai 时使用 OpenAI 官方 API（或 compatible 的 base_url）；
- provider=aliyun 时使用通义千问 DashScope OpenAI 兼容接口；
- provider=gitcode 时使用 api-ai.gitcode.com；provider=atomgit 时使用 AtomGit。
- API Key 通过环境变量（OPENAI_API_KEY/ALIYUN_API_KEY 等）或 config.yaml 的 llm.api_key（勿提交密钥到仓库）。
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import http.client
from pathlib import Path
from typing import Optional

from openai import OpenAI
import pandas as pd
import requests
import yaml
from tqdm import tqdm

from emotion_label_mapping import map_valence_arousal_to_label
from prompt_artifact_paths import (
    apply_audio_description_profile,
    apply_prompt_profile,
    audio_description_version_from_config,
    llm_predictions_csv,
    resolve_test_descriptions_input_path,
    test_descriptions_real_csv,
)
from prompt_loader import load_optional


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


def build_prompt_legacy(description):
    """未配置 prompts 文件时的内置模板：仅输出 valence/arousal，label 由代码映射。"""
    system = (
        "你是音乐情感分析专家。你将收到一段对音乐片段的文字描述，"
        "请根据描述推断 Valence(愉悦度) 与 Arousal(唤醒度)。"
        "DEAM 常用 1~9 标度：1=很低，9=很高。"
        "只输出 JSON，且仅包含 valence、arousal 两个键，不要输出 label。"
    )
    user = (
        "请基于以下音乐描述推断情感：\n\n"
        f"描述：{description}\n\n"
        "输出 JSON 格式如下（数值保留 2 位小数）：\n"
        '{"valence": 0.00, "arousal": 0.00}\n'
        "其中 valence/arousal 取值范围为 1~9。"
    )
    return system, user


def _parse_va_from_dict(parsed):
    """从模型返回的 dict 中解析 valence/arousal；失败返回 (None, None)。"""
    if not isinstance(parsed, dict):
        return None, None
    try:
        v = float(parsed.get("valence"))
        a = float(parsed.get("arousal"))
    except (TypeError, ValueError):
        return None, None
    if not (math.isfinite(v) and math.isfinite(a)):
        return None, None
    return v, a


def build_prompt(description, project_root: Path, config: dict):
    """
    优先从 config.yaml 的 prompts 节加载 system / user 模板；否则使用 build_prompt_legacy。
    返回 (system_msg, user_msg, prompt_version)。
    """
    prompts_cfg = config.get("prompts") or {}
    version = str(prompts_cfg.get("version", "v1"))
    sys_text = load_optional(project_root, prompts_cfg.get("llm_system_file"))
    usr_tpl = load_optional(project_root, prompts_cfg.get("llm_user_template_file"))
    if sys_text and usr_tpl:
        if "<<<DESCRIPTION>>>" in usr_tpl:
            user_msg = usr_tpl.replace("<<<DESCRIPTION>>>", description)
        else:
            user_msg = usr_tpl.replace("{description}", description)
        return sys_text, user_msg, version
    s, u = build_prompt_legacy(description)
    return s, u, version


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
    provider = (llm_cfg or {}).get("provider", "").strip().lower()
    if provider == "openai":
        env_key = os.environ.get("OPENAI_API_KEY")
    elif provider == "atomgit":
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


def run(
    config_path=None,
    prompt_version: Optional[str] = None,
    audio_version: Optional[str] = None,
):
    config, project_root = load_config(config_path)
    if audio_version:
        apply_audio_description_profile(config, audio_version.strip())
        print(f"[INFO] 使用音频描述数据/提示词档: {audio_version}（audio_description_profiles）")
    if prompt_version:
        apply_prompt_profile(config, prompt_version.strip())
        print(f"[INFO] 使用情感提示词配置档: {prompt_version}（profiles）")

    llm_cfg = config.get("llm", {})
    api_key = get_api_key(llm_cfg)
    if not api_key:
        hint = "OPENAI_API_KEY" if (llm_cfg.get("provider") or "").strip().lower() == "openai" else "ALIYUN_API_KEY 等"
        raise RuntimeError(
            f"未配置 API Key。请设置环境变量（如 {hint}）或在 config.yaml 的 llm.api_key 中填写（勿提交仓库）。"
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

    ad_ver = audio_description_version_from_config(config)
    versioned = test_descriptions_real_csv(project_root, config)
    legacy = descriptions_dir / "test_descriptions_real.csv"
    dummy_path = descriptions_dir / "test_descriptions_dummy.csv"
    in_path = resolve_test_descriptions_input_path(project_root, config)
    if in_path == legacy and in_path.exists():
        print(
            f"[WARN] 使用无后缀旧文件 {in_path.name}；"
            f"与当前 audio_description_version={ad_ver!r} 不完全绑定，建议用 audio_to_text 生成 {versioned.name}"
        )
    if not in_path.exists():
        if dummy_path.exists():
            in_path = dummy_path
        else:
            raise FileNotFoundError(
                f"未找到输入描述：期望 {versioned}（或旧版 {legacy}），或 {dummy_path}。"
                f" 请先按当前 audio 提示词版运行: python src/audio_to_text.py [--audio-version {ad_ver}]"
            )

    out_path = llm_predictions_csv(project_root, config)
    try:
        in_rel = in_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        in_rel = in_path
    print(f"[INFO] 输入描述文件: {in_rel}")
    print(f"[INFO] 当前情感 prompts.version 输出: {out_path.name}")

    df = pd.read_csv(in_path)
    required_cols = {"segment_id", "song_id", "audio_path", "description_raw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"输入文件缺少必要列：{missing}")

    # 与 config.yaml 中 llm.max_samples 对齐；null 表示处理描述文件全部行
    max_samples = llm_cfg.get("max_samples")
    if max_samples is not None:
        df = df.head(int(max_samples))

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
        system_msg, user_msg, prompt_ver = build_prompt(description, project_root, config)

        t0 = time.time()
        if provider == "gitcode":
            raw = call_gitcode_api(
                system_msg, user_msg, api_key, model, temperature, max_tokens, base_url=base_url
            )
        elif provider == "atomgit":
            raw = call_atomgit_api(system_msg, user_msg, api_key, model, temperature, max_tokens)
        elif provider == "openai":
            # OpenAI 官方或任意 OpenAI 兼容 /chat/completions 端点
            raw = call_aliyun_api(
                system_msg,
                user_msg,
                api_key,
                model,
                temperature,
                max_tokens,
                base_url=base_url or "https://api.openai.com/v1",
            )
        else:  # aliyun 等：默认通义千问（OpenAI 兼容接口）
            raw = call_aliyun_api(
                system_msg, user_msg, api_key, model, temperature, max_tokens, base_url=base_url
            )
        latency_ms = int((time.time() - t0) * 1000)

        parsed = extract_first_json_obj(raw)
        pred_valence = None
        pred_arousal = None
        pred_label = None
        pred_label_llm = None
        pv, pa = _parse_va_from_dict(parsed)
        if pv is not None:
            pred_valence = round(pv, 4)
            pred_arousal = round(pa, 4)
            pred_label = map_valence_arousal_to_label(pv, pa)
        if isinstance(parsed, dict) and parsed.get("label") is not None:
            pred_label_llm = str(parsed.get("label")).strip() or None

        rec = {
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
            "prompt_version": prompt_ver,
        }
        if pred_label_llm:
            rec["pred_label_llm"] = pred_label_llm
        buffer_records.append(rec)

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
    import argparse

    ap = argparse.ArgumentParser(description="LLM 情感推理；--version 指定 prompts.profiles（如 v1/v3/v4）。")
    ap.add_argument(
        "--version",
        type=str,
        default=None,
        metavar="V",
        help="情感提示词：如 v1、v3、v4、v5，对应 config prompts.profiles",
    )
    ap.add_argument(
        "--audio-version",
        type=str,
        default=None,
        dest="audio_version",
        metavar="A",
        help="音频描述提示词/描述表：如 v2、v3、v4，对应 prompts.audio_description_profiles；"
        "输入为 test_descriptions_real_{A}.csv",
    )
    ap.add_argument("--config", type=str, default=None, help="config.yaml 路径，默认项目根目录")
    args = ap.parse_args()
    run(
        config_path=args.config,
        prompt_version=args.version,
        audio_version=args.audio_version,
    )