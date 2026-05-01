"""
与 prompts 对齐的产物路径，便于多版本提示词与论文一一对应。

- LLM 情感: prompts.version →
    - 预测: data/emotions/llm_predictions_test_{version}.csv
    - 评测: results/prompt_runs/{version}/llm_eval_by_row.csv, llm_eval_summary.json
    - 作图: results/visualizations/prompt_runs/{version}/*.png
- 音频→文本: prompts.audio_description_version →
    - 描述: data/descriptions/test_descriptions_real_{audio_version}.csv
    - 与 prompts.version（情感提示词版）正交，可独立迭代。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def prompt_version_from_config(config: Optional[Dict[str, Any]], fallback: str = "v1") -> str:
    if not config:
        return fallback
    v = (config.get("prompts") or {}).get("version", fallback)
    s = str(v).strip() if v is not None else fallback
    return s or fallback


def rel_emotions_dir(config: Optional[Dict[str, Any]], fallback: str = "./data/emotions") -> str:
    if not config:
        return fallback
    out = config.get("output") or {}
    return str(out.get("emotions_dir", fallback) or fallback)


def rel_results_dir(config: Optional[Dict[str, Any]], fallback: str = "./results") -> str:
    if not config:
        return fallback
    out = config.get("output") or {}
    return str(out.get("results_dir", fallback) or fallback)


def llm_predictions_csv(project_root: Path, config: Dict[str, Any]) -> Path:
    ver = prompt_version_from_config(config)
    emo = rel_emotions_dir(config)
    return project_root / emo.lstrip("./") / f"llm_predictions_test_{ver}.csv"


def llm_results_run_dir(project_root: Path, config: Dict[str, Any]) -> Path:
    ver = prompt_version_from_config(config)
    rd = rel_results_dir(config)
    return project_root / rd.lstrip("./") / "prompt_runs" / ver


def llm_visualizations_run_dir(project_root: Path, config: Dict[str, Any]) -> Path:
    ver = prompt_version_from_config(config)
    rd = rel_results_dir(config)
    return project_root / rd.lstrip("./") / "visualizations" / "prompt_runs" / ver


def rel_descriptions_dir(config: Optional[Dict[str, Any]], fallback: str = "./data/descriptions") -> str:
    if not config:
        return fallback
    out = config.get("output") or {}
    return str(out.get("descriptions_dir", fallback) or fallback)


def audio_description_version_from_config(config: Optional[Dict[str, Any]], fallback: str = "v4") -> str:
    if not config:
        return fallback
    prompts = config.get("prompts") or {}
    v = prompts.get("audio_description_version", fallback)
    s = str(v).strip() if v is not None else fallback
    return s or fallback


def resolve_audio_description_file_rel(config: Dict[str, Any]) -> str:
    """
    返回相对项目根目录的提示词文件路径：优先显式 audio_description_file（非空），
    否则用 audio_description_profiles[audio_description_version]。
    """
    prompts = config.get("prompts") or {}
    explicit = prompts.get("audio_description_file")
    if explicit is not None and str(explicit).strip() and str(explicit).strip().lower() not in (
        "null",
        "none",
        "",
    ):
        return str(explicit).strip()
    ver = audio_description_version_from_config(config)
    profs = prompts.get("audio_description_profiles") or {}
    if ver not in profs:
        raise KeyError(
            f"config prompts.audio_description_profiles 未定义版本 {ver!r}，"
            f"或请设置 prompts.audio_description_file。当前可用: {sorted(profs.keys())}"
        )
    return str(profs[ver])


def test_descriptions_real_csv(project_root: Path, config: Dict[str, Any]) -> Path:
    """按 audio_description_version 版本化的数据描述表路径。"""
    ad_ver = audio_description_version_from_config(config)
    ddir = rel_descriptions_dir(config)
    return project_root / ddir.lstrip("./") / f"test_descriptions_real_{ad_ver}.csv"


def resolve_test_descriptions_input_path(project_root: Path, config: Dict[str, Any]) -> Path:
    """
    优先 test_descriptions_real_{audio_version}.csv；若不存在则回退 test_descriptions_real.csv（旧无后缀命名）。
    """
    vpath = test_descriptions_real_csv(project_root, config)
    if vpath.exists():
        return vpath
    legacy = project_root / rel_descriptions_dir(config).lstrip("./") / "test_descriptions_real.csv"
    if legacy.exists():
        return legacy
    return vpath


def apply_audio_description_profile(config: Dict[str, Any], ad_version: str) -> None:
    """
    设置 prompts.audio_description_version，并令 audio_description_file 指向对应 profiles 条目
   （供 audio_to_text / llm_inference 的 --audio-version 与配置一致）。
    """
    prompts = config.setdefault("prompts", {})
    profs = prompts.get("audio_description_profiles") or {}
    if ad_version not in profs:
        raise KeyError(
            f"config prompts.audio_description_profiles 未定义音频提示词 {ad_version!r}，"
            f"当前可用: {sorted(profs.keys())}"
        )
    prompts["audio_description_version"] = ad_version
    prompts["audio_description_file"] = str(profs[ad_version])


def apply_prompt_profile(config: Dict[str, Any], version: str) -> None:
    """
    用 config['prompts']['profiles'][version] 覆盖 system/user 路径，并设置 prompts.version。
    供命令行 --version v1 与批量脚本使用，无需手改 yaml。
    """
    prompts = config.setdefault("prompts", {})
    profiles = prompts.get("profiles") or {}
    if version not in profiles:
        raise KeyError(
            f"config.yaml 中 prompts.profiles 未定义版本 {version!r}，当前可用: {sorted(profiles.keys())}"
        )
    prof = profiles[version]
    prompts["version"] = version
    if "llm_system_file" in prof:
        prompts["llm_system_file"] = prof["llm_system_file"]
    if "llm_user_template_file" in prof:
        prompts["llm_user_template_file"] = prof["llm_user_template_file"]
