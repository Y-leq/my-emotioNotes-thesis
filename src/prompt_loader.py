"""
从项目根目录下的文本文件加载提示词，供 audio_to_text / llm_inference 使用。

配置文件见 config.yaml 的 prompts 节；路径相对于项目根目录。
"""

from __future__ import annotations

from pathlib import Path


def resolve_under_project(project_root: Path, rel: str) -> Path:
    rel = str(rel).strip().lstrip("./").replace("\\", "/")
    return (project_root / rel).resolve()


def load_text_file(project_root: Path, rel: str) -> str:
    path = resolve_under_project(project_root, rel)
    if not path.is_file():
        raise FileNotFoundError(f"提示词文件不存在: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_optional(project_root: Path, rel: str | None) -> str | None:
    if not rel:
        return None
    try:
        return load_text_file(project_root, str(rel))
    except FileNotFoundError:
        return None
