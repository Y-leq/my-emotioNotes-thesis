"""
依次执行 v1 → v3 → v4 → v5：情感推理 → 评测 → 作图（每版独立产物，便于论文写递进迭代）。

项目根目录下执行:
  python src/run_prompt_versions_pipeline.py

会调用同目录的 llm_inference / eval_llm_predictions / plot_eval_results，各带 --version（当前顺序 v1→v3→v4→v5）。
注意：全量约 181 条 × 4 版，请确认配额。仅跑当前主版本时可直接运行三条命令并省略 --version（与 config 中 prompts.version 一致）或显式加 --version v5。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 与 config prompts.profiles 一致；已弃用 v2（保留在 prompts/ 下旧文件作附录引用时可保留）
VERSIONS = ["v1", "v3", "v4", "v5"]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    py = sys.executable
    steps = [
        ("llm_inference.py", "LLM 批量推理"),
        ("eval_llm_predictions.py", "评测汇总"),
        ("plot_eval_results.py", "可视化"),
    ]
    for v in VERSIONS:
        print(f"\n{'='*20} 提示词 {v} {'='*20}")
        for script, desc in steps:
            cmd = [py, str(root / "src" / script), "--version", v]
            print(f"[RUN] {desc}: {' '.join(cmd)}")
            r = subprocess.run(cmd, cwd=str(root))
            if r.returncode != 0:
                print(f"[ERR] 失败: {script} (exit {r.returncode})")
                return r.returncode
    print(
        f"\n[OK] 全部完成。请查看各版目录:\n"
        f"  results/prompt_runs/{{v1,v3,v4,v5}}/\n"
        f"  results/visualizations/prompt_runs/{{v1,v3,v4,v5}}/\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
