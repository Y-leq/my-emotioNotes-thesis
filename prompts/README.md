# 提示词文件说明（毕设：提示词迭代 / 全量实验对齐）

- 在 **`config.yaml`** 的 **`prompts`** 节中配置 **`version`** 与各 **`*_file`** 路径。
- 修改文案并准备重新跑实验时，请将 **`version`** 递增（如 `v1` → `v2` → `v3`），以便**预测、评测、图表按版本一一对应**，在论文中写清「提示词迭代与指标变化」。

## 与版本绑定的输出（推荐写进 task book / 论文方法）

| 环节 | 输出路径（`version` 以 `v2` 为例） |
|------|----------------------------------|
| 情感推理 | `data/emotions/llm_predictions_v2.csv` |
| LLM 评测 | `results/prompt_runs/v2/llm_eval_by_row.csv`、`llm_eval_summary.json` |
| 可视化 | `results/prompt_runs/v2/visualizations/llm_scatter_valence_arousal.png`、`confusion_matrices.png`、`traditional_scatter_valence_arousal.png` |

- 将 `config.yaml` 中 `prompts.version` 设为 `v1`，并指向 `llm_emotion_system_v1.txt` / `llm_emotion_user_v1.txt`（需自行从 v2 复制重命名或保留历史文件），跑通 `llm_inference.py` → `eval_llm_predictions.py` → `plot_eval_results.py`，即得到 **v1 全套结果**。
- 再改为 `v2`、改提示词文件路径，重跑同样三步，即得到 **v2 独立目录**，与 v1 并存，**互不覆盖**。
- 基线 `traditional_baseline_*.csv` 仍在 `results/` 根目录，与提示词版本无关；图中右侧基线子图在每次作图时复用同一份基线数据。

**命令行（可选，与 config 中 version 一致时效果相同）：**

```text
python src/llm_inference.py
python src/eval_llm_predictions.py
python src/plot_eval_results.py
# 或显式指定旧版重算图：
python src/eval_llm_predictions.py --version v1
python src/plot_eval_results.py --llm-version v1
```

若早期仅有 `llm_predictions_test.csv`，`eval` 在找不到 `llm_predictions_{version}.csv` 时会**回退**该旧文件名并提示重命名。

---

- **音频描述**：`audio_description_v2.txt` → 由 `audio_to_text.py` 读取。
- **情感推理 v3（当前默认）**：`llm_emotion_system_v3.txt` + `llm_emotion_user_v3.txt`（占位符 `<<<DESCRIPTION>>>`）。模型**只输出 valence/arousal**；`pred_label` 由 `emotion_label_mapping.py` 与基线同一规则生成。v2 文件仍保留作对比实验。
- 若删除或留空某个文件路径，脚本会回退到内置默认提示词并打印 `[WARN]`。

推荐流程：小样本调 prompt → 固定 `version` 与文件内容 → 全量 `llm_inference` → `eval` → `plot` → 论文中按版本对比表格/图。
