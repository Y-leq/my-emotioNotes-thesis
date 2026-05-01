# 提示词文件说明（毕设：提示词迭代 / 全量实验对齐）

## 一键跑 v1 → v3 → v4 → v5（论文「逐层递进」）

在项目根目录（已配置 API Key）执行：

```bash
python src/run_prompt_versions_pipeline.py
```

将对 **v1、v3、v4、v5** 依次执行：推理 → 评测 → 作图。全量测试集约 **181 条 × 4 版** LLM 调用，请注意费用与时间。若只跑当前主版本（默认见 `config.yaml` 的 `prompts.version`）：

```bash
python src/llm_inference.py
python src/eval_llm_predictions.py
python src/plot_eval_results.py
```

或显式：

```bash
python src/llm_inference.py --version v5
python src/eval_llm_predictions.py --version v5
python src/plot_eval_results.py --version v5
```

`config.yaml` 中 **`prompts.profiles`** 当前为 **v1 / v3 / v4 / v5**（**v2 已从配置与产物中移除**；`llm_emotion_*_v2.txt` 可保留作附录说明）。

| 版本 | 设计要点（写作可用） |
|------|----------------------|
| **v1** | 最简：仅两维 V/A、短约束（基线式起点） |
| **v3** | 仅输出 V/A；**标签由下游 h 映射**（与基线、评测口径一致；**较 v2 效价误差更优**，故保留） |
| **v4** | 在 v3 上**针对效价系统性偏高**加强反偏置：区分「风格词」与「情绪效价」、中性段 4～6.5、输出前自检；仍只输出两键 JSON |
| **v5** | **以 v4 反偏置为核**，删长示例与「5.5 锚点」等易致条带化的约束；强调**一行纯 JSON**、连续数值；与**音频**提示词通过 `audio_description_profiles` 正交配置 |

## 与 `prompts.version` 绑定的产物（LLM 情感）

| 步骤 | 说明 |
|------|------|
| 情感推理 | `python src/llm_inference.py` → `data/emotions/llm_predictions_test_{version}.csv` |
| 评测 | `python src/eval_llm_predictions.py` → `results/prompt_runs/{version}/` |
| 作图 | `python src/plot_eval_results.py` → `results/visualizations/prompt_runs/{version}/` |

传统基线不随提示词变化；各版可视化目录内仍含与基线对照图。

## 音频 → 文本（独立版本，与情感 `version` 正交）

| 项目 | 说明 |
|------|------|
| 配置 | `prompts.audio_description_version` + `prompts.audio_description_profiles`（v2～v5 对应 `audio_description_v*.txt`） |
| 显式文件 | 若设 `audio_description_file` 为非 null 路径，则**优先生效**为提示词；`audio_description_version` 仍决定描述 CSV 的文件名后缀 |
| 生成描述 | `python src/audio_to_text.py` 或 `python src/audio_to_text.py --audio-version v4` |
| 描述表 | `data/descriptions/test_descriptions_real_{audio_description_version}.csv`（行内列 `audio_description_version`） |
| 情感推理读入 | 默认读上表；若该文件不存在，会**回退** `test_descriptions_real.csv` 并 `[WARN]`（旧无后缀数据） |
| 指定音频版跑 LLM | `python src/llm_inference.py --audio-version v4 --version v5`（先确保已有 `test_descriptions_real_v4.csv`） |

- **v4 音频说明**：与「展示用规则生成描述」**同构**——前四维与第 5 句**情绪+能量**自洽、推荐 `旋律：…；节奏：…` 体例；**v5 音频说明**：前四句更白描、第五句仍双维度，用于与 v4 作听写文风对照。历史：无后缀旧 `test_descriptions_real.csv` 可备份后复制为 `test_descriptions_real_v2.csv` 等与版本对齐。  

## 其它说明

- **与情感 ablation 对齐**：固定 `--audio-version`，分别跑 `--version v4` 与 `v5`，即可在**同一套文本**上比情感提示词。  
- 若曾使用无版本后缀的 `llm_predictions_test.csv`，请复制为 `llm_predictions_test_{version}.csv`（与当前 `prompts.version` 一致）再评测。  
- 推荐：小样本调 prompt → 再全量 `max_samples: null`。
