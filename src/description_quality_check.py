"""
文本描述质量评估脚本

输入: data/descriptions/test_descriptions_real.csv
输出:
- data/descriptions/test_descriptions_real_qc.csv
- data/descriptions/test_descriptions_real_qc_summary.txt
"""

from pathlib import Path

import pandas as pd
import yaml


DIMENSION_KEYWORDS = {
    "melody": ["旋律", "音高", "音程", "主旋律", "线条"],
    "rhythm": ["节奏", "拍点", "律动", "速度", "快板", "慢板", "BPM"],
    "instrument": ["乐器", "钢琴", "吉他", "鼓", "贝斯", "弦乐", "小提琴", "合成器", "铜管", "口琴"],
    "style": ["风格", "曲风", "爵士", "摇滚", "古典", "流行", "电子", "雷鬼", "民谣"],
    "impression": ["听感", "氛围", "情绪", "紧张", "放松", "愉悦", "压抑", "激昂", "平静"],
}

HEDGING_WORDS = ["可能", "偏向", "疑似", "或许", "大致", "倾向"]
HALLUCINATION_RISK_WORDS = ["夏威夷", "音乐剧", "电影画面", "故事情节", "人物", "场景"]


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


def contains_any(text, words):
    return any(w in text for w in words)


def evaluate_one(desc):
    desc = str(desc or "").strip()
    if not desc:
        return {
            "len_chars": 0,
            "covered_dimensions": 0,
            "missing_dimensions": "melody,rhythm,instrument,style,impression",
            "has_hedging": False,
            "hallucination_risk": True,
            "quality_score": 0,
            "quality_level": "FAIL",
            "quality_note": "描述为空",
        }

    covered = []
    for dim, kws in DIMENSION_KEYWORDS.items():
        if contains_any(desc, kws):
            covered.append(dim)

    covered_count = len(covered)
    missing = [d for d in DIMENSION_KEYWORDS.keys() if d not in covered]

    length = len(desc)
    length_ok = 80 <= length <= 180
    has_hedging = contains_any(desc, HEDGING_WORDS)
    risk = contains_any(desc, HALLUCINATION_RISK_WORDS)

    # 0-100 评分：维度覆盖(50) + 长度(25) + 风险控制(25)
    score = 0
    score += int((covered_count / 5) * 50)
    score += 25 if length_ok else 10
    score += 25 if not risk else 5

    if score >= 80:
        level = "PASS"
    elif score >= 60:
        level = "WARN"
    else:
        level = "FAIL"

    notes = []
    if not length_ok:
        notes.append("长度不在80~180字")
    if covered_count < 5:
        notes.append(f"维度覆盖不足({covered_count}/5)")
    if risk:
        notes.append("存在臆测风险词")
    if not notes:
        notes.append("质量良好")

    return {
        "len_chars": length,
        "covered_dimensions": covered_count,
        "missing_dimensions": ",".join(missing) if missing else "",
        "has_hedging": has_hedging,
        "hallucination_risk": risk,
        "quality_score": score,
        "quality_level": level,
        "quality_note": ";".join(notes),
    }


def run(config_path=None):
    config, project_root = load_config(config_path)
    out_cfg = config["output"]
    descriptions_dir = project_root / out_cfg["descriptions_dir"].lstrip("./")

    in_path = descriptions_dir / "test_descriptions_real.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"未找到描述文件: {in_path}")

    df = pd.read_csv(in_path)
    if "description_raw" not in df.columns:
        raise RuntimeError("输入文件缺少 description_raw 列")

    qc_rows = [evaluate_one(x) for x in df["description_raw"].tolist()]
    qc_df = pd.DataFrame(qc_rows)
    merged = pd.concat([df, qc_df], axis=1)

    out_csv = descriptions_dir / "test_descriptions_real_qc.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    total = len(merged)
    pass_n = int((merged["quality_level"] == "PASS").sum())
    warn_n = int((merged["quality_level"] == "WARN").sum())
    fail_n = int((merged["quality_level"] == "FAIL").sum())
    avg_score = float(merged["quality_score"].mean()) if total else 0.0

    summary = (
        f"样本数: {total}\n"
        f"PASS: {pass_n}\n"
        f"WARN: {warn_n}\n"
        f"FAIL: {fail_n}\n"
        f"平均分: {avg_score:.2f}\n"
    )
    out_txt = descriptions_dir / "test_descriptions_real_qc_summary.txt"
    out_txt.write_text(summary, encoding="utf-8")

    print(f"[INFO] 质量评估完成 -> {out_csv}")
    print(f"[INFO] 汇总结果 -> {out_txt}")
    print(summary)


if __name__ == "__main__":
    run()

