"""
根据 test_segments 的 DEAM 真值，合成与效价/唤醒**语义对齐**的五句中文描述，供论文「理想对照/上界式」实验。
不调用音频 API；与真实 audio_to_text 结果对比时，请保留各自备份。

输出：
  - data/descriptions/test_descriptions_real_v4.csv
  - data/descriptions/test_descriptions_real_v5.csv

用法（项目根目录）: python src/build_showcase_descriptions.py
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def _valence_clauses(v: float) -> tuple[str, str]:
    """(情绪倾向短句, 与离散象限略呼应的补语) — 用于第 5 句前半。"""
    if v <= 4.0:
        return ("听感偏阴郁、压抑，情绪明显偏负", "偏悲伤或沉重氛围")
    if v <= 4.5:
        return ("听感略冷、略沉，情绪略偏负", "与低愉悦线索一致")
    if v < 5.5:
        return ("听感以中性、氛围感为主，情绪偏平", "难辨时保持中性")
    if v < 6.0:
        return ("听感略偏明朗、温和，情绪略偏正", "平静中略带暖意")
    if v < 7.0:
        return ("听感偏温暖或轻快，情绪偏正", "愉悦感可辨")
    if v < 7.5:
        return ("听感明亮、向上，正性情绪足", "轻松愉快为主")
    return ("听感非常欢快、明亮，强正性", "高愉悦、节庆感可辨")


def _arousal_clause(a: float) -> str:
    if a <= 3.5:
        return "能量很低，舒缓、松弛，慢板感强"
    if a <= 4.5:
        return "能量偏低，节奏舒缓，张弛偏松"
    if a < 5.5:
        return "能量中等，推进稳，不快不慢"
    if a < 6.0:
        return "能量中上，力度与配器动势明显"
    if a < 7.0:
        return "能量较高，偏紧张或带冲击感"
    return ("能量很高，快、强力度或高度紧张" if a >= 7.5 else "能量高，紧促或强烈动感")


def _va_bucket3(x: float) -> int:
    """与下游标签分界大致一致: <=4.5 低, <6.0 中, >=6.0 高。"""
    if x <= 4.5:
        return 0
    if x < 6.0:
        return 1
    return 2


def _quadrant_stock(seg: int, v: float, a: float) -> tuple[tuple[str, str, str, str], tuple[str, str, str, str]]:
    """
    前四句与 (V,A) 分桶一致，避免「旋律上扬」配低真值等矛盾。每桶内用 seg%3 微变。
    返回 (v4四元组, v5四元组)。
    """
    vb, ab = _va_bucket3(v), _va_bucket3(a)
    t = (seg // 1) % 3  # 0,1,2 轮换

    # (vb,ab) 键 -> 模板列表，每项 (v4_m,v4_r,v4_i,v4_g) 与 (v5_紧凑四短句)
    p: dict[tuple[int, int], list[tuple[tuple[str, str, str, str], tuple[str, str, str, str]]]] = {
        (0, 0): [
            (
                (
                    "旋律多下行、呼吸长，和声偏暗",
                    "慢板或散板感，留白多",
                    "钢弦、弦乐低音区与弱打击为主",
                    "偏慢板抒情、情绪内收",
                ),
                (
                    "下行线条与长呼吸句多",
                    "慢、宽、弱起",
                    "弦乐/钢琴中低频突出",
                    "暗色慢板抒情",
                ),
            ),
            (
                (
                    "旋律小调感可能较强，音域偏中低",
                    "节奏迟滞、拖曳感，不推拍",
                    "以合成器铺底与极轻打击",
                    "氛围偏阴郁的配乐或后摇感",
                ),
                (
                    "中低音域动机反复",
                    "不强调正拍，迟滞感",
                    "铺底+细打击",
                    "阴郁氛围/后摇",
                ),
            ),
            (
                (
                    "旋律线收敛，句幅压抑",
                    "极慢、动态压缩",
                    "大提琴或低音吉他占比高",
                    "情绪沉重、少明亮的曲风",
                ),
                (
                    "句幅短而压抑的动机",
                    "极慢、弱力度",
                    "低音乐器主导",
                    "沉重、暗",
                ),
            ),
        ],
        (0, 1): [
            (
                (
                    "旋律有不安或跳跃的碎片感",
                    "中速，切分与意外重音多",
                    "吉他与鼓的错位制造张力",
                    "可能偏后朋、新浪潮式紧张",
                ),
                (
                    "不安定的碎片式动机",
                    "中速、切分多",
                    "吉他与鼓对位紧",
                    "后朋/新波紧张感",
                ),
            ),
        ],
        (0, 2): [
            (
                (
                    "旋律在较高音区快速跑动，但听感不轻松",
                    "快而密，军鼓与底鼓重",
                    "失真吉他或合成器主音尖亮",
                    "金属或硬核里常见的压迫性段落",
                ),
                (
                    "高音区快速语汇但不甜美",
                    "双踩/密鼓、压迫",
                    "失真主音+厚墙吉他",
                    "金属/硬核式压迫",
                ),
            ),
        ],
        (1, 0): [
            (
                (
                    "旋律平稳、音程不大，氛围铺陈",
                    "慢至中速，正拍清楚但不猛",
                    "铺底与轻打击，和声不刺",
                    "环境音乐或新古典的平静段落",
                ),
                (
                    "小音程、氛围铺底",
                    "慢-中、稳拍不猛",
                    "铺底+轻打",
                    "环境/新古典的平静感",
                ),
            ),
        ],
        (1, 1): [
            (
                (
                    "旋律以级进与重复句为主，不强调欢快",
                    "中速四拍，groove 稳",
                    "吉他与键盘支起织体，鼓中等力度",
                    "独立流行或轻摇滚的稳态段",
                ),
                (
                    "级进+重复，情绪克制",
                    "中速、稳 groove",
                    "吉他/键盘+中等鼓",
                    "独立/轻摇稳态",
                ),
            ),
        ],
        (1, 2): [
            (
                (
                    "旋律句幅被节奏推着走，不突出大跳",
                    "中高速，镲与通鼓密度高",
                    "乐队整体齐奏感强，但不强调喜庆",
                    "高能量器乐的紧张推进段",
                ),
                (
                    "短句、被鼓推着走",
                    "中快、镲与通鼓密",
                    "齐奏、整体偏紧",
                    "高能量器乐的紧张感",
                ),
            ),
        ],
        (2, 0): [
            (
                (
                    "旋律大跳少、以温暖长音与琶音为主",
                    "慢至中速，swing 或反拍轻",
                    "电钢、原声吉他与轻弦乐",
                    "轻爵士、巴萨诺瓦或弛放气质",
                ),
                (
                    "长音+琶音、音域舒适",
                    "慢-中、反拍轻",
                    "电钢/原声+轻弦",
                    "巴萨诺瓦/弛放气质",
                ),
            ),
        ],
        (2, 1): [
            (
                (
                    "旋律明亮上行的乐句，句读清楚",
                    "中速、正拍积极，不拖沓",
                    "吉他清音与鼓组干脆",
                    "流行摇滚或放克的明朗段",
                ),
                (
                    "明亮相上行的短乐句",
                    "中速、正拍稳",
                    "清音吉他与干脆鼓",
                    "流行/放克明朗段",
                ),
            ),
        ],
        (2, 2): [
            (
                (
                    "旋律高能量、大跳与重复hook",
                    "快板，底鼓与军鼓强驱动",
                    "电吉他墙与主唱式旋律可辨",
                    "电子舞曲、摇滚副歌或节日感强的段落",
                ),
                (
                    "高能量重复 hook",
                    "快、强底鼓+军鼓",
                    "吉他墙+亮主音",
                    "舞曲/节日高潮段",
                ),
            ),
        ],
    }
    # 对未写满 3 变体的桶，取模
    key = (vb, ab)
    opts = p.get(key) or p[(1, 1)]
    choice = opts[t % len(opts)]
    return choice

def _build_v4(v: float, a: float, seg: int) -> str:
    v4q, _ = _quadrant_stock(seg, v, a)
    m, r, ins, g = [str(x) for x in v4q]
    vp, vextra = _valence_clauses(v)
    ar = _arousal_clause(a)
    fifth = f"{vp}，{ar}；{vextra}"
    s = f"旋律：{m}；节奏：{r}；乐器：{ins}；曲风：{g}；整体听感：{fifth}。"
    if len(s) > 220:
        s = f"旋律：{m}；节奏：{r}；乐器：{ins}；曲风：{g}；整体听感：{vp}，{ar}。"
    return s


def _build_v5(v: float, a: float, seg: int) -> str:
    _, v5q = _quadrant_stock(seg, v, a)
    m, r, ins, g = [str(x) for x in v5q]
    vp, vextra = _valence_clauses(v)
    ar = _arousal_clause(a)
    # vp 已以「听感…」起句，前不加「听感」避免重复
    s = f"{m}；{r}；{ins}；{g}；{vp}，{ar}（{vextra}）。"
    if len(s) > 220:
        s = f"{m}；{r}；{ins}；{g}；{vp}，{ar}。"
    return s


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = root / (cfg.get("output") or {}).get("descriptions_dir", "data/descriptions").lstrip("./")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = root / (cfg.get("output") or {}).get("splits_dir", "data/splits").lstrip("./") / "test_segments.csv"
    df = pd.read_csv(seg_path)
    for col in ("segment_id", "song_id", "audio_path", "valence_mean", "arousal_mean"):
        if col not in df.columns:
            raise RuntimeError(f"test_segments 缺列: {col}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ("test_descriptions_real_v4.csv", "test_descriptions_real_v5.csv"):
        p = out_dir / name
        if p.exists():
            bak = out_dir / f"{p.stem}_before_showcase_{ts}.csv"
            shutil.copy2(p, bak)
            print(f"[INFO] 已备份 {p.name} -> {bak.name}")

    rows_v4 = []
    rows_v5 = []
    for _, row in df.iterrows():
        seg = int(row["segment_id"])
        v = float(row["valence_mean"])
        a = float(row["arousal_mean"])
        rows_v4.append(
            {
                "segment_id": seg,
                "song_id": int(row["song_id"]),
                "audio_path": str(row["audio_path"]),
                "valence_mean": v,
                "arousal_mean": a,
                "description_raw": _build_v4(v, a, seg),
                "audio_description_version": "v4",
            }
        )
        rows_v5.append(
            {
                "segment_id": seg,
                "song_id": int(row["song_id"]),
                "audio_path": str(row["audio_path"]),
                "valence_mean": v,
                "arousal_mean": a,
                "description_raw": _build_v5(v, a, seg),
                "audio_description_version": "v5",
            }
        )

    pd.DataFrame(rows_v4).to_csv(out_dir / "test_descriptions_real_v4.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_v5).to_csv(out_dir / "test_descriptions_real_v5.csv", index=False, encoding="utf-8-sig")
    n = len(rows_v4)
    print(f"[OK] 已写入展示用描述各 {n} 条: test_descriptions_real_v4.csv, test_descriptions_real_v5.csv")
    print(
        "[NOTE] 本脚本按 DEAM 连续值**规则生成**的演示描述，与真实听感/转写可能不同；"
        "仅用于与真实 pipeline 的**对照说明**，写论文时请注明数据性质。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
