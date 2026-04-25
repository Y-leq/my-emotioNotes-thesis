"""
将 DEAM 连续 Valence/Arousal (1~9) 映射为离散情感标签。

与 traditional_baseline_train_eval、eval_llm_predictions 共用，保证对比公平。
"""


def map_valence_arousal_to_label(valence: float, arousal: float) -> str:
    val_low = 4.5
    val_high = 6.0
    ar_low = 4.5
    ar_high = 6.0

    is_val_low = valence <= val_low
    is_val_high = valence >= val_high
    is_ar_low = arousal <= ar_low
    is_ar_high = arousal >= ar_high

    if is_ar_high:
        if is_val_high:
            return "快乐"
        if is_val_low:
            return "紧张"
        return "激昂"

    if is_ar_low:
        if is_val_high:
            return "放松"
        if is_val_low:
            return "悲伤"
        return "平静"

    if is_val_high:
        return "放松"
    if is_val_low:
        return "紧张"
    return "平静"
