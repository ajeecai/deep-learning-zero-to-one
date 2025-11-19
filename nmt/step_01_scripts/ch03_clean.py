#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean and align bilingual corpus (English-Chinese)
for high-quality datasets like WMT17 or UNv1.0.

Usage:
    python scripts/02_clean.py \
        --en ../data/raw/news-commentary-v12.en-zh.en \
        --zh ../data/raw/news-commentary-v12.en-zh.zh \
        --out ../data/cleaned/news-commentary-v12.cleaned.tsv
"""

import argparse
import re
import sys
import opencc
from pathlib import Path

converter = opencc.OpenCC("t2s")


def remove_control_chars(text: str) -> str:
    """Remove control characters, including C0, C1, and Private Use Area."""
    # PUA: U+E000-U+F8FF
    return re.sub(r"[\x00-\x1F\x7F\x80-\x9F\uE000-\uF8FF]", "", text)


def basic_clean_pair(en: str, zh: str):
    """Basic cleaning for a pair: remove control chars and trim whitespace."""
    en = remove_control_chars(en.strip())
    zh = remove_control_chars(zh.strip())

    # Additional cleaning: remove inline newlines and tabs
    en = en.replace("\n", " ").replace("\t", " ")
    zh = zh.replace("\n", " ").replace("\t", " ")
    return en, zh


def valid_length(en: str, zh: str, min_len=3, max_len=100, ratio_threshold=3.0):
    """
    Filter pairs by length and length ratio constraints, but always return stats.
    Returns a tuple: (is_valid, reason, en_len, zh_len, ratio).
    """
    en_len, zh_len = len(en), len(zh)
    ratio = en_len / max(zh_len, 1)
    if en_len < min_len or zh_len < min_len:
        return False, "too_short", en_len, zh_len, ratio
    if en_len > max_len or zh_len > max_len:
        return False, "too_long", en_len, zh_len, ratio
    if ratio > ratio_threshold or ratio < 1 / ratio_threshold:
        return False, "length_ratio", en_len, zh_len, ratio
    return True, None, en_len, zh_len, ratio


# example usage:
# ./03_clean.py --en ../data/raw/decompressed/huggingface.co-UNv1.0.en-zh/UNv1.0.en-zh.en
#               --zh ../data/raw/decompressed/huggingface.co-UNv1.0.en-zh/UNv1.0.en-zh.zh \
#               --out-prefix ../data/cleaned/huggingface.co-UNv1.0 \
#               --min-len=3 --max-len=100 --ratio=5.5
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", required=True, help="path to English text file")
    parser.add_argument("--zh", required=True, help="path to Chinese text file")
    parser.add_argument(
        "--max-len",
        dest="max_len",
        type=int,
        required=True,
        help="max length of sentences",
    )
    parser.add_argument(
        "--min-len",
        dest="min_len",
        type=int,
        required=True,
        help="min length of sentences",
    )
    parser.add_argument(
        "--ratio", type=float, required=True, help="length ratio threshold"
    )
    parser.add_argument(
        "--out-prefix",
        dest="out_prefix",
        required=True,
        help="output path and prefix for cleaned tsv file",
    )

    parser.add_argument(
        "--t2s",
        action="store_true",
        help="convert traditional Chinese to simplified Chinese",
    )

    parser.add_argument(
        "--drop-debug",
        dest="drop_debug",
        action="store_true",
        help="If set, print detailed messages for each dropped sentence pair.",
    )
    args = parser.parse_args()

    en_path, zh_path, out_prefix = Path(args.en), Path(args.zh), Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_en_path = out_prefix.with_suffix(".en")
    out_zh_path = out_prefix.with_suffix(".zh")
    min_len, max_len, ratio, drop_debug = (
        args.min_len,
        args.max_len,
        args.ratio,
        args.drop_debug,
    )

    kept, dropped = 0, 0
    drop_reasons = {"too_short": 0, "too_long": 0, "length_ratio": 0}

    # --- Variables to track statistics ---
    # Overall stats (for all processed pairs)
    overall_total_en_chars, overall_total_zh_chars = 0, 0
    min_en_len, max_en_len = sys.maxsize, 0
    min_zh_len, max_zh_len = sys.maxsize, 0
    min_ratio, max_ratio = float("inf"), 0
    # Kept stats (for filtered pairs)
    kept_total_en_chars, kept_total_zh_chars = 0, 0

    total_pairs = 0
    with out_en_path.open("w", encoding="utf-8") as f_out_en, out_zh_path.open(
        "w", encoding="utf-8"
    ) as f_out_zh, en_path.open("r", encoding="utf-8") as f_in_en, zh_path.open(
        "r", encoding="utf-8"
    ) as f_in_zh:
        print("Processing sentence pairs...")
        # zip stops at the end of the shorter file
        for i, (en_line, zh_line) in enumerate(zip(f_in_en, f_in_zh)):
            total_pairs += 1
            en, zh = basic_clean_pair(en_line, zh_line)

            valid, reason, en_len, zh_len, current_ratio = valid_length(
                en, zh, min_len, max_len, ratio
            )

            # Update overall total character counts for every pair
            overall_total_en_chars += en_len
            overall_total_zh_chars += zh_len
            # Update min/max stats for ALL processed pairs
            min_en_len = min(min_en_len, en_len)
            max_en_len = max(max_en_len, en_len)
            min_zh_len = min(min_zh_len, zh_len)
            max_zh_len = max(max_zh_len, zh_len)
            min_ratio = min(min_ratio, current_ratio)
            max_ratio = max(max_ratio, current_ratio)

            if not valid:
                drop_reasons[reason] += 1
                dropped += 1
                if drop_debug and reason in ("too_short", "too_long", "length_ratio"):
                    # if reason in ("too_long", "length_ratio"):
                    print(
                        f"Dropping pair {i+1}: {reason}, en:{en_len} vs zh:{zh_len}, ratio:{current_ratio:.2f}\n\ten='{en}', zh='{zh}'"
                    )
                continue

            f_out_en.write(f"{en}\n")
            if args.t2s:
                zh = converter.convert(zh)
            f_out_zh.write(f"{zh}\n")
            kept += 1
            kept_total_en_chars += en_len
            kept_total_zh_chars += zh_len

    # --- Print summary statistics ---
    print("\n=== Cleaning Summary ===")
    print(f"Total pairs processed : {total_pairs:,}")
    print(f"Kept pairs            : {kept:,}")
    print(f"Dropped pairs         : {dropped:,}")
    print("Drop reasons:")
    for k, v in drop_reasons.items():
        print(f"  - {k:<12}: {v:,}")

    print("\n--- Overall Corpus Stats ---")
    print(f"Total EN chars (overall): {overall_total_en_chars:,}")
    print(f"Total ZH chars (overall): {overall_total_zh_chars:,}")
    overall_avg_en = overall_total_en_chars / total_pairs if total_pairs else 0
    overall_avg_zh = overall_total_zh_chars / total_pairs if total_pairs else 0
    print(f"Avg EN length (overall) : {overall_avg_en:.1f} chars")
    print(f"Avg ZH length (overall) : {overall_avg_zh:.1f} chars")
    print(f"Min/Max EN length (overall): {min_en_len} / {max_en_len}")
    print(f"Min/Max ZH length (overall): {min_zh_len} / {max_zh_len}")
    print(f"Min/Max ratio (overall): {min_ratio:.2f} / {max_ratio:.2f}")

    print("\n--- Kept Corpus Stats ---")
    kept_avg_en = kept_total_en_chars / kept if kept else 0
    kept_avg_zh = kept_total_zh_chars / kept if kept else 0
    print(f"Avg EN length (kept): {kept_avg_en:.1f} chars")
    print(f"Avg ZH length (kept): {kept_avg_zh:.1f} chars")

    print(f"\nCleaned corpus saved to: {out_en_path} and {out_zh_path}")


if __name__ == "__main__":
    main()
