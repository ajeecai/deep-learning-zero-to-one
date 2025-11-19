#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sentencepiece as spm
import time
import argparse
import math
from pathlib import Path
from typing import List
from collections import Counter


def analyze_spm(model_path, corpus_path, suffixes, top_k=50, batch_size=1000):
    start_time = time.monotonic()
    print(f"🔍 Loading model: {model_path}")
    sp = spm.SentencePieceProcessor(model_file=model_path)
    vocab_size = sp.vocab_size()
    print(f"✅ Vocab size: {vocab_size:,}\n")

    # Read corpus and tokenize
    print(f"📖 Reading corpus: {corpus_path}")
    p = Path(corpus_path)
    if p.is_dir():
        print(
            f"[INFO] Input is a directory. Searching for files with suffixes: {suffixes}"
        )
        files_to_read = []
        for suffix in suffixes.split(","):
            files_to_read.extend(p.glob(f"*.{suffix.strip()}"))
    elif p.is_file():
        files_to_read = [p]
    else:
        raise FileNotFoundError(
            f"Input path does not exist or is not a file/directory: {corpus_path}"
        )

    if not files_to_read:
        print(f"[WARNING] No files found to analyze in {corpus_path}")
        return

    token_counter = Counter()
    total_lines = 0
    total_words = 0  # Based on language-specific splitting
    batch: List[str] = []

    def process_batch(batch_to_process: List[str], lang_is_chinese: bool):
        nonlocal total_lines, total_words, token_counter
        if not batch_to_process:
            return

        if lang_is_chinese:
            total_words += sum(
                len(s) for s in batch_to_process
            )  # Count chars for Chinese
        else:
            total_words += sum(
                len(s.split()) for s in batch_to_process
            )  # Count space-separated words for others
        total_lines += len(batch_to_process)
        tokenized_batch = sp.encode(batch_to_process, out_type=int)
        for tokens in tokenized_batch:
            token_counter.update(tokens)

    for file_path in files_to_read:
        print(f"  - Reading from {file_path.name}...")
        # Determine the word counting strategy for this file
        is_chinese = file_path.name.endswith(".zh")

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                batch.append(stripped_line)
                if len(batch) >= batch_size:
                    process_batch(batch, is_chinese)
                    batch.clear()
                    if total_lines % (batch_size * 20) == 0:
                        print(f"    Processed {total_lines:,} lines...")
        # Process the final batch for the current file
        process_batch(batch, is_chinese)
        batch.clear()

    if total_lines == 0:
        print("\n[ERROR] No content found in the provided corpus.")
        return

    total_tokens = sum(token_counter.values())
    print(f"\n🧮 Total tokens: {total_tokens:,}")

    # High and low-frequency analysis
    freq_values = list(token_counter.values())
    zero_freq = vocab_size - len(token_counter)
    low_freq = sum(1 for v in freq_values if v < 5)
    low_ratio = (low_freq + zero_freq) / vocab_size

    print(f"📊 Token coverage:")
    print(f"  - Unique tokens seen in corpus: {len(token_counter):,} / {vocab_size:,}")
    print(f"  - Ratio of low-frequency tokens (< 5 occurrences): {low_ratio*100:.2f}%")
    print(f"  - Unused tokens in vocabulary: {zero_freq:,}")

    # Average sentence length
    avg_token_len = total_tokens / total_lines if total_lines > 0 else 0
    avg_word_len = total_words / total_lines if total_lines > 0 else 0
    granularity_r = total_tokens / total_words if total_words > 0 else 0
    print(
        f"📏 Average sentence length (in 'words'): {avg_word_len:.2f} (Note: ZH counts chars, EN counts space-split words)"
    )
    print(f"📏 Average sentence length (in tokens): {avg_token_len:.2f}")
    print(f"⚙️  Granularity (tokens/'word'): {granularity_r:.2f}\n")

    # High-frequency token examples
    print(f"🔥 Top {top_k} most frequent tokens:")
    for tid, freq in token_counter.most_common(top_k):
        print(f"  {sp.id_to_piece(tid):<10} {freq}")

    # Token frequency coverage
    sorted_freq = sorted(freq_values, reverse=True)

    def cov(n):
        return sum(sorted_freq[:n]) / total_tokens * 100

    print("\n📈 Corpus Coverage by Top Tokens:")
    for n in [1000, 5000, 10000, 20000]:
        if n < vocab_size:
            print(f"  - Top {n:>5,d} tokens cover {cov(n):.2f}% of the corpus")

    duration = time.monotonic() - start_time
    print(f"\n⏱️ Analysis finished in {duration:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to spm model")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to a representative corpus file or directory.",
    )
    parser.add_argument(
        "--suffixes",
        default="en,zh,txt",
        help="Comma-separated file suffixes to look for in the corpus directory (e.g., 'en,zh').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Number of top frequent tokens to show."
    )
    args = parser.parse_args()
    analyze_spm(
        args.model,
        args.corpus,
        args.suffixes,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
