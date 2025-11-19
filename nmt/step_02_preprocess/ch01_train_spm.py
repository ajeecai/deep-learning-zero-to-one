#!/usr/bin/env python3
"""
train_spm.py
------------
Train a SentencePiece tokenizer (BPE or Unigram) from bilingual corpus.

Usage:
  python preprocess/train_spm.py \
    --input data/clean/ \
    --output_dir data/spm_model \
    --vocab_size 32000 \
    --model_type unigram \
    --character_coverage 0.9995 \
    --sample_ratio 0.05 \
    --seed 42

If --sample_ratio=1.0 (default), use full corpus.
"""

import argparse
import os
import random
from pathlib import Path
import time
import sentencepiece as spm


def _count_lines_fast(file_path: Path) -> int:
    """Quickly count lines in a file by reading in binary chunks."""
    count = 0
    with file_path.open("rb") as f:
        # Read in 1MB chunks
        chunk_size = 1024 * 1024
        while chunk := f.read(chunk_size):
            count += chunk.count(b"\n")
    return count


def sample_corpus(input_path, sample_ratio=1.0, seed=42, tmp_path=None, max_lines=None):
    """Return path to sampled file (or original if no sampling)."""
    p = Path(input_path)
    if p.is_dir():
        print(f"[INFO] Reading files from directory: {p}")
        files_to_read = (
            list(p.glob("*.en")) + list(p.glob("*.zh")) + list(p.glob("*.txt"))
        )
    elif p.is_file():
        print(f"[INFO] Reading from single file: {p}")
        files_to_read = [p]
    else:
        raise FileNotFoundError(
            f"Input path does not exist or is not a file/directory: {input_path}"
        )

    # If no sampling is needed, return a comma-separated string of all found files.
    if sample_ratio >= 1.0 and not max_lines:
        print("[INFO] Using full corpus for SentencePiece training.")
        # SentencePiece's --input expects a comma-separated list of filenames.
        return ",".join(map(str, files_to_read))

    if tmp_path is None:
        tmp_path = str(p) + ".sampled"

    # Use the fast line counting method
    print("[INFO] Counting total lines (fast method)...")
    total_lines = sum(_count_lines_fast(fp) for fp in files_to_read)
    print(f"[INFO] Found {total_lines} total lines in {len(files_to_read)} file(s).")

    # --- Step 2: Determine number of lines to sample ---
    # Calculate lines from sample_ratio, then cap it with max_lines if provided.
    k = int(total_lines * sample_ratio)
    if max_lines is not None:
        k = min(k, max_lines)

    # --- Step 3: Perform Reservoir Sampling ---
    print(f"[INFO] Sampling {k} lines using Reservoir Sampling ...")
    random.seed(seed)
    reservoir = []
    lines_seen = 0
    for file_path in files_to_read:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                lines_seen += 1
                if len(reservoir) < k:
                    reservoir.append(line)
                else:
                    # Replace elements with decreasing probability to achieve k/N equallly sampling
                    j = random.randint(0, lines_seen - 1)
                    if j < k:
                        reservoir[j] = line

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(reservoir)

    print(
        f"[INFO] Sampled {k} / {total_lines} lines ({100*k/total_lines:.2f}%) to {tmp_path}"
    )
    return tmp_path


def train_spm(
    input_file,
    model_prefix,
    vocab_size=32000,
    character_coverage=0.9995,
    model_type="unigram",
):
    """Train SentencePiece model."""
    cmd = (
        f"--input={input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--model_type={model_type} "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        f"--unk_piece=<unk> --bos_piece=<s> --eos_piece=</s> "
    )

    print(f"[INFO] Training SentencePiece with command:\nspm_train {cmd}\n")
    spm.SentencePieceTrainer.Train(cmd)
    print(f"[INFO] SentencePiece model saved to {model_prefix}.model / .vocab")


""" Usage:
        ./01_train_spm.py --input ../data/cleaned/txt/
"""


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer.")
    parser.add_argument(
        "--input", required=True, help="Clean text file or directory for training"
    )
    parser.add_argument(
        "--output_dir", default="../data/spm_model", help="Output directory"
    )
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", choices=["bpe", "unigram"], default="unigram")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of corpus to sample (0-1)",
    )
    parser.add_argument("--max_lines", type=int, default=None, help="Max sampled lines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Optional Corpus Sampling ---
    print("\n--- Step 1: Optional Corpus Sampling ---")
    start_time = time.monotonic()
    sampled_path = sample_corpus(
        input_path=args.input,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        tmp_path=os.path.join(args.output_dir, "sampled.txt"),
        max_lines=args.max_lines,
    )
    sampling_duration = time.monotonic() - start_time
    print(f"[INFO] Corpus sampling took {sampling_duration:.2f} seconds.")

    # --- Step 2: Train SentencePiece Model ---
    print("\n--- Step 2: Train SentencePiece Model ---")
    start_time = time.monotonic()
    model_prefix = os.path.join(args.output_dir, "spm_bilingual")
    train_spm(
        input_file=sampled_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )
    training_duration = time.monotonic() - start_time
    print(f"[INFO] SentencePiece training took {training_duration:.2f} seconds.")

    # --- Step 3: Print Model Stats ---
    print("\n--- Step 3: Trained Model Stats ---")
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    print(f"Vocabulary size: {sp.vocab_size()}")
    print("Special tokens:")
    print(f"  - PAD: id={sp.pad_id()}, piece='{sp.id_to_piece(sp.pad_id())}'")
    print(f"  - UNK: id={sp.unk_id()}, piece='{sp.id_to_piece(sp.unk_id())}'")
    print(f"  - BOS: id={sp.bos_id()}, piece='{sp.id_to_piece(sp.bos_id())}'")
    print(f"  - EOS: id={sp.eos_id()}, piece='{sp.id_to_piece(sp.eos_id())}'")

    # --- Vocabulary Samples ---
    print("\n--- Vocabulary Samples ---")
    vocab_size = sp.vocab_size()
    # Print first 20 tokens
    print("First 20 tokens:")
    for i in range(min(20, vocab_size)):
        print(f"  id={i:<5} piece='{sp.id_to_piece(i)}' (score={sp.get_score(i):.4f})")

    # Print last 10 tokens if vocab is large enough
    if vocab_size > 30:
        print("\nLast 10 tokens:")
        for i in range(vocab_size - 10, vocab_size):
            print(
                f"  id={i:<5} piece='{sp.id_to_piece(i)}' (score={sp.get_score(i):.4f})"
            )


if __name__ == "__main__":
    main()
