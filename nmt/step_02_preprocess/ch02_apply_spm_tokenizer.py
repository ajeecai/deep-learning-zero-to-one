#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sentencepiece as spm
import argparse
from typing import List
from pathlib import Path


def encode_file(model_path, input_path, output_path, to_str=False, batch_size=1000):
    """Tokenize a file using SentencePiece, with optional batching for speed."""
    sp = spm.SentencePieceProcessor(model_file=model_path)
    # By default (to_str=False), output integer IDs.
    output_type = str if to_str else int

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        lines_batch: List[str] = []
        for line in fin:
            stripped_line = line.strip()
            lines_batch.append(stripped_line)

            if len(lines_batch) == batch_size:
                # Process a batch of lines
                tokenized_batch = sp.encode(lines_batch, out_type=output_type)
                for tokens in tokenized_batch:
                    fout.write(" ".join(map(str, tokens)) + "\n")
                lines_batch.clear()

        # Process any remaining lines in the last batch
        if lines_batch:
            tokenized_batch = sp.encode(lines_batch, out_type=output_type)
            for tokens in tokenized_batch:
                fout.write(" ".join(map(str, tokens)) + "\n")


""" usage: 
        python preprocess/apply_spm.py \
            --model ../data/spm_model/spm_bilingual.model \
            --input ../data/clean/train.en \
            --output ../data/bin/train.en.ids
        # To get subword strings instead of IDs, add --to_str
        # --output ../data/bin/train.en.str --to_str
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a SentencePiece model to a file or a directory of files."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input", required=True, help="Input file or directory to tokenize."
    )
    parser.add_argument("--output", required=True, help="Output file or directory.")
    parser.add_argument(
        "--to_str",
        action="store_true",
        help="If set, output subword strings for debug only. Otherwise, output integer IDs (default).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of lines to process in a batch.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        print(f"[INFO] Input is a directory. Processing all .en and .zh files.")
        output_path.mkdir(parents=True, exist_ok=True)

        files_to_process = list(input_path.glob("*.en")) + list(input_path.glob("*.zh"))
        if not files_to_process:
            print(f"[WARNING] No .en or .zh files found in {input_path}")

        for file_in in files_to_process:
            # Correctly append a suffix, not replace it.
            suffix_to_add = ".str" if args.to_str else ".ids"
            file_out = output_path / (file_in.name + suffix_to_add)
            print(f"  - Tokenizing {file_in} -> {file_out}")
            encode_file(
                args.model, str(file_in), str(file_out), args.to_str, args.batch_size
            )
    elif input_path.is_file():
        print(
            f"[INFO] Input is a single file. Tokenizing {input_path} -> {output_path}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_file(
            args.model, str(input_path), str(output_path), args.to_str, args.batch_size
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
