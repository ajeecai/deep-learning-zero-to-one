#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sentencepiece as spm
import os
import argparse
import math
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from .ch01_transformer_nmt import TranslationTransformer
from ..step_02_preprocess import create_dataloader
from ..utils.logger import setup_logger


def evaluate(
    model: TranslationTransformer,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    sp_model: spm.SentencePieceProcessor,
    device: torch.device,
):
    """
    Evaluates the model on a given dataset to calculate loss, perplexity, and BLEU score.

    Args:
        model: The trained TranslationTransformer model.
        loader: DataLoader for the evaluation dataset.
        loss_fn: The loss function (CrossEntropyLoss).
        sp_model: The SentencePiece processor.
        device: The device to run evaluation on.

    Returns:
        A tuple containing (average_loss, perplexity, bleu).
    """
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_tokens = 0
    hypotheses = []  # List to store model's translations
    references = []  # List to store ground truth translations

    with torch.no_grad():  # Disable gradient calculations
        pbar = tqdm(loader, desc="Evaluating")
        for src, tgt_in, tgt_out in pbar:
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)

            # --- Calculate Loss and Perplexity ---
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            # To calculate total loss correctly, we need to account for the number of non-padding tokens
            num_non_pad_tokens = (tgt_out != loss_fn.ignore_index).sum().item()
            total_loss += loss.item() * num_non_pad_tokens
            total_tokens += num_non_pad_tokens

            # --- Generate Translations for BLEU Score ---
            # Transpose src and tgt_out to be [Batch, SeqLen] for easier iteration
            src_transposed = src.transpose(0, 1)
            tgt_out_transposed = tgt_out.transpose(0, 1)

            for i in range(
                src_transposed.shape[0]
            ):  # Iterate through each sample in the batch
                # Generate hypothesis (translation)
                # Note: This uses a simple greedy decoding. For better results, beam search is recommended.
                memory = model.transformer.encoder(
                    model.pos_encoder(
                        model.token_embed(src[:, i : i + 1]) * (model.d_model**0.5)
                    )
                )
                ys = (
                    torch.ones(1, 1)
                    .fill_(sp_model.bos_id())
                    .type(torch.long)
                    .to(device)
                )
                for _ in range(100):
                    tgt_mask = model._generate_square_subsequent_mask(
                        ys.size(0), device
                    )
                    out = model.transformer.decoder(
                        model.pos_encoder(model.token_embed(ys) * (model.d_model**0.5)),
                        memory,
                        tgt_mask,
                    )
                    prob = model.fc_out(out[-1, :])
                    _, next_word = torch.max(prob, dim=1)
                    next_word = next_word.item()
                    ys = torch.cat(
                        [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
                    )
                    if next_word == sp_model.eos_id():
                        break

                hyp_ids = ys.squeeze().tolist()
                hypotheses.append(sp_model.decode(hyp_ids).split())

                # Prepare reference
                ref_ids = tgt_out_transposed[i].tolist()
                # Remove padding tokens for accurate BLEU score
                ref_ids = [tok for tok in ref_ids if tok != loss_fn.ignore_index]
                references.append(
                    [sp_model.decode(ref_ids).split()]
                )  # Note: bleu_score expects a list of references

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Calculate BLEU score
    # The score is calculated over the entire corpus, not per sentence.
    bleu = bleu_score(hypotheses, references)

    return avg_loss, perplexity, bleu


def main():
    # --- Project Paths (Robust Path Handling) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "logs", "evaluation.log")

    # --- Setup Logger ---
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logger = setup_logger(name="Evaluate", log_file=LOG_FILE_PATH)

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "trained_models", "transformer_nmt.pth"),
        help="Path to the trained model (.pth) file.",
    )
    parser.add_argument(
        "--spm-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "spm_model", "spm_bilingual.model"),
        help="Path to the SentencePiece model file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "bin"),
        help="Path to the directory with tokenized evaluation data (e.g., dev set).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation."
    )
    args = parser.parse_args()

    # --- Setup ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_path)
    logger.info("SentencePiece model loaded.")

    # --- Model Hyperparameters (must match the trained model) ---
    VOCAB_SIZE = sp.vocab_size()
    PAD_ID = sp.pad_id()
    D_MODEL = 128
    NHEAD = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    DIM_FEEDFORWARD = 512
    MAX_LEN = 100

    # Instantiate and load the model
    model = TranslationTransformer(
        vocab_size=VOCAB_SIZE,
        pad_index=PAD_ID,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_len=MAX_LEN,
    ).to(DEVICE)

    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Trained model weights loaded from checkpoint.")

    # --- Data Loading ---
    # Assuming your evaluation files have the same suffix as training files
    eval_loader = create_dataloader(
        args.data_path,
        "en.ids",
        "zh.ids",
        args.batch_size,
        PAD_ID,
        sp.bos_id(),
        sp.eos_id(),
        num_workers=4,
    )

    # --- Evaluation ---
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    avg_loss, perplexity, bleu = evaluate(model, eval_loader, loss_fn, sp, DEVICE)

    # --- Print Results ---
    logger.info(
        "\n"
        + "=" * 30
        + "\n      EVALUATION RESULTS"
        + "\n"
        + "=" * 30
        + f"\n      Average Loss: {avg_loss:.4f}"
        + f"\n        Perplexity: {perplexity:.2f}"
        + f"\n        BLEU Score: {bleu*100:.2f}"
        + "\n"
        + "=" * 30
    )


if __name__ == "__main__":
    main()
