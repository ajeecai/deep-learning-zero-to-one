#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import sentencepiece as spm
import os
from prompt_toolkit import prompt
import argparse
from .ch01_transformer_nmt import TranslationTransformer
from ..utils.logger import setup_logger


def translate_sentence(
    model: TranslationTransformer,
    sentence: str,
    sp_model: spm.SentencePieceProcessor,
    device: torch.device,
    max_len: int,
    bos_id: int,
    eos_id: int,
) -> str:
    """
    Translates a Chinese source sentence into the English target language.

    Args:
        model: The trained TranslationTransformer model.
        sentence: The source sentence string.
        sp_model: The SentencePiece processor.
        device: The device to run inference on (e.g., 'cuda' or 'cpu').
        max_len: The maximum length of the generated translation.
        bos_id: The ID for the beginning-of-sentence token.
        eos_id: The ID for the end-of-sentence token.

    Returns:
        The translated sentence string.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        # 1. Tokenize the source sentence and add special tokens
        src_tokens = [bos_id] + sp_model.encode_as_ids(sentence) + [eos_id]
        src_tensor = torch.tensor(
            src_tokens, dtype=torch.long, device=device
        ).unsqueeze(
            1
        )  # Shape: [S, 1]

        # 2. Initialize the target sequence with the BOS token
        tgt_in_tensor = torch.tensor(
            [bos_id], dtype=torch.long, device=device
        ).unsqueeze(
            1
        )  # Shape: [1, 1]

        # 3. Autoregressive decoding loop (Greedy Search)
        for _ in range(max_len):
            # Forward pass through the model
            # current_T are all in logits due to FFN output, even current_T - 1 are known
            output_logits = model(src_tensor, tgt_in_tensor)  # Shape: [current_T, 1, V]

            # Get the logits for the very last token predicted
            last_token_logits = output_logits[-1, 0, :]  # Shape: [V]

            # Predict the next token ID by finding the one with the highest score
            next_token_id = last_token_logits.argmax(dim=-1).item()

            # Append the predicted token to the target input sequence for the next iteration
            tgt_in_tensor = torch.cat(
                [
                    tgt_in_tensor,
                    torch.tensor([[next_token_id]], dtype=torch.long, device=device),
                ],
                dim=0,
            )

            # If the model predicts the EOS token, stop decoding
            if next_token_id == eos_id:
                break

        # 4. Detokenize the generated sequence
        generated_ids = tgt_in_tensor.squeeze(1).tolist()

        # Remove BOS token from the start
        if generated_ids and generated_ids[0] == bos_id:
            generated_ids = generated_ids[1:]
        # Remove EOS token from the end if it exists
        if generated_ids and generated_ids[-1] == eos_id:
            generated_ids = generated_ids[:-1]

        translated_sentence = sp_model.decode(generated_ids)
        return translated_sentence


def main():
    # --- Project Paths (Robust Path Handling) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (nmt) by going up two levels from the script's location
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "trained_models")
    LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "logs", "translation.log")

    # --- Setup Logger ---
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logger = setup_logger(name="Translate", log_file=LOG_FILE_PATH)

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Interactive Translation with Transformer"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="transformer_nmt.pth",
        help="Path or filename of the model. If only a filename is provided, it's loaded from the default models directory.",
    )
    parser.add_argument(
        "--spm-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "spm_model", "spm_bilingual.model"),
        help="Path to the SentencePiece model file.",
    )
    args = parser.parse_args()

    # --- Construct the final model path ---
    model_path_arg = args.model_path
    # If the provided path is just a filename (no directory separators),
    # prepend the default model directory path.
    if os.path.dirname(model_path_arg) == "":
        model_path_arg = os.path.join(DEFAULT_MODEL_DIR, model_path_arg)

    # --- Setup ---
    if torch.cuda.is_available():
        # Use the last available GPU
        DEVICE = f"cuda:{torch.cuda.device_count() - 1}"
    else:
        DEVICE = "cpu"

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
    checkpoint = torch.load(model_path_arg, map_location=DEVICE)

    # Load model state directly from the checkpoint dictionary.
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Trained model weights loaded from checkpoint.")

    # --- Interactive Loop ---
    print("\nEnter an English sentence for translation (type 'quit' to exit).")
    while True:
        try:
            input_sentence = prompt("EN > ")
            if input_sentence.lower() == "quit":
                break
            translated_output = translate_sentence(
                model, input_sentence, sp, DEVICE, MAX_LEN, sp.bos_id(), sp.eos_id()
            )
            print(f"ZH < {translated_output}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
