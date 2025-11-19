#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim, signal
import sentencepiece as spm
import sys
import os, time
from tqdm import tqdm
from prompt_toolkit import prompt
import matplotlib.pyplot as plt
from .ch01_transformer_nmt import TranslationTransformer
from ..step_02_preprocess import create_dataloader
from ..utils.logger import setup_logger

# ===== 1. Define Hyperparameters =====

# --- Project Paths (Robust Path Handling) ---
# Define paths relative to this script's location to make them robust.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Tokenizer and Model Path ---
SPM_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "spm_model", "spm_bilingual.model")
MODEL_SAVE_PATH = os.path.join(
    PROJECT_ROOT, "data", "trained_models", "transformer_nmt.pth"
)
LOSS_PLOT_PATH = os.path.join(PROJECT_ROOT, "data", "trained_models", "loss_curve.png")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "logs", "training.log")

# --- Setup Logger ---
# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logger = setup_logger(log_file=LOG_FILE_PATH)
logger.info(f"Project root directory: {PROJECT_ROOT}")

# --- Data Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "bin")
SRC_SUFFIX = "en.ids"
TGT_SUFFIX = "zh.ids"

# --- Check for tokenizer model ---
if not os.path.exists(SPM_MODEL_PATH):
    raise FileNotFoundError(
        f"SentencePiece model not found at '{SPM_MODEL_PATH}'. "
        "Please ensure the model is trained and placed in the correct directory."
    )

sp = spm.SentencePieceProcessor()
sp.load(SPM_MODEL_PATH)

# --- Vocabulary Parameters (from tokenizer) ---
VOCAB_SIZE = sp.vocab_size()
PAD_ID = sp.pad_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

# --- Model Architecture Parameters ---
D_MODEL = 128  # Model dimension (embedding size), smaller for quick demo
NHEAD = 4  # Number of attention heads
NUM_ENCODER_LAYERS = 2  # Number of encoder layers
NUM_DECODER_LAYERS = 2  # Number of decoder layers
DIM_FEEDFORWARD = 512  # Dimension of the feedforward network
MAX_LEN = 100  # Maximum sequence length

# --- Device Setup ---
if torch.cuda.is_available():
    # Use the last available GPU
    DEVICE = f"cuda:{torch.cuda.device_count() - 1}"
else:
    DEVICE = "cpu"


# ===== 2. Instantiate Model, Loss Function, and Optimizer =====

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

# Loss function: CrossEntropyLoss, which ignores padding tokens.
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# --- Optimizer and LR Scheduler ---
# The original Transformer paper used a custom LR scheduler.
# Here we implement it using LambdaLR for more effective training.
optimizer = optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)


def get_lr_scheduler(optimizer, d_model, warmup_steps):
    """
    Implements the learning rate schedule from the "Attention Is All You Need" paper.
    lrate = d_model**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
    """

    def lr_lambda(current_step):
        current_step += 1  # Use 1-based indexing for steps
        arg1 = current_step**-0.5
        arg2 = current_step * (warmup_steps**-1.5)
        return (d_model**-0.5) * min(arg1, arg2)

    return LambdaLR(optimizer, lr_lambda)


def plot_loss_curve(losses, save_path, title="Training Loss Curve"):
    """Plots the training loss curve and saves it to a file."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)  # Ensure y-axis starts from 0
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss curve saved to {save_path}")


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    losses,
    interrupted=False,
):
    """Saves a training checkpoint."""
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    if interrupted:
        filename_suffix = f"_interrupted_epoch_{epoch+1}_{timestamp}.pth"
    else:
        filename_suffix = f"_epoch_{epoch+1}_{timestamp}.pth"

    save_path = MODEL_SAVE_PATH.replace(".pth", filename_suffix)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_losses": losses,
    }

    torch.save(checkpoint_data, save_path)
    logger.info(f"Checkpoint saved to {save_path}")
    return save_path


def worker_init_fn(worker_id):
    """Makes worker processes ignore KeyboardInterrupt signals."""
    # This is crucial for allowing the main process to handle Ctrl+C
    # without the DataLoader workers crashing.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ===== 4. Training Loop =====
def train(num_epochs, batch_size, num_workers, resumed_model, warmup_steps=4000):
    start_time = time.time()
    # Create the DataLoader to read from real files
    train_loader = create_dataloader(
        DATA_DIR,
        SRC_SUFFIX,
        TGT_SUFFIX,
        batch_size,
        PAD_ID,
        BOS_ID,
        EOS_ID,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer, D_MODEL, warmup_steps)

    start_epoch = 0
    epoch_losses = []

    # --- Check for checkpoint to resume training, with backward compatibility ---
    if resumed_model:
        # Construct the full path to the specified checkpoint file
        checkpoint_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), resumed_model)
        logger.info(f"Attempting to resume training from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # Load model, optimizer, and scheduler states directly from the checkpoint.
        logger.info("Loading checkpoint with model, optimizer, and scheduler states.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        epoch_losses = checkpoint.get("epoch_losses", [])

        logger.info(f"Resuming from epoch {start_epoch}.")

    else:
        if resumed_model:
            logger.error(f"Resume file '{resumed_model}' not found.")
        logger.info("Starting training from scratch.")

    model.train()  # Set the model to training mode

    # --- Check if training is already complete ---
    if start_epoch >= num_epochs:
        logger.warning(
            f"Training has already been completed up to epoch {start_epoch - 1}."
        )
        logger.warning(
            f"The target number of epochs is {num_epochs}. To train further, please increase the --num-epochs argument. Exiting."
        )
        return

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()

        # Custom bar format to remove the progress bar and default time stats,
        custom_bar_format = "{desc}: {n_fmt}it [{rate_fmt} {postfix}]"
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True,
            bar_format=custom_bar_format,  # Use custom format
        )
        i = 0
        for i, (src, tgt_in, tgt_out) in enumerate(pbar):
            try:
                src = src.to(DEVICE, non_blocking=True)
                tgt_in = tgt_in.to(DEVICE, non_blocking=True)
                tgt_out = tgt_out.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(src, tgt_in)
                loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                current_loss = loss.item()
                epoch_loss += current_loss

                elapsed_time = time.time() - epoch_start_time
                elapsed_time_str = (
                    f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
                )
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "samples": f"{(i + 1) * batch_size:,}",
                        "time": elapsed_time_str,
                    }
                )

            except KeyboardInterrupt:
                # Print a newline to cleanly break away from the tqdm progress bar line
                print()
                logger.warning("Training interrupted by user (Ctrl+C).")
                response = (
                    prompt(
                        "Save and exit (y/yes), exit without saving (n/no), or any other key to continue? [y/n/c]: "
                    )
                    .strip()
                    .lower()
                )

                if response in ("y", "yes"):
                    logger.info(f"Saving intermediate state for epoch {epoch+1}...")
                    save_checkpoint(
                        epoch,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch_losses,
                        interrupted=True,
                    )
                    sys.exit(0)
                elif response in ("n", "no"):
                    logger.info("Exiting without saving.")
                    sys.exit(0)
                else:  # 'c' or any other input
                    logger.info("Cancelling interruption. Resuming training...")
                    # Simply continue to the next batch in the loop
                    continue

        # This part is reached only if the inner loop completes without interruption
        if i > 0:  # Ensure the epoch ran for at least one step
            avg_epoch_loss = epoch_loss / (i + 1)
            epoch_losses.append(avg_epoch_loss)
            final_log = f"Epoch {epoch+1}/{num_epochs} finished. --> Average Epoch Loss: {avg_epoch_loss:.4f}"
            logger.info(final_log)

    end_time = time.time()
    logger.info(
        f"\nTraining finished. Total time: {end_time - start_time:.2f} seconds."
    )

    # --- Save the final model checkpoint ---
    if epoch_losses:  # Only save if at least one epoch completed
        final_model_save_path = save_checkpoint(
            num_epochs - 1,
            model,
            optimizer,
            lr_scheduler,
            epoch_losses,
        )

        # Plot and save the loss curve
        # Also save the loss plot with the same timestamp
        timestamp = (
            os.path.basename(final_model_save_path)
            .replace("transformer_nmt_", "")
            .replace(".pth", "")
        )
        timestamped_loss_plot_path = LOSS_PLOT_PATH.replace(".png", f"_{timestamp}.png")
        plot_loss_curve(epoch_losses, timestamped_loss_plot_path)
    else:
        logger.warning("No epochs were completed. No final model was saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer NMT model.")
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for training."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for DataLoader."
    )
    parser.add_argument(
        "--resumed-model",
        type=str,
        default=None,
        help="Name of the checkpoint file to resume training from (e.g., 'transformer_nmt_20231027153005.pth').",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=4000,
        help="Number of warmup steps for the learning rate scheduler.",
    )
    args = parser.parse_args()

    logger.info(f"Training on {DEVICE}...")
    train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resumed_model=args.resumed_model,
        warmup_steps=args.warmup_steps,
    )
