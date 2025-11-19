#!/usr/bin/env python3
"""Compute length distributions and ratio plot for parallel corpora.

Usage examples:
    # Generate all plots (default) from separate files
    python scripts/02_stats.py --en ../data/raw/UNv1.0.en-zh.en --zh ../data/raw/UNv1.0.en-zh.zh
    # Generate only the histogram plot from a TSV file
    python scripts/02_stats.py --tsv ../data/cleaned/news-commentary-v12.cleaned.tsv --plot hist

Outputs:
 - {prefix}_en_zh_dist.png : KDE plot of lengths for EN and ZH
 - {prefix}_en_zh_hist.png : Histogram of lengths for EN and ZH
 - {prefix}_ratio.png      : KDE plot of en_len / zh_len ratio
"""
import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import numpy as np

# --- Global Matplotlib Settings ---
# Enable minor ticks on both axes for all subsequent plots
matplotlib.rcParams.update({"xtick.minor.visible": True, "ytick.minor.visible": True})


def read_parallel_from_files(en_path, zh_path):
    with open(en_path, "r", encoding="utf-8") as fe, open(
        zh_path, "r", encoding="utf-8"
    ) as fz:
        for enl, zhl in zip(fe, fz):
            enl = enl.strip()
            zhl = zhl.strip()
            if not enl or not zhl:
                continue
            yield enl, zhl


def read_parallel_from_tsv(tsv_path):
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            enl, zhl = parts[0].strip(), parts[1].strip()
            if not enl or not zhl:
                continue
            yield enl, zhl


def compute_lengths(pairs):
    en_lens = []
    zh_lens = []
    ratios = []
    for enl, zhl in pairs:
        le = len(enl)
        lz = len(zhl)
        en_lens.append(le)
        zh_lens.append(lz)
        ratios.append(le / max(1, lz))
    return en_lens, zh_lens, ratios


def apply_custom_grid(ax):
    """Apply a custom grid style with visible major and minor grid lines."""
    # Enable grid for both major and minor ticks with different styles
    ax.grid(which="major", linestyle="-", linewidth="0.5", alpha=1.0)
    ax.grid(which="minor", linestyle="-", linewidth="0.5", alpha=1.0)


def plot_lengths_kde(en_lens, zh_lens, out_prefix="lengths"):
    # KDE (smooth density) plot for EN and ZH lengths using seaborn
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.kdeplot(en_lens, label="EN", fill=False, ax=ax, cut=0)
    sns.kdeplot(zh_lens, label="ZH", fill=False, ax=ax, cut=0)

    ax.set_xlabel("Length (characters) (99.5% capped)")
    ax.set_ylabel("Density")
    ax.set_title("Length distribution: EN vs ZH")
    ax.legend()
    ax.set_xlim(0, np.percentile(en_lens + zh_lens, 99.5))
    ax.set_ylim(bottom=-0.001)
    apply_custom_grid(ax)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_en_zh_kde.png")
    plt.close(fig)


def plot_lengths_hist(en_lens, zh_lens, out_prefix="lengths"):
    """Plot a histogram of sentence lengths for both languages."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use a reasonable number of bins.
    # Using element="step" is good for comparing distributions.
    sns.histplot(en_lens, label="EN", ax=ax, element="step", fill=False, bins=100)
    sns.histplot(zh_lens, label="ZH", ax=ax, element="step", fill=False, bins=100)

    ax.set_xlabel("Length (characters) (99.5% capped)")
    ax.set_ylabel("Count")
    ax.set_title("Length Distribution (Histogram): EN vs ZH")
    ax.legend()
    ax.set_ylim(bottom=-100)
    # Focus on the main distribution
    ax.set_xlim(0, np.percentile(en_lens + zh_lens, 99.5))
    apply_custom_grid(ax)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_en_zh_hist.png")
    plt.close(fig)


def plot_ratio_hist(ratios, out_prefix="lengths"):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(ratios, fill=False, ax=ax, bins=100, element="step")

    ax.set_xlabel("EN length / ZH length (99.5% capped)")
    ax.set_ylabel("Ratio")
    ax.set_title("Length ratio distribution (EN/ZH)")
    ax.set_xlim(0.0001, np.percentile(ratios, 99.5) if ratios else 5)
    ax.set_ylim(bottom=-0.01)

    apply_custom_grid(ax)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_ratio.png")
    plt.close(fig)


# example usage:
#   ./02_stats.py --en ../data/raw/decompressed/object.pouta.csc.fi-OpenSubtitles-v2018.en-zh_cn/OpenSubtitles.en-zh_cn.en
#                 --zh ../data/raw/decompressed/object.pouta.csc.fi-OpenSubtitles-v2018.en-zh_cn/OpenSubtitles.en-zh_cn.zh
#                 --out-prefix ../data/cleaned/object.pouta.csc.fi-Tatoeba-Challenge.eng-zho --plot all
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", help="English file (one sentence per line)")
    parser.add_argument("--zh", help="Chinese file (one sentence per line)")
    parser.add_argument("--tsv", help="TSV file with en\tzh per line")
    parser.add_argument(
        "-o",
        "--out-prefix",
        dest="out_prefix",
        default="stats",
        help="Output prefix for images (default: stats)",
    )
    parser.add_argument(
        "--plot",
        choices=["kde", "hist", "all"],
        default="all",
        help="Type of length distribution plot to generate: 'kde', 'hist', or 'all'. Default is 'all'.",
    )
    args = parser.parse_args()

    if args.tsv:
        pairs = list(read_parallel_from_tsv(args.tsv))
    elif args.en and args.zh:
        pairs = list(read_parallel_from_files(args.en, args.zh))
    else:
        parser.error("Provide either --tsv or both --en and --zh")

    en_lens, zh_lens, ratios = compute_lengths(pairs)
    print(f"Read {len(en_lens):,} parallel pairs")

    if args.plot in ["kde", "all"]:
        print("Generating KDE plot for length distribution...")
        plot_lengths_kde(en_lens, zh_lens, out_prefix=args.out_prefix)
    if args.plot in ["hist", "all"]:
        print("Generating histogram for length distribution...")
        plot_lengths_hist(en_lens, zh_lens, out_prefix=args.out_prefix)

    plot_ratio_hist(ratios, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
