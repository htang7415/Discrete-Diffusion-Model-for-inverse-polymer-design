#!/usr/bin/env python
"""Figure C: Bi-Diffusion generation tradeoff.

Reads metrics_generation.csv and creates a diffusion-only overview:
- Validity (two-star) by representation
- Throughput (valid samples per second) by representation
- Diversity by representation
- Validity vs throughput tradeoff scatter
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "method" in df.columns:
        df = df[df["method"] == "Bi_Diffusion"].copy()
    grouped = (
        df.groupby("representation", as_index=False)
        .agg(
            validity_two_stars=("validity_two_stars", "mean"),
            valid_per_sec=("valid_per_sec", "mean"),
            avg_diversity=("avg_diversity", "mean"),
            novelty=("novelty", "mean"),
        )
        .sort_values("representation")
    )
    return grouped


def plot_diffusion_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    """Create diffusion-only tradeoff plots."""
    grouped = _prepare(df)
    if grouped.empty:
        print("No Bi_Diffusion generation data found. Skipping plot.")
        return

    reps = grouped["representation"].tolist()
    x = range(len(reps))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    bar_colors = [colors[i % len(colors)] for i in x]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.bar(x, grouped["validity_two_stars"], color=bar_colors)
    ax.set_title("Validity (Two Stars)")
    ax.set_ylabel("Fraction")
    ax.set_xticks(list(x))
    ax.set_xticklabels(reps, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(x, grouped["valid_per_sec"], color=bar_colors)
    ax.set_title("Throughput")
    ax.set_ylabel("Valid Samples / Second")
    ax.set_xticks(list(x))
    ax.set_xticklabels(reps, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.bar(x, grouped["avg_diversity"], color=bar_colors)
    ax.set_title("Diversity")
    ax.set_ylabel("Average Diversity")
    ax.set_xticks(list(x))
    ax.set_xticklabels(reps, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    for _, row in grouped.iterrows():
        ax.scatter(row["valid_per_sec"], row["validity_two_stars"], s=110)
        ax.annotate(
            row["representation"],
            (row["valid_per_sec"], row["validity_two_stars"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_title("Tradeoff: Throughput vs Validity")
    ax.set_xlabel("Valid Samples / Second")
    ax.set_ylabel("Validity (Two Stars)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure C to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure C: Bi-Diffusion tradeoff")
    parser.add_argument(
        "--input",
        type=str,
        default="results/aggregate/metrics_generation.csv",
        help="Path to metrics_generation.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/figure_c_diffusion_tradeoff.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_diffusion_tradeoff(df, output_path)


if __name__ == "__main__":
    main()
