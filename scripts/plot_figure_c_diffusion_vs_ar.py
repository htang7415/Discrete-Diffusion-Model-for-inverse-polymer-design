#!/usr/bin/env python
"""Figure C: Diffusion vs AR Tradeoff.

Reads metrics_generation.csv and creates visualization comparing:
- Validity vs throughput (valid samples per second)
- Diffusion vs AR across representations
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_diffusion_vs_ar(df: pd.DataFrame, output_path: Path) -> None:
    """Create diffusion vs AR comparison plots."""
    if df.empty:
        print("No generation data found. Skipping plot.")
        return

    # Group by method and representation
    grouped = df.groupby(['method', 'representation']).agg({
        'validity': 'mean',
        'validity_two_stars': 'mean',
        'novelty': 'mean',
        'uniqueness': 'mean',
        'avg_diversity': 'mean',
        'valid_per_sec': 'mean',
        'mean_sa': 'mean'
    }).reset_index()

    # Separate Diffusion and AR methods
    diffusion = grouped[grouped['method'] == 'Bi_Diffusion']
    ar = grouped[grouped['method'] == 'AR_Transformer']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color mapping for representations
    rep_colors = {
        'SMILES': '#1f77b4',
        'SELFIES': '#ff7f0e',
        'Group_SELFIES': '#2ca02c',
        'Graph': '#d62728'
    }

    # Plot 1: Validity (2-star) comparison
    ax = axes[0, 0]
    reps = sorted(set(grouped['representation']))
    x = np.arange(len(reps))
    width = 0.35

    diff_vals = [diffusion[diffusion['representation'] == r]['validity_two_stars'].values[0]
                 if len(diffusion[diffusion['representation'] == r]) > 0 else 0 for r in reps]
    ar_vals = [ar[ar['representation'] == r]['validity_two_stars'].values[0]
               if len(ar[ar['representation'] == r]) > 0 else 0 for r in reps]

    ax.bar(x - width/2, diff_vals, width, label='Bi_Diffusion', color='#1f77b4')
    ax.bar(x + width/2, ar_vals, width, label='AR_Transformer', color='#ff7f0e')
    ax.set_xlabel('Representation')
    ax.set_ylabel('Validity (2-star)')
    ax.set_title('Validity: Diffusion vs AR')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Throughput comparison
    ax = axes[0, 1]
    diff_throughput = [diffusion[diffusion['representation'] == r]['valid_per_sec'].values[0]
                       if len(diffusion[diffusion['representation'] == r]) > 0 else 0 for r in reps]
    ar_throughput = [ar[ar['representation'] == r]['valid_per_sec'].values[0]
                     if len(ar[ar['representation'] == r]) > 0 else 0 for r in reps]

    ax.bar(x - width/2, diff_throughput, width, label='Bi_Diffusion', color='#1f77b4')
    ax.bar(x + width/2, ar_throughput, width, label='AR_Transformer', color='#ff7f0e')
    ax.set_xlabel('Representation')
    ax.set_ylabel('Valid Samples / Second')
    ax.set_title('Throughput: Diffusion vs AR')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Validity vs Throughput scatter
    ax = axes[1, 0]
    for _, row in grouped.iterrows():
        marker = 'o' if row['method'] == 'Bi_Diffusion' else 's'
        color = rep_colors.get(row['representation'], 'gray')
        label = f"{row['method'][:2]}_{row['representation']}"
        ax.scatter(row['valid_per_sec'], row['validity_two_stars'],
                   marker=marker, color=color, s=100, label=label)

    ax.set_xlabel('Valid Samples / Second')
    ax.set_ylabel('Validity (2-star)')
    ax.set_title('Validity vs Throughput Tradeoff')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Diversity comparison
    ax = axes[1, 1]
    diff_div = [diffusion[diffusion['representation'] == r]['avg_diversity'].values[0]
                if len(diffusion[diffusion['representation'] == r]) > 0 else 0 for r in reps]
    ar_div = [ar[ar['representation'] == r]['avg_diversity'].values[0]
              if len(ar[ar['representation'] == r]) > 0 else 0 for r in reps]

    ax.bar(x - width/2, diff_div, width, label='Bi_Diffusion', color='#1f77b4')
    ax.bar(x + width/2, ar_div, width, label='AR_Transformer', color='#ff7f0e')
    ax.set_xlabel('Representation')
    ax.set_ylabel('Average Diversity')
    ax.set_title('Diversity: Diffusion vs AR')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure C to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure C: Diffusion vs AR Tradeoff")
    parser.add_argument("--input", type=str, default="results/aggregate/metrics_generation.csv",
                        help="Path to metrics_generation.csv")
    parser.add_argument("--output", type=str, default="results/figures/figure_c_diffusion_vs_ar.png",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_diffusion_vs_ar(df, output_path)


if __name__ == "__main__":
    main()
