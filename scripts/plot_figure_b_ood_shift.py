#!/usr/bin/env python
"""Figure B: OOD Embedding Shift D1 vs D2.

Reads metrics_ood.csv and creates visualization showing:
- D1 to D2 distance
- Generated samples to D2 distance
- Fraction of generated samples near D2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ood_shift(df: pd.DataFrame, output_path: Path) -> None:
    """Create OOD shift visualization."""
    if df.empty:
        print("No OOD data found. Skipping plot.")
        return

    # Group by method and representation
    grouped = df.groupby(['method', 'representation']).agg({
        'd1_to_d2_mean_dist': 'mean',
        'generated_to_d2_mean_dist': 'mean',
        'frac_generated_near_d2': 'mean'
    }).reset_index()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Distance comparison
    methods = grouped['method'] + ' - ' + grouped['representation']
    x = np.arange(len(methods))
    width = 0.35

    d1_d2 = grouped['d1_to_d2_mean_dist'].values
    gen_d2 = grouped['generated_to_d2_mean_dist'].values

    bars1 = ax1.bar(x - width/2, d1_d2, width, label='D1 to D2', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, gen_d2, width, label='Generated to D2', color='#ff7f0e')

    ax1.set_xlabel('Method - Representation')
    ax1.set_ylabel('Mean Embedding Distance')
    ax1.set_title('Embedding Distance to Real Polymers (D2)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Fraction near D2
    frac_near = grouped['frac_generated_near_d2'].values
    colors = plt.cm.RdYlGn(frac_near)  # Red to green colormap

    bars3 = ax2.bar(x, frac_near, color=colors, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Method - Representation')
    ax2.set_ylabel('Fraction Near D2')
    ax2.set_title('Fraction of Generated Samples Near Real Polymer Manifold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars3, frac_near):
        ax2.annotate(f'{val:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure B to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure B: OOD Embedding Shift")
    parser.add_argument("--input", type=str, default="results/aggregate/metrics_ood.csv",
                        help="Path to metrics_ood.csv")
    parser.add_argument("--output", type=str, default="results/figures/figure_b_ood_shift.png",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_ood_shift(df, output_path)


if __name__ == "__main__":
    main()
