#!/usr/bin/env python
"""Figure E: Inverse Design Achievement vs Tolerance.

Reads metrics_inverse.csv and creates visualization showing:
- Achievement rates at different tolerance levels (5%, 10%, 15%, 20%)
- Success@K curves
- Comparison across methods and representations
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_inverse_design(df: pd.DataFrame, output_path: Path) -> None:
    """Create inverse design achievement visualization."""
    if df.empty:
        print("No inverse design data found. Skipping plot.")
        return

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get achievement columns
    achievement_cols = ['achievement_5p', 'achievement_10p', 'achievement_15p', 'achievement_20p']
    tolerance_levels = [5, 10, 15, 20]

    # Group by method and representation
    grouped = df.groupby(['method', 'representation']).agg({
        **{col: 'mean' for col in achievement_cols},
        'success_rate': 'mean',
        'validity_two_stars': 'mean',
        'valid_per_compute': 'mean'
    }).reset_index()

    # Plot 1: Achievement vs Tolerance curves
    ax = axes[0, 0]
    color_cycle = plt.cm.tab10.colors

    for idx, (_, row) in enumerate(grouped.iterrows()):
        label = f"{row['method'][:2]}_{row['representation']}"
        achievements = [row.get(col, 0) for col in achievement_cols]
        ax.plot(tolerance_levels, achievements, 'o-',
                color=color_cycle[idx % len(color_cycle)], label=label, linewidth=2)

    ax.set_xlabel('Tolerance (%)')
    ax.set_ylabel('Achievement Rate')
    ax.set_title('Achievement vs Tolerance')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(tolerance_levels)
    ax.grid(True, alpha=0.3)

    # Plot 2: Success rate comparison
    ax = axes[0, 1]
    methods_reps = grouped['method'] + ' - ' + grouped['representation']
    x = np.arange(len(methods_reps))

    colors = ['#1f77b4' if 'Bi_Diffusion' in m else '#ff7f0e' for m in methods_reps]
    ax.bar(x, grouped['success_rate'], color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Method - Representation')
    ax.set_ylabel('Success Rate')
    ax.set_title('Inverse Design Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_reps, rotation=45, ha='right')
    ax.set_ylim(0, min(1.0, grouped['success_rate'].max() * 1.2 + 0.05))
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Valid per compute (efficiency)
    ax = axes[1, 0]
    ax.bar(x, grouped['valid_per_compute'], color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Method - Representation')
    ax.set_ylabel('Valid Samples per Compute')
    ax.set_title('Compute Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_reps, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Achievement at 10% tolerance comparison (heatmap style)
    ax = axes[1, 1]

    # Pivot for heatmap
    if 'achievement_10p' in grouped.columns:
        pivot = grouped.pivot_table(
            values='achievement_10p',
            index='method',
            columns='representation',
            aggfunc='mean'
        ).fillna(0)

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot.index)

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color)

        ax.set_title('Achievement@10% by Method and Representation')
        plt.colorbar(im, ax=ax, label='Achievement Rate')
    else:
        ax.text(0.5, 0.5, 'No achievement data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Achievement@10%')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure E to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure E: Inverse Design Achievement")
    parser.add_argument("--input", type=str, default="results/aggregate/metrics_inverse.csv",
                        help="Path to metrics_inverse.csv")
    parser.add_argument("--output", type=str, default="results/figures/figure_e_inverse_design.png",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_inverse_design(df, output_path)


if __name__ == "__main__":
    main()
