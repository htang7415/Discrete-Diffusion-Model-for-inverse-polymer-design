#!/usr/bin/env python
"""Figure A: Cross-view Retrieval Recall@K.

Reads metrics_alignment.csv and creates grouped bar charts showing
Recall@1, Recall@5, Recall@10 for each view pair.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_retrieval_recall(df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart for cross-view retrieval metrics."""
    if df.empty:
        print("No alignment data found. Skipping plot.")
        return

    # Get unique view pairs
    view_pairs = df['view_pair'].unique()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(view_pairs))
    width = 0.25

    # Extract recall values for each K
    recall_1 = df.groupby('view_pair')['recall_at_1'].mean()
    recall_5 = df.groupby('view_pair')['recall_at_5'].mean()
    recall_10 = df.groupby('view_pair')['recall_at_10'].mean()

    # Ensure order matches
    r1 = [recall_1.get(vp, 0) for vp in view_pairs]
    r5 = [recall_5.get(vp, 0) for vp in view_pairs]
    r10 = [recall_10.get(vp, 0) for vp in view_pairs]

    # Create bars
    bars1 = ax.bar(x - width, r1, width, label='Recall@1', color='#1f77b4')
    bars5 = ax.bar(x, r5, width, label='Recall@5', color='#ff7f0e')
    bars10 = ax.bar(x + width, r10, width, label='Recall@10', color='#2ca02c')

    # Customize plot
    ax.set_xlabel('View Pair')
    ax.set_ylabel('Recall')
    ax.set_title('Cross-View Retrieval Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(view_pairs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars5, bars10]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure A to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure A: Retrieval Recall@K")
    parser.add_argument("--input", type=str, default="results/aggregate/metrics_alignment.csv",
                        help="Path to metrics_alignment.csv")
    parser.add_argument("--output", type=str, default="results/figures/figure_a_retrieval.png",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_retrieval_recall(df, output_path)


if __name__ == "__main__":
    main()
