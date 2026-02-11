#!/usr/bin/env python
"""Figure D: Constraint Failure Taxonomy.

Reads metrics_constraints.csv and creates visualization showing:
- Per-constraint violation rates for SMILES representation
- Comparison across methods
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_constraint_failures(df: pd.DataFrame, output_path: Path) -> None:
    """Create constraint failure taxonomy visualization."""
    if df.empty:
        print("No constraint data found. Skipping plot.")
        return

    # Filter to SMILES representation for the main taxonomy
    smiles_df = df[df['representation'] == 'SMILES']

    if smiles_df.empty:
        print("No SMILES constraint data found. Using all representations.")
        smiles_df = df

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Constraint violation rates by constraint type (SMILES)
    constraint_pivot = smiles_df.pivot_table(
        values='violation_rate',
        index='constraint',
        columns='method',
        aggfunc='mean'
    ).fillna(0)

    x = np.arange(len(constraint_pivot.index))
    width = 0.35
    methods = constraint_pivot.columns.tolist()

    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        values = constraint_pivot[method].values
        ax1.bar(x + offset, values, width, label=method)

    ax1.set_xlabel('Constraint Type')
    ax1.set_ylabel('Violation Rate')
    ax1.set_title('SMILES Constraint Violation Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(constraint_pivot.index, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, min(1.0, constraint_pivot.values.max() * 1.2 + 0.05))
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Overall violation rates by representation
    rep_summary = df.groupby(['method', 'representation']).agg({
        'violations': 'sum',
        'total': 'sum'
    }).reset_index()
    rep_summary['violation_rate'] = rep_summary['violations'] / rep_summary['total'].replace(0, 1)

    rep_pivot = rep_summary.pivot_table(
        values='violation_rate',
        index='representation',
        columns='method',
        aggfunc='mean'
    ).fillna(0)

    x2 = np.arange(len(rep_pivot.index))
    methods2 = rep_pivot.columns.tolist()

    for i, method in enumerate(methods2):
        offset = (i - len(methods2)/2 + 0.5) * width
        values = rep_pivot[method].values
        ax2.bar(x2 + offset, values, width, label=method)

    ax2.set_xlabel('Representation')
    ax2.set_ylabel('Overall Violation Rate')
    ax2.set_title('Constraint Violations by Representation')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rep_pivot.index, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, min(1.0, rep_pivot.values.max() * 1.2 + 0.05))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure D to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure D: Constraint Failures")
    parser.add_argument("--input", type=str, default="results/aggregate/metrics_constraints.csv",
                        help="Path to metrics_constraints.csv")
    parser.add_argument("--output", type=str, default="results/figures/figure_d_constraints.png",
                        help="Output figure path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate_metrics.py first to generate metrics.")
        return

    df = pd.read_csv(input_path)
    plot_constraint_failures(df, output_path)


if __name__ == "__main__":
    main()
