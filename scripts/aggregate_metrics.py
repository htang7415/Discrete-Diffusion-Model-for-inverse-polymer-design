#!/usr/bin/env python
"""Aggregate metrics across all method folders.

This script is intentionally conservative: it aggregates what exists and
creates empty CSVs with the standard schema when data is missing.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from shared.metrics_schema import (
    ALIGNMENT_COLUMNS,
    CONSTRAINT_COLUMNS,
    GENERATION_COLUMNS,
    INVERSE_COLUMNS,
    OOD_COLUMNS,
    PROPERTY_COLUMNS,
    ensure_columns,
    infer_model_size,
    list_method_dirs,
    list_results_dirs,
    parse_method_representation,
)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _aggregate_generation(method_dir: Path) -> List[pd.DataFrame]:
    rows = []
    info = parse_method_representation(method_dir.name)
    for results_dir in list_results_dirs(method_dir):
        model_size = infer_model_size(results_dir)
        metrics_path = results_dir / "step2_sampling" / "metrics" / "sampling_generative_metrics.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
        df["method"] = info.method
        df["representation"] = info.representation
        df["model_size"] = model_size
        df = ensure_columns(df, GENERATION_COLUMNS)
        rows.append(df[GENERATION_COLUMNS])
    return rows


def _infer_property_from_path(path: Path) -> str:
    # Prefer step4_{property} in the path
    for part in path.parts:
        if part.startswith("step4_"):
            return part.replace("step4_", "")
    # Fallback: filename prefix
    return path.stem.replace("_design", "")


def _aggregate_inverse(method_dir: Path) -> List[pd.DataFrame]:
    rows = []
    info = parse_method_representation(method_dir.name)
    for results_dir in list_results_dirs(method_dir):
        model_size = infer_model_size(results_dir)
        for metrics_path in results_dir.glob("step4_*/metrics/*.csv"):
            if "design" not in metrics_path.name:
                continue
            df = _read_csv(metrics_path)
            if df.empty:
                continue
            df["method"] = info.method
            df["representation"] = info.representation
            df["model_size"] = model_size
            if "property" not in df.columns:
                df["property"] = _infer_property_from_path(metrics_path)
            df = ensure_columns(df, INVERSE_COLUMNS)
            rows.append(df[INVERSE_COLUMNS])
    return rows


def _aggregate_property(method_dir: Path) -> List[pd.DataFrame]:
    """Aggregate property prediction metrics from step3 outputs.

    Looks for metrics in step3_{property}/metrics/ directories.
    Individual methods save {property}_data_stats.csv with split-level stats.
    """
    rows = []
    info = parse_method_representation(method_dir.name)
    for results_dir in list_results_dirs(method_dir):
        model_size = infer_model_size(results_dir)
        # Look for step3 property directories
        for step3_dir in results_dir.glob("step3_*"):
            if not step3_dir.is_dir():
                continue
            property_name = step3_dir.name.replace("step3_", "")
            metrics_dir = step3_dir / "metrics"
            if not metrics_dir.exists():
                continue
            # Try standardized metrics_property.csv first
            metrics_path = metrics_dir / "metrics_property.csv"
            if metrics_path.exists():
                df = _read_csv(metrics_path)
                if not df.empty:
                    df["method"] = info.method
                    df["representation"] = info.representation
                    df["model_size"] = model_size
                    if "property" not in df.columns:
                        df["property"] = property_name
                    df = ensure_columns(df, PROPERTY_COLUMNS)
                    rows.append(df[PROPERTY_COLUMNS])
                    continue
            # Fallback: look for {property}_data_stats.csv
            stats_path = metrics_dir / f"{property_name}_data_stats.csv"
            if stats_path.exists():
                df = _read_csv(stats_path)
                if not df.empty and "split" in df.columns:
                    # Extract MAE/RMSE/R² if present
                    df["method"] = info.method
                    df["representation"] = info.representation
                    df["model_size"] = model_size
                    df["property"] = property_name
                    # Rename columns if needed
                    if "MAE" in df.columns:
                        df["mae"] = df["MAE"]
                    if "RMSE" in df.columns:
                        df["rmse"] = df["RMSE"]
                    if "R²" in df.columns or "R2" in df.columns:
                        df["r2"] = df.get("R²", df.get("R2"))
                    df = ensure_columns(df, PROPERTY_COLUMNS)
                    rows.append(df[PROPERTY_COLUMNS])
    return rows


def _aggregate_constraints(method_dir: Path) -> List[pd.DataFrame]:
    rows = []
    info = parse_method_representation(method_dir.name)
    for results_dir in list_results_dirs(method_dir):
        model_size = infer_model_size(results_dir)
        metrics_path = results_dir / "step2_sampling" / "metrics" / "constraint_metrics.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
        df["method"] = info.method
        df["representation"] = info.representation
        df["model_size"] = model_size
        df = ensure_columns(df, CONSTRAINT_COLUMNS)
        rows.append(df[CONSTRAINT_COLUMNS])
    return rows


def _aggregate_ood(method_dir: Path) -> List[pd.DataFrame]:
    rows = []
    info = parse_method_representation(method_dir.name)
    for results_dir in list_results_dirs(method_dir):
        model_size = infer_model_size(results_dir)
        metrics_path = results_dir / "step2_sampling" / "metrics" / "metrics_ood.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
        df["method"] = info.method
        df["representation"] = info.representation
        df["model_size"] = model_size
        df = ensure_columns(df, OOD_COLUMNS)
        rows.append(df[OOD_COLUMNS])
    return rows


def _aggregate_alignment(root: Path) -> List[pd.DataFrame]:
    """Aggregate alignment metrics from Multi_View_Foundation.

    Looks for metrics_alignment.csv in Multi_View_Foundation/results/.
    """
    rows = []
    mvf_dir = root / "Multi_View_Foundation"
    if not mvf_dir.exists():
        return rows

    # Check results directory
    results_dir = mvf_dir / "results"
    if results_dir.exists():
        metrics_path = results_dir / "metrics_alignment.csv"
        if metrics_path.exists():
            df = _read_csv(metrics_path)
            if not df.empty:
                df = ensure_columns(df, ALIGNMENT_COLUMNS)
                rows.append(df[ALIGNMENT_COLUMNS])

    # Also check for results_* directories (different model sizes)
    for subdir in mvf_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("results"):
            metrics_path = subdir / "metrics_alignment.csv"
            if metrics_path.exists():
                df = _read_csv(metrics_path)
                if not df.empty:
                    df = ensure_columns(df, ALIGNMENT_COLUMNS)
                    rows.append(df[ALIGNMENT_COLUMNS])

    return rows


def _aggregate_mvf_property(root: Path) -> List[pd.DataFrame]:
    """Aggregate property metrics from Multi_View_Foundation.

    Looks for metrics_property.csv in Multi_View_Foundation/results/.
    """
    rows = []
    mvf_dir = root / "Multi_View_Foundation"
    if not mvf_dir.exists():
        return rows

    # Check results directory
    results_dir = mvf_dir / "results"
    if results_dir.exists():
        metrics_path = results_dir / "metrics_property.csv"
        if metrics_path.exists():
            df = _read_csv(metrics_path)
            if not df.empty:
                # Add MVF-specific columns
                if "method" not in df.columns:
                    df["method"] = "Multi_View_Foundation"
                if "representation" not in df.columns:
                    df["representation"] = "multi_view"
                if "model_size" not in df.columns:
                    df["model_size"] = "base"
                df = ensure_columns(df, PROPERTY_COLUMNS)
                rows.append(df[PROPERTY_COLUMNS])

    # Also check for results_* directories
    for subdir in mvf_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("results"):
            metrics_path = subdir / "metrics_property.csv"
            if metrics_path.exists():
                df = _read_csv(metrics_path)
                if not df.empty:
                    if "method" not in df.columns:
                        df["method"] = "Multi_View_Foundation"
                    if "representation" not in df.columns:
                        df["representation"] = "multi_view"
                    model_size = infer_model_size(subdir)
                    if "model_size" not in df.columns:
                        df["model_size"] = model_size
                    df = ensure_columns(df, PROPERTY_COLUMNS)
                    rows.append(df[PROPERTY_COLUMNS])

    return rows


def _write_or_empty(df_list: List[pd.DataFrame], columns, output_path: Path) -> None:
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        df = ensure_columns(df, columns)
    else:
        df = pd.DataFrame(columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics across methods")
    parser.add_argument("--root", type=str, default=".", help="Repo root")
    parser.add_argument("--output", type=str, default="results/aggregate", help="Output directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output).resolve()

    gen_rows = []
    inv_rows = []
    prop_rows = []
    cons_rows = []
    ood_rows = []
    align_rows = []

    for method_dir in list_method_dirs(root):
        gen_rows.extend(_aggregate_generation(method_dir))
        inv_rows.extend(_aggregate_inverse(method_dir))
        prop_rows.extend(_aggregate_property(method_dir))
        cons_rows.extend(_aggregate_constraints(method_dir))
        ood_rows.extend(_aggregate_ood(method_dir))

    # Aggregate from Multi_View_Foundation
    align_rows.extend(_aggregate_alignment(root))
    prop_rows.extend(_aggregate_mvf_property(root))

    _write_or_empty(gen_rows, GENERATION_COLUMNS, output_dir / "metrics_generation.csv")
    _write_or_empty(inv_rows, INVERSE_COLUMNS, output_dir / "metrics_inverse.csv")
    _write_or_empty(prop_rows, PROPERTY_COLUMNS, output_dir / "metrics_property.csv")
    _write_or_empty(cons_rows, CONSTRAINT_COLUMNS, output_dir / "metrics_constraints.csv")
    _write_or_empty(ood_rows, OOD_COLUMNS, output_dir / "metrics_ood.csv")
    _write_or_empty(align_rows, ALIGNMENT_COLUMNS, output_dir / "metrics_alignment.csv")

    print(f"Wrote aggregate metrics to: {output_dir}")


if __name__ == "__main__":
    main()
