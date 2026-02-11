#!/usr/bin/env python
"""Aggregate metrics across all method folders.

This script is intentionally conservative: it aggregates what exists and
creates empty CSVs with the standard schema when data is missing.
"""

import argparse
import importlib.util
from pathlib import Path
from typing import List
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _iter_mvf_results_dirs(root: Path) -> List[Path]:
    mvf_dir = root / "Multi_View_Foundation"
    if not mvf_dir.exists():
        return []

    candidates = [mvf_dir / "results"]
    candidates.extend(
        sorted(
            p for p in mvf_dir.iterdir()
            if p.is_dir() and p.name.startswith("results")
        )
    )

    unique_dirs = []
    seen = set()
    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_dirs.append(path)
    return unique_dirs


def _aggregate_alignment(root: Path) -> List[pd.DataFrame]:
    """Aggregate alignment metrics from Multi_View_Foundation.

    Looks for metrics_alignment.csv in Multi_View_Foundation/results/.
    """
    rows = []
    for subdir in _iter_mvf_results_dirs(root):
        metrics_path = subdir / "metrics_alignment.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
        df = ensure_columns(df, ALIGNMENT_COLUMNS)
        rows.append(df[ALIGNMENT_COLUMNS])

    return rows


def _aggregate_mvf_property(root: Path) -> List[pd.DataFrame]:
    """Aggregate property metrics from Multi_View_Foundation.

    Looks for metrics_property.csv in Multi_View_Foundation/results/.
    """
    rows = []
    for subdir in _iter_mvf_results_dirs(root):
        metrics_path = subdir / "metrics_property.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
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


def _aggregate_mvf_inverse(root: Path) -> List[pd.DataFrame]:
    """Aggregate inverse-design metrics from Multi_View_Foundation."""
    rows = []
    for subdir in _iter_mvf_results_dirs(root):
        metrics_path = subdir / "metrics_inverse.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
        if "method" not in df.columns:
            df["method"] = "Multi_View_Foundation"
        if "representation" not in df.columns:
            df["representation"] = "SMILES"
        model_size = infer_model_size(subdir)
        if "model_size" not in df.columns:
            df["model_size"] = model_size
        df = ensure_columns(df, INVERSE_COLUMNS)
        rows.append(df[INVERSE_COLUMNS])

    return rows


def _aggregate_mvf_ood(root: Path) -> List[pd.DataFrame]:
    """Aggregate OOD metrics from Multi_View_Foundation."""
    rows = []
    for subdir in _iter_mvf_results_dirs(root):
        metrics_path = subdir / "metrics_ood.csv"
        if not metrics_path.exists():
            continue
        df = _read_csv(metrics_path)
        if df.empty:
            continue
        if "method" not in df.columns:
            df["method"] = "Multi_View_Foundation"
        if "representation" not in df.columns:
            df["representation"] = "SMILES"
        model_size = infer_model_size(subdir)
        if "model_size" not in df.columns:
            df["model_size"] = model_size
        df = ensure_columns(df, OOD_COLUMNS)
        rows.append(df[OOD_COLUMNS])

    return rows


def _write_or_empty(df_list: List[pd.DataFrame], columns, output_path: Path) -> None:
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        df = ensure_columns(df, columns)
    else:
        df = pd.DataFrame(columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _write_mvf_raw_csvs(root: Path, output_dir: Path) -> None:
    """Copy raw MVF metric CSVs by results directory for paper traceability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "metrics_alignment.csv",
        "metrics_property.csv",
        "metrics_ood.csv",
        "metrics_inverse.csv",
    ]

    for mvf_results_dir in _iter_mvf_results_dirs(root):
        tag = mvf_results_dir.name
        for metric_name in metric_names:
            src = mvf_results_dir / metric_name
            if not src.exists():
                continue
            df = _read_csv(src)
            if df.empty:
                continue
            out_path = output_dir / f"{tag}_{metric_name}"
            df.to_csv(out_path, index=False)


def _save_paper_assets(root: Path, aggregate_dir: Path, paper_output_dir: Path) -> None:
    """Save paper-ready CSV package and figures (A-E)."""
    csv_dir = paper_output_dir / "csv"
    fig_dir = paper_output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Copy aggregate CSVs and create figure-specific CSV aliases.
    csv_aliases = {
        "metrics_alignment.csv": "figure_a_retrieval.csv",
        "metrics_ood.csv": "figure_b_ood_shift.csv",
        "metrics_generation.csv": "figure_c_diffusion_vs_ar.csv",
        "metrics_constraints.csv": "figure_d_constraints.csv",
        "metrics_inverse.csv": "figure_e_inverse_design.csv",
        "metrics_property.csv": "table_property.csv",
    }

    for src_name, alias_name in csv_aliases.items():
        src_path = aggregate_dir / src_name
        if not src_path.exists():
            continue
        df = _read_csv(src_path)
        # Keep canonical aggregate filename.
        df.to_csv(csv_dir / src_name, index=False)
        # Also write the paper-facing alias.
        df.to_csv(csv_dir / alias_name, index=False)

    # Save raw MVF CSVs separately (per results folder).
    _write_mvf_raw_csvs(root, csv_dir / "mvf")

    # Generate figures A-E from aggregate CSVs.
    figure_specs = [
        (
            "plot_figure_a_retrieval.py",
            "plot_retrieval_recall",
            "metrics_alignment.csv",
            "figure_a_retrieval",
        ),
        (
            "plot_figure_b_ood_shift.py",
            "plot_ood_shift",
            "metrics_ood.csv",
            "figure_b_ood_shift",
        ),
        (
            "plot_figure_c_diffusion_vs_ar.py",
            "plot_diffusion_vs_ar",
            "metrics_generation.csv",
            "figure_c_diffusion_vs_ar",
        ),
        (
            "plot_figure_d_constraints.py",
            "plot_constraint_failures",
            "metrics_constraints.csv",
            "figure_d_constraints",
        ),
        (
            "plot_figure_e_inverse_design.py",
            "plot_inverse_design",
            "metrics_inverse.csv",
            "figure_e_inverse_design",
        ),
    ]

    for script_name, fn_name, csv_name, figure_stem in figure_specs:
        csv_path = aggregate_dir / csv_name
        if not csv_path.exists():
            continue
        df = _read_csv(csv_path)
        if df.empty:
            continue
        script_path = SCRIPT_DIR / script_name
        if not script_path.exists():
            print(f"WARNING: figure script not found: {script_path}")
            continue
        try:
            module = _load_module(f"paper_{figure_stem}", script_path)
            plot_fn = getattr(module, fn_name, None)
            if plot_fn is None:
                print(f"WARNING: function {fn_name} not found in {script_name}")
                continue
            plot_fn(df, fig_dir / f"{figure_stem}.png")
            plot_fn(df, fig_dir / f"{figure_stem}.pdf")
        except Exception as exc:
            print(f"WARNING: failed to generate {figure_stem}: {exc}")

    print(f"Wrote paper CSVs to: {csv_dir}")
    print(f"Wrote paper figures to: {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics across methods")
    parser.add_argument("--root", type=str, default=".", help="Repo root")
    parser.add_argument("--output", type=str, default="results/aggregate", help="Output directory")
    parser.add_argument(
        "--save_paper_assets",
        action="store_true",
        help="Also save paper CSV package and Figures A-E.",
    )
    parser.add_argument(
        "--paper_output",
        type=str,
        default="results/paper_package",
        help="Output directory for paper CSVs and figures (used with --save_paper_assets).",
    )
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
    inv_rows.extend(_aggregate_mvf_inverse(root))
    ood_rows.extend(_aggregate_mvf_ood(root))
    align_rows.extend(_aggregate_alignment(root))
    prop_rows.extend(_aggregate_mvf_property(root))

    _write_or_empty(gen_rows, GENERATION_COLUMNS, output_dir / "metrics_generation.csv")
    _write_or_empty(inv_rows, INVERSE_COLUMNS, output_dir / "metrics_inverse.csv")
    _write_or_empty(prop_rows, PROPERTY_COLUMNS, output_dir / "metrics_property.csv")
    _write_or_empty(cons_rows, CONSTRAINT_COLUMNS, output_dir / "metrics_constraints.csv")
    _write_or_empty(ood_rows, OOD_COLUMNS, output_dir / "metrics_ood.csv")
    _write_or_empty(align_rows, ALIGNMENT_COLUMNS, output_dir / "metrics_alignment.csv")

    print(f"Wrote aggregate metrics to: {output_dir}")
    if args.save_paper_assets:
        paper_output_dir = Path(args.paper_output).resolve()
        _save_paper_assets(root, output_dir, paper_output_dir)


if __name__ == "__main__":
    main()
