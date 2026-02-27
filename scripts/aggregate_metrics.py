#!/usr/bin/env python
"""Aggregate Step1/Step2 metrics for five Bi_Diffusion methods and build figures.

Expected workflow:
1) bash scripts/submit_all_5_methods_nrel.sh all
2) python scripts/aggregate_metrics.py --model_sizes all

Step1:
- Uses BPB directly when present in metrics files.
- Falls back to best validation loss and derives BPB via: bpb = val_loss / ln(2).

Step2:
- Collects validity, uniqueness, novelty, diversity (avg_diversity), mean SA, and std SA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Avoid Intel OpenMP shared-memory failures in restricted environments.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.metrics_schema import infer_model_size, list_results_dirs, parse_method_representation


DEFAULT_METHOD_DIRS = [
    "Bi_Diffusion_SMILES",
    "Bi_Diffusion_SMILES_BPE",
    "Bi_Diffusion_SELFIES",
    "Bi_Diffusion_Group_SELFIES",
    "Bi_Diffusion_graph",
]

MODEL_SIZE_SEQUENCE = ["small", "medium", "large", "xl", "base"]
MODEL_SIZE_RANK = {name: idx for idx, name in enumerate(MODEL_SIZE_SEQUENCE)}
METHOD_RANK = {name: idx for idx, name in enumerate(DEFAULT_METHOD_DIRS)}

REP_COLORS = {
    "SMILES": "#2563EB",
    "SMILES_BPE": "#0EA5E9",
    "SELFIES": "#16A34A",
    "Group_SELFIES": "#F97316",
    "Graph": "#7C3AED",
}

STEP1_BPB_KEYS = (
    "bpb",
    "best_bpb",
    "best_val_bpb",
    "val_bpb",
    "bits_per_byte",
    "bits_per_token",
)
STEP1_LOSS_KEYS = (
    "best_val_loss",
    "val_loss",
    "final_val_loss",
    "validation_loss",
)

RAW_COLUMNS = [
    "method_dir",
    "method",
    "representation",
    "model_size",
    "results_dir",
    "step1_bpb",
    "step1_best_val_loss",
    "step1_metric_source",
    "step1_metric_detail",
    "has_step1",
    "step2_validity",
    "step2_uniqueness",
    "step2_novelty",
    "step2_diversity",
    "step2_mean_sa",
    "step2_std_sa",
    "step2_metric_file",
    "has_step2",
]

SUMMARY_COLUMNS = [
    "method_dir",
    "method",
    "representation",
    "model_size",
    "results_dirs",
    "step1_bpb",
    "step1_best_val_loss",
    "step1_metric_source",
    "step1_metric_detail",
    "has_step1",
    "step2_validity",
    "step2_uniqueness",
    "step2_novelty",
    "step2_diversity",
    "step2_mean_sa",
    "step2_std_sa",
    "has_step2",
    "sa_quality_component",
    "step2_quality_score",
]

STEP2_SCORE_COLUMNS = [
    "step2_validity",
    "step2_uniqueness",
    "step2_novelty",
    "step2_diversity",
    "sa_quality_component",
]


def _safe_float(value) -> Optional[float]:
    """Safely parse float, returning None for invalid/nan/inf values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_model_size(value: str) -> str:
    token = value.strip().lower()
    aliases = {
        "s": "small",
        "m": "medium",
        "l": "large",
        "x": "xl",
    }
    return aliases.get(token, token)


def _parse_model_sizes(raw_value: str) -> Optional[Set[str]]:
    """Parse model size filter argument. Returns None for all."""
    if raw_value is None:
        return None
    value = raw_value.strip().lower()
    if not value or value == "all":
        return None
    parsed = {_normalize_model_size(v) for v in raw_value.split(",") if v.strip()}
    allowed = set(MODEL_SIZE_SEQUENCE)
    unknown = sorted(v for v in parsed if v not in allowed)
    if unknown:
        raise ValueError(
            f"Unsupported model size(s): {', '.join(unknown)}. "
            "Expected one or more of: small, medium, large, xl, base, all."
        )
    return parsed


def _parse_method_dirs(raw_value: str) -> List[str]:
    if raw_value is None:
        return list(DEFAULT_METHOD_DIRS)
    methods = [v.strip() for v in raw_value.split(",") if v.strip()]
    return methods if methods else list(DEFAULT_METHOD_DIRS)


def _include_model_size(model_size: str, allowed_sizes: Optional[Set[str]]) -> bool:
    return allowed_sizes is None or model_size in allowed_sizes


def _pick_metric(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[float]:
    for col in aliases:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return None


def _extract_step1_metrics(results_dir: Path) -> Dict[str, object]:
    """Extract Step1 BPB and best val loss from metrics files."""
    metrics_dir = results_dir / "step1_backbone" / "metrics"
    bpb_candidates: List[Tuple[float, str]] = []
    val_loss_candidates: List[Tuple[float, str]] = []

    if metrics_dir.exists():
        for csv_path in sorted(metrics_dir.glob("*.csv")):
            df = _read_csv(csv_path)
            if df.empty:
                continue
            col_lookup = {col.lower(): col for col in df.columns}

            for key in STEP1_BPB_KEYS:
                col = col_lookup.get(key)
                if col is None:
                    continue
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if not values.empty:
                    bpb_candidates.append((float(values.min()), f"{csv_path.name}:{col}"))

            for key in STEP1_LOSS_KEYS:
                col = col_lookup.get(key)
                if col is None:
                    continue
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if not values.empty:
                    val_loss_candidates.append((float(values.min()), f"{csv_path.name}:{col}"))

            val_col = col_lookup.get("val_loss")
            if val_col is not None:
                values = pd.to_numeric(df[val_col], errors="coerce").dropna()
                if not values.empty:
                    val_loss_candidates.append((float(values.min()), f"{csv_path.name}:{val_col}"))

        for json_path in sorted(metrics_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            for key in STEP1_BPB_KEYS:
                if key not in data:
                    continue
                value = _safe_float(data.get(key))
                if value is not None:
                    bpb_candidates.append((value, f"{json_path.name}:{key}"))

            for key in STEP1_LOSS_KEYS:
                if key not in data:
                    continue
                value = _safe_float(data.get(key))
                if value is not None:
                    val_loss_candidates.append((value, f"{json_path.name}:{key}"))

            val_losses = data.get("val_losses")
            if isinstance(val_losses, list):
                numeric_losses = [_safe_float(v) for v in val_losses]
                numeric_losses = [v for v in numeric_losses if v is not None]
                if numeric_losses:
                    val_loss_candidates.append((min(numeric_losses), f"{json_path.name}:val_losses"))

    best_bpb = min(bpb_candidates, key=lambda x: x[0]) if bpb_candidates else None
    best_val_loss = min(val_loss_candidates, key=lambda x: x[0]) if val_loss_candidates else None

    step1_bpb = best_bpb[0] if best_bpb else None
    step1_best_val_loss = best_val_loss[0] if best_val_loss else None
    metric_source = "missing"
    metric_detail = ""

    if step1_bpb is not None:
        metric_source = "reported_bpb"
        metric_detail = best_bpb[1]
        if step1_best_val_loss is None:
            step1_best_val_loss = step1_bpb * math.log(2.0)
    elif step1_best_val_loss is not None:
        step1_bpb = step1_best_val_loss / math.log(2.0)
        metric_source = "derived_from_best_val_loss"
        metric_detail = best_val_loss[1]

    return {
        "step1_bpb": step1_bpb,
        "step1_best_val_loss": step1_best_val_loss,
        "step1_metric_source": metric_source,
        "step1_metric_detail": metric_detail,
        "has_step1": step1_bpb is not None or step1_best_val_loss is not None,
    }


def _extract_step2_metrics(results_dir: Path) -> Dict[str, object]:
    """Extract Step2 quality metrics from sampling_generative_metrics.csv."""
    metrics_path = results_dir / "step2_sampling" / "metrics" / "sampling_generative_metrics.csv"
    if not metrics_path.exists():
        return {
            "step2_validity": None,
            "step2_uniqueness": None,
            "step2_novelty": None,
            "step2_diversity": None,
            "step2_mean_sa": None,
            "step2_std_sa": None,
            "step2_metric_file": "",
            "has_step2": False,
        }

    df = _read_csv(metrics_path)
    if df.empty:
        return {
            "step2_validity": None,
            "step2_uniqueness": None,
            "step2_novelty": None,
            "step2_diversity": None,
            "step2_mean_sa": None,
            "step2_std_sa": None,
            "step2_metric_file": str(metrics_path),
            "has_step2": False,
        }

    validity = _pick_metric(df, ("validity",))
    uniqueness = _pick_metric(df, ("uniqueness",))
    novelty = _pick_metric(df, ("novelty",))
    diversity = _pick_metric(df, ("avg_diversity", "diversity"))
    mean_sa = _pick_metric(df, ("mean_sa", "avg_sa"))
    std_sa = _pick_metric(df, ("std_sa", "sa_std"))

    return {
        "step2_validity": validity,
        "step2_uniqueness": uniqueness,
        "step2_novelty": novelty,
        "step2_diversity": diversity,
        "step2_mean_sa": mean_sa,
        "step2_std_sa": std_sa,
        "step2_metric_file": str(metrics_path),
        "has_step2": any(v is not None for v in (validity, uniqueness, novelty, diversity, mean_sa, std_sa)),
    }


def _collect_rows(
    root: Path,
    method_dirs: Sequence[str],
    allowed_model_sizes: Optional[Set[str]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for method_name in method_dirs:
        method_dir = root / method_name
        if not method_dir.exists():
            print(f"[WARN] Missing method folder: {method_name}")
            continue

        info = parse_method_representation(method_name)
        for results_dir in list_results_dirs(method_dir):
            model_size = infer_model_size(results_dir)
            if not _include_model_size(model_size, allowed_model_sizes):
                continue

            step1 = _extract_step1_metrics(results_dir)
            step2 = _extract_step2_metrics(results_dir)

            if not step1["has_step1"] and not step2["has_step2"]:
                continue

            try:
                rel_results_dir = str(results_dir.relative_to(root))
            except Exception:
                rel_results_dir = str(results_dir)

            row = {
                "method_dir": method_name,
                "method": info.method,
                "representation": info.representation,
                "model_size": model_size,
                "results_dir": rel_results_dir,
            }
            row.update(step1)
            row.update(step2)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)

    raw_df = pd.DataFrame(rows)
    for col in RAW_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = np.nan
    return raw_df[RAW_COLUMNS]


def _combine_sources(values: Iterable[object]) -> str:
    tokens = []
    for value in values:
        if value is None:
            continue
        token = str(value).strip()
        if not token or token.lower() == "nan":
            continue
        tokens.append(token)
    return "; ".join(sorted(set(tokens)))


def _build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    group_cols = ["method_dir", "method", "representation", "model_size"]
    numeric_aggs = {
        "step1_bpb": "min",
        "step1_best_val_loss": "min",
        "step2_validity": "mean",
        "step2_uniqueness": "mean",
        "step2_novelty": "mean",
        "step2_diversity": "mean",
        "step2_mean_sa": "mean",
        "step2_std_sa": "mean",
    }
    summary = raw_df.groupby(group_cols, as_index=False).agg(numeric_aggs)

    details = raw_df.groupby(group_cols, as_index=False).agg(
        results_dirs=("results_dir", _combine_sources),
        step1_metric_source=("step1_metric_source", _combine_sources),
        step1_metric_detail=("step1_metric_detail", _combine_sources),
    )
    summary = summary.merge(details, on=group_cols, how="left")

    summary["has_step1"] = summary["step1_bpb"].notna() | summary["step1_best_val_loss"].notna()
    summary["has_step2"] = summary[
        ["step2_validity", "step2_uniqueness", "step2_novelty", "step2_diversity", "step2_mean_sa"]
    ].notna().any(axis=1)

    summary["method_rank"] = summary["method_dir"].map(METHOD_RANK).fillna(999).astype(int)
    summary["size_rank"] = summary["model_size"].map(MODEL_SIZE_RANK).fillna(999).astype(int)
    summary = summary.sort_values(["method_rank", "size_rank"]).reset_index(drop=True)
    summary = summary.drop(columns=["method_rank", "size_rank"])

    for col in (
        "step1_bpb",
        "step1_best_val_loss",
        "step2_validity",
        "step2_uniqueness",
        "step2_novelty",
        "step2_diversity",
        "step2_mean_sa",
        "step2_std_sa",
    ):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary


def _append_quality_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if df.empty:
        df["sa_quality_component"] = pd.Series(dtype=float)
        df["step2_quality_score"] = pd.Series(dtype=float)
        return df

    sa_values = pd.to_numeric(df["step2_mean_sa"], errors="coerce")
    if sa_values.notna().any():
        sa_min = float(sa_values.min())
        sa_max = float(sa_values.max())
        if abs(sa_max - sa_min) > 1e-12:
            sa_component = (sa_max - sa_values) / (sa_max - sa_min)
        else:
            sa_component = pd.Series(np.where(sa_values.notna(), 1.0, np.nan), index=df.index)
    else:
        sa_component = pd.Series(np.nan, index=df.index)

    df["sa_quality_component"] = sa_component

    score_matrix = df[STEP2_SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["step2_quality_score"] = score_matrix.mean(axis=1, skipna=False)
    return df


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[list(columns)]


def _save_csv_outputs(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    raw_out = output_dir / "metrics_step1_step2_raw.csv"
    summary_out = output_dir / "metrics_step1_step2_summary.csv"
    step1_out = output_dir / "metrics_step1_bpb.csv"
    step2_out = output_dir / "metrics_step2_sampling.csv"
    rank_step1_out = output_dir / "ranking_step1_bpb.csv"
    rank_step2_out = output_dir / "ranking_step2_quality.csv"

    _ensure_columns(raw_df, RAW_COLUMNS).to_csv(raw_out, index=False)
    _ensure_columns(summary_df, SUMMARY_COLUMNS).to_csv(summary_out, index=False)

    step1_cols = [
        "method_dir",
        "representation",
        "model_size",
        "step1_bpb",
        "step1_best_val_loss",
        "step1_metric_source",
        "step1_metric_detail",
        "results_dirs",
    ]
    step2_cols = [
        "method_dir",
        "representation",
        "model_size",
        "step2_validity",
        "step2_uniqueness",
        "step2_novelty",
        "step2_diversity",
        "step2_mean_sa",
        "step2_std_sa",
        "step2_quality_score",
        "results_dirs",
    ]

    step1_df = _ensure_columns(summary_df, step1_cols)
    step2_df = _ensure_columns(summary_df, step2_cols)

    step1_df.to_csv(step1_out, index=False)
    step2_df.to_csv(step2_out, index=False)

    step1_rank = step1_df.dropna(subset=["step1_bpb"]).sort_values("step1_bpb", ascending=True)
    step2_rank = step2_df.dropna(subset=["step2_quality_score"]).sort_values(
        "step2_quality_score", ascending=False
    )
    step1_rank.to_csv(rank_step1_out, index=False)
    step2_rank.to_csv(rank_step2_out, index=False)

    paths["raw"] = raw_out
    paths["summary"] = summary_out
    paths["step1"] = step1_out
    paths["step2"] = step2_out
    paths["rank_step1"] = rank_step1_out
    paths["rank_step2"] = rank_step2_out
    return paths


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#1F2937",
            "axes.labelcolor": "#111827",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "grid.color": "#D1D5DB",
            "grid.alpha": 0.55,
            "font.size": 16,
            "legend.frameon": True,
            "legend.fontsize": 16,
            "legend.edgecolor": "#D1D5DB",
        }
    )


def _standardize_figure_text_and_legend(fig, font_size: int = 16, legend_loc: str = "best") -> None:
    for text_obj in fig.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        try:
            text_obj.set_fontsize(font_size)
        except Exception:
            continue
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_loc(legend_loc)


def _row_label(row: pd.Series) -> str:
    rep = str(row["representation"]).replace("_", " ")
    return f"{rep} | {row['model_size']}"


def _annotated_heatmap(
    ax,
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotation_matrix: Optional[np.ndarray] = None,
):
    masked = np.ma.masked_invalid(data)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#E5E7EB")
    image = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    ax.set_title(title, pad=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    midpoint = None
    if vmin is not None and vmax is not None:
        midpoint = (vmin + vmax) / 2.0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if annotation_matrix is not None:
                text = annotation_matrix[i, j]
            elif np.isnan(value):
                text = "--"
            else:
                text = f"{value:.3f}"

            text_color = "#111827"
            if midpoint is not None and not np.isnan(value):
                text_color = "#FFFFFF" if value >= midpoint else "#111827"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8.8)

    return image


def _plot_step1_bpb_heatmap(summary_df: pd.DataFrame, fig_dir: Path) -> Optional[Path]:
    plot_df = summary_df.dropna(subset=["step1_bpb"]).copy()
    if plot_df.empty:
        print("[INFO] No Step1 BPB data found; skipping BPB heatmap.")
        return None

    agg_df = plot_df.groupby(["representation", "model_size"], as_index=False)["step1_bpb"].min()

    rep_order = []
    for method_dir in DEFAULT_METHOD_DIRS:
        rep_name = parse_method_representation(method_dir).representation
        if rep_name in set(agg_df["representation"]):
            rep_order.append(rep_name)
    for rep_name in sorted(set(agg_df["representation"])):
        if rep_name not in rep_order:
            rep_order.append(rep_name)

    size_order = [size for size in MODEL_SIZE_SEQUENCE if size in set(agg_df["model_size"])]
    for size in sorted(set(agg_df["model_size"])):
        if size not in size_order:
            size_order.append(size)

    pivot = agg_df.pivot(index="representation", columns="model_size", values="step1_bpb")
    pivot = pivot.reindex(index=rep_order, columns=size_order)

    values = pivot.to_numpy(dtype=float)
    row_labels = [label.replace("_", " ") for label in pivot.index]
    col_labels = [label.upper() for label in pivot.columns]

    fig, ax = plt.subplots(figsize=(1.8 + 1.25 * len(col_labels), 2.0 + 0.65 * len(row_labels)))
    image = _annotated_heatmap(
        ax=ax,
        data=values,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Step1 Backbone BPB (Lower Is Better)",
        cmap="viridis_r",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("BPB")
    fig.tight_layout()

    out_path = fig_dir / "fig_01_step1_bpb_heatmap.png"
    _standardize_figure_text_and_legend(fig, font_size=16, legend_loc="best")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_step2_metrics_heatmap(summary_df: pd.DataFrame, fig_dir: Path) -> Optional[Path]:
    metric_cols = ["step2_validity", "step2_uniqueness", "step2_novelty", "step2_diversity", "step2_mean_sa"]
    plot_df = summary_df[summary_df[metric_cols].notna().any(axis=1)].copy()
    if plot_df.empty:
        print("[INFO] No Step2 metrics found; skipping Step2 heatmap.")
        return None

    plot_df["method_rank"] = plot_df["method_dir"].map(METHOD_RANK).fillna(999).astype(int)
    plot_df["size_rank"] = plot_df["model_size"].map(MODEL_SIZE_RANK).fillna(999).astype(int)
    plot_df = plot_df.sort_values(["method_rank", "size_rank"]).reset_index(drop=True)

    row_labels = [_row_label(row) for _, row in plot_df.iterrows()]

    ratio_cols = ["step2_validity", "step2_uniqueness", "step2_novelty", "step2_diversity"]
    ratio_data = plot_df[ratio_cols].to_numpy(dtype=float)
    sa_data = plot_df[["step2_mean_sa"]].to_numpy(dtype=float)

    sa_ann = np.full(sa_data.shape, "--", dtype=object)
    for idx, row in plot_df.iterrows():
        mean_sa = _safe_float(row["step2_mean_sa"])
        std_sa = _safe_float(row["step2_std_sa"])
        if mean_sa is None:
            continue
        if std_sa is None:
            sa_ann[idx, 0] = f"{mean_sa:.2f}"
        else:
            sa_ann[idx, 0] = f"{mean_sa:.2f}\n+-{std_sa:.2f}"

    fig_h = max(4.8, 0.48 * len(row_labels) + 2.4)
    fig = plt.figure(figsize=(12.2, fig_h))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.5], wspace=0.35)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    left_im = _annotated_heatmap(
        ax=ax_left,
        data=ratio_data,
        row_labels=row_labels,
        col_labels=["Validity", "Uniqueness", "Novelty", "Diversity"],
        title="Step2 Generative Quality",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    cbar_left = fig.colorbar(left_im, ax=ax_left, fraction=0.04, pad=0.03)
    cbar_left.set_label("Ratio")

    finite_sa = sa_data[np.isfinite(sa_data)]
    if finite_sa.size:
        sa_vmin = float(finite_sa.min())
        sa_vmax = float(finite_sa.max())
        if abs(sa_vmax - sa_vmin) < 1e-12:
            sa_vmin -= 0.5
            sa_vmax += 0.5
    else:
        sa_vmin, sa_vmax = 0.0, 1.0

    right_im = _annotated_heatmap(
        ax=ax_right,
        data=sa_data,
        row_labels=row_labels,
        col_labels=["Mean SA"],
        title="Synthesis Accessibility",
        cmap="magma_r",
        vmin=sa_vmin,
        vmax=sa_vmax,
        annotation_matrix=sa_ann,
    )
    ax_right.set_yticklabels([])
    cbar_right = fig.colorbar(right_im, ax=ax_right, fraction=0.08, pad=0.04)
    cbar_right.set_label("Mean SA")

    fig.suptitle("Step2 Metrics Across Bi_Diffusion Methods", y=0.995, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.06, wspace=0.35)

    out_path = fig_dir / "fig_02_step2_metrics_heatmap.png"
    _standardize_figure_text_and_legend(fig, font_size=16, legend_loc="best")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_step2_tradeoff(summary_df: pd.DataFrame, fig_dir: Path) -> Optional[Path]:
    required = ["step2_validity", "step2_novelty", "step2_diversity", "step2_mean_sa"]
    plot_df = summary_df.dropna(subset=required).copy()
    if plot_df.empty:
        print("[INFO] Incomplete Step2 metrics for tradeoff plot; skipping.")
        return None

    plot_df["marker_size"] = 120.0 + 380.0 * np.clip(plot_df["step2_diversity"].to_numpy(dtype=float), 0.0, 1.0)
    plot_df["label"] = plot_df.apply(
        lambda row: f"{str(row['representation']).replace('_', '')}-{str(row['model_size'])[0].upper()}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    scatter = ax.scatter(
        plot_df["step2_novelty"],
        plot_df["step2_validity"],
        s=plot_df["marker_size"],
        c=plot_df["step2_mean_sa"],
        cmap="viridis_r",
        alpha=0.9,
        edgecolors="#111827",
        linewidths=0.7,
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["label"],
            (row["step2_novelty"], row["step2_validity"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="#111827",
        )

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Novelty")
    ax.set_ylabel("Validity")
    ax.set_title("Step2 Tradeoff: Validity vs Novelty")
    ax.grid(True, linestyle="--", alpha=0.5)

    size_handles = []
    for diversity_level in (0.2, 0.5, 0.8):
        size_handles.append(
            ax.scatter(
                [],
                [],
                s=120.0 + 380.0 * diversity_level,
                color="#94A3B8",
                edgecolors="#111827",
                linewidths=0.6,
                label=f"Diversity {diversity_level:.1f}",
            )
        )
    ax.legend(handles=size_handles, loc="lower right", title="Marker size")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Mean SA (Lower Is Better)")

    fig.tight_layout()
    out_path = fig_dir / "fig_03_step2_tradeoff_scatter.png"
    _standardize_figure_text_and_legend(fig, font_size=16, legend_loc="best")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_step2_quality_ranking(summary_df: pd.DataFrame, fig_dir: Path) -> Optional[Path]:
    plot_df = summary_df.dropna(subset=["step2_quality_score"]).copy()
    if plot_df.empty:
        print("[INFO] No complete Step2 quality rows found; skipping ranking chart.")
        return None

    plot_df = plot_df.sort_values("step2_quality_score", ascending=True).reset_index(drop=True)
    labels = [_row_label(row) for _, row in plot_df.iterrows()]
    colors = [REP_COLORS.get(rep, "#3B82F6") for rep in plot_df["representation"]]

    fig_h = max(4.4, 0.45 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(9.4, fig_h))
    bars = ax.barh(labels, plot_df["step2_quality_score"], color=colors, edgecolor="#1F2937", linewidth=0.8)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Composite Step2 Score (Higher Is Better)")
    ax.set_title("Step2 Overall Quality Ranking")
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)

    for bar, score in zip(bars, plot_df["step2_quality_score"]):
        if pd.isna(score):
            continue
        x = min(float(score) + 0.01, 0.985)
        y = bar.get_y() + bar.get_height() / 2.0
        ax.text(x, y, f"{float(score):.3f}", va="center", ha="left", fontsize=8.5, color="#111827")

    rep_handles = []
    for rep in sorted(set(plot_df["representation"])):
        rep_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=REP_COLORS.get(rep, "#3B82F6"),
                markeredgecolor="#1F2937",
                markersize=8,
                label=rep.replace("_", " "),
            )
        )
    ax.legend(handles=rep_handles, loc="lower right", title="Representation")

    fig.tight_layout()
    out_path = fig_dir / "fig_04_step2_quality_ranking.png"
    _standardize_figure_text_and_legend(fig, font_size=16, legend_loc="best")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _generate_figures(summary_df: pd.DataFrame, fig_dir: Path) -> List[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    generated: List[Path] = []
    for plot_fn in (
        _plot_step1_bpb_heatmap,
        _plot_step2_metrics_heatmap,
        _plot_step2_tradeoff,
        _plot_step2_quality_ranking,
    ):
        out_path = plot_fn(summary_df, fig_dir)
        if out_path is not None:
            generated.append(out_path)
    return generated


def _copy_paper_assets(output_dir: Path, paper_output: Path) -> None:
    csv_out = paper_output / "csv"
    fig_out = paper_output / "figures"
    csv_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(output_dir.glob("*.csv")):
        shutil.copy2(csv_path, csv_out / csv_path.name)

    source_fig_dir = output_dir / "figures"
    if source_fig_dir.exists():
        for figure_path in sorted(source_fig_dir.glob("*.png")):
            shutil.copy2(figure_path, fig_out / figure_path.name)

    print(f"Wrote paper-style assets to: {paper_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Step1/Step2 metrics for five Bi_Diffusion methods"
    )
    parser.add_argument("--root", type=str, default=".", help="Repo root")
    parser.add_argument(
        "--output",
        type=str,
        default="results/aggregate_step12",
        help="Output directory for aggregate CSVs and figures",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        default="all",
        help="Comma-separated model sizes to include: small,medium,large,xl,base (or all).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHOD_DIRS),
        help="Comma-separated method folders to aggregate.",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip figure generation and write CSV tables only.",
    )
    parser.add_argument(
        "--save_paper_assets",
        action="store_true",
        help="Copy generated CSVs and figures into --paper_output.",
    )
    parser.add_argument(
        "--paper_output",
        type=str,
        default="results/paper_package",
        help="Paper output folder used with --save_paper_assets.",
    )
    parser.add_argument(
        "--step12_only",
        action="store_true",
        help="Legacy no-op flag retained for compatibility.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output).resolve()
    method_dirs = _parse_method_dirs(args.methods)
    allowed_model_sizes = _parse_model_sizes(args.model_sizes)

    if args.step12_only:
        print("[INFO] --step12_only is now implicit and kept only for compatibility.")

    raw_df = _collect_rows(root, method_dirs, allowed_model_sizes)
    summary_df = _build_summary(raw_df)
    summary_df = _append_quality_score(summary_df)

    output_paths = _save_csv_outputs(raw_df, summary_df, output_dir)

    generated_figures: List[Path] = []
    if not args.no_figures:
        generated_figures = _generate_figures(summary_df, output_dir / "figures")

    if args.save_paper_assets:
        _copy_paper_assets(output_dir, Path(args.paper_output).resolve())

    print(f"Wrote aggregate outputs to: {output_dir}")
    print(f"Methods requested: {len(method_dirs)}")
    print(f"Result rows collected: {len(raw_df)}")
    print(f"Summary rows: {len(summary_df)}")
    if allowed_model_sizes is not None:
        print(f"Filtered model sizes: {','.join(sorted(allowed_model_sizes))}")
    print("CSV outputs:")
    for key in ("raw", "summary", "step1", "step2", "rank_step1", "rank_step2"):
        print(f"  - {key}: {output_paths[key]}")
    if generated_figures:
        print("Figure outputs:")
        for fig in generated_figures:
            print(f"  - {fig}")
    elif args.no_figures:
        print("Figure generation skipped (--no_figures).")
    else:
        print("No figures generated (metrics not found yet).")


if __name__ == "__main__":
    main()
