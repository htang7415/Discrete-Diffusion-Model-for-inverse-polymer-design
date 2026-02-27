#!/usr/bin/env python
"""F2: Retrieval evaluation across views."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config, save_config
from src.evaluation.retrieval_metrics import compute_recall_at_k
from src.utils.output_layout import ensure_step_dirs, save_csv

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
}

if plt is not None:
    plt.rcParams.update(PUBLICATION_STYLE)


VIEW_ORDER = ["smiles", "smiles_bpe", "selfies", "group_selfies", "graph"]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _to_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid integer sample cap.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ordered_views(values):
    seen = set()
    ordered = []
    for view in VIEW_ORDER:
        if view in values and view not in seen:
            ordered.append(view)
            seen.add(view)
    for view in values:
        if view not in seen:
            ordered.append(view)
            seen.add(view)
    return ordered


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


def _save_figure_png(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _standardize_figure_text_and_legend(fig, font_size=16, legend_loc="best")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _draw_heatmap(fig, ax, matrix: np.ndarray, row_labels, col_labels, vmin=0.0, vmax=1.0) -> None:
    arr = np.asarray(matrix, dtype=np.float32)
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    n_rows, n_cols = arr.shape[0], arr.shape[1]
    # Scale tick/cell fontsize to matrix size so text fits in cells
    cell_fs = max(7, min(11, 80 // max(n_rows, n_cols, 1)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=cell_fs + 1)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=cell_fs + 1)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            val = arr[r, c]
            if np.isfinite(val):
                text_color = "white" if float(val) > 0.65 * (vmax - vmin) + vmin else "black"
                ax.text(c, r, f"{float(val):.2f}", ha="center", va="center", fontsize=cell_fs, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _parse_view_pair(text: str):
    raw = str(text)
    if "->" not in raw:
        return None
    left, right = raw.split("->", 1)
    if "_" not in left or "_" not in right:
        return None
    l_dataset, l_view = left.split("_", 1)
    r_dataset, r_view = right.split("_", 1)
    return l_dataset, l_view, r_dataset, r_view


def _plot_f2_retrieval_heatmaps(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    df = metrics_df.copy()
    parsed = df["view_pair"].astype(str).map(_parse_view_pair)
    df = df.assign(_parsed=parsed)
    df = df[df["_parsed"].notna()].copy()
    if df.empty:
        return
    df[["src_dataset", "src_view", "tgt_dataset", "tgt_view"]] = pd.DataFrame(df["_parsed"].tolist(), index=df.index)
    df = df[df["src_dataset"] == df["tgt_dataset"]].copy()
    if df.empty:
        return

    datasets = sorted(df["src_dataset"].astype(str).unique().tolist())
    views = _ordered_views(sorted(set(df["src_view"].astype(str)).union(set(df["tgt_view"].astype(str)))))
    if not datasets or not views:
        return

    recall_cols = [c for c in df.columns if str(c).startswith("recall_at_")]
    recall_cols = sorted(recall_cols, key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else 0)
    cos_sim_cols = [c for c in df.columns if c == "mean_cosine_sim_matched"]
    metric_cols = recall_cols + cos_sim_cols + ["match_rate"]

    # Scale panel size so 5×5 cell text remains legible
    n_views = len(views)
    panel_w = max(5.5, 1.1 * n_views)
    panel_h = max(4.5, 1.0 * n_views)
    fig, axes = plt.subplots(
        len(datasets),
        len(metric_cols),
        figsize=(panel_w * len(metric_cols), panel_h * len(datasets)),
        squeeze=False,
    )

    for row_idx, dataset in enumerate(datasets):
        sub = df[df["src_dataset"] == dataset]
        for col_idx, metric in enumerate(metric_cols):
            ax = axes[row_idx, col_idx]
            matrix = np.full((len(views), len(views)), np.nan, dtype=np.float32)
            for i, src in enumerate(views):
                for j, tgt in enumerate(views):
                    r = sub[(sub["src_view"] == src) & (sub["tgt_view"] == tgt)]
                    if r.empty or metric not in r.columns:
                        continue
                    val = pd.to_numeric(r[metric], errors="coerce").mean()
                    if np.isfinite(val):
                        matrix[i, j] = float(val)
            _draw_heatmap(fig, ax, matrix, views, views, vmin=0.0, vmax=1.0)

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f2_retrieval_heatmaps")
    plt.close(fig)


def _plot_f2_cosine_sim_bars(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart of mean cosine similarity per view pair, grouped by dataset."""
    if plt is None or metrics_df.empty or "mean_cosine_sim_matched" not in metrics_df.columns:
        return
    df = metrics_df.copy()
    parsed = df["view_pair"].astype(str).map(_parse_view_pair)
    df = df.assign(_parsed=parsed)
    df = df[df["_parsed"].notna()].copy()
    if df.empty:
        return
    df[["src_dataset", "src_view", "tgt_dataset", "tgt_view"]] = pd.DataFrame(df["_parsed"].tolist(), index=df.index)
    df = df[df["src_dataset"] == df["tgt_dataset"]].copy()
    if df.empty:
        return
    df["pair_label"] = df["src_view"] + u"\u2192" + df["tgt_view"]

    datasets = sorted(df["src_dataset"].astype(str).unique().tolist())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#FF9DA7", "#9C755F"]
    fig, axes = plt.subplots(1, n_datasets, figsize=(max(6, 3.5 * n_datasets), 5.5), squeeze=False)

    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        sub = df[df["src_dataset"] == dataset].sort_values("pair_label")
        pairs = sub["pair_label"].tolist()
        sims = pd.to_numeric(sub["mean_cosine_sim_matched"], errors="coerce").tolist()
        bar_colors = [palette[i % len(palette)] for i in range(len(pairs))]
        bars = ax.bar(range(len(pairs)), sims, color=bar_colors, alpha=0.88, width=0.6)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=40, ha="right", fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Mean Cosine Similarity (matched pairs)", fontsize=13)
        ax.set_title(f"Dataset: {dataset}", fontsize=14)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, sims):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(val) + 0.012,
                    f"{float(val):.3f}",
                    ha="center", va="bottom", fontsize=10,
                )

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f2_cosine_sim_bars")
    plt.close(fig)


def _load_view_embeddings(results_dir: Path, view: str, dataset: str):
    emb_path = results_dir / f"embeddings_{view}_{dataset}.npy"
    if not emb_path.exists() and view == "smiles":
        legacy = results_dir / f"embeddings_{dataset}.npy"
        emb_path = legacy if legacy.exists() else emb_path
    if not emb_path.exists():
        return None
    return np.load(emb_path)


def _load_view_index(results_dir: Path, view: str):
    idx_path = results_dir / f"embedding_index_{view}.csv"
    if not idx_path.exists():
        return None
    return pd.read_csv(idx_path)


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step2_retrieval")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    views = config.get("alignment_views", ["smiles"])
    ks = config.get("evaluation", {}).get("recall_ks", [1, 5, 10])
    max_eval_samples = _to_int_or_none(config.get("evaluation", {}).get("max_samples_per_dataset"))

    view_data = {}
    view_dims = {}
    for view in views:
        idx_df = _load_view_index(results_dir, view)
        if idx_df is None:
            continue
        view_data[view] = {}
        for dataset in ["d1", "d2"]:
            emb = _load_view_embeddings(results_dir, view, dataset)
            if emb is None:
                continue
            subset = idx_df[idx_df["dataset"] == dataset]
            ids = subset.sort_values("row_index")["polymer_id"].astype(str).tolist()
            if max_eval_samples is not None and len(ids) > max_eval_samples:
                ids = ids[:max_eval_samples]
                emb = emb[:max_eval_samples]
            view_data[view][dataset] = {"embeddings": emb, "ids": ids}
            view_dims[view] = emb.shape[1]

    rows = []
    for dataset in ["d1", "d2"]:
        available_views = [v for v in views if dataset in view_data.get(v, {})]
        if available_views:
            sample_sizes = {v: len(view_data[v][dataset]["ids"]) for v in available_views}
            print(f"Retrieval dataset={dataset} views={available_views} sample_sizes={sample_sizes}")
        for src_view in available_views:
            for tgt_view in available_views:
                src = view_data[src_view][dataset]
                tgt = view_data[tgt_view][dataset]
                metrics = compute_recall_at_k(
                    src["embeddings"],
                    tgt["embeddings"],
                    src["ids"],
                    tgt["ids"],
                    ks,
                )
                rows.append({
                    "view_pair": f"{dataset}_{src_view}->{dataset}_{tgt_view}",
                    "view_dropout_mode": "none",
                    **metrics,
                })

    metrics_df = pd.DataFrame(rows)
    save_csv(
        metrics_df,
        step_dirs["metrics_dir"] / "metrics_alignment.csv",
        legacy_paths=[results_dir / "metrics_alignment.csv"],
        index=False,
    )

    eval_cfg = config.get("evaluation", {}) or {}
    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(eval_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F2 figures.")
        generate_figures = False
    if generate_figures:
        _plot_f2_retrieval_heatmaps(metrics_df, step_dirs["figures_dir"])
        _plot_f2_cosine_sim_bars(metrics_df, step_dirs["figures_dir"])

    print(f"Saved metrics_alignment.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
