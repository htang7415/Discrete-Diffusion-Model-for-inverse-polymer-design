#!/usr/bin/env python
"""F2: Retrieval evaluation across views."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config, save_config
from src.evaluation.retrieval_metrics import compute_recall_at_k
from src.utils.output_layout import ensure_step_dirs, save_csv

from src.model.multi_view_model import MultiViewModel

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
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


def _save_figure_png(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _draw_heatmap(fig, ax, matrix: np.ndarray, row_labels, col_labels, vmin=0.0, vmax=1.0) -> None:
    arr = np.asarray(matrix, dtype=np.float32)
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            val = arr[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{float(val):.3f}", ha="center", va="center", fontsize=7, color="black")
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
    metric_cols = recall_cols + ["match_rate"]

    fig, axes = plt.subplots(
        len(datasets),
        len(metric_cols),
        figsize=(4.4 * len(metric_cols), max(3.6, 3.2 * len(datasets))),
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


def _load_alignment_model(results_dir: Path, view_dims: dict, config: dict, checkpoint_override: Optional[str]):
    ckpt_path = _resolve_path(checkpoint_override) if checkpoint_override else results_dir / "step1_alignment" / "alignment_best.pt"
    if not ckpt_path.exists():
        return None
    model_cfg = config.get("model", {})
    model = MultiViewModel(
        view_dims=view_dims,
        projection_dim=int(model_cfg.get("projection_dim", 256)),
        projection_hidden_dims=model_cfg.get("projection_hidden_dims"),
        dropout=float(model_cfg.get("view_dropout", 0.0)),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def _project_embeddings(model: MultiViewModel, view: str, embeddings: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    if embeddings is None or embeddings.size == 0:
        return embeddings
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start:start + batch_size], device=device, dtype=torch.float32)
            z = model.forward(view, batch)
            outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


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

    alignment_model = None
    if args.use_alignment:
        alignment_model = _load_alignment_model(results_dir, view_dims, config, args.alignment_checkpoint)
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for --use_alignment")

    if alignment_model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for view, datasets in view_data.items():
            for dataset in datasets:
                datasets[dataset]["embeddings"] = _project_embeddings(
                    alignment_model,
                    view,
                    datasets[dataset]["embeddings"],
                    device=device,
                )

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
                    "view_dropout_mode": "aligned" if alignment_model is not None else "none",
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

    print(f"Saved metrics_alignment.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
