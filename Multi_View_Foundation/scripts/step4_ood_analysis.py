#!/usr/bin/env python
"""F4: OOD analysis (initial implementation)."""

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from shared.ood_metrics import compute_ood_metrics_from_files, knn_distances
from src.model.multi_view_model import MultiViewModel
from src.utils.output_layout import ensure_step_dirs, save_csv, save_numpy

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


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def _plot_f4_ood_diagnostics(
    *,
    d1_to_d2_dist: np.ndarray,
    gen_to_d2_dist: Optional[np.ndarray],
    metrics_row: dict,
    figures_dir: Path,
) -> None:
    if plt is None:
        return
    d1_vals = np.asarray(d1_to_d2_dist, dtype=np.float32).reshape(-1)
    d1_vals = d1_vals[np.isfinite(d1_vals)]
    if d1_vals.size == 0:
        return

    gen_vals = None
    if gen_to_d2_dist is not None:
        g = np.asarray(gen_to_d2_dist, dtype=np.float32).reshape(-1)
        g = g[np.isfinite(g)]
        if g.size:
            gen_vals = g

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax0, ax1 = axes

    ax0.hist(d1_vals, bins=40, alpha=0.7, color="#4E79A7", label="D1 -> D2")
    if gen_vals is not None:
        ax0.hist(gen_vals, bins=40, alpha=0.6, color="#E15759", label="Generated -> D2")
    ax0.set_xlabel("Distance")
    ax0.set_ylabel("Count")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=15)

    summary_keys = [
        ("d1_to_d2_mean_dist", "D1→D2 mean dist"),
        ("generated_to_d2_mean_dist", "Gen→D2 mean dist"),
        ("frac_generated_near_d2", "Frac gen near D2"),
    ]
    stat_lines = []
    for key, label in summary_keys:
        val = metrics_row.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except Exception:
            continue
        if np.isfinite(val):
            fmt = f"{val:.4f}" if key.startswith("frac") else f"{val:.3f}"
            stat_lines.append((label, fmt))
    # Also add D1 count and Gen count for context
    if d1_vals.size:
        stat_lines.append(("D1 samples", f"{d1_vals.size:,}"))
    if gen_vals is not None:
        stat_lines.append(("Gen samples", f"{gen_vals.size:,}"))
    ax1.set_axis_off()
    if stat_lines:
        col_labels = ["Metric", "Value"]
        table_data = [[lbl, val] for lbl, val in stat_lines]
        tbl = ax1.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(13)
        tbl.scale(1.0, 2.0)
        # Style header row
        for col_idx in range(len(col_labels)):
            tbl[(0, col_idx)].set_facecolor("#4E79A7")
            tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")
        ax1.set_title("OOD Summary Statistics", fontsize=13, pad=4)
    else:
        ax1.text(0.5, 0.5, "No finite summary metrics", ha="center", va="center", fontsize=13)

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f4_ood_diagnostics")
    plt.close(fig)


def _load_model_size(results_dir: Path, config: dict) -> str:
    for meta_path in [
        results_dir / "embedding_meta.json",
        results_dir / "step1_alignment_embeddings" / "files" / "embedding_meta.json",
    ]:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return meta.get("model_size", "base")
    return (
        config.get("smiles_encoder", {}).get("model_size")
        or config.get("smiles_bpe_encoder", {}).get("model_size")
        or "base"
    )


def _load_primary_view(results_dir: Path, config: dict) -> str:
    for meta_path in [
        results_dir / "embedding_meta.json",
        results_dir / "step1_alignment_embeddings" / "files" / "embedding_meta.json",
    ]:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            view = meta.get("view")
            if isinstance(view, str) and view:
                return view
    if (results_dir / "embeddings_smiles_bpe_d1.npy").exists():
        return "smiles_bpe"
    if config.get("smiles_bpe_encoder", {}).get("method_dir") and not config.get("smiles_encoder", {}).get("method_dir"):
        return "smiles_bpe"
    return "smiles"


def _view_to_representation(view: str) -> str:
    return "SMILES_BPE" if view == "smiles_bpe" else "SMILES"


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
    step_dirs = ensure_step_dirs(results_dir, "step4_ood")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")
    primary_view = _load_primary_view(results_dir, config)

    d1_path = results_dir / "embeddings_d1.npy"
    d2_path = results_dir / "embeddings_d2.npy"
    if not d1_path.exists() or not d2_path.exists():
        raise FileNotFoundError("embeddings_d1.npy or embeddings_d2.npy not found. Run F1 first.")

    if args.use_alignment:
        d1 = np.load(d1_path)
        d2 = np.load(d2_path)
        view_dims = {primary_view: d1.shape[1]}
        model = _load_alignment_model(results_dir, view_dims, config, args.alignment_checkpoint)
        if model is None:
            raise FileNotFoundError("Alignment checkpoint not found for --use_alignment")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        d1_proj = _project_embeddings(model, primary_view, d1, device=device)
        d2_proj = _project_embeddings(model, primary_view, d2, device=device)
        d1_path = step_dirs["files_dir"] / "embeddings_d1_aligned.npy"
        d2_path = step_dirs["files_dir"] / "embeddings_d2_aligned.npy"
        save_numpy(d1_proj, d1_path, legacy_paths=[results_dir / "embeddings_d1_aligned.npy"])
        save_numpy(d2_proj, d2_path, legacy_paths=[results_dir / "embeddings_d2_aligned.npy"])

    gen_path = Path(args.generated_embeddings) if args.generated_embeddings else None

    k = args.k or int(config.get("ood", {}).get("nn_k", 1))
    if args.use_faiss is None:
        use_faiss = bool(config.get("ood", {}).get("use_faiss", True))
    else:
        use_faiss = bool(args.use_faiss)
    print(f"OOD settings: k={k}, use_faiss={use_faiss}")
    sig = inspect.signature(compute_ood_metrics_from_files)
    if "use_faiss" in sig.parameters:
        ood_metrics = compute_ood_metrics_from_files(d1_path, d2_path, gen_path, k=k, use_faiss=use_faiss)
    else:
        print("Warning: loaded compute_ood_metrics_from_files() does not support use_faiss; ignoring setting.")
        ood_metrics = compute_ood_metrics_from_files(d1_path, d2_path, gen_path, k=k)

    row = {
        "method": "Multi_View_Foundation",
        "representation": _view_to_representation(primary_view),
        "model_size": _load_model_size(results_dir, config),
        **ood_metrics,
    }

    out_path = step_dirs["metrics_dir"] / "metrics_ood.csv"
    import pandas as pd
    save_csv(pd.DataFrame([row]), out_path, legacy_paths=[results_dir / "metrics_ood.csv"], index=False)

    ood_cfg = config.get("ood", {}) or {}
    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(ood_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F4 figures.")
        generate_figures = False
    if generate_figures:
        d1 = np.load(d1_path)
        d2 = np.load(d2_path)
        d1_to_d2_dist = knn_distances(d1, d2, k=k, use_faiss=use_faiss).mean(axis=1)
        gen_to_d2_dist = None
        if gen_path is not None and gen_path.exists():
            try:
                gen = np.load(gen_path)
                if gen.size:
                    gen_to_d2_dist = knn_distances(gen, d2, k=k, use_faiss=use_faiss).mean(axis=1)
            except Exception:
                gen_to_d2_dist = None

        save_csv(
            pd.DataFrame({"d1_to_d2_distance": d1_to_d2_dist}),
            step_dirs["files_dir"] / "d1_to_d2_distances.csv",
            legacy_paths=[results_dir / "step4_ood" / "d1_to_d2_distances.csv"],
            index=False,
        )
        if gen_to_d2_dist is not None:
            save_csv(
                pd.DataFrame({"generated_to_d2_distance": gen_to_d2_dist}),
                step_dirs["files_dir"] / "generated_to_d2_distances.csv",
                legacy_paths=[results_dir / "step4_ood" / "generated_to_d2_distances.csv"],
                index=False,
            )

        _plot_f4_ood_diagnostics(
            d1_to_d2_dist=d1_to_d2_dist,
            gen_to_d2_dist=gen_to_d2_dist,
            metrics_row=row,
            figures_dir=step_dirs["figures_dir"],
        )

    print(f"Saved metrics_ood.csv to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generated_embeddings", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--use_faiss", dest="use_faiss", action="store_true")
    parser.add_argument("--no_faiss", dest="use_faiss", action="store_false")
    parser.set_defaults(use_faiss=None)
    parser.add_argument("--use_alignment", dest="use_alignment", action="store_true")
    parser.add_argument("--no_alignment", dest="use_alignment", action="store_false")
    parser.set_defaults(use_alignment=False)
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
