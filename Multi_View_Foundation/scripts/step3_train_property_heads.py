#!/usr/bin/env python
"""F3: Property heads on foundation embeddings (multi-view)."""

import argparse
import os
from pathlib import Path
import sys
import importlib.util
import json
import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.metrics import root_mean_squared_error
except Exception:  # pragma: no cover
    root_mean_squared_error = None

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies
from src.model.multi_view_model import MultiViewModel
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json


REPRESENTATION_ORDER = [
    "SMILES",
    "SMILES_BPE",
    "SELFIES",
    "Group_SELFIES",
    "Graph",
    "MultiViewMean",
]

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {
    "train": "#4E79A7",
    "val": "#F28E2B",
    "test": "#E15759",
}

REPRESENTATION_COLORS = {
    "SMILES": "#4E79A7",
    "SMILES_BPE": "#76B7B2",
    "SELFIES": "#F28E2B",
    "Group_SELFIES": "#59A14F",
    "Graph": "#9C755F",
    "MultiViewMean": "#E15759",
}


PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
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


def _resolve_with_base(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sequence_backbone(
    encoder_cfg: dict,
    device: str,
    tokenizer_module: Path,
    tokenizer_class: str,
    tokenizer_filename: str,
):
    method_dir = _resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = _resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    method_cfg = load_config(str(config_path))

    scales_mod = _load_module(
        f"scales_{method_dir.name}",
        method_dir / "src" / "utils" / "model_scales.py",
    )
    tokenizer_mod = _load_module(
        f"tokenizer_{method_dir.name}",
        tokenizer_module,
    )
    backbone_mod = _load_module(
        f"backbone_{method_dir.name}",
        method_dir / "src" / "model" / "backbone.py",
    )

    get_model_config = scales_mod.get_model_config
    get_results_dir = scales_mod.get_results_dir
    TokenizerCls = getattr(tokenizer_mod, tokenizer_class)
    DiffusionBackbone = backbone_mod.DiffusionBackbone

    model_size = encoder_cfg.get("model_size")
    backbone_config = get_model_config(model_size, method_cfg, model_type="sequence")
    diffusion_steps = method_cfg.get("diffusion", {}).get("num_steps", 50)

    base_results_dir = encoder_cfg.get("results_dir")
    if base_results_dir:
        base_results_dir = _resolve_path(base_results_dir)
    else:
        base_results_dir = _resolve_with_base(method_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = _resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / tokenizer_filename
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / tokenizer_filename

    checkpoint_path = encoder_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = _resolve_path(checkpoint_path)
    else:
        step_dir = encoder_cfg.get("step_dir", "step1_backbone")
        checkpoint_name = encoder_cfg.get("checkpoint_name", "backbone_best.pt")
        checkpoint_path = results_dir / step_dir / "checkpoints" / checkpoint_name

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = TokenizerCls.load(str(tokenizer_path))
    backbone = DiffusionBackbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=backbone_config["hidden_size"],
        num_layers=backbone_config["num_layers"],
        num_heads=backbone_config["num_heads"],
        ffn_hidden_size=backbone_config["ffn_hidden_size"],
        max_position_embeddings=backbone_config.get("max_position_embeddings", 256),
        num_diffusion_steps=diffusion_steps,
        dropout=backbone_config.get("dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}

    backbone_state = {
        key[len("backbone."):]: value
        for key, value in cleaned.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = cleaned

    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "model_size": model_size or "base",
        "pooling": encoder_cfg.get("pooling", "mean"),
        "timestep": int(encoder_cfg.get("timestep", 1)),
        "batch_size": int(encoder_cfg.get("batch_size", 256)),
    }


def _load_graph_backbone(encoder_cfg: dict, device: str):
    method_dir = _resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = _resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    method_cfg = load_config(str(config_path))

    scales_mod = _load_module(
        f"graph_scales_{method_dir.name}",
        method_dir / "src" / "utils" / "model_scales.py",
    )
    tokenizer_mod = _load_module(
        f"graph_tokenizer_{method_dir.name}",
        method_dir / "src" / "data" / "graph_tokenizer.py",
    )
    backbone_mod = _load_module(
        f"graph_backbone_{method_dir.name}",
        method_dir / "src" / "model" / "graph_backbone.py",
    )

    get_model_config = scales_mod.get_model_config
    get_results_dir = scales_mod.get_results_dir
    GraphTokenizer = tokenizer_mod.GraphTokenizer
    GraphDiffusionBackbone = backbone_mod.GraphDiffusionBackbone

    model_size = encoder_cfg.get("model_size")
    backbone_config = get_model_config(model_size, method_cfg, model_type="graph")
    diffusion_steps = method_cfg.get("diffusion", {}).get("num_steps", 50)

    base_results_dir = encoder_cfg.get("results_dir")
    if base_results_dir:
        base_results_dir = _resolve_path(base_results_dir)
    else:
        base_results_dir = _resolve_with_base(method_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = _resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / "graph_tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / "graph_tokenizer.json"

    graph_config_path = encoder_cfg.get("graph_config_path")
    if graph_config_path:
        graph_config_path = _resolve_path(graph_config_path)
    else:
        graph_config_path = results_dir / "graph_config.json"
        if not graph_config_path.exists():
            graph_config_path = base_results_dir / "graph_config.json"

    checkpoint_path = encoder_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = _resolve_path(checkpoint_path)
    else:
        step_dir = encoder_cfg.get("step_dir", "step1_backbone")
        checkpoint_name = encoder_cfg.get("checkpoint_name", "graph_backbone_best.pt")
        checkpoint_path = results_dir / step_dir / "checkpoints" / checkpoint_name

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Graph tokenizer not found: {tokenizer_path}")
    if not graph_config_path.exists():
        raise FileNotFoundError(f"Graph config not found: {graph_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Graph checkpoint not found: {checkpoint_path}")

    with open(graph_config_path, "r") as f:
        graph_config = json.load(f)

    tokenizer = GraphTokenizer.load(str(tokenizer_path))
    backbone = GraphDiffusionBackbone(
        atom_vocab_size=graph_config["atom_vocab_size"],
        edge_vocab_size=graph_config["edge_vocab_size"],
        Nmax=graph_config["Nmax"],
        hidden_size=backbone_config["hidden_size"],
        num_layers=backbone_config["num_layers"],
        num_heads=backbone_config["num_heads"],
        ffn_hidden_size=backbone_config["ffn_hidden_size"],
        dropout=backbone_config.get("dropout", 0.1),
        num_diffusion_steps=diffusion_steps,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in cleaned.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = cleaned

    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "model_size": model_size or "base",
        "pooling": encoder_cfg.get("pooling", "mean"),
        "timestep": int(encoder_cfg.get("timestep", 1)),
        "batch_size": int(encoder_cfg.get("batch_size", 64)),
    }


def _embed_sequence(inputs: List[str], assets: dict, device: str) -> np.ndarray:
    if not inputs:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
    embeddings = []
    for start in range(0, len(inputs), assets["batch_size"]):
        batch = inputs[start:start + assets["batch_size"]]
        encoded = assets["tokenizer"].batch_encode(batch)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        timesteps = torch.full((input_ids.size(0),), int(assets["timestep"]), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = assets["backbone"].get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling=assets["pooling"],
            )
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _embed_graph(smiles_list: List[str], assets: dict, device: str) -> tuple[np.ndarray, List[int]]:
    if not smiles_list:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), []
    valid_indices = []
    graph_batches = []
    for idx, smi in enumerate(smiles_list):
        try:
            data = assets["tokenizer"].encode(smi)
            graph_batches.append(data)
            valid_indices.append(idx)
        except Exception:
            continue
    embeddings = []
    for start in range(0, len(graph_batches), assets["batch_size"]):
        batch = graph_batches[start:start + assets["batch_size"]]
        if not batch:
            continue
        X = np.stack([b["X"] for b in batch])
        E = np.stack([b["E"] for b in batch])
        M = np.stack([b["M"] for b in batch])
        X_t = torch.tensor(X, device=device)
        E_t = torch.tensor(E, device=device)
        M_t = torch.tensor(M, device=device)
        timesteps = torch.full((X_t.size(0),), int(assets["timestep"]), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = assets["backbone"].get_node_embeddings(X_t, E_t, timesteps, M_t, pooling=assets["pooling"])
        embeddings.append(pooled.cpu().numpy())
    if embeddings:
        return np.concatenate(embeddings, axis=0), valid_indices
    return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), valid_indices


def _collect_property_files(property_dir: Path, file_list) -> List[Path]:
    if file_list:
        if isinstance(file_list, str):
            file_list = [file_list]
        paths = []
        for name in file_list:
            path = Path(name)
            if not path.is_absolute():
                path = property_dir / name
            paths.append(path)
        return paths
    return sorted(property_dir.glob("*.csv"))


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


def _ordered_representations(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for rep in REPRESENTATION_ORDER:
        if rep in values and rep not in seen:
            ordered.append(rep)
            seen.add(rep)
    for rep in values:
        if rep not in seen:
            ordered.append(rep)
            seen.add(rep)
    return ordered


def _representation_color(rep: str) -> str:
    return REPRESENTATION_COLORS.get(str(rep), "#B07AA1")


def _view_to_representation(view: str) -> str:
    token = str(view or "").strip().lower()
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
        "multiview_mean": "MultiViewMean",
        "multiviewmean": "MultiViewMean",
    }
    return mapping.get(token, token.upper() if token else "UNKNOWN")


def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _save_figure_png(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _draw_heatmap(
    *,
    fig,
    ax,
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "",
    xlabel: str = "Representation",
    ylabel: str = "Property",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    value_fmt: str = ".2f",
    colorbar_label: str = "",
) -> None:
    arr = np.asarray(matrix, dtype=np.float32)
    masked = np.ma.masked_invalid(arr)
    finite_vals = arr[np.isfinite(arr)]
    v0 = vmin if vmin is not None else (float(np.nanmin(finite_vals)) if finite_vals.size else 0.0)
    v1 = vmax if vmax is not None else (float(np.nanmax(finite_vals)) if finite_vals.size else 1.0)
    span = (v1 - v0) if v1 != v0 else 1.0

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=v0, vmax=v1)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=11)
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=6)

    try:
        cmap_fn = plt.get_cmap(cmap)
    except Exception:
        cmap_fn = None

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            val = arr[r, c]
            if np.isfinite(val):
                if cmap_fn is not None:
                    norm_val = (val - v0) / span
                    rgba = cmap_fn(float(np.clip(norm_val, 0, 1)))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    text_color = "white" if lum < 0.45 else "black"
                else:
                    text_color = "black"
                ax.text(c, r, format(float(val), value_fmt), ha="center", va="center", fontsize=10, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03, shrink=0.85)
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.outline.set_linewidth(0.5)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=11)


def _plot_f3_metrics_by_split(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    properties = sorted(metrics_df["property"].astype(str).unique().tolist())
    metric_specs = [
        ("r2", "R2"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
    ]

    n_rows = len(properties)
    n_cols = len(metric_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 3.8 * n_rows), squeeze=False)

    for row_idx, prop in enumerate(properties):
        sub_prop = metrics_df[metrics_df["property"] == prop].copy()
        reps = _ordered_representations(sub_prop["representation"].astype(str).unique().tolist())
        for col_idx, (metric_col, metric_label) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if not reps or metric_col not in sub_prop.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue

            pivot = sub_prop.pivot_table(
                index="representation",
                columns="split",
                values=metric_col,
                aggfunc="mean",
            ).reindex(index=reps)

            x = np.arange(len(reps), dtype=np.float32)
            width = 0.24
            for split_idx, split_name in enumerate(SPLIT_ORDER):
                if split_name not in pivot.columns:
                    continue
                y = pd.to_numeric(pivot[split_name], errors="coerce").to_numpy(dtype=np.float32)
                offset = (split_idx - 1) * width
                ax.bar(
                    x + offset,
                    y,
                    width=width,
                    color=SPLIT_COLORS.get(split_name, "#999999"),
                    label=split_name,
                    alpha=0.9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(reps, rotation=30, ha="right", fontsize=15)
            ax.set_ylabel(metric_label)
            ax.grid(axis="y", alpha=0.25)
            if metric_col == "r2":
                flat = pivot.to_numpy(dtype=np.float32).reshape(-1)
                finite = flat[np.isfinite(flat)]
                low = float(np.min(finite) - 0.05) if finite.size else -0.1
                ax.set_ylim(min(-0.1, low), 1.0)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=15)

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f3_metrics_by_split")
    plt.close(fig)


def _plot_f3_generalization_heatmaps(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    properties = sorted(metrics_df["property"].astype(str).unique().tolist())
    reps = _ordered_representations(metrics_df["representation"].astype(str).unique().tolist())
    if not properties or not reps:
        return

    r2_test = np.full((len(properties), len(reps)), np.nan, dtype=np.float32)
    r2_gap = np.full((len(properties), len(reps)), np.nan, dtype=np.float32)
    rmse_ratio = np.full((len(properties), len(reps)), np.nan, dtype=np.float32)

    for p_idx, prop in enumerate(properties):
        sub_prop = metrics_df[metrics_df["property"] == prop]
        for r_idx, rep in enumerate(reps):
            sub_rep = sub_prop[sub_prop["representation"] == rep]
            if sub_rep.empty:
                continue
            train_r2 = pd.to_numeric(sub_rep[sub_rep["split"] == "train"]["r2"], errors="coerce").mean()
            test_r2 = pd.to_numeric(sub_rep[sub_rep["split"] == "test"]["r2"], errors="coerce").mean()
            train_rmse = pd.to_numeric(sub_rep[sub_rep["split"] == "train"]["rmse"], errors="coerce").mean()
            test_rmse = pd.to_numeric(sub_rep[sub_rep["split"] == "test"]["rmse"], errors="coerce").mean()
            if np.isfinite(test_r2):
                r2_test[p_idx, r_idx] = float(test_r2)
            if np.isfinite(train_r2) and np.isfinite(test_r2):
                r2_gap[p_idx, r_idx] = float(train_r2 - test_r2)
            if np.isfinite(train_rmse) and np.isfinite(test_rmse) and abs(float(train_rmse)) > 1e-12:
                rmse_ratio[p_idx, r_idx] = float(test_rmse / train_rmse)

    panel_h = max(5.0, 0.7 * len(properties) + 2.5)
    fig, axes = plt.subplots(1, 3, figsize=(21, panel_h), constrained_layout=True)

    panels = [
        dict(ax=axes[0], matrix=r2_test,    title="Test R²",                              cmap="YlGn",    vmin=0.0, vmax=1.0, colorbar_label="R²"),
        dict(ax=axes[1], matrix=r2_gap,     title="Generalization Gap\n(Train − Test R²)", cmap="coolwarm",                    colorbar_label="ΔR²"),
        dict(ax=axes[2], matrix=rmse_ratio, title="RMSE Ratio\n(Test / Train)",            cmap="magma",   vmin=0.5,           colorbar_label="Ratio"),
    ]

    for label, spec in zip(["(A)", "(B)", "(C)"], panels):
        _draw_heatmap(fig=fig, row_labels=properties, col_labels=reps, **spec)
        spec["ax"].text(-0.12, 1.04, label, transform=spec["ax"].transAxes,
                        fontsize=14, fontweight="bold", va="bottom", ha="left")

    _save_figure_png(fig, figures_dir / "figure_f3_generalization_heatmaps")
    plt.close(fig)


def _plot_f3_test_head_leaderboard(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    if test_df.empty:
        return

    metric_specs = [
        ("r2", "R2", True),
        ("rmse", "RMSE", False),
        ("mae", "MAE", False),
    ]
    properties = sorted(test_df["property"].astype(str).unique().tolist())
    fig, axes = plt.subplots(
        len(properties),
        len(metric_specs),
        figsize=(5.6 * len(metric_specs), max(3.8, 3.2 * len(properties))),
        squeeze=False,
    )

    ranking_rows = []
    for row_idx, prop in enumerate(properties):
        sub_prop = test_df[test_df["property"] == prop].copy()
        for col_idx, (metric_col, metric_label, higher_is_better) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            metric_values = pd.to_numeric(sub_prop[metric_col], errors="coerce")
            valid = sub_prop.assign(_metric=metric_values).dropna(subset=["_metric"]).copy()
            if valid.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue
            valid = valid.sort_values("_metric", ascending=not higher_is_better).reset_index(drop=True)
            reps = valid["representation"].astype(str).tolist()
            vals = valid["_metric"].to_numpy(dtype=np.float32)

            y = np.arange(len(reps), dtype=np.float32)
            colors = [_representation_color(rep) for rep in reps]
            bars = ax.barh(y, vals, color=colors, alpha=0.92)
            ax.set_yticks(y)
            ax.set_yticklabels(reps, fontsize=15)
            ax.invert_yaxis()
            ax.set_xlabel(metric_label)
            ax.grid(axis="x", alpha=0.25)
            if metric_col == "r2":
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    low = min(-0.1, float(np.min(finite) - 0.05))
                    high = max(1.0, float(np.max(finite) + 0.05))
                    ax.set_xlim(low, high)
            for rank_idx, (bar, rep, val) in enumerate(zip(bars, reps, vals), start=1):
                if not np.isfinite(val):
                    continue
                ax.text(
                    float(val),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"  {val:.3f}",
                    va="center",
                    ha="left",
                    fontsize=15,
                )
                ranking_rows.append(
                    {
                        "property": prop,
                        "metric": metric_col,
                        "rank": rank_idx,
                        "representation": rep,
                        "value": float(val),
                        "split": "test",
                    }
                )

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f3_test_head_leaderboard")
    plt.close(fig)

    if ranking_rows:
        save_csv(
            pd.DataFrame(ranking_rows),
            figures_dir / "figure_f3_test_head_leaderboard.csv",
            index=False,
        )


def _build_f3_fusion_gain_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    if test_df.empty:
        return pd.DataFrame()

    rows = []
    for prop in sorted(test_df["property"].astype(str).unique().tolist()):
        sub_prop = test_df[test_df["property"] == prop].copy()
        fusion = sub_prop[sub_prop["representation"] == "MultiViewMean"].copy()
        single = sub_prop[sub_prop["representation"] != "MultiViewMean"].copy()
        if fusion.empty or single.empty:
            continue
        fusion_row = fusion.iloc[0]

        single_r2 = single.assign(_metric=pd.to_numeric(single["r2"], errors="coerce")).dropna(subset=["_metric"])
        single_rmse = single.assign(_metric=pd.to_numeric(single["rmse"], errors="coerce")).dropna(subset=["_metric"])
        single_mae = single.assign(_metric=pd.to_numeric(single["mae"], errors="coerce")).dropna(subset=["_metric"])
        if single_r2.empty or single_rmse.empty or single_mae.empty:
            continue

        best_r2 = single_r2.loc[single_r2["_metric"].idxmax()]
        best_rmse = single_rmse.loc[single_rmse["_metric"].idxmin()]
        best_mae = single_mae.loc[single_mae["_metric"].idxmin()]

        fusion_r2 = _to_float_or_nan(fusion_row.get("r2"))
        fusion_rmse = _to_float_or_nan(fusion_row.get("rmse"))
        fusion_mae = _to_float_or_nan(fusion_row.get("mae"))
        best_single_r2 = _to_float_or_nan(best_r2["_metric"])
        best_single_rmse = _to_float_or_nan(best_rmse["_metric"])
        best_single_mae = _to_float_or_nan(best_mae["_metric"])

        rows.append(
            {
                "property": prop,
                "fusion_r2": fusion_r2,
                "best_single_r2": best_single_r2,
                "best_single_r2_representation": str(best_r2["representation"]),
                "delta_r2_fusion_minus_best_single": fusion_r2 - best_single_r2,
                "fusion_rmse": fusion_rmse,
                "best_single_rmse": best_single_rmse,
                "best_single_rmse_representation": str(best_rmse["representation"]),
                "delta_rmse_reduction_best_single_minus_fusion": best_single_rmse - fusion_rmse,
                "fusion_mae": fusion_mae,
                "best_single_mae": best_single_mae,
                "best_single_mae_representation": str(best_mae["representation"]),
                "delta_mae_reduction_best_single_minus_fusion": best_single_mae - fusion_mae,
            }
        )
    return pd.DataFrame(rows)


def _plot_f3_fusion_gain(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    gain_df = _build_f3_fusion_gain_table(metrics_df)
    if gain_df.empty:
        return

    metric_specs = [
        ("delta_r2_fusion_minus_best_single", "R2 Gain (fusion - best single)"),
        ("delta_rmse_reduction_best_single_minus_fusion", "RMSE Reduction (best single - fusion)"),
        ("delta_mae_reduction_best_single_minus_fusion", "MAE Reduction (best single - fusion)"),
    ]
    props = gain_df["property"].astype(str).tolist()
    x = np.arange(len(props), dtype=np.float32)

    # For single property, keep figure compact; for many properties scale width
    panel_w = min(5.8, max(3.2, 1.4 * len(props) + 1.8))
    panel_h = max(3.8, 0.8 * len(props) + 2.6)
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(panel_w * len(metric_specs), panel_h))
    if len(metric_specs) == 1:
        axes = [axes]

    for ax, (col, label_text) in zip(axes, metric_specs):
        values = pd.to_numeric(gain_df[col], errors="coerce").to_numpy(dtype=np.float32)
        colors = ["#59A14F" if np.isfinite(v) and v >= 0 else "#E15759" for v in values]
        # Use narrower bars when few properties so they don't fill the entire panel
        bar_width = min(0.55, max(0.3, 0.6 / max(len(props), 1)))
        bars = ax.bar(x, values, color=colors, alpha=0.9, width=bar_width)
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_ylabel(label_text, fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        finite = values[np.isfinite(values)]
        offset = max(0.01, 0.03 * float(np.max(np.abs(finite)))) if finite.size else 0.01
        # Adjust x-axis limits so narrow bars are not flush with edges
        ax.set_xlim(-0.8, len(props) - 0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(props, rotation=30, ha="right")
        for bar, val in zip(bars, values):
            if not np.isfinite(val):
                continue
            text_y = float(val) + (offset if val >= 0 else -offset)
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2.0, text_y, f"{val:+.3f}", ha="center", va=va, fontsize=13)

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f3_fusion_gain_vs_best_single")
    plt.close(fig)

    save_csv(gain_df, figures_dir / "figure_f3_fusion_gain_vs_best_single.csv", index=False)


def _load_f3_meta_table(model_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(model_dir.glob("*_meta.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        prop = str(payload.get("property", "")).strip()
        view = str(payload.get("view", "")).strip()
        if not prop or not view:
            continue
        rep = _view_to_representation(view)
        num_samples = _to_float_or_nan(payload.get("num_samples"))
        num_common = _to_float_or_nan(payload.get("num_common"))
        effective_samples = num_common if np.isfinite(num_common) else num_samples
        rows.append(
            {
                "property": prop,
                "view": view,
                "representation": rep,
                "num_samples": num_samples,
                "num_common": num_common,
                "effective_samples": effective_samples,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _plot_f3_head_coverage(model_dir: Path, figures_dir: Path) -> None:
    if plt is None:
        return
    meta_df = _load_f3_meta_table(model_dir)
    if meta_df.empty:
        return
    properties = sorted(meta_df["property"].astype(str).unique().tolist())
    fig, axes = plt.subplots(len(properties), 1, figsize=(11, max(3.6, 3.1 * len(properties))), squeeze=False)

    for row_idx, prop in enumerate(properties):
        ax = axes[row_idx, 0]
        sub_prop = meta_df[meta_df["property"] == prop].copy()
        reps = _ordered_representations(sub_prop["representation"].astype(str).unique().tolist())
        if not reps:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        sub_prop = sub_prop.set_index("representation").reindex(reps)
        total = pd.to_numeric(sub_prop["num_samples"], errors="coerce").to_numpy(dtype=np.float32)
        effective = pd.to_numeric(sub_prop["effective_samples"], errors="coerce").to_numpy(dtype=np.float32)
        x = np.arange(len(reps), dtype=np.float32)

        total_for_plot = np.where(np.isfinite(total), total, effective)
        ax.bar(x, total_for_plot, width=0.68, color="#CFCFCF", alpha=0.45, label="num_samples")
        colors = [_representation_color(rep) for rep in reps]
        bars = ax.bar(x, effective, width=0.52, color=colors, alpha=0.92, label="effective_samples")
        ax.set_xticks(x)
        ax.set_xticklabels(reps, rotation=30, ha="right", fontsize=15)
        ax.set_ylabel("Samples")
        ax.grid(axis="y", alpha=0.25)

        for bar, val in zip(bars, effective):
            if not np.isfinite(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(val),
                f"{int(round(float(val)))}",
                ha="center",
                va="bottom",
                fontsize=15,
            )

        if row_idx == 0:
            ax.legend(loc="best", fontsize=15)

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f3_head_coverage")
    plt.close(fig)
    save_csv(meta_df, figures_dir / "figure_f3_head_coverage.csv", index=False)


def _parse_trial_file_info(path: Path) -> Tuple[str, str]:
    stem = path.stem
    suffix = "_mlp_hpo_trials"
    base = stem[: -len(suffix)] if stem.endswith(suffix) else stem

    token_to_rep = [
        ("multiview_mean", "MultiViewMean"),
        ("group_selfies", "Group_SELFIES"),
        ("smiles_bpe", "SMILES_BPE"),
        ("selfies", "SELFIES"),
        ("graph", "Graph"),
        ("smiles", "SMILES"),
    ]
    for token, rep in token_to_rep:
        tail = f"_{token}"
        if base.endswith(tail):
            prop = base[: -len(tail)]
            return prop, rep
    return base, "SMILES"


def _plot_f3_hpo_progress(model_dir: Path, figures_dir: Path) -> None:
    if plt is None:
        return
    trial_files = sorted(model_dir.glob("*_mlp_hpo_trials.csv"))
    if not trial_files:
        return

    by_property: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, str]]] = {}
    for path in trial_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"trial", "objective_metric", "objective_value"}
        if not required.issubset(set(df.columns)):
            continue
        df = df.copy()
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
        df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
        df = df.dropna(subset=["trial", "objective_value"])
        if df.empty:
            continue
        df = df.sort_values("trial")
        metric_name = str(df["objective_metric"].iloc[0]).strip().lower()
        y = df["objective_value"].to_numpy(dtype=np.float32)
        if metric_name == "r2":
            best_curve = np.maximum.accumulate(y)
        else:
            best_curve = np.minimum.accumulate(y)
        x = df["trial"].to_numpy(dtype=np.float32)
        prop, rep = _parse_trial_file_info(path)
        by_property.setdefault(prop, []).append((rep, x, best_curve, metric_name))

    if not by_property:
        return

    properties = sorted(by_property.keys())
    n_props = len(properties)
    fig, axes = plt.subplots(
        n_props, 1,
        figsize=(9, max(4.5, 3.8 * n_props)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, prop in enumerate(properties):
        ax = axes[row_idx, 0]
        lines_data = by_property[prop]
        rep_order = _ordered_representations([item[0] for item in lines_data])
        line_map = {item[0]: item for item in lines_data}
        metrics = set()

        endpoints = []
        for rep in rep_order:
            rep_name, x, best_curve, metric_name = line_map[rep]
            metrics.add(metric_name)
            color = REPRESENTATION_COLORS.get(rep_name, "#888888")
            ax.plot(x, best_curve, color=color, linewidth=2.0, label=rep_name)
            ax.plot(x[-1], best_curve[-1], "o", color=color, markersize=5,
                    zorder=5, markeredgewidth=0.6, markeredgecolor="white")
            endpoints.append((float(best_curve[-1]), float(x[-1]), rep_name, color))

        metric_text = ",".join(sorted(metrics)) if metrics else "objective"
        is_r2 = "r2" in metric_text.lower()
        ylabel_text = "Best R²" if is_r2 else f"Best {metric_text}"

        # Expand x-axis to make room for end-value labels
        x_max = max(ep[1] for ep in endpoints) if endpoints else 1.0
        ax.set_xlim(right=x_max * 1.20)

        # Annotate final best values — nudge vertically to avoid overlap
        all_y_vals = [ep[0] for ep in endpoints]
        y_span = (max(all_y_vals) - min(all_y_vals)) if len(all_y_vals) > 1 else 0.1
        min_gap = max(y_span * 0.06, 0.004)
        placed_y: List[float] = []
        for y_val, _x, _name, color in sorted(endpoints, key=lambda e: e[0], reverse=True):
            y_place = y_val
            for prev in placed_y:
                if abs(y_place - prev) < min_gap:
                    y_place = prev - min_gap
            placed_y.append(y_place)
            ax.text(x_max * 1.03, y_place, f"{y_val:.3f}",
                    color=color, va="center", ha="left")

        ax.set_xlabel("Trial")
        ax.set_ylabel(ylabel_text)
        ax.grid(axis="y", alpha=0.2, linestyle="--", color="gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="lower right",
            framealpha=0.9,
            edgecolor="#cccccc",
            borderpad=0.6,
            handlelength=1.5,
        )

    _save_figure_png(fig, figures_dir / "figure_f3_hpo_progress")
    plt.close(fig)


def _generate_f3_figures(
    *,
    metrics_df: pd.DataFrame,
    model_dir: Path,
    figures_dir: Path,
    prop_cfg: dict,
    args,
) -> None:
    generate = args.generate_figures
    if generate is None:
        generate = _to_bool(prop_cfg.get("generate_figures", True), True)
    if not generate:
        return
    if plt is None:
        print("Warning: matplotlib unavailable; skipping F3 figures.")
        return

    _plot_f3_metrics_by_split(metrics_df, figures_dir)
    _plot_f3_generalization_heatmaps(metrics_df, figures_dir)
    _plot_f3_test_head_leaderboard(metrics_df, figures_dir)
    _plot_f3_fusion_gain(metrics_df, figures_dir)
    _plot_f3_head_coverage(model_dir, figures_dir)
    _plot_f3_hpo_progress(model_dir, figures_dir)


def _resolve_property_columns(df: pd.DataFrame, property_name: str) -> tuple[str, str]:
    smiles_col = "p_smiles" if "p_smiles" in df.columns else "SMILES" if "SMILES" in df.columns else None
    if smiles_col is None:
        raise ValueError("Property CSV must contain SMILES or p_smiles column.")

    if property_name in df.columns:
        value_col = property_name
    else:
        candidates = [c for c in df.columns if c != smiles_col]
        if not candidates:
            raise ValueError("Property CSV must contain a value column.")
        value_col = candidates[0]
    return smiles_col, value_col


def _compute_metrics(y_true, y_pred) -> dict:
    if root_mean_squared_error is not None:
        rmse = float(root_mean_squared_error(y_true, y_pred))
    else:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _normalize_numeric_options(options, default: List[float], cast=float) -> List:
    if options is None:
        options = default
    if isinstance(options, (int, float, str)):
        options = [options]
    normalized = []
    for option in options:
        try:
            normalized.append(cast(option))
        except Exception:
            continue
    return normalized or [cast(v) for v in default]


def _stable_seed(seed: int, tag: str) -> int:
    return int(seed + sum(ord(ch) for ch in tag))


class _PropertyMLP(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int, neurons: int, dropout: float):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(max(int(num_layers), 1)):
            layers.append(torch.nn.Linear(current_dim, int(neurons)))
            layers.append(torch.nn.ReLU())
            if float(dropout) > 0:
                layers.append(torch.nn.Dropout(float(dropout)))
            current_dim = int(neurons)
        layers.append(torch.nn.Linear(current_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_tuning_config(hyper_cfg: dict, force_enable: Optional[bool] = None) -> dict:
    hyper_cfg = hyper_cfg or {}
    search = hyper_cfg.get("search_space", {})

    enabled = bool(hyper_cfg.get("enabled", True)) if force_enable is None else bool(force_enable)
    metric = str(hyper_cfg.get("metric", "r2")).strip().lower()
    if metric not in {"r2", "rmse", "mae"}:
        raise ValueError("hyperparameter_tuning.metric must be one of: r2, rmse, mae")

    cfg = {
        "enabled": enabled,
        "n_trials": max(int(hyper_cfg.get("n_trials", 100)), 1),
        "tuning_epochs": max(int(hyper_cfg.get("tuning_epochs", 35)), 1),
        "tuning_patience": max(int(hyper_cfg.get("tuning_patience", 10)), 1),
        "metric": metric,
        "search_space": {
            "num_layers": _normalize_numeric_options(search.get("num_layers"), [3, 4, 5], cast=int),
            "neurons": _normalize_numeric_options(search.get("neurons"), [64, 128, 256, 512, 1024], cast=int),
            "learning_rate": _normalize_numeric_options(search.get("learning_rate"), [4e-4, 6e-4, 8e-4, 1e-3], cast=float),
            "dropout": _normalize_numeric_options(search.get("dropout"), [0.1, 0.2, 0.3], cast=float),
            "batch_size": _normalize_numeric_options(search.get("batch_size"), [8, 16, 32, 64, 128], cast=int),
        },
    }
    return cfg


def _objective_value(metrics: dict, metric: str) -> tuple[float, bool]:
    if metric == "r2":
        return float(metrics["r2"]), True
    if metric == "rmse":
        return float(metrics["rmse"]), False
    if metric == "mae":
        return float(metrics["mae"]), False
    raise ValueError(f"Unsupported tuning metric: {metric}")


def _predict_torch_model(
    model: torch.nn.Module,
    scaler: StandardScaler,
    features: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    transformed = scaler.transform(features).astype(np.float32, copy=False)
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, transformed.shape[0], batch_size):
            batch = torch.tensor(transformed[start:start + batch_size], device=device, dtype=torch.float32)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _train_one_trial(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    trial_params: dict,
    tuning_epochs: int,
    tuning_patience: int,
    objective_metric: str,
    trial_seed: int,
    device: str,
) -> tuple[torch.nn.Module, StandardScaler, dict, int]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X).astype(np.float32, copy=False)
    val_scaled = scaler.transform(val_X).astype(np.float32, copy=False) if val_X.shape[0] > 0 else np.zeros((0, train_scaled.shape[1]), dtype=np.float32)

    model = _PropertyMLP(
        input_dim=int(train_scaled.shape[1]),
        num_layers=int(trial_params["num_layers"]),
        neurons=int(trial_params["neurons"]),
        dropout=float(trial_params["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(trial_params["learning_rate"]))
    criterion = torch.nn.MSELoss()

    x_train = torch.tensor(train_scaled, dtype=torch.float32)
    y_train = torch.tensor(train_y.astype(np.float32, copy=False), dtype=torch.float32)

    batch_size = max(1, int(trial_params["batch_size"]))
    idx = np.arange(train_scaled.shape[0])
    rng = np.random.default_rng(trial_seed)

    use_val = val_scaled.shape[0] > 0
    score_split = "val" if use_val else "train"

    best_state = None
    best_metrics = None
    best_score = -float("inf") if objective_metric == "r2" else float("inf")
    patience_counter = 0
    best_epoch = 0
    trained_epochs = 0

    for epoch in range(1, int(tuning_epochs) + 1):
        trained_epochs = epoch
        rng.shuffle(idx)
        model.train()
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = x_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        if use_val:
            y_pred = _predict_torch_model(model, scaler, val_X, device=device)
            y_true = val_y
        else:
            y_pred = _predict_torch_model(model, scaler, train_X, device=device)
            y_true = train_y

        metrics = _compute_metrics(y_true, y_pred)
        score, maximize_metric = _objective_value(metrics, objective_metric)
        improved = score > best_score if maximize_metric else score < best_score

        if improved:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= int(tuning_patience):
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_metrics = _compute_metrics(train_y, _predict_torch_model(model, scaler, train_X, device=device))
        best_epoch = int(tuning_epochs)

    model.load_state_dict(best_state)
    return model, scaler, {
        "objective_split": score_split,
        "objective_metric": objective_metric,
        "best_epoch": int(best_epoch),
        "rmse": float(best_metrics["rmse"]),
        "mae": float(best_metrics["mae"]),
        "r2": float(best_metrics["r2"]),
    }, int(trained_epochs)


def _fit_mlp_with_hpo(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    hyper_cfg: dict,
    seed: int,
    device: str,
    force_enable: Optional[bool] = None,
) -> tuple[dict, dict, pd.DataFrame]:
    if train_X.shape[0] == 0:
        raise ValueError("Cannot train MLP: empty training set.")

    cfg = _build_tuning_config(hyper_cfg, force_enable=force_enable)
    enabled = bool(cfg["enabled"])
    n_trials = int(cfg["n_trials"]) if enabled else 1
    search = cfg["search_space"]
    objective_metric = str(cfg["metric"])
    final_training_epochs = max(int((hyper_cfg or {}).get("final_training_epochs", 200)), 1)
    final_training_patience = max(
        int((hyper_cfg or {}).get("final_training_patience", final_training_epochs + 1)),
        1,
    )
    final_train_use_val = bool((hyper_cfg or {}).get("final_train_use_val", False))

    rng = np.random.default_rng(seed)
    best_bundle = None
    best_summary = None
    trial_rows = []

    for trial in range(1, n_trials + 1):
        if enabled:
            params = {
                "num_layers": int(search["num_layers"][int(rng.integers(len(search["num_layers"])))]),
                "neurons": int(search["neurons"][int(rng.integers(len(search["neurons"])))]),
                "learning_rate": float(search["learning_rate"][int(rng.integers(len(search["learning_rate"])))]),
                "dropout": float(search["dropout"][int(rng.integers(len(search["dropout"])))]),
                "batch_size": int(search["batch_size"][int(rng.integers(len(search["batch_size"])))]),
            }
        else:
            params = {
                "num_layers": int(search["num_layers"][0]),
                "neurons": int(search["neurons"][0]),
                "learning_rate": float(search["learning_rate"][0]),
                "dropout": float(search["dropout"][0]),
                "batch_size": int(search["batch_size"][0]),
            }

        model, scaler, trial_result, epochs_trained = _train_one_trial(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            trial_params=params,
            tuning_epochs=int(cfg["tuning_epochs"]),
            tuning_patience=int(cfg["tuning_patience"]),
            objective_metric=objective_metric,
            trial_seed=seed + trial,
            device=device,
        )

        row = {
            "trial": trial,
            "tuning_enabled": enabled,
            "objective_split": trial_result["objective_split"],
            "objective_metric": objective_metric,
            "objective_value": float(trial_result[objective_metric]),
            "val_rmse": float(trial_result["rmse"]),
            "val_mae": float(trial_result["mae"]),
            "val_r2": float(trial_result["r2"]),
            "best_epoch": int(trial_result["best_epoch"]),
            "epochs_trained": int(epochs_trained),
            "activation": "relu",
            **params,
        }
        trial_rows.append(row)

        if best_summary is None:
            is_better = True
        elif objective_metric == "r2":
            is_better = row["objective_value"] > best_summary["objective_value"]
        else:
            is_better = row["objective_value"] < best_summary["objective_value"]

        if is_better:
            best_bundle = {
                "model": model,
                "scaler": scaler,
                "params": params,
            }
            best_summary = row

    if best_bundle is None or best_summary is None:
        raise RuntimeError("MLP HPO failed to produce a valid model.")

    if final_train_use_val and val_X.shape[0] > 0:
        final_train_X = np.concatenate([train_X, val_X], axis=0)
        final_train_y = np.concatenate([train_y, val_y], axis=0)
        final_val_X = np.zeros((0, train_X.shape[1]), dtype=np.float32)
        final_val_y = np.zeros((0,), dtype=np.float32)
    else:
        final_train_X = train_X
        final_train_y = train_y
        final_val_X = val_X
        final_val_y = val_y

    final_model, final_scaler, final_result, final_epochs_ran = _train_one_trial(
        train_X=final_train_X,
        train_y=final_train_y,
        val_X=final_val_X,
        val_y=final_val_y,
        trial_params=best_bundle["params"],
        tuning_epochs=final_training_epochs,
        tuning_patience=final_training_patience,
        objective_metric=objective_metric,
        trial_seed=seed + 1000003,
        device=device,
    )
    final_bundle = {
        "model": final_model,
        "scaler": final_scaler,
        "params": best_bundle["params"],
    }

    hpo_best = {
        "tuning_enabled": enabled,
        "objective_metric": objective_metric,
        "objective_value": float(best_summary["objective_value"]),
        "best_rmse_hpo": float(best_summary["val_rmse"]),
        "best_mae_hpo": float(best_summary["val_mae"]),
        "best_r2_hpo": float(best_summary["val_r2"]),
        "best_epoch_hpo": int(best_summary["best_epoch"]),
        "final_training_epochs": int(final_training_epochs),
        "final_training_patience": int(final_training_patience),
        "final_train_use_val": bool(final_train_use_val),
        "final_objective_split": final_result["objective_split"],
        "final_best_epoch": int(final_result["best_epoch"]),
        "final_rmse": float(final_result["rmse"]),
        "final_mae": float(final_result["mae"]),
        "final_r2": float(final_result["r2"]),
        "final_epochs_ran": int(final_epochs_ran),
        "activation": "relu",
        **final_bundle["params"],
    }

    return final_bundle, hpo_best, pd.DataFrame(trial_rows)


def _save_mlp_bundle(model_path: Path, bundle: dict) -> None:
    model = bundle["model"].to("cpu")
    scaler: StandardScaler = bundle["scaler"]
    params = bundle["params"]
    checkpoint = {
        "format": "mvf_torch_mlp",
        "input_dim": int(model.net[0].in_features),
        "num_layers": int(params["num_layers"]),
        "neurons": int(params["neurons"]),
        "dropout": float(params["dropout"]),
        "activation": "relu",
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }
    torch.save(checkpoint, model_path)


def _save_mlp_bundle_with_legacy(model_path: Path, bundle: dict, legacy_paths: Optional[Iterable[Path]] = None) -> None:
    seen = set()
    candidates = [Path(model_path)]
    if legacy_paths:
        candidates.extend(Path(p) for p in legacy_paths)

    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_mlp_bundle(path, bundle)


def _project_embeddings(model: MultiViewModel, view: str, embeddings: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    if embeddings.size == 0:
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
    step_dirs = ensure_step_dirs(results_dir, "step3_property")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    prop_cfg = config.get("property", {})
    if args.figures_only:
        metrics_candidates = [
            step_dirs["metrics_dir"] / "metrics_property.csv",
            step_dirs["step_dir"] / "metrics_property.csv",
            results_dir / "metrics_property.csv",
        ]
        metrics_path = next((p for p in metrics_candidates if p.exists()), None)
        if metrics_path is None:
            raise FileNotFoundError(
                "No metrics_property.csv found. Run F3 training first or provide the expected metrics file."
            )
        metrics_df = pd.read_csv(metrics_path)
        model_dir = step_dirs["files_dir"]
        if not any(model_dir.glob("*_meta.json")) and any(step_dirs["step_dir"].glob("*_meta.json")):
            model_dir = step_dirs["step_dir"]
        _generate_f3_figures(
            metrics_df=metrics_df,
            model_dir=model_dir,
            figures_dir=step_dirs["figures_dir"],
            prop_cfg=prop_cfg,
            args=args,
        )
        for prop in sorted(metrics_df.get("property", pd.Series(dtype=str)).astype(str).unique().tolist()):
            if not prop:
                continue
            prop_dirs = ensure_step_dirs(results_dir, "step3_property", prop)
            prop_model_dir = prop_dirs["files_dir"]
            if not any(prop_model_dir.glob("*_meta.json")):
                fallback_dir = prop_dirs["step_dir"]
                if any(fallback_dir.glob("*_meta.json")):
                    prop_model_dir = fallback_dir
            if not any(prop_model_dir.glob("*_meta.json")):
                continue
            save_config(config, prop_dirs["files_dir"] / "config_used.yaml")
            prop_metrics = metrics_df[metrics_df["property"].astype(str) == prop].copy()
            if prop_metrics.empty:
                continue
            _generate_f3_figures(
                metrics_df=prop_metrics,
                model_dir=prop_model_dir,
                figures_dir=prop_dirs["figures_dir"],
                prop_cfg=prop_cfg,
                args=args,
            )
        print(f"Saved F3 figures to {step_dirs['figures_dir']}")
        return

    property_dir = _resolve_path(config["paths"]["property_dir"])
    model_type = str(prop_cfg.get("model_type", "mlp")).strip().lower()
    if model_type != "mlp":
        raise ValueError("Only property.model_type='mlp' is supported.")
    file_list = prop_cfg.get("files")
    property_files = _collect_property_files(property_dir, file_list)
    if not property_files:
        raise FileNotFoundError(f"No property CSV files found in {property_dir}")

    views = args.views.split(",") if args.views else prop_cfg.get("views") or config.get("alignment_views", ["smiles"])
    views = [v.strip() for v in views if v.strip()]

    encoder_cfgs = {
        "smiles": config.get("smiles_encoder", {}),
        "smiles_bpe": config.get("smiles_bpe_encoder", {}),
        "selfies": config.get("selfies_encoder", {}),
        "group_selfies": config.get("group_selfies_encoder", {}),
        "graph": config.get("graph_encoder", {}),
    }

    device = "auto"
    for view in views:
        device = encoder_cfgs.get(view, {}).get("device", device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    require_cuda = os.environ.get("MVF_REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes"}
    if require_cuda and device != "cuda":
        raise RuntimeError("MVF_REQUIRE_CUDA is set but CUDA is unavailable.")
    if device == "cpu":
        print("Warning: using CPU for property embedding; this can be very slow.")
    else:
        print("Using CUDA for property embedding.")

    view_specs = {
        "smiles": {
            "type": "sequence",
            "encoder_key": "smiles_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "data" / "tokenizer.py",
            "tokenizer_class": "PSmilesTokenizer",
            "tokenizer_file": "tokenizer.json",
        },
        "smiles_bpe": {
            "type": "sequence",
            "encoder_key": "smiles_bpe_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES_BPE" / "src" / "data" / "tokenizer.py",
            "tokenizer_class": "PSmilesTokenizer",
            "tokenizer_file": "tokenizer.json",
        },
        "selfies": {
            "type": "sequence",
            "encoder_key": "selfies_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SELFIES" / "src" / "data" / "selfies_tokenizer.py",
            "tokenizer_class": "SelfiesTokenizer",
            "tokenizer_file": "tokenizer.json",
        },
        "group_selfies": {
            "type": "sequence",
            "encoder_key": "group_selfies_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_Group_SELFIES" / "src" / "data" / "tokenizer.py",
            "tokenizer_class": "GroupSELFIESTokenizer",
            "tokenizer_file": "tokenizer.pkl",
        },
        "graph": {
            "type": "graph",
            "encoder_key": "graph_encoder",
        },
    }

    view_assets: Dict[str, dict] = {}
    for view in views:
        spec = view_specs.get(view)
        if not spec:
            print(f"Skipping unknown view: {view}")
            continue
        encoder_cfg = encoder_cfgs.get(view, {})
        if not encoder_cfg or not encoder_cfg.get("method_dir"):
            print(f"Skipping view {view}: encoder config missing.")
            continue
        if spec["type"] == "sequence":
            view_assets[view] = _load_sequence_backbone(
                encoder_cfg=encoder_cfg,
                device=device,
                tokenizer_module=spec["tokenizer_module"],
                tokenizer_class=spec["tokenizer_class"],
                tokenizer_filename=spec["tokenizer_file"],
            )
        else:
            view_assets[view] = _load_graph_backbone(
                encoder_cfg=encoder_cfg,
                device=device,
            )

    if not view_assets:
        raise RuntimeError("No valid views configured for property prediction.")

    if args.use_alignment is None:
        use_alignment = bool(prop_cfg.get("use_alignment", False))
    else:
        use_alignment = bool(args.use_alignment)
    alignment_model = None
    if use_alignment:
        ckpt_path = args.alignment_checkpoint
        if ckpt_path:
            ckpt_path = _resolve_path(ckpt_path)
        else:
            ckpt_path = results_dir / "step1_alignment" / "alignment_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Alignment checkpoint not found: {ckpt_path}")
        view_dims = {view: view_assets[view]["backbone"].hidden_size for view in view_assets}
        model_cfg = config.get("model", {})
        alignment_model = MultiViewModel(
            view_dims=view_dims,
            projection_dim=int(model_cfg.get("projection_dim", 256)),
            projection_hidden_dims=model_cfg.get("projection_hidden_dims"),
            dropout=float(model_cfg.get("view_dropout", 0.0)),
        )
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        alignment_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        alignment_model.to(device)
        alignment_model.eval()

    train_ratio = float(prop_cfg.get("train_ratio", 0.8))
    val_ratio = float(prop_cfg.get("val_ratio", 0.1))
    test_ratio = float(prop_cfg.get("test_ratio", 0.1))
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Property split ratios must sum to 1.0")

    seed = int(config.get("data", {}).get("random_seed", 42))
    max_prop_samples = _to_int_or_none(prop_cfg.get("max_samples"))
    mlp_hpo_cfg = prop_cfg.get("hyperparameter_tuning")
    if mlp_hpo_cfg is None:
        mlp_hpo_cfg = config.get("hyperparameter_tuning", {})

    model_dir = step_dirs["files_dir"]
    legacy_model_dir = step_dirs["step_dir"]

    # Cache per-view embeddings by raw p-SMILES across property files.
    view_embedding_cache: Dict[str, Dict[str, np.ndarray]] = {view: {} for view in view_assets}
    view_invalid_cache: Dict[str, set] = {view: set() for view in view_assets}

    rows = []
    rows_by_property: Dict[str, List[dict]] = {}
    for prop_path in property_files:
        df = pd.read_csv(prop_path)
        prop_name = prop_path.stem
        prop_step_dirs = ensure_step_dirs(results_dir, "step3_property", prop_name)
        save_config(config, prop_step_dirs["files_dir"] / "config_used.yaml")
        prop_model_dir = prop_step_dirs["files_dir"]
        prop_legacy_model_dir = prop_step_dirs["step_dir"]
        smiles_col, value_col = _resolve_property_columns(df, prop_name)

        df = df[[smiles_col, value_col]].dropna()
        df = df.rename(columns={smiles_col: "p_smiles", value_col: "target"})
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = df.dropna(subset=["target"])
        if max_prop_samples is not None:
            df = df.head(max_prop_samples)

        if df.empty:
            print(f"Skipping {prop_name}: no valid rows.")
            continue

        smiles_list = df["p_smiles"].astype(str).tolist()
        targets = df["target"].values.astype(np.float32)
        num_samples = len(smiles_list)

        indices = np.arange(num_samples)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )
        val_size = val_ratio / max(val_ratio + test_ratio, 1e-9)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=seed,
            shuffle=True,
        )
        splits = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        }

        view_data: Dict[str, Dict] = {}
        for view, assets in view_assets.items():
            cache = view_embedding_cache[view]
            invalid = view_invalid_cache[view]

            unique_missing = []
            seen = set()
            for smi in smiles_list:
                if smi in seen:
                    continue
                seen.add(smi)
                if smi in cache or smi in invalid:
                    continue
                unique_missing.append(smi)

            if unique_missing:
                if view == "selfies":
                    seq_inputs = []
                    seq_smiles = []
                    for smi in unique_missing:
                        s = smiles_to_selfies(smi)
                        if s:
                            seq_inputs.append(s)
                            seq_smiles.append(smi)
                        else:
                            invalid.add(smi)
                    if seq_inputs:
                        new_emb = _embed_sequence(seq_inputs, assets, device)
                        if use_alignment and alignment_model is not None:
                            new_emb = _project_embeddings(alignment_model, view, new_emb, device)
                        for i, smi in enumerate(seq_smiles):
                            cache[smi] = new_emb[i]
                elif view == "graph":
                    new_emb, valid_indices = _embed_graph(unique_missing, assets, device)
                    if use_alignment and alignment_model is not None and new_emb.size:
                        new_emb = _project_embeddings(alignment_model, view, new_emb, device)
                    valid_set = set(valid_indices)
                    emb_row = 0
                    for i, smi in enumerate(unique_missing):
                        if i in valid_set:
                            cache[smi] = new_emb[emb_row]
                            emb_row += 1
                        else:
                            invalid.add(smi)
                else:
                    new_emb = _embed_sequence(unique_missing, assets, device)
                    if use_alignment and alignment_model is not None:
                        new_emb = _project_embeddings(alignment_model, view, new_emb, device)
                    for i, smi in enumerate(unique_missing):
                        cache[smi] = new_emb[i]

            emb_rows = []
            view_indices = []
            for idx, smi in enumerate(smiles_list):
                emb = cache.get(smi)
                if emb is None:
                    continue
                emb_rows.append(emb)
                view_indices.append(idx)

            if emb_rows:
                embeddings = np.stack(emb_rows, axis=0).astype(np.float32, copy=False)
            else:
                embeddings = np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)

            index_map = {orig_idx: row_idx for row_idx, orig_idx in enumerate(view_indices)}
            view_data[view] = {
                "embeddings": embeddings,
                "index_map": index_map,
                "model_size": assets["model_size"],
            }

        for view, data in view_data.items():
            embeddings = data["embeddings"]
            index_map = data["index_map"]

            if not index_map:
                continue

            train_indices = [i for i in splits["train"] if i in index_map]
            train_rows = [index_map[i] for i in train_indices]
            if not train_rows:
                continue
            val_indices = [i for i in splits["val"] if i in index_map]
            val_rows = [index_map[i] for i in val_indices]

            model_bundle, hpo_best, trial_df = _fit_mlp_with_hpo(
                train_X=embeddings[train_rows],
                train_y=targets[train_indices],
                val_X=embeddings[val_rows] if val_rows else np.zeros((0, embeddings.shape[1]), dtype=np.float32),
                val_y=targets[val_indices] if val_indices else np.zeros((0,), dtype=np.float32),
                hyper_cfg=mlp_hpo_cfg,
                seed=_stable_seed(seed, f"{prop_name}:{view}"),
                device=device,
                force_enable=args.tune,
            )
            model = model_bundle["model"]
            scaler = model_bundle["scaler"]

            for split_name, split_indices in splits.items():
                valid_indices = [i for i in split_indices if i in index_map]
                split_rows = [index_map[i] for i in valid_indices]
                if not split_rows:
                    metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
                else:
                    preds = _predict_torch_model(model, scaler, embeddings[split_rows], device=device)
                    metrics = _compute_metrics(targets[valid_indices], preds)
                rep = "Group_SELFIES" if view == "group_selfies" else "Graph" if view == "graph" else view.upper()
                row = {
                    "method": "Multi_View_Foundation",
                    "representation": rep,
                    "model_size": data["model_size"],
                    "property": prop_name,
                    "split": split_name,
                    **metrics,
                }
                rows.append(row)
                rows_by_property.setdefault(prop_name, []).append(row)

            model_path = prop_model_dir / f"{prop_name}_{view}_mlp.pt"
            if view == "smiles":
                model_path = prop_model_dir / f"{prop_name}_mlp.pt"
            _save_mlp_bundle_with_legacy(
                model_path,
                model_bundle,
                legacy_paths=[
                    model_dir / model_path.name,
                    prop_legacy_model_dir / model_path.name,
                    legacy_model_dir / model_path.name,
                ],
            )

            trial_path = prop_model_dir / f"{prop_name}_{view}_mlp_hpo_trials.csv"
            save_csv(
                trial_df,
                trial_path,
                legacy_paths=[
                    model_dir / trial_path.name,
                    prop_legacy_model_dir / trial_path.name,
                    legacy_model_dir / trial_path.name,
                ],
                index=False,
            )

            meta_path = prop_model_dir / f"{prop_name}_{view}_meta.json"
            save_json(
                {
                    "property": prop_name,
                    "view": view,
                    "num_samples": num_samples,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio,
                    "model_type": "mlp",
                    "hpo_trials": int(mlp_hpo_cfg.get("n_trials", 100)),
                    "hpo_best": hpo_best,
                    "use_alignment": use_alignment,
                    "model_path": str(model_path),
                    "hpo_trials_path": str(trial_path),
                },
                meta_path,
                legacy_paths=[
                    model_dir / meta_path.name,
                    prop_legacy_model_dir / meta_path.name,
                    legacy_model_dir / meta_path.name,
                ],
            )

        if len(view_data) >= 2:
            common = None
            for view, data in view_data.items():
                idx_set = set(data["index_map"].keys())
                common = idx_set if common is None else common.intersection(idx_set)
            common = sorted(common) if common else []
            if common:
                fused_embeddings = []
                for view, data in view_data.items():
                    rows_idx = [data["index_map"][i] for i in common]
                    fused_embeddings.append(data["embeddings"][rows_idx])
                fused = np.mean(np.stack(fused_embeddings), axis=0)
                fused_map = {orig_idx: row_idx for row_idx, orig_idx in enumerate(common)}

                train_indices = [i for i in splits["train"] if i in fused_map]
                train_rows = [fused_map[i] for i in train_indices]
                val_indices = [i for i in splits["val"] if i in fused_map]
                val_rows = [fused_map[i] for i in val_indices]
                if train_rows:
                    model_bundle, hpo_best, trial_df = _fit_mlp_with_hpo(
                        train_X=fused[train_rows],
                        train_y=targets[train_indices],
                        val_X=fused[val_rows] if val_rows else np.zeros((0, fused.shape[1]), dtype=np.float32),
                        val_y=targets[val_indices] if val_indices else np.zeros((0,), dtype=np.float32),
                        hyper_cfg=mlp_hpo_cfg,
                        seed=_stable_seed(seed, f"{prop_name}:multiview_mean"),
                        device=device,
                        force_enable=args.tune,
                    )
                    model = model_bundle["model"]
                    scaler = model_bundle["scaler"]

                    for split_name, split_indices in splits.items():
                        valid_indices = [i for i in split_indices if i in fused_map]
                        split_rows = [fused_map[i] for i in valid_indices]
                        if not split_rows:
                            metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
                        else:
                            preds = _predict_torch_model(model, scaler, fused[split_rows], device=device)
                            metrics = _compute_metrics(targets[valid_indices], preds)
                        row = {
                            "method": "Multi_View_Foundation",
                            "representation": "MultiViewMean",
                            "model_size": "mixed",
                            "property": prop_name,
                            "split": split_name,
                            **metrics,
                        }
                        rows.append(row)
                        rows_by_property.setdefault(prop_name, []).append(row)

                    model_path = prop_model_dir / f"{prop_name}_multiview_mean_mlp.pt"
                    _save_mlp_bundle_with_legacy(
                        model_path,
                        model_bundle,
                        legacy_paths=[
                            model_dir / model_path.name,
                            prop_legacy_model_dir / model_path.name,
                            legacy_model_dir / model_path.name,
                        ],
                    )

                    trial_path = prop_model_dir / f"{prop_name}_multiview_mean_mlp_hpo_trials.csv"
                    save_csv(
                        trial_df,
                        trial_path,
                        legacy_paths=[
                            model_dir / trial_path.name,
                            prop_legacy_model_dir / trial_path.name,
                            legacy_model_dir / trial_path.name,
                        ],
                        index=False,
                    )

                    meta_path = prop_model_dir / f"{prop_name}_multiview_mean_meta.json"
                    save_json(
                        {
                            "property": prop_name,
                            "view": "multiview_mean",
                            "num_samples": num_samples,
                            "num_common": len(common),
                            "train_ratio": train_ratio,
                            "val_ratio": val_ratio,
                            "test_ratio": test_ratio,
                            "model_type": "mlp",
                            "hpo_trials": int(mlp_hpo_cfg.get("n_trials", 100)),
                            "hpo_best": hpo_best,
                            "use_alignment": use_alignment,
                            "model_path": str(model_path),
                            "hpo_trials_path": str(trial_path),
                        },
                        meta_path,
                        legacy_paths=[
                            model_dir / meta_path.name,
                            prop_legacy_model_dir / meta_path.name,
                            legacy_model_dir / meta_path.name,
                        ],
                    )

        prop_metrics_rows = rows_by_property.get(prop_name, [])
        if prop_metrics_rows:
            prop_metrics_df = pd.DataFrame(prop_metrics_rows)
            save_csv(
                prop_metrics_df,
                prop_step_dirs["metrics_dir"] / "metrics_property.csv",
                legacy_paths=[prop_step_dirs["step_dir"] / "metrics_property.csv"],
                index=False,
            )
            _generate_f3_figures(
                metrics_df=prop_metrics_df,
                model_dir=prop_model_dir,
                figures_dir=prop_step_dirs["figures_dir"],
                prop_cfg=prop_cfg,
                args=args,
            )

    metrics_df = pd.DataFrame(rows)
    save_csv(
        metrics_df,
        step_dirs["metrics_dir"] / "metrics_property.csv",
        legacy_paths=[results_dir / "metrics_property.csv"],
        index=False,
    )
    _generate_f3_figures(
        metrics_df=metrics_df,
        model_dir=model_dir,
        figures_dir=step_dirs["figures_dir"],
        prop_cfg=prop_cfg,
        args=args,
    )
    print(f"Saved metrics_property.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--use_alignment", dest="use_alignment", action="store_true")
    parser.add_argument("--no_alignment", dest="use_alignment", action="store_false")
    parser.set_defaults(use_alignment=None)
    parser.add_argument("--tune", dest="tune", action="store_true")
    parser.add_argument("--no_tune", dest="tune", action="store_false")
    parser.set_defaults(tune=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    parser.add_argument("--figures_only", action="store_true", help="Regenerate F3 figures from existing metrics/models without retraining.")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    parser.add_argument("--views", type=str, default=None, help="comma-separated list of views")
    main(parser.parse_args())
