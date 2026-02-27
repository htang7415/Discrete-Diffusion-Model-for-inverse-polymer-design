#!/usr/bin/env python
"""F1: Extract backbone embeddings for multiple views."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import importlib.util
from typing import Any, List

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json, save_numpy

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


def _resolve_with_base(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


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


def _load_f1_meta_table(results_dir: Path, step_dirs: dict) -> pd.DataFrame:
    rows = []
    files_dir = step_dirs["files_dir"]
    for path in sorted(files_dir.glob("embedding_meta_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        view = str(payload.get("view", "")).strip()
        if not view:
            continue
        rows.append(
            {
                "view": view,
                "model_size": str(payload.get("model_size", "")),
                "d1_samples": float(payload.get("d1_samples", 0) or 0),
                "d2_samples": float(payload.get("d2_samples", 0) or 0),
                "embedding_dim": float(payload.get("embedding_dim", 0) or 0),
                "d1_time_sec": float(payload.get("d1_time_sec", 0) or 0),
                "d2_time_sec": float(payload.get("d2_time_sec", 0) or 0),
            }
        )
    if not rows:
        for path in sorted(results_dir.glob("embedding_meta_*.json")):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            view = str(payload.get("view", "")).strip()
            if not view:
                continue
            rows.append(
                {
                    "view": view,
                    "model_size": str(payload.get("model_size", "")),
                    "d1_samples": float(payload.get("d1_samples", 0) or 0),
                    "d2_samples": float(payload.get("d2_samples", 0) or 0),
                    "embedding_dim": float(payload.get("embedding_dim", 0) or 0),
                    "d1_time_sec": float(payload.get("d1_time_sec", 0) or 0),
                    "d2_time_sec": float(payload.get("d2_time_sec", 0) or 0),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["total_time_sec"] = pd.to_numeric(df["d1_time_sec"], errors="coerce").fillna(0.0) + pd.to_numeric(df["d2_time_sec"], errors="coerce").fillna(0.0)
    return df.sort_values("view").reset_index(drop=True)


def _plot_f1_embedding_summary(meta_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or meta_df.empty:
        return
    views = meta_df["view"].astype(str).tolist()
    x = np.arange(len(views), dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    d1 = pd.to_numeric(meta_df["d1_samples"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    d2 = pd.to_numeric(meta_df["d2_samples"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    width = 0.36
    ax0.bar(x - width / 2.0, d1, width=width, color="#4E79A7", alpha=0.9, label="D1")
    ax0.bar(x + width / 2.0, d2, width=width, color="#F28E2B", alpha=0.9, label="D2")
    ax0.set_xticks(x)
    ax0.set_xticklabels(views, rotation=30, ha="right", fontsize=15)
    ax0.set_ylabel("Samples")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="best", fontsize=15)

    total_time = pd.to_numeric(meta_df["total_time_sec"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    bars = ax1.bar(views, total_time, color="#59A14F", alpha=0.9)
    ax1.set_ylabel("Seconds (D1 + D2)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, total_time):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, float(val), f"{float(val):.1f}", ha="center", va="bottom", fontsize=15)

    # Encoding throughput: (D1 + D2 samples) / total_time_sec
    total_samples = d1 + d2
    speed = np.where(total_time > 0, total_samples / total_time, 0.0).astype(np.float32)
    bars2 = ax2.bar(views, speed, color="#B07AA1", alpha=0.9)
    ax2.set_ylabel("Throughput (samples/sec)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars2, speed):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, float(val),
                     f"{float(val):.0f}", ha="center", va="bottom", fontsize=11)

    # Summary table
    ax3.axis("off")
    model_sizes_list = meta_df["model_size"].astype(str).tolist()
    emb_dims = pd.to_numeric(meta_df["embedding_dim"], errors="coerce").fillna(0).astype(int).tolist()
    table_data = [
        [v, f"{int(d1_v):,}", f"{int(d2_v):,}", f"{dim}", m]
        for v, d1_v, d2_v, dim, m in zip(views, d1, d2, emb_dims, model_sizes_list)
    ]
    col_labels = ["View", "D1 samples", "D2 samples", "Dim", "Model size"]
    tbl = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)
    for col_idx in range(len(col_labels)):
        tbl[(0, col_idx)].set_facecolor("#4E79A7")
        tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")
    ax3.set_title(f"Embedding summary  (total time: {float(np.sum(total_time)):.1f}s)", fontsize=12, pad=4)

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f1_embedding_summary")
    plt.close(fig)


def _plot_alignment_loss_curve(curve_path: Path, output_base: Path) -> bool:
    if plt is None or not curve_path.exists():
        return False
    try:
        df = pd.read_csv(curve_path)
    except Exception:
        return False
    if df.empty or "epoch" not in df.columns:
        return False
    train_loss = pd.to_numeric(df.get("train_loss", pd.Series(dtype=float)), errors="coerce")
    val_loss = pd.to_numeric(df.get("val_loss", pd.Series(dtype=float)), errors="coerce")
    epochs = pd.to_numeric(df["epoch"], errors="coerce")
    mask = epochs.notna()
    if not mask.any():
        return False
    x = epochs[mask].to_numpy(dtype=np.float32)
    train_arr = train_loss[mask].to_numpy(dtype=np.float32) if len(train_loss) else np.array([], dtype=np.float32)
    val_arr = val_loss[mask].to_numpy(dtype=np.float32) if len(val_loss) else np.array([], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if train_loss.notna().any() and train_arr.size:
        ax.plot(x, train_arr, label="train_loss", color="#4E79A7", linewidth=1.8)
    if val_loss.notna().any() and val_arr.size:
        ax.plot(x, val_arr, label="val_loss", color="#E15759", linewidth=1.8)
        finite = np.isfinite(val_arr) & np.isfinite(x)
        if finite.any():
            x_finite = x[finite]
            val_finite = val_arr[finite]
            best_idx = int(np.argmin(val_finite))
            best_epoch = float(x_finite[best_idx])
            best_val = float(val_finite[best_idx])
            ax.axvline(best_epoch, color="#111111", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.scatter([best_epoch], [best_val], color="#111111", s=28, zorder=5)
            ax.annotate(
                f"best val: epoch {int(round(best_epoch))}\nloss={best_val:.4f}",
                xy=(best_epoch, best_val),
                xytext=(6, 8),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=15,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="#999999"),
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=15)
    fig.tight_layout()
    _save_figure_png(fig, output_base)
    plt.close(fig)
    return True


def _generate_f1_figures(*, results_dir: Path, step_dirs: dict, cfg_f1: dict, args) -> None:
    generate = args.generate_figures
    if generate is None:
        generate = _to_bool(cfg_f1.get("generate_figures", True), True)
    if not generate:
        return
    if plt is None:
        print("Warning: matplotlib unavailable; skipping F1 figures.")
        return

    meta_df = _load_f1_meta_table(results_dir, step_dirs)
    if not meta_df.empty:
        _plot_f1_embedding_summary(meta_df, step_dirs["figures_dir"])
        save_csv(meta_df, step_dirs["figures_dir"] / "figure_f1_embedding_summary.csv", index=False)

    _plot_alignment_loss_curve(
        results_dir / "step1_alignment" / "alignment_loss_curve.csv",
        step_dirs["figures_dir"] / "figure_f1_alignment_loss_frozen",
    )
    _plot_alignment_loss_curve(
        results_dir / "step1_alignment_e2e" / "alignment_loss_curve.csv",
        step_dirs["figures_dir"] / "figure_f1_alignment_loss_e2e",
    )


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

    incompatible = backbone.load_state_dict(backbone_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Warning: mismatch when loading backbone weights.")
        if incompatible.missing_keys:
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "model_size": model_size or "base",
        "diffusion_steps": diffusion_steps,
    }


def _embed_sequence(
    inputs,
    tokenizer,
    backbone,
    device: str,
    batch_size: int,
    pooling: str,
    timestep: int,
):
    if not inputs:
        return np.zeros((0, backbone.hidden_size), dtype=np.float32)

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **kwargs):
            return x

    embeddings = []
    num_samples = len(inputs)
    for start in tqdm(range(0, num_samples, batch_size), desc="Embedding", leave=False):
        batch = inputs[start:start + batch_size]
        encoded = tokenizer.batch_encode(batch)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        timesteps = torch.full((input_ids.size(0),), int(timestep), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling=pooling,
            )
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


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

    incompatible = backbone.load_state_dict(backbone_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Warning: mismatch when loading graph backbone weights.")
        if incompatible.missing_keys:
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "graph_config_path": graph_config_path,
        "model_size": model_size or "base",
        "diffusion_steps": diffusion_steps,
    }


def _embed_graph(
    smiles_list,
    tokenizer,
    backbone,
    device: str,
    batch_size: int,
    pooling: str,
    timestep: int,
):
    if not smiles_list:
        return np.zeros((0, backbone.hidden_size), dtype=np.float32), []

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **kwargs):
            return x

    valid_indices = []
    graph_batches = []
    for idx, smi in enumerate(smiles_list):
        try:
            data = tokenizer.encode(smi)
            graph_batches.append(data)
            valid_indices.append(idx)
        except Exception:
            continue

    embeddings = []
    for start in tqdm(range(0, len(graph_batches), batch_size), desc="Embedding", leave=False):
        batch = graph_batches[start:start + batch_size]
        if not batch:
            continue
        X = np.stack([b["X"] for b in batch])
        E = np.stack([b["E"] for b in batch])
        M = np.stack([b["M"] for b in batch])
        X_t = torch.tensor(X, device=device)
        E_t = torch.tensor(E, device=device)
        M_t = torch.tensor(M, device=device)
        timesteps = torch.full((X_t.size(0),), int(timestep), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = backbone.get_node_embeddings(X_t, E_t, timesteps, M_t, pooling=pooling)
        embeddings.append(pooled.cpu().numpy())

    if embeddings:
        return np.concatenate(embeddings, axis=0), valid_indices
    return np.zeros((0, backbone.hidden_size), dtype=np.float32), valid_indices


def _prepare_inputs(df: pd.DataFrame, column: str):
    col = df[column].fillna("").astype(str)
    mask = col.str.len() > 0
    return df.loc[mask, "polymer_id"].tolist(), col[mask].tolist()


def _ensure_selfies_column(df: pd.DataFrame) -> pd.DataFrame:
    if "selfies" in df.columns:
        df = df.copy()
        missing = df["selfies"].isna() | (df["selfies"].astype(str).str.len() == 0)
        if missing.any():
            df.loc[missing, "selfies"] = df.loc[missing, "p_smiles"].apply(lambda s: smiles_to_selfies(s) or "")
        return df
    df = df.copy()
    df["selfies"] = df["p_smiles"].apply(lambda s: smiles_to_selfies(s) or "")
    return df


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step1_alignment_embeddings")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")
    cfg_f1 = config.get("alignment_embeddings", {}) or {}

    if args.figures_only:
        _generate_f1_figures(results_dir=results_dir, step_dirs=step_dirs, cfg_f1=cfg_f1, args=args)
        print(f"Saved F1 figures to {step_dirs['figures_dir']}")
        return

    paired_index_path = _resolve_path(config["paths"].get("paired_index", str(results_dir / "paired_index.csv")))
    if not paired_index_path.exists():
        raise FileNotFoundError(f"paired_index.csv not found: {paired_index_path}")

    paired_df = pd.read_csv(paired_index_path)
    if "p_smiles" not in paired_df.columns:
        if "SMILES" in paired_df.columns:
            paired_df = paired_df.rename(columns={"SMILES": "p_smiles"})
        else:
            raise ValueError("paired_index.csv must include a p_smiles column.")

    if "polymer_id" not in paired_df.columns:
        paired_df["polymer_id"] = [f"row_{i}" for i in range(len(paired_df))]

    if "dataset" in paired_df.columns:
        d1_df = paired_df[paired_df["dataset"] == "d1"].copy()
        d2_df = paired_df[paired_df["dataset"] == "d2"].copy()
    else:
        d1_df = paired_df.copy()
        d2_df = paired_df.iloc[0:0].copy()

    views = config.get("alignment_views", ["smiles"])
    data_cfg = config.get("data", {})
    views_cfg = config.get("views", {})
    primary_smiles_view = None
    for candidate in views:
        if candidate in {"smiles", "smiles_bpe"}:
            primary_smiles_view = candidate
            break

    device = "auto"
    for view in views:
        encoder_cfg = config.get(f"{view}_encoder", config.get("smiles_encoder", {}))
        device = encoder_cfg.get("device", device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    require_cuda = os.environ.get("MVF_REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes"}
    if require_cuda and device != "cuda":
        raise RuntimeError("MVF_REQUIRE_CUDA is set but CUDA is unavailable.")
    if device == "cpu":
        print("Warning: using CPU for embedding extraction; this can be very slow.")
    else:
        print("Using CUDA for embedding extraction.")

    view_specs = {
        "smiles": {
            "type": "sequence",
            "input_col": "p_smiles",
            "encoder_key": "smiles_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "data" / "tokenizer.py",
            "tokenizer_class": "PSmilesTokenizer",
            "tokenizer_file": "tokenizer.json",
        },
        "smiles_bpe": {
            "type": "sequence",
            "input_col": "p_smiles",
            "encoder_key": "smiles_bpe_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES_BPE" / "src" / "data" / "tokenizer.py",
            "tokenizer_class": "PSmilesTokenizer",
            "tokenizer_file": "tokenizer.json",
        },
        "selfies": {
            "type": "sequence",
            "input_col": "selfies",
            "encoder_key": "selfies_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SELFIES" / "src" / "data" / "selfies_tokenizer.py",
            "tokenizer_class": "SelfiesTokenizer",
            "tokenizer_file": "tokenizer.json",
        },
        "group_selfies": {
            "type": "sequence",
            "input_col": "p_smiles",
            "encoder_key": "group_selfies_encoder",
            "tokenizer_module": REPO_ROOT / "Bi_Diffusion_Group_SELFIES" / "src" / "data" / "tokenizer.py",
            "tokenizer_class": "GroupSELFIESTokenizer",
            "tokenizer_file": "tokenizer.pkl",
        },
        "graph": {
            "type": "graph",
            "input_col": "p_smiles",
            "encoder_key": "graph_encoder",
        },
    }

    active_views = []
    for view in views:
        if view in views_cfg and not views_cfg[view].get("enabled", True):
            print(f"Skipping {view}: disabled in config.")
            continue
        spec = view_specs.get(view)
        if not spec:
            print(f"Skipping unknown view: {view}")
            continue

        encoder_cfg = config.get(spec["encoder_key"], {})
        if not encoder_cfg or not encoder_cfg.get("method_dir"):
            print(f"Skipping {view}: encoder config missing.")
            continue
        max_d1 = encoder_cfg.get("max_samples_d1") or data_cfg.get("max_samples_d1")
        max_d2 = encoder_cfg.get("max_samples_d2") or data_cfg.get("max_samples_d2")

        view_d1 = d1_df.copy()
        view_d2 = d2_df.copy()
        if view == "selfies":
            view_d1 = _ensure_selfies_column(view_d1)
            view_d2 = _ensure_selfies_column(view_d2)

        if max_d1:
            view_d1 = view_d1.head(int(max_d1))
        if max_d2:
            view_d2 = view_d2.head(int(max_d2))

        print(f"\nEmbedding view: {view}")
        if spec["type"] == "sequence":
            assets = _load_sequence_backbone(
                encoder_cfg=encoder_cfg,
                device=device,
                tokenizer_module=spec["tokenizer_module"],
                tokenizer_class=spec["tokenizer_class"],
                tokenizer_filename=spec["tokenizer_file"],
            )
            pooling = encoder_cfg.get("pooling", "mean")
            timestep = encoder_cfg.get("timestep", 1)
            batch_size = int(encoder_cfg.get("batch_size", 256))

            d1_ids, d1_inputs = _prepare_inputs(view_d1, spec["input_col"])
            d2_ids, d2_inputs = _prepare_inputs(view_d2, spec["input_col"])

            t0 = time.time()
            d1_embeddings = _embed_sequence(d1_inputs, assets["tokenizer"], assets["backbone"], device, batch_size, pooling, timestep)
            d1_time = time.time() - t0
            t1 = time.time()
            d2_embeddings = _embed_sequence(d2_inputs, assets["tokenizer"], assets["backbone"], device, batch_size, pooling, timestep)
            d2_time = time.time() - t1
        else:
            assets = _load_graph_backbone(encoder_cfg=encoder_cfg, device=device)
            pooling = encoder_cfg.get("pooling", "mean")
            timestep = encoder_cfg.get("timestep", 1)
            batch_size = int(encoder_cfg.get("batch_size", 64))

            d1_ids, d1_inputs = _prepare_inputs(view_d1, spec["input_col"])
            d2_ids, d2_inputs = _prepare_inputs(view_d2, spec["input_col"])

            t0 = time.time()
            d1_embeddings, d1_valid = _embed_graph(d1_inputs, assets["tokenizer"], assets["backbone"], device, batch_size, pooling, timestep)
            d1_time = time.time() - t0
            t1 = time.time()
            d2_embeddings, d2_valid = _embed_graph(d2_inputs, assets["tokenizer"], assets["backbone"], device, batch_size, pooling, timestep)
            d2_time = time.time() - t1
            d1_ids = [d1_ids[i] for i in d1_valid]
            d2_ids = [d2_ids[i] for i in d2_valid]

        emb_d1_path = step_dirs["files_dir"] / f"embeddings_{view}_d1.npy"
        emb_d2_path = step_dirs["files_dir"] / f"embeddings_{view}_d2.npy"
        save_numpy(d1_embeddings, emb_d1_path, legacy_paths=[results_dir / f"embeddings_{view}_d1.npy"])
        save_numpy(d2_embeddings, emb_d2_path, legacy_paths=[results_dir / f"embeddings_{view}_d2.npy"])

        if view == primary_smiles_view:
            save_numpy(
                d1_embeddings,
                step_dirs["files_dir"] / "embeddings_d1.npy",
                legacy_paths=[results_dir / "embeddings_d1.npy"],
            )
            save_numpy(
                d2_embeddings,
                step_dirs["files_dir"] / "embeddings_d2.npy",
                legacy_paths=[results_dir / "embeddings_d2.npy"],
            )

        index_rows = []
        for idx, pid in enumerate(d1_ids):
            index_rows.append({"polymer_id": pid, "dataset": "d1", "row_index": idx})
        for idx, pid in enumerate(d2_ids):
            index_rows.append({"polymer_id": pid, "dataset": "d2", "row_index": idx})
        save_csv(
            pd.DataFrame(index_rows),
            step_dirs["files_dir"] / f"embedding_index_{view}.csv",
            legacy_paths=[results_dir / f"embedding_index_{view}.csv"],
            index=False,
        )

        meta = {
            "view": view,
            "model_size": assets.get("model_size", "base"),
            "checkpoint_path": str(assets.get("checkpoint_path", "")),
            "tokenizer_path": str(assets.get("tokenizer_path", "")),
            "pooling": pooling,
            "timestep": int(timestep),
            "device": device,
            "d1_samples": int(d1_embeddings.shape[0]),
            "d2_samples": int(d2_embeddings.shape[0]),
            "embedding_dim": int(d1_embeddings.shape[1]) if d1_embeddings.size else int(d2_embeddings.shape[1]) if d2_embeddings.size else 0,
            "d1_time_sec": round(d1_time, 2),
            "d2_time_sec": round(d2_time, 2),
        }
        meta_path = step_dirs["files_dir"] / f"embedding_meta_{view}.json"
        save_json(meta, meta_path, legacy_paths=[results_dir / f"embedding_meta_{view}.json"])

        if view == primary_smiles_view:
            save_json(
                meta,
                step_dirs["files_dir"] / "embedding_meta.json",
                legacy_paths=[results_dir / "embedding_meta.json"],
            )

        print(f"Saved embeddings for {view} to {emb_d1_path} and {emb_d2_path}")
        active_views.append(view)

    _generate_f1_figures(results_dir=results_dir, step_dirs=step_dirs, cfg_f1=cfg_f1, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    parser.add_argument("--figures_only", action="store_true", help="Regenerate F1 figures from existing outputs without recomputing embeddings.")
    main(parser.parse_args())
