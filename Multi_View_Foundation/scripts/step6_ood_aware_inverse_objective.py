#!/usr/bin/env python
"""F6: Conservative OOD-aware inverse objective on scored candidates.

This step ranks candidate polymers with a joint objective:
  target property violation + OOD distance + predictive uncertainty
  + optional soft constraints on other properties + SA penalty.
Lower objective is better.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.foundation_inverse import (
    compute_property_error,
    compute_property_hits,
)
from src.utils.config import load_config, save_config
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json
from shared.ood_metrics import knn_distances

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors, rdMolDescriptors
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    rdBase = None


SUPPORTED_VIEWS = ("smiles", "smiles_bpe", "selfies", "group_selfies", "graph")
SUPPORTED_DESCRIPTOR_CONSTRAINTS = {
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "sa_score",
}


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

_STEP5_MODULE = None


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid integer value.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid float value.")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


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


def _plot_f6_objective_diagnostics(
    *,
    valid_df: pd.DataFrame,
    top_df: pd.DataFrame,
    property_name: str,
    target_mode: str,
    normalized_term_weights: Dict[str, float],
    figures_dir: Path,
) -> None:
    if plt is None or valid_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    # A) objective distribution.
    obj_all = pd.to_numeric(valid_df.get("conservative_objective", pd.Series(dtype=float)), errors="coerce").dropna()
    obj_top = pd.to_numeric(top_df.get("conservative_objective", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(obj_all):
        ax0.hist(obj_all.to_numpy(dtype=np.float32), bins=40, color="#4E79A7", alpha=0.65, label="All scored")
    if len(obj_top):
        ax0.hist(obj_top.to_numpy(dtype=np.float32), bins=25, color="#E15759", alpha=0.75, label="Top-k")
    ax0.set_xlabel("Objective (lower is better)")
    ax0.set_ylabel("Count")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=15)

    # B) property-vs-OOD trade-off.
    mode = str(target_mode).strip().lower()
    if mode in {"ge", "le"} and "target_excess" in valid_df.columns:
        x = pd.to_numeric(valid_df.get("d2_distance", pd.Series(dtype=float)), errors="coerce")
        y = pd.to_numeric(valid_df.get("target_excess", pd.Series(dtype=float)), errors="coerce")
        x_label = "D2 distance"
        y_label = "Target excess (>=0 is hit)"
    else:
        x = pd.to_numeric(valid_df.get("d2_distance_objective", pd.Series(dtype=float)), errors="coerce")
        y = pd.to_numeric(valid_df.get("property_error_objective", pd.Series(dtype=float)), errors="coerce")
        x_label = "OOD objective term"
        y_label = "Property objective term"
    c = pd.to_numeric(valid_df.get("conservative_objective", pd.Series(dtype=float)), errors="coerce")
    mask = x.notna() & y.notna() & c.notna()
    if mask.any():
        sc = ax1.scatter(x[mask], y[mask], c=c[mask], cmap="viridis", s=18, alpha=0.7)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Conservative objective")
    else:
        ax1.text(0.5, 0.5, "No finite objective points", ha="center", va="center")
        ax1.set_axis_off()

    # C) objective term weights.
    weight_items = [(k, float(v)) for k, v in normalized_term_weights.items() if np.isfinite(float(v))]
    if weight_items:
        labels = [k for k, _ in weight_items]
        values = np.asarray([v for _, v in weight_items], dtype=np.float32)
        bars = ax2.bar(labels, values, color="#59A14F", alpha=0.85)
        ax2.set_ylim(0, max(1.0, float(np.max(values) + 0.05)))
        ax2.set_ylabel("Weight")
        ax2.grid(axis="y", alpha=0.25)
        ax2.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, float(val), f"{float(val):.2f}", ha="center", va="bottom", fontsize=15)
    else:
        ax2.text(0.5, 0.5, "No active objective terms", ha="center", va="center")
        ax2.set_axis_off()

    # D) cumulative hits by objective rank.
    if "property_hit" in valid_df.columns and "conservative_rank" in valid_df.columns:
        ranked = valid_df.sort_values("conservative_rank").copy()
        hits = ranked["property_hit"].astype(bool).to_numpy(dtype=np.int64)
        cum_hits = np.cumsum(hits)
        x_rank = np.arange(1, len(cum_hits) + 1, dtype=np.int64)
        ax3.plot(x_rank, cum_hits, color="#4E79A7", linewidth=2.0)
        ax3.set_xlabel("Objective rank")
        ax3.set_ylabel("Cumulative hits")
        ax3.grid(alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "Missing rank/hit columns", ha="center", va="center")
        ax3.set_axis_off()

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / f"figure_f6_ood_objective_diagnostics_{property_name}")
    plt.close(fig)


def _normalize_property_name(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    return text.strip()


def _parse_property_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(x).strip() for x in value]
    else:
        raw = [str(value).strip()]
    props: list[str] = []
    for item in raw:
        name = _normalize_property_name(item)
        if name and name not in props:
            props.append(name)
    return props


def _parse_view_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.lower() == "all":
            return ["all"]
        raw = [x.strip() for x in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(x).strip() for x in value]
    else:
        raw = [str(value).strip()]

    views: list[str] = []
    for item in raw:
        if not item:
            continue
        low = item.lower()
        if low == "all":
            return ["all"]
        if low not in SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported view '{item}'. Supported: {', '.join(SUPPORTED_VIEWS)}")
        if low not in views:
            views.append(low)
    return views


def _normalize_property_map(raw: Optional[dict]) -> dict:
    out = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        name = _normalize_property_name(key)
        if name:
            out[name] = value
    return out


def _merge_property_maps(*maps: Optional[dict]) -> dict:
    merged = {}
    for mp in maps:
        normalized = _normalize_property_map(mp)
        for key, value in normalized.items():
            merged[key] = value
    return merged


def _normalize_descriptor_constraints(raw: Any) -> Dict[str, dict]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("descriptor_constraints must be a dict mapping descriptor->spec.")

    out: Dict[str, dict] = {}
    for key, value in raw.items():
        name = str(key).strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_DESCRIPTOR_CONSTRAINTS:
            raise ValueError(
                f"Unsupported descriptor constraint '{name}'. "
                f"Supported: {sorted(SUPPORTED_DESCRIPTOR_CONSTRAINTS)}"
            )

        if isinstance(value, dict):
            target = _to_float_or_none(value.get("target"))
            mode = str(value.get("mode", "ge")).strip().lower() or "ge"
            epsilon = _to_float_or_none(value.get("epsilon"))
            weight = _to_float_or_none(value.get("weight"))
        else:
            target = _to_float_or_none(value)
            mode = "ge"
            epsilon = None
            weight = None

        if target is None:
            continue
        if mode not in {"window", "ge", "le"}:
            raise ValueError(f"descriptor constraint mode must be window|ge|le (got {mode}).")
        if epsilon is None:
            epsilon = 0.0
        if weight is None:
            weight = 1.0
        if float(weight) <= 0:
            continue

        out[name] = {
            "target": float(target),
            "mode": mode,
            "epsilon": float(epsilon),
            "weight": float(weight),
        }
    return out


def _smiles_to_mol(smiles: str):
    if Chem is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text.replace("*", "[H]"))
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def _descriptor_value(mol, descriptor_name: str) -> float:
    if mol is None:
        return float("nan")
    if descriptor_name == "ring_count":
        return float(Descriptors.RingCount(mol)) if Descriptors is not None else float("nan")
    if descriptor_name == "aromatic_ring_count":
        return float(rdMolDescriptors.CalcNumAromaticRings(mol)) if rdMolDescriptors is not None else float("nan")
    if descriptor_name == "fraction_csp3":
        return float(rdMolDescriptors.CalcFractionCSP3(mol)) if rdMolDescriptors is not None else float("nan")
    if descriptor_name == "rotatable_bonds":
        return float(Descriptors.NumRotatableBonds(mol)) if Descriptors is not None else float("nan")
    if descriptor_name == "tpsa":
        return float(Descriptors.TPSA(mol)) if Descriptors is not None else float("nan")
    if descriptor_name == "logp":
        return float(Descriptors.MolLogP(mol)) if Descriptors is not None else float("nan")
    if descriptor_name == "sa_score":
        return float("nan")
    raise ValueError(f"Unsupported descriptor_name={descriptor_name}")


def _descriptor_vector_from_smiles(smiles_list: list[str], descriptor_name: str) -> np.ndarray:
    values = np.full((len(smiles_list),), np.nan, dtype=np.float32)
    if descriptor_name == "sa_score":
        return values
    if Chem is None:
        return values
    for idx, text in enumerate(smiles_list):
        mol = _smiles_to_mol(str(text))
        values[idx] = _descriptor_value(mol, descriptor_name)
    return values


def _load_d1_to_d2_mean_distance(results_dir: Path, k: int) -> float:
    metric_candidates = [
        results_dir / "step4_ood" / "metrics" / "metrics_ood.csv",
        results_dir / "metrics_ood.csv",
        results_dir / "step4_ood" / "metrics_ood.csv",
    ]
    for path in metric_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "d1_to_d2_mean_dist" not in df.columns or df.empty:
            continue
        val = _to_float_or_none(df.iloc[0]["d1_to_d2_mean_dist"])
        if val is not None and np.isfinite(val):
            return float(val)

    d1_candidates = [
        results_dir / "step4_ood" / "files" / "embeddings_d1_aligned.npy",
        results_dir / "embeddings_d1_aligned.npy",
        results_dir / "embeddings_d1.npy",
    ]
    d2_candidates = [
        results_dir / "step4_ood" / "files" / "embeddings_d2_aligned.npy",
        results_dir / "embeddings_d2_aligned.npy",
        results_dir / "embeddings_d2.npy",
    ]
    d1_path = next((p for p in d1_candidates if p.exists()), None)
    d2_path = next((p for p in d2_candidates if p.exists()), None)
    if d1_path is None or d2_path is None:
        return float("nan")
    d1 = np.load(d1_path)
    d2 = np.load(d2_path)
    d1_to_d2 = knn_distances(d1, d2, k=max(int(k), 1))
    return float(np.mean(d1_to_d2)) if d1_to_d2.size else float("nan")


def _save_augmented_ood_metrics(
    *,
    results_dir: Path,
    representation: str,
    model_size: str,
    generated_d2_distance: np.ndarray,
    ood_k: int,
) -> None:
    finite = np.asarray(generated_d2_distance, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return

    d1_to_d2_mean = _load_d1_to_d2_mean_distance(results_dir, k=ood_k)
    gen_mean = float(np.mean(finite))
    frac_near = float(np.mean(finite <= d1_to_d2_mean)) if np.isfinite(d1_to_d2_mean) else np.nan

    row = {
        "method": "Multi_View_Foundation",
        "representation": representation,
        "model_size": model_size,
        "d1_to_d2_mean_dist": round(float(d1_to_d2_mean), 4) if np.isfinite(d1_to_d2_mean) else np.nan,
        "generated_to_d2_mean_dist": round(gen_mean, 4),
        "frac_generated_near_d2": round(frac_near, 4) if np.isfinite(frac_near) else np.nan,
    }
    out_path = results_dir / "step4_ood" / "metrics" / "metrics_ood.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(
        pd.DataFrame([row]),
        out_path,
        legacy_paths=[
            results_dir / "metrics_ood.csv",
            results_dir / "step4_ood" / "metrics_ood.csv",
        ],
        index=False,
    )


def _normalize_scores(values: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    norm_mode = str(mode).strip().lower()
    if norm_mode in {"none", ""}:
        return x
    if norm_mode == "rank":
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(x.size, dtype=np.float32)
        denom = max(float(x.size - 1), 1.0)
        return ranks / denom
    if norm_mode != "minmax":
        raise ValueError(f"Unsupported normalization={mode}. Use minmax|rank|none.")
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = x_max - x_min
    if span <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / span


def _compute_target_excess(predictions: np.ndarray, target_value: float, target_mode: str) -> np.ndarray:
    """Signed distance to target boundary; >=0 means meeting directional target."""
    preds = np.asarray(predictions, dtype=np.float32).reshape(-1)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return (float(target_value) - preds).astype(np.float32, copy=False)
    return (preds - float(target_value)).astype(np.float32, copy=False)


def _target_violation_from_excess(target_excess: np.ndarray, target_mode: str) -> np.ndarray:
    mode = str(target_mode).strip().lower()
    excess = np.asarray(target_excess, dtype=np.float32).reshape(-1)
    if mode == "window":
        return np.abs(excess).astype(np.float32, copy=False)
    return np.maximum(0.0, -excess).astype(np.float32, copy=False)


def _resolve_prediction_columns(df: pd.DataFrame, property_name: str) -> tuple[str, Optional[str], Optional[str]]:
    prop = _normalize_property_name(property_name)
    mean_candidates = [f"pred_{prop}_mean", "prediction"]
    std_candidates = [f"pred_{prop}_std", "prediction_std"]
    count_candidates = [f"pred_{prop}_n_models", "prediction_n_models"]

    mean_col = next((c for c in mean_candidates if c in df.columns), None)
    if mean_col is None:
        raise ValueError(
            f"Could not find target prediction column for property={property_name}. "
            f"Expected one of {mean_candidates}."
        )
    std_col = next((c for c in std_candidates if c in df.columns), None)
    count_col = next((c for c in count_candidates if c in df.columns), None)
    return mean_col, std_col, count_col


def _load_step5_module():
    global _STEP5_MODULE
    if _STEP5_MODULE is not None:
        return _STEP5_MODULE

    step5_path = BASE_DIR / "scripts" / "step5_foundation_inverse.py"
    spec = importlib.util.spec_from_file_location("mvf_step5_foundation_inverse", step5_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import step5 module from {step5_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STEP5_MODULE = module
    return module


def _select_encoder_view(config: dict, override: Optional[str], cfg_step6: dict) -> str:
    requested = override
    if requested is None:
        requested = str(cfg_step6.get("encoder_view", "")).strip() or None

    step5 = _load_step5_module()
    return step5._select_encoder_view(config, requested)


def _available_views(config: dict) -> list[str]:
    step5 = _load_step5_module()
    ordered = []
    for view in config.get("alignment_views", list(SUPPORTED_VIEWS)):
        if view in SUPPORTED_VIEWS and view not in ordered:
            ordered.append(view)
    for view in SUPPORTED_VIEWS:
        if view not in ordered:
            ordered.append(view)

    available: list[str] = []
    for view in ordered:
        try:
            if not step5._is_view_enabled(config, view):
                continue
            encoder_key = step5.VIEW_SPECS[view]["encoder_key"]
            if not config.get(encoder_key, {}).get("method_dir"):
                continue
            available.append(view)
        except Exception:
            continue
    return available


def _select_ood_views(
    config: dict,
    override: Optional[str],
    cfg_step6: dict,
    cfg_f5: dict,
    fallback_view: str,
) -> list[str]:
    raw = override
    if raw is None:
        raw = cfg_step6.get("ood_views", None)
    if raw is None or str(raw).strip() == "":
        raw = cfg_f5.get("proposal_views", "all")

    parsed = _parse_view_list(raw)
    available = _available_views(config)
    if not available:
        return [fallback_view]

    if parsed == ["all"] or not parsed:
        selected = list(available)
    else:
        selected = []
        for view in parsed:
            if view not in available:
                raise ValueError(
                    f"ood view '{view}' is unavailable (disabled or encoder config missing)."
                )
            if view not in selected:
                selected.append(view)

    if fallback_view in available and fallback_view not in selected:
        selected.append(fallback_view)
    if not selected:
        selected = [fallback_view]
    return selected


def _view_to_representation(view: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
    }
    return mapping.get(view, view)


def _candidate_scores_paths_for_property(results_dir: Path, property_name: str) -> list[Path]:
    prop = _normalize_property_name(property_name)
    candidates: list[Path] = []
    if prop:
        candidates.append(results_dir / "step5_foundation_inverse" / prop / "files" / f"candidate_scores_{prop}.csv")
        candidates.append(results_dir / "step5_foundation_inverse" / prop / "files" / "candidate_scores.csv")
        candidates.append(results_dir / "step5_foundation_inverse" / prop / f"candidate_scores_{prop}.csv")
        candidates.append(results_dir / "step5_foundation_inverse" / prop / "candidate_scores.csv")
        candidates.append(results_dir / "step5_foundation_inverse" / "files" / f"candidate_scores_{prop}.csv")
        candidates.append(results_dir / "step5_foundation_inverse" / f"candidate_scores_{prop}.csv")
    candidates.append(results_dir / "step5_foundation_inverse" / "files" / "candidate_scores.csv")
    candidates.append(results_dir / "step5_foundation_inverse" / "candidate_scores.csv")
    return candidates


def _resolve_candidate_scores_path(results_dir: Path, property_name: str) -> Path:
    candidates = _candidate_scores_paths_for_property(results_dir, property_name)
    for path in candidates:
        if path.exists():
            return path
    # Return preferred path for error reporting if none exists.
    return candidates[0]


def _compute_d2_distance_single_view(
    *,
    config: dict,
    results_dir: Path,
    smiles_list: list[str],
    encoder_view: str,
    ood_k: int,
    use_alignment: bool,
    alignment_checkpoint: Optional[str],
) -> np.ndarray:
    if not smiles_list:
        return np.zeros((0,), dtype=np.float32)

    step5 = _load_step5_module()
    encoder_key = step5.VIEW_SPECS[encoder_view]["encoder_key"]
    encoder_cfg = config.get(encoder_key, {})
    device = encoder_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assets = step5._load_view_assets(config=config, view=encoder_view, device=device)
    embeddings, kept_indices = step5._embed_candidates(
        view=encoder_view,
        smiles_list=smiles_list,
        assets=assets,
        device=device,
    )

    d2_full = np.full((len(smiles_list),), np.nan, dtype=np.float32)
    if embeddings.size == 0 or not kept_indices:
        return d2_full

    if use_alignment:
        view_hidden_dim = int(getattr(assets["backbone"], "hidden_size", 0)) or int(config.get("model", {}).get("projection_dim", 256))
        alignment_model = step5._load_alignment_model(
            results_dir=results_dir,
            view_dims={encoder_view: view_hidden_dim},
            config=config,
            checkpoint_override=alignment_checkpoint,
        )
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for OOD-aware objective with use_alignment=True")
        projection_device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = step5._project_embeddings(alignment_model, encoder_view, embeddings, device=projection_device)

    d2_embeddings = step5._load_d2_embeddings(results_dir, encoder_view)
    if use_alignment:
        view_hidden_dim = int(getattr(assets["backbone"], "hidden_size", 0)) or int(config.get("model", {}).get("projection_dim", 256))
        alignment_model = step5._load_alignment_model(
            results_dir=results_dir,
            view_dims={encoder_view: view_hidden_dim},
            config=config,
            checkpoint_override=alignment_checkpoint,
        )
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for OOD-aware objective with use_alignment=True")
        projection_device = "cuda" if torch.cuda.is_available() else "cpu"
        d2_embeddings = step5._project_embeddings(alignment_model, encoder_view, d2_embeddings, device=projection_device)

    distances = knn_distances(embeddings, d2_embeddings, k=int(ood_k))
    mean_dist = distances.mean(axis=1).astype(np.float32, copy=False)
    for local_i, global_i in enumerate(kept_indices):
        d2_full[int(global_i)] = mean_dist[local_i]
    return d2_full


def _compute_d2_distance_column(
    *,
    config: dict,
    results_dir: Path,
    smiles_list: list[str],
    ood_views: list[str],
    ood_k: int,
    use_alignment: bool,
    alignment_checkpoint: Optional[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    n = len(smiles_list)
    if n == 0:
        return np.zeros((0,), dtype=np.float32), {}, []

    per_view: dict[str, np.ndarray] = {}
    used_views: list[str] = []
    for view in ood_views:
        try:
            d2_vec = _compute_d2_distance_single_view(
                config=config,
                results_dir=results_dir,
                smiles_list=smiles_list,
                encoder_view=view,
                ood_k=ood_k,
                use_alignment=use_alignment,
                alignment_checkpoint=alignment_checkpoint,
            )
        except FileNotFoundError as exc:
            print(f"[F6] Warning: skipping OOD view '{view}' ({exc})")
            continue
        per_view[view] = np.asarray(d2_vec, dtype=np.float32).reshape(-1)
        used_views.append(view)

    if not used_views:
        raise FileNotFoundError(
            "No usable OOD views for D2-distance computation. "
            "Ensure F1 generated D2 embeddings for at least one enabled view."
        )

    stack = np.column_stack([per_view[v] for v in used_views]).astype(np.float32, copy=False)
    valid_counts = np.sum(np.isfinite(stack), axis=1)
    with np.errstate(invalid="ignore"):
        agg = np.nanmean(stack, axis=1).astype(np.float32, copy=False)
    agg[valid_counts == 0] = np.nan
    return agg, per_view, used_views


def main(args):
    config = load_config(args.config)
    cfg_step6 = config.get("ood_aware_inverse", {}) or {}
    cfg_f5 = config.get("foundation_inverse", {}) or {}

    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step6_ood_aware_inverse")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    encoder_view = _select_encoder_view(config, args.encoder_view, cfg_step6)
    if encoder_view not in SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported encoder_view={encoder_view}. Supported: {', '.join(SUPPORTED_VIEWS)}")
    ood_views = _select_ood_views(
        config=config,
        override=args.ood_views,
        cfg_step6=cfg_step6,
        cfg_f5=cfg_f5,
        fallback_view=encoder_view,
    )

    property_name = args.property
    if property_name is None:
        property_name = str(cfg_step6.get("property", "")).strip() or str(cfg_f5.get("property", "")).strip() or "property"
    property_step_dirs = ensure_step_dirs(results_dir, "step6_ood_aware_inverse", property_name)
    save_config(config, property_step_dirs["files_dir"] / "config_used.yaml")

    candidate_scores_path = _resolve_candidate_scores_path(results_dir, property_name)
    if not candidate_scores_path.exists():
        searched = [str(p) for p in _candidate_scores_paths_for_property(results_dir, property_name)]
        raise FileNotFoundError(
            "Candidate scores file not found for F6 auto-discovery. "
            f"property={property_name}. searched={searched}"
        )

    property_key = _normalize_property_name(property_name)
    target_map = _merge_property_maps(cfg_f5.get("targets"), cfg_step6.get("targets"))
    target_mode_map = _merge_property_maps(cfg_f5.get("target_modes"), cfg_step6.get("target_modes"))
    epsilon_map = _merge_property_maps(cfg_f5.get("epsilons"), cfg_step6.get("epsilons"))

    target = _to_float_or_none(args.target)
    if target is None:
        target = _to_float_or_none(target_map.get(property_key))
    if target is None:
        target = _to_float_or_none(cfg_step6.get("target"))
    if target is None:
        target = _to_float_or_none(cfg_f5.get("target"))
    if target is None:
        raise ValueError(
            "target is required (set --target or per-property targets map "
            "or ood_aware_inverse.target/foundation_inverse.target)."
        )

    target_mode = (
        args.target_mode
        or str(target_mode_map.get(property_key, "")).strip()
        or str(cfg_step6.get("target_mode", "")).strip()
        or str(cfg_f5.get("target_mode", "window")).strip()
        or "window"
    )
    epsilon = _to_float_or_none(args.epsilon)
    if epsilon is None:
        epsilon = _to_float_or_none(epsilon_map.get(property_key))
    if epsilon is None:
        epsilon = _to_float_or_none(cfg_step6.get("epsilon"))
    if epsilon is None:
        epsilon = _to_float_or_none(cfg_f5.get("epsilon"))
    if epsilon is None:
        epsilon = 30.0

    top_k = _to_int_or_none(args.top_k)
    if top_k is None:
        top_k = _to_int_or_none(cfg_step6.get("top_k"))
    if top_k is None:
        top_k = 100
    if int(top_k) <= 0:
        raise ValueError("top_k must be > 0.")
    top_k = int(top_k)

    property_weight = _to_float_or_none(args.property_weight)
    if property_weight is None:
        property_weight = _to_float_or_none(cfg_step6.get("property_weight"))
    if property_weight is None:
        property_weight = 0.6

    ood_weight = _to_float_or_none(args.ood_weight)
    if ood_weight is None:
        ood_weight = _to_float_or_none(cfg_step6.get("ood_weight"))
    if ood_weight is None:
        ood_weight = 0.2

    uncertainty_weight = _to_float_or_none(args.uncertainty_weight)
    if uncertainty_weight is None:
        uncertainty_weight = _to_float_or_none(cfg_step6.get("uncertainty_weight"))
    if uncertainty_weight is None:
        uncertainty_weight = 0.15

    constraint_weight = _to_float_or_none(args.constraint_weight)
    if constraint_weight is None:
        constraint_weight = _to_float_or_none(cfg_step6.get("constraint_weight"))
    if constraint_weight is None:
        constraint_weight = 0.05

    sa_weight = _to_float_or_none(args.sa_weight)
    if sa_weight is None:
        sa_weight = _to_float_or_none(cfg_step6.get("sa_weight"))
    if sa_weight is None:
        sa_weight = 0.0

    descriptor_weight = _to_float_or_none(args.descriptor_weight)
    if descriptor_weight is None:
        descriptor_weight = _to_float_or_none(cfg_step6.get("descriptor_weight"))
    if descriptor_weight is None:
        descriptor_weight = 0.0

    descriptor_constraints = _normalize_descriptor_constraints(cfg_step6.get("descriptor_constraints"))

    normalization = str(args.normalization or cfg_step6.get("normalization", "minmax")).strip().lower()
    ood_k = _to_int_or_none(args.ood_k)
    if ood_k is None:
        ood_k = _to_int_or_none(cfg_step6.get("ood_k"))
    if ood_k is None:
        ood_k = _to_int_or_none(cfg_f5.get("ood_k"))
    if ood_k is None:
        ood_k = 5
    ood_k = int(ood_k)
    if ood_k <= 0:
        raise ValueError("ood_k must be > 0.")

    target_map[property_key] = float(target)
    target_mode_map[property_key] = str(target_mode).strip().lower()

    constraint_weights_map = _normalize_property_map(cfg_step6.get("constraint_weights"))
    constraint_properties = _parse_property_list(args.constraint_properties)
    if not constraint_properties:
        constraint_properties = _parse_property_list(cfg_step6.get("constraint_properties"))

    if args.use_alignment is None:
        use_alignment = _to_bool(cfg_step6.get("use_alignment", cfg_f5.get("use_alignment", False)), False)
    else:
        use_alignment = bool(args.use_alignment)

    alignment_checkpoint = args.alignment_checkpoint
    if alignment_checkpoint:
        alignment_checkpoint = str(_resolve_path(alignment_checkpoint))
    elif str(cfg_step6.get("alignment_checkpoint", "")).strip():
        alignment_checkpoint = str(_resolve_path(str(cfg_step6.get("alignment_checkpoint", ""))))
    elif str(cfg_f5.get("alignment_checkpoint", "")).strip():
        alignment_checkpoint = str(_resolve_path(str(cfg_f5.get("alignment_checkpoint", ""))))
    else:
        alignment_checkpoint = None

    compute_if_missing = args.compute_d2_distance_if_missing
    if compute_if_missing is None:
        compute_if_missing = _to_bool(cfg_step6.get("compute_d2_distance_if_missing", True), True)
    force_recompute = bool(args.recompute_d2_distance)

    df = pd.read_csv(candidate_scores_path)
    if "smiles" not in df.columns:
        raise ValueError(f"Candidate scores must include 'smiles' column: {candidate_scores_path}")

    df = df.copy()
    if "property" in df.columns:
        prop_series = df["property"].astype(str).str.strip()
        match_mask = prop_series == property_name
        if not bool(match_mask.any()):
            seen = [x for x in sorted(prop_series.unique().tolist()) if x]
            seen_preview = ",".join(seen[:6]) if seen else "(empty)"
            raise RuntimeError(
                "Candidate scores property mismatch: "
                f"requested property={property_name} but file has {seen_preview}. "
                "Use the correct candidate_scores_<PROPERTY>.csv for this run."
            )
        df = df.loc[match_mask].copy()

    target_pred_col, target_unc_col, target_n_models_col = _resolve_prediction_columns(df, property_name)
    df[target_pred_col] = pd.to_numeric(df[target_pred_col], errors="coerce")
    if target_unc_col is not None:
        df[target_unc_col] = pd.to_numeric(df[target_unc_col], errors="coerce")
    if target_n_models_col is not None:
        df[target_n_models_col] = pd.to_numeric(df[target_n_models_col], errors="coerce")

    # Keep canonical aliases for downstream consumers.
    df["prediction"] = pd.to_numeric(df[target_pred_col], errors="coerce")
    if target_unc_col is not None:
        df["prediction_uncertainty"] = pd.to_numeric(df[target_unc_col], errors="coerce")
    elif "prediction_uncertainty" not in df.columns:
        df["prediction_uncertainty"] = np.nan

    d2_source = "existing"
    ood_views_used = []
    has_d2 = "d2_distance" in df.columns
    if has_d2:
        df["d2_distance"] = pd.to_numeric(df["d2_distance"], errors="coerce")
        for view in ood_views:
            col = f"d2_distance_{view}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    needs_compute = force_recompute
    if not needs_compute and compute_if_missing:
        needs_compute = (not has_d2) or bool(df["d2_distance"].isna().any())
        if (not needs_compute) and len(ood_views) > 1:
            present_cols = [f"d2_distance_{v}" for v in ood_views if f"d2_distance_{v}" in df.columns]
            if len(present_cols) < len(ood_views):
                needs_compute = True

    if needs_compute:
        d2_source = "recomputed_multiview" if len(ood_views) > 1 else "recomputed"
        smiles_list = df["smiles"].astype(str).tolist()
        d2_values, d2_per_view, used_views = _compute_d2_distance_column(
            config=config,
            results_dir=results_dir,
            smiles_list=smiles_list,
            ood_views=ood_views,
            ood_k=ood_k,
            use_alignment=use_alignment,
            alignment_checkpoint=alignment_checkpoint,
        )
        df["d2_distance"] = d2_values
        for view_name, values in d2_per_view.items():
            df[f"d2_distance_{view_name}"] = values
        ood_views_used = list(used_views)
    elif not has_d2:
        raise ValueError(
            "d2_distance column is missing and compute_d2_distance_if_missing is disabled. "
            "Enable compute or provide d2_distance in candidate_scores.csv."
        )
    else:
        ood_views_used = [v for v in ood_views if f"d2_distance_{v}" in df.columns]
        if not ood_views_used:
            ood_views_used = [encoder_view]

    valid_mask = (
        df["prediction"].notna()
        & pd.to_numeric(df["d2_distance"], errors="coerce").notna()
    )

    valid_df = df.loc[valid_mask].copy()
    dropped = int((~valid_mask).sum())
    if valid_df.empty:
        raise RuntimeError("No valid candidates remain after filtering for prediction and d2_distance.")

    pred = valid_df["prediction"].to_numpy(dtype=np.float32)
    d2 = valid_df["d2_distance"].to_numpy(dtype=np.float32)
    pred_unc = pd.to_numeric(valid_df.get("prediction_uncertainty", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float32)
    if pred_unc.size:
        pred_unc = np.nan_to_num(pred_unc, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        pred_unc = np.zeros((len(valid_df),), dtype=np.float32)
    if pred_unc.size:
        unc_span = float(np.max(pred_unc) - np.min(pred_unc))
        uncertainty_has_signal = bool(np.isfinite(unc_span) and unc_span > 1e-8)
    else:
        uncertainty_has_signal = False
    if (not uncertainty_has_signal) and float(uncertainty_weight) > 0:
        print("[F6] Warning: uncertainty term disabled because prediction_uncertainty has no variation.")

    hit_mask = compute_property_hits(pred, target_value=float(target), epsilon=float(epsilon), target_mode=target_mode)
    target_excess = _compute_target_excess(pred, target_value=float(target), target_mode=target_mode)
    target_violation_raw = _target_violation_from_excess(target_excess, target_mode=target_mode)
    property_error_raw = compute_property_error(pred, target_value=float(target), target_mode=target_mode)
    property_error_obj = _normalize_scores(property_error_raw, normalization)
    d2_obj = _normalize_scores(d2, normalization)
    if uncertainty_has_signal:
        uncertainty_obj = _normalize_scores(pred_unc, normalization)
    else:
        uncertainty_obj = np.zeros((len(valid_df),), dtype=np.float32)
    effective_uncertainty_weight = float(uncertainty_weight) if uncertainty_has_signal else 0.0

    available_committee_props = sorted(
        {
            _normalize_property_name(col[len("pred_"):-len("_mean")])
            for col in valid_df.columns
            if col.startswith("pred_") and col.endswith("_mean")
        }
    )
    if not constraint_properties:
        constraint_properties = [
            name for name in available_committee_props
            if name and name != _normalize_property_name(property_name) and name in target_map
        ]

    constraint_raw_total = np.zeros((len(valid_df),), dtype=np.float32)
    constraint_obj = np.zeros((len(valid_df),), dtype=np.float32)
    active_constraint_props: list[str] = []
    raw_den = 0.0
    obj_den = 0.0
    for prop_name in constraint_properties:
        if prop_name == _normalize_property_name(property_name):
            continue
        col = f"pred_{prop_name}_mean"
        if col not in valid_df.columns:
            continue
        target_value_prop = _to_float_or_none(target_map.get(prop_name))
        if target_value_prop is None:
            continue
        mode_prop = str(target_mode_map.get(prop_name, "window")).strip().lower() or "window"
        prop_pred = pd.to_numeric(valid_df[col], errors="coerce").to_numpy(dtype=np.float32)
        finite_prop = prop_pred[np.isfinite(prop_pred)]
        fill_prop = float(np.median(finite_prop)) if finite_prop.size else 0.0
        prop_pred = np.nan_to_num(prop_pred, nan=fill_prop, posinf=fill_prop, neginf=fill_prop)
        violation_raw = compute_property_error(prop_pred, target_value=float(target_value_prop), target_mode=mode_prop)
        violation_obj = _normalize_scores(violation_raw, normalization)
        valid_df[f"constraint_violation_{prop_name}"] = violation_raw
        w_prop = _to_float_or_none(constraint_weights_map.get(prop_name))
        if w_prop is None:
            w_prop = 1.0
        if w_prop <= 0:
            continue
        constraint_raw_total += float(w_prop) * violation_raw
        constraint_obj += float(w_prop) * violation_obj
        raw_den += float(w_prop)
        obj_den += float(w_prop)
        active_constraint_props.append(prop_name)
    if raw_den > 0:
        constraint_raw_total = (constraint_raw_total / raw_den).astype(np.float32, copy=False)
    if obj_den > 0:
        constraint_obj = (constraint_obj / obj_den).astype(np.float32, copy=False)

    if "sa_score" in valid_df.columns:
        sa_raw = pd.to_numeric(valid_df["sa_score"], errors="coerce").to_numpy(dtype=np.float32)
        if sa_raw.size:
            finite_sa = sa_raw[np.isfinite(sa_raw)]
            fill_sa = float(np.nanmedian(finite_sa)) if finite_sa.size else 0.0
            sa_raw = np.nan_to_num(sa_raw, nan=fill_sa, posinf=fill_sa, neginf=fill_sa)
        else:
            sa_raw = np.zeros((len(valid_df),), dtype=np.float32)
    else:
        sa_raw = np.zeros((len(valid_df),), dtype=np.float32)
    sa_obj = _normalize_scores(sa_raw, normalization)

    descriptor_raw_total = np.zeros((len(valid_df),), dtype=np.float32)
    descriptor_obj = np.zeros((len(valid_df),), dtype=np.float32)
    active_descriptor_constraints: list[str] = []
    descriptor_den = 0.0
    if descriptor_constraints and float(descriptor_weight) > 0:
        smiles_values = valid_df["smiles"].astype(str).tolist()
        for descriptor_name, spec in descriptor_constraints.items():
            if descriptor_name == "sa_score" and "sa_score" in valid_df.columns:
                descriptor_values = pd.to_numeric(valid_df["sa_score"], errors="coerce").to_numpy(dtype=np.float32)
            else:
                descriptor_values = _descriptor_vector_from_smiles(smiles_values, descriptor_name)
            finite_desc = descriptor_values[np.isfinite(descriptor_values)]
            if finite_desc.size == 0:
                continue
            fill_desc = float(np.nanmedian(finite_desc))
            descriptor_values = np.nan_to_num(descriptor_values, nan=fill_desc, posinf=fill_desc, neginf=fill_desc)
            valid_df[f"descriptor_{descriptor_name}"] = descriptor_values

            violation_raw = compute_property_error(
                descriptor_values,
                target_value=float(spec["target"]),
                target_mode=str(spec["mode"]),
            )
            violation_obj = _normalize_scores(violation_raw, normalization)
            valid_df[f"descriptor_violation_{descriptor_name}"] = violation_raw

            w_desc = float(spec["weight"])
            if w_desc <= 0:
                continue
            descriptor_raw_total += w_desc * violation_raw
            descriptor_obj += w_desc * violation_obj
            descriptor_den += w_desc
            active_descriptor_constraints.append(descriptor_name)
    if descriptor_den > 0:
        descriptor_raw_total = (descriptor_raw_total / descriptor_den).astype(np.float32, copy=False)
        descriptor_obj = (descriptor_obj / descriptor_den).astype(np.float32, copy=False)
    elif descriptor_constraints and float(descriptor_weight) > 0:
        print("[F6] Warning: descriptor constraints configured but no valid descriptor values were available.")

    term_specs = [
        ("property", property_error_obj, float(property_weight)),
        ("ood", d2_obj, float(ood_weight)),
        ("uncertainty", uncertainty_obj, float(effective_uncertainty_weight)),
        ("constraint", constraint_obj, float(constraint_weight) if active_constraint_props else 0.0),
        ("sa", sa_obj, float(sa_weight) if "sa_score" in valid_df.columns else 0.0),
        ("descriptor", descriptor_obj, float(descriptor_weight) if active_descriptor_constraints else 0.0),
    ]
    active_terms = [(name, values, weight) for name, values, weight in term_specs if float(weight) > 0]
    if not active_terms:
        active_terms = [("property", property_error_obj, 0.7), ("ood", d2_obj, 0.3)]
    total_w = sum(float(weight) for _, _, weight in active_terms)
    normalized_term_weights = {name: float(weight) / max(total_w, 1e-12) for name, _, weight in active_terms}

    objective = np.zeros((len(valid_df),), dtype=np.float32)
    for name, values, _ in active_terms:
        objective += float(normalized_term_weights[name]) * np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(objective, kind="mergesort")
    objective_rank = np.empty_like(order, dtype=np.int64)
    objective_rank[order] = np.arange(objective.shape[0], dtype=np.int64)

    valid_df["property_hit"] = hit_mask.astype(bool)
    valid_df["target_excess"] = target_excess
    valid_df["target_violation"] = target_violation_raw
    valid_df["property_error_normed"] = property_error_raw
    valid_df["property_error_objective"] = property_error_obj
    valid_df["d2_distance_objective"] = d2_obj
    valid_df["prediction_uncertainty"] = pred_unc
    valid_df["uncertainty_objective"] = uncertainty_obj
    valid_df["constraint_violation_total"] = constraint_raw_total
    valid_df["constraint_objective"] = constraint_obj
    valid_df["sa_objective"] = sa_obj
    valid_df["descriptor_violation_total"] = descriptor_raw_total
    valid_df["descriptor_objective"] = descriptor_obj
    valid_df["conservative_objective"] = objective
    valid_df["conservative_rank"] = objective_rank
    # Backward-compatibility aliases.
    valid_df["ood_aware_objective"] = objective
    valid_df["ood_aware_rank"] = objective_rank
    valid_df["property"] = property_name

    order = np.argsort(valid_df["conservative_objective"].to_numpy(dtype=np.float32))
    k = min(top_k, len(order))
    top_idx = order[:k]
    top_df = valid_df.iloc[top_idx].copy()
    top_df["property"] = property_name

    # Baselines for comparison: property-only and OOD-only ranking.
    prop_order = np.argsort(valid_df["property_error_objective"].to_numpy(dtype=np.float32))
    ood_order = np.argsort(valid_df["d2_distance_objective"].to_numpy(dtype=np.float32))
    top_prop = valid_df.iloc[prop_order[:k]]
    top_ood = valid_df.iloc[ood_order[:k]]

    model_size = config.get(f"{encoder_view}_encoder", {}).get("model_size", "base")
    top_hits = int(top_df["property_hit"].sum()) if k > 0 else 0
    top_hit_rate = float(top_hits / max(k, 1))
    representation = "MultiView" if len(set(ood_views_used)) > 1 else _view_to_representation(encoder_view)

    metrics_row = {
        "method": "Multi_View_Foundation",
        "representation": representation,
        "model_size": model_size,
        "property": property_name,
        "target_value": float(target),
        "target_mode": target_mode,
        "epsilon": float(epsilon),
        "normalization": normalization,
        "prediction_column": target_pred_col,
        "prediction_uncertainty_column": target_unc_col or "",
        "prediction_n_models_column": target_n_models_col or "",
        "objective_property_weight": float(normalized_term_weights.get("property", 0.0)),
        "objective_ood_weight": float(normalized_term_weights.get("ood", 0.0)),
        "objective_uncertainty_weight": float(normalized_term_weights.get("uncertainty", 0.0)),
        "objective_constraint_weight": float(normalized_term_weights.get("constraint", 0.0)),
        "objective_sa_weight": float(normalized_term_weights.get("sa", 0.0)),
        "objective_descriptor_weight": float(normalized_term_weights.get("descriptor", 0.0)),
        "n_constraint_properties": int(len(active_constraint_props)),
        "constraint_properties": ",".join(active_constraint_props),
        "n_descriptor_constraints": int(len(active_descriptor_constraints)),
        "descriptor_constraints": ",".join(active_descriptor_constraints),
        "ood_views_requested": ",".join(ood_views),
        "ood_views_used": ",".join(ood_views_used),
        "ood_view_aggregation": "mean",
        "ood_k": int(ood_k),
        "use_alignment_for_ood": bool(use_alignment),
        "d2_distance_source": d2_source,
        "candidate_scores_path": str(candidate_scores_path),
        "n_candidates_total": int(len(df)),
        "n_candidates_scored": int(len(valid_df)),
        "n_candidates_dropped": int(dropped),
        "top_k": int(k),
        "top_k_hits": int(top_hits),
        "top_k_hit_rate": round(top_hit_rate, 4),
        "top_k_hit_rate_property_only": round(float(top_prop["property_hit"].mean()) if len(top_prop) else 0.0, 4),
        "top_k_hit_rate_ood_only": round(float(top_ood["property_hit"].mean()) if len(top_ood) else 0.0, 4),
        "top_k_mean_prediction": round(float(top_df["prediction"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_prediction_uncertainty": round(float(top_df["prediction_uncertainty"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_abs_error": round(float(top_df["target_violation"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_target_excess": round(float(top_df["target_excess"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_target_violation": round(float(top_df["target_violation"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_d2_distance": round(float(top_df["d2_distance"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_constraint_violation": round(float(top_df["constraint_violation_total"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_descriptor_violation": round(float(top_df["descriptor_violation_total"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_objective": round(float(top_df["ood_aware_objective"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_conservative_objective": round(float(top_df["conservative_objective"].mean()) if len(top_df) else 0.0, 6),
    }

    save_csv(
        valid_df.sort_values("ood_aware_rank"),
        property_step_dirs["files_dir"] / "ood_objective_scores.csv",
        legacy_paths=[
            step_dirs["files_dir"] / "ood_objective_scores.csv",
            results_dir / "step6_ood_aware_inverse" / "ood_objective_scores.csv",
        ],
        index=False,
    )
    save_csv(
        valid_df.sort_values("ood_aware_rank"),
        property_step_dirs["files_dir"] / f"ood_objective_scores_{property_name}.csv",
        legacy_paths=[
            step_dirs["files_dir"] / f"ood_objective_scores_{property_name}.csv",
            results_dir / "step6_ood_aware_inverse" / f"ood_objective_scores_{property_name}.csv",
        ],
        index=False,
    )
    save_csv(
        top_df.sort_values("ood_aware_rank"),
        property_step_dirs["files_dir"] / "ood_objective_topk.csv",
        legacy_paths=[
            step_dirs["files_dir"] / "ood_objective_topk.csv",
            results_dir / "step6_ood_aware_inverse" / "ood_objective_topk.csv",
        ],
        index=False,
    )
    save_csv(
        top_df.sort_values("ood_aware_rank"),
        property_step_dirs["files_dir"] / f"ood_objective_topk_{property_name}.csv",
        legacy_paths=[
            step_dirs["files_dir"] / f"ood_objective_topk_{property_name}.csv",
            results_dir / "step6_ood_aware_inverse" / f"ood_objective_topk_{property_name}.csv",
        ],
        index=False,
    )
    save_csv(
        pd.DataFrame([metrics_row]),
        property_step_dirs["metrics_dir"] / "metrics_inverse_ood_objective.csv",
        legacy_paths=[
            step_dirs["metrics_dir"] / "metrics_inverse_ood_objective.csv",
            results_dir / "metrics_inverse_ood_objective.csv",
        ],
        index=False,
    )
    save_csv(
        pd.DataFrame([metrics_row]),
        property_step_dirs["metrics_dir"] / f"metrics_inverse_ood_objective_{property_name}.csv",
        legacy_paths=[
            step_dirs["metrics_dir"] / f"metrics_inverse_ood_objective_{property_name}.csv",
            results_dir / f"metrics_inverse_ood_objective_{property_name}.csv",
        ],
        index=False,
    )
    save_json(
        {
            "encoder_view": encoder_view,
            "property": property_name,
            "target_value": float(target),
            "target_mode": target_mode,
            "epsilon": float(epsilon),
            "top_k": int(k),
            "normalization": normalization,
            "prediction_column": target_pred_col,
            "prediction_uncertainty_column": target_unc_col or "",
            "prediction_n_models_column": target_n_models_col or "",
            "objective_property_weight": float(normalized_term_weights.get("property", 0.0)),
            "objective_ood_weight": float(normalized_term_weights.get("ood", 0.0)),
            "objective_uncertainty_weight": float(normalized_term_weights.get("uncertainty", 0.0)),
            "objective_constraint_weight": float(normalized_term_weights.get("constraint", 0.0)),
            "objective_sa_weight": float(normalized_term_weights.get("sa", 0.0)),
            "objective_descriptor_weight": float(normalized_term_weights.get("descriptor", 0.0)),
            "constraint_properties": active_constraint_props,
            "descriptor_constraints": active_descriptor_constraints,
            "ood_views_requested": ood_views,
            "ood_views_used": ood_views_used,
            "ood_view_aggregation": "mean",
            "ood_k": int(ood_k),
            "use_alignment_for_ood": bool(use_alignment),
            "candidate_scores_path": str(candidate_scores_path),
            "d2_distance_source": d2_source,
        },
        property_step_dirs["files_dir"] / "run_meta.json",
        legacy_paths=[
            step_dirs["files_dir"] / "run_meta.json",
            results_dir / "step6_ood_aware_inverse" / "run_meta.json",
        ],
    )
    save_json(
        {
            "encoder_view": encoder_view,
            "property": property_name,
            "target_value": float(target),
            "target_mode": target_mode,
            "epsilon": float(epsilon),
            "top_k": int(k),
            "normalization": normalization,
            "prediction_column": target_pred_col,
            "prediction_uncertainty_column": target_unc_col or "",
            "prediction_n_models_column": target_n_models_col or "",
            "objective_property_weight": float(normalized_term_weights.get("property", 0.0)),
            "objective_ood_weight": float(normalized_term_weights.get("ood", 0.0)),
            "objective_uncertainty_weight": float(normalized_term_weights.get("uncertainty", 0.0)),
            "objective_constraint_weight": float(normalized_term_weights.get("constraint", 0.0)),
            "objective_sa_weight": float(normalized_term_weights.get("sa", 0.0)),
            "objective_descriptor_weight": float(normalized_term_weights.get("descriptor", 0.0)),
            "constraint_properties": active_constraint_props,
            "descriptor_constraints": active_descriptor_constraints,
            "ood_views_requested": ood_views,
            "ood_views_used": ood_views_used,
            "ood_view_aggregation": "mean",
            "ood_k": int(ood_k),
            "use_alignment_for_ood": bool(use_alignment),
            "candidate_scores_path": str(candidate_scores_path),
            "d2_distance_source": d2_source,
        },
        property_step_dirs["files_dir"] / f"run_meta_{property_name}.json",
        legacy_paths=[
            step_dirs["files_dir"] / f"run_meta_{property_name}.json",
            results_dir / "step6_ood_aware_inverse" / f"run_meta_{property_name}.json",
        ],
    )

    _save_augmented_ood_metrics(
        results_dir=results_dir,
        representation=representation,
        model_size=str(model_size),
        generated_d2_distance=pd.to_numeric(df.get("d2_distance", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float32),
        ood_k=int(ood_k),
    )

    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(cfg_step6.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F6 figures.")
        generate_figures = False
    if generate_figures:
        _plot_f6_objective_diagnostics(
            valid_df=valid_df,
            top_df=top_df,
            property_name=property_name,
            target_mode=target_mode,
            normalized_term_weights=normalized_term_weights,
            figures_dir=property_step_dirs["figures_dir"],
        )

    print(f"Saved metrics_inverse_ood_objective.csv to {property_step_dirs['step_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--encoder_view", type=str, default=None, choices=list(SUPPORTED_VIEWS))
    parser.add_argument("--ood_views", type=str, default=None, help="Comma-separated views or 'all' for multi-view OOD distance.")
    parser.add_argument("--property", type=str, default=None)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--target_mode", type=str, default=None, choices=["window", "ge", "le"])
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--property_weight", type=float, default=None)
    parser.add_argument("--ood_weight", type=float, default=None)
    parser.add_argument("--uncertainty_weight", type=float, default=None)
    parser.add_argument("--constraint_weight", type=float, default=None)
    parser.add_argument("--sa_weight", type=float, default=None)
    parser.add_argument("--descriptor_weight", type=float, default=None)
    parser.add_argument("--constraint_properties", type=str, default=None, help="Comma-separated soft-constraint properties.")
    parser.add_argument("--normalization", type=str, default=None, choices=["minmax", "rank", "none"])
    parser.add_argument("--ood_k", type=int, default=None)
    parser.add_argument("--use_alignment", dest="use_alignment", action="store_true")
    parser.add_argument("--no_alignment", dest="use_alignment", action="store_false")
    parser.set_defaults(use_alignment=None)
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    parser.add_argument("--compute_d2_distance_if_missing", dest="compute_d2_distance_if_missing", action="store_true")
    parser.add_argument("--no_compute_d2_distance_if_missing", dest="compute_d2_distance_if_missing", action="store_false")
    parser.set_defaults(compute_d2_distance_if_missing=None)
    parser.add_argument("--recompute_d2_distance", action="store_true")
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
