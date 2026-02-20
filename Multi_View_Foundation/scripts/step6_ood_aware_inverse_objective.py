#!/usr/bin/env python
"""F6: OOD-aware inverse design objective on scored candidates.

This step ranks candidate polymers with a joint objective:
  property term + OOD term (distance to D2 manifold).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.foundation_inverse import (
    compute_ood_aware_objective,
    compute_property_hits,
)
from src.utils.config import load_config, save_config
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json
from shared.ood_metrics import knn_distances


SUPPORTED_VIEWS = ("smiles", "smiles_bpe", "selfies", "group_selfies", "graph")

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


def _view_to_representation(view: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
    }
    return mapping.get(view, view)


def _default_candidate_scores_path(results_dir: Path) -> Path:
    preferred = results_dir / "step5_foundation_inverse" / "files" / "candidate_scores.csv"
    if preferred.exists():
        return preferred
    legacy = results_dir / "step5_foundation_inverse" / "candidate_scores.csv"
    if legacy.exists():
        return legacy
    return preferred


def _resolve_candidate_scores_path(args, cfg_step6: dict, results_dir: Path) -> Path:
    if args.candidate_scores_csv:
        return _resolve_path(args.candidate_scores_csv)
    cfg_path = str(cfg_step6.get("candidate_scores_csv", "")).strip()
    if cfg_path:
        return _resolve_path(cfg_path)
    return _default_candidate_scores_path(results_dir)


def _compute_d2_distance_column(
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


def main(args):
    config = load_config(args.config)
    cfg_step6 = config.get("ood_aware_inverse", {}) or {}
    cfg_f5 = config.get("foundation_inverse", {}) or {}

    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step6_ood_aware_inverse")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    candidate_scores_path = _resolve_candidate_scores_path(args, cfg_step6, results_dir)
    if not candidate_scores_path.exists():
        raise FileNotFoundError(f"Candidate scores CSV not found: {candidate_scores_path}")

    encoder_view = _select_encoder_view(config, args.encoder_view, cfg_step6)
    if encoder_view not in SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported encoder_view={encoder_view}. Supported: {', '.join(SUPPORTED_VIEWS)}")

    property_name = args.property
    if property_name is None:
        property_name = str(cfg_step6.get("property", "")).strip() or str(cfg_f5.get("property", "")).strip() or "property"

    target = _to_float_or_none(args.target)
    if target is None:
        target = _to_float_or_none(cfg_step6.get("target"))
    if target is None:
        target = _to_float_or_none(cfg_f5.get("target"))
    if target is None:
        raise ValueError("target is required (set --target or ood_aware_inverse.target/foundation_inverse.target).")

    target_mode = args.target_mode or str(cfg_step6.get("target_mode", "")).strip() or str(cfg_f5.get("target_mode", "window")).strip() or "window"
    epsilon = _to_float_or_none(args.epsilon)
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
        property_weight = 0.7

    ood_weight = _to_float_or_none(args.ood_weight)
    if ood_weight is None:
        ood_weight = _to_float_or_none(cfg_step6.get("ood_weight"))
    if ood_weight is None:
        ood_weight = 0.3

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

    if args.use_alignment is None:
        use_alignment = _to_bool(cfg_step6.get("use_alignment", cfg_f5.get("use_alignment", True)), True)
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
    if "prediction" not in df.columns:
        raise ValueError(f"Candidate scores must include 'prediction' column: {candidate_scores_path}")

    df = df.copy()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
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

    d2_source = "existing"
    has_d2 = "d2_distance" in df.columns
    if has_d2:
        df["d2_distance"] = pd.to_numeric(df["d2_distance"], errors="coerce")

    needs_compute = force_recompute
    if not needs_compute and compute_if_missing:
        needs_compute = (not has_d2) or bool(df["d2_distance"].isna().any())

    if needs_compute:
        d2_source = "recomputed"
        smiles_list = df["smiles"].astype(str).tolist()
        d2_values = _compute_d2_distance_column(
            config=config,
            results_dir=results_dir,
            smiles_list=smiles_list,
            encoder_view=encoder_view,
            ood_k=ood_k,
            use_alignment=use_alignment,
            alignment_checkpoint=alignment_checkpoint,
        )
        df["d2_distance"] = d2_values
    elif not has_d2:
        raise ValueError(
            "d2_distance column is missing and compute_d2_distance_if_missing is disabled. "
            "Enable compute or provide d2_distance in candidate_scores.csv."
        )

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
    hit_mask = compute_property_hits(pred, target_value=float(target), epsilon=float(epsilon), target_mode=target_mode)
    score_pack = compute_ood_aware_objective(
        predictions=pred,
        d2_distances=d2,
        target_value=float(target),
        target_mode=target_mode,
        property_weight=float(property_weight),
        ood_weight=float(ood_weight),
        normalization=normalization,
    )

    valid_df["property_hit"] = hit_mask.astype(bool)
    valid_df["property_error_normed"] = score_pack["property_error"]
    valid_df["property_error_objective"] = score_pack["property_error_norm"]
    valid_df["d2_distance_objective"] = score_pack["d2_distance_norm"]
    valid_df["ood_aware_objective"] = score_pack["objective"]
    valid_df["ood_aware_rank"] = score_pack["objective_rank"]
    valid_df["property"] = property_name

    order = np.argsort(valid_df["ood_aware_objective"].to_numpy(dtype=np.float32))
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

    metrics_row = {
        "method": "Multi_View_Foundation",
        "representation": _view_to_representation(encoder_view),
        "model_size": model_size,
        "property": property_name,
        "target_value": float(target),
        "target_mode": target_mode,
        "epsilon": float(epsilon),
        "normalization": normalization,
        "objective_property_weight": float(score_pack["property_weight_norm"][0]),
        "objective_ood_weight": float(score_pack["ood_weight_norm"][0]),
        "ood_k": int(ood_k),
        "use_alignment_for_ood": bool(use_alignment),
        "d2_distance_source": d2_source,
        "candidate_scores_csv": str(candidate_scores_path),
        "n_candidates_total": int(len(df)),
        "n_candidates_scored": int(len(valid_df)),
        "n_candidates_dropped": int(dropped),
        "top_k": int(k),
        "top_k_hits": int(top_hits),
        "top_k_hit_rate": round(top_hit_rate, 4),
        "top_k_hit_rate_property_only": round(float(top_prop["property_hit"].mean()) if len(top_prop) else 0.0, 4),
        "top_k_hit_rate_ood_only": round(float(top_ood["property_hit"].mean()) if len(top_ood) else 0.0, 4),
        "top_k_mean_prediction": round(float(top_df["prediction"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_abs_error": round(float(np.abs(top_df["prediction"] - float(target)).mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_d2_distance": round(float(top_df["d2_distance"].mean()) if len(top_df) else 0.0, 6),
        "top_k_mean_objective": round(float(top_df["ood_aware_objective"].mean()) if len(top_df) else 0.0, 6),
    }

    save_csv(
        valid_df.sort_values("ood_aware_rank"),
        step_dirs["files_dir"] / "ood_objective_scores.csv",
        legacy_paths=[results_dir / "step6_ood_aware_inverse" / "ood_objective_scores.csv"],
        index=False,
    )
    save_csv(
        valid_df.sort_values("ood_aware_rank"),
        step_dirs["files_dir"] / f"ood_objective_scores_{property_name}.csv",
        legacy_paths=[results_dir / "step6_ood_aware_inverse" / f"ood_objective_scores_{property_name}.csv"],
        index=False,
    )
    save_csv(
        top_df.sort_values("ood_aware_rank"),
        step_dirs["files_dir"] / "ood_objective_topk.csv",
        legacy_paths=[results_dir / "step6_ood_aware_inverse" / "ood_objective_topk.csv"],
        index=False,
    )
    save_csv(
        top_df.sort_values("ood_aware_rank"),
        step_dirs["files_dir"] / f"ood_objective_topk_{property_name}.csv",
        legacy_paths=[results_dir / "step6_ood_aware_inverse" / f"ood_objective_topk_{property_name}.csv"],
        index=False,
    )
    save_csv(
        pd.DataFrame([metrics_row]),
        step_dirs["metrics_dir"] / "metrics_inverse_ood_objective.csv",
        legacy_paths=[results_dir / "metrics_inverse_ood_objective.csv"],
        index=False,
    )
    save_csv(
        pd.DataFrame([metrics_row]),
        step_dirs["metrics_dir"] / f"metrics_inverse_ood_objective_{property_name}.csv",
        legacy_paths=[results_dir / f"metrics_inverse_ood_objective_{property_name}.csv"],
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
            "objective_property_weight": float(score_pack["property_weight_norm"][0]),
            "objective_ood_weight": float(score_pack["ood_weight_norm"][0]),
            "ood_k": int(ood_k),
            "use_alignment_for_ood": bool(use_alignment),
            "candidate_scores_csv": str(candidate_scores_path),
            "d2_distance_source": d2_source,
        },
        step_dirs["files_dir"] / "run_meta.json",
        legacy_paths=[results_dir / "step6_ood_aware_inverse" / "run_meta.json"],
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
            "objective_property_weight": float(score_pack["property_weight_norm"][0]),
            "objective_ood_weight": float(score_pack["ood_weight_norm"][0]),
            "ood_k": int(ood_k),
            "use_alignment_for_ood": bool(use_alignment),
            "candidate_scores_csv": str(candidate_scores_path),
            "d2_distance_source": d2_source,
        },
        step_dirs["files_dir"] / f"run_meta_{property_name}.json",
        legacy_paths=[results_dir / "step6_ood_aware_inverse" / f"run_meta_{property_name}.json"],
    )

    print(f"Saved metrics_inverse_ood_objective.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--candidate_scores_csv", type=str, default=None)
    parser.add_argument("--encoder_view", type=str, default=None, choices=list(SUPPORTED_VIEWS))
    parser.add_argument("--property", type=str, default=None)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--target_mode", type=str, default=None, choices=["window", "ge", "le"])
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--property_weight", type=float, default=None)
    parser.add_argument("--ood_weight", type=float, default=None)
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
    main(parser.parse_args())
