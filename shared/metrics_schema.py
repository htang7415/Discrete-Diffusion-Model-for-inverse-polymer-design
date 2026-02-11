"""Shared metrics schemas and helpers for cross-method aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


METHOD_DIR_PREFIXES = ("Bi_Diffusion_", "AR_Transformer_")

REPRESENTATION_ALIASES = {
    "SMILES": "SMILES",
    "SMILES_BPE": "SMILES_BPE",
    "SELFIES": "SELFIES",
    "Group_SELFIES": "Group_SELFIES",
    "graph": "Graph",
    "Graph": "Graph",
}

MODEL_SIZES = ("small", "medium", "large", "xl")

GENERATION_COLUMNS = [
    "method",
    "representation",
    "model_size",
    "sample_id",
    "n_total",
    "n_valid",
    "validity",
    "validity_two_stars",
    "frac_star_eq_2",
    "uniqueness",
    "novelty",
    "avg_diversity",
    "mean_sa",
    "std_sa",
    "min_sa",
    "max_sa",
    "mean_length",
    "std_length",
    "min_length",
    "max_length",
    "samples_per_sec",
    "valid_per_sec",
]

INVERSE_COLUMNS = [
    "method",
    "representation",
    "model_size",
    "property",
    "target_value",
    "epsilon",
    "n_generated",
    "n_valid",
    "n_hits",
    "success_rate",
    "validity",
    "validity_two_stars",
    "uniqueness",
    "novelty",
    "avg_diversity",
    "achievement_5p",
    "achievement_10p",
    "achievement_15p",
    "achievement_20p",
    "sampling_time_sec",
    "valid_per_compute",
    "rerank_applied",
    "rerank_strategy",
    "rerank_top_k",
    "rerank_hits",
    "rerank_success_rate",
    "rerank_reason",
    "valid_per_compute_rerank",
]

PROPERTY_COLUMNS = [
    "method",
    "representation",
    "model_size",
    "property",
    "split",
    "mae",
    "rmse",
    "r2",
]

CONSTRAINT_COLUMNS = [
    "method",
    "representation",
    "model_size",
    "constraint",
    "total",
    "violations",
    "violation_rate",
]

OOD_COLUMNS = [
    "method",
    "representation",
    "model_size",
    "d1_to_d2_mean_dist",
    "generated_to_d2_mean_dist",
    "frac_generated_near_d2",
]

ALIGNMENT_COLUMNS = [
    "view_pair",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "view_dropout_mode",
]

CONSTRAINTS_BY_REPRESENTATION = {
    "SMILES": [
        "star_count",
        "bond_placement",
        "paren_balance",
        "empty_parens",
        "ring_closure",
    ],
    "SMILES_BPE": [
        "star_count",
        "bond_placement",
        "paren_balance",
        "empty_parens",
        "ring_closure",
    ],
    "SELFIES": [
        "placeholder_count",
        "conversion_failure",
    ],
    "Group_SELFIES": [
        "star_count",
        "bond_placement",
        "paren_balance",
        "empty_parens",
        "ring_closure",
    ],
    "Graph": [
        "star_count",
        "star_degree",
        "star_star_bond",
        "edge_symmetry",
        "valence",
    ],
}


@dataclass(frozen=True)
class MethodInfo:
    method: str
    representation: str


def parse_method_representation(folder_name: str) -> MethodInfo:
    """Infer method and representation from a method directory name."""
    if folder_name.startswith("Bi_Diffusion_"):
        method = "Bi_Diffusion"
        rep = folder_name.replace("Bi_Diffusion_", "")
    elif folder_name.startswith("AR_Transformer_"):
        method = "AR_Transformer"
        rep = folder_name.replace("AR_Transformer_", "")
    else:
        method = "Unknown"
        rep = folder_name

    rep = REPRESENTATION_ALIASES.get(rep, rep)
    return MethodInfo(method=method, representation=rep)


def infer_model_size(results_dir: Path) -> str:
    """Infer model size from a results directory name."""
    name = results_dir.name
    for size in MODEL_SIZES:
        if name.endswith(f"_{size}") or name == f"results_{size}":
            return size
    return "base"


def ensure_columns(df, columns: Iterable[str], fill_value=None):
    """Ensure DataFrame has all columns; fill missing with fill_value."""
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def list_method_dirs(root: Path) -> List[Path]:
    """List method directories at repo root."""
    dirs = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(METHOD_DIR_PREFIXES):
            dirs.append(path)
    return sorted(dirs)


def list_results_dirs(method_dir: Path) -> List[Path]:
    """List results directories for a method folder."""
    results = []
    for path in method_dir.iterdir():
        if path.is_dir() and path.name.startswith("results"):
            results.append(path)
    # ensure base results first if present
    results_sorted = sorted(results, key=lambda p: (p.name != "results", p.name))
    return results_sorted
