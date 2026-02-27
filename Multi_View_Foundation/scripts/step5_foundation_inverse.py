#!/usr/bin/env python
"""F5: Foundation-enhanced inverse design with reranking via resampling."""

import argparse
import json
from pathlib import Path
import sys
import time
import importlib
import importlib.util
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies
from shared.ood_metrics import knn_distances
from src.model.multi_view_model import MultiViewModel
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import RDConfig
    # Keep F5 logs readable when many invalid candidates are rejected.
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
except Exception:  # pragma: no cover
    Chem = None
    RDConfig = None
    rdBase = None

try:
    import os
    if RDConfig is not None:
        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
    import sascorer  # type: ignore
except Exception:  # pragma: no cover
    sascorer = None


SUPPORTED_VIEWS = ("smiles", "smiles_bpe", "selfies", "group_selfies", "graph")

VIEW_SPECS = {
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

DEFAULT_POLYMER_PATTERNS = {
    "polyimide": "[#6](=O)-[#7]-[#6](=O)",
    "polyester": "[#6](=O)-[#8]-[#6]",
    "polyamide": "[#6](=O)-[#7]-[#6]",
    "polyurethane": "[#8]-[#6](=O)-[#7]",
    "polyether": "[#6]-[#8]-[#6]",
    "polysiloxane": "[Si]-[#8]-[Si]",
    "polycarbonate": "[#8]-[#6](=O)-[#8]",
    "polysulfone": "[#6]-[S](=O)(=O)-[#6]",
    "polyacrylate": "[#6]-[#6](=O)-[#8]",
    "polystyrene": "[#6]-[#6](c1ccccc1)-[#6]",
}

DEFAULT_F5_RESAMPLE_SETTINGS = {
    "property_model_mode": "all",  # single|all
    "proposal_views": "all",  # all|<comma-separated views>
    "committee_properties": None,  # defaults to property.files
    "sampling_target": 100,
    "sampling_num_per_batch": 512,
    "sampling_batch_size": 128,
    "sampling_max_batches": 200,
    "sampling_temperature": None,
    "sampling_num_atoms": None,
    "target_class": "",
    "require_validity": True,
    "require_two_stars": True,
    "require_novel": True,
    "require_unique": True,
    "max_sa": 4.5,
    "per_view_min_hits": 0,
    "per_view_quota_relax_after_batches": 0,
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


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _resolve_with_base(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def _view_to_representation(view: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
    }
    return mapping.get(view, view)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int_or_none(value: Any) -> Optional[int]:
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


def _plot_f5_diagnostics(
    *,
    scored_df: pd.DataFrame,
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    source_meta: dict,
    figures_dir: Path,
) -> None:
    if plt is None or scored_df.empty:
        return

    df = scored_df.copy()
    mode = str(target_mode).strip().lower()
    accepted_mask = pd.to_numeric(df.get("accepted", pd.Series([0] * len(df), index=df.index)), errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    # A) score distribution.
    if mode in {"ge", "le"}:
        if "target_excess" in df.columns:
            excess_all = pd.to_numeric(df["target_excess"], errors="coerce").dropna()
            excess_acc = pd.to_numeric(df.loc[accepted_mask, "target_excess"], errors="coerce").dropna()
        else:
            pred_vals = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce")
            pred_acc_vals = pd.to_numeric(df.loc[accepted_mask, "prediction"] if "prediction" in df.columns else pd.Series(dtype=float), errors="coerce")
            excess_all = pd.Series(_compute_target_excess(pred_vals.to_numpy(dtype=np.float32), float(target_value), mode)).dropna()
            excess_acc = pd.Series(_compute_target_excess(pred_acc_vals.to_numpy(dtype=np.float32), float(target_value), mode)).dropna()
        if len(excess_all):
            ax0.hist(excess_all.to_numpy(dtype=np.float32), bins=40, color="#4E79A7", alpha=0.65, label="Scored")
        if len(excess_acc):
            ax0.hist(excess_acc.to_numpy(dtype=np.float32), bins=30, color="#E15759", alpha=0.75, label="Accepted")
        ax0.axvline(0.0, color="#222222", linestyle="--", linewidth=1.1, label="Target boundary")
        ax0.set_xlabel(_target_excess_axis_label(property_name, mode))
    else:
        pred = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce").dropna()
        pred_accepted = pd.to_numeric(df.loc[accepted_mask, "prediction"] if "prediction" in df.columns else pd.Series(dtype=float), errors="coerce").dropna()
        if len(pred):
            ax0.hist(pred.to_numpy(dtype=np.float32), bins=40, color="#4E79A7", alpha=0.65, label="Scored")
        if len(pred_accepted):
            ax0.hist(pred_accepted.to_numpy(dtype=np.float32), bins=30, color="#E15759", alpha=0.75, label="Accepted")
        ax0.axvline(float(target_value), color="#222222", linestyle="--", linewidth=1.1, label="Target")
        ax0.axvspan(float(target_value) - float(epsilon), float(target_value) + float(epsilon), color="#59A14F", alpha=0.15)
        ax0.set_xlabel(f"Predicted {property_name}")
    ax0.set_ylabel("Count")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=15)

    # B) prediction vs OOD distance.
    if {"prediction", "d2_distance"}.issubset(set(df.columns)):
        scatter_df = df.dropna(subset=["prediction", "d2_distance"]).copy()
        if not scatter_df.empty:
            if mode in {"ge", "le"}:
                if "target_excess" in scatter_df.columns:
                    y_vals = pd.to_numeric(scatter_df["target_excess"], errors="coerce")
                else:
                    y_vals = pd.Series(
                        _compute_target_excess(
                            pd.to_numeric(scatter_df["prediction"], errors="coerce").to_numpy(dtype=np.float32),
                            float(target_value),
                            mode,
                        ),
                        index=scatter_df.index,
                    )
                mask = y_vals.notna()
                scatter_df = scatter_df.loc[mask].copy()
                y_vals = y_vals.loc[mask]
                if not scatter_df.empty:
                    sc = ax1.scatter(
                        scatter_df["d2_distance"],
                        y_vals,
                        c=y_vals,
                        cmap="RdYlGn",
                        s=24,
                        alpha=0.82,
                    )
                    ax1.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
                    ax1.set_xlabel("D2 distance")
                    ax1.set_ylabel("Target excess (≥0 = hit)")
                    ax1.grid(alpha=0.25)
                    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Target excess")
                else:
                    ax1.text(0.5, 0.5, "No finite points", ha="center", va="center")
                    ax1.set_axis_off()
            else:
                hit_vals = scatter_df.get("property_hit", pd.Series([False] * len(scatter_df), index=scatter_df.index)).astype(bool)
                base = scatter_df[~hit_vals]
                hit = scatter_df[hit_vals]
                if not base.empty:
                    ax1.scatter(base["d2_distance"], base["prediction"], s=16, alpha=0.35, color="#9ECAE1", label="Non-hit")
                if not hit.empty:
                    ax1.scatter(hit["d2_distance"], hit["prediction"], s=24, alpha=0.9, color="#D94801", label="Hit")
                ax1.axhline(float(target_value), color="#222222", linestyle="--", linewidth=1.0)
                ax1.set_xlabel("D2 distance")
                ax1.set_ylabel(f"Predicted {property_name}")
                ax1.grid(alpha=0.25)
                ax1.legend(loc="best", fontsize=15)
        else:
            ax1.text(0.5, 0.5, "No finite points", ha="center", va="center")
            ax1.set_axis_off()
    else:
        ax1.text(0.5, 0.5, "No d2_distance column", ha="center", va="center")
        ax1.set_axis_off()

    # C) Sampling funnel: generated → structural valid → scored → accepted (property hit).
    # If multiple views, show per-view breakdown; otherwise show aggregate funnel.
    stats_dict = source_meta.get("stats", {})
    funnel_drawn = False
    if "proposal_view" in df.columns:
        all_proposal_views = sorted(df["proposal_view"].dropna().astype(str).unique().tolist())
        if len(all_proposal_views) > 1:
            # Multi-view: grouped bar showing generated / scored / accepted per view
            x_labels = all_proposal_views
            n_gen = np.array([float(stats_dict.get(f"n_generated_{v}", 0)) for v in all_proposal_views], dtype=np.float32)
            n_scored = np.array([float(stats_dict.get(f"n_scored_{v}", 0)) for v in all_proposal_views], dtype=np.float32)
            n_accepted = np.array([float(df.loc[accepted_mask & (df["proposal_view"] == v)].shape[0]) for v in all_proposal_views], dtype=np.float32)
            x = np.arange(len(x_labels), dtype=np.float32)
            w = 0.28
            ax2.bar(x - w, n_gen, width=w, label="Generated", color="#AEC7E8", alpha=0.85)
            ax2.bar(x, n_scored, width=w, label="Scored", color="#4E79A7", alpha=0.85)
            ax2.bar(x + w, n_accepted, width=w, label="Accepted", color="#59A14F", alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=12)
            ax2.set_ylabel("Count")
            ax2.legend(loc="best", fontsize=11)
            ax2.grid(axis="y", alpha=0.25)
            funnel_drawn = True
    if not funnel_drawn:
        # Single/no view: show aggregate funnel as horizontal bar chart
        n_gen_total = float(stats_dict.get("n_generated", len(df)))
        n_valid = float(stats_dict.get("n_structural_valid", stats_dict.get("n_valid_any", 0)))
        n_scored_total = float(stats_dict.get("n_scored", len(df)))
        n_accepted_total = int(np.sum(accepted_mask))
        funnel_labels = ["Generated", "Struct. valid", "Scored", "Accepted"]
        funnel_vals = np.array([n_gen_total, n_valid, n_scored_total, float(n_accepted_total)], dtype=np.float32)
        colors = ["#AEC7E8", "#6BAED6", "#4E79A7", "#59A14F"]
        y_pos = np.arange(len(funnel_labels), dtype=np.float32)
        bars = ax2.barh(y_pos, funnel_vals, color=colors, alpha=0.9)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(funnel_labels, fontsize=12)
        ax2.set_xlabel("Count")
        ax2.set_title("Sampling funnel", fontsize=12)
        ax2.grid(axis="x", alpha=0.25)
        for bar, val in zip(bars, funnel_vals):
            if val > 0:
                ax2.text(float(val), bar.get_y() + bar.get_height() / 2.0,
                         f"  {int(val):,}", va="center", ha="left", fontsize=11)

    # D) cumulative hits by ranking.
    hit_col = df.get("property_hit", pd.Series([False] * len(df))).astype(bool).to_numpy(dtype=bool)
    if len(df):
        if mode in {"ge", "le"}:
            if "target_excess" in df.columns:
                excess = pd.to_numeric(df["target_excess"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float32)
            else:
                pred_vals = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
                excess = _compute_target_excess(pred_vals, float(target_value), mode)
                excess = np.nan_to_num(excess, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
            order = np.argsort(-excess)
        else:
            if "target_violation" in df.columns:
                order_vals = pd.to_numeric(df["target_violation"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float32)
            else:
                order_vals = pd.to_numeric(df.get("abs_error", pd.Series(dtype=float)), errors="coerce").fillna(np.inf).to_numpy(dtype=np.float32)
            order = np.argsort(order_vals)
        ordered_hits = hit_col[order].astype(np.int64, copy=False)
        cum_hits = np.cumsum(ordered_hits)
        x = np.arange(1, len(cum_hits) + 1, dtype=np.int64)
        ax3.plot(x, cum_hits, color="#4E79A7", linewidth=2.0)
        ax3.set_xlabel("Ranked candidate index")
        ax3.set_ylabel("Cumulative property hits")
        ax3.grid(alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "No scored rows", ha="center", va="center")
        ax3.set_axis_off()

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / f"figure_f5_inverse_diagnostics_{property_name}")
    plt.close(fig)

    # Save compact accepted-by-view table for downstream inspection.
    accepted_rows = []
    if "proposal_view" in df.columns:
        grp = df.loc[accepted_mask].groupby("proposal_view").size().sort_values(ascending=False)
        accepted_rows.extend(
            {"proposal_view": str(k), "accepted_count": int(v)}
            for k, v in grp.items()
        )
    if not accepted_rows:
        accepted_by_view = source_meta.get("accepted_by_view", {})
        if isinstance(accepted_by_view, dict):
            accepted_rows.extend(
                {"proposal_view": str(k), "accepted_count": int(v)}
                for k, v in accepted_by_view.items()
            )
    if accepted_rows:
        save_csv(
            pd.DataFrame(accepted_rows),
            figures_dir / f"figure_f5_inverse_diagnostics_{property_name}_accepted_by_view.csv",
            index=False,
        )


def _rdkit_mol_from_polymer_smiles(smiles: str):
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


def _count_rings(smiles: str) -> Optional[int]:
    mol = _rdkit_mol_from_polymer_smiles(smiles)
    if mol is None:
        return None
    try:
        return int(mol.GetRingInfo().NumRings())
    except Exception:
        return None


def _count_heavy_atoms(smiles: str) -> Optional[int]:
    mol = _rdkit_mol_from_polymer_smiles(smiles)
    if mol is None:
        return None
    try:
        return int(mol.GetNumHeavyAtoms())
    except Exception:
        return None


def _short_text(text: Any, max_len: int = 44) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)] + "..."


def _build_f5_accepted_polymer_report(
    *,
    accepted_df: pd.DataFrame,
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    sampling_target: int,
) -> Tuple[pd.DataFrame, dict]:
    mode = str(target_mode).strip().lower()
    report = accepted_df.copy()
    if report.empty:
        summary = {
            "property": property_name,
            "target_value": float(target_value),
            "target_mode": mode,
            "epsilon": float(epsilon),
            "sampling_target": int(sampling_target),
            "n_accepted": 0,
            "acceptance_ratio_vs_target": 0.0,
        }
        return report, summary

    report["prediction"] = pd.to_numeric(report.get("prediction"), errors="coerce")
    report["target_excess"] = pd.to_numeric(report.get("target_excess"), errors="coerce")
    report["target_violation"] = pd.to_numeric(report.get("target_violation"), errors="coerce")
    report["d2_distance"] = pd.to_numeric(report.get("d2_distance"), errors="coerce")
    report["sa_score"] = pd.to_numeric(report.get("sa_score"), errors="coerce")
    report["prediction_std"] = pd.to_numeric(report.get("prediction_std"), errors="coerce")

    # Ranking policy:
    # - ge/le: maximize target_excess
    # - window: minimize target_violation
    if mode in {"ge", "le"}:
        primary = -report["target_excess"].fillna(-np.inf)
        rank_score = report["target_excess"].fillna(np.nan)
    else:
        fallback_violation = pd.to_numeric(report.get("abs_error"), errors="coerce")
        violation = report["target_violation"].where(report["target_violation"].notna(), fallback_violation)
        primary = violation.fillna(np.inf)
        rank_score = -violation.fillna(np.nan)

    report["_rank_primary"] = primary
    report["_rank_secondary"] = report["d2_distance"].fillna(np.inf)
    report["_rank_tertiary"] = report["prediction_std"].fillna(np.inf)
    report["_rank_score"] = rank_score
    report = report.sort_values(
        ["_rank_primary", "_rank_secondary", "_rank_tertiary"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    report["rank"] = np.arange(1, len(report) + 1, dtype=np.int64)
    report["rank_score"] = pd.to_numeric(report.get("_rank_score"), errors="coerce")

    if "canonical_smiles" not in report.columns:
        report["canonical_smiles"] = report["smiles"].astype(str).map(_canonicalize_smiles)
    else:
        report["canonical_smiles"] = report["canonical_smiles"].astype(str)

    if "star_count" not in report.columns:
        report["star_count"] = report["smiles"].astype(str).map(lambda x: int(str(x).count("*")))
    else:
        report["star_count"] = pd.to_numeric(report["star_count"], errors="coerce")

    if "aromatic_ring_count" not in report.columns:
        report["aromatic_ring_count"] = report["smiles"].astype(str).map(_count_aromatic_rings)
    if "ring_count" not in report.columns:
        report["ring_count"] = report["smiles"].astype(str).map(_count_rings)
    if "heavy_atom_count" not in report.columns:
        report["heavy_atom_count"] = report["smiles"].astype(str).map(_count_heavy_atoms)

    if report["sa_score"].isna().any():
        missing_mask = report["sa_score"].isna()
        computed = report.loc[missing_mask, "smiles"].astype(str).map(_compute_sa_score)
        report.loc[missing_mask, "sa_score"] = pd.to_numeric(computed, errors="coerce")

    mean_pred = float(report["prediction"].mean()) if report["prediction"].notna().any() else np.nan
    mean_excess = float(report["target_excess"].mean()) if report["target_excess"].notna().any() else np.nan
    mean_violation = float(report["target_violation"].mean()) if report["target_violation"].notna().any() else np.nan
    mean_d2 = float(report["d2_distance"].mean()) if report["d2_distance"].notna().any() else np.nan
    mean_sa = float(report["sa_score"].mean()) if report["sa_score"].notna().any() else np.nan
    unique_ratio = float(report["canonical_smiles"].nunique() / max(len(report), 1))
    novelty_ratio = float(pd.to_numeric(report.get("is_novel"), errors="coerce").fillna(0).mean()) if "is_novel" in report.columns else np.nan

    summary = {
        "property": property_name,
        "target_value": float(target_value),
        "target_mode": mode,
        "epsilon": float(epsilon),
        "sampling_target": int(sampling_target),
        "n_accepted": int(len(report)),
        "acceptance_ratio_vs_target": round(float(len(report) / max(sampling_target, 1)), 4),
        "mean_prediction": round(mean_pred, 6) if np.isfinite(mean_pred) else np.nan,
        "mean_target_excess": round(mean_excess, 6) if np.isfinite(mean_excess) else np.nan,
        "mean_target_violation": round(mean_violation, 6) if np.isfinite(mean_violation) else np.nan,
        "mean_d2_distance": round(mean_d2, 6) if np.isfinite(mean_d2) else np.nan,
        "mean_sa_score": round(mean_sa, 6) if np.isfinite(mean_sa) else np.nan,
        "unique_ratio_canonical": round(unique_ratio, 4),
        "novelty_ratio": round(novelty_ratio, 4) if np.isfinite(novelty_ratio) else np.nan,
    }

    ordered_cols = [
        "rank",
        "smiles",
        "canonical_smiles",
        "proposal_view",
        "matched_class",
        "prediction",
        "prediction_std",
        "target_excess",
        "target_violation",
        "d2_distance",
        "sa_score",
        "star_count",
        "aromatic_ring_count",
        "ring_count",
        "heavy_atom_count",
        "is_novel",
        "batch_idx",
        "rank_score",
    ]
    existing_cols = [c for c in ordered_cols if c in report.columns]
    other_cols = [c for c in report.columns if c not in existing_cols and not c.startswith("_")]
    report = report[existing_cols + other_cols]
    return report, summary


def _plot_f5_accepted_polymer_overview(
    *,
    report_df: pd.DataFrame,
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    figures_dir: Path,
) -> None:
    if plt is None or report_df.empty:
        return

    mode = str(target_mode).strip().lower()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    # A) Ranked score bars.
    top_n = min(30, len(report_df))
    top_df = report_df.head(top_n).copy()
    y = np.arange(top_n, dtype=np.float32)
    if mode in {"ge", "le"}:
        vals = pd.to_numeric(top_df.get("target_excess"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        x_label = _target_excess_axis_label(property_name, mode)
        colors = ["#2A9D8F" if v >= 0 else "#E76F51" for v in vals]
    else:
        violation = pd.to_numeric(top_df.get("target_violation"), errors="coerce").fillna(np.inf).to_numpy(dtype=np.float32)
        vals = -violation
        x_label = "Negative target violation (higher is better)"
        colors = ["#2A9D8F"] * len(vals)

    labels = [f"#{int(r)}" for r in top_df["rank"].tolist()]
    ax0.barh(y, vals, color=colors, alpha=0.9)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    ax0.set_xlabel(x_label)
    ax0.set_title(f"Top accepted {property_name} candidates", fontsize=14, fontweight="bold")
    ax0.grid(axis="x", alpha=0.25)

    # B) Prediction vs OOD distance scatter.
    scatter_df = report_df.dropna(subset=["prediction", "d2_distance"]).copy()
    if not scatter_df.empty:
        color_vals = pd.to_numeric(scatter_df.get("target_excess"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        sc = ax1.scatter(
            scatter_df["d2_distance"].to_numpy(dtype=np.float32),
            scatter_df["prediction"].to_numpy(dtype=np.float32),
            c=color_vals,
            cmap="RdYlGn",
            s=34,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
        )
        ax1.axhline(float(target_value), color="#111111", linestyle="--", linewidth=1.0, label="Target")
        if mode == "window":
            ax1.axhspan(float(target_value) - float(epsilon), float(target_value) + float(epsilon), color="#2A9D8F", alpha=0.12)
        ax1.set_xlabel("D2 distance (lower is better)")
        ax1.set_ylabel(f"Predicted {property_name}")
        ax1.set_title("Accepted candidates: property vs OOD", fontsize=14, fontweight="bold")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best", fontsize=11)
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Target excess")
    else:
        ax1.text(0.5, 0.5, "No finite prediction/d2 points", ha="center", va="center")
        ax1.set_axis_off()

    # C) Descriptor profile boxplots.
    descriptor_cols = [
        ("d2_distance", "D2"),
        ("sa_score", "SA"),
        ("aromatic_ring_count", "Aro rings"),
        ("ring_count", "Rings"),
        ("heavy_atom_count", "Heavy atoms"),
    ]
    arrays = []
    labels_c = []
    for col, label in descriptor_cols:
        if col in report_df.columns:
            arr = pd.to_numeric(report_df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if arr.size:
                arrays.append(arr)
                labels_c.append(label)
    if arrays:
        bp = ax2.boxplot(arrays, labels=labels_c, patch_artist=True, showfliers=False)
        palette = ["#4E79A7", "#2A9D8F", "#F4A261", "#264653", "#E76F51"]
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax2.set_title("Accepted polymer descriptor profile", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "No descriptor values", ha="center", va="center")
        ax2.set_axis_off()

    # D) Top-ranked table.
    show_n = min(12, len(report_df))
    table_df = report_df.head(show_n).copy()
    cols = []
    if "rank" in table_df.columns:
        cols.append(("rank", "Rank"))
    cols.append(("smiles", "Polymer p-SMILES"))
    if "prediction" in table_df.columns:
        cols.append(("prediction", "Pred"))
    if "target_excess" in table_df.columns:
        cols.append(("target_excess", "Excess"))
    if "d2_distance" in table_df.columns:
        cols.append(("d2_distance", "D2"))
    if "sa_score" in table_df.columns:
        cols.append(("sa_score", "SA"))

    col_labels = [x[1] for x in cols]
    cell_rows = []
    for _, row in table_df.iterrows():
        row_vals = []
        for key, _ in cols:
            value = row.get(key, "")
            if key == "smiles":
                row_vals.append(_short_text(value, max_len=38))
            elif isinstance(value, (float, np.floating)):
                row_vals.append(f"{float(value):.3f}" if np.isfinite(value) else "")
            else:
                row_vals.append(str(value))
        cell_rows.append(row_vals)

    ax3.axis("off")
    if cell_rows:
        table = ax3.table(cellText=cell_rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.15)
        ax3.set_title("Top accepted polymers (detailed info)", fontsize=14, fontweight="bold", pad=8)
    else:
        ax3.text(0.5, 0.5, "No accepted candidates", ha="center", va="center")

    fig.suptitle(f"F5 accepted polymer report: {property_name}", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save_figure_png(fig, figures_dir / f"figure_f5_accepted_polymer_overview_{property_name}")
    plt.close(fig)


def _plot_f5_accepted_polymer_gallery(
    *,
    report_df: pd.DataFrame,
    property_name: str,
    figures_dir: Path,
    top_n: int = 36,
) -> None:
    if report_df.empty or Chem is None:
        return
    try:
        from rdkit.Chem import Draw  # type: ignore
    except Exception:
        return

    gallery_df = report_df.head(max(int(top_n), 1)).copy()
    mols = []
    legends = []
    for _, row in gallery_df.iterrows():
        smi = str(row.get("smiles", "")).strip()
        mol = _rdkit_mol_from_polymer_smiles(smi)
        if mol is None:
            continue
        rank_val = row.get("rank", "")
        pred_val = pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]
        d2_val = pd.to_numeric(pd.Series([row.get("d2_distance")]), errors="coerce").iloc[0]
        rank_text = str(int(rank_val)) if pd.notna(rank_val) else "?"
        legend = f"#{rank_text} | pred={pred_val:.2f} | d2={d2_val:.2f}" if np.isfinite(pred_val) and np.isfinite(d2_val) else f"#{rank_text}"
        mols.append(mol)
        legends.append(legend)

    if not mols:
        return

    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=6,
            subImgSize=(260, 200),
            legends=legends,
            useSVG=False,
        )
        out_path = figures_dir / f"figure_f5_accepted_polymer_gallery_{property_name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(img, "save"):
            img.save(str(out_path))
    except Exception:
        return


def _f5_cfg(config: dict) -> dict:
    merged = dict(DEFAULT_F5_RESAMPLE_SETTINGS)
    merged.update(config.get("foundation_inverse", {}) or {})
    return merged


def _is_view_enabled(config: dict, view: str) -> bool:
    views_cfg = config.get("views", {})
    if view in views_cfg and not views_cfg[view].get("enabled", True):
        return False
    return True


def _select_encoder_view(config: dict, override: Optional[str]) -> str:
    requested = override
    if requested is None:
        requested = str(config.get("foundation_inverse", {}).get("encoder_view", "")).strip() or None

    if requested:
        if requested not in SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported encoder_view={requested}. Supported: {', '.join(SUPPORTED_VIEWS)}")
        if not _is_view_enabled(config, requested):
            raise ValueError(f"encoder_view={requested} is disabled in config.views.")
        encoder_key = VIEW_SPECS[requested]["encoder_key"]
        if not config.get(encoder_key, {}).get("method_dir"):
            raise ValueError(f"Encoder config missing for view={requested} ({encoder_key}.method_dir).")
        return requested

    candidate_views = []
    configured_order = config.get("alignment_views", list(SUPPORTED_VIEWS))
    for view in configured_order:
        if view in SUPPORTED_VIEWS:
            candidate_views.append(view)
    for view in SUPPORTED_VIEWS:
        if view not in candidate_views:
            candidate_views.append(view)

    for view in candidate_views:
        if not _is_view_enabled(config, view):
            continue
        encoder_key = VIEW_SPECS[view]["encoder_key"]
        if config.get(encoder_key, {}).get("method_dir"):
            return view
    return "smiles"


def _normalize_property_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    return text.strip()


def _parse_property_names_from_files(values: Any) -> List[str]:
    names: List[str] = []
    if values is None:
        return names
    if isinstance(values, str):
        raw_items = [x.strip() for x in values.split(",")]
    elif isinstance(values, (list, tuple, set)):
        raw_items = [str(x).strip() for x in values]
    else:
        raw_items = [str(values).strip()]
    for item in raw_items:
        name = _normalize_property_name(item)
        if name and name not in names:
            names.append(name)
    return names


def _resolve_committee_properties(config: dict, target_property: str, override: Optional[str]) -> List[str]:
    f5_cfg = _f5_cfg(config)
    names: List[str] = []

    if override is not None:
        names = _parse_property_names_from_files(override)
    if not names:
        names = _parse_property_names_from_files(f5_cfg.get("committee_properties"))
    if not names:
        names = _parse_property_names_from_files((config.get("property", {}) or {}).get("files"))

    target_name = _normalize_property_name(target_property)
    if target_name and target_name not in names:
        names.insert(0, target_name)
    if not names and target_name:
        names = [target_name]
    return names


def _parse_view_list(value: Any) -> List[str]:
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

    views: List[str] = []
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


def _select_proposal_views(config: dict, override: Optional[str], fallback_view: str) -> List[str]:
    f5_cfg = _f5_cfg(config)
    raw = override if override is not None else f5_cfg.get("proposal_views", "all")
    parsed = _parse_view_list(raw)

    available: List[str] = []
    ordered = []
    for view in config.get("alignment_views", list(SUPPORTED_VIEWS)):
        if view in SUPPORTED_VIEWS and view not in ordered:
            ordered.append(view)
    for view in SUPPORTED_VIEWS:
        if view not in ordered:
            ordered.append(view)
    for view in ordered:
        if not _is_view_enabled(config, view):
            continue
        encoder_key = VIEW_SPECS[view]["encoder_key"]
        if not config.get(encoder_key, {}).get("method_dir"):
            continue
        available.append(view)

    if parsed == ["all"] or not parsed:
        selected = list(available)
    else:
        selected = []
        for view in parsed:
            if view not in available:
                raise ValueError(
                    f"proposal view '{view}' is unavailable (disabled or encoder config missing)."
                )
            if view not in selected:
                selected.append(view)

    if fallback_view not in selected:
        selected.append(fallback_view)
    if not selected:
        selected = [fallback_view]
    return selected


def _resolve_view_device(config: dict, view: str) -> str:
    encoder_cfg = config.get(VIEW_SPECS[view]["encoder_key"], {})
    device = str(encoder_cfg.get("device", "auto")).strip() or "auto"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_module(module_name: str, path: Path):
    module_path = path.resolve()

    # Prefer package import when the module is within this repo, so relative
    # imports inside the target module (e.g., from ..utils import ...) work.
    import_name = None
    try:
        rel = module_path.relative_to(REPO_ROOT.resolve())
        if rel.suffix == ".py":
            parts = list(rel.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if parts and all(part.isidentifier() for part in parts):
                import_name = ".".join(parts)
    except Exception:
        import_name = None

    if import_name:
        try:
            return importlib.import_module(import_name)
        except Exception:
            # Fall back to direct file import for backward compatibility.
            pass

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    tokenizer_cls = getattr(tokenizer_mod, tokenizer_class)
    diffusion_backbone = backbone_mod.DiffusionBackbone

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

    tokenizer = tokenizer_cls.load(str(tokenizer_path))
    backbone = diffusion_backbone(
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
        "method_dir": method_dir,
        "method_cfg": method_cfg,
        "results_dir": results_dir,
        "base_results_dir": base_results_dir,
        "checkpoint_path": checkpoint_path,
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
    graph_tokenizer = tokenizer_mod.GraphTokenizer
    graph_backbone = backbone_mod.GraphDiffusionBackbone

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

    tokenizer = graph_tokenizer.load(str(tokenizer_path))
    backbone = graph_backbone(
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
        "method_dir": method_dir,
        "method_cfg": method_cfg,
        "results_dir": results_dir,
        "base_results_dir": base_results_dir,
        "checkpoint_path": checkpoint_path,
        "graph_config": graph_config,
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


def _embed_graph(smiles_list: List[str], assets: dict, device: str) -> Tuple[np.ndarray, List[int]]:
    if not smiles_list:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), []
    valid_indices = []
    graph_batches = []
    for idx, smi in enumerate(smiles_list):
        try:
            data = assets["tokenizer"].encode(smi)
            graph_batches.append(data)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[F5] Warning: graph encoding failed for molecule {idx}: {str(e)[:80]}")
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


def _load_view_assets(config: dict, view: str, device: str) -> dict:
    spec = VIEW_SPECS[view]
    encoder_cfg = config.get(spec["encoder_key"], {})
    if not encoder_cfg or not encoder_cfg.get("method_dir"):
        raise ValueError(f"Encoder config missing for view={view}")

    if spec["type"] == "graph":
        return _load_graph_backbone(encoder_cfg=encoder_cfg, device=device)
    return _load_sequence_backbone(
        encoder_cfg=encoder_cfg,
        device=device,
        tokenizer_module=spec["tokenizer_module"],
        tokenizer_class=spec["tokenizer_class"],
        tokenizer_filename=spec["tokenizer_file"],
    )


def _sanitize_sequence_inputs(inputs: List[str], tokenizer) -> Tuple[List[str], List[int]]:
    valid_inputs = []
    valid_indices = []
    for idx, text in enumerate(inputs):
        try:
            tokenizer.encode(text, add_special_tokens=True, padding=True, return_attention_mask=True)
            valid_inputs.append(text)
            valid_indices.append(idx)
        except Exception:
            continue
    return valid_inputs, valid_indices


def _embed_candidates(view: str, smiles_list: List[str], assets: dict, device: str) -> Tuple[np.ndarray, List[int]]:
    if view == "graph":
        return _embed_graph(smiles_list, assets, device)

    if view == "selfies":
        seq_inputs = []
        seq_indices = []
        for idx, smi in enumerate(smiles_list):
            converted = smiles_to_selfies(smi)
            if not converted:
                continue
            seq_inputs.append(converted)
            seq_indices.append(idx)
        seq_inputs, filtered_local_indices = _sanitize_sequence_inputs(seq_inputs, assets["tokenizer"])
        filtered_indices = [seq_indices[i] for i in filtered_local_indices]
        embeddings = _embed_sequence(seq_inputs, assets, device) if seq_inputs else np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
        return embeddings, filtered_indices

    seq_inputs, valid_indices = _sanitize_sequence_inputs(smiles_list, assets["tokenizer"])
    embeddings = _embed_sequence(seq_inputs, assets, device) if seq_inputs else np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
    return embeddings, valid_indices


def _load_d2_embeddings(results_dir: Path, view: str) -> np.ndarray:
    direct_path = results_dir / f"embeddings_{view}_d2.npy"
    if direct_path.exists():
        return np.load(direct_path)
    if view == "smiles":
        legacy_path = results_dir / "embeddings_d2.npy"
        if legacy_path.exists():
            return np.load(legacy_path)
    raise FileNotFoundError(
        f"D2 embeddings not found for view={view}. Expected {direct_path} "
        f"(or legacy embeddings_d2.npy for smiles). Run F1 with this view enabled first."
    )


def _default_property_model_path(results_dir: Path, property_name: str, view: str) -> Path:
    prop = _normalize_property_name(property_name)
    model_dir = results_dir / "step3_property"
    if prop:
        model_dir = model_dir / prop / "files"
    else:
        model_dir = model_dir / "files"
    if view == "smiles":
        return model_dir / f"{property_name}_mlp.pt"
    return model_dir / f"{property_name}_{view}_mlp.pt"


def _discover_property_model_paths(results_dir: Path, property_name: str) -> Dict[str, Path]:
    model_dir = results_dir / "step3_property"
    files_dir = model_dir / "files"
    prop = _normalize_property_name(property_name)
    prop_files_dir = model_dir / prop / "files" if prop else None
    prop_step_dir = model_dir / prop if prop else None
    discovered: Dict[str, Path] = {}

    for view in SUPPORTED_VIEWS:
        if view == "smiles":
            filename = f"{property_name}_mlp.pt"
        else:
            filename = f"{property_name}_{view}_mlp.pt"
        candidates = []
        if prop_files_dir is not None:
            candidates.append(prop_files_dir / filename)
        if prop_step_dir is not None:
            candidates.append(prop_step_dir / filename)
        candidates.extend([files_dir / filename, model_dir / filename])
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is not None:
            discovered[view] = model_path

    mv_filename = f"{property_name}_multiview_mean_mlp.pt"
    mv_candidates = []
    if prop_files_dir is not None:
        mv_candidates.append(prop_files_dir / mv_filename)
    if prop_step_dir is not None:
        mv_candidates.append(prop_step_dir / mv_filename)
    mv_candidates.extend([files_dir / mv_filename, model_dir / mv_filename])
    mv_path = next((p for p in mv_candidates if p.exists()), None)
    if mv_path is not None:
        discovered["multiview_mean"] = mv_path

    return discovered


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


class _TorchPropertyPredictor:
    def __init__(self, checkpoint: dict):
        if checkpoint.get("format") != "mvf_torch_mlp":
            raise ValueError("Unsupported torch property model format.")
        self.mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.scale = np.where(np.abs(self.scale) < 1e-12, 1.0, self.scale).astype(np.float32, copy=False)
        self.model = _PropertyMLP(
            input_dim=int(checkpoint["input_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            neurons=int(checkpoint["neurons"]),
            dropout=float(checkpoint.get("dropout", 0.0)),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features is None or len(features) == 0:
            return np.zeros((0,), dtype=np.float32)
        x = np.asarray(features, dtype=np.float32)
        x = (x - self.mean) / self.scale
        with torch.no_grad():
            preds = self.model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
        return preds


def _load_property_model(model_path: Path):
    if model_path.suffix in {".pt", ".pth"}:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "mvf_torch_mlp":
            return _TorchPropertyPredictor(checkpoint)
        raise ValueError(f"Unsupported torch property model checkpoint: {model_path}")

    if joblib is not None:
        return joblib.load(model_path)
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _build_prediction_columns_from_all_models(
    *,
    config: dict,
    results_dir: Path,
    smiles_list: List[str],
    model_paths: Dict[str, Path],
    use_alignment: bool,
    alignment_checkpoint: Optional[str],
) -> Dict[str, Any]:
    n = len(smiles_list)
    if n == 0:
        return {
            "prediction_by_model": {},
            "prediction_ensemble": np.zeros((0,), dtype=np.float32),
            "prediction_std": np.zeros((0,), dtype=np.float32),
            "prediction_valid_count": np.zeros((0,), dtype=np.int64),
            "model_order": [],
            "models_used": {},
        }

    view_cache: Dict[str, Dict[str, Any]] = {}
    prediction_by_model: Dict[str, np.ndarray] = {}
    models_used: Dict[str, str] = {}

    def _view_pack(view: str) -> Dict[str, Any]:
        cached = view_cache.get(view)
        if cached is not None:
            return cached

        encoder_cfg = config.get(VIEW_SPECS[view]["encoder_key"], {})
        device = encoder_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assets = _load_view_assets(config=config, view=view, device=device)
        embeddings, kept_indices = _embed_candidates(
            view=view,
            smiles_list=smiles_list,
            assets=assets,
            device=device,
        )
        if embeddings.size and use_alignment:
            view_hidden_dim = int(getattr(assets["backbone"], "hidden_size", 0)) or int(config.get("model", {}).get("projection_dim", 256))
            alignment_model = _load_alignment_model(
                results_dir=results_dir,
                view_dims={view: view_hidden_dim},
                config=config,
                checkpoint_override=alignment_checkpoint,
            )
            if alignment_model is None:
                raise FileNotFoundError("Alignment checkpoint not found for all-model F5 scoring with use_alignment=True")
            projection_device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings = _project_embeddings(alignment_model, view, embeddings, device=projection_device)
        packed = {
            "embeddings": embeddings,
            "kept_indices": [int(i) for i in kept_indices],
        }
        view_cache[view] = packed
        return packed

    # Per-view models
    for view in SUPPORTED_VIEWS:
        model_path = model_paths.get(view)
        if model_path is None:
            continue
        packed = _view_pack(view)
        pred_col = np.full((n,), np.nan, dtype=np.float32)
        if packed["embeddings"].size and packed["kept_indices"]:
            model = _load_property_model(model_path)
            preds = np.asarray(model.predict(packed["embeddings"]), dtype=np.float32).reshape(-1)
            for local_i, global_i in enumerate(packed["kept_indices"]):
                pred_col[global_i] = preds[local_i]
        prediction_by_model[view] = pred_col
        models_used[view] = str(model_path)

    # Multiview-mean model
    mv_path = model_paths.get("multiview_mean")
    if mv_path is not None:
        available_views = [v for v in SUPPORTED_VIEWS if v in prediction_by_model]
        common_indices = None
        for view in available_views:
            idx_set = set(view_cache[view]["kept_indices"])
            common_indices = idx_set if common_indices is None else common_indices.intersection(idx_set)
        common = sorted(common_indices) if common_indices else []

        pred_col = np.full((n,), np.nan, dtype=np.float32)
        if common:
            dim_set = set()
            fused_blocks = []
            for view in available_views:
                packed = view_cache[view]
                idx_to_row = {idx: row_i for row_i, idx in enumerate(packed["kept_indices"])}
                block = packed["embeddings"][[idx_to_row[i] for i in common]]
                fused_blocks.append(block)
                dim_set.add(int(block.shape[1]))
            if len(dim_set) == 1:
                fused = np.mean(np.stack(fused_blocks, axis=0), axis=0)
                mv_model = _load_property_model(mv_path)
                preds = np.asarray(mv_model.predict(fused), dtype=np.float32).reshape(-1)
                for local_i, global_i in enumerate(common):
                    pred_col[global_i] = preds[local_i]
            else:
                print(
                    "[F5] Warning: skipping multiview_mean model due to embedding-dimension mismatch "
                    f"across views: {sorted(dim_set)}"
                )

        prediction_by_model["multiview_mean"] = pred_col
        models_used["multiview_mean"] = str(mv_path)

    if not prediction_by_model:
        return {
            "prediction_by_model": {},
            "prediction_ensemble": np.full((n,), np.nan, dtype=np.float32),
            "prediction_std": np.full((n,), np.nan, dtype=np.float32),
            "prediction_valid_count": np.zeros((n,), dtype=np.int64),
            "model_order": [],
            "models_used": {},
        }

    ordered = [k for k in list(SUPPORTED_VIEWS) + ["multiview_mean"] if k in prediction_by_model]
    pred_matrix = np.column_stack([prediction_by_model[k] for k in ordered]).astype(np.float32, copy=False)
    valid_counts = np.sum(np.isfinite(pred_matrix), axis=1)
    with np.errstate(invalid="ignore"):
        ensemble = np.nanmean(pred_matrix, axis=1).astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        ensemble_std = np.nanstd(pred_matrix, axis=1).astype(np.float32, copy=False)
    ensemble[valid_counts == 0] = np.nan
    ensemble_std[valid_counts == 0] = np.nan

    return {
        "prediction_by_model": prediction_by_model,
        "prediction_ensemble": ensemble,
        "prediction_std": ensemble_std,
        "prediction_valid_count": valid_counts.astype(np.int64, copy=False),
        "model_order": ordered,
        "models_used": models_used,
    }


def _check_validity(smiles: str) -> bool:
    if Chem is None:
        return True
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _count_stars(smiles: str) -> int:
    return smiles.count("*") if isinstance(smiles, str) else 0


def _load_training_smiles(path: Path) -> set:
    if not path.exists():
        return set()
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    col = "p_smiles" if "p_smiles" in df.columns else "SMILES" if "SMILES" in df.columns else None
    if col is None:
        return set()
    return set(df[col].astype(str).tolist())


def _achievement_rates(preds: np.ndarray, target: float) -> dict:
    if preds is None or len(preds) == 0:
        return {
            "achievement_5p": 0.0,
            "achievement_10p": 0.0,
            "achievement_15p": 0.0,
            "achievement_20p": 0.0,
        }
    denom = max(abs(float(target)), 1e-9)
    return {
        "achievement_5p": float(np.mean(np.abs(preds - target) <= 0.05 * denom)),
        "achievement_10p": float(np.mean(np.abs(preds - target) <= 0.10 * denom)),
        "achievement_15p": float(np.mean(np.abs(preds - target) <= 0.15 * denom)),
        "achievement_20p": float(np.mean(np.abs(preds - target) <= 0.20 * denom)),
    }


def _compute_hits(preds: np.ndarray, target: float, epsilon: float, target_mode: str) -> np.ndarray:
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(preds - target) <= epsilon
    if mode == "ge":
        return preds >= target
    if mode == "le":
        return preds <= target
    raise ValueError(f"Unsupported target_mode={target_mode}. Use window|ge|le.")


def _compute_target_excess(preds: np.ndarray, target: float, target_mode: str) -> np.ndarray:
    """Signed distance to target boundary.

    Positive values indicate better achievement:
    - window: pred - target
    - ge: pred - target
    - le: target - pred
    """
    arr = np.asarray(preds, dtype=np.float32).reshape(-1)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return (float(target) - arr).astype(np.float32, copy=False)
    return (arr - float(target)).astype(np.float32, copy=False)


def _compute_target_violation(preds: np.ndarray, target: float, target_mode: str) -> np.ndarray:
    """Non-negative boundary violation in property units."""
    excess = _compute_target_excess(preds, target, target_mode)
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(excess).astype(np.float32, copy=False)
    return np.maximum(0.0, -excess).astype(np.float32, copy=False)


def _target_excess_axis_label(property_name: str, target_mode: str) -> str:
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return f"Target excess (target - predicted {property_name})"
    return f"Target excess (predicted {property_name} - target)"


_SA_SCORE_FN = None


def _get_sa_score_fn():
    global _SA_SCORE_FN
    if _SA_SCORE_FN is not None:
        return _SA_SCORE_FN
    chem_path = REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "utils" / "chemistry.py"
    if not chem_path.exists():
        _SA_SCORE_FN = False
        return None
    try:
        chem_mod = _load_module("mvf_sa_chemistry", chem_path)
        _SA_SCORE_FN = getattr(chem_mod, "compute_sa_score", False)
    except Exception:
        _SA_SCORE_FN = False
    return _SA_SCORE_FN if callable(_SA_SCORE_FN) else None


def _compute_sa_score(smiles: str) -> Optional[float]:
    fn = _get_sa_score_fn()
    if fn is None:
        return None
    try:
        score = fn(smiles)
    except Exception:
        return None
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def _count_aromatic_rings(smiles: str) -> Optional[int]:
    """Count aromatic rings in a SMILES string using RDKit."""
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.replace("*", "[H]"))
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem import rdMolDescriptors
        return int(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        return None


def _compute_pairwise_tanimoto_diversity(smiles_list: List[str], max_pairs: int = 10000) -> Optional[float]:
    """Compute mean pairwise Tanimoto diversity (1 - similarity) over accepted candidates."""
    if Chem is None or len(smiles_list) < 2:
        return None
    try:
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return None
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except Exception:
            continue
    n = len(fps)
    if n < 2:
        return None
    # For large sets, subsample pairs to avoid O(n^2) blowup
    if n * (n - 1) // 2 <= max_pairs:
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                total_dist += 1.0 - sim
                count += 1
        return total_dist / count if count > 0 else None
    else:
        rng = np.random.RandomState(42)
        indices = rng.randint(0, n, size=(max_pairs, 2))
        # Ensure different indices in each pair
        mask = indices[:, 0] != indices[:, 1]
        indices = indices[mask]
        total_dist = 0.0
        count = 0
        for i, j in indices:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_dist += 1.0 - sim
            count += 1
        return total_dist / count if count > 0 else None


def _parse_descriptor_min_constraints(config: dict) -> Dict[str, int]:
    """Extract hard minimum constraints from ood_aware_inverse.descriptor_constraints."""
    desc_cfg = config.get("ood_aware_inverse", {}).get("descriptor_constraints", {})
    if not desc_cfg or not isinstance(desc_cfg, dict):
        return {}
    mins: Dict[str, int] = {}
    for name, spec in desc_cfg.items():
        if isinstance(spec, dict) and spec.get("min") is not None:
            try:
                mins[str(name).strip().lower()] = int(spec["min"])
            except (ValueError, TypeError):
                pass
    return mins


def _canonicalize_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text or Chem is None:
        return text
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return text
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return text


def _parse_target_classes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [x.strip() for x in str(value).split(",")]
    return [x for x in items if x]


def _compile_polymer_patterns(patterns: Dict[str, str]) -> Dict[str, Any]:
    if Chem is None:
        return {}
    compiled = {}
    for name, smarts in (patterns or {}).items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                compiled[str(name)] = patt
        except Exception:
            continue
    return compiled


def _match_polymer_class(smiles: str, target_classes: List[str], compiled_patterns: Dict[str, Any]) -> Tuple[bool, str]:
    if not target_classes:
        return True, ""
    if Chem is None:
        return False, ""
    try:
        mol = Chem.MolFromSmiles(smiles.replace("*", "[*]"))
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return False, ""
    for class_name in target_classes:
        patt = compiled_patterns.get(class_name)
        if patt is None:
            continue
        try:
            if mol.HasSubstructMatch(patt):
                return True, class_name
        except Exception:
            continue
    return False, ""


def _create_generator(view: str, assets: dict, device: str, sampling_temperature: Optional[float], sampling_num_atoms: Optional[int]):
    method_dir = assets["method_dir"]
    method_cfg = assets.get("method_cfg", {}) or {}
    diffusion_cfg = method_cfg.get("diffusion", {}) or {}
    sampling_cfg = method_cfg.get("sampling", {}) or {}
    num_steps = int(diffusion_cfg.get("num_steps", 50))
    use_constraints = _to_bool(sampling_cfg.get("use_constraints", True), True)
    temperature = sampling_temperature if sampling_temperature is not None else float(sampling_cfg.get("temperature", 1.0))

    if view == "graph":
        graph_diffusion_mod = _load_module(
            f"graph_diffusion_{method_dir.name}",
            method_dir / "src" / "model" / "graph_diffusion.py",
        )
        graph_sampler_mod = _load_module(
            f"graph_sampler_{method_dir.name}",
            method_dir / "src" / "sampling" / "graph_sampler.py",
        )
        graph_cfg = assets.get("graph_config", {}) or {}
        diffusion_model = graph_diffusion_mod.GraphMaskingDiffusion(
            backbone=assets["backbone"],
            num_steps=num_steps,
            beta_min=float(diffusion_cfg.get("beta_min", 1e-4)),
            beta_max=float(diffusion_cfg.get("beta_max", 2e-2)),
            force_clean_t0=_to_bool(diffusion_cfg.get("force_clean_t0", False), False),
            node_mask_id=assets["tokenizer"].mask_id,
            edge_mask_id=assets["tokenizer"].edge_vocab.get("MASK", 5),
            node_pad_id=assets["tokenizer"].pad_id,
            edge_none_id=assets["tokenizer"].edge_vocab.get("NONE", 0),
            lambda_node=float(diffusion_cfg.get("lambda_node", 1.0)),
            lambda_edge=float(diffusion_cfg.get("lambda_edge", 0.5)),
        )
        diffusion_model.to(device)
        diffusion_model.eval()
        sampler = graph_sampler_mod.GraphSampler(
            backbone=diffusion_model.backbone,
            graph_tokenizer=assets["tokenizer"],
            num_steps=num_steps,
            device=device,
            atom_count_distribution=graph_cfg.get("atom_count_distribution"),
            use_constraints=use_constraints,
        )
        return {
            "view": view,
            "sampler": sampler,
            "temperature": float(temperature),
            "num_atoms": sampling_num_atoms,
        }

    diffusion_mod = _load_module(
        f"diffusion_{method_dir.name}",
        method_dir / "src" / "model" / "diffusion.py",
    )
    sampler_mod = _load_module(
        f"sampler_{method_dir.name}",
        method_dir / "src" / "sampling" / "sampler.py",
    )
    diffusion_model = diffusion_mod.DiscreteMaskingDiffusion(
        backbone=assets["backbone"],
        num_steps=num_steps,
        beta_min=float(diffusion_cfg.get("beta_min", 1e-4)),
        force_clean_t0=_to_bool(diffusion_cfg.get("force_clean_t0", False), False),
        mask_token_id=assets["tokenizer"].mask_token_id,
        pad_token_id=assets["tokenizer"].pad_token_id,
        bos_token_id=assets["tokenizer"].bos_token_id,
        eos_token_id=assets["tokenizer"].eos_token_id,
    )
    diffusion_model.to(device)
    diffusion_model.eval()
    sampler = sampler_mod.ConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=assets["tokenizer"],
        num_steps=num_steps,
        temperature=float(temperature),
        use_constraints=use_constraints,
        device=device,
    )

    selfies_to_psmiles = None
    if view == "selfies":
        selfies_utils_mod = _load_module(
            f"selfies_utils_{method_dir.name}",
            method_dir / "src" / "utils" / "selfies_utils.py",
        )
        selfies_to_psmiles = getattr(selfies_utils_mod, "selfies_to_psmiles")

    return {
        "view": view,
        "sampler": sampler,
        "seq_length": int(getattr(assets["tokenizer"], "max_length", 256)),
        "selfies_to_psmiles": selfies_to_psmiles,
    }


def _sample_batch_from_generator(generator: dict, n: int, batch_size: int) -> List[str]:
    if n <= 0:
        return []
    view = generator["view"]
    sampler = generator["sampler"]
    if view == "graph":
        raw = sampler.sample_batch(
            num_samples=int(n),
            batch_size=int(batch_size),
            show_progress=False,
            temperature=float(generator.get("temperature", 1.0)),
            num_atoms=generator.get("num_atoms"),
        )
        return [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]

    _, outputs = sampler.sample_batch(
        num_samples=int(n),
        seq_length=int(generator["seq_length"]),
        batch_size=int(batch_size),
        show_progress=False,
    )
    if view == "selfies":
        converter = generator.get("selfies_to_psmiles")
        converted = []
        for text in outputs:
            try:
                psmiles = converter(text) if converter is not None else None
            except Exception:
                psmiles = None
            if psmiles:
                converted.append(str(psmiles).strip())
        return [s for s in converted if s]

    return [str(x).strip() for x in outputs if isinstance(x, str) and str(x).strip()]


def _resample_candidates_until_target(
    *,
    config: dict,
    args,
    proposal_views: List[str],
    scoring_view: str,
    scoring_assets: dict,
    scoring_device: str,
    property_model,
    alignment_model,
    training_set: set,
    target_value: float,
    epsilon: float,
    target_mode: str,
) -> dict:
    f5_cfg = _f5_cfg(config)
    sampling_target = int(args.sampling_target if args.sampling_target is not None else f5_cfg.get("sampling_target", 100))
    sampling_num_per_batch = int(args.sampling_num_per_batch if args.sampling_num_per_batch is not None else f5_cfg.get("sampling_num_per_batch", 512))
    sampling_batch_size = int(args.sampling_batch_size if args.sampling_batch_size is not None else f5_cfg.get("sampling_batch_size", 128))
    sampling_max_batches = int(args.sampling_max_batches if args.sampling_max_batches is not None else f5_cfg.get("sampling_max_batches", 200))

    sampling_temperature = args.sampling_temperature if args.sampling_temperature is not None else f5_cfg.get("sampling_temperature", None)
    sampling_num_atoms = args.sampling_num_atoms if args.sampling_num_atoms is not None else f5_cfg.get("sampling_num_atoms", None)
    if sampling_num_atoms in ("", "none", None):
        sampling_num_atoms = None
    elif sampling_num_atoms is not None:
        sampling_num_atoms = int(sampling_num_atoms)
    if sampling_temperature in ("", "none", None):
        sampling_temperature = None
    elif sampling_temperature is not None:
        sampling_temperature = float(sampling_temperature)

    target_class = args.target_class if args.target_class is not None else f5_cfg.get("target_class", "")
    require_validity = _to_bool(f5_cfg.get("require_validity", True), True)
    require_two_stars = _to_bool(f5_cfg.get("require_two_stars", True), True)
    require_novel = _to_bool(f5_cfg.get("require_novel", True), True)
    require_unique = _to_bool(f5_cfg.get("require_unique", True), True)
    max_sa = args.max_sa if args.max_sa is not None else f5_cfg.get("max_sa", None)
    per_view_min_hits = _to_int_or_none(
        args.per_view_min_hits if args.per_view_min_hits is not None else f5_cfg.get("per_view_min_hits", 0)
    )
    per_view_quota_relax_after_batches = _to_int_or_none(
        args.per_view_quota_relax_after_batches
        if args.per_view_quota_relax_after_batches is not None
        else f5_cfg.get("per_view_quota_relax_after_batches", 0)
    )
    if max_sa in ("", "none", None):
        max_sa = None
    else:
        max_sa = float(max_sa)
    if per_view_min_hits is None:
        per_view_min_hits = 0
    per_view_min_hits = int(per_view_min_hits)
    if per_view_min_hits < 0:
        raise ValueError("per_view_min_hits must be >= 0")
    if per_view_quota_relax_after_batches is None:
        per_view_quota_relax_after_batches = 0
    per_view_quota_relax_after_batches = int(per_view_quota_relax_after_batches)
    if per_view_quota_relax_after_batches < 0:
        raise ValueError("per_view_quota_relax_after_batches must be >= 0")
    if per_view_min_hits > 0 and per_view_quota_relax_after_batches == 0:
        # Keep quota phase short by default, then relax to avoid stalling when
        # some views cannot satisfy the requested hit floor.
        per_view_quota_relax_after_batches = max(1, len(proposal_views) * 2)

    if sampling_target <= 0:
        raise ValueError("sampling_target must be > 0")
    if sampling_num_per_batch <= 0 or sampling_batch_size <= 0:
        raise ValueError("sampling_num_per_batch and sampling_batch_size must both be > 0")
    if sampling_max_batches <= 0:
        raise ValueError("sampling_max_batches must be > 0")
    if not proposal_views:
        raise ValueError("proposal_views must be non-empty.")

    patterns = config.get("polymer_classes") or f5_cfg.get("polymer_class_patterns") or DEFAULT_POLYMER_PATTERNS
    target_classes = _parse_target_classes(target_class)
    compiled_patterns = _compile_polymer_patterns(patterns)
    for class_name in target_classes:
        if class_name not in compiled_patterns:
            raise ValueError(f"Unknown target class '{class_name}'. Available classes: {sorted(compiled_patterns.keys())}")
    if target_classes:
        print(f"[F5 resample] target_class filter enabled: {','.join(target_classes)}")
    if per_view_min_hits > 0:
        print(
            "[F5 resample] per-view balancing enabled: "
            f"min_hits_per_view={per_view_min_hits}, relax_after_batch={per_view_quota_relax_after_batches}"
        )

    descriptor_min_constraints = _parse_descriptor_min_constraints(config)
    if descriptor_min_constraints:
        print(f"[F5 resample] descriptor min constraints: {descriptor_min_constraints}")

    scoring_generator = _create_generator(
        view=scoring_view,
        assets=scoring_assets,
        device=scoring_device,
        sampling_temperature=sampling_temperature,
        sampling_num_atoms=sampling_num_atoms,
    )
    projection_device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = Counter()
    scored_records: List[dict] = []
    scored_embeddings: List[np.ndarray] = []
    structural_valid_smiles: List[str] = []
    generated_smiles_all: List[str] = []
    seen_keys: set = set()
    accepted = 0
    accepted_by_view: Counter = Counter()
    active_generator_view: Optional[str] = None
    active_generator_assets: Optional[dict] = None
    active_generator: Optional[dict] = None
    available_proposal_views = list(proposal_views)
    disabled_proposal_views: Dict[str, str] = {}

    def _get_proposal_generator(view: str) -> dict:
        nonlocal active_generator_view, active_generator_assets, active_generator
        if view == scoring_view:
            return scoring_generator
        if active_generator_view == view and active_generator is not None:
            return active_generator

        # Keep at most one non-scoring proposal generator resident to reduce peak memory.
        if active_generator_assets is not None:
            try:
                active_generator_assets.get("backbone").to("cpu")
            except Exception:
                pass
            active_generator_assets = None
            active_generator = None
            active_generator_view = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        proposal_device = _resolve_view_device(config, view)
        proposal_assets = _load_view_assets(config=config, view=view, device=proposal_device)
        proposal_generator = _create_generator(
            view=view,
            assets=proposal_assets,
            device=proposal_device,
            sampling_temperature=sampling_temperature,
            sampling_num_atoms=sampling_num_atoms,
        )
        active_generator_assets = proposal_assets
        active_generator = proposal_generator
        active_generator_view = view
        return proposal_generator

    for batch_idx in range(1, sampling_max_batches + 1):
        if accepted >= sampling_target:
            break

        if not available_proposal_views:
            stats["stop_no_proposal_views"] += 1
            print("[F5 resample] no proposal views remain; stopping early.")
            break

        proposal_view = available_proposal_views[(batch_idx - 1) % len(available_proposal_views)]
        try:
            generator = _get_proposal_generator(proposal_view)
        except Exception as exc:
            stats["proposal_view_init_failed"] += 1
            stats[f"proposal_view_init_failed_{proposal_view}"] += 1
            disabled_proposal_views[proposal_view] = str(exc)
            available_proposal_views = [v for v in available_proposal_views if v != proposal_view]
            print(f"[F5 resample] disabling view={proposal_view} due to init error: {exc}")
            continue

        try:
            batch_smiles = _sample_batch_from_generator(generator, sampling_num_per_batch, sampling_batch_size)
        except Exception as exc:
            stats["proposal_view_sample_failed"] += 1
            stats[f"proposal_view_sample_failed_{proposal_view}"] += 1
            disabled_proposal_views[proposal_view] = str(exc)
            available_proposal_views = [v for v in available_proposal_views if v != proposal_view]
            print(f"[F5 resample] disabling view={proposal_view} due to sample error: {exc}")
            continue

        stats[f"n_generated_{proposal_view}"] += len(batch_smiles)
        stats[f"n_batches_{proposal_view}"] += 1
        stats["n_generated"] += len(batch_smiles)
        generated_smiles_all.extend(batch_smiles)
        if not batch_smiles:
            stats["empty_batches"] += 1
            continue

        prefilter_smiles: List[str] = []
        prefilter_meta: List[dict] = []

        def _bump(key: str) -> None:
            stats[key] += 1
            stats[f"{key}_{proposal_view}"] += 1

        for smi in batch_smiles:
            text = str(smi).strip()
            if not text:
                _bump("reject_empty")
                continue

            is_valid = _check_validity(text)
            is_two_star = _count_stars(text) == 2
            if is_valid:
                stats["n_valid_any"] += 1
                stats[f"n_valid_any_{proposal_view}"] += 1
            if is_valid and is_two_star:
                stats["n_structural_valid"] += 1
                stats[f"n_structural_valid_{proposal_view}"] += 1
                structural_valid_smiles.append(text)

            if require_validity and not is_valid:
                _bump("reject_invalid")
                continue
            if require_two_stars and not is_two_star:
                _bump("reject_two_star")
                continue

            canonical_key = _canonicalize_smiles(text)
            if require_unique and canonical_key in seen_keys:
                _bump("reject_duplicate")
                continue

            is_novel = text not in training_set
            if require_novel and not is_novel:
                _bump("reject_non_novel")
                continue

            class_ok, matched_class = _match_polymer_class(text, target_classes, compiled_patterns)
            if target_classes and not class_ok:
                _bump("reject_class")
                continue

            sa_value = None
            if max_sa is not None:
                sa_value = _compute_sa_score(text)
                if sa_value is None or sa_value >= max_sa:
                    _bump("reject_sa")
                    continue

            # Hard descriptor minimum constraints (e.g., aromatic_ring_count >= 2)
            descriptor_ok = True
            if descriptor_min_constraints:
                for desc_name, min_val in descriptor_min_constraints.items():
                    if desc_name == "aromatic_ring_count":
                        count = _count_aromatic_rings(text)
                        if count is not None and count < min_val:
                            descriptor_ok = False
                            break
                    elif desc_name == "ring_count":
                        if Chem is not None:
                            try:
                                mol = Chem.MolFromSmiles(text.replace("*", "[H]"))
                                if mol is None:
                                    mol = Chem.MolFromSmiles(text)
                                if mol is not None and mol.GetRingInfo().NumRings() < min_val:
                                    descriptor_ok = False
                                    break
                            except Exception:
                                pass
            if not descriptor_ok:
                _bump("reject_descriptor_min")
                continue

            prefilter_smiles.append(text)
            prefilter_meta.append(
                {
                    "smiles": text,
                    "canonical_smiles": canonical_key,
                    "proposal_view": proposal_view,
                    "is_valid": bool(is_valid),
                    "is_two_star": bool(is_two_star),
                    "is_novel": bool(is_novel),
                    "class_match": bool(class_ok) if target_classes else True,
                    "matched_class": matched_class,
                    "sa_score": sa_value,
                    "batch_idx": batch_idx,
                }
            )
            if require_unique:
                seen_keys.add(canonical_key)

        if not prefilter_smiles:
            print(
                f"[F5 resample] batch={batch_idx} view={proposal_view} generated={len(batch_smiles)} prefilter=0 "
                f"accepted={accepted}/{sampling_target}"
            )
            continue

        embeddings, kept_indices = _embed_candidates(
            view=scoring_view,
            smiles_list=prefilter_smiles,
            assets=scoring_assets,
            device=scoring_device,
        )
        stats["n_prefilter"] += len(prefilter_smiles)
        stats[f"n_prefilter_{proposal_view}"] += len(prefilter_smiles)
        if len(kept_indices) < len(prefilter_smiles):
            dropped = len(prefilter_smiles) - len(kept_indices)
            stats["reject_embed"] += dropped
            stats[f"reject_embed_{proposal_view}"] += dropped

        if embeddings.size and alignment_model is not None:
            embeddings = _project_embeddings(alignment_model, scoring_view, embeddings, device=projection_device)

        if embeddings.size == 0:
            print(
                f"[F5 resample] batch={batch_idx} generated={len(batch_smiles)} prefilter={len(prefilter_smiles)} scored=0 accepted={accepted}/{sampling_target}"
            )
            continue

        kept_meta = [prefilter_meta[idx] for idx in kept_indices]
        preds = property_model.predict(embeddings)
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        hits = _compute_hits(preds, target_value, epsilon, target_mode)

        scored_this_batch = 0
        for row_idx, meta in enumerate(kept_meta):
            pred_value = float(preds[row_idx])
            hit = bool(hits[row_idx])
            excess_value = float(_compute_target_excess(np.asarray([pred_value], dtype=np.float32), target_value, target_mode)[0])
            violation_value = float(_compute_target_violation(np.asarray([pred_value], dtype=np.float32), target_value, target_mode)[0])
            record = {
                **meta,
                "prediction": pred_value,
                "abs_error": abs(pred_value - target_value),
                "target_excess": excess_value,
                "target_violation": violation_value,
                "property_hit": hit,
                "accepted": False,
            }
            if hit and accepted < sampling_target:
                if per_view_min_hits > 0 and batch_idx <= per_view_quota_relax_after_batches:
                    underfilled = [v for v in proposal_views if accepted_by_view[v] < per_view_min_hits]
                    if underfilled and meta.get("proposal_view") not in underfilled:
                        stats["reject_view_quota"] += 1
                        stats[f"reject_view_quota_{proposal_view}"] += 1
                        hit = False

            if hit and accepted < sampling_target:
                record["accepted"] = True
                accepted += 1
                stats["n_hits"] += 1
                stats[f"n_hits_{proposal_view}"] += 1
                accepted_by_view[proposal_view] += 1
            scored_records.append(record)
            scored_embeddings.append(embeddings[row_idx].astype(np.float32, copy=False))
            scored_this_batch += 1
            if accepted >= sampling_target:
                break

        stats["n_scored"] += scored_this_batch
        stats[f"n_scored_{proposal_view}"] += scored_this_batch
        print(
            f"[F5 resample] batch={batch_idx} view={proposal_view} generated={len(batch_smiles)} prefilter={len(prefilter_smiles)} "
            f"scored={scored_this_batch} accepted={accepted}/{sampling_target}"
        )
        if accepted >= sampling_target:
            break

    if active_generator_assets is not None:
        try:
            active_generator_assets.get("backbone").to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored_df = pd.DataFrame(scored_records)
    if scored_df.empty:
        scored_df = pd.DataFrame(
            columns=[
                "smiles",
                "prediction",
                "abs_error",
                "target_excess",
                "target_violation",
                "property_hit",
                "accepted",
            ]
        )
    accepted_df = scored_df[scored_df["accepted"] == True].head(sampling_target).copy()  # noqa: E712

    scored_embeddings_np = None
    if scored_embeddings:
        scored_embeddings_np = np.stack(scored_embeddings, axis=0)

    if len(accepted_df) == 0 and target_classes and stats.get("reject_class", 0) > 0:
        print(
            f"[F5 resample] warning: class filter rejected {int(stats.get('reject_class', 0))} "
            "candidates; consider relaxing foundation_inverse.target_class."
        )

    per_view_stats = {}
    tracked_keys = [
        "n_generated",
        "n_batches",
        "n_valid_any",
        "n_structural_valid",
        "n_prefilter",
        "n_scored",
        "n_hits",
        "reject_empty",
        "reject_invalid",
        "reject_two_star",
        "reject_duplicate",
        "reject_non_novel",
        "reject_class",
        "reject_sa",
        "reject_embed",
        "reject_view_quota",
    ]
    for view in proposal_views:
        row = {}
        for key in tracked_keys:
            row[key] = int(stats.get(f"{key}_{view}", 0))
        row["accepted"] = int(accepted_by_view.get(view, 0))
        per_view_stats[view] = row

    result = {
        "candidate_source": "resample",
        "generated_smiles": generated_smiles_all,
        "structurally_valid_smiles": structural_valid_smiles,
        "scored_df": scored_df,
        "accepted_df": accepted_df,
        "scored_embeddings": scored_embeddings_np,
        "sampling_target": sampling_target,
        "sampling_num_per_batch": sampling_num_per_batch,
        "sampling_batch_size": sampling_batch_size,
        "sampling_max_batches": sampling_max_batches,
        "proposal_views": list(proposal_views),
        "proposal_views_remaining": list(available_proposal_views),
        "proposal_views_disabled": dict(disabled_proposal_views),
        "scoring_view": scoring_view,
        "target_classes": target_classes,
        "require_validity": require_validity,
        "require_two_stars": require_two_stars,
        "require_novel": require_novel,
        "require_unique": require_unique,
        "max_sa": max_sa,
        "per_view_min_hits": per_view_min_hits,
        "per_view_quota_relax_after_batches": per_view_quota_relax_after_batches,
        "accepted_by_view": {k: int(v) for k, v in accepted_by_view.items()},
        "per_view_stats": per_view_stats,
        "stats": dict(stats),
        "completed": len(accepted_df) >= sampling_target,
    }
    return result


def main(args):
    config = load_config(args.config)
    f5_cfg = _f5_cfg(config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step5_foundation_inverse")
    property_step_dirs = ensure_step_dirs(results_dir, "step5_foundation_inverse", args.property)
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")
    save_config(config, property_step_dirs["files_dir"] / "config_used.yaml")

    encoder_view = _select_encoder_view(config, args.encoder_view)
    proposal_views = _select_proposal_views(config, args.proposal_views, fallback_view=encoder_view)
    committee_properties = _resolve_committee_properties(config, args.property, args.committee_properties)

    encoder_device = _resolve_view_device(config, encoder_view)
    assets = _load_view_assets(config=config, view=encoder_view, device=encoder_device)
    view_hidden_dim = int(getattr(assets["backbone"], "hidden_size", 0)) or int(config.get("model", {}).get("projection_dim", 256))

    alignment_model = None
    if args.use_alignment:
        view_dims = {encoder_view: view_hidden_dim}
        alignment_model = _load_alignment_model(results_dir, view_dims, config, args.alignment_checkpoint)
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for --use_alignment")

    property_model_mode = args.property_model_mode
    if property_model_mode is None:
        property_model_mode = str(f5_cfg.get("property_model_mode", "all"))
    property_model_mode = str(property_model_mode).strip().lower()
    if property_model_mode not in {"single", "all"}:
        raise ValueError("property_model_mode must be one of: single|all")
    if property_model_mode == "all" and args.property_model_path:
        raise ValueError("--property_model_path is only supported when property_model_mode=single")

    model_path = args.property_model_path
    if model_path is None:
        model_path = _default_property_model_path(results_dir, args.property, encoder_view)
    else:
        model_path = _resolve_path(model_path)

    if not Path(model_path).exists():
        discovered = _discover_property_model_paths(results_dir, args.property)
        fallback = discovered.get(encoder_view) or discovered.get("smiles")
        if fallback is not None and Path(fallback).exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Property model not found: {model_path}")

    model = _load_property_model(Path(model_path))
    target_value = float(args.target)
    epsilon = float(args.epsilon)
    target_mode = str(args.target_mode).strip().lower()

    train_smiles_path = _resolve_path(config["paths"]["polymer_file"])
    training_set = _load_training_smiles(train_smiles_path)

    t0 = time.time()
    source_meta = {"candidate_source": "resample"}
    sampled = _resample_candidates_until_target(
        config=config,
        args=args,
        proposal_views=proposal_views,
        scoring_view=encoder_view,
        scoring_assets=assets,
        scoring_device=encoder_device,
        property_model=model,
        alignment_model=alignment_model,
        training_set=training_set,
        target_value=target_value,
        epsilon=epsilon,
        target_mode=target_mode,
    )
    smiles_list = sampled["generated_smiles"]
    structurally_valid_smiles = sampled["structurally_valid_smiles"]
    scored_df = sampled["scored_df"]
    scored_embeddings = sampled["scored_embeddings"]
    source_meta.update(
        {
            "sampling_target": int(sampled["sampling_target"]),
            "sampling_num_per_batch": int(sampled["sampling_num_per_batch"]),
            "sampling_batch_size": int(sampled["sampling_batch_size"]),
            "sampling_max_batches": int(sampled["sampling_max_batches"]),
            "proposal_views": sampled["proposal_views"],
            "scoring_view": sampled["scoring_view"],
            "target_classes": sampled["target_classes"],
            "require_validity": bool(sampled["require_validity"]),
            "require_two_stars": bool(sampled["require_two_stars"]),
            "require_novel": bool(sampled["require_novel"]),
            "require_unique": bool(sampled["require_unique"]),
            "max_sa": sampled["max_sa"],
            "per_view_min_hits": int(sampled.get("per_view_min_hits", 0)),
            "per_view_quota_relax_after_batches": int(sampled.get("per_view_quota_relax_after_batches", 0)),
            "accepted_by_view": sampled.get("accepted_by_view", {}),
            "per_view_stats": sampled.get("per_view_stats", {}),
            "stats": sampled["stats"],
            "sampling_completed": bool(sampled["completed"]),
        }
    )

    # Optional all-model aggregation:
    # score with all available F3 property heads across committee properties,
    # then use target-property committee mean/std for acceptance.
    if property_model_mode == "all" and not scored_df.empty:
        committee_models_used: Dict[str, Dict[str, str]] = {}
        committee_model_order: Dict[str, List[str]] = {}
        missing_properties: List[str] = []
        smiles_for_committee = scored_df["smiles"].astype(str).tolist()
        for committee_prop in committee_properties:
            all_model_paths = _discover_property_model_paths(results_dir, committee_prop)
            if committee_prop == args.property and encoder_view not in all_model_paths and Path(model_path).exists():
                # Keep backward compatibility for primary scoring model location.
                all_model_paths[encoder_view] = Path(model_path)
            if not all_model_paths:
                if committee_prop == args.property:
                    raise FileNotFoundError(
                        f"No F3 property models found for target property={args.property} under {results_dir / 'step3_property'}"
                    )
                missing_properties.append(committee_prop)
                continue

            pred_pack = _build_prediction_columns_from_all_models(
                config=config,
                results_dir=results_dir,
                smiles_list=smiles_for_committee,
                model_paths=all_model_paths,
                use_alignment=bool(args.use_alignment),
                alignment_checkpoint=args.alignment_checkpoint,
            )
            if not pred_pack["prediction_by_model"]:
                if committee_prop == args.property:
                    raise RuntimeError(
                        f"Failed to compute committee predictions for target property={args.property}."
                    )
                missing_properties.append(committee_prop)
                continue

            for model_name, pred_values in pred_pack["prediction_by_model"].items():
                scored_df[f"pred_{committee_prop}_{model_name}"] = pred_values
                if committee_prop == args.property:
                    # Preserve legacy target-property column names.
                    scored_df[f"prediction_{model_name}"] = pred_values

            scored_df[f"pred_{committee_prop}_mean"] = np.asarray(pred_pack["prediction_ensemble"], dtype=np.float32).reshape(-1)
            scored_df[f"pred_{committee_prop}_std"] = np.asarray(pred_pack["prediction_std"], dtype=np.float32).reshape(-1)
            scored_df[f"pred_{committee_prop}_n_models"] = np.asarray(pred_pack["prediction_valid_count"], dtype=np.int64).reshape(-1)
            committee_models_used[committee_prop] = pred_pack["models_used"]
            committee_model_order[committee_prop] = [str(x) for x in pred_pack.get("model_order", [])]

        target_mean_col = f"pred_{args.property}_mean"
        target_std_col = f"pred_{args.property}_std"
        target_count_col = f"pred_{args.property}_n_models"
        if target_mean_col not in scored_df.columns:
            raise RuntimeError(
                f"Missing target committee prediction column '{target_mean_col}'. "
                "Ensure step3 models exist for the target property."
            )

        ensemble_pred = pd.to_numeric(scored_df[target_mean_col], errors="coerce").to_numpy(dtype=np.float32)
        scored_df["prediction"] = ensemble_pred
        if target_std_col in scored_df.columns:
            scored_df["prediction_std"] = pd.to_numeric(scored_df[target_std_col], errors="coerce")
        else:
            scored_df["prediction_std"] = np.nan
        if target_count_col in scored_df.columns:
            scored_df["prediction_n_models"] = pd.to_numeric(scored_df[target_count_col], errors="coerce")
        else:
            scored_df["prediction_n_models"] = np.nan

        scored_df["abs_error"] = np.abs(ensemble_pred - target_value)
        scored_df["target_excess"] = _compute_target_excess(ensemble_pred, target_value, target_mode)
        scored_df["target_violation"] = _compute_target_violation(ensemble_pred, target_value, target_mode)
        valid_pred = np.isfinite(ensemble_pred)
        hit_mask = _compute_hits(ensemble_pred, target_value, epsilon, target_mode)
        hit_mask = np.logical_and(hit_mask, valid_pred)
        scored_df["property_hit"] = hit_mask
        scored_df["accepted"] = hit_mask

        source_meta.update(
            {
                "property_model_mode": "all",
                "prediction_aggregation": "mean",
                "committee_properties_requested": committee_properties,
                "committee_properties_scored": sorted(committee_models_used.keys()),
                "committee_properties_missing": missing_properties,
                "property_models_used": committee_models_used,
                "property_model_order": committee_model_order,
                "target_prediction_column": target_mean_col,
                "target_uncertainty_column": target_std_col,
            }
        )
    elif property_model_mode == "all":
        source_meta.update(
            {
                "property_model_mode": "all",
                "prediction_aggregation": "mean",
                "committee_properties_requested": committee_properties,
                "committee_properties_scored": [],
                "committee_properties_missing": committee_properties,
                "property_models_used": {},
                "property_model_order": {},
            }
        )
    else:
        source_meta.update(
            {
                "property_model_mode": "single",
                "property_models_used": {encoder_view: str(Path(model_path))},
            }
        )

    elapsed_sec = time.time() - t0
    if scored_df.empty:
        scored_df = pd.DataFrame(
            columns=[
                "smiles",
                "prediction",
                "abs_error",
                "target_excess",
                "target_violation",
                "property_hit",
                "accepted",
            ]
        )

    scored_smiles = scored_df["smiles"].astype(str).tolist() if "smiles" in scored_df.columns else []
    preds = scored_df["prediction"].to_numpy(dtype=np.float32) if "prediction" in scored_df.columns else np.array([], dtype=np.float32)
    hits = scored_df["property_hit"].to_numpy(dtype=bool) if "property_hit" in scored_df.columns else np.array([], dtype=bool)
    accepted_mask = scored_df["accepted"].to_numpy(dtype=bool) if "accepted" in scored_df.columns else hits

    n_generated = len(smiles_list)
    n_valid = len(structurally_valid_smiles)
    n_scored = len(scored_smiles)
    n_hits = int(np.sum(accepted_mask)) if n_scored else 0
    success_rate = n_hits / n_valid if n_valid else 0.0

    validity = n_valid / n_generated if n_generated else 0.0
    uniqueness = len(set(structurally_valid_smiles)) / n_valid if n_valid else 0.0
    novelty = sum(1 for s in structurally_valid_smiles if s not in training_set) / n_valid if n_valid else 0.0

    d2_distance_scores = None
    rerank_metrics = {
        "rerank_applied": False,
    }

    if args.rerank_strategy == "d2_distance" and scored_embeddings is not None and len(scored_smiles):
        d2_embeddings = _load_d2_embeddings(results_dir, encoder_view)
        if args.use_alignment and alignment_model is not None:
            device_proj = "cuda" if torch.cuda.is_available() else "cpu"
            d2_embeddings = _project_embeddings(alignment_model, encoder_view, d2_embeddings, device=device_proj)
        distances = knn_distances(scored_embeddings, d2_embeddings, k=args.ood_k)
        d2_distance_scores = distances.mean(axis=1)
        order = np.argsort(d2_distance_scores)
        top_k = min(int(args.rerank_top_k), len(order))
        if top_k > 0:
            top_hits = hits[order[:top_k]]
            rerank_metrics = {
                "rerank_applied": True,
                "rerank_strategy": "d2_distance",
                "rerank_top_k": top_k,
                "rerank_hits": int(top_hits.sum()),
                "rerank_success_rate": round(float(top_hits.sum()) / top_k, 4),
            }
        else:
            rerank_metrics = {"rerank_applied": False}

    # Compute pairwise Tanimoto diversity over accepted candidates
    accepted_smiles_for_div = scored_df.loc[accepted_mask, "smiles"].astype(str).tolist() if n_hits > 1 else []
    avg_diversity = _compute_pairwise_tanimoto_diversity(accepted_smiles_for_div)
    if avg_diversity is not None:
        avg_diversity = round(avg_diversity, 4)

    target_excess = _compute_target_excess(preds, target_value, target_mode) if preds.size else np.array([], dtype=np.float32)
    target_violation = _compute_target_violation(preds, target_value, target_mode) if preds.size else np.array([], dtype=np.float32)
    mean_target_excess = float(np.nanmean(target_excess)) if target_excess.size else np.nan
    mean_target_violation = float(np.nanmean(target_violation)) if target_violation.size else np.nan

    metrics_row = {
        "method": "Multi_View_Foundation",
        "representation": _view_to_representation(encoder_view),
        "model_size": assets["model_size"],
        "property": args.property,
        "target_value": target_value,
        "target_mode": target_mode,
        "epsilon": epsilon,
        "candidate_source": "resample",
        "n_generated": n_generated,
        "n_valid": n_valid,
        "n_scored": n_scored,
        "n_hits": n_hits,
        "success_rate": round(success_rate, 4),
        "validity": round(validity, 4),
        "validity_two_stars": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "novelty": round(novelty, 4),
        "avg_diversity": avg_diversity,
        "mean_target_excess": round(mean_target_excess, 6) if np.isfinite(mean_target_excess) else np.nan,
        "mean_target_violation": round(mean_target_violation, 6) if np.isfinite(mean_target_violation) else np.nan,
        **_achievement_rates(preds, target_value),
        "sampling_time_sec": round(elapsed_sec, 2),
        "valid_per_compute": round(n_valid / max(elapsed_sec, 1e-9), 4) if n_valid else 0.0,
        "hits_per_compute": round(n_hits / max(elapsed_sec, 1e-9), 4) if n_hits else 0.0,
        **rerank_metrics,
    }

    if rerank_metrics.get("rerank_applied"):
        metrics_row["valid_per_compute_rerank"] = round(rerank_metrics.get("rerank_hits", 0) / max(elapsed_sec, 1e-9), 4)
    else:
        metrics_row["rerank_strategy"] = args.rerank_strategy

    save_csv(
        pd.DataFrame([metrics_row]),
        property_step_dirs["metrics_dir"] / "metrics_inverse.csv",
        legacy_paths=[
            step_dirs["metrics_dir"] / "metrics_inverse.csv",
            results_dir / "metrics_inverse.csv",
        ],
        index=False,
    )

    out_dir = step_dirs["step_dir"]
    files_dir = property_step_dirs["files_dir"]
    legacy_files_dir = step_dirs["files_dir"]

    if d2_distance_scores is not None and len(scored_df) == len(d2_distance_scores):
        scored_df = scored_df.copy()
        scored_df["d2_distance"] = d2_distance_scores
    if "property" not in scored_df.columns:
        scored_df = scored_df.copy()
    scored_df["property"] = args.property

    accepted_df = scored_df[scored_df["accepted"] == True].copy() if "accepted" in scored_df.columns else scored_df[hits].copy()  # noqa: E712
    if "property" not in accepted_df.columns:
        accepted_df = accepted_df.copy()
    accepted_df["property"] = args.property

    save_csv(
        scored_df,
        files_dir / "candidate_scores.csv",
        legacy_paths=[
            legacy_files_dir / "candidate_scores.csv",
            out_dir / "candidate_scores.csv",
        ],
        index=False,
    )
    save_csv(
        scored_df,
        files_dir / f"candidate_scores_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"candidate_scores_{args.property}.csv",
            out_dir / f"candidate_scores_{args.property}.csv",
        ],
        index=False,
    )
    save_csv(
        accepted_df,
        files_dir / "accepted_candidates.csv",
        legacy_paths=[
            legacy_files_dir / "accepted_candidates.csv",
            out_dir / "accepted_candidates.csv",
        ],
        index=False,
    )
    save_csv(
        accepted_df,
        files_dir / f"accepted_candidates_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"accepted_candidates_{args.property}.csv",
            out_dir / f"accepted_candidates_{args.property}.csv",
        ],
        index=False,
    )

    report_df, report_summary = _build_f5_accepted_polymer_report(
        accepted_df=accepted_df,
        property_name=args.property,
        target_value=target_value,
        target_mode=target_mode,
        epsilon=epsilon,
        sampling_target=int(source_meta.get("sampling_target", len(accepted_df))),
    )
    save_csv(
        report_df,
        files_dir / "accepted_polymer_report.csv",
        legacy_paths=[
            legacy_files_dir / "accepted_polymer_report.csv",
            out_dir / "accepted_polymer_report.csv",
        ],
        index=False,
    )
    save_csv(
        report_df,
        files_dir / f"accepted_polymer_report_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"accepted_polymer_report_{args.property}.csv",
            out_dir / f"accepted_polymer_report_{args.property}.csv",
        ],
        index=False,
    )
    save_json(
        report_summary,
        files_dir / "accepted_polymer_summary.json",
        legacy_paths=[
            legacy_files_dir / "accepted_polymer_summary.json",
            out_dir / "accepted_polymer_summary.json",
        ],
    )
    save_json(
        report_summary,
        files_dir / f"accepted_polymer_summary_{args.property}.json",
        legacy_paths=[
            legacy_files_dir / f"accepted_polymer_summary_{args.property}.json",
            out_dir / f"accepted_polymer_summary_{args.property}.json",
        ],
    )

    save_json(
        {
            **source_meta,
            "encoder_view": encoder_view,
            "property": args.property,
            "target_value": target_value,
            "target_mode": target_mode,
            "epsilon": epsilon,
            "rerank_strategy": args.rerank_strategy,
            "rerank_top_k": int(args.rerank_top_k),
            "use_alignment": bool(args.use_alignment),
            "n_generated": int(n_generated),
            "n_structurally_valid": int(n_valid),
            "n_scored": int(n_scored),
            "n_hits": int(n_hits),
        },
        files_dir / "run_meta.json",
        legacy_paths=[
            legacy_files_dir / "run_meta.json",
            out_dir / "run_meta.json",
        ],
    )
    save_json(
        {
            **source_meta,
            "encoder_view": encoder_view,
            "property": args.property,
            "target_value": target_value,
            "target_mode": target_mode,
            "epsilon": epsilon,
            "rerank_strategy": args.rerank_strategy,
            "rerank_top_k": int(args.rerank_top_k),
            "use_alignment": bool(args.use_alignment),
            "n_generated": int(n_generated),
            "n_structurally_valid": int(n_valid),
            "n_scored": int(n_scored),
            "n_hits": int(n_hits),
        },
        files_dir / f"run_meta_{args.property}.json",
        legacy_paths=[
            legacy_files_dir / f"run_meta_{args.property}.json",
            out_dir / f"run_meta_{args.property}.json",
        ],
    )

    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(f5_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F5 figures.")
        generate_figures = False
    if generate_figures:
        _plot_f5_diagnostics(
            scored_df=scored_df,
            property_name=args.property,
            target_value=target_value,
            target_mode=target_mode,
            epsilon=epsilon,
            source_meta=source_meta,
            figures_dir=property_step_dirs["figures_dir"],
        )
        _plot_f5_accepted_polymer_overview(
            report_df=report_df,
            property_name=args.property,
            target_value=target_value,
            target_mode=target_mode,
            epsilon=epsilon,
            figures_dir=property_step_dirs["figures_dir"],
        )
        _plot_f5_accepted_polymer_gallery(
            report_df=report_df,
            property_name=args.property,
            figures_dir=property_step_dirs["figures_dir"],
            top_n=min(36, int(source_meta.get("sampling_target", 100))),
        )

    print(f"Saved metrics_inverse.csv to {property_step_dirs['step_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--encoder_view", type=str, default=None, choices=list(SUPPORTED_VIEWS))
    parser.add_argument("--proposal_views", type=str, default=None, help="Comma list or 'all' for F5 proposal generation views.")
    parser.add_argument("--property", type=str, required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--target_mode", type=str, default="window", choices=["window", "ge", "le"])
    parser.add_argument("--epsilon", type=float, default=30.0)
    parser.add_argument("--sampling_target", type=int, default=None)
    parser.add_argument("--sampling_num_per_batch", type=int, default=None)
    parser.add_argument("--sampling_batch_size", type=int, default=None)
    parser.add_argument("--sampling_max_batches", type=int, default=None)
    parser.add_argument("--sampling_temperature", type=float, default=None)
    parser.add_argument("--sampling_num_atoms", type=int, default=None)
    parser.add_argument("--target_class", type=str, default=None)
    parser.add_argument("--max_sa", type=float, default=None)
    parser.add_argument(
        "--per_view_min_hits",
        type=int,
        default=None,
        help="Minimum accepted hits per proposal view during the initial balancing phase.",
    )
    parser.add_argument(
        "--per_view_quota_relax_after_batches",
        type=int,
        default=None,
        help="Batch index after which per-view minimum-hit quota is relaxed (0=auto).",
    )
    parser.add_argument("--property_model_mode", type=str, default=None, choices=["single", "all"])
    parser.add_argument("--committee_properties", type=str, default=None, help="Comma list for committee property exports; defaults to property.files.")
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--rerank_strategy", type=str, default="d2_distance")
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--ood_k", type=int, default=5)
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
