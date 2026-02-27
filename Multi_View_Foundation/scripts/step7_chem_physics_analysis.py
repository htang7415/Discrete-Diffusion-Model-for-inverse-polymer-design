#!/usr/bin/env python
"""F7: Chemistry + physics analysis for OOD-aware inverse design results.

This step summarizes science-facing evidence across properties (Tg/Tm/Td/Eg):
- descriptor shifts (reference vs F5 candidates vs F6 top-k)
- polymer motif enrichment
- physics-consistency checks from predefined heuristic rules
- nearest-neighbor explanations against property reference datasets
- paper-style figures (per-property + cross-property summary)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:  # pragma: no cover
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    rdBase.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None
    rdMolDescriptors = None


DEFAULT_PROPERTIES = ["Tg", "Tm", "Td", "Eg"]

DESCRIPTOR_COLUMNS = [
    "mol_wt",
    "heavy_atoms",
    "ring_count",
    "aromatic_ring_count",
    "aromatic_atom_fraction",
    "fraction_csp3",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "hba",
    "hbd",
    "hetero_atom_fraction",
    "bertz_ct",
    "sa_score",
    "star_count",
]

POLYMER_MOTIF_SMARTS = {
    "polyimide": "[#6](=O)-[#7]-[#6](=O)",
    "polyester": "[#6](=O)-[#8]-[#6]",
    "polyamide": "[#6](=O)-[#7]-[#6]",
    "polyurethane": "[#8]-[#6](=O)-[#7]",
    "polyether": "[#6]-[#8]-[#6]",
    "polysiloxane": "[Si]-[#8]-[Si]",
    "polycarbonate": "[#8]-[#6](=O)-[#8]",
    "polysulfone": "[#6]-[S](=O)(=O)-[#6]",
    "polyacrylate": "[#6]-[#6](=O)-[#8]",
    "polystyrene": "c1ccccc1",
}

# Heuristic direction: +1 means higher in top-k is more physics-consistent,
# -1 means lower in top-k is more physics-consistent.
PHYSICS_RULES = {
    "Tg": [
        ("aromatic_ring_count", +1, "Higher chain rigidity tends to increase Tg."),
        ("ring_count", +1, "More cyclic content can reduce segmental mobility."),
        ("rotatable_bonds", -1, "Fewer rotatable bonds generally increase Tg."),
        ("fraction_csp3", -1, "Lower aliphatic flexibility often correlates with higher Tg."),
        ("polyimide", +1, "Imide-like motifs are commonly associated with high thermal performance."),
    ],
    "Tm": [
        ("ring_count", +1, "More rigid repeat units can support higher melting transitions."),
        ("rotatable_bonds", -1, "Reduced flexibility can improve ordered packing."),
        ("aromatic_ring_count", +1, "Aromatic/rigid motifs often raise thermal transitions."),
        ("polyamide", +1, "Hydrogen-bonding motifs may support higher Tm."),
    ],
    "Td": [
        ("aromatic_ring_count", +1, "Aromatic stabilization is often linked to higher Td."),
        ("ring_count", +1, "More rigid structures can improve thermal decomposition resistance."),
        ("rotatable_bonds", -1, "Excess flexibility can reduce thermal robustness."),
        ("polysulfone", +1, "Sulfone-containing motifs are often thermally stable."),
    ],
    "Eg": [
        ("aromatic_ring_count", -1, "For bandgap-like Eg, stronger conjugation often lowers Eg."),
        ("fraction_csp3", +1, "Less conjugated/aliphatic character can increase Eg."),
        ("ring_count", -1, "Extensive aromatic cyclic systems can reduce Eg."),
        ("polystyrene", -1, "Aromatic-rich motifs may correspond to lower Eg."),
    ],
}

PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.frameon": False,
    "legend.fontsize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 600,
}

COLOR_PRIMARY = "#2A5CAA"
COLOR_SECONDARY = "#E76F51"
COLOR_ACCENT = "#2A9D8F"
COLOR_MUTED = "#9AA3AF"


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


def _numeric_array(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)
    return arr[np.isfinite(arr)]


def _normalize_property_name(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    return text.strip()


def _parse_properties(args, cfg: dict) -> List[str]:
    raw = args.properties if args.properties else cfg.get("properties", DEFAULT_PROPERTIES)
    if isinstance(raw, str):
        props = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        props = [str(x).strip() for x in raw if str(x).strip()]
    if not props:
        return list(DEFAULT_PROPERTIES)
    return props


def _safe_mkdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sa_score_fn():
    chem_path = REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "utils" / "chemistry.py"
    if not chem_path.exists():
        return None
    try:
        mod = _load_module("mvf_chemistry_utils", chem_path)
    except Exception:
        return None
    fn = getattr(mod, "compute_sa_score", None)
    return fn if callable(fn) else None


def _compile_motifs() -> Dict[str, object]:
    if Chem is None:
        return {}
    compiled = {}
    for name, smarts in POLYMER_MOTIF_SMARTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                compiled[name] = patt
        except Exception:
            continue
    return compiled


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


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    y_std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    pooled = np.sqrt(max((x_std * x_std + y_std * y_std) / 2.0, 1e-12))
    return float((x_mean - y_mean) / pooled)


def _prediction_uncertainty_series(df: pd.DataFrame, property_name: str) -> pd.Series:
    prop = _normalize_property_name(property_name)
    candidates = [
        "prediction_uncertainty",
        f"pred_{prop}_std",
        "prediction_std",
    ]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _resolve_property_columns(df: pd.DataFrame, property_name: str) -> Tuple[str, str]:
    smiles_candidates = ["SMILES", "smiles", "p_smiles", "psmiles"]
    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError("Property CSV must contain a SMILES column.")

    prop_lower = str(property_name).strip().lower()
    value_col = None
    for col in df.columns:
        if col.lower() == prop_lower:
            value_col = col
            break
    if value_col is None:
        for col in df.columns:
            c = col.lower()
            if c in {"smiles", "p_smiles", "psmiles", "pid", "polymer_id", "id"}:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break
    if value_col is None:
        raise ValueError(f"Could not determine property value column for {property_name}.")
    return smiles_col, value_col


def _load_property_reference(property_dir: Path, property_name: str, max_samples: Optional[int]) -> pd.DataFrame:
    path = property_dir / f"{property_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing property file: {path}")
    df = pd.read_csv(path)
    smiles_col, value_col = _resolve_property_columns(df, property_name)
    out = df[[smiles_col, value_col]].copy()
    out = out.rename(columns={smiles_col: "smiles", value_col: "property_value"})
    out["property_value"] = pd.to_numeric(out["property_value"], errors="coerce")
    out = out.dropna(subset=["smiles", "property_value"])
    out["smiles"] = out["smiles"].astype(str)
    if max_samples is not None and max_samples > 0:
        out = out.head(int(max_samples)).copy()
    out["property"] = property_name
    return out


def _resolve_template_path(template: Optional[str], property_name: str) -> Optional[Path]:
    if template is None:
        return None
    text = str(template).strip()
    if not text:
        return None
    try:
        text = text.format(property=property_name)
    except Exception:
        pass
    return _resolve_path(text)


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _default_candidate_paths(results_dir: Path, property_name: str) -> List[Path]:
    return [
        results_dir / "step5_foundation_inverse" / property_name / "files" / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / property_name / "files" / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / property_name / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / property_name / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / "files" / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / "files" / f"{property_name}_candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / "files" / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / "candidate_scores.csv",
    ]


def _default_topk_paths(results_dir: Path, property_name: str) -> List[Path]:
    return [
        results_dir / "step6_ood_aware_inverse" / property_name / "files" / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / property_name / "files" / "ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / property_name / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / property_name / "ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / "files" / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / "files" / f"{property_name}_ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / "files" / "ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / "ood_objective_topk.csv",
    ]


def _resolve_property_file_inputs(
    *,
    property_name: str,
    results_dir: Path,
    candidate_template: Optional[str],
    topk_template: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    candidate_path = _resolve_template_path(candidate_template, property_name)
    if candidate_path is not None and not candidate_path.exists():
        candidate_path = None
    if candidate_path is None:
        candidate_path = _find_first_existing(_default_candidate_paths(results_dir, property_name))

    topk_path = _resolve_template_path(topk_template, property_name)
    if topk_path is not None and not topk_path.exists():
        topk_path = None
    if topk_path is None:
        topk_path = _find_first_existing(_default_topk_paths(results_dir, property_name))
    return candidate_path, topk_path


def _is_generic_scores_file(path: Optional[Path]) -> bool:
    if path is None:
        return False
    return path.name in {"candidate_scores.csv", "ood_objective_scores.csv", "ood_objective_topk.csv"}


def _filter_property_rows_if_available(
    df: pd.DataFrame,
    *,
    property_name: str,
    source_path: Path,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if "property" not in df.columns:
        return df, None
    prop_series = df["property"].astype(str).str.strip()
    match_mask = prop_series == property_name
    if bool(match_mask.any()):
        return df.loc[match_mask].copy(), None
    seen = [x for x in sorted(prop_series.unique().tolist()) if x]
    seen_preview = ",".join(seen[:6]) if seen else "(empty)"
    msg = (
        f"{property_name}: property column mismatch in {source_path}. "
        f"Found properties={seen_preview}."
    )
    return pd.DataFrame(columns=df.columns), msg


def _descriptor_and_motif_rows(
    smiles_list: List[str],
    property_name: str,
    split: str,
    sa_score_fn,
    compiled_motifs: Dict[str, object],
) -> pd.DataFrame:
    rows: List[dict] = []
    for smiles in smiles_list:
        mol = _smiles_to_mol(smiles)
        if mol is None:
            continue
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        aromatic_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))
        hetero_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
        row = {
            "property": property_name,
            "split": split,
            "smiles": str(smiles),
            "mol_wt": float(Descriptors.MolWt(mol)) if Descriptors is not None else np.nan,
            "heavy_atoms": heavy_atoms,
            "ring_count": float(Descriptors.RingCount(mol)) if Descriptors is not None else np.nan,
            "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)) if rdMolDescriptors is not None else np.nan,
            "aromatic_atom_fraction": float(aromatic_atoms / max(heavy_atoms, 1.0)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)) if rdMolDescriptors is not None else np.nan,
            "rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)) if Descriptors is not None else np.nan,
            "tpsa": float(Descriptors.TPSA(mol)) if Descriptors is not None else np.nan,
            "logp": float(Descriptors.MolLogP(mol)) if Descriptors is not None else np.nan,
            "hba": float(Descriptors.NumHAcceptors(mol)) if Descriptors is not None else np.nan,
            "hbd": float(Descriptors.NumHDonors(mol)) if Descriptors is not None else np.nan,
            "hetero_atom_fraction": float(hetero_atoms / max(heavy_atoms, 1.0)),
            "bertz_ct": float(Descriptors.BertzCT(mol)) if Descriptors is not None else np.nan,
            "star_count": float(str(smiles).count("*")),
            "sa_score": np.nan,
        }
        if sa_score_fn is not None:
            try:
                score = sa_score_fn(str(smiles))
                if score is not None:
                    row["sa_score"] = float(score)
            except Exception:
                pass
        for motif_name, patt in compiled_motifs.items():
            key = f"motif_{motif_name}"
            try:
                row[key] = bool(mol.HasSubstructMatch(patt))
            except Exception:
                row[key] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_stats(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(values)), float(np.std(values)), float(np.median(values))


def _descriptor_shift_table(
    property_name: str,
    ref_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for desc in DESCRIPTOR_COLUMNS:
        x_ref = ref_df[desc].dropna().to_numpy(dtype=np.float32) if desc in ref_df.columns else np.zeros((0,), dtype=np.float32)
        x_cand = cand_df[desc].dropna().to_numpy(dtype=np.float32) if desc in cand_df.columns else np.zeros((0,), dtype=np.float32)
        x_top = topk_df[desc].dropna().to_numpy(dtype=np.float32) if desc in topk_df.columns else np.zeros((0,), dtype=np.float32)

        ref_mean, ref_std, ref_med = _summary_stats(x_ref)
        cand_mean, cand_std, cand_med = _summary_stats(x_cand)
        top_mean, top_std, top_med = _summary_stats(x_top)
        rows.append(
            {
                "property": property_name,
                "descriptor": desc,
                "ref_n": int(x_ref.size),
                "cand_n": int(x_cand.size),
                "topk_n": int(x_top.size),
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "ref_median": ref_med,
                "candidate_mean": cand_mean,
                "candidate_std": cand_std,
                "candidate_median": cand_med,
                "topk_mean": top_mean,
                "topk_std": top_std,
                "topk_median": top_med,
                "delta_topk_vs_ref": (top_mean - ref_mean) if np.isfinite(top_mean) and np.isfinite(ref_mean) else np.nan,
                "delta_topk_vs_candidate": (top_mean - cand_mean) if np.isfinite(top_mean) and np.isfinite(cand_mean) else np.nan,
                "cohens_d_topk_vs_ref": _cohens_d(x_top, x_ref),
                "cohens_d_topk_vs_candidate": _cohens_d(x_top, x_cand),
            }
        )
    return pd.DataFrame(rows)


def _motif_enrichment_table(
    property_name: str,
    ref_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> pd.DataFrame:
    motif_cols = [c for c in ref_df.columns if c.startswith("motif_")]
    rows = []
    eps = 1e-6
    for col in motif_cols:
        ref = ref_df[col].astype(float).to_numpy(dtype=np.float32) if col in ref_df.columns else np.zeros((0,), dtype=np.float32)
        cand = cand_df[col].astype(float).to_numpy(dtype=np.float32) if col in cand_df.columns else np.zeros((0,), dtype=np.float32)
        top = topk_df[col].astype(float).to_numpy(dtype=np.float32) if col in topk_df.columns else np.zeros((0,), dtype=np.float32)
        ref_freq = float(np.mean(ref)) if ref.size else 0.0
        cand_freq = float(np.mean(cand)) if cand.size else 0.0
        top_freq = float(np.mean(top)) if top.size else 0.0
        ratio = (top_freq + eps) / (ref_freq + eps)
        rows.append(
            {
                "property": property_name,
                "motif": col.replace("motif_", ""),
                "ref_freq": ref_freq,
                "candidate_freq": cand_freq,
                "topk_freq": top_freq,
                "delta_freq_topk_vs_ref": top_freq - ref_freq,
                "delta_freq_topk_vs_candidate": top_freq - cand_freq,
                "enrichment_ratio_topk_vs_ref": ratio,
                "log2_enrichment_topk_vs_ref": float(np.log2(ratio)),
            }
        )
    return pd.DataFrame(rows)


def _physics_consistency_table(
    property_name: str,
    descriptor_shift_df: pd.DataFrame,
    motif_enrich_df: pd.DataFrame,
) -> pd.DataFrame:
    rules = PHYSICS_RULES.get(property_name, [])
    rows = []
    for feature, expected, rationale in rules:
        if feature in set(descriptor_shift_df["descriptor"].astype(str)):
            row = descriptor_shift_df[descriptor_shift_df["descriptor"] == feature].iloc[0]
            observed = float(row["delta_topk_vs_ref"]) if pd.notna(row["delta_topk_vs_ref"]) else np.nan
            source = "descriptor"
        elif feature in set(motif_enrich_df["motif"].astype(str)):
            row = motif_enrich_df[motif_enrich_df["motif"] == feature].iloc[0]
            observed = float(row["delta_freq_topk_vs_ref"]) if pd.notna(row["delta_freq_topk_vs_ref"]) else np.nan
            source = "motif"
        else:
            observed = np.nan
            source = "missing"
        sign_match = bool(observed * expected > 0) if np.isfinite(observed) else False
        rows.append(
            {
                "property": property_name,
                "feature": feature,
                "source": source,
                "expected_direction": int(expected),
                "observed_delta_topk_vs_ref": observed,
                "sign_match": sign_match,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def _fingerprint(mol):
    if AllChem is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None


def _nearest_neighbor_explanations(
    property_name: str,
    ref_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    ref_desc: pd.DataFrame,
    top_desc: pd.DataFrame,
) -> pd.DataFrame:
    if Chem is None or DataStructs is None:
        return pd.DataFrame()
    if ref_df.empty or topk_df.empty:
        return pd.DataFrame()

    ref_smiles = ref_df["smiles"].astype(str).tolist()
    ref_values = ref_df["property_value"].to_numpy(dtype=np.float32)
    ref_mols = [_smiles_to_mol(s) for s in ref_smiles]
    ref_fps = [_fingerprint(m) if m is not None else None for m in ref_mols]

    valid_ref_idx = [i for i, fp in enumerate(ref_fps) if fp is not None]
    if not valid_ref_idx:
        return pd.DataFrame()
    ref_fps_valid = [ref_fps[i] for i in valid_ref_idx]

    ref_desc_map = {str(r["smiles"]): r for _, r in ref_desc.iterrows()}
    top_desc_map = {str(r["smiles"]): r for _, r in top_desc.iterrows()}

    rows = []
    for _, top_row in topk_df.iterrows():
        top_smiles = str(top_row.get("smiles", ""))
        mol = _smiles_to_mol(top_smiles)
        if mol is None:
            continue
        fp = _fingerprint(mol)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps_valid)
        if not sims:
            continue
        local_idx = int(np.argmax(np.asarray(sims, dtype=np.float32)))
        best_ref_global_idx = valid_ref_idx[local_idx]
        best_ref_smiles = ref_smiles[best_ref_global_idx]
        best_sim = float(sims[local_idx])
        best_ref_value = float(ref_values[best_ref_global_idx])

        top_desc_row = top_desc_map.get(top_smiles)
        ref_desc_row = ref_desc_map.get(best_ref_smiles)
        delta_aromatic = np.nan
        delta_rotatable = np.nan
        delta_mw = np.nan
        if top_desc_row is not None and ref_desc_row is not None:
            delta_aromatic = float(top_desc_row.get("aromatic_ring_count", np.nan) - ref_desc_row.get("aromatic_ring_count", np.nan))
            delta_rotatable = float(top_desc_row.get("rotatable_bonds", np.nan) - ref_desc_row.get("rotatable_bonds", np.nan))
            delta_mw = float(top_desc_row.get("mol_wt", np.nan) - ref_desc_row.get("mol_wt", np.nan))

        rows.append(
            {
                "property": property_name,
                "topk_smiles": top_smiles,
                "topk_prediction": pd.to_numeric(top_row.get("prediction"), errors="coerce"),
                "topk_d2_distance": pd.to_numeric(top_row.get("d2_distance"), errors="coerce"),
                "nearest_reference_smiles": best_ref_smiles,
                "nearest_reference_value": best_ref_value,
                "nearest_tanimoto": best_sim,
                "delta_aromatic_ring_count": delta_aromatic,
                "delta_rotatable_bonds": delta_rotatable,
                "delta_mol_wt": delta_mw,
            }
        )
    return pd.DataFrame(rows)


def _property_error_from_predictions(
    pred: np.ndarray,
    target: Optional[float],
    target_mode: str,
) -> np.ndarray:
    if target is None:
        return np.full_like(pred, np.nan, dtype=np.float32)
    target = float(target)
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(pred - target).astype(np.float32, copy=False)
    if mode == "ge":
        return np.maximum(0.0, target - pred).astype(np.float32, copy=False)
    if mode == "le":
        return np.maximum(0.0, pred - target).astype(np.float32, copy=False)
    return np.abs(pred - target).astype(np.float32, copy=False)


def _target_excess_from_predictions(
    pred: np.ndarray,
    target: Optional[float],
    target_mode: str,
) -> np.ndarray:
    if target is None:
        return np.full_like(pred, np.nan, dtype=np.float32)
    target = float(target)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return (target - pred).astype(np.float32, copy=False)
    return (pred - target).astype(np.float32, copy=False)


def _target_excess_axis_label(property_name: str, target_mode: str) -> str:
    mode = str(target_mode).strip().lower()
    if mode == "ge":
        return f"Target excess (predicted {property_name} - target)"
    if mode == "le":
        return f"Target excess (target - predicted {property_name})"
    return "Signed error (prediction - target)"


def _resolve_target_config(
    property_name: str,
    cfg_step7: dict,
    cfg_f5: dict,
) -> Tuple[Optional[float], str, Optional[float]]:
    targets = cfg_step7.get("targets", {}) or {}
    target_modes = cfg_step7.get("target_modes", {}) or {}
    epsilons = cfg_step7.get("epsilons", {}) or {}

    target = _to_float_or_none(targets.get(property_name))
    if target is None and str(cfg_f5.get("property", "")).strip() == property_name:
        target = _to_float_or_none(cfg_f5.get("target"))

    target_mode = str(target_modes.get(property_name, "")).strip()
    if not target_mode and str(cfg_f5.get("property", "")).strip() == property_name:
        target_mode = str(cfg_f5.get("target_mode", "window")).strip()
    if not target_mode:
        target_mode = "window"

    epsilon = _to_float_or_none(epsilons.get(property_name))
    if epsilon is None and str(cfg_f5.get("property", "")).strip() == property_name:
        epsilon = _to_float_or_none(cfg_f5.get("epsilon"))
    return target, target_mode, epsilon


def _resolve_reference_class(
    *,
    property_name: str,
    cfg_step7: dict,
    cfg_f5: dict,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    override_class: Optional[str] = None,
) -> str:
    text = str(override_class or "").strip().lower()
    if text:
        return text

    class_map = cfg_step7.get("reference_classes", {}) or {}
    if isinstance(class_map, dict):
        text = str(class_map.get(property_name, "")).strip().lower()
        if text:
            return text

    text = str(cfg_step7.get("reference_class", "")).strip().lower()
    if text:
        return text

    # Use F5 target_class when this property is the active F5 target.
    f5_prop = str(cfg_f5.get("property", "")).strip()
    if f5_prop == property_name:
        text = str(cfg_f5.get("target_class", "")).strip().lower()
        if text:
            return text

    # Optional per-property class map from F5 configs.
    f5_class_map = cfg_f5.get("target_classes", {}) or {}
    if isinstance(f5_class_map, dict):
        text = str(f5_class_map.get(property_name, "")).strip().lower()
        if text:
            return text

    # Last fallback: infer single matched_class from candidate/top-k tables.
    for df in (topk_df, candidate_df):
        if "matched_class" not in df.columns:
            continue
        series = df["matched_class"].astype(str).str.strip().str.lower()
        uniq = sorted({x for x in series.tolist() if x})
        if len(uniq) == 1:
            return uniq[0]
    return ""


def _apply_reference_class_filter(
    *,
    property_name: str,
    ref_df: pd.DataFrame,
    ref_desc: pd.DataFrame,
    reference_class: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    cls = str(reference_class).strip().lower()
    if not cls:
        return ref_df, ref_desc, None
    motif_col = f"motif_{cls}"
    if motif_col not in ref_desc.columns:
        return ref_df, ref_desc, f"{property_name}: reference class filter '{cls}' not found in motif columns; using unfiltered reference."

    mask = ref_desc[motif_col].fillna(False).astype(bool)
    n_match = int(mask.sum())
    if n_match <= 0:
        return ref_df, ref_desc, f"{property_name}: reference class filter '{cls}' matched 0 rows; using unfiltered reference."

    filtered_desc = ref_desc.loc[mask].copy()
    allowed_smiles = set(filtered_desc["smiles"].astype(str).tolist())
    filtered_ref = ref_df[ref_df["smiles"].astype(str).isin(allowed_smiles)].copy()
    if filtered_ref.empty:
        return ref_df, ref_desc, f"{property_name}: reference class filter '{cls}' removed all property rows; using unfiltered reference."
    return filtered_ref, filtered_desc, None


def _prepare_candidate_error_df(candidate_df: pd.DataFrame, target: Optional[float], target_mode: str) -> pd.DataFrame:
    cdf = candidate_df.copy()
    if "prediction" in cdf.columns:
        cdf["prediction"] = pd.to_numeric(cdf["prediction"], errors="coerce")
    if "d2_distance" in cdf.columns:
        cdf["d2_distance"] = pd.to_numeric(cdf["d2_distance"], errors="coerce")

    if "target_excess" in cdf.columns:
        cdf["target_excess"] = pd.to_numeric(cdf["target_excess"], errors="coerce")
    elif "prediction" in cdf.columns and target is not None:
        cdf["target_excess"] = _target_excess_from_predictions(
            cdf["prediction"].to_numpy(dtype=np.float32),
            target=target,
            target_mode=target_mode,
        )
    else:
        cdf["target_excess"] = np.nan

    if "target_violation" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["target_violation"], errors="coerce")
    elif "prediction" in cdf.columns and target is not None:
        cdf["target_violation"] = _property_error_from_predictions(
            cdf["prediction"].to_numpy(dtype=np.float32),
            target=target,
            target_mode=target_mode,
        )
    elif "abs_error" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["abs_error"], errors="coerce")
    elif "property_error_normed" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["property_error_normed"], errors="coerce")
    else:
        cdf["target_violation"] = np.nan

    cdf["property_error"] = cdf["target_violation"]
    return cdf


def _plot_descriptor_detail_figure(property_name: str, descriptor_shift_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or descriptor_shift_df.empty:
        return
    ds = descriptor_shift_df.copy().replace([np.inf, -np.inf], np.nan)
    ds["cohens_d_topk_vs_ref"] = pd.to_numeric(ds.get("cohens_d_topk_vs_ref"), errors="coerce")
    ds["ref_median"] = pd.to_numeric(ds.get("ref_median"), errors="coerce")
    ds["topk_median"] = pd.to_numeric(ds.get("topk_median"), errors="coerce")
    ds = ds.dropna(subset=["descriptor", "cohens_d_topk_vs_ref"])
    if ds.empty:
        return
    ds["abs_d"] = np.abs(ds["cohens_d_topk_vs_ref"])
    ds = ds.sort_values("abs_d", ascending=False).head(12).iloc[::-1]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        ax0, ax1 = axes

        colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in ds["cohens_d_topk_vs_ref"].to_numpy(dtype=np.float32)]
        ax0.barh(ds["descriptor"].astype(str), ds["cohens_d_topk_vs_ref"], color=colors, alpha=0.9)
        ax0.axvline(0.0, color="#111111", linewidth=1.0)
        ax0.set_xlabel("Cohen's d (top-k vs reference)")
        ax0.grid(axis="x", alpha=0.25)

        pair_df = ds.dropna(subset=["ref_median", "topk_median"]).copy()
        if not pair_df.empty:
            ax1.scatter(pair_df["ref_median"], pair_df["topk_median"], s=50, color=COLOR_PRIMARY, alpha=0.9)
            lo = float(np.nanmin(np.concatenate([pair_df["ref_median"].to_numpy(dtype=np.float32), pair_df["topk_median"].to_numpy(dtype=np.float32)])))
            hi = float(np.nanmax(np.concatenate([pair_df["ref_median"].to_numpy(dtype=np.float32), pair_df["topk_median"].to_numpy(dtype=np.float32)])))
            if np.isfinite(lo) and np.isfinite(hi):
                ax1.plot([lo, hi], [lo, hi], linestyle="--", color=COLOR_MUTED, linewidth=1.1)
            for _, row in pair_df.iterrows():
                ax1.text(float(row["ref_median"]), float(row["topk_median"]), str(row["descriptor"]), fontsize=15, alpha=0.8)
            ax1.set_xlabel("Reference median")
            ax1.set_ylabel("Top-k median")
            ax1.grid(alpha=0.25)
        else:
            ax1.text(0.5, 0.5, "No median values", ha="center", va="center")
            ax1.set_axis_off()

        fig.tight_layout()
        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_motif_detail_figure(property_name: str, motif_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or motif_df.empty:
        return
    mf = motif_df.copy().replace([np.inf, -np.inf], np.nan)
    mf["log2_enrichment_topk_vs_ref"] = pd.to_numeric(mf.get("log2_enrichment_topk_vs_ref"), errors="coerce")
    mf["ref_freq"] = pd.to_numeric(mf.get("ref_freq"), errors="coerce")
    mf["candidate_freq"] = pd.to_numeric(mf.get("candidate_freq"), errors="coerce")
    mf["topk_freq"] = pd.to_numeric(mf.get("topk_freq"), errors="coerce")
    mf = mf.dropna(subset=["motif", "log2_enrichment_topk_vs_ref"])
    if mf.empty:
        return
    mf["abs_log2"] = np.abs(mf["log2_enrichment_topk_vs_ref"])
    mf = mf.sort_values("abs_log2", ascending=False).head(12).iloc[::-1]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        ax0, ax1 = axes

        colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in mf["log2_enrichment_topk_vs_ref"].to_numpy(dtype=np.float32)]
        ax0.barh(mf["motif"].astype(str), mf["log2_enrichment_topk_vs_ref"], color=colors, alpha=0.9)
        ax0.axvline(0.0, color="#111111", linewidth=1.0)
        ax0.set_xlabel("log2(top-k / reference)")
        ax0.grid(axis="x", alpha=0.25)

        x = np.arange(len(mf), dtype=np.float32)
        width = 0.25
        ax1.bar(x - width, mf["ref_freq"].fillna(0.0).to_numpy(dtype=np.float32), width=width, color=COLOR_MUTED, alpha=0.8, label="Reference")
        ax1.bar(x, mf["candidate_freq"].fillna(0.0).to_numpy(dtype=np.float32), width=width, color=COLOR_PRIMARY, alpha=0.8, label="Candidate")
        ax1.bar(x + width, mf["topk_freq"].fillna(0.0).to_numpy(dtype=np.float32), width=width, color=COLOR_ACCENT, alpha=0.9, label="Top-k")
        ax1.set_xticks(x)
        ax1.set_xticklabels(mf["motif"].astype(str), rotation=30, ha="right")
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Frequency")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(loc="best")

        fig.tight_layout()
        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_physics_rule_detail_figure(property_name: str, physics_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or physics_df.empty:
        return
    pf = physics_df.copy().replace([np.inf, -np.inf], np.nan)
    pf["observed_delta_topk_vs_ref"] = pd.to_numeric(pf.get("observed_delta_topk_vs_ref"), errors="coerce")
    pf = pf.dropna(subset=["feature", "observed_delta_topk_vs_ref"])
    if pf.empty:
        return

    pf["abs_delta"] = np.abs(pf["observed_delta_topk_vs_ref"])
    pf = pf.sort_values("abs_delta", ascending=False).copy()

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        ax0, ax1 = axes

        labels = [f"{r['feature']} ({r['source']})" for _, r in pf.iterrows()]
        vals = pf["observed_delta_topk_vs_ref"].to_numpy(dtype=np.float32)
        ok = pf["sign_match"].astype(bool).to_numpy(dtype=bool)
        colors = [COLOR_ACCENT if m else COLOR_SECONDARY for m in ok]
        y = np.arange(len(labels), dtype=np.float32)
        ax0.barh(y, vals, color=colors, alpha=0.9)
        ax0.set_yticks(y)
        ax0.set_yticklabels(labels)
        ax0.axvline(0.0, color="#111111", linewidth=1.0)
        ax0.set_xlabel("Observed delta (top-k vs reference)")
        ax0.grid(axis="x", alpha=0.25)

        n_match = int(np.sum(ok))
        n_total = int(len(ok))
        n_mismatch = max(n_total - n_match, 0)
        ax1.bar(["Match", "Mismatch"], [n_match, n_mismatch], color=[COLOR_ACCENT, COLOR_SECONDARY], alpha=0.9)
        ax1.set_ylabel("Count")
        ax1.grid(axis="y", alpha=0.25)
        if n_total > 0:
            ax1.text(0.5, max(n_match, n_mismatch, 1) * 1.02, f"Consistency: {n_match / n_total:.2f}", ha="center", va="bottom")

        fig.tight_layout()
        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_property_ood_landscape_figure(
    property_name: str,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    target: Optional[float],
    target_mode: str,
    output_base: Path,
) -> None:
    if plt is None or candidate_df.empty:
        return
    cdf = _prepare_candidate_error_df(candidate_df, target=target, target_mode=target_mode)
    use_target_excess = str(target_mode).strip().lower() in {"ge", "le"} and cdf["target_excess"].notna().any()
    y_col = "target_excess" if use_target_excess else "property_error"
    y_label = (
        f"Target excess (≥0 = hit)"
        if use_target_excess
        else "Property error"
    )
    if {"d2_distance", y_col, "smiles"}.issubset(set(cdf.columns)) is False:
        return

    plot_df = cdf.dropna(subset=["d2_distance", y_col]).copy()
    if plot_df.empty:
        return
    if len(plot_df) > 12000:
        plot_df = plot_df.sample(n=12000, random_state=42)
    topk_set = set(topk_df["smiles"].astype(str).tolist()) if "smiles" in topk_df.columns else set()
    plot_df["is_topk"] = plot_df["smiles"].astype(str).isin(topk_set)
    base = plot_df[~plot_df["is_topk"]]
    tk = plot_df[plot_df["is_topk"]]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        ax0, ax1 = axes

        if not base.empty:
            ax0.scatter(base["d2_distance"], base[y_col], s=10, alpha=0.22, color=COLOR_PRIMARY, label="Candidates")
        if not tk.empty:
            ax0.scatter(tk["d2_distance"], tk[y_col], s=26, alpha=0.9, color=COLOR_SECONDARY, label="F6 top-k")
        if use_target_excess:
            ax0.axhline(0.0, color="#111111", linewidth=1.0, linestyle="--")
        ax0.set_xlabel("D2 distance (lower is better)")
        ax0.set_ylabel(y_label)
        ax0.grid(alpha=0.25)
        ax0.legend(loc="best")

        err_cand = _numeric_array(base[y_col] if not base.empty else [])
        err_top = _numeric_array(tk[y_col] if not tk.empty else [])
        if err_cand.size:
            ax1.hist(err_cand, bins=40, color=COLOR_PRIMARY, alpha=0.5, label="Candidates")
        if err_top.size:
            ax1.hist(err_top, bins=25, color=COLOR_SECONDARY, alpha=0.75, label="F6 top-k")
        if err_cand.size or err_top.size:
            ax1.set_xlabel(y_label)
            ax1.set_ylabel("Count")
            ax1.grid(alpha=0.25)
            ax1.legend(loc="best")
        else:
            ax1.text(0.5, 0.5, "No finite error values", ha="center", va="center")
            ax1.set_axis_off()

        fig.tight_layout()
        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_nearest_neighbor_detail_figure(property_name: str, nn_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or nn_df.empty:
        return
    ndf = nn_df.copy()
    ndf["nearest_tanimoto"] = pd.to_numeric(ndf.get("nearest_tanimoto"), errors="coerce")
    ndf["topk_prediction"] = pd.to_numeric(ndf.get("topk_prediction"), errors="coerce")
    ndf["nearest_reference_value"] = pd.to_numeric(ndf.get("nearest_reference_value"), errors="coerce")

    sims = _numeric_array(ndf["nearest_tanimoto"])
    if sims.size == 0:
        return

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        ax0, ax1, ax2 = axes

        ax0.hist(sims, bins=18, color="#6D597A", alpha=0.85)
        ax0.axvline(float(np.mean(sims)), color="#111111", linestyle="--", linewidth=1.1, label="Mean")
        ax0.set_xlabel("Tanimoto")
        ax0.set_ylabel("Count")
        ax0.grid(alpha=0.25)
        ax0.legend(loc="best")

        pair_df = ndf.dropna(subset=["topk_prediction", "nearest_reference_value"]).copy()
        if not pair_df.empty:
            ax1.scatter(pair_df["nearest_reference_value"], pair_df["topk_prediction"], s=24, alpha=0.75, color=COLOR_PRIMARY)
            lo = float(np.nanmin(np.concatenate([pair_df["nearest_reference_value"].to_numpy(dtype=np.float32), pair_df["topk_prediction"].to_numpy(dtype=np.float32)])))
            hi = float(np.nanmax(np.concatenate([pair_df["nearest_reference_value"].to_numpy(dtype=np.float32), pair_df["topk_prediction"].to_numpy(dtype=np.float32)])))
            if np.isfinite(lo) and np.isfinite(hi):
                ax1.plot([lo, hi], [lo, hi], linestyle="--", color=COLOR_MUTED, linewidth=1.1)
            ax1.set_xlabel("Nearest reference value")
            ax1.set_ylabel("Top-k prediction")
            ax1.grid(alpha=0.25)
        else:
            ax1.text(0.5, 0.5, "No paired values", ha="center", va="center")
            ax1.set_axis_off()

        # Show structural/count descriptors only (comparable scales);
        # large-magnitude descriptors (mol_wt, tpsa, logp) displayed as text annotations.
        bar_cols = [
            ("delta_aromatic_ring_count", "Δ aromatic rings"),
            ("delta_ring_count", "Δ ring count"),
            ("delta_rotatable_bonds", "Δ rotatable bonds"),
            ("delta_fraction_csp3", "Δ frac CSP3"),
        ]
        note_cols = [
            ("delta_mol_wt", "Δ mol wt"),
            ("delta_tpsa", "Δ TPSA"),
            ("delta_logp", "Δ logP"),
        ]
        labels, vals = [], []
        for col, label in bar_cols:
            if col in ndf.columns:
                arr = _numeric_array(ndf[col])
                if arr.size:
                    labels.append(label)
                    vals.append(float(np.mean(arr)))
        note_lines = []
        for col, label in note_cols:
            if col in ndf.columns:
                arr = _numeric_array(ndf[col])
                if arr.size:
                    note_lines.append(f"{label}: {float(np.mean(arr)):+.2f}")
        if vals:
            colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in vals]
            y_pos = np.arange(len(labels), dtype=np.float32)
            ax2.barh(y_pos, vals, color=colors, alpha=0.85)
            ax2.axvline(0.0, color="#111111", linewidth=1.0)
            # No y-ticks — embed label + value as white text inside each bar
            ax2.set_yticks([])
            ax2.set_xlabel("Mean Δ (top-k vs nearest reference)", fontsize=9)
            ax2.set_title("Descriptor shifts\n(top-k vs nearest ref)", fontsize=10)
            ax2.grid(axis="x", alpha=0.25)
            x_range = max(abs(v) for v in vals) if vals else 1.0
            for y_pos_i, v, label in zip(y_pos, vals, labels):
                # Embed combined "label: value" just inside the zero end of the bar
                inner_x = -0.03 * x_range if v < 0 else 0.03 * x_range
                ha_inner = "right" if v < 0 else "left"
                ax2.text(inner_x, y_pos_i, f"{label}: {v:+.2f}",
                         va="center", ha=ha_inner, fontsize=8,
                         color="white", fontweight="bold")
            # Show large-magnitude descriptors as a text block below the plot
            if note_lines:
                note_text = "  ".join(note_lines)
                ax2.text(0.5, -0.18, note_text, transform=ax2.transAxes,
                         ha="center", va="top", fontsize=8, color="#555555",
                         style="italic")
        else:
            ax2.text(0.5, 0.5, "No descriptor delta columns", ha="center", va="center")
            ax2.set_axis_off()

        fig.tight_layout()
        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_property_figure_suite(
    *,
    property_name: str,
    descriptor_shift_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    physics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    target: Optional[float],
    target_mode: str,
    figures_dir: Path,
) -> None:
    _plot_property_figure(
        property_name=property_name,
        descriptor_shift_df=descriptor_shift_df,
        motif_df=motif_df,
        candidate_df=candidate_df,
        topk_df=topk_df,
        nn_df=nn_df,
        target=target,
        target_mode=target_mode,
        output_base=figures_dir / f"figure_f7_chem_physics_{property_name}",
    )
    _plot_descriptor_detail_figure(
        property_name=property_name,
        descriptor_shift_df=descriptor_shift_df,
        output_base=figures_dir / f"figure_f7_descriptors_{property_name}",
    )
    _plot_motif_detail_figure(
        property_name=property_name,
        motif_df=motif_df,
        output_base=figures_dir / f"figure_f7_motif_enrichment_{property_name}",
    )
    _plot_physics_rule_detail_figure(
        property_name=property_name,
        physics_df=physics_df,
        output_base=figures_dir / f"figure_f7_physics_rules_{property_name}",
    )
    _plot_property_ood_landscape_figure(
        property_name=property_name,
        candidate_df=candidate_df,
        topk_df=topk_df,
        target=target,
        target_mode=target_mode,
        output_base=figures_dir / f"figure_f7_property_ood_landscape_{property_name}",
    )
    _plot_nearest_neighbor_detail_figure(
        property_name=property_name,
        nn_df=nn_df,
        output_base=figures_dir / f"figure_f7_nearest_neighbor_{property_name}",
    )


def _plot_property_figure(
    property_name: str,
    descriptor_shift_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    target: Optional[float],
    target_mode: str,
    output_base: Path,
) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel A: descriptor effect sizes (top-k vs reference)
    ds = descriptor_shift_df.copy()
    ds = ds.replace([np.inf, -np.inf], np.nan).dropna(subset=["cohens_d_topk_vs_ref"])
    if not ds.empty:
        ds = ds.assign(abs_d=np.abs(ds["cohens_d_topk_vs_ref"]))
        ds = ds.sort_values("abs_d", ascending=False).head(10).iloc[::-1]
        ax1.barh(ds["descriptor"], ds["cohens_d_topk_vs_ref"], color="#2C7FB8")
        ax1.axvline(0.0, color="black", linewidth=1)
        ax1.set_xlabel("Effect size")
    else:
        ax1.text(0.5, 0.5, "No descriptor shift data", ha="center", va="center")
        ax1.set_axis_off()

    # Panel B: motif enrichment
    mf = motif_df.copy()
    mf = mf.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2_enrichment_topk_vs_ref"])
    if not mf.empty:
        mf = mf.assign(abs_log2=np.abs(mf["log2_enrichment_topk_vs_ref"]))
        mf = mf.sort_values("abs_log2", ascending=False).head(10).iloc[::-1]
        ax2.barh(mf["motif"], mf["log2_enrichment_topk_vs_ref"], color="#41AB5D")
        ax2.axvline(0.0, color="black", linewidth=1)
        ax2.set_xlabel("log2 enrichment")
    else:
        ax2.text(0.5, 0.5, "No motif data", ha="center", va="center")
        ax2.set_axis_off()

    # Panel C: property error vs OOD distance
    cdf = _prepare_candidate_error_df(candidate_df, target=target, target_mode=target_mode)
    use_target_excess = str(target_mode).strip().lower() in {"ge", "le"} and cdf["target_excess"].notna().any()
    y_col = "target_excess" if use_target_excess else "property_error"
    y_label = (
        f"{_target_excess_axis_label(property_name, target_mode)} (>=0 is hit)"
        if use_target_excess
        else "Property error/proxy"
    )

    if {"d2_distance", y_col, "smiles"}.issubset(set(cdf.columns)):
        plot_df = cdf.dropna(subset=["d2_distance", y_col]).copy()
        if not plot_df.empty:
            topk_set = set(topk_df["smiles"].astype(str).tolist()) if "smiles" in topk_df.columns else set()
            plot_df["is_topk"] = plot_df["smiles"].astype(str).isin(topk_set)
            base = plot_df[~plot_df["is_topk"]]
            tk = plot_df[plot_df["is_topk"]]
            if not base.empty:
                ax3.scatter(base["d2_distance"], base[y_col], s=12, alpha=0.35, label="Candidates", color="#9ECAE1")
            if not tk.empty:
                ax3.scatter(tk["d2_distance"], tk[y_col], s=24, alpha=0.9, label="F6 top-k", color="#D94801")
            if use_target_excess:
                ax3.axhline(0.0, color="#111111", linewidth=1.0, linestyle="--")
            ax3.set_xlabel("D2 distance (lower is closer)")
            ax3.set_ylabel(y_label)
            ax3.legend(frameon=False, loc="best", fontsize=15)
        else:
            ax3.text(0.5, 0.5, "No valid scatter points", ha="center", va="center")
            ax3.set_axis_off()
    else:
        ax3.text(0.5, 0.5, "Need prediction + d2_distance", ha="center", va="center")
        ax3.set_axis_off()

    # Panel D: nearest-neighbor similarity
    if not nn_df.empty and "nearest_tanimoto" in nn_df.columns:
        vals = pd.to_numeric(nn_df["nearest_tanimoto"], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if vals.size:
            ax4.hist(vals, bins=15, color="#756BB1", alpha=0.85)
            ax4.set_xlabel("Tanimoto similarity")
            ax4.set_ylabel("Count")
        else:
            ax4.text(0.5, 0.5, "No NN similarity values", ha="center", va="center")
            ax4.set_axis_off()
    else:
        ax4.text(0.5, 0.5, "No NN explanation data", ha="center", va="center")
        ax4.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure_png(fig, output_base)
    plt.close(fig)


def _plot_summary_figure(metrics_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        df = metrics_df.copy().sort_values("property")
        props = df["property"].astype(str).tolist()

        axes[0].bar(props, pd.to_numeric(df["descriptor_consistency_rate"], errors="coerce").fillna(0.0), color=COLOR_PRIMARY)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Rate")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(props, pd.to_numeric(df["topk_hit_rate"], errors="coerce").fillna(0.0), color=COLOR_ACCENT)
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", alpha=0.25)

        axes[2].bar(props, pd.to_numeric(df["mean_nn_similarity"], errors="coerce").fillna(0.0), color="#6D597A")
        axes[2].set_ylim(0, 1)
        axes[2].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        _save_figure_png(fig, output_base)
        plt.close(fig)


def main(args):
    if Chem is None:
        raise RuntimeError("RDKit is required for step7_chem_physics_analysis.py")

    config = load_config(args.config)
    cfg_step7 = config.get("chem_physics_analysis", {}) or {}
    cfg_f5 = config.get("foundation_inverse", {}) or {}

    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step7_chem_physics_analysis")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    properties = _parse_properties(args, cfg_step7)
    property_dir = _resolve_path(config["paths"]["property_dir"])

    candidate_template = args.candidate_scores_template
    if candidate_template is None:
        candidate_template = str(cfg_step7.get("candidate_scores_template", "")).strip() or None
    topk_template = args.topk_scores_template
    if topk_template is None:
        topk_template = str(cfg_step7.get("topk_scores_template", "")).strip() or None

    max_ref_samples = _to_int_or_none(args.max_reference_samples)
    if max_ref_samples is None:
        max_ref_samples = _to_int_or_none(cfg_step7.get("max_reference_samples"))

    top_k_override = _to_int_or_none(args.top_k)
    if top_k_override is None:
        top_k_override = _to_int_or_none(cfg_step7.get("top_k"))
    if top_k_override is None:
        top_k_override = 100

    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(cfg_step7.get("generate_figures", True), True)
    generate_figures = bool(generate_figures and plt is not None)

    skip_missing = args.skip_missing_property
    if skip_missing is None:
        skip_missing = True
    skip_missing = bool(skip_missing)

    reference_class_override = str(args.reference_class or "").strip().lower() or None
    reference_class_filter_enabled = args.reference_class_filter_enabled
    if reference_class_filter_enabled is None:
        reference_class_filter_enabled = _to_bool(cfg_step7.get("reference_class_filter_enabled", True), True)
    reference_class_filter_enabled = bool(reference_class_filter_enabled)

    sa_score_fn = _load_sa_score_fn()
    compiled_motifs = _compile_motifs()

    all_descriptor_shift = []
    all_motif = []
    all_physics = []
    all_nn = []
    metric_rows = []
    file_rows = []
    skipped = []
    multi_property_run = len(properties) > 1

    for prop in properties:
        prop_step_dirs = ensure_step_dirs(results_dir, "step7_chem_physics_analysis", prop)
        save_config(config, prop_step_dirs["files_dir"] / "config_used.yaml")
        try:
            ref_df = _load_property_reference(property_dir, prop, max_samples=max_ref_samples)
        except Exception as exc:
            msg = f"{prop}: failed to load reference property data ({exc})"
            if skip_missing:
                skipped.append(msg)
                continue
            raise

        cand_path, topk_path = _resolve_property_file_inputs(
            property_name=prop,
            results_dir=results_dir,
            candidate_template=candidate_template,
            topk_template=topk_template,
        )
        file_rows.append(
            {
                "property": prop,
                "candidate_scores_path": str(cand_path) if cand_path else "",
                "topk_scores_path": str(topk_path) if topk_path else "",
            }
        )
        save_csv(
            pd.DataFrame(
                [
                    {
                        "property": prop,
                        "candidate_scores_path": str(cand_path) if cand_path else "",
                        "topk_scores_path": str(topk_path) if topk_path else "",
                    }
                ]
            ),
            prop_step_dirs["files_dir"] / "property_input_files.csv",
            index=False,
        )

        if topk_path is None:
            msg = f"{prop}: top-k file not found (set step6 output or topk_scores_template)."
            if skip_missing:
                skipped.append(msg)
                continue
            raise FileNotFoundError(msg)

        topk_df = pd.read_csv(topk_path)
        if topk_df.empty:
            msg = f"{prop}: top-k file is empty: {topk_path}"
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)
        if multi_property_run and _is_generic_scores_file(topk_path) and "property" not in topk_df.columns:
            msg = (
                f"{prop}: only generic top-k file found ({topk_path}) without a property column. "
                "Use property-specific file naming (ood_objective_topk_<PROPERTY>.csv)."
            )
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)

        topk_df, topk_property_msg = _filter_property_rows_if_available(
            topk_df,
            property_name=prop,
            source_path=topk_path,
        )
        if topk_property_msg is not None:
            if skip_missing:
                skipped.append(topk_property_msg)
                continue
            raise RuntimeError(topk_property_msg)
        if topk_df.empty:
            msg = f"{prop}: top-k file has no usable rows after property filtering: {topk_path}"
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)
        if "smiles" not in topk_df.columns:
            raise ValueError(f"{prop}: top-k file must contain 'smiles': {topk_path}")

        if top_k_override is not None and top_k_override > 0 and len(topk_df) > top_k_override:
            if "ood_aware_rank" in topk_df.columns:
                topk_df = topk_df.sort_values("ood_aware_rank").head(int(top_k_override)).copy()
            else:
                topk_df = topk_df.head(int(top_k_override)).copy()

        if cand_path is not None and cand_path.exists():
            candidate_df = pd.read_csv(cand_path)
            if multi_property_run and _is_generic_scores_file(cand_path) and "property" not in candidate_df.columns:
                skipped.append(
                    f"{prop}: candidate file {cand_path} is generic without property column; using top-k rows instead."
                )
                candidate_df = topk_df.copy()
            else:
                candidate_df, cand_property_msg = _filter_property_rows_if_available(
                    candidate_df,
                    property_name=prop,
                    source_path=cand_path,
                )
                if cand_property_msg is not None:
                    skipped.append(f"{cand_property_msg} Falling back to top-k rows for candidate analysis.")
                    candidate_df = topk_df.copy()
                elif candidate_df.empty:
                    skipped.append(f"{prop}: candidate file has no rows after property filtering; using top-k rows.")
                    candidate_df = topk_df.copy()
        else:
            candidate_df = topk_df.copy()

        target, target_mode, epsilon = _resolve_target_config(prop, cfg_step7, cfg_f5)

        resolved_reference_class = _resolve_reference_class(
            property_name=prop,
            cfg_step7=cfg_step7,
            cfg_f5=cfg_f5,
            candidate_df=candidate_df,
            topk_df=topk_df,
            override_class=reference_class_override,
        )

        ref_desc = _descriptor_and_motif_rows(
            ref_df["smiles"].astype(str).tolist(),
            property_name=prop,
            split="reference",
            sa_score_fn=sa_score_fn,
            compiled_motifs=compiled_motifs,
        )
        ref_df_used = ref_df
        ref_desc_used = ref_desc
        if reference_class_filter_enabled and resolved_reference_class:
            ref_df_used, ref_desc_used, ref_filter_msg = _apply_reference_class_filter(
                property_name=prop,
                ref_df=ref_df,
                ref_desc=ref_desc,
                reference_class=resolved_reference_class,
            )
            if ref_filter_msg:
                skipped.append(ref_filter_msg)
            else:
                print(
                    f"[F7] Applied reference class filter '{resolved_reference_class}' "
                    f"for {prop}: {len(ref_df_used)}/{len(ref_df)} reference rows kept."
                )
        cand_desc = _descriptor_and_motif_rows(
            candidate_df["smiles"].astype(str).tolist(),
            property_name=prop,
            split="candidate",
            sa_score_fn=sa_score_fn,
            compiled_motifs=compiled_motifs,
        )
        top_desc = _descriptor_and_motif_rows(
            topk_df["smiles"].astype(str).tolist(),
            property_name=prop,
            split="topk",
            sa_score_fn=sa_score_fn,
            compiled_motifs=compiled_motifs,
        )

        if ref_desc_used.empty or top_desc.empty:
            msg = f"{prop}: insufficient descriptor rows after RDKit parsing."
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)

        shift_df = _descriptor_shift_table(prop, ref_desc_used, cand_desc, top_desc)
        motif_df = _motif_enrichment_table(prop, ref_desc_used, cand_desc, top_desc)
        physics_df = _physics_consistency_table(prop, shift_df, motif_df)
        nn_df = _nearest_neighbor_explanations(prop, ref_df_used, topk_df, ref_desc_used, top_desc)

        all_descriptor_shift.append(shift_df)
        all_motif.append(motif_df)
        all_physics.append(physics_df)
        if not nn_df.empty:
            all_nn.append(nn_df)

        save_csv(shift_df, prop_step_dirs["files_dir"] / "descriptor_shifts.csv", index=False)
        save_csv(motif_df, prop_step_dirs["files_dir"] / "motif_enrichment.csv", index=False)
        save_csv(physics_df, prop_step_dirs["files_dir"] / "physics_consistency.csv", index=False)
        save_csv(nn_df, prop_step_dirs["files_dir"] / "nearest_neighbor_explanations.csv", index=False)

        if "property_hit" in topk_df.columns:
            hit_vals = pd.to_numeric(topk_df["property_hit"], errors="coerce")
            topk_hit_rate = float(np.nanmean(hit_vals)) if hit_vals.notna().any() else np.nan
        elif "prediction" in topk_df.columns and target is not None:
            pred_vals = pd.to_numeric(topk_df["prediction"], errors="coerce").to_numpy(dtype=np.float32)
            if epsilon is None:
                epsilon = 0.0
            if target_mode == "window":
                hits = np.abs(pred_vals - float(target)) <= float(epsilon)
            elif target_mode == "ge":
                hits = pred_vals >= float(target)
            elif target_mode == "le":
                hits = pred_vals <= float(target)
            else:
                hits = np.abs(pred_vals - float(target)) <= float(epsilon)
            topk_hit_rate = float(np.mean(hits)) if hits.size else np.nan
        else:
            topk_hit_rate = np.nan

        physics_valid = physics_df["sign_match"].astype(bool)
        consistency_rate = float(physics_valid.mean()) if len(physics_valid) else np.nan
        mean_nn = float(pd.to_numeric(nn_df.get("nearest_tanimoto", pd.Series(dtype=float)), errors="coerce").mean()) if not nn_df.empty else np.nan
        mean_d2 = float(pd.to_numeric(topk_df.get("d2_distance", pd.Series(dtype=float)), errors="coerce").mean()) if "d2_distance" in topk_df.columns else np.nan
        unc_series = _prediction_uncertainty_series(topk_df, prop)
        mean_unc = float(unc_series.mean()) if unc_series.notna().any() else np.nan
        if "conservative_objective" in topk_df.columns:
            mean_obj = float(pd.to_numeric(topk_df["conservative_objective"], errors="coerce").mean())
        else:
            mean_obj = float(pd.to_numeric(topk_df.get("ood_aware_objective", pd.Series(dtype=float)), errors="coerce").mean()) if "ood_aware_objective" in topk_df.columns else np.nan
        mean_constraint = float(pd.to_numeric(topk_df.get("constraint_violation_total", pd.Series(dtype=float)), errors="coerce").mean()) if "constraint_violation_total" in topk_df.columns else np.nan
        metric_row = {
            "method": "Multi_View_Foundation",
            "property": prop,
            "n_reference": int(len(ref_df_used)),
            "n_candidates": int(len(candidate_df)),
            "n_topk": int(len(topk_df)),
            "reference_class_filter_enabled": bool(reference_class_filter_enabled and bool(resolved_reference_class)),
            "reference_class_filter": resolved_reference_class if resolved_reference_class else "",
            "descriptor_consistency_rate": round(consistency_rate, 4) if np.isfinite(consistency_rate) else np.nan,
            "mean_nn_similarity": round(mean_nn, 4) if np.isfinite(mean_nn) else np.nan,
            "topk_hit_rate": round(topk_hit_rate, 4) if np.isfinite(topk_hit_rate) else np.nan,
            "topk_mean_d2_distance": round(mean_d2, 6) if np.isfinite(mean_d2) else np.nan,
            "topk_mean_prediction_uncertainty": round(mean_unc, 6) if np.isfinite(mean_unc) else np.nan,
            "topk_mean_conservative_objective": round(mean_obj, 6) if np.isfinite(mean_obj) else np.nan,
            "topk_mean_constraint_violation": round(mean_constraint, 6) if np.isfinite(mean_constraint) else np.nan,
            "target_value": target,
            "target_mode": target_mode,
            "epsilon": epsilon,
        }
        metric_rows.append(metric_row)
        save_csv(pd.DataFrame([metric_row]), prop_step_dirs["metrics_dir"] / "metrics_chem_physics.csv", index=False)

        if generate_figures:
            _plot_property_figure_suite(
                property_name=prop,
                descriptor_shift_df=shift_df,
                motif_df=motif_df,
                physics_df=physics_df,
                candidate_df=candidate_df,
                topk_df=topk_df,
                nn_df=nn_df,
                target=target,
                target_mode=target_mode,
                figures_dir=prop_step_dirs["figures_dir"],
            )

        save_json(
            {
                "property": prop,
                "candidate_scores_path": str(cand_path) if cand_path else "",
                "topk_scores_path": str(topk_path) if topk_path else "",
                "reference_class_filter_enabled": bool(reference_class_filter_enabled),
                "reference_class_filter": resolved_reference_class if resolved_reference_class else "",
                "target_value": target,
                "target_mode": target_mode,
                "epsilon": epsilon,
                "generate_figures": bool(generate_figures),
            },
            prop_step_dirs["files_dir"] / "run_meta.json",
        )

    if not metric_rows:
        raise RuntimeError(
            "Step7 found no analyzable properties. "
            "Provide valid F5/F6 outputs per property or set templates."
        )

    descriptor_shift_all = pd.concat(all_descriptor_shift, ignore_index=True) if all_descriptor_shift else pd.DataFrame()
    motif_all = pd.concat(all_motif, ignore_index=True) if all_motif else pd.DataFrame()
    physics_all = pd.concat(all_physics, ignore_index=True) if all_physics else pd.DataFrame()
    nn_all = pd.concat(all_nn, ignore_index=True) if all_nn else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    files_df = pd.DataFrame(file_rows)

    save_csv(
        descriptor_shift_all,
        step_dirs["files_dir"] / "descriptor_shifts.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "descriptor_shifts.csv"],
        index=False,
    )
    save_csv(
        motif_all,
        step_dirs["files_dir"] / "motif_enrichment.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "motif_enrichment.csv"],
        index=False,
    )
    save_csv(
        physics_all,
        step_dirs["files_dir"] / "physics_consistency.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "physics_consistency.csv"],
        index=False,
    )
    save_csv(
        nn_all,
        step_dirs["files_dir"] / "nearest_neighbor_explanations.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "nearest_neighbor_explanations.csv"],
        index=False,
    )
    save_csv(
        files_df,
        step_dirs["files_dir"] / "property_input_files.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "property_input_files.csv"],
        index=False,
    )
    save_csv(
        metrics_df,
        step_dirs["metrics_dir"] / "metrics_chem_physics.csv",
        legacy_paths=[results_dir / "metrics_chem_physics.csv"],
        index=False,
    )

    if generate_figures:
        _plot_summary_figure(
            metrics_df=metrics_df,
            output_base=step_dirs["figures_dir"] / "figure_f7_chem_physics_all_properties",
        )

    save_json(
        {
            "properties_requested": properties,
            "properties_processed": metrics_df["property"].astype(str).tolist(),
            "skipped": skipped,
            "candidate_scores_template": candidate_template,
            "topk_scores_template": topk_template,
            "reference_class_filter_enabled": bool(reference_class_filter_enabled),
            "reference_class_override": reference_class_override or "",
            "generate_figures": bool(generate_figures),
        },
        step_dirs["files_dir"] / "run_meta.json",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "run_meta.json"],
    )

    if skipped:
        print("Step7 warnings:")
        for msg in skipped:
            print(f"  - {msg}")
    print(f"Saved metrics_chem_physics.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--properties", type=str, default=None, help="Comma-separated property list, e.g. Tg,Tm,Td,Eg")
    parser.add_argument("--candidate_scores_template", type=str, default=None, help="Path template for F5 candidate CSV; supports {property}")
    parser.add_argument("--topk_scores_template", type=str, default=None, help="Path template for F6 top-k CSV; supports {property}")
    parser.add_argument("--max_reference_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--reference_class", type=str, default=None, help="Reference class filter (e.g., polyamide)")
    parser.add_argument("--enable_reference_class_filter", dest="reference_class_filter_enabled", action="store_true")
    parser.add_argument("--disable_reference_class_filter", dest="reference_class_filter_enabled", action="store_false")
    parser.set_defaults(reference_class_filter_enabled=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    parser.add_argument("--skip_missing_property", dest="skip_missing_property", action="store_true")
    parser.add_argument("--strict", dest="skip_missing_property", action="store_false")
    parser.set_defaults(skip_missing_property=None)
    main(parser.parse_args())
