"""
baseline_b4_random_sampling.py

B4 baseline: unconditional SELFIES diffusion sampling with no property guidance.
Generates polymers, applies validity/class/SA filters (identical to step5),
then scores with the single SELFIES property model.

Purpose: establishes the hit rate a naive sampler achieves before any
multi-view committee or OOD-aware ranking is applied.

Outputs (to --output_dir):
  metrics_b4_<property>.csv    — summary metrics row (matches step5 columns)
  candidate_scores_b4_<property>.csv — per-candidate scores
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR.parent))   # for shared.*

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _check_rdkit():
    try:
        from rdkit import Chem
        return Chem
    except ImportError:
        raise RuntimeError("RDKit is required: pip install rdkit")


def _is_valid_smiles(smi: str, Chem) -> bool:
    if not smi or not isinstance(smi, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except Exception:
        return False


def _count_stars(smi: str) -> int:
    return smi.count("*")


def _check_polyamide(smi: str, Chem) -> bool:
    """Simple amide-bond substructure check for polyamide class matching."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        amide = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
        return mol.HasSubstructMatch(amide)
    except Exception:
        return False


def _sa_score(smi: str) -> Optional[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem.rdchem import RWMol
        # Try sascorer from RDKit contrib
        from rdkit.Contrib.SA_Score import sascorer
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return sascorer.calculateScore(mol)
    except Exception:
        pass
    try:
        # Fallback: load from Bi_Diffusion_SMILES utils
        chem_path = REPO_DIR.parent / "Bi_Diffusion_SMILES" / "src" / "utils" / "chemistry.py"
        if chem_path.exists():
            chem_mod = _load_module("b4_chem_smiles", chem_path)
            if hasattr(chem_mod, "compute_sa_score"):
                return chem_mod.compute_sa_score(smi)
    except Exception:
        pass
    return None


# ── SELFIES → p-SMILES conversion ────────────────────────────────────────────
def _load_selfies_converter(config: dict):
    """Load the selfies→polymer-SMILES converter used in step5."""
    try:
        data_mod = _load_module(
            "b4_selfies_data",
            REPO_DIR / "src" / "data" / "view_converters.py",
        )
        if hasattr(data_mod, "selfies_to_psmiles"):
            return data_mod.selfies_to_psmiles
    except Exception:
        pass

    # Fallback: use selfies library directly
    try:
        import selfies as sf

        def converter(selfies_str: str) -> Optional[str]:
            try:
                smi = sf.decoder(selfies_str)
                return smi if smi else None
            except Exception:
                return None

        return converter
    except ImportError:
        raise RuntimeError("Could not load SELFIES→SMILES converter. Install selfies: pip install selfies")


# ── SELFIES sampler loader ────────────────────────────────────────────────────
def _load_selfies_sampler(config: dict, device: str) -> dict:
    """Load the trained SELFIES diffusion backbone and build a ConstrainedSampler."""
    enc_cfg = config.get("selfies_encoder", {})
    method_dir = _resolve(enc_cfg.get("method_dir", REPO_DIR.parent / "Bi_Diffusion_SELFIES"))

    cfg_path = enc_cfg.get("config_path")
    cfg_path = _resolve(cfg_path) if cfg_path else method_dir / "configs" / "config.yaml"
    method_cfg = _load_config(cfg_path)

    scales_mod = _load_module("b4_scales_selfies", method_dir / "src" / "utils" / "model_scales.py")
    tok_mod    = _load_module("b4_tok_selfies",    method_dir / "src" / "data" / "selfies_tokenizer.py")
    backbone_mod = _load_module("b4_backbone_selfies", method_dir / "src" / "model" / "backbone.py")
    sampler_mod  = _load_module("b4_sampler_selfies",  method_dir / "src" / "sampling" / "sampler.py")
    diffusion_mod = _load_module("b4_diffusion_selfies", method_dir / "src" / "model" / "diffusion.py")

    model_size = enc_cfg.get("model_size", "small")
    backbone_config = scales_mod.get_model_config(model_size, method_cfg, model_type="sequence")

    base_results_dir = enc_cfg.get("results_dir")
    if base_results_dir:
        base_results_dir = _resolve(base_results_dir)
    else:
        base_results_dir = _resolve(method_cfg["paths"]["results_dir"]) \
            if not Path(method_cfg["paths"]["results_dir"]).is_absolute() \
            else Path(method_cfg["paths"]["results_dir"])
        if not base_results_dir.is_absolute():
            base_results_dir = method_dir / base_results_dir

    results_dir = Path(scales_mod.get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = enc_cfg.get("tokenizer_path")
    tokenizer_path = _resolve(tokenizer_path) if tokenizer_path \
        else results_dir / "tokenizer.pkl"
    if not tokenizer_path.exists():
        tokenizer_path = base_results_dir / "tokenizer.pkl"

    ckpt_path = enc_cfg.get("checkpoint_path")
    if ckpt_path:
        ckpt_path = _resolve(ckpt_path)
    else:
        step_dir  = enc_cfg.get("step_dir", "step1_backbone")
        ckpt_name = enc_cfg.get("checkpoint_name", "backbone_best.pt")
        ckpt_path = results_dir / step_dir / "checkpoints" / ckpt_name

    print(f"[B4] Tokenizer: {tokenizer_path}")
    print(f"[B4] Checkpoint: {ckpt_path}")

    tokenizer = tok_mod.SelfiesTokenizer.load(str(tokenizer_path))

    diffusion_steps = method_cfg.get("diffusion", {}).get("num_steps", 50)
    backbone = backbone_mod.DiffusionBackbone(
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    backbone_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    if not backbone_state:
        backbone_state = state
    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(device).eval()

    # Wrap in a minimal diffusion model shell the sampler expects
    class _DiffusionShell:
        def __init__(self, bb):
            self.backbone = bb
            self.eval = bb.eval

    diffusion_model = _DiffusionShell(backbone)

    seq_length = backbone_config.get("max_position_embeddings", 256)

    sampler = sampler_mod.ConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=diffusion_steps,
        temperature=1.0,
        use_constraints=True,
        device=device,
    )

    return {"sampler": sampler, "seq_length": seq_length, "tokenizer": tokenizer}


# ── Property model loader ─────────────────────────────────────────────────────
def _get_step5():
    """Load step5 module once and cache it."""
    if "b4_step5" not in sys.modules:
        _load_module("b4_step5", REPO_DIR / "scripts" / "step5_foundation_inverse.py")
    return sys.modules["b4_step5"]


def _load_property_model(model_path: str | Path):
    """Load a step3 MLP property model."""
    return _get_step5()._load_property_model(_resolve(model_path))


def _score_smiles(smiles_list: List[str], model, encoder_assets: dict, device: str) -> np.ndarray:
    """Encode SMILES with the SELFIES backbone and run through the property MLP."""
    # _embed_sequence(inputs, assets, device) → np.ndarray [N, D]
    embeddings = _get_step5()._embed_sequence(smiles_list, encoder_assets, device)
    if embeddings is None or len(embeddings) == 0:
        return np.full(len(smiles_list), np.nan)
    preds = np.asarray(model.predict(embeddings), dtype=np.float32).reshape(-1)
    return preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="B4 random sampling baseline")
    parser.add_argument("--config",               type=str, required=True)
    parser.add_argument("--results_dir",          type=str, required=True,
                        help="Base results dir (for property model discovery).")
    parser.add_argument("--output_dir",           type=str, required=True)
    parser.add_argument("--property",             type=str, required=True)
    parser.add_argument("--target",               type=float, required=True)
    parser.add_argument("--target_mode",          type=str, default="ge",
                        choices=["window", "ge", "le"])
    parser.add_argument("--epsilon",              type=float, default=20.0)
    parser.add_argument("--target_class",         type=str, default="polyamide")
    parser.add_argument("--max_sa",               type=float, default=4.5)
    parser.add_argument("--sampling_num_per_batch", type=int, default=512)
    parser.add_argument("--sampling_batch_size",  type=int, default=128)
    parser.add_argument("--sampling_max_batches", type=int, default=200)
    parser.add_argument("--device",               type=str, default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"[B4] Device: {device}")

    config      = _load_config(args.config)
    results_dir = _resolve(args.results_dir)
    output_dir  = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Chem = _check_rdkit()

    # Load SELFIES sampler
    print("[B4] Loading SELFIES sampler...")
    sampler_assets = _load_selfies_sampler(config, device)
    sampler    = sampler_assets["sampler"]
    seq_length = sampler_assets["seq_length"]

    # Load SELFIES→pSMILES converter
    print("[B4] Loading SELFIES converter...")
    selfies_to_psmiles = _load_selfies_converter(config)

    # Load SELFIES property model
    prop_model_path = results_dir / "step3_property" / "files" / f"{args.property}_selfies_mlp.pt"
    if not prop_model_path.exists():
        prop_model_path = results_dir / "step3_property" / f"{args.property}_selfies_mlp.pt"
    print(f"[B4] Property model: {prop_model_path}")
    prop_model = _load_property_model(prop_model_path, device)

    # Load encoder assets for scoring (reuse step5 helpers)
    step5 = _get_step5()
    enc_cfg = config.get("selfies_encoder", {})
    encoder_assets = step5._load_sequence_backbone(enc_cfg, device)

    # ── Sampling loop ─────────────────────────────────────────────────────────
    n_generated     = 0
    n_valid         = 0
    n_two_star      = 0
    n_class_pass    = 0
    n_sa_pass       = 0
    n_scored        = 0

    candidates = []  # list of dicts
    t_start = time.time()

    print(f"[B4] Sampling {args.sampling_max_batches} batches × {args.sampling_num_per_batch} sequences...")

    for batch_idx in range(args.sampling_max_batches):
        _, selfies_list = sampler.sample_batch(
            num_samples=args.sampling_num_per_batch,
            seq_length=seq_length,
            batch_size=args.sampling_batch_size,
            show_progress=False,
        )
        n_generated += len(selfies_list)

        # Convert SELFIES → pSMILES
        batch_smiles = []
        for sf_str in selfies_list:
            try:
                psmiles = selfies_to_psmiles(sf_str) if selfies_to_psmiles else None
            except Exception:
                psmiles = None
            if psmiles and isinstance(psmiles, str) and psmiles.strip():
                batch_smiles.append(psmiles.strip())

        # Filter: validity
        valid_smiles = [s for s in batch_smiles if _is_valid_smiles(s, Chem)]
        n_valid += len(valid_smiles)

        # Filter: exactly 2 stars (polymer repeat unit)
        two_star = [s for s in valid_smiles if _count_stars(s) == 2]
        n_two_star += len(two_star)

        # Filter: polyamide class
        if args.target_class and args.target_class.lower() == "polyamide":
            class_pass = [s for s in two_star if _check_polyamide(s, Chem)]
        else:
            class_pass = two_star
        n_class_pass += len(class_pass)

        # Filter: SA score
        sa_pass = []
        for s in class_pass:
            sa = _sa_score(s)
            if sa is None or sa <= args.max_sa:
                sa_pass.append((s, sa))
        n_sa_pass += len(sa_pass)

        if not sa_pass:
            if (batch_idx + 1) % 20 == 0:
                print(f"[B4] batch {batch_idx+1}/{args.sampling_max_batches} "
                      f"gen={n_generated} valid={n_valid} class={n_class_pass} sa={n_sa_pass}")
            continue

        # Score with property model
        smiles_to_score = [s for s, _ in sa_pass]
        sa_map = {s: sa for s, sa in sa_pass}

        preds = _score_smiles(smiles_to_score, prop_model, encoder_assets, device)

        for smi, pred in zip(smiles_to_score, preds):
            if np.isnan(pred):
                continue
            n_scored += 1
            # Hit check (raw threshold — no epsilon, this is the *unconditional* baseline)
            is_hit = bool(
                (args.target_mode == "ge"     and pred >= args.target) or
                (args.target_mode == "le"     and pred <= args.target) or
                (args.target_mode == "window" and abs(pred - args.target) <= args.epsilon)
            )
            candidates.append({
                "smiles":       smi,
                "sa_score":     sa_map.get(smi),
                "batch_idx":    batch_idx,
                "prediction":   float(pred),
                "property_hit": is_hit,
                "target_excess": float(pred - args.target) if args.target_mode == "ge"
                                 else float(args.target - pred) if args.target_mode == "le"
                                 else float(abs(pred - args.target)),
            })

        if (batch_idx + 1) % 20 == 0:
            print(f"[B4] batch {batch_idx+1}/{args.sampling_max_batches} "
                  f"gen={n_generated} valid={n_valid} class={n_class_pass} "
                  f"sa={n_sa_pass} scored={n_scored} hits={sum(c['property_hit'] for c in candidates)}")

    t_elapsed = time.time() - t_start
    n_hits = sum(c["property_hit"] for c in candidates)

    # ── Metrics ───────────────────────────────────────────────────────────────
    hit_rate          = n_hits / n_scored       if n_scored > 0 else 0.0
    validity          = n_valid / n_generated   if n_generated > 0 else 0.0
    class_pass_rate   = n_class_pass / n_valid  if n_valid > 0 else 0.0
    sa_pass_rate      = n_sa_pass / n_class_pass if n_class_pass > 0 else 0.0
    mean_pred         = float(np.mean([c["prediction"] for c in candidates])) if candidates else float("nan")
    mean_pred_hits    = float(np.mean([c["prediction"] for c in candidates if c["property_hit"]])) \
                        if n_hits > 0 else float("nan")
    mean_target_excess = float(np.mean([c["target_excess"] for c in candidates if c["property_hit"]])) \
                         if n_hits > 0 else float("nan")

    metrics = {
        "method":              "B4_random",
        "representation":      "SELFIES",
        "model_size":          "small",
        "property":            args.property,
        "target_value":        args.target,
        "target_mode":         args.target_mode,
        "epsilon":             args.epsilon,
        "n_generated":         n_generated,
        "n_valid":             n_valid,
        "n_two_star":          n_two_star,
        "n_class_pass":        n_class_pass,
        "n_sa_pass":           n_sa_pass,
        "n_scored":            n_scored,
        "n_hits":              n_hits,
        "hit_rate":            round(hit_rate, 6),
        "validity":            round(validity, 4),
        "class_pass_rate":     round(class_pass_rate, 4),
        "sa_pass_rate":        round(sa_pass_rate, 4),
        "mean_prediction":     round(mean_pred, 4),
        "mean_prediction_hits": round(mean_pred_hits, 4) if not np.isnan(mean_pred_hits) else None,
        "mean_target_excess":  round(mean_target_excess, 4) if not np.isnan(mean_target_excess) else None,
        "sampling_time_sec":   round(t_elapsed, 2),
        "hits_per_compute":    round(n_hits / t_elapsed, 6) if t_elapsed > 0 else 0.0,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics_path = output_dir / f"metrics_b4_{args.property}.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[B4] Metrics saved: {metrics_path}")

    if candidates:
        cand_path = output_dir / f"candidate_scores_b4_{args.property}.csv"
        with open(cand_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
            writer.writeheader()
            writer.writerows(candidates)
        print(f"[B4] Candidate scores saved: {cand_path}")

    # Print summary
    print("\n[B4] ── Summary ──────────────────────────────────────────")
    print(f"  Generated:          {n_generated:>8,}")
    print(f"  Valid SMILES:       {n_valid:>8,}  ({validity:.1%})")
    print(f"  Two-star:           {n_two_star:>8,}")
    print(f"  Class pass:         {n_class_pass:>8,}  ({class_pass_rate:.1%} of valid)")
    print(f"  SA pass:            {n_sa_pass:>8,}  ({sa_pass_rate:.1%} of class)")
    print(f"  Scored:             {n_scored:>8,}")
    print(f"  Hits (pred≥{args.target:.0f}):  {n_hits:>8,}")
    print(f"  Hit rate:           {hit_rate:.4f}  ({hit_rate:.2%})")
    print(f"  Mean prediction:    {mean_pred:.1f}")
    print(f"  Sampling time:      {t_elapsed:.1f}s")
    print("[B4] ─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
