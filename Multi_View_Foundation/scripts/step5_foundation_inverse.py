#!/usr/bin/env python
"""F5: Foundation-enhanced inverse design with reranking (initial)."""

import argparse
import json
from pathlib import Path
import sys
import time
import importlib.util
from typing import Optional

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from shared.ood_metrics import knn_distances
from src.model.multi_view_model import MultiViewModel

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _resolve_with_base(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def _view_to_representation(view: str) -> str:
    return "SMILES_BPE" if view == "smiles_bpe" else "SMILES"


def _select_smiles_view(config: dict, override: Optional[str]) -> str:
    if override:
        return override
    if config.get("smiles_encoder", {}).get("method_dir"):
        return "smiles"
    if config.get("smiles_bpe_encoder", {}).get("method_dir"):
        return "smiles_bpe"
    return "smiles"


def _load_module(module_name: str, path: Path):
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


def _load_backbone(encoder_cfg: dict, device: str):
    method_dir = _resolve_path(encoder_cfg.get("method_dir", "../Bi_Diffusion_SMILES"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = _resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Bi_Diffusion_SMILES config not found: {config_path}")

    smiles_cfg = load_config(str(config_path))

    scales_mod = _load_module(
        "bdiff_scales",
        method_dir / "src" / "utils" / "model_scales.py",
    )
    tokenizer_mod = _load_module(
        "bdiff_tokenizer",
        method_dir / "src" / "data" / "tokenizer.py",
    )
    backbone_mod = _load_module(
        "bdiff_backbone",
        method_dir / "src" / "model" / "backbone.py",
    )

    get_model_config = scales_mod.get_model_config
    get_results_dir = scales_mod.get_results_dir
    PSmilesTokenizer = tokenizer_mod.PSmilesTokenizer
    DiffusionBackbone = backbone_mod.DiffusionBackbone

    model_size = encoder_cfg.get("model_size")
    backbone_config = get_model_config(model_size, smiles_cfg, model_type="sequence")
    diffusion_steps = smiles_cfg.get("diffusion", {}).get("num_steps", 50)

    base_results_dir = encoder_cfg.get("results_dir")
    if base_results_dir:
        base_results_dir = _resolve_path(base_results_dir)
    else:
        base_results_dir = _resolve_with_base(smiles_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = _resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / "tokenizer.json"

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

    tokenizer = PSmilesTokenizer.load(str(tokenizer_path))
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
    }


def _embed_smiles(smiles_list, tokenizer, backbone, device: str, batch_size: int, pooling: str, timestep: int):
    if not smiles_list:
        return np.zeros((0, backbone.hidden_size), dtype=np.float32)

    embeddings = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start:start + batch_size]
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


def _load_property_model(model_path: Path):
    if joblib is not None:
        return joblib.load(model_path)
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


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


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, results_dir / "config_used.yaml")

    candidates_path = _resolve_path(args.candidates_csv)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_path}")

    candidates_df = pd.read_csv(candidates_path)
    smiles_col = args.smiles_column
    if smiles_col is None:
        for candidate in ["smiles", "p_smiles", "psmiles"]:
            if candidate in candidates_df.columns:
                smiles_col = candidate
                break
    if smiles_col is None or smiles_col not in candidates_df.columns:
        raise ValueError("Candidates CSV must include a SMILES column.")

    smiles_list = candidates_df[smiles_col].astype(str).tolist()
    validity_mask = [
        _check_validity(smi) and _count_stars(smi) == 2
        for smi in smiles_list
    ]

    valid_smiles = [s for s, v in zip(smiles_list, validity_mask) if v]

    encoder_view = _select_smiles_view(config, args.encoder_view)
    encoder_cfg = config.get(f"{encoder_view}_encoder", {})
    if not encoder_cfg:
        raise ValueError(f"Encoder config not found for view: {encoder_view}")
    device = encoder_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assets = _load_backbone(encoder_cfg=encoder_cfg, device=device)
    pooling = encoder_cfg.get("pooling", "mean")
    timestep = encoder_cfg.get("timestep", 1)
    batch_size = int(encoder_cfg.get("batch_size", 256))

    t0 = time.time()
    embeddings = _embed_smiles(valid_smiles, assets["tokenizer"], assets["backbone"], device, batch_size, pooling, timestep)
    embed_time = time.time() - t0

    alignment_model = None
    if args.use_alignment:
        view_dims = {encoder_view: embeddings.shape[1] if embeddings.size else assets["backbone"].hidden_size}
        alignment_model = _load_alignment_model(results_dir, view_dims, config, args.alignment_checkpoint)
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for --use_alignment")
        device_proj = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = _project_embeddings(alignment_model, encoder_view, embeddings, device=device_proj)

    model_path = args.property_model_path
    if model_path is None:
        model_path = results_dir / "step3_property" / f"{args.property}_ridge.pkl"
    else:
        model_path = _resolve_path(model_path)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Property model not found: {model_path}")

    model = _load_property_model(Path(model_path))
    preds = model.predict(embeddings) if len(valid_smiles) else np.array([])

    target_value = float(args.target)
    epsilon = float(args.epsilon)
    hits = np.abs(preds - target_value) <= epsilon

    n_generated = len(smiles_list)
    n_valid = len(valid_smiles)
    n_hits = int(hits.sum()) if n_valid else 0
    success_rate = n_hits / n_valid if n_valid else 0.0

    validity = n_valid / n_generated if n_generated else 0.0
    uniqueness = len(set(valid_smiles)) / n_valid if n_valid else 0.0

    train_smiles_path = _resolve_path(config["paths"]["polymer_file"])
    training_set = _load_training_smiles(train_smiles_path)
    novelty = 0.0
    if n_valid:
        novelty = sum(1 for s in valid_smiles if s not in training_set) / n_valid

    d2_distance_scores = None
    rerank_metrics = {
        "rerank_applied": False,
    }

    if args.rerank_strategy == "d2_distance":
        d2_path = results_dir / "embeddings_d2.npy"
        if not d2_path.exists():
            raise FileNotFoundError("embeddings_d2.npy not found. Run F1 first.")
        d2_embeddings = np.load(d2_path)
        if args.use_alignment and alignment_model is not None:
            device_proj = "cuda" if torch.cuda.is_available() else "cpu"
            d2_embeddings = _project_embeddings(alignment_model, encoder_view, d2_embeddings, device=device_proj)
        distances = knn_distances(embeddings, d2_embeddings, k=args.ood_k)
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

    metrics_row = {
        "method": "Multi_View_Foundation",
        "representation": _view_to_representation(encoder_view),
        "model_size": assets["model_size"],
        "property": args.property,
        "target_value": target_value,
        "epsilon": epsilon,
        "n_generated": n_generated,
        "n_valid": n_valid,
        "n_hits": n_hits,
        "success_rate": round(success_rate, 4),
        "validity": round(validity, 4),
        "validity_two_stars": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "novelty": round(novelty, 4),
        "avg_diversity": None,
        **_achievement_rates(preds, target_value),
        "sampling_time_sec": round(embed_time, 2),
        "valid_per_compute": round(n_valid / max(embed_time, 1e-9), 4) if n_valid else 0.0,
        **rerank_metrics,
    }

    if rerank_metrics.get("rerank_applied"):
        metrics_row["valid_per_compute_rerank"] = metrics_row["valid_per_compute"]
    else:
        metrics_row["rerank_strategy"] = args.rerank_strategy

    pd.DataFrame([metrics_row]).to_csv(results_dir / "metrics_inverse.csv", index=False)

    out_dir = results_dir / "step5_foundation_inverse"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = pd.DataFrame({
        "smiles": valid_smiles,
        "prediction": preds,
        "abs_error": np.abs(preds - target_value) if len(preds) else [],
        "hit": hits,
        "d2_distance": d2_distance_scores if d2_distance_scores is not None else None,
    })
    valid_rows.to_csv(out_dir / "candidate_scores.csv", index=False)

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump({
            "candidates_csv": str(candidates_path),
            "smiles_column": smiles_col,
            "encoder_view": encoder_view,
            "property": args.property,
            "target_value": target_value,
            "epsilon": epsilon,
            "rerank_strategy": args.rerank_strategy,
            "rerank_top_k": int(args.rerank_top_k),
            "use_alignment": bool(args.use_alignment),
        }, f, indent=2)

    print(f"Saved metrics_inverse.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--encoder_view", type=str, default=None)
    parser.add_argument("--candidates_csv", type=str, required=True)
    parser.add_argument("--smiles_column", type=str, default=None)
    parser.add_argument("--property", type=str, required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=30.0)
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--rerank_strategy", type=str, default="d2_distance")
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--ood_k", type=int, default=5)
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    main(parser.parse_args())
