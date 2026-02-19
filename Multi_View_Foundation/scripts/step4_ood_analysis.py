#!/usr/bin/env python
"""F4: OOD analysis (initial implementation)."""

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from shared.ood_metrics import compute_ood_metrics_from_files
from src.model.multi_view_model import MultiViewModel


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _load_model_size(results_dir: Path, config: dict) -> str:
    meta_path = results_dir / "embedding_meta.json"
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
    meta_path = results_dir / "embedding_meta.json"
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
    save_config(config, results_dir / "config_used.yaml")
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
        d1_path = results_dir / "embeddings_d1_aligned.npy"
        d2_path = results_dir / "embeddings_d2_aligned.npy"
        np.save(d1_path, d1_proj)
        np.save(d2_path, d2_proj)

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

    out_path = results_dir / "metrics_ood.csv"
    import pandas as pd
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved metrics_ood.csv to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generated_embeddings", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--use_faiss", dest="use_faiss", action="store_true")
    parser.add_argument("--no_faiss", dest="use_faiss", action="store_false")
    parser.set_defaults(use_faiss=None)
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    main(parser.parse_args())
