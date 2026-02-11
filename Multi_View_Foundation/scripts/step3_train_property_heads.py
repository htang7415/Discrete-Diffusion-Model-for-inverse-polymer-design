#!/usr/bin/env python
"""F3: Property heads on foundation embeddings (multi-view)."""

import argparse
from pathlib import Path
import sys
import importlib.util
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies
from src.model.multi_view_model import MultiViewModel


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
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


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
    save_config(config, results_dir / "config_used.yaml")

    property_dir = _resolve_path(config["paths"]["property_dir"])
    prop_cfg = config.get("property", {})
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

    use_alignment = args.use_alignment or bool(prop_cfg.get("use_alignment", False))
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
    ridge_alpha = float(prop_cfg.get("ridge_alpha", 1.0))

    model_dir = results_dir / "step3_property"
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for prop_path in property_files:
        df = pd.read_csv(prop_path)
        prop_name = prop_path.stem
        smiles_col, value_col = _resolve_property_columns(df, prop_name)

        df = df[[smiles_col, value_col]].dropna()
        df = df.rename(columns={smiles_col: "p_smiles", value_col: "target"})
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = df.dropna(subset=["target"])

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
            if view == "selfies":
                view_inputs = []
                view_indices = []
                for idx, smi in enumerate(smiles_list):
                    s = smiles_to_selfies(smi)
                    if s:
                        view_inputs.append(s)
                        view_indices.append(idx)
                embeddings = _embed_sequence(view_inputs, assets, device)
            elif view == "graph":
                embeddings, valid_indices = _embed_graph(smiles_list, assets, device)
                view_indices = valid_indices
            else:
                view_inputs = smiles_list
                view_indices = list(range(num_samples))
                embeddings = _embed_sequence(view_inputs, assets, device)

            if use_alignment and alignment_model is not None:
                embeddings = _project_embeddings(alignment_model, view, embeddings, device)

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

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ])

            train_indices = [i for i in splits["train"] if i in index_map]
            train_rows = [index_map[i] for i in train_indices]
            if not train_rows:
                continue
            model.fit(embeddings[train_rows], targets[train_indices])

            for split_name, split_indices in splits.items():
                valid_indices = [i for i in split_indices if i in index_map]
                split_rows = [index_map[i] for i in valid_indices]
                if not split_rows:
                    metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
                else:
                    preds = model.predict(embeddings[split_rows])
                    metrics = _compute_metrics(targets[valid_indices], preds)
                rep = "Group_SELFIES" if view == "group_selfies" else "Graph" if view == "graph" else view.upper()
                rows.append({
                    "method": "Multi_View_Foundation",
                    "representation": rep,
                    "model_size": data["model_size"],
                    "property": prop_name,
                    "split": split_name,
                    **metrics,
                })

            model_path = model_dir / f"{prop_name}_{view}_ridge.pkl"
            if view == "smiles":
                model_path = model_dir / f"{prop_name}_ridge.pkl"
            if joblib is not None:
                joblib.dump(model, model_path)
            else:
                import pickle
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            meta_path = model_dir / f"{prop_name}_{view}_meta.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "property": prop_name,
                    "view": view,
                    "num_samples": num_samples,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio,
                    "ridge_alpha": ridge_alpha,
                    "use_alignment": use_alignment,
                    "model_path": str(model_path),
                }, f, indent=2)

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

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=ridge_alpha)),
                ])
                train_indices = [i for i in splits["train"] if i in fused_map]
                train_rows = [fused_map[i] for i in train_indices]
                if train_rows:
                    model.fit(fused[train_rows], targets[train_indices])

                    for split_name, split_indices in splits.items():
                        valid_indices = [i for i in split_indices if i in fused_map]
                        split_rows = [fused_map[i] for i in valid_indices]
                        if not split_rows:
                            metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
                        else:
                            preds = model.predict(fused[split_rows])
                            metrics = _compute_metrics(targets[valid_indices], preds)
                        rows.append({
                            "method": "Multi_View_Foundation",
                            "representation": "MultiViewMean",
                            "model_size": "mixed",
                            "property": prop_name,
                            "split": split_name,
                            **metrics,
                        })

                    model_path = model_dir / f"{prop_name}_multiview_mean.pkl"
                    if joblib is not None:
                        joblib.dump(model, model_path)
                    else:
                        import pickle
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)

                    meta_path = model_dir / f"{prop_name}_multiview_mean_meta.json"
                    with open(meta_path, "w") as f:
                        json.dump({
                            "property": prop_name,
                            "view": "multiview_mean",
                            "num_samples": num_samples,
                            "num_common": len(common),
                            "train_ratio": train_ratio,
                            "val_ratio": val_ratio,
                            "test_ratio": test_ratio,
                            "ridge_alpha": ridge_alpha,
                            "use_alignment": use_alignment,
                            "model_path": str(model_path),
                        }, f, indent=2)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results_dir / "metrics_property.csv", index=False)
    print(f"Saved metrics_property.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    parser.add_argument("--views", type=str, default=None, help="comma-separated list of views")
    main(parser.parse_args())
