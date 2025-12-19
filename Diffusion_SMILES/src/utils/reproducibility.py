"""Reproducibility utilities for experiments."""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Seed value.
        deterministic: Whether to enable deterministic cuDNN behavior.

    Returns:
        Dictionary with seed metadata.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "cuda_available": torch.cuda.is_available()
    }


def save_run_metadata(
    output_dir: Path,
    config_path: Optional[str],
    seed_info: Dict[str, Any]
) -> None:
    """Save run metadata to JSON.

    Args:
        output_dir: Directory to write metadata into.
        config_path: Path to the config file used.
        seed_info: Seed metadata dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(config_path) if config_path else None,
        "seed": seed_info.get("seed"),
        "deterministic": seed_info.get("deterministic"),
        "cuda_available": seed_info.get("cuda_available"),
        "torch_version": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
