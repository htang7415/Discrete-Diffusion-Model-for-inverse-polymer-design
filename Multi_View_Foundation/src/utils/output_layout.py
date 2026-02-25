"""Output layout helpers for MVF pipeline steps.

Provides a consistent per-step structure:
  <results_dir>/<step_name>/{files,metrics,figures}
or, for property-scoped artifacts:
  <results_dir>/<step_name>/<property>/{files,metrics,figures}

Writers can optionally mirror artifacts to legacy locations for backward
compatibility with existing scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _normalize_optional_property_name(property_name: Optional[str]) -> Optional[str]:
    if property_name is None:
        return None
    text = str(property_name).strip()
    if not text:
        return None
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    text = text.replace("/", "_").replace("\\", "_").strip()
    return text or None


def ensure_step_dirs(results_dir: Path, step_name: str, property_name: Optional[str] = None) -> dict:
    root_step_dir = Path(results_dir) / step_name
    property_token = _normalize_optional_property_name(property_name)
    step_dir = root_step_dir / property_token if property_token else root_step_dir
    files_dir = step_dir / "files"
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    for directory in (root_step_dir, step_dir, files_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root_step_dir": root_step_dir,
        "step_dir": step_dir,
        "files_dir": files_dir,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
        "property_name": property_token,
    }


def _iter_unique_paths(primary_path: Path, legacy_paths: Optional[Iterable[Path]]) -> Iterable[Path]:
    seen = set()
    candidates = [Path(primary_path)]
    if legacy_paths:
        candidates.extend(Path(p) for p in legacy_paths)
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        path.parent.mkdir(parents=True, exist_ok=True)
        yield path


def save_csv(df, primary_path: Path, legacy_paths: Optional[Iterable[Path]] = None, index: bool = False) -> None:
    for path in _iter_unique_paths(primary_path, legacy_paths):
        df.to_csv(path, index=index)


def save_numpy(array, primary_path: Path, legacy_paths: Optional[Iterable[Path]] = None) -> None:
    arr = np.asarray(array)
    for path in _iter_unique_paths(primary_path, legacy_paths):
        np.save(path, arr)


def save_json(payload, primary_path: Path, legacy_paths: Optional[Iterable[Path]] = None, indent: int = 2) -> None:
    for path in _iter_unique_paths(primary_path, legacy_paths):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
