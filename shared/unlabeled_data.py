"""Utilities for shared unlabeled train/val splits across subprojects."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


DEFAULT_UNLABELED_COLUMNS: Tuple[str, str] = ("p_smiles", "sa_score")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], csv_path: Path | None = None) -> None:
    """Validate that required columns are present in dataframe."""
    required = list(columns)
    missing = [col for col in required if col not in df.columns]
    if not missing:
        return
    location = f" in {csv_path}" if csv_path is not None else ""
    raise ValueError(f"Missing required columns{location}: {missing}")


def _select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Select required columns and return a copy."""
    selected = list(columns)
    _require_columns(df, selected)
    return df[selected].copy().reset_index(drop=True)


def get_shared_unlabeled_paths(repo_root: Path) -> Tuple[Path, Path]:
    """Return canonical shared train/val unlabeled paths.

    Preference order:
    1) .csv.gz (new default to save disk)
    2) .csv (backward compatibility)
    """
    shared_dir = Path(repo_root) / "Data" / "Polymer"

    train_gz = shared_dir / "train_unlabeled.csv.gz"
    val_gz = shared_dir / "val_unlabeled.csv.gz"
    train_csv = shared_dir / "train_unlabeled.csv"
    val_csv = shared_dir / "val_unlabeled.csv"

    train_path = train_gz if train_gz.exists() else train_csv
    val_path = val_gz if val_gz.exists() else val_csv

    # If neither variant exists, default to .csv.gz for new creations.
    if not train_path.exists() and not train_csv.exists():
        train_path = train_gz
    if not val_path.exists() and not val_csv.exists():
        val_path = val_gz

    return train_path, val_path


def require_preprocessed_unlabeled_splits(repo_root: Path) -> Tuple[Path, Path]:
    """Return preprocessed shared train/val split paths (.csv.gz only).

    This enforces the project convention where data preprocessing and splitting
    are performed once via ``Data/Polymer/split_unlabeled_full_columns.py``,
    and all method pipelines consume the resulting compressed files directly.
    """
    shared_dir = Path(repo_root) / "Data" / "Polymer"
    train_path = shared_dir / "train_unlabeled.csv.gz"
    val_path = shared_dir / "val_unlabeled.csv.gz"

    missing = [str(p) for p in (train_path, val_path) if not p.exists()]
    if missing:
        missing_str = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Preprocessed shared unlabeled splits were not found.\n"
            f"Missing files:\n{missing_str}\n"
            "Generate them first with:\n"
            "  python Data/Polymer/split_unlabeled_full_columns.py --overwrite"
        )

    return train_path, val_path


def load_or_create_shared_unlabeled_splits(
    data_loader,
    repo_root: Path,
    columns: Iterable[str] = DEFAULT_UNLABELED_COLUMNS,
    create_if_missing: bool = False,
) -> Dict[str, object]:
    """Load shared unlabeled splits and optionally create when missing.

    Args:
        data_loader: Subproject data loader with prepare_unlabeled_data().
        repo_root: Repository root path.
        columns: Columns to keep in shared files.
        create_if_missing: Whether to create train/val shared CSVs from
            ``data_loader.prepare_unlabeled_data()`` when they do not exist.

    Returns:
        Dict with:
            - train_df / val_df
            - train_path / val_path
            - created (bool): whether files were created in this run
    """
    selected_cols = list(columns)
    train_path, val_path = get_shared_unlabeled_paths(Path(repo_root))
    train_path.parent.mkdir(parents=True, exist_ok=True)

    if train_path.exists() and val_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        _require_columns(train_df, selected_cols, train_path)
        _require_columns(val_df, selected_cols, val_path)
        return {
            "train_df": train_df[selected_cols].copy().reset_index(drop=True),
            "val_df": val_df[selected_cols].copy().reset_index(drop=True),
            "train_path": train_path,
            "val_path": val_path,
            "created": False,
        }

    if not create_if_missing:
        raise FileNotFoundError(
            "Shared unlabeled splits were not found. Expected files:\n"
            f"  - {train_path}\n"
            f"  - {val_path}\n"
            "Create them first using:\n"
            "  python Data/Polymer/split_unlabeled_full_columns.py --overwrite"
        )

    unlabeled_data = data_loader.prepare_unlabeled_data()
    train_df = _select_columns(unlabeled_data["train"], selected_cols)
    val_df = _select_columns(unlabeled_data["val"], selected_cols)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_path": train_path,
        "val_path": val_path,
        "created": True,
    }


def _safe_resolve(path: Path) -> Path:
    """Resolve paths while tolerating broken symlinks."""
    try:
        return path.resolve()
    except FileNotFoundError:
        return path.absolute()


def _link_or_copy(src: Path, dst: Path) -> str:
    """Create symlink from dst -> src; copy if symlink creation fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink():
        existing_target = _safe_resolve(dst)
        if existing_target == _safe_resolve(src):
            return "symlink"
        dst.unlink()
    elif dst.exists():
        dst.unlink()

    rel_src = os.path.relpath(src, start=dst.parent)
    try:
        dst.symlink_to(rel_src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def link_local_unlabeled_splits(results_dir: Path, train_src: Path, val_src: Path) -> Dict[str, str]:
    """Expose shared train/val splits via local results paths.

    This keeps legacy scripts (that expect results/train_unlabeled.csv) working
    without storing duplicate large CSV files.
    """
    results_dir = Path(results_dir)
    train_dst = results_dir / "train_unlabeled.csv"
    val_dst = results_dir / "val_unlabeled.csv"

    train_mode = _link_or_copy(train_src, train_dst)
    val_mode = _link_or_copy(val_src, val_dst)

    return {
        "train_mode": train_mode,
        "val_mode": val_mode,
        "train_dst": str(train_dst),
        "val_dst": str(val_dst),
        "train_src": str(train_src),
        "val_src": str(val_src),
    }
