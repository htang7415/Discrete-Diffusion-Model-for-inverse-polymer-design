#!/usr/bin/env python
"""F8: Build a paper-ready package from F1-F7 outputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_property_name(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    return text.strip()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _existing_dirs(paths: Iterable[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def _copy_file(
    src: Path,
    dst: Path,
    *,
    category: str,
    copied: list[dict],
    results_dir: Path,
    output_dir: Path,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(
        {
            "category": category,
            "source": _safe_rel(src, results_dir),
            "destination": _safe_rel(dst, output_dir),
        }
    )


def _copy_first(
    candidates: Iterable[Path],
    dst: Path,
    *,
    category: str,
    copied: list[dict],
    missing: list[dict],
    results_dir: Path,
    output_dir: Path,
    missing_label: str,
) -> Optional[Path]:
    src = _first_existing(candidates)
    if src is None:
        missing.append({"category": category, "name": missing_label})
        return None
    _copy_file(src, dst, category=category, copied=copied, results_dir=results_dir, output_dir=output_dir)
    return src


def _collect_glob_unique(source_dirs: Iterable[Path], patterns: Iterable[str]) -> list[Path]:
    collected: list[Path] = []
    seen_names: set[str] = set()
    for directory in source_dirs:
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.name in seen_names:
                    continue
                seen_names.add(path.name)
                collected.append(path)
    return collected


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _f1_embedding_summary(results_dir: Path) -> pd.DataFrame:
    meta_candidates = _collect_glob_unique(
        _existing_dirs(
            [
                results_dir / "step1_alignment_embeddings" / "files",
                results_dir,
            ]
        ),
        ["embedding_meta_*.json"],
    )

    rows = []
    for path in meta_candidates:
        try:
            meta = _read_json(path)
        except Exception:
            continue
        rows.append(
            {
                "view": str(meta.get("view", "")).strip(),
                "model_size": str(meta.get("model_size", "")).strip(),
                "embedding_dim": meta.get("embedding_dim", None),
                "d1_samples": meta.get("d1_samples", None),
                "d2_samples": meta.get("d2_samples", None),
                "pooling": str(meta.get("pooling", "")).strip(),
                "timestep": meta.get("timestep", None),
                "device": str(meta.get("device", "")).strip(),
                "d1_time_sec": meta.get("d1_time_sec", None),
                "d2_time_sec": meta.get("d2_time_sec", None),
                "checkpoint_path": str(meta.get("checkpoint_path", "")).strip(),
                "tokenizer_path": str(meta.get("tokenizer_path", "")).strip(),
                "source_meta": str(path),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "view",
                "model_size",
                "embedding_dim",
                "d1_samples",
                "d2_samples",
                "pooling",
                "timestep",
                "device",
                "d1_time_sec",
                "d2_time_sec",
                "checkpoint_path",
                "tokenizer_path",
                "source_meta",
            ]
        )

    df = pd.DataFrame(rows)
    if "view" in df.columns:
        df["view"] = df["view"].astype(str)
        df = df.sort_values("view").reset_index(drop=True)
    return df


def _configured_properties(config: dict) -> list[str]:
    prop_cfg = config.get("property", {}) or {}
    files = prop_cfg.get("files")
    if files is None:
        return []
    if isinstance(files, str):
        files = [files]
    props = []
    for item in files:
        name = _normalize_property_name(item)
        if name and name not in props:
            props.append(name)
    return props


def _discover_properties(
    config: dict,
    step5_dirs: list[Path],
    step6_dirs: list[Path],
    step7_metric_path: Optional[Path],
) -> list[str]:
    props = _configured_properties(config)
    prop_set = set(props)

    regexes = [
        re.compile(r"^candidate_scores_(.+)\.csv$"),
        re.compile(r"^accepted_candidates_(.+)\.csv$"),
        re.compile(r"^ood_objective_topk_(.+)\.csv$"),
        re.compile(r"^ood_objective_scores_(.+)\.csv$"),
        re.compile(r"^metrics_inverse_ood_objective_(.+)\.csv$"),
    ]
    for directory in step5_dirs + step6_dirs:
        for path in directory.glob("*.csv"):
            name = path.name
            for regex in regexes:
                m = regex.match(name)
                if not m:
                    continue
                prop = _normalize_property_name(m.group(1))
                if prop and prop not in prop_set:
                    prop_set.add(prop)
                    props.append(prop)

    if step7_metric_path is not None and step7_metric_path.exists():
        try:
            df = pd.read_csv(step7_metric_path)
            if "property" in df.columns:
                for prop in df["property"].dropna().astype(str).tolist():
                    name = _normalize_property_name(prop)
                    if name and name not in prop_set:
                        prop_set.add(name)
                        props.append(name)
        except Exception:
            pass

    return props


def main(args):
    config = load_config(args.config)
    paper_cfg = config.get("paper_results", {}) or {}

    enabled = _to_bool(paper_cfg.get("enabled", True), True)
    if args.disable:
        enabled = False
    if not enabled:
        print("Paper package export disabled by config/flag.")
        return

    include_large_csv = _to_bool(paper_cfg.get("include_large_csv", True), True)
    include_figures = _to_bool(paper_cfg.get("include_figures", True), True)
    if args.skip_large_csv:
        include_large_csv = False
    if args.no_figures:
        include_figures = False

    if args.results_dir:
        results_dir = _resolve_path(args.results_dir)
    else:
        results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg_output = str(paper_cfg.get("output_dir", "")).strip()
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    elif cfg_output:
        cfg_path = Path(cfg_output)
        output_dir = cfg_path if cfg_path.is_absolute() else (results_dir / cfg_path)
    else:
        output_dir = results_dir / "paper_package"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    manifest_dir = output_dir / "manifest"
    tables_main_dir = output_dir / "tables" / "main"
    tables_supp_dir = output_dir / "tables" / "supplementary"
    figures_dir = output_dir / "figures"
    run_meta_dir = manifest_dir / "run_meta"
    for directory in [manifest_dir, tables_main_dir, tables_supp_dir, figures_dir, run_meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    missing: list[dict] = []
    step_rows: list[dict] = []

    source_config_candidates = [
        results_dir / "config_used.yaml",
        _resolve_path(args.config),
    ]
    _copy_first(
        source_config_candidates,
        manifest_dir / "config_used.yaml",
        category="manifest",
        copied=copied,
        missing=missing,
        results_dir=results_dir,
        output_dir=output_dir,
        missing_label="config_used.yaml",
    )

    # F1 summary table from embedding meta.
    f1_df = _f1_embedding_summary(results_dir)
    f1_table_path = tables_main_dir / "table_f1_embedding_summary.csv"
    if not f1_df.empty:
        f1_df.to_csv(f1_table_path, index=False)
        copied.append(
            {
                "category": "table_main",
                "source": "generated_from_embedding_meta",
                "destination": _safe_rel(f1_table_path, output_dir),
            }
        )
        f1_status = "completed"
    else:
        missing.append({"category": "table_main", "name": "table_f1_embedding_summary.csv"})
        f1_status = "missing"
    step_rows.append(
        {
            "step_id": "F1",
            "step_name": "alignment_embeddings",
            "status": f1_status,
            "source_metric": "embedding_meta_*.json",
            "paper_table": _safe_rel(f1_table_path, output_dir),
        }
    )

    step_metric_map = [
        (
            "F2",
            "retrieval",
            [
                results_dir / "step2_retrieval" / "metrics" / "metrics_alignment.csv",
                results_dir / "metrics_alignment.csv",
                results_dir / "step2_retrieval" / "metrics_alignment.csv",
            ],
            tables_main_dir / "table_f2_retrieval.csv",
        ),
        (
            "F3",
            "property_heads",
            [
                results_dir / "step3_property" / "metrics" / "metrics_property.csv",
                results_dir / "metrics_property.csv",
                results_dir / "step3_property" / "metrics_property.csv",
            ],
            tables_main_dir / "table_f3_property_heads.csv",
        ),
        (
            "F4",
            "ood_analysis",
            [
                results_dir / "step4_ood" / "metrics" / "metrics_ood.csv",
                results_dir / "metrics_ood.csv",
                results_dir / "step4_ood" / "metrics_ood.csv",
            ],
            tables_main_dir / "table_f4_ood_analysis.csv",
        ),
        (
            "F5",
            "foundation_inverse",
            [
                results_dir / "step5_foundation_inverse" / "metrics" / "metrics_inverse.csv",
                results_dir / "metrics_inverse.csv",
                results_dir / "step5_foundation_inverse" / "metrics_inverse.csv",
            ],
            tables_main_dir / "table_f5_inverse_design.csv",
        ),
        (
            "F6",
            "ood_aware_inverse",
            [
                results_dir / "step6_ood_aware_inverse" / "metrics" / "metrics_inverse_ood_objective.csv",
                results_dir / "metrics_inverse_ood_objective.csv",
                results_dir / "step6_ood_aware_inverse" / "metrics_inverse_ood_objective.csv",
            ],
            tables_main_dir / "table_f6_ood_aware_objective.csv",
        ),
        (
            "F7",
            "chem_physics_analysis",
            [
                results_dir / "step7_chem_physics_analysis" / "metrics" / "metrics_chem_physics.csv",
                results_dir / "metrics_chem_physics.csv",
                results_dir / "step7_chem_physics_analysis" / "metrics_chem_physics.csv",
            ],
            tables_main_dir / "table_f7_chem_physics.csv",
        ),
    ]

    step_metric_sources: dict[str, Optional[Path]] = {}
    for step_id, step_name, candidates, dst in step_metric_map:
        src = _copy_first(
            candidates,
            dst,
            category="table_main",
            copied=copied,
            missing=missing,
            results_dir=results_dir,
            output_dir=output_dir,
            missing_label=dst.name,
        )
        step_metric_sources[step_id] = src
        step_rows.append(
            {
                "step_id": step_id,
                "step_name": step_name,
                "status": "completed" if src is not None else "missing",
                "source_metric": _safe_rel(src, results_dir) if src is not None else "",
                "paper_table": _safe_rel(dst, output_dir),
            }
        )

    step5_dirs = _existing_dirs(
        [
            results_dir / "step5_foundation_inverse" / "files",
            results_dir / "step5_foundation_inverse",
        ]
    )
    step6_dirs = _existing_dirs(
        [
            results_dir / "step6_ood_aware_inverse" / "files",
            results_dir / "step6_ood_aware_inverse",
        ]
    )
    step7_files_dirs = _existing_dirs(
        [
            results_dir / "step7_chem_physics_analysis" / "files",
            results_dir / "step7_chem_physics_analysis",
        ]
    )

    properties = _discover_properties(config, step5_dirs, step6_dirs, step_metric_sources.get("F7"))

    f5_patterns = ["accepted_candidates*.csv"]
    f6_patterns = ["ood_objective_topk*.csv"]
    if include_large_csv:
        f5_patterns = ["candidate_scores*.csv", "accepted_candidates*.csv"]
        f6_patterns = ["ood_objective_scores*.csv", "ood_objective_topk*.csv"]

    for src in _collect_glob_unique(step5_dirs, f5_patterns):
        _copy_file(
            src,
            tables_supp_dir / src.name,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )
    for src in _collect_glob_unique(step6_dirs, f6_patterns):
        _copy_file(
            src,
            tables_supp_dir / src.name,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )
    for src in _collect_glob_unique(
        step7_files_dirs,
        [
            "descriptor_shifts.csv",
            "motif_enrichment.csv",
            "physics_consistency.csv",
            "nearest_neighbor_explanations.csv",
            "property_input_files.csv",
        ],
    ):
        _copy_file(
            src,
            tables_supp_dir / src.name,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    if include_figures:
        step7_figure_dirs = _existing_dirs(
            [
                results_dir / "step7_chem_physics_analysis" / "figures",
                results_dir / "step7_chem_physics_analysis",
            ]
        )
        for src in _collect_glob_unique(step7_figure_dirs, ["figure_f7_chem_physics*.png", "figure_f7_chem_physics*.pdf"]):
            _copy_file(
                src,
                figures_dir / src.name,
                category="figure",
                copied=copied,
                results_dir=results_dir,
                output_dir=output_dir,
            )

    for src in _collect_glob_unique(
        step5_dirs + step6_dirs + step7_files_dirs,
        ["run_meta*.json"],
    ):
        _copy_file(
            src,
            run_meta_dir / src.name,
            category="manifest",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    manifest_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_dir": str(results_dir),
        "paper_package_dir": str(output_dir),
        "config_path": str(_resolve_path(args.config)),
        "properties_detected": properties,
        "include_large_csv": bool(include_large_csv),
        "include_figures": bool(include_figures),
        "steps": step_rows,
        "copied_artifacts": copied,
        "missing_artifacts": missing,
    }

    with open(manifest_dir / "pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    pd.DataFrame(step_rows).to_csv(manifest_dir / "step_status.csv", index=False)

    print(f"Saved paper package to {output_dir}")
    print(f"Manifest: {manifest_dir / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_large_csv", action="store_true")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--clean", action="store_true")
    main(parser.parse_args())
