"""Smoke checks for SELFIES unlabeled CSV schema standardization."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_STEP0_SCRIPTS = [
    "Bi_Diffusion_SMILES/scripts/step0_prepare_data.py",
    "Bi_Diffusion_SELFIES/scripts/step0_prepare_data.py",
    "Bi_Diffusion_Group_SELFIES/scripts/step0_prepare_data.py",
    "Bi_Diffusion_graph/scripts/step0_prepare_data.py",
    "AR_Transformer_SMILES/scripts/step0_prepare_data.py",
    "AR_Transformer_SELFIES/scripts/step0_prepare_data.py",
    "AR_Transformer_Group_SELFIES/scripts/step0_prepare_data.py",
]

SELFIES_STEP0_SCRIPTS = [
    "Bi_Diffusion_SELFIES/scripts/step0_prepare_data.py",
    "AR_Transformer_SELFIES/scripts/step0_prepare_data.py",
]

STEP1_SCRIPTS = [
    "Bi_Diffusion_SELFIES/scripts/step1_train_backbone.py",
    "AR_Transformer_SELFIES/scripts/step1_train_backbone.py",
]

STEP6_SCRIPTS = [
    "Bi_Diffusion_SELFIES/scripts/step6_hyperparameter_tuning.py",
    "AR_Transformer_SELFIES/scripts/step6_hyperparameter_tuning.py",
]

LENGTH_SAMPLING_SCRIPTS = [
    "Bi_Diffusion_SELFIES/scripts/step2_sample_and_evaluate.py",
    "AR_Transformer_SELFIES/scripts/step2_sample_and_evaluate.py",
    "Bi_Diffusion_SELFIES/scripts/step4_inverse_design.py",
    "AR_Transformer_SELFIES/scripts/step4_inverse_design.py",
    "Bi_Diffusion_SELFIES/scripts/step5_class_design.py",
    "AR_Transformer_SELFIES/scripts/step5_class_design.py",
]

UTILS_FILES = [
    "Bi_Diffusion_SELFIES/src/utils/selfies_utils.py",
    "AR_Transformer_SELFIES/src/utils/selfies_utils.py",
]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_step0_uses_preprocessed_shared_unlabeled_splits():
    for rel_path in ALL_STEP0_SCRIPTS:
        source = _read(rel_path)
        assert "require_preprocessed_unlabeled_splits" in source
        assert "load_or_create_shared_unlabeled_splits" not in source
        assert "link_local_unlabeled_splits" not in source
        assert "train_shared_path" in source
        assert "val_shared_path" in source
        assert "to_csv(results_dir / 'train_unlabeled.csv'" not in source
        assert "to_csv(results_dir / 'val_unlabeled.csv'" not in source


def test_selfies_step0_rebuilds_selfies_view_from_shared_schema():
    for rel_path in SELFIES_STEP0_SCRIPTS:
        source = _read(rel_path)
        assert "ensure_selfies_column" in source


def test_step1_and_step6_rebuild_selfies_when_missing():
    for rel_path in STEP1_SCRIPTS + STEP6_SCRIPTS:
        source = _read(rel_path)
        assert "ensure_selfies_column" in source


def test_length_sampling_uses_dataframe_fallback_helper():
    for rel_path in LENGTH_SAMPLING_SCRIPTS:
        source = _read(rel_path)
        assert "sample_selfies_from_dataframe" in source


def test_selfies_utils_define_new_compat_helpers():
    for rel_path in UTILS_FILES:
        source = _read(rel_path)
        assert "def ensure_selfies_column(" in source
        assert "def sample_selfies_from_dataframe(" in source
