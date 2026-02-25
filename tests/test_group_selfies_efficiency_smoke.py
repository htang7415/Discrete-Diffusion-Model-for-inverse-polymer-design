"""Smoke checks for Group SELFIES Step1 efficiency wiring."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

GROUP_STEP1_SCRIPTS = [
    "Bi_Diffusion_Group_SELFIES/scripts/step1_train_backbone.py",
]

GROUP_TRAINERS = [
    "Bi_Diffusion_Group_SELFIES/src/training/trainer_backbone.py",
]

GROUP_CONFIGS = [
    "Bi_Diffusion_Group_SELFIES/configs/config.yaml",
]

GROUP_SAMPLERS = [
    "Bi_Diffusion_Group_SELFIES/src/data/samplers.py",
]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_group_step1_scripts_wire_dynamic_padding_and_bucket_sampler():
    for rel_path in GROUP_STEP1_SCRIPTS:
        source = _read(rel_path)
        assert "dynamic_collate_fn" in source
        assert "LengthBucketBatchSampler" in source
        assert "pad_to_max_length=not dynamic_padding" in source
        assert "batch_sampler=train_batch_sampler" in source


def test_group_trainers_seed_batch_sampler_each_epoch():
    for rel_path in GROUP_TRAINERS:
        source = _read(rel_path)
        assert 'batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)' in source
        assert 'if hasattr(batch_sampler, "set_epoch"):' in source


def test_group_configs_define_dynamic_padding_and_bucketing_controls():
    for rel_path in GROUP_CONFIGS:
        source = _read(rel_path)
        assert "dynamic_padding: true" in source
        assert "length_bucket_sampler: false" in source
        assert "bucket_size_multiplier: 50" in source


def test_group_sampler_module_is_present():
    for rel_path in GROUP_SAMPLERS:
        source = _read(rel_path)
        assert "class LengthBucketBatchSampler" in source
        assert "def set_epoch(self, epoch: int) -> None:" in source
