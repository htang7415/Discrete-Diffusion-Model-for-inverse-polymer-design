"""Smoke tests for trainer safety/correctness patches."""

from pathlib import Path


SEQUENCE_BACKBONE_TRAINERS = [
    Path("Bi_Diffusion_SMILES/src/training/trainer_backbone.py"),
    Path("Bi_Diffusion_SELFIES/src/training/trainer_backbone.py"),
    Path("Bi_Diffusion_Group_SELFIES/src/training/trainer_backbone.py"),
]

PROPERTY_TRAINERS = [
    Path("Bi_Diffusion_SMILES/src/training/trainer_property.py"),
    Path("Bi_Diffusion_SELFIES/src/training/trainer_property.py"),
    Path("Bi_Diffusion_Group_SELFIES/src/training/trainer_property.py"),
    Path("Bi_Diffusion_graph/src/training/trainer_property.py"),
]

GRAPH_BACKBONE_TRAINERS = [
    Path("Bi_Diffusion_graph/src/training/trainer_backbone.py"),
    Path("Bi_Diffusion_graph/src/training/graph_trainer_backbone.py"),
]


def test_sequence_backbone_trainers_use_device_index_and_explicit_val_steps():
    for file_path in SEQUENCE_BACKBONE_TRAINERS:
        source = file_path.read_text(encoding="utf-8")
        assert "get_device_properties(device_index)" in source
        assert "get_device_properties(0)" not in source
        assert "self.val_steps.append(self.global_step)" in source
        assert "step': self.val_steps[:paired_count]" in source


def test_property_trainers_use_should_step_not_global_step_modulo():
    for file_path in PROPERTY_TRAINERS:
        source = file_path.read_text(encoding="utf-8")
        assert "should_step = (" in source
        assert "def _train_step(self, batch: Dict[str, torch.Tensor], should_step: bool)" in source
        assert "(self.global_step + 1) % self.grad_accum_steps == 0" not in source


def test_graph_backbone_trainers_use_should_step_and_val_steps():
    for file_path in GRAPH_BACKBONE_TRAINERS:
        source = file_path.read_text(encoding="utf-8")
        assert "should_step = (" in source
        assert "self.val_steps.append(self.global_step)" in source
