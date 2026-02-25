"""Smoke tests for attention masking NaN stability."""

from pathlib import Path


BACKBONE_FILES = [
    Path("Bi_Diffusion_SMILES/src/model/backbone.py"),
    Path("Bi_Diffusion_SELFIES/src/model/backbone.py"),
    Path("Bi_Diffusion_Group_SELFIES/src/model/backbone.py"),
    Path("Bi_Diffusion_graph/src/model/backbone.py"),
]


def test_attention_mask_uses_finite_negative_fill():
    for file_path in BACKBONE_FILES:
        source = file_path.read_text(encoding="utf-8")
        assert "masked_fill(mask == 0, -1e9)" in source, f"Missing finite mask fill in {file_path}"
        assert "masked_fill(mask == 0, float('-inf'))" not in source, (
            f"Unsafe -inf mask fill still present in {file_path}"
        )
