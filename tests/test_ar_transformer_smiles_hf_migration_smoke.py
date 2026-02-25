"""Smoke tests for AR_Transformer_SMILES HF migration compatibility."""

from pathlib import Path
import tempfile
import sys

import torch
import torch.nn.functional as F

# Ensure AR_Transformer_SMILES/src is importable as package `src`.
REPO_ROOT = Path(__file__).resolve().parents[1]
AR_ROOT = REPO_ROOT / "AR_Transformer_SMILES"
if str(AR_ROOT) not in sys.path:
    sys.path.insert(0, str(AR_ROOT))

from src.data.hf_tokenizer import HFPSmilesTokenizer, load_polymer_tokenizer
from src.model.backbone import DiffusionBackbone
from src.model.hf_ar import (
    PolymerARConfig,
    PolymerARForCausalLM,
    build_and_load_polymer_ar_model,
    build_polymer_ar_model,
    resolve_ar_backbone_path,
)


def test_hf_tokenizer_save_load_roundtrip_parity():
    tokenizer = HFPSmilesTokenizer(max_length=32)
    tokenizer.build_vocab(["*CC*", "*c1ccccc1*", "C(=O)O*"])
    sample = "*c1ccccc1*"
    encoded_before = tokenizer.encode(sample)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tokenizer.save(root / "tokenizer.json")
        tokenizer.save_pretrained(str(root / "tokenizer_hf"))

        from_hf_dir = load_polymer_tokenizer(root)
        from_legacy_file = load_polymer_tokenizer(root / "does_not_exist", root)

        assert encoded_before == from_hf_dir.encode(sample)
        assert encoded_before == from_legacy_file.encode(sample)
        assert tokenizer.decode(encoded_before["input_ids"]) == from_hf_dir.decode(encoded_before["input_ids"])


def test_hf_model_matches_backbone_logits_and_manual_loss():
    torch.manual_seed(0)
    backbone = DiffusionBackbone(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=64,
        max_position_embeddings=32,
        num_diffusion_steps=50,
        dropout=0.0,
        pad_token_id=0,
    )

    config = PolymerARConfig(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=64,
        max_position_embeddings=32,
        num_diffusion_steps=50,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    hf_model = PolymerARForCausalLM(config)
    hf_model.backbone.load_state_dict(backbone.state_dict())

    input_ids = torch.randint(0, 32, (3, 10))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0

    logits = backbone(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_mask == 0, 0)
    manual_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,
    )

    out_hf = hf_model(input_ids=input_ids, attention_mask=attention_mask)

    max_logit_diff = (logits - out_hf.logits).abs().max().item()
    loss_diff = abs(manual_loss.item() - out_hf.loss.item())

    assert max_logit_diff < 1e-6
    assert loss_diff < 1e-6


def test_build_and_load_model_supports_legacy_and_hf_paths():
    tokenizer = HFPSmilesTokenizer(max_length=32)
    tokenizer.build_vocab(["*CC*", "C(=O)O*"])
    backbone_cfg = {
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_hidden_size": 64,
        "max_position_embeddings": 64,
        "dropout": 0.0,
    }
    diff_cfg = {"num_steps": 50}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ckpt_dir = root / "step1_backbone" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = build_polymer_ar_model(backbone_cfg, tokenizer, diff_cfg)

        legacy_path = ckpt_dir / "backbone_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, legacy_path)

        hf_dir = ckpt_dir / "backbone_best_hf"
        model.save_pretrained(str(hf_dir))

        # Preference order should pick HF directory when both exist.
        assert resolve_ar_backbone_path(root) == hf_dir

        loaded_from_hf = build_and_load_polymer_ar_model(
            backbone_cfg, tokenizer, diff_cfg, checkpoint_path=hf_dir
        )
        loaded_from_legacy = build_and_load_polymer_ar_model(
            backbone_cfg, tokenizer, diff_cfg, checkpoint_path=legacy_path
        )

        x = torch.randint(0, tokenizer.vocab_size, (2, 8))
        m = torch.ones_like(x)
        assert loaded_from_hf(input_ids=x, attention_mask=m).logits.shape == (
            loaded_from_legacy(input_ids=x, attention_mask=m).logits.shape
        )
