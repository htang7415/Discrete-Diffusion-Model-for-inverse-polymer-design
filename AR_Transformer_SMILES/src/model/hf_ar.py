"""Hugging Face compatible AR model wrappers with legacy parity.

This module keeps the exact legacy backbone architecture and loss behavior,
while exposing a transformers-native model/config interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from .backbone import DiffusionBackbone


class PolymerARConfig(PretrainedConfig):
    """Configuration for the polymer autoregressive causal LM."""

    model_type = "polymer_ar"

    def __init__(
        self,
        vocab_size: int = 0,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_hidden_size: int = 3072,
        max_position_embeddings: int = 256,
        num_diffusion_steps: int = 100,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        causal: bool = True,
        use_cache: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.use_cache = bool(use_cache)


class PolymerARForCausalLM(PreTrainedModel, GenerationMixin):
    """Transformers-compatible causal LM backed by the legacy backbone."""

    config_class = PolymerARConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"

    def __init__(self, config: PolymerARConfig):
        super().__init__(config)
        self.pad_token_id = int(config.pad_token_id)
        self.backbone = DiffusionBackbone(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_hidden_size=config.ffn_hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            num_diffusion_steps=config.num_diffusion_steps,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id,
            causal=config.causal,
        )

    def get_input_embeddings(self):
        return self.backbone.token_embedding

    def set_input_embeddings(self, value):
        self.backbone.token_embedding = value

    def get_output_embeddings(self):
        return self.backbone.output_proj

    def set_output_embeddings(self, value):
        self.backbone.output_proj = value

    def initialize_weights(self):  # type: ignore[override]
        """Weights are initialized by the wrapped legacy backbone constructor."""
        return

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Expose legacy helper used by downstream property pipelines."""
        return self.backbone.get_hidden_states(input_ids, attention_mask, layer_idx)

    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Expose legacy helper used by downstream property pipelines."""
        return self.backbone.get_pooled_output(input_ids, attention_mask, pooling)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        # This model does not use KV cache in generation; keep full-prefix decoding.
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutput:
        """Forward pass with legacy next-token loss semantics."""
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        logits = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if labels is None:
            labels = input_ids

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_mask == 0, self.pad_token_id)

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(loss=loss, logits=logits)


def build_polymer_ar_config(
    backbone_config: Mapping[str, Any],
    tokenizer: Any,
    diffusion_config: Mapping[str, Any],
) -> PolymerARConfig:
    """Build HF config from the existing YAML + tokenizer settings."""
    return PolymerARConfig(
        vocab_size=int(tokenizer.vocab_size),
        hidden_size=int(backbone_config["hidden_size"]),
        num_layers=int(backbone_config["num_layers"]),
        num_heads=int(backbone_config["num_heads"]),
        ffn_hidden_size=int(backbone_config["ffn_hidden_size"]),
        max_position_embeddings=int(backbone_config["max_position_embeddings"]),
        num_diffusion_steps=int(diffusion_config["num_steps"]),
        dropout=float(backbone_config["dropout"]),
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        causal=True,
        use_cache=False,
    )


def build_polymer_ar_model(
    backbone_config: Mapping[str, Any],
    tokenizer: Any,
    diffusion_config: Mapping[str, Any],
) -> PolymerARForCausalLM:
    """Instantiate HF model with legacy-compatible config values."""
    hf_config = build_polymer_ar_config(backbone_config, tokenizer, diffusion_config)
    return PolymerARForCausalLM(hf_config)


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize state dict keys across compile/DDP/trainer wrappers."""
    normalized = dict(state_dict)

    if any(k.startswith("_orig_mod.") for k in normalized):
        normalized = {k.replace("_orig_mod.", "", 1): v for k, v in normalized.items()}

    if normalized and all(k.startswith("module.") for k in normalized):
        normalized = {k.replace("module.", "", 1): v for k, v in normalized.items()}

    if normalized and all(k.startswith("model.") for k in normalized):
        normalized = {k.replace("model.", "", 1): v for k, v in normalized.items()}

    return normalized


def extract_backbone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract a backbone-only state dict (no `backbone.` prefix)."""
    normalized = _strip_common_prefixes(state_dict)

    if any(k.startswith("backbone.") for k in normalized):
        return {
            k.replace("backbone.", "", 1): v
            for k, v in normalized.items()
            if k.startswith("backbone.")
        }

    return normalized


def load_polymer_ar_state_dict(
    model: PolymerARForCausalLM,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Load checkpoints from either legacy wrapper or backbone-only exports."""
    normalized = _strip_common_prefixes(state_dict)

    # Preferred path: full LM state dict (keys include backbone.*)
    try:
        model.load_state_dict(normalized)
        return
    except RuntimeError:
        pass

    # Fallback: backbone-only checkpoints.
    model.backbone.load_state_dict(extract_backbone_state_dict(normalized))


def load_polymer_ar_checkpoint(
    model: PolymerARForCausalLM,
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint from legacy .pt file or HF `save_pretrained` directory."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        loaded = PolymerARForCausalLM.from_pretrained(str(checkpoint_path), local_files_only=True)
        model.load_state_dict(loaded.state_dict())
        return {"source": "hf_pretrained", "path": str(checkpoint_path)}

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} missing 'model_state_dict'")
    load_polymer_ar_state_dict(model, checkpoint["model_state_dict"])
    return checkpoint


def build_and_load_polymer_ar_model(
    backbone_config: Mapping[str, Any],
    tokenizer: Any,
    diffusion_config: Mapping[str, Any],
    checkpoint_path: Optional[Union[str, Path]] = None,
    map_location: str = "cpu",
) -> PolymerARForCausalLM:
    """Build model from config and optionally load weights from checkpoint/path."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.is_dir():
            return PolymerARForCausalLM.from_pretrained(str(path), local_files_only=True)

    model = build_polymer_ar_model(backbone_config, tokenizer, diffusion_config)
    if checkpoint_path is not None:
        load_polymer_ar_checkpoint(model, checkpoint_path, map_location=map_location)
    return model


def resolve_ar_backbone_path(
    results_dir: Union[str, Path],
    preferred: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve checkpoint path with preference order: explicit > HF dir > legacy .pt."""
    if preferred is not None:
        return Path(preferred)

    ckpt_dir = Path(results_dir) / "step1_backbone" / "checkpoints"
    hf_dir = ckpt_dir / "backbone_best_hf"
    legacy_file = ckpt_dir / "backbone_best.pt"

    if hf_dir.is_dir():
        return hf_dir
    return legacy_file


try:
    AutoConfig.register(PolymerARConfig.model_type, PolymerARConfig)
except ValueError:
    pass

try:
    AutoModelForCausalLM.register(PolymerARConfig, PolymerARForCausalLM)
except ValueError:
    pass
