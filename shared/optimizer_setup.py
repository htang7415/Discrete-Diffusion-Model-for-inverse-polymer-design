"""Optimizer setup helpers for Step1 backbone training."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.optim import AdamW, Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def _to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r} ({type(value).__name__})")


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be integer-like, got {value!r} ({type(value).__name__})")


def _cfg_value(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Read config value where explicit null should still use the default."""
    value = cfg.get(key, default)
    return default if value is None else value


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Get underlying module from DDP/torch.compile wrappers."""
    unwrapped = model
    if isinstance(unwrapped, DDP):
        unwrapped = unwrapped.module
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod
    return unwrapped


def _zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalized update used by Muon-style optimizers."""
    if grad.ndim != 2:
        raise ValueError(f"Muon update expects 2D tensors, got shape {tuple(grad.shape)}")

    # Use reduced precision matmuls on CUDA for speed, FP32 on CPU for compatibility.
    work_dtype = torch.bfloat16 if grad.is_cuda else torch.float32
    x = grad.to(work_dtype)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.transpose(0, 1)
        transposed = True

    eps = torch.tensor(1e-7, device=x.device, dtype=x.dtype)
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(max(1, int(steps))):
        xx_t = x @ x.transpose(0, 1)
        poly = b * xx_t + c * (xx_t @ xx_t)
        x = a * x + poly @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(grad.dtype)


class MuonAdamW(Optimizer):
    """Single optimizer with mixed Muon/AdamW parameter groups."""

    def __init__(self, params: Iterable[Dict[str, Any]]) -> None:
        defaults = dict(
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            kind="adamw",
            momentum=0.95,
            nesterov=True,
            backend_steps=5,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            kind = group.get("kind", "adamw")
            if kind not in {"adamw", "muon"}:
                raise ValueError(f"Unsupported optimizer group kind: {kind}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            kind = group.get("kind", "adamw")
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))

            if kind == "adamw":
                beta1, beta2 = group.get("betas", (0.9, 0.95))
                eps = float(group.get("eps", 1e-8))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("MuonAdamW AdamW path does not support sparse gradients.")

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step_t = state["step"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step_t
                    bias_correction2 = 1.0 - beta2 ** step_t
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1

                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                continue

            # Muon group: momentum + orthogonalized matrix updates.
            momentum = float(group.get("momentum", 0.95))
            nesterov = bool(group.get("nesterov", True))
            backend_steps = int(group.get("backend_steps", 5))
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon path does not support sparse gradients.")
                if grad.ndim != 2:
                    raise RuntimeError(
                        f"Muon path only supports 2D tensors, got shape {tuple(grad.shape)}"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )

                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz5(update, steps=backend_steps)

                rows, cols = p.shape
                matrix_scale = max(1.0, rows / max(1, cols))
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-lr * matrix_scale)

        return loss


def _build_muon_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    base_learning_rate: float,
    muon_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Split model params into Muon and AdamW groups."""
    layer_tokens = tuple(muon_cfg.get("layer_name_tokens", ["layers.", "blocks."]))
    exclude_tokens = tuple(
        muon_cfg.get(
            "exclude_name_tokens",
            [
                "token_embedding",
                "position_embedding",
                "timestep_embedding",
                "time_embedding",
                "output_proj",
                "node_head",
                "edge_head",
                "edge_embedding",
            ],
        )
    )

    adamw_params: List[torch.nn.Parameter] = []
    muon_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_matrix = param.ndim == 2
        in_transformer_blocks = any(tok in name for tok in layer_tokens)
        is_excluded = any(tok in name for tok in exclude_tokens)
        if is_matrix and in_transformer_blocks and not is_excluded:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        return [], {
            "using_muon": False,
            "reason": "no_muon_eligible_parameters",
            "num_muon_tensors": 0,
            "num_adamw_tensors": len(adamw_params),
        }

    adamw_lr = _to_float(
        _cfg_value(muon_cfg, "adamw_lr", base_learning_rate),
        "optimizer.muon_adamw.adamw_lr",
    )
    muon_lr_raw = muon_cfg.get("muon_lr", None)
    if muon_lr_raw is None:
        muon_lr_multiplier = _to_float(
            _cfg_value(muon_cfg, "muon_lr_multiplier", 20.0),
            "optimizer.muon_adamw.muon_lr_multiplier",
        )
        muon_lr = adamw_lr * muon_lr_multiplier
    else:
        muon_lr = _to_float(muon_lr_raw, "optimizer.muon_adamw.muon_lr")

    adamw_betas_raw = _cfg_value(muon_cfg, "adamw_betas", [0.9, 0.95])
    if not isinstance(adamw_betas_raw, (list, tuple)) or len(adamw_betas_raw) != 2:
        raise ValueError("optimizer.muon_adamw.adamw_betas must be a length-2 list.")
    adamw_betas = (float(adamw_betas_raw[0]), float(adamw_betas_raw[1]))
    adamw_eps = _to_float(
        _cfg_value(muon_cfg, "adamw_eps", 1.0e-8), "optimizer.muon_adamw.adamw_eps"
    )
    muon_momentum = _to_float(
        _cfg_value(muon_cfg, "muon_momentum", 0.95), "optimizer.muon_adamw.muon_momentum"
    )
    muon_nesterov = bool(_cfg_value(muon_cfg, "muon_nesterov", True))
    backend_steps = _to_int(
        _cfg_value(muon_cfg, "backend_steps", 5), "optimizer.muon_adamw.backend_steps"
    )

    param_groups = [
        {
            "params": adamw_params,
            "kind": "adamw",
            "lr": adamw_lr,
            "weight_decay": weight_decay,
            "betas": adamw_betas,
            "eps": adamw_eps,
        },
        {
            "params": muon_params,
            "kind": "muon",
            "lr": muon_lr,
            "weight_decay": weight_decay,
            "momentum": muon_momentum,
            "nesterov": muon_nesterov,
            "backend_steps": backend_steps,
        },
    ]

    return param_groups, {
        "using_muon": True,
        "reason": None,
        "num_muon_tensors": len(muon_params),
        "num_adamw_tensors": len(adamw_params),
        "adamw_lr": adamw_lr,
        "muon_lr": muon_lr,
        "muon_lr_multiplier": (muon_lr / max(1e-12, adamw_lr)),
        "backend_steps": backend_steps,
        "muon_nesterov": muon_nesterov,
    }


def build_step1_backbone_optimizer(
    model: torch.nn.Module,
    optimization_config: Dict[str, Any],
    base_learning_rate: float,
    weight_decay: float,
) -> Tuple[Optimizer, Dict[str, Any]]:
    """Build Step1 optimizer: plain AdamW or Muon+AdamW grouped setup."""
    optimizer_cfg = optimization_config.get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adamw")).strip().lower()

    if optimizer_type != "muon_adamw":
        return (
            AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay),
            {"type": "adamw", "using_muon": False},
        )

    muon_cfg = optimizer_cfg.get("muon_adamw", {})
    if not bool(muon_cfg.get("enabled", True)):
        return (
            AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay),
            {"type": "adamw", "using_muon": False, "reason": "muon_disabled"},
        )

    unwrapped = _unwrap_model(model)
    param_groups, info = _build_muon_param_groups(
        model=unwrapped,
        weight_decay=weight_decay,
        base_learning_rate=base_learning_rate,
        muon_cfg=muon_cfg,
    )

    if not info.get("using_muon", False):
        return (
            AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay),
            {"type": "adamw", "using_muon": False, "reason": info.get("reason")},
        )

    return MuonAdamW(param_groups), {"type": "muon_adamw", **info}
