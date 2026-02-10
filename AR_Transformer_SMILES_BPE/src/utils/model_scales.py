"""Model scale definitions for GPT-style diffusion backbones.

This module provides model architecture and training configurations for
scaling law experiments with 4 model sizes: small, medium, large, xl.
"""

from typing import Dict, Any, Optional

# Valid model sizes
VALID_SIZES = ['small', 'medium', 'large', 'xl']

# Sequence model scales (for p-SMILES, SELFIES, Group SELFIES)
SEQUENCE_MODEL_SCALES = {
    'small': {
        # Architecture (~12M params)
        'hidden_size': 384,
        'num_layers': 6,
        'num_heads': 6,
        'ffn_hidden_size': 1536,
        'dropout': 0.1,
        'max_position_embeddings': 256,
        # Training
        'max_steps': 100000,
        'warmup_steps': 1000,
        'batch_size': 256,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3.0e-4,
    },
    'medium': {
        # Architecture (~50M params)
        'hidden_size': 640,
        'num_layers': 10,
        'num_heads': 10,
        'ffn_hidden_size': 2560,
        'dropout': 0.1,
        'max_position_embeddings': 256,
        # Training
        'max_steps': 200000,
        'warmup_steps': 2000,
        'batch_size': 256,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3.0e-4,
    },
    'large': {
        # Architecture (~150M params)
        'hidden_size': 960,
        'num_layers': 14,
        'num_heads': 12,
        'ffn_hidden_size': 3840,
        'dropout': 0.1,
        'max_position_embeddings': 256,
        # Training
        'max_steps': 300000,
        'warmup_steps': 3000,
        'batch_size': 128,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1.0e-4,
    },
    'xl': {
        # Architecture (~400M params)
        'hidden_size': 1280,
        'num_layers': 20,
        'num_heads': 16,
        'ffn_hidden_size': 5120,
        'dropout': 0.1,
        'max_position_embeddings': 256,
        # Training
        'max_steps': 400000,
        'warmup_steps': 4000,
        'batch_size': 64,
        'gradient_accumulation_steps': 16,
        'learning_rate': 1.0e-4,
    },
}

# Graph model scales (adjusted for edge attention overhead)
GRAPH_MODEL_SCALES = {
    'small': {
        # Architecture (~14M params)
        'hidden_size': 384,
        'num_layers': 6,
        'num_heads': 6,
        'ffn_hidden_size': 1536,
        'dropout': 0.1,
        # Training
        'max_steps': 100000,
        'warmup_steps': 1000,
        'batch_size': 256,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3.0e-4,
    },
    'medium': {
        # Architecture (~55M params)
        'hidden_size': 640,
        'num_layers': 10,
        'num_heads': 10,
        'ffn_hidden_size': 2560,
        'dropout': 0.1,
        # Training
        'max_steps': 200000,
        'warmup_steps': 2000,
        'batch_size': 256,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3.0e-4,
    },
    'large': {
        # Architecture (~140M params)
        'hidden_size': 896,
        'num_layers': 14,
        'num_heads': 8,
        'ffn_hidden_size': 3584,
        'dropout': 0.1,
        # Training
        'max_steps': 300000,
        'warmup_steps': 3000,
        'batch_size': 128,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1.0e-4,
    },
    'xl': {
        # Architecture (~350M params)
        'hidden_size': 1152,
        'num_layers': 18,
        'num_heads': 16,
        'ffn_hidden_size': 4608,
        'dropout': 0.1,
        # Training
        'max_steps': 400000,
        'warmup_steps': 4000,
        'batch_size': 64,
        'gradient_accumulation_steps': 16,
        'learning_rate': 1.0e-4,
    },
}


def get_model_config(
    model_size: Optional[str],
    config: Dict[str, Any],
    model_type: str = 'sequence'
) -> Dict[str, Any]:
    """Get model architecture configuration based on size argument.

    Args:
        model_size: One of 'small', 'medium', 'large', 'xl', or None.
                   If None, uses config['backbone'] directly.
        config: Full configuration dictionary.
        model_type: 'sequence' or 'graph'.

    Returns:
        Dictionary with model architecture hyperparameters.

    Raises:
        ValueError: If model_size is invalid.
    """
    if model_size is None:
        # Use existing backbone config
        return config['backbone'].copy()

    if model_size not in VALID_SIZES:
        raise ValueError(
            f"Invalid model_size '{model_size}'. "
            f"Must be one of: {VALID_SIZES}"
        )

    # Priority: 1) config['model_sizes'] if present, 2) built-in scales
    if 'model_sizes' in config and model_size in config['model_sizes']:
        scale_config = config['model_sizes'][model_size].copy()
    else:
        # Use built-in scales
        if model_type == 'graph':
            scale_config = GRAPH_MODEL_SCALES[model_size].copy()
        else:
            scale_config = SEQUENCE_MODEL_SCALES[model_size].copy()

    # Extract only architecture params
    arch_keys = ['hidden_size', 'num_layers', 'num_heads', 'ffn_hidden_size',
                 'dropout', 'max_position_embeddings']
    return {k: scale_config[k] for k in arch_keys if k in scale_config}


def get_training_config(
    model_size: str,
    config: Dict[str, Any],
    model_type: str = 'sequence'
) -> Dict[str, Any]:
    """Get training hyperparameters for a model size.

    Args:
        model_size: One of 'small', 'medium', 'large', 'xl'.
        config: Full configuration dictionary.
        model_type: 'sequence' or 'graph'.

    Returns:
        Dictionary with training hyperparameters.
    """
    # Priority: 1) config['model_sizes'] if present, 2) built-in scales
    if 'model_sizes' in config and model_size in config['model_sizes']:
        scale_config = config['model_sizes'][model_size]
    else:
        if model_type == 'graph':
            scale_config = GRAPH_MODEL_SCALES[model_size]
        else:
            scale_config = SEQUENCE_MODEL_SCALES[model_size]

    # Extract only training params
    training_keys = ['max_steps', 'warmup_steps', 'batch_size',
                     'gradient_accumulation_steps', 'learning_rate']
    return {k: scale_config[k] for k in training_keys if k in scale_config}


def estimate_params(
    model_config: Dict[str, Any],
    vocab_size: int,
    model_type: str = 'sequence',
    num_diffusion_steps: int = 50
) -> int:
    """Estimate number of parameters for a model configuration.

    Args:
        model_config: Model hyperparameters dictionary.
        vocab_size: Vocabulary size.
        model_type: 'sequence' or 'graph'.
        num_diffusion_steps: Number of diffusion timesteps.

    Returns:
        Estimated parameter count.
    """
    h = model_config['hidden_size']
    L = model_config['num_layers']
    ffn = model_config['ffn_hidden_size']
    max_pos = model_config.get('max_position_embeddings', 256)

    # Embeddings
    token_emb = vocab_size * h
    pos_emb = max_pos * h
    time_emb = (num_diffusion_steps + 1) * h

    # Transformer layers: 4 projections (Q,K,V,O) + 2 FFN + LayerNorms
    attn_params = 4 * h * h  # Q, K, V, O projections
    ffn_params = 2 * h * ffn  # fc1, fc2
    ln_params = 4 * h  # 2 LayerNorms per layer
    layer_params = L * (attn_params + ffn_params + ln_params)

    # Output
    final_ln = 2 * h
    output_proj = vocab_size * h

    total = token_emb + pos_emb + time_emb + layer_params + final_ln + output_proj

    if model_type == 'graph':
        # Add edge attention embeddings (edge_vocab * num_heads per layer)
        edge_vocab = 6  # Typical edge vocab size
        num_heads = model_config['num_heads']
        edge_attn_emb = L * edge_vocab * num_heads

        # Edge prediction head: 2h -> h -> edge_vocab
        edge_head = 2 * h * h + h * edge_vocab
        total += edge_attn_emb + edge_head

    return total


def get_results_dir(model_size: Optional[str], base_dir: str = 'results') -> str:
    """Get results directory path based on model size.

    Args:
        model_size: One of 'small', 'medium', 'large', 'xl', or None.
        base_dir: Base results directory name.

    Returns:
        Results directory path string.
    """
    if model_size is None:
        return base_dir
    return f"{base_dir}_{model_size}"


def print_model_info(model_size: str, model_config: Dict[str, Any],
                     training_config: Dict[str, Any], vocab_size: int,
                     model_type: str = 'sequence'):
    """Print model and training configuration info.

    Args:
        model_size: Model size name.
        model_config: Model architecture config.
        training_config: Training config.
        vocab_size: Vocabulary size.
        model_type: 'sequence' or 'graph'.
    """
    num_params = estimate_params(model_config, vocab_size, model_type)

    print(f"\nModel Size: {model_size.upper()}")
    print(f"  Architecture:")
    print(f"    hidden_size: {model_config['hidden_size']}")
    print(f"    num_layers: {model_config['num_layers']}")
    print(f"    num_heads: {model_config['num_heads']}")
    print(f"    ffn_hidden_size: {model_config['ffn_hidden_size']}")
    print(f"    Estimated parameters: {num_params:,}")
    print(f"  Training:")
    print(f"    max_steps: {training_config['max_steps']:,}")
    print(f"    batch_size: {training_config['batch_size']}")
    print(f"    gradient_accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"    effective_batch: {training_config['batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"    learning_rate: {training_config['learning_rate']}")
