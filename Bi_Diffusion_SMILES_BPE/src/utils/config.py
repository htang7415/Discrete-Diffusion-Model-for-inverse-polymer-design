"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
