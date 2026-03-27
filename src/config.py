"""Configuration loader.

Loads settings from config/settings.yaml and provides typed access.
"""

from pathlib import Path
from typing import Any, Optional

import yaml


_config: Optional[dict] = None


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from YAML file.

    Args:
        path: Path to settings.yaml. If None, uses default location.

    Returns:
        Configuration dictionary
    """
    global _config

    if path is None:
        path = str(Path(__file__).parent.parent / "config" / "settings.yaml")

    with open(path, "r") as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> dict:
    """Get the loaded configuration, loading defaults if needed."""
    global _config
    if _config is None:
        return load_config()
    return _config


def get(section: str, key: str, default: Any = None) -> Any:
    """Get a specific config value.

    Args:
        section: Top-level section (e.g., 'board', 'preprocessing')
        key: Key within section (e.g., 'board_id')
        default: Default value if key not found

    Returns:
        Config value
    """
    cfg = get_config()
    return cfg.get(section, {}).get(key, default)
