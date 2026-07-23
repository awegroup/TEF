"""Shared infrastructure: YAML read/write helpers and project paths.

The TEF project root is resolved from this file's location inside the
``src`` layout, so it works both for editable installs and direct
``sys.path`` use. Set the ``TEF_ROOT`` environment variable to override
(e.g. when the package is installed non-editable).
"""

import os
from pathlib import Path

import numpy as np
import yaml


def _to_builtin(value):
    """Recursively convert numpy scalars/arrays and Paths to plain
    Python types so yaml.safe_dump can represent them."""
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dictionary.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, path: Path) -> None:
    """Write a dictionary to a YAML file, preserving key order.

    Args:
        data (dict): Data to write.
        path (Path): Destination file path. Parent directories are
            created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(_to_builtin(data), f, sort_keys=False,
                       default_flow_style=False)


def project_root() -> Path:
    """Return the TEF project root directory."""
    env_root = os.environ.get('TEF_ROOT')
    if env_root:
        return Path(env_root)
    # .../TEF/src/tef/io.py -> TEF/
    return Path(__file__).resolve().parents[2]


def base_config_dir() -> Path:
    """Return the default base configuration directory."""
    return project_root() / 'config' / 'base'


def studies_config_dir() -> Path:
    """Return the study configuration directory."""
    return project_root() / 'config' / 'studies'


def results_dir() -> Path:
    """Return the default results directory."""
    return project_root() / 'results'
