"""Configuration utilities for XyloMaMi-Bench training and evaluation."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    raise ModuleNotFoundError(
        "PyYAML is required for XyloMaMi-Bench experiment configs. "
        "Install it with `pip install pyyaml`."
    ) from exc


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` without mutating either input."""

    merged: dict[str, Any] = deepcopy(dict(base))
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge_dicts(base_value, override_value)
        else:
            merged[key] = deepcopy(override_value)
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file '{path}' must contain a YAML mapping at the top level.")
    return dict(payload)


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_values(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _expand_env_values(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with recursive `base_configs` support."""

    config_path = Path(path).resolve()
    payload = _load_yaml_file(config_path)
    base_configs_raw = payload.pop("base_configs", [])
    if isinstance(base_configs_raw, str):
        base_configs = [base_configs_raw]
    else:
        base_configs = list(base_configs_raw)

    merged: dict[str, Any] = {}
    for base_entry in base_configs:
        base_path = (config_path.parent / str(base_entry)).resolve()
        merged = deep_merge_dicts(merged, load_config(base_path))

    merged = deep_merge_dicts(merged, payload)
    merged = _expand_env_values(merged)
    merged["_meta"] = {
        "config_path": str(config_path),
        "config_name": config_path.stem,
    }
    return merged


def set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using dot notation."""

    path_tokens = [token.strip() for token in dotted_key.split(".") if token.strip()]
    if not path_tokens:
        raise ValueError(f"Invalid override key: '{dotted_key}'.")

    current: dict[str, Any] = config
    for token in path_tokens[:-1]:
        next_value = current.get(token)
        if next_value is None:
            current[token] = {}
            next_value = current[token]
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot set nested override '{dotted_key}': '{token}' is not a mapping."
            )
        current = next_value
    current[path_tokens[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply CLI overrides of the form `a.b.c=value`."""

    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}'. Expected the form key=value."
            )
        key, raw_value = override.split("=", 1)
        parsed_value = yaml.safe_load(raw_value)
        set_nested_value(updated, key, parsed_value)
    return updated


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a config mapping to YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_serializable(config), handle, sort_keys=False, allow_unicode=True)


def to_serializable(value: Any) -> Any:
    """Convert Paths and nested containers into YAML/JSON-friendly primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    return value
