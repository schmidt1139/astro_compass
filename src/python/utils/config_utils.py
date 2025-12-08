import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tomli
from utils.path_utils import PROJECT_ROOT

# Keys that should be expanded to absolute paths relative to PROJECT_ROOT
PATH_KEYS = {
    "output_dir",
    "path_training_data",
    "path_replay_buffer",
    "path_SAC_model_load",
    "path_SAC_model_save",
}


def _flatten_to_last_key(data: Dict) -> Dict[str, object]:
    """Flatten a nested dict, keeping only the final key segment.

    This mirrors the legacy behavior of read_toml_config_file so callers
    continue to use flat param names like "training_steps" even when the
    source TOML is nested.
    """

    flat: Dict[str, object] = {}

    def _recurse(prefix: str | None, obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _recurse(k if prefix is None else f"{prefix}.{k}", v)
        else:
            key = prefix.split(".")[-1] if isinstance(prefix, str) else prefix
            flat[key] = obj

    _recurse(None, data)
    return flat


def _expand_paths(params: Dict[str, object]) -> Dict[str, object]:
    for key in PATH_KEYS:
        val = params.get(key)
        if isinstance(val, str):
            params[key] = os.path.abspath(os.path.expanduser(val))
    return params


def load_toml(path: Path) -> Dict:
    with path.open("rb") as f:
        return tomli.load(f)


def load_config(
    base_files: Iterable[str],
    experiment_file: str,
    base_dir: Path | None = None,
    allow_new_keys: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Load and merge layered TOML configs.

    - base_files: sequence of TOML files merged in order (later wins).
    - experiment_file: TOML with an [overwrite] table whose keys must already
      exist in the merged base. Unknown keys raise unless allow_new_keys=True.
    Returns (params, meta) where meta captures sources and overwrites.
    """

    if experiment_file is None:
        raise ValueError("experiment_file must be provided explicitly")

    base_dir = base_dir or PROJECT_ROOT / "data" / "config"
    merged: Dict[str, object] = {}
    sources: List[str] = []

    for fname in base_files:
        path = Path(fname)
        if not path.is_absolute():
            path = base_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Base config not found: {path}")
        data = load_toml(path)
        merged.update(_flatten_to_last_key(data))
        sources.append(str(path))

    exp_path = Path(experiment_file)
    if not exp_path.is_absolute():
        exp_path = base_dir / experiment_file
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    exp_data = load_toml(exp_path)
    if "overwrite" not in exp_data or not isinstance(exp_data["overwrite"], dict):
        raise ValueError("Experiment config must define an [overwrite] table")

    overrides = _flatten_to_last_key(exp_data["overwrite"])
    unknown = set(overrides.keys()) - set(merged.keys())
    if unknown and not allow_new_keys:
        raise ValueError(
            f"Experiment overrides unknown keys: {sorted(unknown)}. "
            "Add them to a base file first."
        )

    overwrite_log: List[Tuple[str, object, object]] = []
    for key, new_val in overrides.items():
        old_val = merged.get(key)
        if key in merged and old_val != new_val:
            overwrite_log.append((key, old_val, new_val))
        merged[key] = new_val

    merged = _expand_paths(merged)

    meta = {
        "sources": sources,
        "experiment": str(exp_path),
        "overwrites": overwrite_log,
    }
    return merged, meta


def write_config_sources(meta: Dict[str, object], output_dir: Path) -> None:
    """Persist the list of config sources and overwrites for traceability."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = meta.get("sources", [])
    experiment = meta.get("experiment")
    overwrites = meta.get("overwrites", [])

    with (output_dir / "config_sources.txt").open("w") as f:
        f.write("Config sources (in merge order):\n")
        for src in sources:
            f.write(f"- {src}\n")
        if experiment:
            f.write(f"Experiment: {experiment}\n")

        if overwrites:
            f.write("\nApplied overwrites:\n")
            for key, old, new in overwrites:
                f.write(f"- {key}: {old} -> {new}\n")


__all__ = ["load_config", "write_config_sources"]
