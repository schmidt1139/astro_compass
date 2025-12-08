import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tomli
from utils.path_utils import PROJECT_ROOT

# Keys that should be expanded to absolute paths relative to PROJECT_ROOT
PATH_KEYS = {
    ("paths", "output_dir"),
    ("paths", "path_training_data"),
    ("paths", "path_replay_buffer"),
    ("paths", "path_SAC_model_load"),
    ("paths", "path_SAC_model_save"),
}


def _expand_paths(params: Dict[str, object]) -> Dict[str, object]:
    for section, key in PATH_KEYS:
        sec = params.get(section, {})
        if not isinstance(sec, dict):
            continue
        val = sec.get(key)
        if isinstance(val, str):
            sec[key] = os.path.abspath(os.path.expanduser(val))
    return params


def _merge_dicts(dest: Dict, src: Dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dest.get(k), dict):
            _merge_dicts(dest[k], v)
        else:
            dest[k] = v


def _flatten_paths(prefix: str, obj: object) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_paths(f"{prefix}.{k}" if prefix else k, v))
    else:
        out[prefix] = obj
    return out


def _validate_schema(required: Dict[str, object], cfg: Dict[str, object], path: str = "") -> List[str]:
    missing: List[str] = []
    for key, req_val in required.items():
        cur_path = f"{path}.{key}" if path else key
        if isinstance(req_val, dict):
            if key not in cfg or not isinstance(cfg.get(key), dict):
                missing.append(cur_path)
            else:
                missing.extend(_validate_schema(req_val, cfg[key], cur_path))
        else:
            if key not in cfg:
                missing.append(cur_path)
    return missing


REQUIRED_SCHEMA = {
    "paths": {
        "output_dir": None,
        "path_training_data": None,
        "path_replay_buffer": None,
    },
    "environment": {
        "env_type": None,
        "mu": None,
        "max_T": None,
        "ISP": None,
        "l_star": None,
        "m_star": None,
        "t_star": None,
        "g0": None,
        "env_step_size": None,
    },
    "model": {
        "learning_rate": None,
        "tau": None,
        "train_freq": None,
        "gradient_steps": None,
        "max_grad_norm": None,
        "eval_device": None,
    },
    "training": {
        "training_steps": None,
        "buffer_size": None,
        "batch_size": None,
    },
}


def load_toml(path: Path) -> Dict:
    with path.open("rb") as f:
        return tomli.load(f)


def load_config(
    base_files: Iterable[str],
    experiment_file: str,
    base_dir: Path | None = None,
    allow_new_keys: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Load and merge layered TOML configs preserving hierarchy and validating schema."""

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
        _merge_dicts(merged, data)
        sources.append(str(path))

    exp_path = Path(experiment_file)
    if not exp_path.is_absolute():
        exp_path = base_dir / experiment_file
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    exp_data = load_toml(exp_path)
    if "overwrite" not in exp_data or not isinstance(exp_data["overwrite"], dict):
        raise ValueError("Experiment config must define an [overwrite] table")

    overrides = exp_data["overwrite"]
    flat_base = _flatten_paths("", merged)
    flat_over = _flatten_paths("", overrides)
    unknown = set(flat_over.keys()) - set(flat_base.keys())
    if unknown and not allow_new_keys:
        raise ValueError(
            f"Experiment overrides unknown keys: {sorted(unknown)}. "
            "Add them to a base file first."
        )

    _merge_dicts(merged, overrides)

    missing = _validate_schema(REQUIRED_SCHEMA, merged)
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    merged = _expand_paths(merged)

    overwrite_log: List[Tuple[str, object, object]] = []
    for key, new_val in flat_over.items():
        old_val = flat_base.get(key)
        if old_val != new_val:
            overwrite_log.append((key, old_val, new_val))

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
