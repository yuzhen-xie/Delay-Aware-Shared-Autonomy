"""Configuration loading, validation and experiment provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, override: str | Path | None = None) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if override:
        with open(override, encoding="utf-8") as stream:
            config = merge(config, yaml.safe_load(stream))
    validate(config)
    return config


def validate(c: dict) -> None:
    sim = c["simulation"]
    dt, decision = sim["control_dt"], sim["decision_dt"]
    if dt <= 0 or decision < dt or not np.isclose(decision / dt, round(decision / dt)):
        raise ValueError("decision_dt must be a positive integer multiple of control_dt")
    if sim["mass"] <= 0 or sim["damping"] < 0:
        raise ValueError("Invalid plant mass or damping")
    for field in ("feedback_delay", "command_delay", "jitter_seconds"):
        if sim[field] < 0:
            raise ValueError(f"{field} must be nonnegative")
    if c["motion"]["distance_max"] <= c["motion"]["distance_min"]:
        raise ValueError("Collision-distance thresholds must be ordered")
    p = c["passivity"]
    if not 0 <= p["initial_energy"] <= p["maximum_energy"]:
        raise ValueError("Invalid initial energy budget")
    if not 0 <= p["input_efficiency"] <= 1:
        raise ValueError("Energy input efficiency must lie in [0, 1]")
    goals = np.asarray(c["scene"]["goals"], dtype=float)
    if goals.shape != (3, 3) or not np.isfinite(goals).all():
        raise ValueError("This benchmark requires three finite 3D candidate goals")
    if c["intent"]["hidden_dim"] % c["intent"]["transformer_heads"]:
        raise ValueError("Transformer dimension must be divisible by its head count")
    if c["intent"]["deviation_objective"] not in {"auxiliary", "joint"}:
        raise ValueError("Unknown deviation loss objective")
    if c["motion"]["risk_distance"] not in {"center", "surface"}:
        raise ValueError("Risk distance must be center or surface")
    if c["skill"]["trajectory_samples"] < c["skill"]["states"]:
        raise ValueError("Each HSMM state needs at least one time sample")
    if c["intent"]["epochs"] < 1:
        raise ValueError("At least one training epoch is required")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False), encoding="utf-8")


def manifest(config: dict, command: str, inputs: list[Path] | None = None) -> dict:
    encoded = json.dumps(config, sort_keys=True).encode()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {
        "command": command, "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "data_provenance": "synthetic_task_space_simulation",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs or []},
    }
