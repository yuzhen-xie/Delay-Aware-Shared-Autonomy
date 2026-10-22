from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from dasa.config import write_json
from dasa.data import Demonstration


DEFAULT_GRID = {"intent.beta": [1., 3., 5.], "motion.lambda_gain": [5., 10., 15.],
                "arbitration.roughness_gamma": [.006, .012, .024]}


def _set(config, path, value):
    names = path.split(".")
    target = config
    for name in names[:-1]:
        target = target[name]
    if names[-1] not in target:
        raise ValueError(f"Unknown tuning parameter: {path}")
    target[names[-1]] = value


def plan(config, prepared, output, parameters=None):
    prepared, output = Path(prepared), Path(output)
    split_path = prepared / "dataset/splits.json"
    if not split_path.is_file():
        split_path = prepared / "splits.json"
    splits = json.loads(split_path.read_text())
    validation = [Demonstration.load(split_path.parent / name) for name in splits["validation"]]
    if not validation:
        raise ValueError("Measured validation episodes are required")
    requested = parameters or ["intent.beta"]
    grid = config.get("tuning", {}).get("grid", DEFAULT_GRID)
    if any(name not in grid for name in requested):
        raise ValueError("Every requested parameter needs an explicitly configured candidate grid")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("A validation plan requires a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    candidates, rows = [], []
    for index, values in enumerate(itertools.product(*(grid[name] for name in requested))):
        candidate = copy.deepcopy(config)
        settings = dict(zip(requested, values))
        for key, value in settings.items():
            _set(candidate, key, value)
        identifier = f"candidate_{index:03d}"
        path = output / (identifier + ".yaml")
        path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        candidates.append({"candidate_id": identifier, "settings": settings,
                           "config_file": path.name, "config_sha256": digest})
        for trial, demo in enumerate(validation):
            rows.append({"trial_id": f"validation_{identifier}_{trial:04d}",
                         "candidate_id": identifier, "participant_id": demo.metadata["operator_id"],
                         "condition": "C3", "delay_seconds": config["task"]["feedback_delay"],
                         "command_delay_seconds": 0., "intended_goal": demo.goal,
                         "stage": "parameter_validation", "record_status": "planned",
                         "config_file": str(path.resolve()), "config_sha256": digest,
                         "matched_validation_episode": demo.metadata["episode_id"],
                         "timeout_seconds": config["task"]["max_seconds"]})
    order = np.random.default_rng(config["seed"]).permutation(len(rows))
    rows = [{**rows[int(index)], "order": j} for j, index in enumerate(order)]
    with (output / "schedule.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "plan.json", {"record_status": "prospective", "candidates": candidates,
        "validation_operators": sorted({d.metadata["operator_id"] for d in validation}),
        "planned_trials": rows, "selection": "minimum_mean_timeout_penalized_seconds",
        "tie_break": "candidate_id_lexicographic", "test_records_opened": False})
    return rows


def select(config, plan_path, trials_path, output):
    plan_path, output = Path(plan_path), Path(output)
    specification = json.loads(plan_path.read_text())
    planned = {row["trial_id"]: row for row in specification["planned_trials"]}
    with Path(trials_path).open(newline="", encoding="utf-8-sig") as stream:
        measured = list(csv.DictReader(stream))
    if len(measured) != len(planned) or len({r["trial_id"] for r in measured}) != len(measured):
        raise ValueError("Selection requires every planned validation trial once")
    scores = {}
    for row in measured:
        assignment = planned.get(row["trial_id"])
        if (assignment is None or row.get("provenance") != "UR10_measured_teleoperation"
                or row["participant_id"] != assignment["participant_id"]):
            raise ValueError("Validation observation does not match the physical validation plan")
        if row.get("config_sha256") != assignment["config_sha256"]:
            raise ValueError("Observed parameter configuration is not bound to the planned configuration")
        success = row["success"].strip().lower() in {"true", "1"}
        value = float(row["completion_seconds"]) if success else float(assignment["timeout_seconds"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid observed validation outcome")
        scores.setdefault(assignment["candidate_id"], []).append(value)
    summaries = [{"candidate_id": name, "mean_timeout_penalized_seconds": float(np.mean(values)),
                  "n_trials": len(values)} for name, values in scores.items()]
    chosen = min(summaries, key=lambda row: (row["mean_timeout_penalized_seconds"], row["candidate_id"]))
    selected = next(c for c in specification["candidates"] if c["candidate_id"] == chosen["candidate_id"])
    source = plan_path.parent / selected["config_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != selected["config_sha256"]:
        raise ValueError("Frozen candidate configuration changed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Selection output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected.yaml").write_bytes(source.read_bytes())
    write_json(output / "selection.json", {"selected": selected, "candidate_scores": summaries,
        "input_sha256": hashlib.sha256(Path(trials_path).read_bytes()).hexdigest(),
        "provenance": "UR10_measured_teleoperation", "test_records_opened": False})
    return selected
