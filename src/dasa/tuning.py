from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from dasa.config import manifest, validate, write_json
from dasa.data import Demonstration
from dasa.metrics import trial_metrics
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import load_models
from dasa.simulation import run_trial


RUNTIME_PARAMETERS = {
    "intent.beta", "motion.lambda_gain", "motion.distance_min", "motion.distance_max",
    "arbitration.roughness_gamma", "motion.kalman_process", "motion.kalman_measurement",
    "motion.attractive_weight", "motion.repulsive_weight",
}


def _candidate_configs(config: dict, parameters: list[str] | None) -> list[tuple]:
    tuning = config.get("tuning", {})
    if tuning.get("split") != "validation":
        raise ValueError("Sensitivity selection may use only the validation split")
    if tuning.get("primary_metric", "success_rate") != "success_rate" or tuning.get(
            "tie_breakers", ["minimum_clearance_m", "completion_seconds"]) != [
                "minimum_clearance_m", "completion_seconds"]:
        raise ValueError("Expected success, clearance, then successful-completion selection rule")
    grids = tuning.get("grids")
    if not isinstance(grids, dict) or not grids:
        raise ValueError("tuning.grids must be a nonempty mapping of parameter paths to values")
    if isinstance(parameters, str):
        raise ValueError("parameters must be a list of exact dotted parameter paths")
    names = list(grids) if parameters is None else list(parameters)
    if not names or len(names) != len(set(names)) or any(name not in grids for name in names):
        raise ValueError("Selected parameter paths must be distinct configured grid keys")
    result = []
    for parameter_index, name in enumerate(names):
        if name not in RUNTIME_PARAMETERS:
            raise ValueError(f"Grid parameter is not a supported frozen-model runtime setting: {name}")
        values = grids[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid must contain at least one numeric value: {name}")
        section, field = name.split(".")
        if field not in config[section]:
            raise ValueError(f"Grid parameter has no explicit base value: {name}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not np.isfinite(value) for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Grid values must be distinct finite numbers: {name}")
        for value_index, value in enumerate(values):
            candidate = copy.deepcopy(config)
            candidate[section][field] = value
            validate(candidate)
            if value < 0 or (name in {"motion.lambda_gain", "motion.distance_max",
                                     "motion.kalman_measurement"} and value == 0):
                raise ValueError(f"Invalid runtime parameter value: {name}={value}")
            result.append((f"p{parameter_index:02d}_v{value_index:02d}", name, value, candidate))
    return result


def _validation_data(root: Path) -> tuple[list[Demonstration], list[Path], dict]:

    splits_path = root / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    paths, owner = {}, {}
    for name in ("train", "validation", "test"):
        if not isinstance(splits.get(name), list):
            raise ValueError(f"Missing explicit dataset split: {name}")
        paths[name] = []
        for relative in splits[name]:
            if not isinstance(relative, str):
                raise ValueError("Dataset split entries must be relative path strings")
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or path.suffix != ".npz":
                raise ValueError("Dataset split path escapes its root or is not an NPZ recording")
            if path in owner:
                raise ValueError("Dataset split paths overlap or are duplicated")
            owner[path] = name
            paths[name].append(path)
    if not paths["validation"]:
        raise ValueError("Validation split is empty")
    demos = [Demonstration.load(path) for path in paths["validation"]]
    ids = [demo.metadata["episode_id"] for demo in demos]
    if len(set(ids)) != len(ids):
        raise ValueError("Validation episode IDs are duplicated")
    return demos, [splits_path, *paths["validation"]], {
        "validation_episode_ids": ids,
        "loaded_demonstration_splits": ["validation"],
        "train_test_path_overlap_checked": True,
        "train_test_signals_opened": False,
        "operator_partition_scope": "requires the prepared dataset partition audit; test operators not inspected"}


def _summary(rows: list[dict]) -> dict:

    units = sorted({row["unit"] for row in rows})
    summaries = []
    for unit in units:
        records = [row for row in rows if row["unit"] == unit]
        completed = [row["completion_seconds"] for row in records
                     if row["success_rate"] == 1 and row["completion_seconds"] is not None]
        summaries.append({"unit": unit, "n_trials": len(records),
                          "success_rate": float(np.mean([row["success_rate"] for row in records])),
                          "minimum_clearance_m": float(np.mean([
                              row["minimum_clearance_m"] for row in records])),
                          "completion_seconds": float(np.mean(completed)) if completed else None})
    completions = [row["completion_seconds"] for row in summaries if row["completion_seconds"] is not None]
    return {"n_units": len(units), "n_trials": len(rows),
            "n_successful_trials": sum(row["success_rate"] == 1 for row in rows),
            "success_rate": float(np.mean([row["success_rate"] for row in summaries])),
            "minimum_clearance_m": float(np.mean([row["minimum_clearance_m"] for row in summaries])),
            "completion_seconds": float(np.mean(completions)) if completions else None,
            "unit_summaries": summaries}


def sweep(config: dict, prepared_root: Path, output: Path,
          parameters: list[str] | None = None) -> list[dict]:





    prepared_root, output = Path(prepared_root), Path(output)
    validate(config)
    candidates = _candidate_configs(config, parameters)
    demos, input_files, split_audit = _validation_data(prepared_root / "dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation sweep output must be empty")
    model_directory = prepared_root / "models"
    model_files = [model_directory / "bootstrap_hsmm.npz", *sorted(model_directory.glob("*.pt"))]
    audit_file = model_directory / "training-audit.json"
    if audit_file.exists():
        model_files.append(audit_file)
    hashes_before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in model_files}
    skill = TPHSMM.load(config, model_directory / "bootstrap_hsmm.npz")
    cvae, _ = load_models(config, model_directory)
    scenes = [{"validation_index": index, "episode_id": demo.metadata["episode_id"],
               "source_operator_id": demo.metadata["operator_id"], "scene": demo.scene.to_dict(),
               "goal": demo.goal, "operator_seed": config["seed"] + 900000 + index,
               "noisy_simulated_operator": bool(index % 2)} for index, demo in enumerate(demos)]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation-scenes.json", scenes)
    rows, summaries = [], []
    for candidate_id, parameter, value, candidate in candidates:
        candidate_rows = []
        for demo, entry in zip(demos, scenes):
            trial_id = f"validation_{candidate_id}_e{entry['validation_index']:04d}"
            trial = run_trial(candidate, demo.scene, skill, cvae, entry["operator_seed"],
                              demo.goal, "C3", f"validation_group_{demo.metadata['operator_id']}",
                              noisy=entry["noisy_simulated_operator"], trial_id=trial_id)
            trial.save(output / "trials" / f"{trial_id}.npz")
            record = {**trial_metrics(trial, candidate), "candidate_id": candidate_id,
                      "parameter": parameter, "value": value, "split": "validation",
                      "source_episode_id": entry["episode_id"],
                      "source_operator_id": entry["source_operator_id"],
                      "simulated_operator_noisy": entry["noisy_simulated_operator"],
                      "termination": trial.metadata["termination"],
                      "failure_retained": not trial.metadata["success"]}
            rows.append(record)
            candidate_rows.append(record)
        summaries.append({"candidate_id": candidate_id, "parameter": parameter, "value": value,
                          **_summary(candidate_rows)})
    rankings, selections = [], []
    for parameter in dict.fromkeys(candidate[1] for candidate in candidates):
        section, field = parameter.split(".")
        base = config[section][field]
        selected = [row for row in summaries if row["parameter"] == parameter]
        selected.sort(key=lambda row: (-row["success_rate"], -row["minimum_clearance_m"],
                                       row["completion_seconds"] if row["completion_seconds"] is not None
                                       else np.inf, abs(row["value"] - base),
                                       row["candidate_id"]))
        rankings.extend({"rank": index + 1, **row} for index, row in enumerate(selected))
        selections.append({"parameter": parameter, "base_value": base,
                           "base_value_in_grid": any(row["value"] == base for row in selected),
                           "selected_validation_value": selected[0]["value"],
                           "selected_candidate_id": selected[0]["candidate_id"],
                           "status": "prospective_validation_selection_only",
                           "combined_configuration_validated": False,
                           "historical_default_optimized": False})
    hashes_after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in model_files}
    if hashes_after != hashes_before:
        raise RuntimeError("Prepared model artifacts changed during the validation sweep")
    write_json(output / "metrics.json", rows)
    write_json(output / "ranking.json", rankings)
    write_json(output / "selection.json", {
        "selection_split": "validation", "design": "one_at_a_time_not_cartesian",
        "ranking_rule": ["unit_mean_success_descending", "unit_mean_minimum_clearance_descending",
                         "mean_successful_completion_ascending", "distance_to_base_ascending",
                         "grid_order"],
        "completion_scope": "successful trials only; absent if no successes; tertiary tie breaker",
        "unit_scope": "source-validation-operator groups; responses are simulated, not participant observations",
        "failure_rule": "retain every collision, bounds violation, wrong-object and timeout trial",
        "no_automatic_test_evaluation": True, "selections": selections, **split_audit})
    run_manifest = manifest(config, "validation-sweep", [*input_files, *model_files])
    run_manifest.update({"frozen_model_sha256": hashes_before, "models_unchanged": True,
                         "training_objective": cvae.training_metadata["objective"],
                         "used_parameter_paths": list(dict.fromkeys(row[1] for row in candidates)),
                         "test_access": "none; split filenames inspected only to reject path overlap",
                         "validation_scenes_sha256": hashlib.sha256(
                             (output / "validation-scenes.json").read_bytes()).hexdigest()})
    write_json(output / "manifest.json", run_manifest)
    return rows
