from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import uuid

import numpy as np
import yaml

from dasa.config import write_json
from dasa.control.arbitration import CONDITIONS
from dasa.geometry import Scene
from dasa.robotics.session import run_workstation


def _load_hardware(path):
    from dasa.robotics.configuration import load_hardware
    return load_hardware(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _resolve(value, parent):
    path = Path(value).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _identifier(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError(f"{field} must be a plain anonymous identifier")
    return value


def _finite(value, field, positive=False):
    parsed = float(value)
    if not np.isfinite(parsed) or (parsed <= 0 if positive else parsed < 0):
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'nonnegative'}")
    return parsed


def _integer(value, field):
    try:
        parsed = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be an integer")
    return parsed


def _scene_delta(left, right):
    if left["frame"] != right["frame"] or left["obstacle_ids"] != right["obstacle_ids"]:
        return float("inf")
    differences = []
    for key in ("goals", "place", "start", "centers", "radii"):
        a, b = np.asarray(left[key], float), np.asarray(right[key], float)
        if a.shape != b.shape:
            return float("inf")
        if a.size:
            differences.append(float(np.max(np.abs(a - b))) if key == "radii"
                               else float(np.max(np.linalg.norm((a - b).reshape(-1, 3), axis=1))))
    return max(differences, default=0.)


def _append(path, event):
    encoded = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ledger_rows(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except json.JSONDecodeError as exc:
        raise ValueError("Execution ledger is incomplete; preserve it and reconcile the interrupted acquisition") from exc


def _software():
    package = Path(__file__).resolve().parents[1]
    root = package.parent.parent
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True)
    return {"python": platform.python_version(), "numpy": np.__version__,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "source_sha256": {str(path.relative_to(package)): _sha(path)
                              for path in sorted(package.rglob("*.py"))}}


def _matched_scene(rows, assignment, tolerance):
    maximum, matched = 0., []
    if assignment["scene_seed"] is None:
        return {"matched_acquisition_ids": [], "maximum_geometry_difference_m": None}
    for event in rows:
        same_layout_group = bool(assignment.get("layout_group")) and event.get("layout_group") == assignment["layout_group"]
        same_person_seed = (event.get("participant_id") == assignment["participant_id"]
                            and event.get("scene_seed") == assignment["scene_seed"])
        if (event.get("event") != "recorded" or not event.get("execute")
                or not (same_layout_group or same_person_seed)):
            continue
        difference = _scene_delta(event["scene"], assignment["scene"])
        if difference > tolerance:
            raise ValueError("Measured layout differs from the matched condition scene beyond the allowed tolerance")
        maximum = max(maximum, difference)
        matched.append(event["acquisition_id"])
    return {"matched_acquisition_ids": matched,
            "maximum_geometry_difference_m": maximum if matched else None}


def _validate_recording(recording, assignment, execute):
    if (recording.get("provenance") != "UR10_workstation_measurement"
            or recording.get("execute") is not execute or recording.get("condition") != assignment["condition"]):
        raise ValueError("Workstation recording mode, condition or measured provenance differs from the assignment")
    timing = recording.get("timing", {})
    if (not np.isclose(float(timing.get("feedback_delay_seconds", -1)), assignment["feedback_delay_seconds"])
            or not np.isclose(float(timing.get("command_delay_seconds", -1)), 0.)):
        raise ValueError("Recorded channel delays differ from the assigned experiment")
    if _scene_delta(recording["scene"], assignment["scene"]) > 1e-12:
        raise ValueError("Recorded scene differs from the frozen measured scene")
    samples = recording.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Acquisition produced no measured samples")
    times = np.asarray([sample["time"] for sample in samples], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Measured sample times must be finite and strictly increasing")
    for sample in samples:
        for name in ("position", "velocity", "raw_force"):
            signal = np.asarray(sample[name], float)
            if signal.shape != (3,) or not np.isfinite(signal).all():
                raise ValueError(f"Invalid measured signal: {name}")
        if not execute and sample.get("executed") is True:
            raise ValueError("Read-only acquisition contains an executed motion command")
    return len(samples), any(sample.get("executed") is True for sample in samples)


def _recording_assignment(assignment, common):
    fields = ("trial_id", "participant_id", "intended_goal", "scene_seed", "scene_id", "acquisition_id",
              "practice", "planned_order", "timeout_seconds", "robot_radius_m", "stage", "policy_sha256",
              "training_repeat", "training_condition", "refinement_budget", "scenario", "layout_group",
              "candidate_id", "candidate_config_sha256", "config_sha256", "schedule_sha256",
              "actor_type", "evaluation_index", "matched_block_id", "matched_block_index",
              "within_block_order", "randomization_seed")
    return {**{key: assignment[key] for key in fields if key in assignment},
            "assignment_sha256": common.get("assignment_sha256")}


def run_experiment(config, schedule_csv, hardware_path, models, output, execute=False,
                   participant=None, trial_id=None, skill_path=None):
    if not isinstance(execute, bool):
        raise ValueError("execute must be boolean")
    schedule_csv, output = Path(schedule_csv).resolve(), Path(output).resolve()
    models = Path(models).resolve() if models is not None else None
    with schedule_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [row.get("trial_id") for row in rows]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Schedule requires unique, nonempty trial IDs")
    selected = [row for row in rows if (participant is None or row.get("participant_id") == participant)
                and (trial_id is None or row.get("trial_id") == trial_id)]
    if len(selected) != 1:
        raise ValueError("Select exactly one planned trial with participant and trial ID; batches never execute automatically")
    row = selected[0]
    if models is None and row.get("models_dir"):
        models = _resolve(row["models_dir"], schedule_csv.parent)
    candidate_config_sha256 = None
    if row.get("config_file"):
        from dasa.config import load_config
        candidate_path = _resolve(row["config_file"], schedule_csv.parent)
        candidate_config_sha256 = _sha(candidate_path)
        if candidate_config_sha256 != row["config_sha256"]:
            raise ValueError("The frozen validation configuration changed")
        config = load_config(candidate_path)
    trial = _identifier(row["trial_id"], "trial_id")
    identifier = _identifier(row["participant_id"], "participant_id")
    if row.get("record_status", "planned") not in {"planned", "prospective"}:
        raise ValueError("Only a planned schedule row may start a new acquisition")
    condition = row["condition"]
    if condition not in CONDITIONS:
        raise ValueError("Unsupported participant control condition")
    stage = row.get("stage") or "human_shared_control"
    if condition == "autonomous" and stage != "autonomous_evaluation":
        raise ValueError("Autonomous control is restricted to explicitly assigned physical policy evaluations")
    if models is None and condition != "C1":
        raise ValueError("Model-free acquisition is available only for manual demonstrations")
    delay = _finite(row["delay_seconds"], "delay_seconds")
    if _finite(row.get("command_delay_seconds") or 0, "command_delay_seconds") != 0:
        raise ValueError("The measured workstation implements zero command-channel buffering")
    intended_goal = _integer(row["intended_goal"], "intended_goal")
    seed = _integer(row["scene_seed"], "scene_seed") if row.get("scene_seed") else None
    if skill_path is not None:
        selected_skill = Path(skill_path).resolve()
    elif row.get("policy_file") or row.get("skill_file"):
        selected_skill = _resolve(row.get("policy_file") or row["skill_file"], schedule_csv.parent)
    else:
        selected_skill = models / "bootstrap_hsmm.npz" if models is not None else None
    if selected_skill is not None and not selected_skill.is_file():
        raise ValueError("The selected skill checkpoint must be an existing file")
    selected_skill_sha256 = _sha(selected_skill) if selected_skill is not None else None
    if row.get("policy_sha256") and row["policy_sha256"] != selected_skill_sha256:
        raise ValueError("The selected policy differs from the scheduled frozen checkpoint")
    hardware_source = _resolve(row["hardware_file"], schedule_csv.parent) if row.get("hardware_file") \
        else Path(hardware_path).resolve()
    hardware = copy.deepcopy(_load_hardware(hardware_source))
    scene = Scene.from_dict(hardware["scene"]).to_dict()
    if not 0 <= intended_goal < len(scene["goals"]):
        raise ValueError("intended_goal is outside the measured candidate set")
    if row.get("scene_file") or row.get("planned_scene_file"):
        expected = _resolve(row.get("planned_scene_file") or row["scene_file"], schedule_csv.parent)
        document = json.loads(expected.read_text(encoding="utf-8"))
        if _scene_delta(document.get("scene", document), scene) > config["study"].get("hardware_scene_tolerance_m", .005):
            raise ValueError("Measured geometry differs from the scheduled physical layout")
    if row.get("scene_id") and hardware.get("scene_id") and row["scene_id"] != hardware["scene_id"]:
        raise ValueError("Schedule scene_id is not bound to this hardware configuration")
    timeout = _finite(row.get("timeout_seconds") or config.get("study", {}).get(
        "max_seconds", hardware["maximum_seconds"]), "timeout_seconds", positive=True)
    hardware["feedback_delay_seconds"], hardware["maximum_seconds"] = delay, timeout
    required = ["cvae.pt", "lstm.pt", "transformer.pt"]
    if models is not None and (not models.is_dir() or any(not (models / name).is_file() for name in required)):
        raise ValueError("The prepared models directory requires TP-HSMM and all three network checkpoints")
    tolerance = _finite(config.get("study", {}).get("hardware_scene_tolerance_m", .005),
                        "hardware_scene_tolerance_m")
    mode = "execute" if execute else "read-only"
    target = output / "acquisition" / trial / mode
    if target.exists():
        raise FileExistsError("This trial/mode already has an acquisition; existing measurements are immutable")
    runtime = copy.deepcopy(config)
    runtime["task"].update(feedback_delay=delay, command_delay=0., max_seconds=timeout,
                                 decision_dt=hardware["loop_seconds"])
    assignment = {
        "schema_version": 1, "acquisition_id": uuid.uuid4().hex, "trial_id": trial,
        "participant_id": identifier, "condition": condition, "intended_goal": intended_goal,
        "scene_seed": seed, "scene_seed_scope": "schedule_label_not_a_source_of_measured_coordinates",
        "scene_id": row.get("scene_id") or hardware.get("scene_id") or f"measured_{_json_sha(scene)[:16]}",
        "scene": scene, "scene_geometry_sha256": _json_sha(scene), "execute": execute,
        "robot_radius_m": float(runtime["task"]["robot_radius"]),
        "feedback_delay_seconds": delay, "command_delay_seconds": 0., "timeout_seconds": timeout,
        "practice": str(row.get("practice", "false")).lower() == "true",
        "planned_order": _integer(row.get("planned_order") or row["order"], "order") if row.get("planned_order") or row.get("order") else None,
        "schedule_sha256": _sha(schedule_csv), "schedule_file": str(schedule_csv),
        "hardware_source_sha256": _sha(hardware_source), "model_source": str(models),
        "selected_skill_source": str(selected_skill), "selected_skill_sha256": selected_skill_sha256,
        "config_sha256": _json_sha(runtime), "base_config_sha256": _json_sha(config),
        "runtime_config": runtime, "software": _software(),
        "outcome_source": "independent_task_annotation_after_acquisition",
        "stage": stage, "candidate_config_sha256": candidate_config_sha256,
        "policy_sha256": selected_skill_sha256,
    }
    for field in ("training_repeat", "refinement_budget", "attempt", "evaluation_index", "matched_block_index",
                  "within_block_order", "randomization_seed"):
        if row.get(field) is not None and row[field] != "":
            assignment[field] = _integer(row[field], field)
    for field in ("training_condition", "scenario", "layout_group", "candidate_id", "actor_type", "matched_block_id"):
        if row.get(field):
            assignment[field] = row[field]
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".hardware-trial.lock"
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Another acquisition owns this experiment output") from exc
    ledger = output / "execution-ledger.jsonl"
    common = {key: assignment[key] for key in (
        "acquisition_id", "trial_id", "participant_id", "condition", "intended_goal", "scene_seed",
        "scene_id", "scene_geometry_sha256", "execute", "selected_skill_source", "selected_skill_sha256")}
    common.update({key: assignment[key] for key in ("stage", "layout_group", "training_repeat", "training_condition") if key in assignment})
    created = False
    try:
        with os.fdopen(lock, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid(), **common}))
        assignment["matched_scene_audit"] = _matched_scene(_ledger_rows(ledger), assignment, tolerance)
        target.mkdir(parents=True, exist_ok=False)
        created = True
        frozen_models, inputs = target / "models", target / "inputs"
        frozen_models.mkdir()
        inputs.mkdir()
        for source in sorted(models.iterdir()) if models is not None else []:
            if source.is_file() and source.suffix in {".pt", ".npz", ".json"}:
                shutil.copyfile(source, frozen_models / source.name)
        override_skill = skill_path is not None or row.get("policy_file") or row.get("skill_file")
        frozen_skill = frozen_models / ("selected_skill.npz" if override_skill else "bootstrap_hsmm.npz")
        if selected_skill is not None and override_skill:
            shutil.copyfile(selected_skill, frozen_skill)
        if selected_skill is not None and _sha(frozen_skill) != selected_skill_sha256:
            raise ValueError("Selected skill checkpoint changed while freezing the experiment")
        assignment["selected_skill_snapshot"] = str(frozen_skill.relative_to(target))
        assignment["model_sha256"] = {path.name: _sha(path) for path in sorted(frozen_models.iterdir())}
        shutil.copyfile(hardware_source, inputs / "hardware.source.yaml")
        shutil.copyfile(schedule_csv, inputs / "schedule.csv")
        for key, name in (("calibration_record", "calibration.json"), ("snapshot_record", "snapshot.json"),
                          ("perception_record", "perception.json"), ("tool_record", "tool.json")):
            if hardware.get(key):
                source = _resolve(hardware[key], hardware_source.parent)
                shutil.copyfile(source, inputs / name)
                hardware[key] = str((inputs / name).resolve())
        resolved = target / "hardware.resolved.yaml"
        resolved.write_text(yaml.safe_dump(hardware, sort_keys=False), encoding="utf-8")
        _load_hardware(resolved)
        assignment["hardware_resolved_sha256"] = _sha(resolved)
        write_json(target / "assignment.json", assignment)
        common["assignment_sha256"] = _sha(target / "assignment.json")
        common["assignment_path"] = str((target / "assignment.json").relative_to(output))
        _append(ledger, {**common, "event": "started", "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        recording_path = target / "recording.json"
        options = {"execute": execute, "condition": condition, "stage": stage}
        if selected_skill is not None and override_skill:
            options["skill_path"] = frozen_skill
        if condition == "autonomous":
            options["autonomous_goal"] = intended_goal
        run_workstation(runtime, resolved, frozen_models if models is not None else None, recording_path, **options)
        recording = json.loads(recording_path.read_text(encoding="utf-8"))
        count, motion = _validate_recording(recording, assignment, execute)
        recording["experiment_assignment"] = _recording_assignment(assignment, common)
        temporary = target / "recording.linked.tmp.json"
        write_json(temporary, recording)
        os.replace(temporary, recording_path)
        result = {**common, "event": "recorded", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "recording_path": str(recording_path.relative_to(output)), "recording_sha256": _sha(recording_path),
                  "n_samples": count, "motion_observed": motion,
                  "eligible_for_demonstration_annotation": bool(execute and motion and count >= 3),
                  "scene": scene, "stop_reason": recording.get("stop_reason"),
                  "outcome": "independent_annotation_required", "provenance": "measured_workstation_acquisition"}
        _append(ledger, result)
        write_json(target / "acquisition_status.json", result)
        return result
    except BaseException as exc:
        if created:
            saved = target / "recording.json"
            if saved.is_file():
                partial = json.loads(saved.read_text(encoding="utf-8"))
                partial["experiment_assignment"] = _recording_assignment(assignment, common)
                partial["acquisition_error"] = {"type": type(exc).__name__, "message": str(exc)}
                write_json(saved, partial)
            failure = {**common, "event": "failed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "error_type": type(exc).__name__, "error": str(exc), "outcome": "not_inferred"}
            _append(ledger, failure)
            write_json(target / "acquisition_status.json", failure)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
