import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dasa.robotics import experiment


def _write_schedule(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def experiment_inputs(config, tmp_path, monkeypatch):
    hardware = {
        "robot": {"host": "192.0.2.10", "model": "UR10"},
        "loop_seconds": .02, "maximum_seconds": 60., "feedback_delay_seconds": .8,
        "calibration_record": str(tmp_path / "calibration.json"),
        "scene_id": "measured_layout_01", "scene": {
            "goals": config["scene"]["goals"], "place": config["scene"]["place"],
            "start": config["scene"]["start"],
            "centers": [r["center"] for r in config["scene"]["obstacles"]],
            "radii": [r["radius"] for r in config["scene"]["obstacles"]],
            "obstacle_ids": [r["id"] for r in config["scene"]["obstacles"]],
            "frame": "base", "provenance": "calibrated_scene"}}
    (tmp_path / "calibration.json").write_text(json.dumps({"target_frame": "base", "rms_m": .001}))
    hardware_file = tmp_path / "hardware.yaml"
    hardware_file.write_text(yaml.safe_dump(hardware))
    models = tmp_path / "trained_models"
    models.mkdir()
    for name in ("bootstrap_hsmm.npz", "cvae.pt", "lstm.pt", "transformer.pt"):
        (models / name).write_bytes(f"mock-checkpoint-{name}".encode())
    (models / "training-audit.json").write_text('{"source":"unit_test"}')
    schedule = tmp_path / "schedule.csv"
    row = dict(trial_id="P01_r00_c02", participant_id="P01", condition="C2", delay_seconds=1.5,
               command_delay_seconds=0., intended_goal=2, scene_seed=101, order=3,
               scene_id="measured_layout_01", practice="false", record_status="planned")
    _write_schedule(schedule, [row])
    calls = []

    def load_hardware(path):
        parsed = yaml.safe_load(Path(path).read_text())
        assert parsed["scene"]["provenance"] == "calibrated_scene"
        return parsed

    def acquire(runtime, hardware_path, frozen_models, output, execute=False, condition="C3", skill_path=None):
        parsed = yaml.safe_load(Path(hardware_path).read_text())
        calls.append(dict(config=copy.deepcopy(runtime), hardware=parsed, models=frozen_models,
                          execute=execute, condition=condition, output=output, skill_path=skill_path))
        samples = [dict(time=.02 * (i+1), position=[.2, -.1, .4], velocity=[0., 0., 0.],
                        raw_force=[1., 0., 0.], executed=execute) for i in range(3)]
        output.write_text(json.dumps({"schema_version": 2, "provenance": "UR10_workstation_measurement",
                                     "execute": execute, "condition": condition, "scene": parsed["scene"],
                                     "samples": samples, "stop_reason": "duration",
                                     "timing": {"feedback_delay_seconds": parsed["feedback_delay_seconds"],
                                                "command_delay_seconds": 0.},
                                     "outcome": "requires_independent_task_annotation"}))

    monkeypatch.setattr(experiment, "_load_hardware", load_hardware)
    monkeypatch.setattr(experiment, "run_workstation", acquire)
    return dict(config=config, directory=tmp_path, hardware=hardware, hardware_file=hardware_file,
                models=models, schedule=schedule, row=row, calls=calls, output=tmp_path / "experiment")


def _run(inputs, **kwargs):
    return experiment.run_experiment(inputs["config"], inputs["schedule"], inputs["hardware_file"],
                                     inputs["models"], inputs["output"], **kwargs)


def test_single_selected_measured_trial_records_exact_assignment_and_versions(experiment_inputs):
    p = experiment_inputs
    result = _run(p, execute=True, participant="P01", trial_id=p["row"]["trial_id"])
    assert len(p["calls"]) == 1
    call = p["calls"][0]
    assert call["condition"] == "C2" and call["execute"] is True
    assert call["hardware"]["feedback_delay_seconds"] == 1.5
    assert call["config"]["simulation"]["feedback_delay"] == 1.5
    assert call["hardware"]["scene"] == p["hardware"]["scene"]
    assert call["models"] != p["models"]
    assignment_file = p["output"] / result["assignment_path"]
    assignment = json.loads(assignment_file.read_text())
    assert assignment["intended_goal"] == 2 and assignment["scene_seed"] == 101
    assert assignment["software"]["source_sha256"]["robotics/experiment.py"] == hashlib.sha256(
        Path(experiment.__file__).read_bytes()).hexdigest()
    assert set(assignment["model_sha256"]) == {p.name for p in p["models"].iterdir()}
    for name, digest in assignment["model_sha256"].items():
        assert hashlib.sha256((call["models"] / name).read_bytes()).hexdigest() == digest
    raw = json.loads((p["output"] / result["recording_path"]).read_text())
    assert raw["experiment_assignment"]["trial_id"] == p["row"]["trial_id"]
    assert "success" not in raw and result["outcome"] == "independent_annotation_required"
    assert result["eligible_for_demonstration_annotation"] is True
    events = [json.loads(line) for line in (p["output"] / "execution-ledger.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "recorded"]
    assert not (p["output"] / ".hardware-trial.lock").exists()


def test_read_only_acquisition_never_becomes_executed_demonstration(experiment_inputs):
    p = experiment_inputs
    result = _run(p)
    assert p["calls"][0]["execute"] is False
    assert result["n_samples"] == 3 and result["motion_observed"] is False
    assert result["eligible_for_demonstration_annotation"] is False
    assert "/read-only/" in result["recording_path"]


def test_multiple_rows_require_single_trial_selection_and_never_call_hardware(experiment_inputs):
    p = experiment_inputs
    second = {**p["row"], "trial_id": "P01_r00_c03", "condition": "C3"}
    _write_schedule(p["schedule"], [p["row"], second])
    with pytest.raises(ValueError, match="exactly one"):
        _run(p, participant="P01")
    assert not p["calls"] and not p["output"].exists()
    _run(p, participant="P01", trial_id=second["trial_id"])
    assert p["calls"][0]["condition"] == "C3"


def test_immutable_acquisition_rejects_rerun_and_preserves_all_raw_bytes(experiment_inputs):
    p = experiment_inputs
    result = _run(p)
    raw_path = p["output"] / result["recording_path"]
    before = raw_path.read_bytes()
    ledger = (p["output"] / "execution-ledger.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        _run(p)
    assert raw_path.read_bytes() == before and len(p["calls"]) == 1
    assert (p["output"] / "execution-ledger.jsonl").read_bytes() == ledger


@pytest.mark.parametrize("change", [{"command_delay_seconds": .1}, {"delay_seconds": "nan"},
                                    {"trial_id": "../outside"}, {"record_status": "collected"}])
def test_invalid_assignment_has_no_output_or_motion(experiment_inputs, change):
    p = experiment_inputs
    _write_schedule(p["schedule"], [{**p["row"], **change}])
    with pytest.raises(ValueError):
        _run(p, execute=True)
    assert not p["calls"] and not p["output"].exists()


def test_empty_acquisition_is_failed_not_fabricated_as_a_trial(experiment_inputs, monkeypatch):
    p = experiment_inputs
    acquire = experiment.run_workstation

    def empty(*args, **kwargs):
        acquire(*args, **kwargs)
        output = args[3]
        record = json.loads(output.read_text())
        record["samples"] = []
        output.write_text(json.dumps(record))

    monkeypatch.setattr(experiment, "run_workstation", empty)
    with pytest.raises(ValueError, match="no measured samples"):
        _run(p, execute=True)
    events = [json.loads(line) for line in (p["output"] / "execution-ledger.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "failed"]
    assert events[-1]["outcome"] == "not_inferred"
    assert (p["output"] / "acquisition" / p["row"]["trial_id"] / "execute" / "recording.json").exists()
    assert not (p["output"] / ".hardware-trial.lock").exists()


def test_interrupted_workstation_is_audited_and_lock_is_released(experiment_inputs, monkeypatch):
    p = experiment_inputs

    def interrupted(*args, **kwargs):
        raise RuntimeError("connection interrupted")

    monkeypatch.setattr(experiment, "run_workstation", interrupted)
    with pytest.raises(RuntimeError, match="connection interrupted"):
        _run(p, execute=True)
    events = [json.loads(line) for line in (p["output"] / "execution-ledger.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "failed" and events[-1]["error_type"] == "RuntimeError"
    assert not (p["output"] / ".hardware-trial.lock").exists()


def test_matched_conditions_require_same_measured_geometry(experiment_inputs):
    p = experiment_inputs
    _run(p, execute=True)
    changed = copy.deepcopy(p["hardware"])
    changed["scene"]["goals"][0][0] += .02
    p["hardware_file"].write_text(yaml.safe_dump(changed))
    second = {**p["row"], "trial_id": "P01_r00_c03", "condition": "C3"}
    _write_schedule(p["schedule"], [second])
    with pytest.raises(ValueError, match="matched condition scene"):
        _run(p, execute=True)
    assert len(p["calls"]) == 1
    assert not (p["output"] / "acquisition" / second["trial_id"]).exists()
    changed["scene"]["goals"][0][0] -= .018
    p["hardware_file"].write_text(yaml.safe_dump(changed))
    result = _run(p, execute=True)
    assignment = json.loads((p["output"] / result["assignment_path"]).read_text())
    assert np.isclose(assignment["matched_scene_audit"]["maximum_geometry_difference_m"], .002)
    assert len(assignment["matched_scene_audit"]["matched_acquisition_ids"]) == 1


def test_row_specific_hardware_and_scene_file_are_resolved_from_schedule(experiment_inputs):
    p = experiment_inputs
    scene_file = p["directory"] / "scene.json"
    scene_file.write_text(json.dumps(p["hardware"]["scene"]))
    _write_schedule(p["schedule"], [{**p["row"], "hardware_file": "hardware.yaml", "scene_file": "scene.json"}])
    result = experiment.run_experiment(p["config"], p["schedule"], p["directory"] / "unused.yaml",
                                       p["models"], p["output"])
    assert result["n_samples"] == 3
    assert p["calls"][0]["hardware"]["scene"] == p["hardware"]["scene"]


def test_existing_execution_lock_prevents_parallel_hardware_acquisition(experiment_inputs):
    p = experiment_inputs
    p["output"].mkdir()
    lock = p["output"] / ".hardware-trial.lock"
    lock.write_text("other-owner")
    with pytest.raises(RuntimeError, match="Another acquisition"):
        _run(p, execute=True)
    assert lock.read_text() == "other-owner" and not p["calls"]


def test_refined_skill_is_frozen_and_its_identity_is_recorded(experiment_inputs):
    p = experiment_inputs
    refined = p["directory"] / "refined_skill.npz"
    refined.write_bytes(b"refined-checkpoint-unit-test")
    result = _run(p, skill_path=refined)
    call = p["calls"][0]
    assert call["skill_path"] != refined
    assert call["skill_path"].read_bytes() == refined.read_bytes()
    assert result["selected_skill_sha256"] == hashlib.sha256(refined.read_bytes()).hexdigest()
    assignment = json.loads((p["output"] / result["assignment_path"]).read_text())
    assert assignment["selected_skill_snapshot"] == "models/selected_skill.npz"
    assert assignment["selected_skill_sha256"] == result["selected_skill_sha256"]


def test_real_configuration_validation_is_preserved_when_freezing_inputs(experiment_inputs, monkeypatch):
    from dasa.robotics.configuration import default_profile, load_hardware

    p = experiment_inputs
    calibration_file = p["directory"] / "calibration.json"
    calibration_file.write_text(json.dumps({"source_frame": "camera_optical", "target_frame": "base",
                                           "transform": np.eye(4).tolist(), "rms_m": .001}))
    hardware = default_profile(p["config"])
    hardware.update(calibration_record=str(calibration_file), scene=p["hardware"]["scene"],
                    scene_id="measured_layout_01")
    hardware["robot"]["host"] = "192.0.2.10"
    p["hardware_file"].write_text(yaml.safe_dump(hardware))
    monkeypatch.setattr(experiment, "_load_hardware", load_hardware)
    result = _run(p)
    frozen = p["calls"][0]["hardware"]
    assert Path(frozen["calibration_record"]).parent.name == "inputs"
    assert frozen["calibration_sha256"] == hashlib.sha256(calibration_file.read_bytes()).hexdigest()
    assert result["n_samples"] == 3
