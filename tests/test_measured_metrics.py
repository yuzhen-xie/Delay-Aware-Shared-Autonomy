import copy
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from dasa.data import Demonstration
from dasa.geometry import Scene
from dasa.robotics.analysis import export_metrics


def _measured(path, config, success=True, practice=False):
    time = np.array([1., 1.2, 1.55, 1.9, 2.7])
    position = np.c_[.2 + .02 * time, np.full(len(time), -.1), np.full(len(time), .4)]
    velocity = np.c_[np.full(len(time), .02), np.zeros((len(time), 2))]
    scene = Scene(np.asarray(config["scene"]["goals"]), np.asarray(config["scene"]["place"]), position[0],
                  np.array([[.7, .4, .2]]), np.array([.05]), ["obstacle"], "calibrated_scene", "base")
    metadata = {"episode_id": path.stem, "operator_id": "P01", "provenance": "UR10_measured_teleoperation",
                "success": success, "collision": False, "bounds_violation": False, "placed": success,
                "grasped_goal": 0 if success else None, "source_condition": "C3", "delay_seconds": .8,
                "termination": "gripper_release" if success else "duration",
                "source_recording_sha256": "a" * 64,
                "execution_contract": {"maximum_seconds": 60., "robot_radius_m": config["simulation"]["robot_radius"]},
                "experiment_assignment": {"trial_id": path.stem, "participant_id": "P01", "intended_goal": 0,
                                          "practice": practice, "planned_order": 1, "timeout_seconds": 60.}}
    demo = Demonstration(time, position, velocity, np.zeros_like(position), np.c_[position, velocity],
                         np.zeros(len(time), int), scene, 0, metadata)
    demo.save(path)
    return demo


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_nonuniform_measured_timestamps_and_source_hash_are_preserved(config, tmp_path):
    root, output = tmp_path / "processed", tmp_path / "metrics.csv"
    source = root / "nested" / "P01_trial01.npz"
    demo = _measured(source, config)
    report = export_metrics(config, root, output)
    rows = _rows(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["trial_id"] == demo.metadata["episode_id"] and row["participant_id"] == "P01"
    assert row["condition"] == "C3" and float(row["delay_seconds"]) == .8
    assert row["success"].lower() == "true" and row["record_status"] == "collected"
    assert np.isclose(float(row["completion_seconds"]), 1.7)
    assert float(row["jerk_rms"]) < 1e-12
    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row["source_file"] == str(source.relative_to(root))
    assert report["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["sources"][0]["time_window_seconds"] == [1., 2.7]
    assert report == json.loads(output.with_suffix(".audit.json").read_text())


def test_swept_clearance_detects_crossing_between_safe_sample_positions(config, tmp_path):
    path = tmp_path / "crossing.npz"
    demo = _measured(path, config, success=False)
    demo.time = np.array([0., 2., 3.])
    demo.position = np.array([[-1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    demo.velocity = np.tile([1., 0., 0.], (3, 1))
    demo.raw_force, demo.phase = np.zeros((3, 3)), np.zeros(3, int)
    demo.observed_state = np.c_[demo.position, demo.velocity]
    demo.scene.centers, demo.scene.radii = np.zeros((1, 3)), np.array([.1])
    demo.metadata["collision"] = True
    demo.save(path)
    assert demo.scene.clearance(demo.position, config["simulation"]["robot_radius"]).min() > 0
    output = tmp_path / "metrics.csv"
    report = export_metrics(config, path, output)
    row = _rows(output)[0]
    assert np.isclose(float(row["minimum_clearance_m"]), -.1-config["simulation"]["robot_radius"])
    assert row["success"].lower() == "false" and row["completion_seconds"] == ""
    assert float(row["elapsed_seconds"]) == 3. and float(row["timeout_seconds"]) == 60.
    assert report["n_failures"] == 1 and report["n_success"] == 0


def test_practice_and_quadratic_velocity_jerk_are_not_filtered(config, tmp_path):
    path = tmp_path / "practice.npz"
    demo = _measured(path, config, practice=True)
    demo.velocity[:, 0] = demo.time ** 2
    demo.save(path)
    output = tmp_path / "metrics.csv"
    report = export_metrics(config, path, output)
    assert report["n_practice"] == 1
    assert _rows(output)[0]["practice"].lower() == "true"
    assert np.isclose(float(_rows(output)[0]["jerk_rms"]), 2., atol=1e-12)


@pytest.mark.parametrize("change", [{"provenance": "synthetic_quintic_demonstration"},
                                    {"collision": "false"}, {"collision": True},
                                    {"grasped_goal": 1}, {"delay_seconds": "0.8"}])
def test_unmeasured_or_contradictory_annotations_do_not_create_outputs(config, tmp_path, change):
    path = tmp_path / "trial.npz"
    demo = _measured(path, config)
    demo.metadata.update(change)
    demo.save(path)
    output = tmp_path / "metrics.csv"
    with pytest.raises(ValueError):
        export_metrics(config, path, output)
    assert not output.exists() and not output.with_suffix(".audit.json").exists()


def test_duplicate_episodes_are_rejected_before_export(config, tmp_path):
    root = tmp_path / "processed"
    _measured(root / "trial.npz", config)
    shutil.copyfile(root / "trial.npz", root / "duplicate.npz")
    output = tmp_path / "metrics.csv"
    with pytest.raises(ValueError, match="Duplicate trial"):
        export_metrics(config, root, output)
    assert not output.exists()


def test_actual_radius_must_match_configuration_and_export_is_immutable(config, tmp_path):
    path = tmp_path / "trial.npz"
    _measured(path, config)
    changed = copy.deepcopy(config)
    changed["simulation"]["robot_radius"] += .01
    output = tmp_path / "metrics.csv"
    with pytest.raises(ValueError, match="radius differs"):
        export_metrics(changed, path, output)
    export_metrics(config, path, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_metrics(config, path, output)
    assert output.read_bytes() == before
