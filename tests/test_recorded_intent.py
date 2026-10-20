import hashlib
import json

import numpy as np
import pytest

import dasa.recorded as recorded
from dasa.data import Demonstration


def prepared_dataset(tmp_path, scene):
    root = tmp_path / "prepared"
    dataset = root / "dataset"
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        time = 100 + np.arange(6) * .1
        position = np.c_[.15 + index * .01 + np.arange(6) * .01,
                         np.full(6, -.1), np.full(6, .4)]
        raw_force = np.c_[1 + index * .1 + np.arange(6) * .2, np.zeros((6, 2))]
        velocity = np.zeros((6, 3))
        observed = np.c_[position - [.02, 0, 0], velocity]
        demo = Demonstration(time, position, velocity, raw_force, observed,
                             np.array([0, 0, 0, 0, 2, 3]), scene, 0,
                             {"episode_id": f"{split}_episode", "operator_id": f"{split}_unit",
                              "provenance": "test_fixture_recorded", "success": True,
                              "delay_seconds": .8})
        demo.save(dataset / split / "recording.npz")
        splits[split] = [f"{split}/recording.npz"]
    (dataset / "splits.json").write_text(json.dumps(splits))
    models = root / "models"
    models.mkdir()
    (models / "fixture.pt").write_bytes(b"test-model-fixture")
    return root


def spy_estimators(monkeypatch):
    calls = {}

    class Estimator:
        def __init__(self, method):
            self.method = method
            self.previous = np.zeros(2)
            calls.setdefault(method, [])

        def step(self, action, observed, current, scene, age):
            calls[self.method].append((action.copy(), observed.copy(), current.copy(), age))
            # A causal, label-free stand-in; it deliberately consumes only this sample.
            probability = np.array([action[0] + observed[0], 1., 1.])
            probability /= probability.sum()
            self.previous += [action[0], observed[0]]
            return probability, self.previous.copy()

    monkeypatch.setattr(recorded.TPHSMM, "load", lambda config, path: object())
    monkeypatch.setattr(recorded.DMP, "load", lambda path: object())
    monkeypatch.setattr(recorded, "load_models", lambda config, path: (object(), {}))
    monkeypatch.setattr(recorded, "ReferenceBank", lambda skill, scene: object())
    monkeypatch.setattr(recorded, "IntentEstimator", lambda config, bank, model, kind, network: Estimator(kind))
    monkeypatch.setattr(recorded, "DMPIntent", lambda dmp, scene, beta: Estimator("dmp"))
    return calls


def append_source_time(path, source):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    np.savez_compressed(path, **arrays, observed_source_time=source)


def test_recorded_intent_replays_identical_actual_samples_without_simulation(config, scene, tmp_path, monkeypatch):
    import dasa.simulation

    def forbid_simulation(*args, **kwargs):
        raise AssertionError("Recorded evaluation must never generate an interaction")

    monkeypatch.setattr(dasa.simulation, "run_trial", forbid_simulation)
    root = prepared_dataset(tmp_path, scene)
    calls = spy_estimators(monkeypatch)
    output = tmp_path / "evaluation"
    rows = recorded.intent_recordings(config, root, output)
    original = Demonstration.load(root / "dataset/test/recording.npz")
    assert len(rows) == 8
    for method in recorded.METHODS:
        assert len(calls[method]) == 4
        for index, (action, observed, current, age) in enumerate(calls[method]):
            assert np.array_equal(action, original.raw_force[index])
            assert np.array_equal(observed, original.observed_state[index])
            assert np.array_equal(current, np.r_[original.position[index], original.velocity[index]])
            assert age == .8
        with np.load(output / f"intent/predictions/{method}_episode00000.npz", allow_pickle=False) as prediction:
            assert np.allclose(prediction["time"], [0, .1, .2, .3])
            assert np.array_equal(prediction["recorded_time"], original.time[:4])
            assert prediction["posterior"].shape == (4, 3)
    inventory = json.loads((output / "intent/input_inventory.json").read_text())
    path = root / "dataset/test/recording.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert inventory[0]["source_sha256"] == digest
    assert inventory[0]["age_source"].startswith("nominal_metadata_only")
    assert {row["provenance"] for row in rows} == {"test_fixture_recorded"}
    record = json.loads((output / "manifest.json").read_text())
    assert record["input_sha256"][str(path)] == digest
    statistics = json.loads((output / "intent/statistics.json").read_text())
    assert {row["delay_seconds"] for row in statistics} == set(config["experiments"]["intent_delays"])
    assert all(row["p_raw"] is None for row in statistics)
    with pytest.raises(ValueError, match="must be empty"):
        recorded.intent_recordings(config, root, output)


def test_label_and_future_signal_changes_do_not_change_earlier_predictions(config, scene, tmp_path, monkeypatch):
    root = prepared_dataset(tmp_path, scene)
    spy_estimators(monkeypatch)
    recorded.intent_recordings(config, root, tmp_path / "first")
    path = root / "dataset/test/recording.npz"
    demo = Demonstration.load(path)
    demo.goal = 2
    demo.raw_force[3:] = [100., 2., 3.]
    demo.observed_state[3:, 0] += 1
    demo.save(path)
    recorded.intent_recordings(config, root, tmp_path / "second")
    for method in recorded.METHODS:
        with np.load(tmp_path / f"first/intent/predictions/{method}_episode00000.npz") as first, \
                np.load(tmp_path / f"second/intent/predictions/{method}_episode00000.npz") as second:
            assert np.array_equal(first["posterior"][:3], second["posterior"][:3])
            assert not np.array_equal(first["posterior"][3], second["posterior"][3])
            assert first["true_goal"] == 0 and second["true_goal"] == 2


def test_recorded_timestamps_used_and_missing_delay_rejected(config, scene, tmp_path, monkeypatch):
    root = prepared_dataset(tmp_path, scene)
    path = root / "dataset/test/recording.npz"
    demo = Demonstration.load(path)
    demo.metadata["provenance"] = "synthetic_fixture"
    demo.save(path)
    ages = np.linspace(.1, .6, len(demo.time))
    append_source_time(path, demo.time - ages)
    calls = spy_estimators(monkeypatch)
    rows = recorded.intent_recordings(config, root, tmp_path / "sourced")
    assert {row["provenance"] for row in rows} == {"synthetic_fixture"}
    assert np.allclose([row[-1] for row in calls["full"]], ages[:4])
    inventory = json.loads((tmp_path / "sourced/intent/input_inventory.json").read_text())
    assert inventory[0]["age_source"].startswith("recorded_timestamps")
    demo.metadata.pop("delay_seconds")
    demo.save(path)
    with pytest.raises(ValueError, match="delay_seconds"):
        recorded.intent_recordings(config, root, tmp_path / "invalid")


def test_future_observation_timestamp_is_rejected(config, scene, tmp_path, monkeypatch):
    root = prepared_dataset(tmp_path, scene)
    path = root / "dataset/test/recording.npz"
    demo = Demonstration.load(path)
    append_source_time(path, demo.time + 1)
    spy_estimators(monkeypatch)
    with pytest.raises(ValueError, match="future"):
        recorded.intent_recordings(config, root, tmp_path / "invalid")
