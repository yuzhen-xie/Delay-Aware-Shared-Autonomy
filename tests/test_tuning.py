import copy
import json

import pytest
import torch

from dasa.data import Demonstration, synthetic_demonstration
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import ConditionalVAE, SequenceClassifier
from dasa.tuning import _summary, sweep


def test_sweep_runs_actual_validation_trials_without_opening_train_or_test(config, tmp_path,
                                                                          monkeypatch):
    config["simulation"]["max_seconds"] = 0.04
    config["skill"].update(states=3, trajectory_samples=24, em_iterations=1)
    config["intent"].update(hidden_dim=8, latent_dim=2, transformer_heads=2, transformer_layers=1)
    config["tuning"] = {"split": "validation", "grids": {"intent.beta": [1.0, 3.0]}}
    before = copy.deepcopy(config)
    prepared = tmp_path / "prepared"
    data, models = prepared / "dataset", prepared / "models"
    data.mkdir(parents=True)
    models.mkdir()
    train = [synthetic_demonstration(config, index, f"train_{index}", f"train_{index}", index)
             for index in range(3)]
    TPHSMM(config).fit(train).save(models / "bootstrap_hsmm.npz")
    torch.save(ConditionalVAE(8, 2).state_dict(), models / "cvae.pt")
    for kind in ("lstm", "transformer"):
        torch.save(SequenceClassifier(kind, config).state_dict(), models / f"{kind}.pt")
    validation = synthetic_demonstration(config, 31, "validation_operator", "validation_only", 0)
    validation.save(data / "validation/only.npz")
    (data / "splits.json").write_text(json.dumps({
        "train": ["forbidden_train.npz"], "validation": ["validation/only.npz"],
        "test": ["forbidden_test.npz"]}))
    opened = []
    original_load = Demonstration.load

    def validation_only(cls, path):
        opened.append(path.name)
        assert path.name == "only.npz"
        return original_load(path)

    monkeypatch.setattr(Demonstration, "load", classmethod(validation_only))
    output = tmp_path / "sweep"
    rows = sweep(config, prepared, output)
    assert opened == ["only.npz"]
    assert config == before
    assert len(rows) == 2
    assert {row["source_episode_id"] for row in rows} == {"validation_only"}
    assert all(row["failure_retained"] and row["termination"] == "timeout" for row in rows)
    assert len(list((output / "trials").glob("*.npz"))) == 2
    selection = json.loads((output / "selection.json").read_text())
    assert selection["loaded_demonstration_splits"] == ["validation"]
    assert not selection["train_test_signals_opened"]
    assert not selection["selections"][0]["historical_default_optimized"]
    assert selection["selections"][0]["selected_validation_value"] in (1.0, 3.0)
    ranking = json.loads((output / "ranking.json").read_text())
    assert all(row["completion_seconds"] is None for row in ranking)
    audit = json.loads((output / "manifest.json").read_text())
    assert audit["models_unchanged"]
    assert audit["training_objective"]["name"] == "unrecorded"


def test_invalid_grid_rejected_before_loading_data_or_creating_output(config, tmp_path):
    config["tuning"] = {"split": "validation", "grids": {
        "motion.distance_min": [config["motion"]["distance_max"] + 0.1]}}
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="thresholds must be ordered"):
        sweep(config, tmp_path / "does_not_exist", output)
    assert not output.exists()
    config["tuning"] = {"split": "test", "grids": {"intent.beta": [3.0]}}
    with pytest.raises(ValueError, match="only the validation split"):
        sweep(config, tmp_path / "does_not_exist", output)


def test_summary_weights_source_operator_groups_and_retains_failure_clearance():
    rows = [{"unit": "a", "success_rate": 1, "minimum_clearance_m": .1,
             "completion_seconds": 2.0},
            {"unit": "a", "success_rate": 1, "minimum_clearance_m": .1,
             "completion_seconds": 4.0},
            {"unit": "b", "success_rate": 0, "minimum_clearance_m": -.05,
             "completion_seconds": None}]
    summary = _summary(rows)
    assert summary["success_rate"] == .5
    assert summary["minimum_clearance_m"] == pytest.approx(.025)
    assert summary["completion_seconds"] == 3.0
    assert summary["n_successful_trials"] == 2
