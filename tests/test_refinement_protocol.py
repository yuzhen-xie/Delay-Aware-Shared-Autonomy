import copy
import json

import numpy as np
import pytest

from dasa.data import generate_dataset, synthetic_demonstration
from dasa.models.hsmm import TPHSMM, demonstration_fingerprint, refine


def small_config(config):
    result = copy.deepcopy(config)
    result["skill"].update(states=3, trajectory_samples=21, em_iterations=0,
                           refinement_count=2, original_weight=1.0, new_weight=2.0)
    result["scene"].update(training_translation=0,
                           obstacles=[{"id": "far", "center": [.9, -.5, .2], "radius": .01}])
    return result


def demonstration(config, seed, episode, operator="collection", condition="C1"):
    demo = synthetic_demonstration(config, seed, operator, episode, 0)
    demo.metadata.update(source_condition=condition, delay_seconds=.8)
    return demo


def test_refinement_rejects_copied_signals_and_records_exact_weights(config, tmp_path):
    config = small_config(config)
    original = demonstration(config, 1, "original", "bootstrap")
    first = demonstration(config, 2, "new_1")
    copied = copy.deepcopy(first)
    copied.metadata["episode_id"] = "renamed_copy"
    failed = demonstration(config, 3, "failed")
    failed.metadata["success"] = False
    model, audit, accepted = refine(config, [original], [first, copied, failed])
    assert [d.metadata["episode_id"] for d in accepted] == ["new_1"]
    assert "duplicate_or_original_overlap" in audit[1]["reasons"]
    assert "task_incomplete" in audit[2]["reasons"]
    assert model.fit_manifest["refinement"]["comparison_status"] == "diagnostic_only_incomplete_quota"
    assert [r["weight"] for r in model.fit_manifest["records"]] == [1, 2]
    assert model.fit_manifest["total_weight"] == 3
    assert len(audit[0]["signal_sha256"]) == 64
    model.save(tmp_path / "fit.npz")
    restored = TPHSMM.load(config, tmp_path / "fit.npz")
    assert restored.fit_manifest == model.fit_manifest
    wrong_horizon = copy.deepcopy(config)
    wrong_horizon["skill"]["trajectory_samples"] += 1
    with pytest.raises(ValueError, match="trajectory horizon"):
        TPHSMM.load(wrong_horizon, tmp_path / "fit.npz")


def test_signal_hash_does_not_allow_new_labels_to_hide_data_leakage(config):
    config = small_config(config)
    original = demonstration(config, 1, "original")
    copied = copy.deepcopy(original)
    copied.metadata.update(episode_id="different", operator_id="different")
    assert demonstration_fingerprint(original)["signal_sha256"] == demonstration_fingerprint(copied)["signal_sha256"]
    assert demonstration_fingerprint(original)["record_sha256"] != demonstration_fingerprint(copied)["record_sha256"]
    with pytest.raises(ValueError, match="Duplicate demonstration signals"):
        TPHSMM(config).fit([original, copied])
    with pytest.raises(ValueError, match="positive"):
        TPHSMM(config).fit([original], np.array([np.nan]))


def test_budget_zero_is_exact_bootstrap_and_prefix_order_is_fixed(config):
    config = small_config(config)
    original = demonstration(config, 1, "original")
    candidate = demonstration(config, 2, "candidate")
    config["skill"]["refinement_count"] = 0
    refined, audit, accepted = refine(config, [original], [candidate])
    baseline = TPHSMM(config).fit([original])
    assert not accepted and audit[0]["reasons"] == ["quota_reached"]
    assert np.array_equal(refined.means, baseline.means)
    assert refined.fit_manifest["refinement"]["quota_met"]


def test_generalization_keeps_missing_budget_unavailable_and_units_as_repeats(config, tmp_path,
                                                                            monkeypatch):
    import dasa.generalization as module

    config = small_config(config)
    config["experiments"].update(bootstrap_demonstrations=3, validation_demonstrations=1,
                                 test_demonstrations=1, independent_repeats=2,
                                 autonomy_trials=1, scenarios=["demonstrated"],
                                 refinement_budgets=[0, 1, 2])
    prepared = tmp_path / "prepared"
    generate_dataset(config, prepared / "dataset")
    candidates = tmp_path / "collections"
    for repeat in range(2):
        for index, condition in enumerate(("C1", "C2", "C3")):
            demonstration(config, 100 + repeat * 10 + index,
                          f"new_{repeat}_{condition}", f"operator_{repeat}", condition).save(
                candidates / f"r{repeat}" / condition / "000.npz")
    simulation_bootstrap = demonstration(config, 500, "initial_collection_bootstrap", "bootstrap_r0", "bootstrap")
    simulation_bootstrap.metadata["delay_seconds"] = 0.0
    simulation_bootstrap.save(candidates.parent / "bootstrap/r0/000.npz")
    monkeypatch.setattr(module, "load_models", lambda *_: (None, {}))

    class RecordedTrial:
        def __init__(self, unit):
            self.unit = unit

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test fixture")

    monkeypatch.setattr(module, "run_trial", lambda c, scene, model, cvae, seed, goal,
                        mode, unit, trial_id: RecordedTrial(unit))
    monkeypatch.setattr(module, "trial_metrics", lambda trial, c: {
        "unit": trial.unit, "delay_seconds": .8, "success_rate": 1.0,
        "reference_error_m": .01})
    output = tmp_path / "analysis"
    rows = module.generalization(config, prepared, candidates, output)
    assert len(rows) == 8  # baseline + three complete one-demonstration fits per repeat
    assert all(row["refinement_budget"] in (0, 1) for row in rows)
    assert sum(row["condition"] == "bootstrap" for row in rows) == 2
    budget_rows = json.loads((output / "budgets.json").read_text())
    assert {row["bootstrap_policy"] for row in budget_rows if row["repeat"] == 0} == {
        "recorded_simulation_per_repeat_bootstrap"}
    assert {row["bootstrap_policy"] for row in budget_rows if row["repeat"] == 1} == {
        "shared_fixed_bootstrap_conditional_inference"}
    model = TPHSMM.load(config, output / "models/r0/bootstrap.npz")
    assert model.fit_ids == ["initial_collection_bootstrap"]
    missing = [row for row in budget_rows if row["refinement_budget"] == 2]
    assert len(missing) == 6 and all(not row["confirmatory_eligible"] for row in missing)
    status = json.loads((output / "protocol_status.json").read_text())
    assert status["procedure_completed"] and not status["protocol_complete"]
    stats = json.loads((output / "statistics.json").read_text())
    assert all(row["n_units"] == 2 for row in stats if row["refinement_budget"] == 1)
    assert all(row["n_units"] == 0 for row in stats if row["refinement_budget"] == 2)
    assert all(row["p_raw"] is None for row in stats if row["refinement_budget"] == 2)
    with pytest.raises(ValueError, match="empty"):
        module.generalization(config, prepared, candidates, output)


def test_candidate_acquisition_manifest_detects_omission_and_path_escape(config, tmp_path):
    from dasa.generalization import load_candidates

    config = small_config(config)
    demonstration(config, 1, "first").save(tmp_path / "a.npz")
    demonstration(config, 2, "second").save(tmp_path / "b.npz")
    (tmp_path / "order.json").write_text(json.dumps(["b.npz", "a.npz"]))
    demos, _ = load_candidates(tmp_path)
    assert [d.metadata["episode_id"] for d in demos] == ["second", "first"]
    (tmp_path / "order.json").write_text(json.dumps(["a.npz"]))
    with pytest.raises(ValueError, match="each NPZ"):
        load_candidates(tmp_path)
    (tmp_path / "order.json").write_text(json.dumps(["../a.npz", "b.npz"]))
    with pytest.raises(ValueError, match="escape"):
        load_candidates(tmp_path)


def test_prepared_dataset_snapshot_preserves_bytes_and_only_declared_inputs(config, tmp_path):
    import hashlib
    from dasa.experiments import snapshot_dataset, load_split

    config = small_config(config)
    config["experiments"].update(bootstrap_demonstrations=3, validation_demonstrations=1,
                                 test_demonstrations=1)
    source = tmp_path / "source"
    splits = generate_dataset(config, source)
    (source / "private_notes.txt").write_text("not a declared recording")
    prepared = tmp_path / "prepared"
    copied = snapshot_dataset(source, prepared)
    assert copied == prepared / "dataset"
    assert not (copied / "private_notes.txt").exists()
    for name in ["splits.json", *(p for files in splits.values() for p in files)]:
        assert hashlib.sha256((copied / name).read_bytes()).digest() == hashlib.sha256((source / name).read_bytes()).digest()
    assert len(load_split(copied, "train")) == 3
    assert snapshot_dataset(copied, prepared) == copied
    with pytest.raises(ValueError, match="already contains"):
        snapshot_dataset(source, prepared)


def test_simulation_collection_does_not_replace_measured_bootstrap(config, tmp_path):
    from dasa.data import Demonstration
    from dasa.experiments import refinement_experiment

    config = small_config(config)
    config["experiments"].update(bootstrap_demonstrations=1, validation_demonstrations=1,
                                 test_demonstrations=1)
    splits = generate_dataset(config, tmp_path / "dataset")
    path = tmp_path / "dataset" / splits["train"][0]
    demo = Demonstration.load(path)
    demo.metadata["provenance"] = "measured_annotated_recording"
    demo.save(path)
    with pytest.raises(ValueError, match="measured collections use refine-data/generalization"):
        refinement_experiment(config, tmp_path)


def test_collection_loader_distinguishes_trial_and_demonstration_schema_one(config, tmp_path):
    from dasa.generalization import load_candidates
    from dasa.simulation import Trial

    config = small_config(config)
    demo = demonstration(config, 1, "processed")
    demo.save(tmp_path / "000.npz")
    trial = Trial({key: getattr(demo, key) for key in (
        "time", "position", "velocity", "raw_force", "observed_state", "phase")}, {
            "schema_version": 1, "trial_id": "raw_trial", "operator_id": "operator",
            "scene": demo.scene.to_dict(), "intended_goal": 0, "success": True,
            "collision": False, "bounds_violation": False, "grasped_goal": 0,
            "condition": "C1", "delay_seconds": .8, "seed": 42,
            "provenance": "synthetic_task_space_simulation"})
    trial.save(tmp_path / "001.npz")
    loaded, _ = load_candidates(tmp_path)
    assert [d.metadata["episode_id"] for d in loaded] == ["processed", "raw_trial"]
    assert loaded[1].metadata["simulation_seed"] == 42
    assert loaded[1].metadata["recording_format"] == "simulation_trial"
