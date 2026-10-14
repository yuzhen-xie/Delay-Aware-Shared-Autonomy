import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dasa.config import seed_everything
from dasa.data import synthetic_demonstration
from dasa.models.hsmm import TPHSMM, refine
from dasa.models.intent import ConditionalVAE
from dasa.simulation import run_trial, Trial
from dasa.study import analyze


@pytest.fixture
def small_skill(config):
    config["skill"].update(trajectory_samples=42, states=5, em_iterations=2)
    demos = [synthetic_demonstration(config, i, f"o{i}", f"e{i}", i % 3) for i in range(9)]
    return TPHSMM(config).fit(demos), demos


def test_em_frame_factor_likelihood_does_not_decrease(small_skill):
    model, _ = small_skill
    values = [h["weighted_log_likelihood"] for h in model.history]
    assert np.min(np.diff(values)) >= -1e-5


def test_complete_autonomous_task_and_energy_record(config, scene, small_skill, tmp_path):
    model, _ = small_skill
    seed_everything(12)
    cvae = ConditionalVAE(config["intent"]["hidden_dim"], config["intent"]["latent_dim"])
    config["simulation"]["feedback_delay"] = .4
    trial = run_trial(config, scene, model, cvae, 22, 0, "autonomous")
    assert trial.metadata["success"], trial.metadata["termination"]
    assert {e["event"] for e in trial.metadata["events"]} == {"grasp", "lift", "place"}
    assert np.max(np.abs(trial.arrays["energy_residual"])) < 1e-8
    assert np.min(trial.arrays["tank_energy"]) >= 0
    assert np.allclose(trial.arrays["raw_force"], 0)
    assert np.allclose(trial.arrays["current_state"][0, :3], scene.start)
    assert trial.arrays["decision_time"][0] == 0
    assert np.all(trial.arrays["decision_time"] < trial.arrays["time"])
    trial.save(tmp_path / "trial.npz")
    restored = Trial.load(tmp_path / "trial.npz")
    assert np.array_equal(restored.arrays["position"], trial.arrays["position"])


def test_refinement_preserves_original_and_selects_successful_only(config, small_skill):
    _, originals = small_skill
    new = synthetic_demonstration(config, 30, "new", "new_success", 0)
    failed = synthetic_demonstration(config, 31, "new", "new_failed", 1)
    failed.metadata["success"] = False
    model, audit, accepted = refine(config, originals, [new, failed])
    assert len(accepted) == 1
    assert len(model.fit_ids) == len(originals) + 1
    assert audit[0]["accepted"]
    assert not audit[1]["accepted"]


def write_csv(path: Path, keys, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_real_study_import_uses_participants_as_units_and_actual_tlx(config, tmp_path):
    protocol = {k: "test_record" for k in ("approval_id", "ethics_body", "consent_procedure", "screening",
                "recruitment", "experience_definition", "practice_trials", "analysis_plan_frozen_on")}
    (tmp_path / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    participants = [{"participant_id": f"T{i}", "age_years": 25 + i, "gender": "not_reported",
                     "teleoperation_experience": "test_category", "consented": "true"} for i in range(2)]
    write_csv(tmp_path / "participants.csv", participants[0].keys(), participants)
    rows, tlx = [], []
    for i in range(2):
        for name in ("C1", "C2", "C3", "C4"):
            for repetition in range(2):
                rows.append({"trial_id": f"{i}_{name}_{repetition}", "participant_id": f"T{i}",
                             "condition": name, "delay_seconds": .8, "success": "true",
                             "completion_seconds": 10 + repetition, "minimum_clearance_m": .1,
                             "jerk_rms": 2, "practice": "false"})
            tlx.append({"participant_id": f"T{i}", "condition": name, "delay_seconds": .8,
                        **{s: 40 for s in ("mental", "physical", "temporal", "performance", "effort", "frustration")}})
    write_csv(tmp_path / "trials.csv", rows[0].keys(), rows)
    write_csv(tmp_path / "tlx.csv", tlx[0].keys(), tlx)
    output = tmp_path / "analysis"
    analyze(config, tmp_path / "trials.csv", tmp_path / "participants.csv", tmp_path / "protocol.yaml",
            tmp_path / "tlx.csv", output)
    result = json.loads((output / "statistics.json").read_text())
    assert all(r["n_units"] == 2 for r in result)
    ratings = json.loads((output / "tlx.json").read_text())
    assert all(r["rtl_x"] == 40 for r in ratings)
