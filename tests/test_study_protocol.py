import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from dasa.statistics import comparisons, holm, paired_inference
from dasa.study import analyze, read_csv, validate_protocol, write_schedule


def test_williams_schedule_is_reproducible_and_nearly_position_balanced(config, tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_schedule(config, first)
    write_schedule(config, second)
    assert first.read_bytes() == second.read_bytes()
    rows = read_csv(first)
    assert len(rows) == 320
    counts = Counter((r["order"], r["condition"], r["delay_seconds"]) for r in rows)
    assert set(counts.values()) == {1, 2}
    for identifier in {r["participant_id"] for r in rows}:
        selected = [r for r in rows if r["participant_id"] == identifier]
        for delay in {r["delay_seconds"] for r in selected}:
            paired = [r for r in selected if r["delay_seconds"] == delay]
            assert len({r["scene_seed"] for r in paired}) == 1
            assert len({r["intended_goal"] for r in paired}) == 1
    practices = read_csv(tmp_path / "a_practice.csv")
    assert len(practices) == 20 * 8
    assert {r["scene_seed"] for r in rows}.isdisjoint(r["scene_seed"] for r in practices)
    assert all(r["record_status"] == "planned" for r in practices + rows)


def test_collected_protocol_requires_analysis_contract_and_rejects_planned_status():
    root = Path(__file__).resolve().parents[1]
    template = yaml.safe_load((root / "study/protocol.template.yaml").read_text())
    with pytest.raises(ValueError, match="collected"):
        validate_protocol(template, [])
    with pytest.raises(ValueError, match="study_id"):
        validate_protocol({}, [])
    minimal = {"study_id": "unit-test-study", "analysis_plan_frozen_on": "2026-10-20"}
    with pytest.raises(ValueError, match="collected"):
        validate_protocol({**minimal, "record_status": "planned"}, [])


def test_multiplicity_can_include_unestimable_planned_contrasts():
    assert holm([.01, None], family_size=2) == [.02, None]
    with pytest.raises(ValueError, match="probabilities"):
        holm([float("nan")])
    with pytest.raises(ValueError, match="equal length"):
        paired_inference([1, 2], [1], 0)


def test_missing_pairs_and_full_declared_family_are_reported(config):
    rows = [dict(unit="P1", condition="C3", delay_seconds=.8, success_rate=1),
            dict(unit="P2", condition="C3", delay_seconds=.8, success_rate=0),
            dict(unit="P1", condition="C1", delay_seconds=.8, success_rate=0)]
    stats = comparisons(rows, "C3", ["C1"], ["success_rate"], config, planned_delays=[0, .8])
    assert len(stats) == 2
    assert stats[0]["p_raw"] is None
    assert stats[1]["n_units"] == 1
    assert stats[1]["reference_only_units"] == ["P2"]
    assert stats[1]["family_size_planned"] == 2
    assert stats[1]["units"] == ["P1"]


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_actual_failure_timeout_and_audit_are_preserved(config, tmp_path):
    protocol = dict(study_id="unit-test-study", analysis_plan_frozen_on="2026-10-20",
                    trial_timeout_seconds=60, conditions=["C1", "C3"], feedback_delays_seconds=[.8],
                    expected_participants=20)
    (tmp_path / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    participants = [dict(participant_id=f"T{i}", inclusion_status="included") for i in range(2)]
    _csv(tmp_path / "participants.csv", participants)
    trials = [dict(trial_id=f"{i}_{method}", participant_id=f"T{i}", condition=method,
                   delay_seconds=.8, success="false" if method == "C1" else "true",
                   completion_seconds="" if method == "C1" else 30, elapsed_seconds=60 if method == "C1" else 30,
                   minimum_clearance_m=.01, jerk_rms=2, practice="false")
              for i in range(2) for method in ("C1", "C3")]
    _csv(tmp_path / "trials.csv", trials)
    out = tmp_path / "out"
    analyze(config, tmp_path / "trials.csv", tmp_path / "participants.csv", tmp_path / "protocol.yaml", None, out)
    rows = json.loads((out / "trial_metrics.json").read_text())
    assert all(r["completion_seconds"] is None and r["timeout_penalized_seconds"] == 60
               for r in rows if r["condition"] == "C1")
    stats = json.loads((out / "statistics.json").read_text())
    time = next(r for r in stats if r["metric"] == "completion_seconds")
    assert time["p_raw"] is None and time["n_units"] == 0
    penalty = next(r for r in stats if r["metric"] == "timeout_penalized_seconds")
    assert penalty["difference_mean"] == -30
    metadata = json.loads((out / "cohort_summary.json").read_text())
    assert metadata["n_registered"] == 2 and metadata["declared_expected_participants"] == 20
    assert metadata["n_included"] == 2 and metadata["n_withdrawn"] == 0
    assert not (out / "participants.csv").exists()
    assert (out / "statistics.csv").exists()


def test_monte_carlo_metadata_and_effect_intervals_are_finite():
    left = np.linspace(0, 1, 20)
    stats = paired_inference(left, np.zeros(20), seed=17, bootstrap=1000, permutations=2000)
    assert stats["test"] == "monte_carlo_paired_sign_permutation"
    assert stats["permutations_used"] == 2000
    assert 0 < stats["p_raw"] <= 1
    assert stats["monte_carlo_standard_error"] > 0
    assert len(stats["effect_dz_ci95"]) == 2
    assert stats == paired_inference(left, np.zeros(20), seed=17, bootstrap=1000, permutations=2000)


def test_anonymous_registration_filters_withdrawals_and_ignores_extra_fields(config, tmp_path):
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load((root / "study/protocol.template.yaml").read_text())
    protocol.update(record_status="collected", conditions=["C1", "C3"], feedback_delays_seconds=[.8],
                    private_extra="unexported_protocol_value")
    (tmp_path / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    participants = [dict(participant_id="T0", inclusion_status="included", private_extra="unexported_roster_value"),
                    dict(participant_id="T1", inclusion_status="withdrawn", private_extra="unused")]
    _csv(tmp_path / "participants.csv", participants)
    trials = [dict(trial_id=f"trial_{i}", participant_id=f"T{i}", condition="C3", delay_seconds=.8,
                   success="true", completion_seconds=30, elapsed_seconds=30,
                   minimum_clearance_m=.1, jerk_rms=1, practice="false") for i in range(2)]
    _csv(tmp_path / "trials.csv", trials)
    output = tmp_path / "output"
    analyze(config, tmp_path / "trials.csv", tmp_path / "participants.csv", tmp_path / "protocol.yaml", None, output)
    summary = json.loads((output / "cohort_summary.json").read_text())
    assert summary["n_registered"] == 2 and summary["n_included"] == 1 and summary["n_withdrawn"] == 1
    audit = json.loads((output / "analysis_audit.json").read_text())
    assert audit["excluded_trials"] == [{"trial_id": "trial_1", "reason": "participant_not_included"}]
    for path in output.glob("*.json"):
        assert "unexported_" not in path.read_text()
