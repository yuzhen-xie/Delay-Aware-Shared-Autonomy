import csv
import json

import numpy as np
import pytest

from dasa.figures import build, eligible, unit_summary


def test_unit_summary_weights_independent_units_and_retains_missingness():
    rows = [
        {"condition": "C3", "delay_seconds": .8, "unit": "a", "score": 0},
        {"condition": "C3", "delay_seconds": .8, "unit": "a", "score": 0},
        {"condition": "C3", "delay_seconds": .8, "unit": "b", "score": 1},
        {"condition": "C3", "delay_seconds": .8, "unit": "c", "score": None},
    ]
    units, summary = unit_summary(rows, ["score"], ("condition", "delay_seconds"), 7, 500)
    assert len(units) == 2
    assert summary[0]["mean"] == .5
    assert summary[0]["n_units"] == 2
    assert summary[0]["n_units_missing"] == 1
    assert summary[0]["n_observed_trials"] == 3
    assert summary[0]["ci95"] is not None
    assert unit_summary(rows, ["score"], ("condition", "delay_seconds"), 7, 500)[1] == summary


def test_single_unit_has_no_zero_width_inferential_interval():
    rows = [{"condition": "C3", "scenario": "novel", "repeat": 0, "score": 1}]
    _, summary = unit_summary(rows, ["score"], ("condition", "scenario"), 7, 500)
    assert summary[0]["ci95"] is None
    assert summary[0]["ci_method"] == "insufficient_independent_units"
    with pytest.raises(ValueError, match="Nonfinite"):
        unit_summary([{**rows[0], "score": np.nan}], ["score"], ("condition", "scenario"), 7)


def test_figure_exports_exclude_unmet_budget_and_record_input_hashes(config, tmp_path):
    config["experiments"]["bootstrap_samples"] = 100
    directory = tmp_path / "refinement"
    directory.mkdir()
    rows = [
        {"condition": "C1", "scenario": "novel", "unit": "r0", "repeat": 0,
         "provenance": "synthetic_task_space_simulation", "success_rate": .4,
         "reference_error_m": .1, "refinement_quota_met": True},
        {"condition": "C3", "scenario": "novel", "unit": "r0", "repeat": 0,
         "provenance": "synthetic_task_space_simulation", "success_rate": .9,
         "reference_error_m": .01, "refinement_quota_met": False},
    ]
    source = directory / "metrics.json"
    source.write_text(json.dumps(rows))
    (directory / "statistics.json").write_text("[]")
    assert eligible(rows[0])
    assert not eligible(rows[1])
    record = build(config, tmp_path)
    assert record["figure_map"]["learning"]["n_incomplete_budget_rows_excluded"] == 1
    assert record["figure_map"]["figure5"]["status"] == "missing_input"
    assert str(source) in record["input_sha256"]
    output = tmp_path / "figures"
    for extension in ("png", "pdf", "svg"):
        assert (output / f"learning.{extension}").stat().st_size > 100
    with (output / "learning_units.csv").open() as stream:
        units = list(csv.DictReader(stream))
    assert {row["condition"] for row in units} == {"C1"}
    assert (output / "learning_incomplete_budget_diagnostics.csv").exists()
    assert (output / "learning_statistics.csv").exists()
