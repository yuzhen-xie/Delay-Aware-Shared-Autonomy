import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from dasa.config import write_json
from dasa.statistics import comparisons


TLX_SCALES = ["mental", "physical", "temporal", "performance", "effort", "frustration"]
METRICS = ["success_rate", "completion_seconds", "minimum_clearance_m", "jerk_rms"]


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
                         for r in rows)


def write_schedule(config: dict, output: Path):

    study = config.get("study", {})
    conditions = list(study.get("conditions", ["C1", "C2", "C3", "C4"]))
    delays = list(study.get("feedback_delays_seconds", config["experiments"]["sharing_delays"]))
    cells = [(name, delay) for delay in delays for name in conditions]
    participants, repeats = int(study.get("participants", 20)), int(study.get("trial_repeats", 1))
    if participants < 1 or repeats < 1 or len(cells) < 2:
        raise ValueError("Positive participant/repeat counts and at least two conditions required")
    n = len(cells)
    first = [0]
    for i in range(1, n):
        first.append((i + 1) // 2 if i % 2 else n - i // 2)
    sequences = [[(x + row) % n for x in first] for row in range(n)]
    if n % 2:
        sequences += [list(reversed(row)) for row in sequences]
    rng = np.random.default_rng(config["seed"])
    label_permutation = rng.permutation(n)
    assigned = rng.permutation(np.arange(participants) % len(sequences))
    rows, practice = [], []
    practice_cells = [("C1", 0.), ("C1", 0.), ("C1", .4), ("C1", .8),
                      ("C3", 0.), ("C3", .4), ("C3", .8), ("C3", .8)]
    for participant in range(participants):
        identifier = f"P{participant + 1:02d}"
        for repeat in range(repeats):
            sequence = sequences[(int(assigned[participant]) + repeat) % len(sequences)]
            for index, position in enumerate(sequence):
                cell = int(label_permutation[position])
                name, delay = cells[cell]
                delay_index = delays.index(delay)
                rows.append({"trial_id": f"{identifier}_r{repeat:02d}_c{cell:02d}",
                             "participant_id": identifier, "order": repeat * n + index + 1,
                             "condition": name, "delay_seconds": delay,
                             "command_delay_seconds": float(study.get("command_delay_seconds", 0.)),
                             "scene_seed": int(config["seed"]) + 1000000 + participant * 10000
                                           + repeat * 100 + delay_index,
                             "intended_goal": (participant + repeat + delay_index) % 3,
                             "practice": False, "repeat": repeat,
                             "order_sequence": int(assigned[participant]), "record_status": "planned"})
        for index, (name, delay) in enumerate(practice_cells):
            practice.append({"trial_id": f"{identifier}_practice_{index + 1:02d}",
                             "participant_id": identifier, "order": index + 1, "condition": name,
                             "delay_seconds": delay, "command_delay_seconds": 0.,
                             "scene_seed": int(config["seed"]) + 9000000 + participant * 100 + index,
                             "intended_goal": index % 3, "practice": True,
                             "record_status": "planned"})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    _write_csv(output.with_name(output.stem + "_practice.csv"), practice)
    write_json(output.with_suffix(".design.json"), {
        "status": "prospective_schedule_not_a_record_of_completed_trials",
        "design": "Williams_balanced_first_order_carryover_sequences",
        "seed": config["seed"], "n_participants_planned": participants,
        "n_cells": n, "trial_repeats": repeats, "cell_label_permutation": label_permutation.tolist(),
        "sequence_assignments": assigned.tolist(), "practice_trials_per_participant": 8,
        "scene_matching": "same_participant_delay_repeat_has_same_scene_seed_and_goal_across_methods",
        "balance_limit": "complete_balance_requires_multiples_of_the_number_of_sequences",
    })


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _filled(value):
    return value not in (None, "")


def _boolean(value, name):
    if str(value).lower() not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return str(value).lower() == "true"


def _number(value, name, nonnegative=False):
    parsed = float(value)
    if not np.isfinite(parsed) or (nonnegative and parsed < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return parsed


def _participant_status(participant):
    status = participant.get("inclusion_status")
    if status:
        if status not in {"included", "excluded", "withdrawn"}:
            raise ValueError("inclusion_status must be included, excluded or withdrawn")
        return status
    if participant.get("included"):
        return "included" if _boolean(participant["included"], "included") else "excluded"
    return "included"


def validate_protocol(protocol: dict, participants: list[dict]):
    for field in ("study_id", "analysis_plan_frozen_on"):
        if not _filled(protocol.get(field)):
            raise ValueError(f"Study configuration required: {field}")
    if protocol.get("record_status", "collected") != "collected":
        raise ValueError("Only a collected protocol record may be analyzed")
    if int(protocol.get("schema_version", 1)) >= 2:
        for field in ("trial_timeout_seconds", "scoring", "exclusion_rules"):
            if not _filled(protocol.get(field)):
                raise ValueError(f"Study configuration required: {field}")
        if _number(protocol["trial_timeout_seconds"], "trial_timeout_seconds") <= 0:
            raise ValueError("trial_timeout_seconds must be positive")
    if not participants:
        raise ValueError("No anonymous participant registrations")
    ids = [p["participant_id"] for p in participants]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate participant identifiers")
    for participant in participants:
        _participant_status(participant)


PUBLIC_PROTOCOL_FIELDS = {
    "schema_version", "record_status", "study_id", "analysis_plan_frozen_on", "design",
    "expected_participants", "conditions", "feedback_delays_seconds", "command_delay_seconds",
    "tlx_delays_seconds", "workload_figure_delay_seconds", "trial_timeout_seconds",
    "timing_tolerance_seconds", "randomization_seed", "primary_outcomes", "secondary_outcomes",
    "unit_of_analysis", "test", "multiple_comparison_correction", "confidence_intervals",
    "task_family", "workload_family", "missingness", "tlx_performance_direction", "scoring",
    "exclusion_rules", "practice_trials", "practice_trial_timeout_seconds", "practice_sequence",
}


def analyze(config, trials_path, participants_path, protocol_path, tlx_path, output):

    participants = read_csv(participants_path)
    with Path(protocol_path).open(encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    validate_protocol(protocol, participants)
    protocol = {key: value for key, value in protocol.items() if key in PUBLIC_PROTOCOL_FIELDS}
    known = {p["participant_id"] for p in participants}
    included = {p["participant_id"] for p in participants if _participant_status(p) == "included"}
    conditions = protocol.get("conditions", ["C1", "C2", "C3", "C4"])
    declared_delays = protocol.get("feedback_delays_seconds")
    raw = read_csv(trials_path)
    rows, excluded = [], []
    ids = set()
    timeout = protocol.get("trial_timeout_seconds")
    for r in raw:
        if r["participant_id"] not in known or not r["trial_id"] or r["trial_id"] in ids:
            raise ValueError("Unknown participant, empty identifier or duplicate trial")
        ids.add(r["trial_id"])
        if r["participant_id"] not in included:
            excluded.append({"trial_id": r["trial_id"], "reason": "participant_not_included"})
            continue
        if r.get("record_status", "collected") != "collected":
            raise ValueError("Planned trial rows are not observations")
        if _boolean(r.get("practice", "false"), "practice"):
            excluded.append({"trial_id": r["trial_id"], "reason": "practice"})
            continue
        if r.get("exclude_reason"):
            excluded.append({"trial_id": r["trial_id"], "reason": r["exclude_reason"]})
            continue
        if r["condition"] not in conditions:
            raise ValueError("Trial condition not declared in the collected protocol")
        delay = _number(r["delay_seconds"], "delay_seconds", True)
        if declared_delays is not None and delay not in declared_delays:
            raise ValueError("Trial delay not declared in the collected protocol")
        success = _boolean(r["success"], "success")
        row = {"unit": r["participant_id"], "trial_id": r["trial_id"], "condition": r["condition"],
               "delay_seconds": delay, "success_rate": float(success),
               "provenance": "actual_participant_measurement", "termination": r.get("termination") or None}
        for field in ("completion_seconds", "minimum_clearance_m", "jerk_rms"):
            row[field] = _number(r[field], field, field != "minimum_clearance_m") if r.get(field) else None
        if success and row["completion_seconds"] is None:
            raise ValueError("A successful trial requires an observed completion time")
        if not success:
            row["completion_seconds"] = None
        if timeout is not None:
            elapsed = _number(r["elapsed_seconds"], "elapsed_seconds", True) \
                if r.get("elapsed_seconds") else row["completion_seconds"]
            if elapsed is None:
                raise ValueError("Failed trials require elapsed_seconds when timeout is declared")
            if elapsed > float(timeout) + float(protocol.get("timing_tolerance_seconds", .02)):
                raise ValueError("Elapsed trial duration exceeds the declared timeout")
            row["timeout_penalized_seconds"] = row["completion_seconds"] if success else float(timeout)
            row["elapsed_seconds"] = elapsed
        rows.append(row)
    if not rows:
        raise ValueError("No analyzable non-practice observations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = METRICS + (["timeout_penalized_seconds"] if timeout is not None else [])
    statistics = comparisons(rows, "C3", [c for c in conditions if c != "C3"], metrics, config,
                             family="task_performance_all_methods_delays_outcomes",
                             planned_delays=declared_delays)
    write_json(output / "trial_metrics.json", rows)
    write_json(output / "statistics.json", statistics)
    _write_csv(output / "statistics.csv", statistics)
    workload_audit = {}
    if tlx_path:
        ratings, seen = [], set()
        for r in read_csv(tlx_path):
            if r["participant_id"] not in known or r["condition"] not in conditions:
                raise ValueError("Unknown participant or condition in workload ratings")
            if r["participant_id"] not in included:
                continue
            delay = _number(r["delay_seconds"], "delay_seconds", True)
            if declared_delays is not None and delay not in declared_delays:
                raise ValueError("Workload delay not declared in the collected protocol")
            key = (r["participant_id"], r["condition"], delay)
            if key in seen:
                raise ValueError("Duplicate condition-level workload rating")
            seen.add(key)
            values = np.array([_number(r[s], s) for s in TLX_SCALES])
            if np.any((values < 0) | (values > 100)):
                raise ValueError("Workload scales must be finite ratings from 0 to 100")
            ratings.append({"unit": key[0], "condition": key[1], "delay_seconds": key[2],
                            "rtl_x": float(values.mean()), "provenance": "actual_participant_measurement",
                            **dict(zip(TLX_SCALES, values.tolist()))})
        tlx_delays = protocol.get("tlx_delays_seconds", sorted({r["delay_seconds"] for r in ratings}))
        tlx_statistics = comparisons(ratings, "C3", [c for c in conditions if c != "C3"],
                                     ["rtl_x", *TLX_SCALES], config,
                                     family="workload_RTLX_and_six_subscales_all_methods_delays",
                                     planned_delays=tlx_delays)
        write_json(output / "tlx.json", ratings)
        write_json(output / "tlx_statistics.json", tlx_statistics)
        _write_csv(output / "tlx_statistics.csv", tlx_statistics)
        from dasa.study_reporting import render_workload
        render_workload(ratings, output, delay=float(protocol.get("workload_figure_delay_seconds", .8)))
        workload_audit = {"n_ratings": len(ratings), "declared_delays": tlx_delays,
                          "missing_cells": [dict(participant_id=p, condition=c, delay_seconds=d)
                                            for p in sorted(included) for c in conditions for d in tlx_delays
                                            if (p, c, d) not in seen]}
    counts = {}
    for row in rows:
        key = (row["unit"], row["condition"], row["delay_seconds"])
        counts[key] = counts.get(key, 0) + 1
    delays = declared_delays if declared_delays is not None else sorted({r["delay_seconds"] for r in rows})
    missing_cells = [dict(participant_id=p, condition=c, delay_seconds=d)
                     for p in sorted(included) for c in conditions for d in delays if (p, c, d) not in counts]
    write_json(output / "analysis_audit.json", {
        "input_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in [trials_path, participants_path, protocol_path, *([tlx_path] if tlx_path else [])]},
        "n_input_trials": len(raw), "n_analyzed_trials": len(rows), "excluded_trials": excluded,
        "missing_task_cells": missing_cells, "workload": workload_audit,
        "missing_metric_counts": {m: sum(r.get(m) is None for r in rows) for m in metrics},
        "unit_trial_counts": [dict(participant_id=p, condition=c, delay_seconds=d, n_trials=n)
                              for (p, c, d), n in sorted(counts.items())],
        "completion_time_scope": "successful_trials_only; failures_retained_in_success_rate",
        "timeout_penalized_scope": "timeout_for_every_failure; diagnostic_estimand_not_actual_completion_time"
                                   if timeout is not None else "not_available_without_recorded_timeout",
    })
    write_json(output / "cohort_summary.json", {
        "n_registered": len(participants), "n_included": len(included),
        "n_withdrawn": sum(_participant_status(p) == "withdrawn" for p in participants),
        "n_excluded": sum(_participant_status(p) == "excluded" for p in participants),
        "n_with_analyzed_trials": len({row["unit"] for row in rows}),
        "declared_expected_participants": protocol.get("expected_participants"),
        "registration_scope": "anonymous_participant_ID_and_analysis_inclusion_only",
    })
    write_json(output / "study_metadata.json", {
        "protocol": protocol,
        "protocol_schema_version": int(protocol.get("schema_version", 1)),
        "validation_level": "version2_analysis_contract" if int(protocol.get("schema_version", 1)) >= 2
                            else "legacy_analysis_contract",
        "analysis_scope": "paired participant means; incomplete cells retained and pairwise missingness reported",
    })
    write_json(output / "analysis_plan.json", {
        "unit": "participant", "aggregation": "arithmetic_mean_of_repeated_trials_within_participant_condition_delay",
        "null": "paired_mean_difference_zero", "direction": "C3_minus_comparator",
        "test": "two_sided_paired_sign_permutation; exact_n<=18_otherwise_Monte_Carlo_with_plus_one_correction",
        "multiplicity": "Holm_FWER_.05_separate_prespecified_task_and_workload_families",
        "uncertainty": "95_percent_pointwise_paired_participant_percentile_bootstrap; not_simultaneous_CI",
        "effect": "raw_mean_difference_and_Cohen_dz; undefined_for_zero_difference_SD",
        "assumptions": "independent_participants; sign_exchangeability_of_paired_differences_under_null",
        "missingness": "pairwise_complete_participant_cells; no_imputation; failures_do_not_have_completion_times",
        "selection_caution": "success_conditioned_completion_times_can_exhibit_selection_bias",
        "metric_units": {"success_rate": "fraction", "completion_seconds": "s", "minimum_clearance_m": "m",
                         "jerk_rms": "m/s^3", "timeout_penalized_seconds": "s", "rtl_x": "0_to_100_points"},
    })
