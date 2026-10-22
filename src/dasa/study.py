import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from dasa.config import write_json
from dasa.statistics import comparisons


TLX_SCALES = ["mental", "physical", "temporal", "performance", "effort", "frustration"]
METRICS = ["success_rate", "completion_seconds", "stability_index", "safety_index",
           "minimum_clearance_m", "jerk_rms"]
MEASURED_PROVENANCE = {"actual_participant_measurement", "UR10_measured_teleoperation"}
TASK_FAILURE_REASONS = {"collision", "bounds", "bounds_violation", "wrong_goal", "wrong_object",
                        "timeout", "task_failure", "failed"}


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
    practice_cells = study.get("practice_sequence", [("C1", 0.)] * 4 + [("C3", 0.)] * 4)
    if (not isinstance(practice_cells, list) or len(practice_cells) != int(study.get("practice_trials", 8))
            or any(not isinstance(cell, (list, tuple)) or len(cell) != 2 or cell[0] not in conditions
                   or isinstance(cell[1], bool) or not isinstance(cell[1], (int, float))
                   or not np.isfinite(cell[1]) or cell[1] < 0 for cell in practice_cells)):
        raise ValueError("Practice sequence must explicitly match the configured condition and trial count")
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
        "sequence_assignments": assigned.tolist(), "practice_trials_per_participant": len(practice_cells),
        "scene_matching": "same_participant_delay_repeat_has_same_scene_seed_and_goal_across_methods",
        "balance_limit": "complete_balance_requires_multiples_of_the_number_of_sequences",
    })


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_representation_schedule(config, participants_csv, output):
    registrations = read_csv(participants_csv)
    identifiers = [row.get("participant_id") for row in registrations]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Representation acquisition requires unique registered participant IDs")
    included = [row for row in registrations if _participant_status(row) == "included"]
    if not included or any(row.get("record_status", "collected") != "collected" for row in included):
        raise ValueError("Representation acquisition requires actual included participant registrations")
    rows = []
    for index, participant in enumerate(included):
        cells = [(goal, repeat) for goal in range(3) for repeat in range(2)]
        order = np.random.default_rng(config["seed"] + 12000000 + index).permutation(len(cells))
        for position, cell in enumerate(order):
            goal, repeat = cells[int(cell)]
            rows.append({"trial_id": f"{participant['participant_id']}_repr_g{goal}_r{repeat:02d}",
                         "participant_id": participant["participant_id"], "order": position + 1,
                         "condition": "C1", "delay_seconds": .4, "command_delay_seconds": 0.,
                         "scene_seed": int(config["seed"]) + 12000000 + repeat,
                         "intended_goal": goal, "repeat": repeat, "practice": False,
                         "stage": "intent_representation", "record_status": "planned",
                         "layout_group": f"intent_representation_r{repeat:02d}",
                         "experience_group": participant.get("experience_group", "")})
    output = Path(output)
    if output.exists() or output.with_suffix(".design.json").exists():
        raise FileExistsError("Representation schedules preserve existing assignments")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    write_json(output.with_suffix(".design.json"), {
        "status": "prospective_acquisition_schedule", "stage": "intent_representation",
        "participant_registry_sha256": hashlib.sha256(Path(participants_csv).read_bytes()).hexdigest(),
        "n_registered_included": len(included), "goals": [0, 1, 2], "repetitions_per_goal": 2,
        "physical_acquisition_condition": "C1", "one_way_feedback_delay_seconds": .4,
        "evaluation": "same_recorded_episodes_across_offline_estimators",
        "scene_matching": "common_measured_layout_group_by_repeat_across_registered_IDs_and_goals",
        "seed": config["seed"], "n_planned_trials": len(rows), "outcomes_generated": False,
    })


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
    if protocol.get("record_status") not in {"prospective", "collected"}:
        raise ValueError("Analysis requires a frozen prospective or collected protocol")
    try:
        datetime.fromisoformat(str(protocol["analysis_plan_frozen_on"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("The analysis freeze must contain an actual ISO date or timestamp") from exc
    if protocol["record_status"] == "prospective" and (not isinstance(protocol.get("config_sha256"), str)
            or len(protocol["config_sha256"]) != 64):
        raise ValueError("A prospective analysis plan requires its frozen configuration hash")
    if protocol["record_status"] == "prospective" and (not isinstance(protocol.get("source_sha256"), dict)
            or not protocol["source_sha256"]):
        raise ValueError("A prospective analysis plan requires its frozen source hashes")
    if int(protocol.get("schema_version", 1)) >= 2:
        for field in ("trial_timeout_seconds", "scoring", "exclusion_rules"):
            if not _filled(protocol.get(field)):
                raise ValueError(f"Study configuration required: {field}")
        if _number(protocol["trial_timeout_seconds"], "trial_timeout_seconds") <= 0:
            raise ValueError("trial_timeout_seconds must be positive")
        if not isinstance(protocol["scoring"], dict) or not protocol["scoring"]:
            raise ValueError("Scoring definitions must be an explicit nonempty mapping")
    if not isinstance(protocol.get("exclusion_rules"), dict):
        raise ValueError("Technical exclusion rules must be an explicit mapping")
    exclusions = protocol["exclusion_rules"].get("excluded")
    if (not isinstance(exclusions, list) or not all(isinstance(value, str) and value for value in exclusions)
            or TASK_FAILURE_REASONS.intersection(exclusions)):
        raise ValueError("Declare technical exclusion codes; task failures cannot be exclusion rules")
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
    "data_provenance", "questionnaire", "trial_repeats",
    "config_sha256", "source_sha256", "source_git_commit", "source_provenance",
}


def _measurement_provenance(row, protocol):
    provenance = row.get("provenance") or protocol.get("data_provenance")
    if provenance not in MEASURED_PROVENANCE:
        raise ValueError("Collected study records require explicit physical-measurement provenance")
    return provenance


def _questionnaire(protocol, config):
    specification = protocol.get("questionnaire", config.get("analysis", {}).get("questionnaire"))
    if not isinstance(specification, dict):
        raise ValueError("Declare the original questionnaire scale, anchors and performance direction")
    lower = _number(specification.get("scale_min"), "questionnaire.scale_min")
    upper = _number(specification.get("scale_max"), "questionnaire.scale_max")
    direction = specification.get("performance_direction")
    if lower >= upper or direction not in {"higher_is_worse", "higher_is_better"}:
        raise ValueError("Invalid declared questionnaire scale or performance direction")
    anchors = specification.get("anchors")
    if not isinstance(anchors, dict) or any(not isinstance(anchors.get(scale), dict)
            or not all(isinstance(anchors[scale].get(edge), str) and anchors[scale][edge].strip()
                       for edge in ("low", "high")) for scale in TLX_SCALES):
        raise ValueError("Each questionnaire scale requires explicit low and high anchors")
    return {"scale_min": lower, "scale_max": upper, "performance_direction": direction,
            "anchors": {scale: dict(anchors[scale]) for scale in TLX_SCALES},
            "normalization": "100_times_original_burden_score_minus_min_divided_by_scale_range",
            "performance_reversal": "min_plus_max_minus_original" if direction == "higher_is_better" else "none",
            "score_assumption": "declared_scores_treated_as_equal_intervals_for_arithmetic_RTLX",
            "parameter_provenance": "configured_operational_values"}


def _original_statistics(statistics, specification):
    factor = (specification["scale_max"] - specification["scale_min"]) / 100.
    return [{**row, "difference_mean": row["difference_mean"] * factor
             if row.get("difference_mean") is not None else None,
             "ci95": [value * factor for value in row["ci95"]] if row.get("ci95") is not None else None,
             "score_scale": "original_burden_oriented", "score_unit": "original_scale_points",
             "p_value_source": "tlx_statistics.json", "affine_scale_factor": factor}
            for row in statistics]


def analyze(config, trials_path, participants_path, protocol_path, tlx_path, output):

    participants = read_csv(participants_path)
    with Path(protocol_path).open(encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    validate_protocol(protocol, participants)
    if protocol.get("config_sha256") and protocol["config_sha256"] != hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError("The analysis configuration differs from the frozen study plan")
    protocol = {key: value for key, value in protocol.items() if key in PUBLIC_PROTOCOL_FIELDS}
    known = {p["participant_id"] for p in participants}
    included = {p["participant_id"] for p in participants if _participant_status(p) == "included"}
    conditions = protocol.get("conditions", ["C1", "C2", "C3", "C4"])
    declared_delays = protocol.get("feedback_delays_seconds")
    raw = read_csv(trials_path)
    rows, excluded = [], []
    ids = set()
    timeout = protocol.get("trial_timeout_seconds")
    allowed_exclusions = set(protocol["exclusion_rules"]["excluded"])
    for r in raw:
        if r["participant_id"] not in known or not r["trial_id"] or r["trial_id"] in ids:
            raise ValueError("Unknown participant, empty identifier or duplicate trial")
        ids.add(r["trial_id"])
        if r.get("record_status") != "collected":
            raise ValueError("Every observed trial row must explicitly declare record_status=collected")
        provenance = _measurement_provenance(r, protocol)
        if r["participant_id"] not in included:
            excluded.append({"trial_id": r["trial_id"], "reason": "participant_not_included"})
            continue
        if _boolean(r.get("practice", "false"), "practice"):
            excluded.append({"trial_id": r["trial_id"], "reason": "practice"})
            continue
        if r.get("exclude_reason"):
            reason = r["exclude_reason"].strip()
            if reason in TASK_FAILURE_REASONS or reason not in allowed_exclusions:
                raise ValueError("Only declared technical exclusions are allowed; retain task failures")
            if reason == "practice":
                raise ValueError("Practice exclusions require a recorded practice=true flag")
            excluded.append({"trial_id": r["trial_id"], "reason": reason})
            continue
        if r["condition"] not in conditions:
            raise ValueError("Trial condition not declared in the collected protocol")
        delay = _number(r["delay_seconds"], "delay_seconds", True)
        if declared_delays is not None and delay not in declared_delays:
            raise ValueError("Trial delay not declared in the collected protocol")
        success = _boolean(r["success"], "success")
        row = {"unit": r["participant_id"], "trial_id": r["trial_id"], "condition": r["condition"],
               "delay_seconds": delay, "success_rate": float(success),
               "provenance": provenance, "termination": r.get("termination") or None}
        for field in ("completion_seconds", "stability_index", "safety_index", "minimum_clearance_m", "jerk_rms"):
            row[field] = _number(r[field], field, field != "minimum_clearance_m") if r.get(field) else None
        if any(row[field] is not None and row[field] > 1 for field in ("stability_index", "safety_index")):
            raise ValueError("Safety and stability indices must lie in [0, 1]")
        if success and row["completion_seconds"] is None:
            raise ValueError("A successful trial requires an observed completion time")
        if not success:
            row["completion_seconds"] = None
        if timeout is not None:
            if r.get("timeout_seconds") and not np.isclose(float(timeout), _number(
                    r["timeout_seconds"], "recorded_timeout_seconds", True), atol=1e-9, rtol=0):
                raise ValueError("Collected trial timeout differs from the frozen study protocol")
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
    metric_digest = hashlib.sha256((output / "trial_metrics.json").read_bytes()).hexdigest()
    for row in statistics:
        row.update(metric_source_sha256=metric_digest, metric_source_file="trial_metrics.json")
    write_json(output / "statistics.json", statistics)
    _write_csv(output / "statistics.csv", statistics)
    write_json(output / "sharing/metrics.json", rows)
    write_json(output / "sharing/statistics.json", statistics)
    _write_csv(output / "sharing/statistics.csv", statistics)
    workload_audit = {}
    if tlx_path:
        specification = _questionnaire(protocol, config)
        lower, upper = specification["scale_min"], specification["scale_max"]
        ratings, original, original_burden, seen = [], [], [], set()
        for r in read_csv(tlx_path):
            if r["participant_id"] not in known or r["condition"] not in conditions:
                raise ValueError("Unknown participant or condition in workload ratings")
            if r.get("record_status") != "collected":
                raise ValueError("Every workload row must explicitly declare record_status=collected")
            provenance = _measurement_provenance(r, protocol)
            if r["participant_id"] not in included:
                continue
            delay = _number(r["delay_seconds"], "delay_seconds", True)
            if declared_delays is not None and delay not in declared_delays:
                raise ValueError("Workload delay not declared in the collected protocol")
            if protocol.get("tlx_delays_seconds") is not None and delay not in protocol["tlx_delays_seconds"]:
                raise ValueError("Workload delay differs from the frozen questionnaire collection plan")
            key = (r["participant_id"], r["condition"], delay)
            if key in seen:
                raise ValueError("Duplicate condition-level workload rating")
            seen.add(key)
            values = np.array([_number(r[s], s) for s in TLX_SCALES])
            if np.any((values < lower) | (values > upper)):
                raise ValueError("A workload rating falls outside its explicitly declared original scale")
            burden = values.copy()
            if specification["performance_direction"] == "higher_is_better":
                burden[TLX_SCALES.index("performance")] = lower + upper - burden[TLX_SCALES.index("performance")]
            normalized = 100. * (burden - lower) / (upper - lower)
            identity = {"unit": key[0], "condition": key[1], "delay_seconds": key[2], "provenance": provenance}
            original.append({**identity, **dict(zip(TLX_SCALES, values.tolist())),
                             "performance_burden": float(burden[TLX_SCALES.index("performance")]),
                             "rtl_x_burden": float(burden.mean()), "score_scale": "original_literal_responses"})
            original_burden.append({**identity, "rtl_x": float(burden.mean()),
                                   **dict(zip(TLX_SCALES, burden.tolist())),
                                   "score_scale": "original_burden_oriented"})
            ratings.append({**identity, "rtl_x": float(normalized.mean()),
                            **dict(zip(TLX_SCALES, normalized.tolist())),
                            "score_scale": "normalized_0_to_100_burden", "original_scale_min": lower,
                            "original_scale_max": upper})
        if not ratings:
            raise ValueError("The supplied questionnaire file has no included collected responses")
        tlx_delays = protocol.get("tlx_delays_seconds", sorted({r["delay_seconds"] for r in ratings}))
        tlx_statistics = comparisons(ratings, "C3", [c for c in conditions if c != "C3"],
                                     ["rtl_x", *TLX_SCALES], config,
                                     family="workload_RTLX_and_six_subscales_all_methods_delays",
                                     planned_delays=tlx_delays)
        for row in tlx_statistics:
            row.update(score_scale="normalized_0_to_100_burden", score_unit="normalized_points",
                       original_scale_min=lower, original_scale_max=upper,
                       original_statistics_file="tlx_original_statistics.csv")
        write_json(output / "tlx.json", ratings)
        rating_digest = hashlib.sha256((output / "tlx.json").read_bytes()).hexdigest()
        for row in tlx_statistics:
            row.update(metric_source_sha256=rating_digest, metric_source_file="tlx.json")
        _write_csv(output / "tlx_normalized.csv", ratings)
        write_json(output / "tlx_original.json", original)
        _write_csv(output / "tlx_original.csv", original)
        write_json(output / "tlx_original_burden.json", original_burden)
        _write_csv(output / "tlx_original_burden.csv", original_burden)
        write_json(output / "tlx_statistics.json", tlx_statistics)
        _write_csv(output / "tlx_statistics.csv", tlx_statistics)
        original_statistics = _original_statistics(tlx_statistics, specification)
        write_json(output / "tlx_original_statistics.json", original_statistics)
        _write_csv(output / "tlx_original_statistics.csv", original_statistics)
        write_json(output / "questionnaire_contract.json", specification)
        from dasa.study_reporting import render_workload
        render_workload(original_burden, output,
                        delay=float(protocol.get("workload_figure_delay_seconds", .8)),
                        specification=specification, normalized=ratings)
        workload_audit = {"n_ratings": len(ratings), "declared_delays": tlx_delays,
                          "questionnaire": specification,
                          "statistical_family": "one_family_seven_burden_outcomes_not_duplicated_by_affine_scale",
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
    success_rates = []
    for participant in sorted(included):
        for condition in conditions:
            for delay in delays:
                observed = [r for r in rows if r["unit"] == participant and r["condition"] == condition
                            and r["delay_seconds"] == delay]
                successes = sum(r["success_rate"] for r in observed)
                success_rates.append({"participant_id": participant, "condition": condition,
                                      "delay_seconds": delay, "n_trials": len(observed),
                                      "n_successes": int(successes),
                                      "success_rate": successes / len(observed) if observed else None,
                                      "provenance": "actual_participant_measurement"})
    write_json(output / "success_rates.json", success_rates)
    _write_csv(output / "success_rates.csv", success_rates)
    write_json(output / "analysis_audit.json", {
        "input_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in [trials_path, participants_path, protocol_path, *([tlx_path] if tlx_path else [])]},
        "n_input_trials": len(raw), "n_analyzed_trials": len(rows), "excluded_trials": excluded,
        "missing_task_cells": missing_cells, "workload": workload_audit,
        "missing_metric_counts": {m: sum(r.get(m) is None for r in rows) for m in metrics},
        "unit_trial_counts": [dict(participant_id=p, condition=c, delay_seconds=d, n_trials=n)
                              for (p, c, d), n in sorted(counts.items())],
        "declared_repetitions_per_cell": protocol.get("trial_repeats", config.get("study", {}).get("trial_repeats", 1)),
        "success_rate_denominator": "actual_nonexcluded_binary_trial_outcomes_in_each_ID_condition_delay_cell",
        "success_rate_table": "success_rates.csv",
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
    planned_count = protocol.get("expected_participants", config.get("study", {}).get("participants"))
    task_complete = not missing_cells and all(row["n_trials"] == protocol.get(
        "trial_repeats", config.get("study", {}).get("trial_repeats", 1)) for row in success_rates)
    workload_complete = not protocol.get("tlx_delays_seconds") or bool(
        tlx_path and workload_audit.get("n_ratings") and not workload_audit.get("missing_cells"))
    write_json(output / "study_metadata.json", {
        "protocol": protocol,
        "analysis_record_status": "collected_measurements_analyzed",
        "task_cell_coverage_complete": task_complete, "workload_cell_coverage_complete": workload_complete,
        "cohort_coverage_complete": bool(task_complete and workload_complete
                                         and (planned_count is None or len(included) == planned_count)),
        "protocol_schema_version": int(protocol.get("schema_version", 1)),
        "validation_level": "version2_analysis_contract" if int(protocol.get("schema_version", 1)) >= 2
                            else "legacy_analysis_contract",
        "analysis_scope": "paired participant means; incomplete cells retained and pairwise missingness reported",
    })
    write_json(output / "analysis_plan.json", {
        "unit": "participant", "aggregation": "arithmetic_mean_of_repeated_trials_within_participant_condition_delay",
        "null": "paired_mean_difference_zero", "direction": "C3_minus_comparator",
        "test": "two_sided_paired_sign_permutation_exact_n_at_most_20_else_Monte_Carlo_plus_one",
        "multiplicity": "Holm_FWER_.05_separate_prespecified_task_and_workload_families",
        "uncertainty": "95_percent_pointwise_paired_participant_percentile_bootstrap; not_simultaneous_CI",
        "effect": "raw_mean_difference_and_Cohen_dz; undefined_for_zero_difference_SD",
        "assumptions": "independent_participants; sign_exchangeability_of_paired_differences_under_null",
        "missingness": "pairwise_complete_participant_cells; no_imputation; failures_do_not_have_completion_times",
        "selection_caution": "success_conditioned_completion_times_can_exhibit_selection_bias",
        "metric_units": {"success_rate": "fraction", "completion_seconds": "s", "stability_index": "1",
                         "safety_index": "1", "minimum_clearance_m": "m",
                         "jerk_rms": "m/s^3", "timeout_penalized_seconds": "s", "rtl_x": "0_to_100_points"},
    })
    artifacts = ["statistics.json", "statistics.csv", "analysis_plan.json", "success_rates.csv"]
    if tlx_path:
        artifacts += ["tlx_statistics.json", "tlx_statistics.csv", "tlx_original_statistics.json",
                      "tlx_original_statistics.csv", "questionnaire_contract.json", "tlx_original.csv",
                      "tlx_original_burden.csv", "tlx_normalized.csv"]
    write_json(output / "analysis_artifacts.json", {
        "provenance": "actual_participant_measurement", "task_family": statistics[0]["family"] if statistics else None,
        "task_family_size": len(statistics), "comparison_direction": "C3_minus_comparator",
        "p_value_columns": ["p_raw", "p_holm"], "effect_columns": ["difference_mean", "effect_dz"],
        "interval_columns": ["ci95", "effect_dz_ci95"], "interval_scope": "pointwise_not_simultaneous",
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifacts},
        "frozen_plan_sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
        "analysis_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "analysis_source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("study.py", "statistics.py", "metrics.py", "study_reporting.py")},
        "figure_statistics": {"figure6": "statistics.csv", "figure8": "tlx_original_statistics.csv" if tlx_path else None},
    })
