"""Human-study schedules, metadata checks and analysis of actual collected observations."""

import csv
from pathlib import Path

import numpy as np
import yaml

from dasa.config import write_json
from dasa.statistics import comparisons


TLX_SCALES = ["mental", "physical", "temporal", "performance", "effort", "frustration"]


def write_schedule(config: dict, output: Path):
    conditions = ["C1", "C2", "C3", "C4"]
    cells = [(name, delay) for delay in config["experiments"]["sharing_delays"] for name in conditions]
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(cells))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["participant_id", "order", "condition", "delay_seconds",
                                                   "scene_seed", "intended_goal", "practice"])
        writer.writeheader()
        for participant in range(12):
            for index, cell in enumerate(np.roll(order, participant)):
                name, delay = cells[cell]
                writer.writerow({"participant_id": f"P{participant + 1:02d}", "order": index + 1,
                                 "condition": name, "delay_seconds": delay,
                                 "scene_seed": config["seed"] + int(cell),
                                 "intended_goal": int(cell) % 3, "practice": False})


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_protocol(protocol: dict, participants: list[dict]):
    for field in ("approval_id", "ethics_body", "consent_procedure", "screening", "recruitment",
                  "experience_definition", "practice_trials", "analysis_plan_frozen_on"):
        if protocol.get(field) in (None, "", "TO_FILL"):
            raise ValueError(f"Actual study record required: {field}")
    if not participants:
        raise ValueError("No participant records")
    ids = [p["participant_id"] for p in participants]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate participant identifiers")
    for participant in participants:
        if participant.get("consented", "").lower() != "true":
            raise ValueError("Every analyzed participant must have documented consent")
        if not participant.get("teleoperation_experience") or not participant.get("age_years"):
            raise ValueError("Missing demographics or experience metadata")


def analyze(config, trials_path, participants_path, protocol_path, tlx_path, output):
    participants = read_csv(participants_path)
    with protocol_path.open(encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    validate_protocol(protocol, participants)
    known = {p["participant_id"] for p in participants}
    raw = read_csv(trials_path)
    rows = []
    ids = set()
    for r in raw:
        if r["participant_id"] not in known or r["trial_id"] in ids:
            raise ValueError("Unknown participant or duplicate trial")
        ids.add(r["trial_id"])
        if r.get("practice", "false").lower() == "true":
            continue
        outcome = r["success"].lower()
        if outcome not in {"true", "false"}:
            raise ValueError("success must be true or false")
        success = outcome == "true"
        row = {"unit": r["participant_id"], "trial_id": r["trial_id"], "condition": r["condition"],
               "delay_seconds": float(r["delay_seconds"]), "success_rate": float(success),
               "provenance": "actual_participant_measurement"}
        for field in ("completion_seconds", "minimum_clearance_m", "jerk_rms"):
            row[field] = float(r[field]) if r.get(field) else None
        if not success:
            row["completion_seconds"] = None
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "trial_metrics.json", rows)
    write_json(output / "statistics.json", comparisons(rows, "C3", ["C1", "C2", "C4"],
                ["success_rate", "completion_seconds", "minimum_clearance_m", "jerk_rms"], config))
    if tlx_path:
        ratings, seen = [], set()
        for r in read_csv(tlx_path):
            if r["participant_id"] not in known:
                raise ValueError("Unknown participant in workload ratings")
            key = (r["participant_id"], r["condition"], float(r["delay_seconds"]))
            if key in seen:
                raise ValueError("Duplicate condition-level workload rating")
            seen.add(key)
            values = np.array([float(r[s]) for s in TLX_SCALES])
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
                raise ValueError("Workload scales must be finite ratings from 0 to 100")
            ratings.append({"unit": key[0], "condition": key[1], "delay_seconds": key[2],
                            "rtl_x": float(values.mean()),
                            **dict(zip(TLX_SCALES, values.tolist()))})
        write_json(output / "tlx.json", ratings)
        write_json(output / "tlx_statistics.json", comparisons(ratings, "C3", ["C1", "C2", "C4"],
                                                               ["rtl_x", *TLX_SCALES], config))
    ages = [float(p["age_years"]) for p in participants]
    write_json(output / "study_metadata.json", {
        "protocol": protocol, "n_participants": len(participants), "age_mean": np.mean(ages),
        "age_range": [min(ages), max(ages)],
        "experience_counts": {v: sum(p["teleoperation_experience"] == v for p in participants)
                              for v in {p["teleoperation_experience"] for p in participants}},
        "gender_counts": {v: sum(p.get("gender", "not_reported") == v for p in participants)
                          for v in {p.get("gender", "not_reported") for p in participants}},
        "analysis_scope": "paired participant means; incomplete cells retained and pairwise missingness reported"})
