from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dasa.config import manifest, write_json
from dasa.data import Demonstration, successful
from dasa.experiments import (MEASURED_PROVENANCE, assignment, load_split, measured_bootstrap,
                              require_measured, split_files, validate_refinement_collection)
from dasa.geometry import Scene
from dasa.models.hsmm import TPHSMM, demonstration_fingerprint, refine, select_demonstrations
from dasa.scenes import generate_manifest, load_scene_manifest, validate_scene
from dasa.statistics import comparisons, holm


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, rows):
    if not rows:
        raise ValueError("A physical schedule must contain explicit prospective trials")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Use an empty output directory to preserve physical experiment records")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _acquisition_time(record):
    try:
        timestamp = datetime.fromisoformat(record.metadata["acquisition_started_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("Physical acquisition requires the recorded acquisition_started_utc timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Acquisition timestamps must include their measured UTC offset")
    return timestamp.astimezone(timezone.utc)


def load_candidates(directory):
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise ValueError(f"No imported physical demonstration files exist in {directory}")
    records = [Demonstration.load(path) for path in files]
    for record in records:
        require_measured(record)
    def acquisition_key(item):
        record = item[1]
        return _acquisition_time(record), record.metadata["source_recording_sha256"]
    ordered_records = sorted(zip(files, records), key=acquisition_key)
    files, records = map(list, zip(*ordered_records))
    order_path = directory / "order.json"
    inputs = list(files)
    if order_path.exists():
        order = json.loads(order_path.read_text(encoding="utf-8"))
        if not isinstance(order, list) or not all(isinstance(name, str) for name in order):
            raise ValueError("order.json must list the actual acquisition filenames")
        ordered = [(directory / name).resolve() for name in order]
        if (any(path.parent != directory.resolve() for path in ordered)
                or len(set(ordered)) != len(ordered)
                or set(ordered) != {path.resolve() for path in files}):
            raise ValueError("Acquisition order must identify every local NPZ exactly once")
        if ordered != [path.resolve() for path in files]:
            raise ValueError("order.json cannot reorder the measured acquisition timestamps")
        inputs.append(order_path)
    return records, inputs


def _bootstrap_checkpoint(config, prepared, original):
    path = prepared / "models/bootstrap_hsmm.npz"
    model = TPHSMM.load(config, path)
    fitted = model.fit_manifest.get("records", [])
    expected = {record["episode_id"]: (record["signal_sha256"], record["source_recording_sha256"])
                for record in map(demonstration_fingerprint, original)}
    observed = {record["episode_id"]: (record["signal_sha256"], record.get("source_recording_sha256"))
                for record in fitted}
    if expected != observed or len(fitted) != len(original):
        raise ValueError("The initial checkpoint does not match the supplied physical bootstrap corpus")
    return path, model


def _freeze_models(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cvae.pt", "lstm.pt", "transformer.pt", "bootstrap_hsmm.npz"):
        if not (source / name).is_file():
            raise ValueError(f"A prepared physical model checkpoint is required: {name}")
    result = {}
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {".pt", ".npz", ".json"}:
            digest = _sha(path)
            shutil.copyfile(path, output / path.name)
            if _sha(output / path.name) != digest:
                raise ValueError("A model checkpoint changed while freezing the acquisition")
            result[path.name] = digest
    return result


def _scene_config(config, scene):
    result = copy.deepcopy(config)
    result["scene"].update(goals=scene.goals.tolist(), place=scene.place.tolist(),
                           start=scene.start.tolist(), frame=scene.frame,
                           obstacles=[{"id": name, "center": center.tolist(), "radius": float(radius)}
                                      for name, center, radius in
                                      zip(scene.obstacle_ids, scene.centers, scene.radii)])
    return result


def _layouts(config, original, output, source=None):
    count = config["experiments"]["autonomy_trials"]
    scenarios = config["experiments"]["scenarios"]
    if source is not None:
        document = json.loads(Path(source).read_text(encoding="utf-8"))
        if document.get("schema") == "physical_evaluation_layouts_v1":
            entries = document["scenes"]
        else:
            single = load_scene_manifest(source)
            entries = [{**row, "scenario": single["scenario"], "goal": index % 3}
                       for index, row in enumerate(single["scenes"])]
    else:
        entries = []
        for scenario_index, scenario in enumerate(scenarios):
            for index in range(count):
                reference = original[index % len(original)]
                seed = config["seed"] + 700000 + scenario_index * 10000 + index
                generated = generate_manifest(_scene_config(config, reference.scene), 1, scenario, seed)
                entry = generated["scenes"][0]
                entries.append({**entry, "scene_id": f"{scenario}_{index:04d}",
                                "scenario": scenario, "goal": index % 3,
                                "master_seed": seed,
                                "reference_episode_id": reference.metadata["episode_id"],
                                "generator_version": generated["generator_version"],
                                "constraints": generated["constraints"]})
    ids = set()
    for entry in entries:
        identifier = entry["scene_id"]
        if (not isinstance(identifier, str) or Path(identifier).name != identifier
                or identifier in {".", ".."} or identifier in ids
                or entry["scenario"] not in scenarios or type(entry["goal"]) is not int
                or entry["goal"] not in range(3)):
            raise ValueError("Physical scene identifiers, conditions and assigned goals must be explicit and unique")
        ids.add(identifier)
        if validate_scene(Scene.from_dict(entry["scene"]), config):
            raise ValueError("A prescribed physical layout violates the geometric setup contract")
        entry["record_status"] = "planned_requires_measured_confirmation"
        write_json(output / "layouts" / f"{identifier}.json", entry)
    for scenario in scenarios:
        if sum(entry["scenario"] == scenario for entry in entries) != count:
            raise ValueError(f"Physical evaluation requires exactly {count} layouts/trial assignments for {scenario}")
    result = {"schema": "physical_evaluation_layouts_v1", "scenes": entries,
              "status": "prospective_physical_placement_plan",
              "path_selection": "workspace_and_nonoverlap_only_no_reference_path_filter",
              "target_displacement": "Euclidean_base_xy_radius_in_metres_without_clipping"}
    write_json(output / "scenes.json", result)
    return entries


def refinement_schedule(config, prepared_root, operators_csv, output):
    prepared, operators_csv = Path(prepared_root), Path(operators_csv)
    original = load_split(prepared / "dataset", "train")
    measured_bootstrap(config, original)
    initial, _ = _bootstrap_checkpoint(config, prepared, original)
    held_out = load_split(prepared / "dataset", "validation") + load_split(prepared / "dataset", "test")
    excluded = {demo.metadata["operator_id"] for demo in held_out}
    with operators_csv.open(newline="", encoding="utf-8-sig") as stream:
        roster = list(csv.DictReader(stream))
    operators = [row.get("operator_id", row.get("participant_id", "")).strip() for row in roster]
    if not operators or any(not operator for operator in operators) or len(set(operators)) != len(operators):
        raise ValueError("Supply unique actual anonymous operator IDs in operators.csv")
    if set(operators) & excluded:
        raise ValueError("Refinement operators cannot belong to held-out validation or test partitions")
    output = _empty_output(output)
    hashes = _freeze_models(prepared / "models", output / "models")
    quota, repeats = config["skill"]["refinement_count"], config["experiments"]["independent_repeats"]
    cap = config["experiments"].get("collection_attempt_cap", 150)
    if type(cap) is not int or cap < quota:
        raise ValueError("The declared attempt cap must accommodate the successful-demonstration quota")
    rng = np.random.default_rng(config["seed"] + 900000)
    rows, extensions = [], []
    for repeat in range(repeats):
        for attempt in range(cap):
            operator = operators[(repeat * cap + attempt) % len(operators)]
            reference = original[(repeat * cap + attempt) % len(original)]
            scene_id = f"collection_r{repeat:02d}_a{attempt:03d}"
            write_json(output / "layouts" / f"{scene_id}.json", {
                "scene_id": scene_id, "scene": reference.scene.to_dict(),
                "record_status": "planned_requires_measured_confirmation",
                "reference_episode_id": reference.metadata["episode_id"]})
            for condition in rng.permutation(["C1", "C2", "C3"]):
                row = {"trial_id": f"refinement_r{repeat:02d}_{condition}_a{attempt:03d}",
                       "participant_id": operator, "condition": str(condition),
                       "stage": "human_refinement", "training_repeat": repeat,
                       "training_condition": str(condition), "refinement_budget": quota,
                       "collection_id": f"r{repeat:02d}_{condition}", "attempt": attempt,
                       "delay_seconds": .8, "command_delay_seconds": 0.,
                       "intended_goal": (repeat + attempt) % 3,
                       "scene_seed": config["seed"] + 900000 + repeat * cap + attempt,
                       "scene_id": scene_id, "planned_scene_file": f"layouts/{scene_id}.json",
                       "models_dir": "models", "skill_file": "models/bootstrap_hsmm.npz",
                       "policy_sha256": _sha(initial), "cvae_sha256": hashes["cvae.pt"],
                       "timeout_seconds": config["task"]["max_seconds"],
                       "robot_radius_m": config["task"]["robot_radius"],
                       "practice": False, "record_status": "planned",
                       "planned_order": len(rows) + len(extensions) + 1,
                       "stop_after_accepted_successes": quota}
                (rows if attempt < quota else extensions).append(row)
    _write_csv(output / "schedule.csv", rows)
    if extensions:
        _write_csv(output / "schedule.extensions.csv", extensions)
    write_json(output / "protocol_status.json", {
        "stage": "human_refinement", "status": "prospective_not_collected",
        "required_successes_per_repeat_and_condition": quota, "attempt_cap": cap,
        "training_repeats": repeats, "conditions": ["C1", "C2", "C3"],
        "selection": "first_unique_successes_by_measured_acquisition_started_utc_then_source_recording_sha256",
        "extension_policy": "use_additional_assigned_attempts_only_until_the_success_quota_is_reached",
        "initial_policy": "fixed_for_the_entire_collection_block",
        "frozen_intent_model_sha256": hashes["cvae.pt"]})
    _, inputs = split_files(prepared / "dataset")
    write_json(output / "manifest.json", manifest(config, "refinement-schedule", [*inputs, operators_csv, initial]))
    result = {"schedule": str(output / "schedule.csv")}
    if extensions:
        result["extensions"] = str(output / "schedule.extensions.csv")
    return result


def generalization(config, prepared_root, candidates_dir, output, scene_manifest_path=None):
    prepared, candidates = Path(prepared_root), Path(candidates_dir)
    original = load_split(prepared / "dataset", "train")
    measured_bootstrap(config, original)
    initial, _ = _bootstrap_checkpoint(config, prepared, original)
    held_out = load_split(prepared / "dataset", "validation") + load_split(prepared / "dataset", "test")
    repeats, quota = config["experiments"]["independent_repeats"], config["skill"]["refinement_count"]
    cvae_hash = _sha(prepared / "models/cvae.pt")
    blocks, all_ids, all_hashes, raw_hashes, inputs = [], set(), set(), set(), []
    for repeat in range(repeats):
        for condition in ("C1", "C2", "C3"):
            directory = candidates / f"r{repeat}" / condition
            demos, files = load_candidates(directory)
            validate_refinement_collection(config, original, demos, held_out, condition, repeat)
            for demo in demos:
                fingerprint = demonstration_fingerprint(demo)
                raw_hash = demo.metadata["source_recording_sha256"]
                if (fingerprint["episode_id"] in all_ids or fingerprint["signal_sha256"] in all_hashes
                        or raw_hash in raw_hashes):
                    raise ValueError("Actual new demonstration IDs/signals cannot be reused across collections or conditions")
                all_ids.add(fingerprint["episode_id"])
                all_hashes.add(fingerprint["signal_sha256"])
                raw_hashes.add(raw_hash)
                hashes = demo.metadata.get("model_hashes", {})
                recorded_policy = hashes.get("skill", assignment(demo).get("policy_sha256"))
                recorded_cvae = hashes.get("cvae.pt", assignment(demo).get("cvae_sha256"))
                if condition != "C1" and (recorded_policy != _sha(initial) or recorded_cvae != cvae_hash):
                    raise ValueError("C2/C3 collection must use the fixed initial skill and frozen measured intent model")
                if condition == "C1" and recorded_policy not in (None, _sha(initial)):
                    raise ValueError("C1 collection metadata identify a different initial policy")
                if condition == "C1" and recorded_cvae not in (None, cvae_hash):
                    raise ValueError("C1 collection metadata identify a different frozen intent model")
            accepted, audit = select_demonstrations(config, original, demos)
            if len(accepted) != quota:
                raise ValueError(f"Collection r{repeat}/{condition} supplies {len(accepted)}/{quota} successful physical demonstrations")
            inputs.extend(files)
            blocks.append((repeat, condition, accepted, audit))
    output = _empty_output(output)
    frozen = _freeze_models(prepared / "models", output / "intent_models")
    scenes = _layouts(config, original, output, scene_manifest_path)
    rows, inventories = [], []
    for repeat, condition, accepted, audit in blocks:
        model, _, _ = refine(config, original, accepted)
        model.fit_manifest["physical_collection"] = {"stage": "human_refinement",
            "training_repeat": repeat, "training_condition": condition,
            "cvae_sha256": cvae_hash, "initial_policy_sha256": _sha(initial),
            "bootstrap_policy": "fixed_measured_corpus_conditional_inference"}
        policy = output / f"models/r{repeat}/{condition}_n{quota}.npz"
        model.save(policy)
        write_json(output / f"collections/r{repeat}_{condition}.json", {
            "stage": "human_refinement", "repeat": repeat, "condition": condition,
            "accepted_ids": [demo.metadata["episode_id"] for demo in accepted],
            "selection_audit": audit, "initial_policy_sha256": _sha(initial),
            "candidate_order": "measured_acquisition_started_utc_then_source_recording_sha256",
            "new_data_independence": "distinct_actual_episode_ids_and_signal_hashes",
            "operator_scope": "conditional_on_recorded_operators_and_fixed_bootstrap_corpus"})
        inventory = {"training_repeat": repeat, "training_condition": condition,
                     "refinement_budget": quota, "policy_file": str(policy.relative_to(output)),
                     "policy_sha256": _sha(policy), "accepted_demonstrations": len(accepted),
                     "bootstrap_demonstrations": len(original), "cvae_sha256": cvae_hash}
        inventories.append(inventory)
        for index, entry in enumerate(scenes):
            rows.append({"trial_id": f"autonomy_r{repeat:02d}_{condition}_{entry['scene_id']}",
                         "participant_id": f"robot_policy_r{repeat:02d}", "actor_type": "physical_autonomy",
                         "condition": "autonomous", "stage": "autonomous_evaluation",
                         "training_repeat": repeat, "training_condition": condition,
                         "refinement_budget": quota, "scenario": entry["scenario"],
                         "evaluation_index": index, "intended_goal": entry["goal"],
                         "scene_id": entry["scene_id"],
                         "scene_seed": entry.get("master_seed", config["seed"] + index),
                         "planned_scene_file": f"layouts/{entry['scene_id']}.json",
                         "models_dir": "intent_models", "skill_file": inventory["policy_file"],
                         "policy_sha256": inventory["policy_sha256"], "cvae_sha256": cvae_hash,
                         "delay_seconds": 0., "command_delay_seconds": 0.,
                         "timeout_seconds": config["task"]["max_seconds"],
                         "robot_radius_m": config["task"]["robot_radius"],
                         "practice": False, "record_status": "planned",
                         "planned_order": len(rows) + 1})
    randomization_seed = config["seed"] + 1100000
    rng = np.random.default_rng(randomization_seed)
    lookup = {(row["training_repeat"], row["training_condition"], row["scene_id"]): row for row in rows}
    randomized = []
    for repeat in range(repeats):
        for block, index in enumerate(rng.permutation(len(scenes))):
            scene_id = scenes[index]["scene_id"]
            for within_block, condition in enumerate(rng.permutation(["C1", "C2", "C3"])):
                row = lookup[(repeat, str(condition), scene_id)]
                row.update(randomization_seed=randomization_seed,
                           matched_block_id=f"autonomy_r{repeat:02d}_b{block:04d}",
                           matched_block_index=block, within_block_order=within_block + 1,
                           planned_order=len(randomized) + 1)
                randomized.append(row)
    _write_csv(output / "evaluation_schedule.csv", randomized)
    write_json(output / "policy_inventory.json", inventories)
    write_json(output / "protocol_status.json", {
        "stage": "autonomous_evaluation", "human_refinement_status": "fitted_from_measured_collections",
        "evaluation_status": "prospective_not_collected", "expected_physical_trials": len(rows),
        "trials_per_policy_and_scenario": config["experiments"]["autonomy_trials"],
        "training_repeats": repeats, "fixed_new_demonstration_budget": quota,
        "randomization_seed": randomization_seed,
        "condition_order": "randomized_within_matched_scene_blocks_with_independent_scene_order_per_training_repeat",
        "bootstrap_policy": "fixed_real_corpus_conditional_inference",
        "frozen_intent_checkpoint_sha256": frozen["cvae.pt"],
        "inference_unit": "training_collection_repeat_not_individual_robot_trial",
        "scene_confirmation": "physical_placement_and_calibrated_detection_required_before_execution"})
    _, bootstrap_files = split_files(prepared / "dataset")
    if scene_manifest_path is not None:
        inputs.append(Path(scene_manifest_path))
    run_manifest = manifest(config, "generalization", [*bootstrap_files, *inputs, initial, prepared / "models/cvae.pt"])
    run_manifest.update(data_provenance=MEASURED_PROVENANCE, stage="measured_policy_refit_and_physical_schedule")
    write_json(output / "manifest.json", run_manifest)
    return {"trained_policies": len(inventories), "scheduled_physical_trials": len(rows),
            "schedule": str(output / "evaluation_schedule.csv")}


def aggregate_measured_evaluations(config, experiment_root, recordings, output):
    from dasa.metrics import measured_motion_metrics

    experiment, recordings = Path(experiment_root), Path(recordings)
    schedule_path = experiment / "evaluation_schedule.csv"
    schedule_digest = _sha(schedule_path)
    with schedule_path.open(newline="", encoding="utf-8-sig") as stream:
        schedule_rows = list(csv.DictReader(stream))
    schedule = {row["trial_id"]: row for row in schedule_rows}
    repeats = config["experiments"]["independent_repeats"]
    scenarios = config["experiments"]["scenarios"]
    required = config["experiments"]["autonomy_trials"]
    quota = config["skill"]["refinement_count"]
    expected_cells = {(str(repeat), condition, scenario)
                      for repeat in range(repeats) for condition in ("C1", "C2", "C3") for scenario in scenarios}
    cell_counts = {key: 0 for key in expected_cells}
    cell_policies = {}
    layout_entries = json.loads((experiment / "scenes.json").read_text(encoding="utf-8"))["scenes"]
    layouts_by_scenario = {scenario: {entry["scene_id"] for entry in layout_entries
                                     if entry["scenario"] == scenario} for scenario in scenarios}
    if (any(len(identifiers) != required for identifiers in layouts_by_scenario.values())
            or len({entry["scene_id"] for entry in layout_entries}) != len(layout_entries)):
        raise ValueError("Matched physical evaluation needs unique explicit layouts at every configured scenario count")
    cell_layouts = {key: set() for key in expected_cells}
    for planned in schedule_rows:
        key = (planned["training_repeat"], planned["training_condition"], planned["scenario"])
        if (key not in expected_cells or planned["stage"] != "autonomous_evaluation"
                or planned["condition"] != "autonomous" or planned["refinement_budget"] != str(quota)
                or planned["intended_goal"] not in {"0", "1", "2"}
                or float(planned["delay_seconds"]) != 0. or float(planned["command_delay_seconds"]) != 0.):
            raise ValueError("The frozen physical evaluation schedule differs from the configured fixed-budget protocol")
        cell_counts[key] += 1
        if planned["scene_id"] not in layouts_by_scenario[key[2]] or planned["scene_id"] in cell_layouts[key]:
            raise ValueError("Every learned policy must evaluate each frozen matched scenario layout exactly once")
        cell_layouts[key].add(planned["scene_id"])
        if key in cell_policies and cell_policies[key] != planned["policy_sha256"]:
            raise ValueError("A physical policy/scenario cell cannot mix trained checkpoint versions")
        cell_policies[key] = planned["policy_sha256"]
        for field in ("skill_file", "planned_scene_file"):
            if not (experiment / planned[field]).resolve().is_relative_to(experiment.resolve()):
                raise ValueError("Physical schedule paths must remain inside their frozen experiment directory")
    if len(schedule) != len(schedule_rows) or any(count != required for count in cell_counts.values()):
        raise ValueError("The physical schedule must contain every distinct planned trial at the configured repeat and scenario counts")
    for repeat in range(repeats):
        for condition in ("C1", "C2", "C3"):
            policies = {cell_policies[(str(repeat), condition, scenario)] for scenario in scenarios}
            if len(policies) != 1:
                raise ValueError("Matched generalization scenarios must evaluate the same trained checkpoint")
    paths = sorted(recordings.rglob("*.npz")) if recordings.is_dir() else [recordings]
    if not paths:
        raise ValueError("No imported physical autonomous evaluation recordings were supplied")
    rows, seen_ids, seen_hashes, raw_hashes, source_hashes, motion_audits = [], set(), set(), set(), {}, {}
    actual_order = []
    for path in paths:
        digest_before = _sha(path)
        demo = Demonstration.load(path)
        require_measured(demo, "autonomous_evaluation")
        record, metadata = assignment(demo), demo.metadata
        identifier = metadata["episode_id"]
        fingerprint = demonstration_fingerprint(demo)
        if (identifier not in schedule or identifier in seen_ids or fingerprint["signal_sha256"] in seen_hashes
                or metadata["source_recording_sha256"] in raw_hashes):
            raise ValueError("Each actual physical evaluation must match one unique scheduled trial")
        seen_ids.add(identifier)
        seen_hashes.add(fingerprint["signal_sha256"])
        raw_hashes.add(metadata["source_recording_sha256"])
        planned = schedule[identifier]
        if record.get("schedule_sha256") != schedule_digest:
            raise ValueError("The actual acquisition is not bound to this exact frozen physical evaluation schedule")
        actual_order.append((_acquisition_time(demo), metadata["source_recording_sha256"], identifier))
        for key in ("training_repeat", "training_condition", "refinement_budget", "scenario", "policy_sha256", "scene_id"):
            if str(record.get(key)) != str(planned[key]):
                raise ValueError(f"Measured autonomous assignment differs from the frozen schedule: {key}")
        if metadata.get("source_condition") != "autonomous" or demo.goal != int(planned["intended_goal"]):
            raise ValueError("Autonomous evaluations require the assigned goal without human arbitration")
        if metadata.get("model_hashes", {}).get("skill") != planned["policy_sha256"]:
            raise ValueError("The physically executed skill differs from the scheduled learned policy")
        if metadata.get("model_hashes", {}).get("cvae.pt") != planned["cvae_sha256"]:
            raise ValueError("A physical evaluation changed the frozen intent checkpoint")
        policy_file = experiment / planned["skill_file"]
        if _sha(policy_file) != planned["policy_sha256"]:
            raise ValueError("A scheduled physical policy checkpoint changed")
        planned_scene = json.loads((experiment / planned["planned_scene_file"]).read_text())["scene"]
        actual_scene = demo.scene.to_dict()
        tolerance = config.get("study", {}).get("hardware_scene_tolerance_m", .005)
        if actual_scene["obstacle_ids"] != planned_scene["obstacle_ids"]:
            raise ValueError("Measured obstacle identities differ from the scheduled physical layout")
        for key in ("goals", "place", "start", "centers", "radii"):
            actual, expected = np.asarray(actual_scene[key]), np.asarray(planned_scene[key])
            if actual.shape != expected.shape or np.max(np.abs(actual - expected), initial=0.) > tolerance:
                raise ValueError("Measured placement differs from the planned physical layout")
        with np.load(path, allow_pickle=False) as archive:
            if "reference" not in archive or "executed" not in archive:
                raise ValueError("Physical evaluation import must preserve the executed mask and original online reference")
            reference, executed = np.asarray(archive["reference"], float), np.asarray(archive["executed"])
            scene_mode = metadata.get("hardware_timing", {}).get("scene_mode")
            measured_clearance = None
            if scene_mode == "online":
                if "surface_clearance_m" not in archive:
                    raise ValueError("Online physical geometry requires measured surface_clearance_m at every recorded sample")
                measured_clearance = np.asarray(archive["surface_clearance_m"], float)
            elif scene_mode != "static_once":
                raise ValueError("Physical autonomous evaluation requires an explicit static_once or online acquisition mode")
        if (reference.shape != demo.position.shape or executed.shape != demo.time.shape
                or executed.dtype != np.bool_ or not np.isfinite(reference).all() or not executed.any()):
            raise ValueError("Online references and actual executed sample masks must be finite and aligned")
        accepted, reasons = successful(demo, config)
        if metadata["success"] and not accepted:
            raise ValueError("A successful annotation contradicts physical geometry/task evidence: " + ",".join(reasons))
        contract = metadata["execution_contract"]
        metrics, motion_audit, _ = measured_motion_metrics(
            config, demo.time, demo.position, demo.velocity, demo.scene, contract["robot_radius_m"], measured_clearance)
        motion_audit["scene_mode"] = scene_mode
        motion_audits[identifier] = motion_audit
        error = float(np.linalg.norm(demo.position[executed] - reference[executed], axis=1).mean())
        duration = float(demo.time[-1] - demo.time[0])
        rows.append({**metrics, "trial_id": identifier,
                     "unit": f"training_repeat_{record['training_repeat']}",
                     "repeat": int(record["training_repeat"]), "condition": record["training_condition"],
                     "scenario": record["scenario"], "refinement_budget": int(record["refinement_budget"]),
                     "provenance": metadata["provenance"], "stage": "autonomous_evaluation",
                     "delay_seconds": 0., "success_rate": float(metadata["success"]),
                     "completion_seconds": duration if metadata["success"] else None,
                     "elapsed_seconds": duration, "reference_error_m": error,
                     "successful_reference_error_m": error if metadata["success"] else None,
                     "reference_error_scope": "original_online_target_at_matching_executed_samples_failures_retained",
                     "collision_rate": float(metadata["collision"]), "policy_sha256": planned["policy_sha256"]})
        source_hashes[str(path)] = _sha(path)
        if source_hashes[str(path)] != digest_before:
            raise ValueError("A physical evaluation recording changed while its metrics were computed")
    if _sha(schedule_path) != schedule_digest:
        raise ValueError("The physical schedule changed during analysis")
    missing = sorted(set(schedule) - seen_ids)
    order_audit = {}
    for repeat in range(repeats):
        acquired = [item for item in actual_order if schedule[item[2]]["training_repeat"] == str(repeat)]
        chronological = [item[2] for item in sorted(acquired)]
        planned_order = sorted(chronological, key=lambda identifier: int(schedule[identifier]["planned_order"]))
        order_audit[f"training_repeat_{repeat}"] = {
            "matches_assigned_order": chronological == planned_order,
            "actual_trial_sequence": chronological, "assigned_collected_trial_sequence": planned_order}
    order_valid = all(audit["matches_assigned_order"] for audit in order_audit.values())
    output = _empty_output(output)
    write_json(output / "metrics.json", rows)
    write_json(output / "acquisition_status.json", {
        "stage": "autonomous_evaluation", "expected_trials": len(schedule), "collected_trials": len(rows),
        "complete": not missing, "missing_trial_ids": missing,
        "inference_status": ("fixed_budget_complete" if not missing and order_valid
                             else "incomplete_descriptive_only" if missing else "order_deviation_descriptive_only"),
        "actual_acquisition_order_audit": order_audit,
        "failure_policy": "all collected annotated failures remain in the success denominator",
        "reference_error_policy": "all executed traces plus separately reported successful-only error",
        "reference_alignment": "original recorded nominal online LQT target at each controller sample",
        "source_sha256": source_hashes, "motion_metric_audits": motion_audits})
    results = []
    eligible = []
    for row in rows:
        count = sum(other["repeat"] == row["repeat"] and other["condition"] == row["condition"]
                    and other["scenario"] == row["scenario"] for other in rows)
        row["confirmatory_eligible"] = not missing and order_valid and count == required
        if row["confirmatory_eligible"]:
            eligible.append(row)
    for scenario in scenarios:
        subset = [row for row in eligible if row["scenario"] == scenario]
        contrasts = comparisons(subset, "C3", ["C1", "C2"],
                                ["success_rate", "reference_error_m", "successful_reference_error_m"],
                                config, family=f"physical_autonomy_{scenario}", planned_delays=[0.])
        results.extend({**contrast, "scenario": scenario} for contrast in contrasts)
    corrected = holm([row["p_raw"] for row in results], family_size=len(results))
    for row, adjusted in zip(results, corrected):
        row.update(p_holm=adjusted, family="all_physical_autonomy_scenarios_methods_metrics",
                   family_size_planned=len(results), family_size_estimable=sum(item["p_raw"] is not None for item in results),
                   significant_holm=adjusted < .05 if adjusted is not None else None,
                   independent_unit="actual_training_collection_repeat_conditional_on_fixed_bootstrap_and_recorded_operators")
    write_json(output / "metrics.json", rows)
    metric_digest = _sha(output / "metrics.json")
    for row in results:
        row.update(metric_source_sha256=metric_digest, metric_source_file="metrics.json")
    write_json(output / "statistics.json", results)
    run_manifest = manifest(config, "generalization-analyze", [experiment / "evaluation_schedule.csv", *paths])
    run_manifest.update(data_provenance=MEASURED_PROVENANCE, stage="measured_autonomous_evaluation_analysis")
    write_json(output / "manifest.json", run_manifest)
    return {"collected_trials": len(rows), "expected_trials": len(schedule), "complete": not missing,
            "confirmatory_eligible": not missing and order_valid}
