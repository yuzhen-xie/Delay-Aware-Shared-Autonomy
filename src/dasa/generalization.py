from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from dasa.config import manifest, seed_everything, write_json
from dasa.data import Demonstration, bootstrap_admission
from dasa.experiments import load_split
from dasa.geometry import Scene
from dasa.scenes import generate_manifest, load_scene_manifest, validate_scene
from dasa.metrics import trial_metrics
from dasa.models.hsmm import TPHSMM, demonstration_fingerprint, refine
from dasa.models.intent import load_models
from dasa.simulation import Trial, run_trial
from dasa.statistics import comparisons, holm


def load_candidates(directory: Path) -> tuple[list[Demonstration], list[Path]]:






    files = sorted(directory.glob("*.npz"))
    order_path = directory / "order.json"
    if order_path.exists():
        names = json.loads(order_path.read_text())
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("order.json must list relative recording filenames")
        ordered = [(directory / n).resolve() for n in names]
        if any(p.parent != directory.resolve() for p in ordered):
            raise ValueError("Acquisition order cannot escape its collection directory")
        if len(set(ordered)) != len(ordered) or set(ordered) != {p.resolve() for p in files}:
            raise ValueError("Acquisition order must contain each NPZ exactly once")
        files = ordered
    demos = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema_version") == 1 and "goal" in metadata:
            demo = Demonstration.load(path)
        elif (metadata.get("schema_version") == 1 and "trial_id" in metadata
              and "intended_goal" in metadata
              and metadata.get("provenance") == "synthetic_task_space_simulation"):
            demo = Trial.load(path).demonstration()
            demo.metadata["simulation_seed"] = metadata.get("seed")
            demo.metadata["recording_format"] = "simulation_trial"
            demo.validate()
        else:
            raise ValueError(f"Unsupported recording schema: {path.name}")
        demos.append(demo)
    return demos, files


def _freeze_scenes(config: dict, output: Path, source: Path | None) -> dict:
    if source:
        result = json.loads(source.read_text())
        if result.get("generator_version") == "constrained_translation_v1":
            single = load_scene_manifest(source)
            entries = [{**row, "scenario": single["scenario"],
                        "seed": config["seed"] + 700000 + index, "goal": index % 3}
                       for index, row in enumerate(single["scenes"])]
            result = {"schema": "dasa_evaluation_scenes_v1", "scenes": entries,
                      "source_manifest_sha256": single["content_sha256"],
                      "scope": "constrained static known-scene translational simulation"}
        elif result.get("schema") != "dasa_evaluation_scenes_v1":
            raise ValueError("Expected a constrained scene manifest or dasa_evaluation_scenes_v1")
    else:
        entries, generated = [], {}
        for scenario_index, scenario in enumerate(config["experiments"]["scenarios"]):
            seed = config["seed"] + 700000 + scenario_index * 10000
            scene_manifest = generate_manifest(config, config["experiments"]["autonomy_trials"],
                                               scenario, seed)
            write_json(output / f"scene_manifests/{scenario}.json", scene_manifest)
            generated[scenario] = scene_manifest["content_sha256"]
            entries.extend({**row, "scenario": scenario, "seed": seed + index,
                            "goal": index % 3} for index, row in enumerate(scene_manifest["scenes"]))
        result = {"schema": "dasa_evaluation_scenes_v1",
                  "generator": "constrained_translation_v1_frozen_evaluation",
                  "seed": config["seed"], "scenes": entries, "source_manifests_sha256": generated,
                  "scope": "static known-scene translational simulation; no full-arm feasibility guarantee"}
    if not result.get("scenes"):
        raise ValueError("Evaluation manifest must contain scenes")
    ids = [row["scene_id"] for row in result["scenes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation scene IDs must be unique")
    for row in result["scenes"]:
        scene_id = row["scene_id"]
        if (not isinstance(scene_id, str) or Path(scene_id).name != scene_id
                or scene_id in {".", ".."} or not isinstance(row["seed"], int) or row["seed"] < 0):
            raise ValueError("Scene identifiers must be simple filenames and seeds nonnegative integers")
        scene = Scene.from_dict(row["scene"])
        if validate_scene(scene, config) or not 0 <= row["goal"] < len(scene.goals):
            raise ValueError("Invalid evaluation goal or coordinate frame")
    write_json(output / "scenes.json", result)
    return result


def generalization(config: dict, prepared_root: Path, candidates_dir: Path,
                   output: Path, scene_manifest_path: Path | None = None) -> list[dict]:








    if output.exists() and any(output.iterdir()):
        raise ValueError("Generalization output must be empty to preserve frozen analyses")
    output.mkdir(parents=True, exist_ok=True)
    budgets = config["experiments"].get("refinement_budgets", [0, 5, 10, 20])
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] != 0 or any(
            not isinstance(budget, int) or budget < 0 for budget in budgets):
        raise ValueError("refinement_budgets must be sorted unique nonnegative integers starting at 0")
    conditions = config["experiments"].get("refinement_conditions", ["C1", "C2", "C3"])
    if ("C3" not in conditions or len(set(conditions)) != len(conditions)
            or not set(conditions) <= {"C1", "C2", "C3"}):
        raise ValueError("Refinement conditions must be unique and include C3")
    repeats = config["experiments"]["independent_repeats"]
    cvae, _ = load_models(config, prepared_root / "models")
    evaluation = _freeze_scenes(config, output, scene_manifest_path)
    data_files, rows, budget_rows, audits = [], [], [], []
    global_collection_ids, global_collection_hashes = set(), set()
    for repeat in range(repeats):
        seed_everything(config["seed"] + 800000 + repeat)
        repeat_root = candidates_dir / f"r{repeat}"
        bootstrap_data = repeat_root / "bootstrap"
        simulation_bootstrap = candidates_dir.parent / "bootstrap" / f"r{repeat}"
        if (bootstrap_data / "splits.json").exists():
            bootstrap_policy = "supplied_per_repeat_bootstrap"
            original = load_split(bootstrap_data, "train")
            heldout = load_split(bootstrap_data, "validation") + load_split(bootstrap_data, "test")
            data_files.extend([bootstrap_data / "splits.json", *sorted(bootstrap_data.glob("*/*.npz"))])
        elif list(simulation_bootstrap.glob("*.npz")):
            bootstrap_policy = "recorded_simulation_per_repeat_bootstrap"
            original, bootstrap_files = load_candidates(simulation_bootstrap)
            if not all(d.metadata["provenance"].startswith("synthetic")
                       and d.metadata.get("source_condition") == "bootstrap" for d in original):
                raise ValueError("Sibling simulation bootstrap must contain explicitly synthetic bootstrap records")
            heldout = load_split(prepared_root / "dataset", "validation") + load_split(prepared_root / "dataset", "test")
            data_files.extend(bootstrap_files)
            data_files.extend([prepared_root / "dataset/splits.json",
                               *sorted((prepared_root / "dataset").glob("*/*.npz"))])
        else:
            bootstrap_data = prepared_root / "dataset"
            bootstrap_policy = "shared_fixed_bootstrap_conditional_inference"
            original = load_split(bootstrap_data, "train")
            heldout = load_split(bootstrap_data, "validation") + load_split(bootstrap_data, "test")
            data_files.extend([bootstrap_data / "splits.json", *sorted(bootstrap_data.glob("*/*.npz"))])
        heldout_hashes = {demonstration_fingerprint(d)["signal_sha256"] for d in heldout}
        heldout_operators = {d.metadata["operator_id"] for d in heldout}
        admission = bootstrap_admission(original, config)
        base = TPHSMM(config).fit(original)
        base.fit_manifest["bootstrap_admission"] = admission
        base.save(output / f"models/r{repeat}/bootstrap.npz")
        fits = [("bootstrap", 0, base, [])]
        budget_rows.append({"repeat": repeat, "condition": "bootstrap", "refinement_budget": 0,
                            "accepted_demonstrations": 0, "confirmatory_eligible": True,
                            "bootstrap_policy": bootstrap_policy})
        for condition in conditions:
            directory = repeat_root / condition
            candidates, files = load_candidates(directory)
            data_files.extend(files)
            if (directory / "order.json").exists():
                data_files.append(directory / "order.json")
            for demo in candidates:
                record = demonstration_fingerprint(demo)
                if demo.metadata.get("source_condition") != condition:
                    raise ValueError("Candidate condition does not match its collection directory")
                if not np.isclose(demo.metadata.get("delay_seconds", -1), 0.8):
                    raise ValueError("Refinement collection requires recorded 0.8 s feedback delay")
                if record["operator_id"] in heldout_operators or record["signal_sha256"] in heldout_hashes:
                    raise ValueError("Refinement collection overlaps held-out data")
                if record["episode_id"] in global_collection_ids or record["signal_sha256"] in global_collection_hashes:
                    raise ValueError("Collection episodes are duplicated across repeats or conditions")
                global_collection_ids.add(record["episode_id"])
                global_collection_hashes.add(record["signal_sha256"])
            selection_config = copy.deepcopy(config)
            selection_config["skill"]["refinement_count"] = budgets[-1]
            _, audit, accepted = refine(selection_config, original, candidates)
            audits.extend({"repeat": repeat, "condition": condition, **item} for item in audit)
            for budget in budgets[1:]:
                available = len(accepted) >= budget
                budget_rows.append({"repeat": repeat, "condition": condition,
                                    "refinement_budget": budget,
                                    "accepted_demonstrations": min(budget, len(accepted)),
                                    "available_successes": len(accepted),
                                    "confirmatory_eligible": available,
                                    "status": "eligible" if available else "unavailable_incomplete_quota",
                                    "bootstrap_policy": bootstrap_policy,
                                    "candidate_order": "order.json" if (directory / "order.json").exists()
                                    else "filename_lexicographic_order"})
                if not available:
                    continue
                fit_config = copy.deepcopy(config)
                fit_config["skill"]["refinement_count"] = budget
                model, _, selected = refine(fit_config, original, accepted[:budget])
                model.fit_manifest["bootstrap_admission"] = admission
                model.save(output / f"models/r{repeat}/{condition}_n{budget}.npz")
                fits.append((condition, budget, model, selected))
        for condition, budget, model, selected in fits:
            for entry in evaluation["scenes"]:
                seed = entry["seed"] + repeat * 1000000
                trial = run_trial(config, Scene.from_dict(entry["scene"]), model, cvae,
                                  seed, entry["goal"], "autonomous", f"training_repeat_{repeat}",
                                  trial_id=f"generalization_r{repeat}_{condition}_n{budget}_{entry['scene_id']}")
                trial.save(output / f"trials/r{repeat}/{condition}_n{budget}/{entry['scene_id']}.npz")
                rows.append({**trial_metrics(trial, config), "condition": condition,
                             "repeat": repeat, "scenario": entry["scenario"],
                             "scene_id": entry["scene_id"], "refinement_budget": budget,
                             "accepted_demonstrations": len(selected),
                             "confirmatory_eligible": True, "refinement_quota_met": True,
                             "bootstrap_policy": bootstrap_policy})
    write_json(output / "metrics.json", rows)
    write_json(output / "budgets.json", budget_rows)
    write_json(output / "selection_audit.json", audits)
    statistics = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for budget in budgets[1:]:
            subset = [r for r in rows if r["scenario"] == scenario and r["refinement_budget"] == budget]
            subset += [r for r in rows if r["scenario"] == scenario and r["condition"] == "bootstrap"]
            stats = comparisons(subset, "C3", [c for c in conditions if c != "C3"] + ["bootstrap"],
                                ["success_rate", "reference_error_m"], config,
                                family=f"generalization_{scenario}_n{budget}",
                                planned_delays=[config["simulation"]["feedback_delay"]])
            statistics.extend({"scenario": scenario, "refinement_budget": budget, **r} for r in stats)
    corrected = holm([r["p_raw"] for r in statistics], family_size=len(statistics))
    for row, adjusted in zip(statistics, corrected):
        row.update(p_holm=adjusted,
                   family="all_planned_generalization_method_budget_scenario_metric_comparisons",
                   family_size_planned=len(statistics),
                   family_size_estimable=sum(r["p_raw"] is not None for r in statistics),
                   correction="Holm_step_down_FWER",
                   significant_holm=adjusted < .05 if adjusted is not None else None)
        row["independent_unit"] = "training_repeat_conditional_on_bootstrap_corpus"
    write_json(output / "statistics.json", statistics)
    summaries = []
    for repeat in range(repeats):
        for condition, budget, scenario in sorted({(r["condition"], r["refinement_budget"], r["scenario"])
                                                  for r in rows if r["repeat"] == repeat}):
            selected = [r for r in rows if r["repeat"] == repeat and r["condition"] == condition
                        and r["refinement_budget"] == budget and r["scenario"] == scenario]
            summaries.append({"repeat": repeat, "condition": condition, "refinement_budget": budget,
                              "scenario": scenario, "n_trials": len(selected),
                              "success_rate": float(np.mean([r["success_rate"] for r in selected])),
                              "reference_error_m": float(np.mean([r["reference_error_m"] for r in selected])),
                              "reference_error_scope": "all executed samples, including failed partial trials"})
    write_json(output / "repeat_summary.json", summaries)
    source_files = sorted(set(data_files))
    if scene_manifest_path:
        source_files.append(scene_manifest_path)
    run_manifest = manifest(config, "generalization", source_files)
    run_manifest["evaluation_scene_sha256"] = hashlib.sha256((output / "scenes.json").read_bytes()).hexdigest()
    run_manifest["trained_intent_model_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((prepared_root / "models").glob("*.pt"))}
    write_json(output / "manifest.json", run_manifest)
    write_json(output / "protocol_status.json", {
        "procedure_completed": True,
        "protocol_complete": all(b["confirmatory_eligible"] for b in budget_rows),
        "unavailable_fits": [b for b in budget_rows if not b["confirmatory_eligible"]],
        "independent_unit": "training_repeat_not_individual_autonomous_trial",
        "inference_scope": "conditional on the supplied bootstrap corpus and frozen scene manifest",
        "selection": "first unique successful demonstrations in frozen acquisition order",
        "budget_zero": "one bootstrap-only fit per repeat, shared reference for all conditions"})
    return rows
