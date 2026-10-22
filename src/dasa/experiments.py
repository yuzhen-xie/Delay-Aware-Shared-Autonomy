from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from dasa.config import manifest, seed_everything, write_json
from dasa.data import Demonstration, bootstrap_admission, validate_partitions
from dasa.models.dmp import DMP
from dasa.models.hsmm import TPHSMM, demonstration_fingerprint, refine, select_demonstrations
from dasa.models.intent import train_intent, training_arrays


MEASURED_PROVENANCE = "UR10_measured_teleoperation"


def assignment(demo):
    value = demo.metadata.get("experiment_assignment", {})
    if not isinstance(value, dict):
        raise ValueError("Physical recordings require a structured acquisition assignment")
    return value


def require_measured(demo, stage=None):
    demo.validate()
    metadata = demo.metadata
    actual_stage = metadata.get("stage", assignment(demo).get("stage"))
    allowed = {MEASURED_PROVENANCE, "UR10_measured_autonomous"} \
        if actual_stage == "autonomous_evaluation" else {MEASURED_PROVENANCE}
    if metadata.get("provenance") not in allowed:
        raise ValueError("Only imported physical UR10 measurements are accepted")
    if demo.scene.provenance not in {"calibrated_scene", "calibrated_rgbd_geometry"}:
        raise ValueError("Physical demonstrations require measured calibrated scene geometry")
    digest = metadata.get("source_recording_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())):
        raise ValueError("Physical recordings require their original recording SHA-256")
    if not isinstance(metadata.get("execution_contract"), dict):
        raise ValueError("Physical recordings require their actual execution limits")
    if (actual_stage in {"human_bootstrap", "human_refinement"}
            and metadata.get("hardware_timing", {}).get("scene_mode") != "static_once"):
        raise ValueError("Bootstrap and refinement learning require fixed task frames from static_once calibrated acquisition")
    for field in ("success", "placed", "collision", "bounds_violation"):
        if not isinstance(metadata.get(field), bool):
            raise ValueError(f"Independent physical annotations must contain a JSON boolean: {field}")
    if stage is not None and actual_stage != stage:
        raise ValueError(f"Expected physical acquisition stage {stage}, received {actual_stage}")
    return actual_stage


def split_files(root):
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    validate_partitions(root, splits)
    return splits, [root / "splits.json", *sorted({root / name for paths in splits.values() for name in paths})]


def load_split(root: Path, split: str):
    root = Path(root)
    splits, _ = split_files(root)
    if split not in splits or not splits[split]:
        raise ValueError(f"A nonempty measured {split} partition is required")
    demos = [Demonstration.load(root / name) for name in splits[split]]
    for demo in demos:
        require_measured(demo)
    return demos


def snapshot_dataset(data: Path, root: Path) -> Path:
    data, root = Path(data), Path(root)
    splits, _ = split_files(data)
    target = root / "dataset"
    if data.resolve() == target.resolve():
        return data
    if target.exists() and any(target.iterdir()):
        raise ValueError("A prepared dataset snapshot already exists")
    for names in splits.values():
        for name in names:
            source = (data / name).resolve()
            if not source.is_relative_to(data.resolve()):
                raise ValueError("Dataset entries must remain inside their dataset directory")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    shutil.copyfile(data / "splits.json", target / "splits.json")
    validate_partitions(target, splits)
    return target


def measured_bootstrap(config, demos):
    expected = config["experiments"]["bootstrap_demonstrations"]
    if len(demos) != expected:
        raise ValueError(f"Bootstrap training requires exactly {expected} measured training demonstrations")
    for demo in demos:
        require_measured(demo, "human_bootstrap")
        if not np.isclose(demo.metadata.get("delay_seconds", -1), 0., atol=1e-12, rtol=0):
            raise ValueError("The initial human bootstrap corpus must use zero configured feedback delay")
    if {demo.goal for demo in demos} != {0, 1, 2}:
        raise ValueError("The measured bootstrap corpus must include every candidate goal")
    if len({demo.metadata["source_recording_sha256"] for demo in demos}) != len(demos):
        raise ValueError("Bootstrap episodes cannot reuse the same original physical recording")
    return bootstrap_admission(demos, config)


def train_dataset(config: dict, data: Path, root: Path):
    data, root = Path(data), Path(root)
    seed_everything(config["seed"])
    train, validation = load_split(data, "train"), load_split(data, "validation")
    admission = measured_bootstrap(config, train)
    for demo in validation:
        require_measured(demo)
    data = snapshot_dataset(data, root)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    skill = TPHSMM(config).fit(train)
    skill.fit_manifest["bootstrap_admission"] = admission
    skill.fit_manifest["acquisition_stage"] = "human_bootstrap"
    skill.save(models / "bootstrap_hsmm.npz")
    write_json(models / "bootstrap_inventory.json", skill.fit_manifest)
    DMP().fit(train).save(models / "dmp.npz")
    history = train_intent(config, training_arrays(train, skill, config),
                           training_arrays(validation, skill, config), models)
    write_json(models / "training.json", history)
    splits, files = split_files(data)
    run_manifest = manifest(config, "train-data", files)
    run_manifest.update(data_provenance=MEASURED_PROVENANCE, stage="measured_bootstrap_training",
                        training_demonstrations=len(train), validation_demonstrations=len(validation))
    write_json(root / "manifest.json", run_manifest)
    write_json(root / "dataset_inventory.json", {
        "split_policy": "operator_episode_and_signal_disjoint",
        "splits": {name: [{"file": path, **demonstration_fingerprint(Demonstration.load(data / path))}
                          for path in names] for name, names in splits.items()}})
    return run_manifest


def validate_refinement_collection(config, original, candidates, held_out, condition=None, repeat=None):
    measured_bootstrap(config, original)
    excluded_operators = {demo.metadata["operator_id"] for demo in held_out}
    excluded_hashes = {demonstration_fingerprint(demo)["signal_sha256"] for demo in held_out}
    excluded_recordings = {demo.metadata["source_recording_sha256"] for demo in original + held_out}
    for demo in candidates:
        require_measured(demo, "human_refinement")
        record = assignment(demo)
        actual_condition = demo.metadata.get("source_condition")
        if actual_condition not in {"C1", "C2", "C3"} or (condition is not None and actual_condition != condition):
            raise ValueError("Human refinement recordings must match their assigned C1/C2/C3 condition")
        if not np.isclose(demo.metadata.get("delay_seconds", -1), .8, atol=1e-12, rtol=0):
            raise ValueError("Human refinement recordings require the recorded 0.8-second feedback setting")
        if repeat is not None and record.get("training_repeat") != repeat:
            raise ValueError("The measured training-repeat assignment differs from its collection")
        if (demo.metadata["operator_id"] in excluded_operators
                or demonstration_fingerprint(demo)["signal_sha256"] in excluded_hashes
                or demo.metadata["source_recording_sha256"] in excluded_recordings):
            raise ValueError("Refinement recordings overlap held-out operators or signals")


def refine_dataset(config: dict, data: Path, candidates: Path, output: Path):
    from dasa.generalization import load_candidates

    data, candidates, output = Path(data), Path(candidates), Path(output)
    original = load_split(data, "train")
    new, files = load_candidates(candidates)
    held_out = load_split(data, "validation") + load_split(data, "test")
    validate_refinement_collection(config, original, new, held_out)
    accepted, audit = select_demonstrations(config, original, new)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection_audit.json", audit)
    requested = config["skill"]["refinement_count"]
    complete = len(accepted) == requested
    write_json(output / "refinement_metadata.json", {
        "stage": "human_refinement", "original_count": len(original), "accepted_count": len(accepted),
        "requested": requested, "quota_met": complete, "confirmatory_eligible": complete,
        "candidate_order": "measured_acquisition_started_utc_then_source_recording_sha256",
        "order_file_verified": (candidates / "order.json").exists(),
        "updated_components": ["TP-HSMM"], "frozen_components": ["cVAE", "goal_classifier", "intent_baselines"]})
    if not complete:
        raise ValueError(f"Only {len(accepted)} of {requested} required successful physical demonstrations were available")
    model, _, _ = refine(config, original, accepted)
    model.fit_manifest["bootstrap_admission"] = measured_bootstrap(config, original)
    model.save(output / "refined_hsmm.npz")
    _, original_files = split_files(data)
    write_json(output / "manifest.json", manifest(config, "refine-data", [*original_files, *files]))
    return model.fit_manifest
