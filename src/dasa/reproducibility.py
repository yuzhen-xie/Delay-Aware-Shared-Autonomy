"""Resolved parameter tables and source-bound experiment inventories."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from dasa.config import manifest, write_json


# Units apply to the implemented discrete model; selection provenance is separate.
SPECS = {
    "simulation.control_dt": ("s", "Implicit-midpoint integration and energy ledger interval"),
    "simulation.decision_dt": ("s", "Held nominal command and inference interval"),
    "simulation.mass": ("kg", "Translational point-mass plant; not UR10 inertia identification"),
    "simulation.damping": ("N s/m", "Translational viscous damping"),
    "simulation.force_limit": ("N", "Norm limit before the energy constraint"),
    "simulation.robot_radius": ("m", "TCP collision sphere radius"),
    "simulation.workspace_min": ("m", "Base-frame Cartesian lower bounds"),
    "simulation.workspace_max": ("m", "Base-frame Cartesian upper bounds"),
    "simulation.max_seconds": ("s", "Simulation timeout; independent from study/device timeout"),
    "simulation.grasp_tolerance": ("m", "Abstract grasp-center distance tolerance"),
    "simulation.place_tolerance": ("m", "Abstract placement-center tolerance"),
    "simulation.dwell_seconds": ("s", "Required continuous low-speed dwell"),
    "simulation.max_grasp_speed": ("m/s", "Grasp and placement speed threshold"),
    "simulation.lift_height": ("m", "Base-frame lift height"),
    "simulation.feedback_delay": ("s, one-way", "Slave observation-to-master channel"),
    "simulation.command_delay": ("s, one-way", "Master command-to-slave channel"),
    "simulation.jitter_seconds": ("s", "Uniform per-query jitter half-width, clipped at zero delay"),
    "scene.goals": ("m", "Three candidate grasp centers in base coordinates"),
    "scene.start": ("m", "TCP initial position in base coordinates"),
    "scene.place": ("m", "Shared placement center in base coordinates"),
    "scene.object_size_m": ("m", "Proposed static envelope for constrained manifests; not simulated contact geometry"),
    "scene.training_translation": ("m", "Common xyz uniform half-width; z is multiplied by 0.25"),
    "scene.novel_translation": ("m per xy axis", "Independent signed shifts, followed by clipping"),
    "scene.added_obstacle_radius": ("m", "Novel obstacle sphere radius"),
    "motion.attractive_gain": ("N/m", "k_att; fixed engineering default"),
    "motion.repulsive_gain": ("N m", "k_rep in Gaussian radial field"),
    "motion.repulsive_sigma": ("m", "Gaussian radial width"),
    "motion.repulsive_limit": ("N per obstacle", "Per-obstacle repulsion norm cap"),
    "motion.attractive_weight": ("1", "beta_1 alias: multiplier of attractive field"),
    "motion.repulsive_weight": ("1", "beta_2 alias: multiplier of repulsive field"),
    "motion.distance_min": ("m", "d_min / ell_min in the configured distance convention"),
    "motion.distance_max": ("m", "d_max / ell_max in the configured distance convention"),
    "motion.lambda_gain": ("1", "K_lambda: sigmoid slope for distance normalized by d_max-d_min"),
    "motion.kalman_process": ("N^2/sample", "Q=q I_3, not continuous-time noise density"),
    "motion.kalman_measurement": ("N^2", "R=r I_3"),
    "motion.kalman_initial_variance": ("N^2", "P_0=p I_3; x_0=0 N"),
    "arbitration.stiffness": ("N/m", "K^P=k I_3"),
    "arbitration.damping": ("N s/m", "K^V=b I_3"),
    "arbitration.covariance_lower": ("m^2", "Lower trace threshold for c_A"),
    "arbitration.covariance_upper": ("m^2", "Upper trace threshold for c_A"),
    "arbitration.roughness_gamma": ("s/N", "gamma_j; one common sensitivity, not participant-specific fitted values"),
    "arbitration.roughness_window": ("samples", "Moving mean of force-derivative norms"),
    "arbitration.epsilon": ("1", "Positive denominator regularizer for relative authority"),
    "arbitration.rho_window_seconds": ("s", "Sliding window for removed/nominal positive work"),
    "arbitration.low_confidence_threshold": ("1", "Guarded extension only"),
    "arbitration.correction_distance": ("m", "Guarded extension uses surface clearance"),
    "arbitration.fixed_weight": ("1", "C4 alpha"),
    "arbitration.guarded_corrective_max": ("1", "Guarded extension alpha cap"),
    "passivity.initial_energy": ("J", "Initial shared-control storage"),
    "passivity.maximum_energy": ("J", "Storage capacity; autonomous mode starts at this value"),
    "passivity.input_efficiency": ("1", "Credited fraction of positive sampled human input work"),
    "passivity.tolerance": ("J", "Numerical budget violation tolerance"),
    "intent.beta": ("rad^-1", "Boltzmann inverse angular cost scale; not a KL weight"),
    "intent.confidence_threshold": ("1", "IPA and sustained early-prediction threshold"),
    "intent.sustained_seconds": ("s", "Continuous correct/confident period for EPT"),
    "intent.memory_decay": ("1", "Memory baseline evidence forgetting factor"),
    "intent.learning_rate": ("1/update", "Adam learning rate"),
    "intent.kl_weight": ("1", "Exactly one for standard ELBO modes"),
    "intent.deviation_weight": ("1", "One for joint ELBO; zero for action-only ELBO; explicit legacy otherwise"),
    "skill.covariance_floor": ("m^2", "Gaussian covariance eigenvalue lower bound"),
    "skill.duration_floor": ("resampled samples", "Duration standard-deviation lower bound"),
    "skill.lqt_control_cost": ("relative cost", "R=r I_3 in the LQT discretization"),
    "skill.original_weight": ("1 per episode", "Original demonstration likelihood weight"),
    "skill.new_weight": ("1 per episode", "Accepted new demonstration likelihood weight"),
    "skill.proximity_threshold_m": ("m", "Advance path cursor near the current reference"),
    "skill.place_snap_distance_m": ("m", "Snap final reference to placement center"),
    "bootstrap.segment_min_seconds": ("s", "Synthetic segment duration lower bound"),
    "bootstrap.cruise_speed_m_s": ("m/s", "Synthetic quintic duration sampling range"),
    "bootstrap.lateral_std_m": ("m", "Synthetic per-segment zero-endpoint lateral variation"),
    "bootstrap.force_noise_std_N": ("N", "Synthetic inverse-dynamics force observation noise"),
    "operator.kp": ("N/m", "Simulated operator PD position gain"),
    "operator.kd": ("N s/m", "Simulated operator PD velocity gain"),
    "operator.expert_noise": ("N", "Synthetic correlated force-noise innovation scale"),
    "operator.noisy_noise": ("N", "Synthetic correlated force-noise innovation scale"),
    "operator.corrective_gain": ("1", "Simulated outward correction multiplier"),
    "experiments.intent_delays": ("s, one-way feedback", "No command delay in the primary profile"),
    "experiments.sharing_delays": ("s, one-way feedback", "No command delay in the primary profile"),
    "study.max_seconds": ("s", "Proposed human-study timeout; historical value requires protocol record"),
}


def flatten(value: dict, prefix=""):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from flatten(item, name)
        else:
            yield name, item


def parameter_rows(config: dict):
    rows = []
    for path, value in flatten(config):
        unit, description = SPECS.get(path, (
            "count" if isinstance(value, int) and not isinstance(value, bool) else "1 / category",
            "Resolved execution setting; see module contract for structural meaning"))
        if path.startswith("study."):
            provenance = "prospective_protocol_setting; verify against actual study records"
        elif path.startswith("tuning."):
            provenance = "prospective_validation_grid; not evidence of completed tuning"
        else:
            provenance = "implementation_setting; historical experimental selection unverified"
        rows.append({"parameter": path, "value": value, "unit": unit,
                     "definition": description, "selection_provenance": provenance})
    return rows


def export_protocol(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows = parameter_rows(config)
    write_json(output / "parameters.json", rows)
    (output / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (output / "parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "value": json.dumps(r["value"])} for r in rows)
    table = ["# Resolved execution parameters", "",
             "These values describe this configuration. They do not establish values used in an earlier study.",
             "", "| Parameter | Value | Unit | Definition |", "| --- | --- | --- | --- |"]
    table.extend(f"| `{r['parameter']}` | `{json.dumps(r['value'])}` | {r['unit']} | {r['definition']} |"
                 for r in rows)
    (output / "parameters.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(output / "provenance.json", manifest(config, "protocol"))


def audit_run(config: dict, root: Path):
    """Export the geometry actually embedded in records, including legacy samplers."""
    records, malformed = [], []
    for path in sorted(root.rglob("*.npz")):
        if "audit" in path.relative_to(root).parts:
            continue
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if "metadata" not in arrays:
                    continue
                metadata = json.loads(str(arrays["metadata"]))
                if "scene" not in metadata:
                    continue
                records.append({"path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "trial_id": metadata.get("trial_id"), "episode_id": metadata.get("episode_id"),
                    "seed": metadata.get("seed"), "unit": metadata.get("operator_id"),
                    "goal": metadata.get("intended_goal", metadata.get("goal")),
                    "condition": metadata.get("condition", metadata.get("source_condition")),
                    "provenance": metadata.get("provenance"), "scene": metadata["scene"]})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            malformed.append({"path": str(path.relative_to(root)), "error": str(error)})
    write_json(root / "audit/actual_scene_records.json", {"schema_version": 1,
        "records": records, "malformed_records": malformed,
        "interpretation": "record inventory, not a count of independent trials or operators"})
    export_protocol(config, root / "audit")
    write_json(root / "audit/run_inventory.json", manifest(config, "audit-run", [
        root / r["path"] for r in records]))
    return {"record_count": len(records), "malformed_count": len(malformed)}
