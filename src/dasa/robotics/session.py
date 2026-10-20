"""Workstation loop and offline replay of the same nominal hierarchy.

The workstation performs translational admittance at a measured host cadence.
Its velocity interface is not the simulated force plant. Hardware energy traces
are observations and must not be reported as the simulation's passivity theorem.
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np
import yaml

from dasa.config import write_json
from dasa.control.hierarchy import HierarchicalController
from dasa.data import Demonstration
from dasa.delay import DelayBuffer
from dasa.geometry import Scene, clamp_norm
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import load_models
from dasa.robotics.interfaces import FeedbackLedger, MasterUDP, RobotiqSocket, URRTDE
from dasa.robotics.kinematics import jacobian
from dasa.simulation import Trial


def replay(config: dict, models: Path, trial_path: Path, output: Path, condition="C3"):
    trial = Trial.load(trial_path)
    scene = Scene.from_dict(trial.metadata["scene"])
    skill = TPHSMM.load(config, models / "bootstrap_hsmm.npz")
    cvae, _ = load_models(config, models)
    controller = HierarchicalController(config, scene, skill, cvae, condition)
    records, grasped = [], None
    for k in range(len(trial.arrays["time"])):
        a = trial.arrays
        phase = int(a["phase"][k])
        if phase > 0:
            grasped = trial.metadata["grasped_goal"]
        current = a["current_state"][k] if "current_state" in a else np.r_[a["position"][k], a["velocity"][k]]
        decision = controller.decide(a["raw_force"][k], a["observed_state"][k], current,
                                     a["feedback_age"][k], phase, grasped, a["rho"][k],
                                     bool(a["corrective"][k]))
        velocity = clamp_norm(a["velocity"][k] + config["simulation"]["decision_dt"] *
                              decision["nominal_force"] / config["simulation"]["mass"], .08)
        records.append({"time": a["time"][k], "nominal_force": decision["nominal_force"],
                        "bounded_velocity": velocity, "motion_sent": False})
    write_json(output, {"mode": "offline_replay", "condition": condition, "records": records})


def run_workstation(config: dict, hardware_file: Path, models: Path, output: Path,
                    execute=False, condition="C3"):
    hardware = yaml.safe_load(hardware_file.read_text(encoding="utf-8"))
    if hardware["robot"]["host"] == "TO_FILL":
        raise ValueError("Set the actual controller IP in the hardware configuration")
    if hardware["scene"]["provenance"] != "calibrated_scene":
        raise ValueError("A measured base-frame scene is required for workstation operation")
    scene = Scene.from_dict(hardware["scene"])
    if not hardware.get("calibration_record") or not Path(hardware["calibration_record"]).exists():
        raise ValueError("An existing camera/task calibration record is required")
    calibration = json.loads(Path(hardware["calibration_record"]).read_text())
    if calibration["target_frame"] != "base" or calibration["rms_m"] > hardware["limits"]["calibration_rms_m"]:
        raise ValueError("Calibration record is inconsistent or exceeds the allowed residual")
    acknowledgement_deadline = hardware["limits"].get(
        "feedback_acknowledgement_seconds", hardware["limits"]["watchdog_seconds"])
    if (hardware["feedback_delay_seconds"] < 0 or acknowledgement_deadline <= 0
            or hardware["loop_seconds"] <= 0 or not np.isfinite(
                [hardware["feedback_delay_seconds"], acknowledgement_deadline,
                 hardware["loop_seconds"]]).all()):
        raise ValueError("Feedback delay, host cadence and acknowledgement deadline must be finite and valid")
    skill = TPHSMM.load(config, models / "bootstrap_hsmm.npz")
    cvae, _ = load_models(config, models)
    c = copy.deepcopy(config)
    c["simulation"]["decision_dt"] = hardware["loop_seconds"]
    controller = HierarchicalController(c, scene, skill, cvae, condition)
    if hardware["gripper"]["type"] not in {"robotiq_socket", "external"}:
        raise ValueError("Select robotiq_socket or an independently controlled external gripper")
    robot = URRTDE(hardware["robot"]["host"], execute, hardware["limits"]["maximum_speed_m_s"])
    try:
        master = MasterUDP(hardware["master"]["udp_port"], hardware["limits"]["watchdog_seconds"])
    except Exception:
        robot.close()
        raise
    gripper = None
    if hardware["gripper"]["type"] == "robotiq_socket":
        gripper = RobotiqSocket(hardware["robot"]["host"], hardware["gripper"]["port"], execute)
    initial = robot.state()
    observations = DelayBuffer(np.r_[initial.position, initial.velocity, initial.wrench],
                               hardware["feedback_delay_seconds"] + 1)
    feedback_frames = FeedbackLedger()
    started, phase, grasped, grip_previous = time.monotonic(), 0, None, False
    samples, commands, reflection_work, stop_reason = [], [], 0.0, "duration"
    previous_receive_time, previous_reflected = started, np.zeros(3)
    try:
        while time.monotonic() - started < hardware["maximum_seconds"]:
            tick = time.monotonic()
            state = robot.state()
            elapsed = tick - started
            observations.push(elapsed, np.r_[state.position, state.velocity, state.wrench])
            delayed = observations.observe(elapsed, hardware["feedback_delay_seconds"])
            try:
                packet, address = master.read()
            except socket.timeout:
                robot.stop()
                stop_reason = "master_watchdog"
                break
            except (ValueError, KeyError, TypeError):
                robot.stop()
                stop_reason = "invalid_master_packet"
                break
            received_time = packet["host_receive_monotonic"]
            reflection_work += (received_time - previous_receive_time) * float(
                previous_reflected @ np.asarray(packet["velocity_m_s"]))
            previous_receive_time = received_time
            acknowledged = feedback_frames.acknowledge(
                packet.get("observation_sequence"), received_time, acknowledgement_deadline)
            if np.any(state.position < hardware["limits"]["workspace_min_m"]) or np.any(
                    state.position > hardware["limits"]["workspace_max_m"]):
                robot.stop()
                stop_reason = "workspace_boundary"
                break
            if np.linalg.svd(jacobian(state.joints, hardware["robot"]["model"]), compute_uv=False)[-1] < hardware["limits"]["minimum_jacobian_singular_value"]:
                robot.stop()
                stop_reason = "kinematic_singularity"
                break
            raw = clamp_norm(packet["force_N"], hardware["limits"]["maximum_master_force_N"])
            if phase == 1 and state.position[2] >= c["simulation"]["lift_height"] - .05:
                phase = 2
            phase_at_decision = phase
            decision, desired, observed, source_time, feedback_age = None, np.zeros(3), None, None, None
            if acknowledged is not None:
                observed = acknowledged.state
                source_time = acknowledged.source_time
                feedback_age = received_time - acknowledged.host_source_time
                decision = controller.decide(raw, observed, np.r_[state.position, state.velocity],
                                             feedback_age, phase, grasped, 0.0)
                desired = clamp_norm(state.velocity + hardware["loop_seconds"] * decision["nominal_force"] /
                                      c["simulation"]["mass"], hardware["limits"]["maximum_speed_m_s"])
            # Stop before entry into an inflated obstacle; this is a workstation interlock.
            next_position = state.position + desired * hardware["limits"]["watchdog_seconds"]
            clearance = scene.swept_clearance(state.position, next_position,
                                              c["simulation"]["robot_radius"] + hardware["limits"]["obstacle_margin_m"])
            enabled = bool(packet["enable"] and acknowledged is not None and clearance > 0
                           and state.safety_mode in {1, 2})
            executed = robot.speed(desired, hardware["loop_seconds"], enabled)
            if enabled and (executed or not execute) and packet["grip"] != grip_previous:
                if gripper:
                    gripper.command(packet["grip"])
                grip_previous = packet["grip"]
                if grip_previous:
                    grasped = int(np.argmin(np.linalg.norm(scene.goals - state.position, axis=1)))
                    phase = 1
                elif phase >= 1:
                    phase = 3
            # A real sample must first spend the declared interval in the host buffer.
            feedback_ready = elapsed - delayed.source_time >= hardware["feedback_delay_seconds"] - 1e-9
            feedback = None
            if feedback_ready:
                feedback = feedback_frames.issue(delayed.state[:6], delayed.state[6:],
                                                  delayed.source_time,
                                                  started + delayed.source_time, time.monotonic())
            reflected = clamp_norm(-delayed.state[6:9], hardware["limits"]["maximum_reflection_force_N"])
            if not enabled or not executed or feedback is None:
                reflected = np.zeros(3)
            master.reflect_force(reflected, address, feedback.payload() if feedback else None)
            previous_reflected = reflected.copy()
            samples.append({"time": elapsed, "robot_timestamp": state.timestamp,
                            "position": state.position, "velocity": state.velocity,
                            "raw_force": raw, "observed_state": observed,
                            "source_time": source_time, "feedback_age": feedback_age,
                            "feedback_acknowledged": acknowledged is not None,
                            "acknowledged_observation_sequence": (
                                acknowledged.observation_sequence if acknowledged else None),
                            "sent_observation_sequence": feedback.observation_sequence if feedback else None,
                            "sent_feedback_source_time": feedback.source_time if feedback else None,
                            "master_sequence": packet["sequence"],
                            "host_receive_time": received_time - started, "phase": phase_at_decision,
                            "measured_wrench": state.wrench,
                            "delayed_feedback_wrench": delayed.state[6:],
                            "reflected_force": reflected,
                            "master_velocity": packet["velocity_m_s"],
                            "cumulative_reflection_work_J": reflection_work,
                            "actual_cadence_seconds": time.monotonic() - tick,
                            "enabled": enabled, "executed": bool(executed)})
            commands.append(decision)
            remaining = hardware["loop_seconds"] - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining > hardware["limits"]["watchdog_seconds"]:
                robot.stop()
                stop_reason = "host_cadence_watchdog"
                break
    finally:
        robot.close()
        master.close()
        write_json(output, {"schema_version": 2, "provenance": "UR10_workstation_measurement",
                            "execute": execute, "condition": condition, "stop_reason": stop_reason,
                            "scene": scene.to_dict(), "samples": samples, "decisions": commands,
                            "timing": {"feedback_delay_seconds": hardware["feedback_delay_seconds"],
                                       "feedback_channels": ["cartesian_state", "measured_wrench"],
                                       "feedback_definition": "one_way_host_buffer_then_bridge_acknowledgement",
                                       "acknowledgement_deadline_seconds": acknowledgement_deadline,
                                       "nominal_cadence_seconds": hardware["loop_seconds"],
                                       "robot_model": hardware["robot"]["model"],
                                       "command_delay_seconds": 0.0},
                            "outcome": "requires_independent_task_annotation",
                            "reflection_work_scope": "host_interval_held_force_estimate_from_received_master_velocity",
                            "passivity_scope": "measured trace only; no hardware passivity assertion"})


def import_recording(recording: Path, annotations: Path, output: Path):
    data = json.loads(recording.read_text())
    metadata = json.loads(annotations.read_text())
    for field in ("episode_id", "operator_id", "intended_goal", "success", "collision", "placed",
                  "bounds_violation", "source_condition", "delay_seconds"):
        if metadata.get(field) is None:
            raise ValueError(f"Missing independent task annotation: {field}")
    samples = data["samples"]
    if (len(samples) < 3 or not data["execute"]
            or data.get("provenance") != "UR10_workstation_measurement"):
        raise ValueError("Only executed, nonempty measured recordings may become hardware demonstrations")
    for key in ("success", "collision", "placed", "bounds_violation"):
        if not isinstance(metadata[key], bool):
            raise ValueError(f"Annotation {key} must be boolean")
    if "grasped_goal" not in metadata or (metadata["success"] and metadata["grasped_goal"] is None):
        raise ValueError("Record grasped_goal explicitly; successful trials require a grasped object ID")
    if data.get("schema_version", 0) < 2 or any(
            sample.get("executed") and sample.get("feedback_acknowledged") is not True
            for sample in samples):
        raise ValueError("Executed demonstrations require acknowledged displayed feedback")
    acknowledged_indices = [index for index, sample in enumerate(samples)
                            if sample.get("feedback_acknowledged") is True]
    if not acknowledged_indices:
        raise ValueError("The recording has no acknowledged feedback samples")
    prefix = acknowledged_indices[0]
    samples = samples[prefix:]
    if len(samples) < 3 or not any(sample.get("executed") is True for sample in samples):
        raise ValueError("At least three acknowledged samples and an executed command are required")
    if any(sample.get("feedback_acknowledged") is not True for sample in samples):
        raise ValueError("An unacknowledged feedback gap cannot be silently removed from a demonstration")
    source = np.asarray([sample["source_time"] for sample in samples], float)
    age = np.asarray([sample["feedback_age"] for sample in samples], float)
    receive = np.asarray([sample["host_receive_time"] for sample in samples], float)
    sequences = [sample["acknowledged_observation_sequence"] for sample in samples]
    timing = data.get("timing", {})
    declared_delay = timing.get("feedback_delay_seconds")
    if (declared_delay is None or not np.isfinite(declared_delay) or declared_delay < 0
            or not np.isfinite(metadata["delay_seconds"])
            or not np.isclose(metadata["delay_seconds"], declared_delay)):
        raise ValueError("Annotated delay must match the recorded one-way feedback buffer setting")
    if (not np.isfinite(np.r_[source, age, receive]).all() or np.any(source < 0)
            or np.any(np.diff(source) < 0) or np.any(age < declared_delay - 1e-8)
            or not np.allclose(receive - source, age, atol=1e-7, rtol=0)
            or any(not isinstance(sequence, int) or isinstance(sequence, bool)
                   or sequence < 0 for sequence in sequences)
            or np.any(np.diff(sequences) < 0)):
        raise ValueError("Recorded feedback timing or acknowledgement sequence is inconsistent")
    if data.get("condition") != metadata["source_condition"]:
        raise ValueError("Annotated source condition must match the workstation recording")
    metadata["provenance"] = "UR10_measured_teleoperation"
    metadata["source_recording"] = str(recording)
    metadata["source_recording_sha256"] = hashlib.sha256(recording.read_bytes()).hexdigest()
    metadata["hardware_timing"] = timing
    metadata["feedback_timing_audit"] = {
        "initial_unacknowledged_samples": prefix,
        "clock": "host_session_relative_seconds",
        "timing_array_names": ["observed_source_time", "feedback_age", "command_receive_time",
                               "acknowledged_observation_sequence", "master_sequence", "executed"],
        "minimum_feedback_age_seconds": float(age.min()),
        "maximum_feedback_age_seconds": float(age.max())}
    demo = Demonstration(np.asarray([s["time"] for s in samples]),
                          *[np.asarray([s[k] for s in samples]) for k in
                            ("position", "velocity", "raw_force", "observed_state", "phase")],
                          Scene.from_dict(data["scene"]), metadata.pop("intended_goal"), metadata)
    # Preserve the standard demonstration schema and its validated timing extensions.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.stem + ".import-source.npz")
    temporary = output.with_name(output.stem + ".import-timing.tmp.npz")
    try:
        demo.save(staging)
        with np.load(staging, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays.update(observed_source_time=source, feedback_age=age, command_receive_time=receive,
                      acknowledged_observation_sequence=np.asarray(sequences, dtype=np.int64),
                      master_sequence=np.asarray([sample["master_sequence"] for sample in samples],
                                                 dtype=np.int64),
                      executed=np.asarray([sample["executed"] for sample in samples], dtype=bool))
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
    finally:
        staging.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
