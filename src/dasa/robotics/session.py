from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.control.hierarchy import HierarchicalController
from dasa.control.motion import ForceKalman
from dasa.data import Demonstration
from dasa.delay import DelayBuffer
from dasa.geometry import Scene, clamp_norm
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import load_models
from dasa.robotics.interfaces import FeedbackLedger, MasterUDP, RobotiqSocket, URRTDE
from dasa.robotics.configuration import load_hardware
from dasa.robotics.kinematics import jacobian
from dasa.simulation import Trial


class ManualController:
    def __init__(self, config: dict):
        self.maximum_force = config["simulation"]["force_limit"]
        self.kalman = ForceKalman(config["motion"]["kalman_process"],
                                  config["motion"]["kalman_measurement"],
                                  config["motion"].get("kalman_initial_variance", 1.0))

    def decide(self, raw, observed, current, age, phase, grasped_goal, rho):
        human = self.kalman.step(np.asarray(raw, float))
        return {"human_force": human,
                "nominal_force": clamp_norm(human, self.maximum_force),
                "alpha": 0.0, "lambda": 0.0}


def admittance_velocity(force, velocity, config: dict, dt: float, maximum_speed=.08):
    force, velocity = np.asarray(force, float), np.asarray(velocity, float)
    mass, damping = config["simulation"]["mass"], config["simulation"]["damping"]
    if (force.shape != (3,) or velocity.shape != (3,)
            or not np.isfinite(np.r_[force, velocity, mass, damping, dt, maximum_speed]).all()
            or mass <= 0 or damping < 0 or dt <= 0 or maximum_speed <= 0):
        raise ValueError("Admittance needs finite 3D signals and positive mass, step and speed limit")
    desired = ((2 * mass - dt * damping) * velocity + 2 * dt * force) / (2 * mass + dt * damping)
    return clamp_norm(desired, maximum_speed)


def replay(config: dict, models: Path, trial_path: Path, output: Path, condition="C3",
           skill_path: Path | None = None):
    trial = Trial.load(trial_path)
    scene = Scene.from_dict(trial.metadata["scene"])
    skill_file = Path(skill_path) if skill_path is not None else models / "bootstrap_hsmm.npz"
    skill = TPHSMM.load(config, skill_file)
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
        velocity = admittance_velocity(decision["nominal_force"], a["velocity"][k], config,
                                        config["simulation"]["decision_dt"])
        records.append({"time": a["time"][k], "nominal_force": decision["nominal_force"],
                        "bounded_velocity": velocity, "motion_sent": False})
    write_json(output, {"mode": "offline_replay", "condition": condition, "records": records,
                        "skill_file": skill_file.name,
                        "skill_sha256": hashlib.sha256(skill_file.read_bytes()).hexdigest()})


def run_workstation(config: dict, hardware_file: Path, models: Path | None, output: Path,
                    execute=False, condition="C3", skill_path: Path | None = None):
    if models is None and (condition != "C1" or skill_path is not None):
        raise ValueError("Model-free acquisition supports manual C1 without a learned skill")
    hardware = load_hardware(hardware_file)
    scene = Scene.from_dict(hardware["scene"])
    acknowledgement_deadline = hardware["limits"].get(
        "feedback_acknowledgement_seconds", hardware["limits"]["watchdog_seconds"])
    c = copy.deepcopy(config)
    c["simulation"]["decision_dt"] = hardware["loop_seconds"]
    model_hashes = {}
    skill_file, skill_digest = None, None
    if models is None:
        controller = ManualController(c)
        controller_mode = "manual_bootstrap"
    else:
        models = Path(models)
        skill_file = Path(skill_path) if skill_path is not None else models / "bootstrap_hsmm.npz"
        skill_digest = hashlib.sha256(skill_file.read_bytes()).hexdigest()
        skill = TPHSMM.load(config, skill_file)
        cvae, _ = load_models(config, models)
        controller = HierarchicalController(c, scene, skill, cvae, condition)
        controller_mode = "trained_shared_control"
        model_hashes = {"skill": skill_digest, **{
            name: hashlib.sha256((models / name).read_bytes()).hexdigest()
            for name in ("cvae.pt", "lstm.pt", "transformer.pt") if (models / name).is_file()}}
    robot, master, gripper = None, None, None
    address = None
    samples, commands, gripper_commands = [], [], []
    reflection_work, stop_reason = 0.0, "initialization"
    cleanup_errors = []
    try:
        robot = URRTDE(hardware["robot"]["host"], execute, hardware["limits"]["maximum_speed_m_s"])
        master = MasterUDP(hardware["master"]["udp_port"], hardware["limits"]["watchdog_seconds"])
        if hardware["gripper"]["type"] == "robotiq_socket":
            gc = hardware["gripper"]
            gripper = RobotiqSocket(hardware["robot"]["host"], gc["port"], execute,
                                     gc.get("speed", 64), gc.get("force", 64),
                                     gc.get("timeout_seconds", 1.0))
            if execute:
                status = gripper.status()
                if not status["active"] or status["activation_status"] != 3 or status["fault"]:
                    raise RuntimeError("Activate the Robotiq gripper before starting a measured trial")
        initial = robot.state()
        if np.linalg.norm(initial.position - scene.start) > hardware["limits"].get("start_tolerance_m", .05):
            raise ValueError("Measured TCP position exceeds the declared initial-pose tolerance")
        observations = DelayBuffer(np.r_[initial.position, initial.velocity, initial.wrench],
                                   hardware["feedback_delay_seconds"] + 1)
        feedback_frames = FeedbackLedger()
        started, phase, grasped, grip_previous = time.monotonic(), 0, None, False
        previous_receive_time, previous_reflected = started, np.zeros(3)
        stop_reason = "duration"
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
                desired = admittance_velocity(decision["nominal_force"], state.velocity, c,
                                                hardware["loop_seconds"],
                                                hardware["limits"]["maximum_speed_m_s"])

            next_position = state.position + desired * hardware["limits"]["watchdog_seconds"]
            clearance = scene.swept_clearance(state.position, next_position,
                                              c["simulation"]["robot_radius"] + hardware["limits"]["obstacle_margin_m"])
            enabled = bool(packet["enable"] and acknowledged is not None and clearance > 0
                           and state.safety_mode in {1, 2})
            executed = robot.speed(desired, hardware["loop_seconds"], enabled)
            if enabled and (executed or not execute) and packet["grip"] != grip_previous:
                if gripper:
                    gripper_commands.append({"time": elapsed, **gripper.command(packet["grip"])})
                grip_previous = packet["grip"]
                if grip_previous:
                    grasped = int(np.argmin(np.linalg.norm(scene.goals - state.position, axis=1)))
                    phase = 1
                elif phase >= 1:
                    phase = 3

            feedback_ready = elapsed - delayed.source_time >= hardware["feedback_delay_seconds"] - 1e-9
            feedback = None
            if feedback_ready:
                feedback = feedback_frames.issue(delayed.state[:6], delayed.state[6:],
                                                  delayed.source_time,
                                                  started + delayed.source_time, time.monotonic())
            reflected = clamp_norm(-delayed.state[6:9], hardware["limits"]["maximum_reflection_force_N"])
            if not enabled or not executed or feedback is None or phase == 3:
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
                            "master_skipped_packets": packet.get("skipped_packets", 0),
                            "master_invalid_packets": packet.get("invalid_packets", 0),
                            "host_receive_time": received_time - started, "phase": phase_at_decision,
                            "phase_after_action": phase,
                            "measured_wrench": state.wrench,
                            "delayed_feedback_wrench": delayed.state[6:],
                            "reflected_force": reflected,
                            "master_velocity": packet["velocity_m_s"],
                            "cumulative_reflection_work_J": reflection_work,
                            "actual_cadence_seconds": time.monotonic() - tick,
                            "enabled": enabled, "executed": bool(executed)})
            commands.append(decision)
            if phase == 3:
                robot.stop()
                stop_reason = "gripper_release"
                break
            remaining = hardware["loop_seconds"] - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining > hardware["limits"]["watchdog_seconds"]:
                robot.stop()
                stop_reason = "host_cadence_watchdog"
                break
    except BaseException as exc:
        stop_reason = f"exception_{type(exc).__name__}"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        if master is not None and address is not None:
            try:
                master.reflect_force(np.zeros(3), address, feedback=None)
            except Exception as exc:
                cleanup_errors.append({"device": "master_zero_force", "error_type": type(exc).__name__})
        for name, device in (("robot", robot), ("master", master)):
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    cleanup_errors.append({"device": name, "error_type": type(exc).__name__})
        write_json(output, {"schema_version": 2, "provenance": "UR10_workstation_measurement",
                            "execute": execute, "condition": condition, "stop_reason": stop_reason,
                            "scene": scene.to_dict(), "samples": samples, "decisions": commands,
                            "controller_mode": controller_mode, "model_hashes": model_hashes,
                            **({"skill_file": skill_file.name, "skill_sha256": skill_digest}
                               if skill_file is not None else {}),
                            "calibration_sha256": hardware["calibration_sha256"],
                            "snapshot_sha256": hardware.get("snapshot_sha256"),
                            "gripper_commands": gripper_commands, "cleanup_errors": cleanup_errors,
                            "execution_contract": {
                                "maximum_seconds": hardware["maximum_seconds"],
                                "robot_radius_m": c["simulation"]["robot_radius"],
                                "place_tolerance_m": c["simulation"]["place_tolerance"],
                                "workspace_min_m": hardware["limits"]["workspace_min_m"],
                                "workspace_max_m": hardware["limits"]["workspace_max_m"]},
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
        if cleanup_errors and not unwinding:
            raise RuntimeError("Workstation device cleanup reported an error")


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
    assignment = data.get("experiment_assignment")
    if assignment is not None:
        for annotation_key, assignment_key in (("intended_goal", "intended_goal"),
                                                ("operator_id", "participant_id"),
                                                ("episode_id", "trial_id")):
            if metadata[annotation_key] != assignment.get(assignment_key):
                raise ValueError(f"Annotation {annotation_key} conflicts with the measured trial assignment")
        metadata["experiment_assignment"] = assignment
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
    metadata["controller_mode"] = data.get("controller_mode", "recording_mode_unspecified")
    metadata["model_hashes"] = data.get("model_hashes", {})
    metadata["source_recording"] = str(recording)
    metadata["source_recording_sha256"] = hashlib.sha256(recording.read_bytes()).hexdigest()
    metadata["hardware_timing"] = timing
    if "execution_contract" in data:
        metadata["execution_contract"] = data["execution_contract"]
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
