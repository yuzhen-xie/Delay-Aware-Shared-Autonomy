from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.control.hierarchy import HierarchicalController
from dasa.control.motion import ForceKalman
from dasa.control.physical_passivity import PhysicalPassivity
from dasa.data import Demonstration
from dasa.delay import DelayBuffer
from dasa.geometry import Scene, clamp_norm
from dasa.models.hsmm import TPHSMM
from dasa.models.intent import load_models
from dasa.robotics.camera import CameraUnavailable, RGBDStream, RealSenseCamera, VideoServer
from dasa.robotics.configuration import load_hardware
from dasa.robotics.interfaces import FeedbackLedger, MasterUDP, RobotiqSocket, URRTDE
from dasa.robotics.kinematics import jacobian
from dasa.robotics.task import GraspPlaceTask, GripperWorker
from dasa.robotics.master import MasterRecorder


class ManualController:
    def __init__(self, config):
        self.config = config
        self.maximum_force = config["task"]["force_limit"]
        self.kalman = ForceKalman(config["motion"]["kalman_process"],
                                  config["motion"]["kalman_measurement"],
                                  config["motion"].get("kalman_initial_variance", 1.),
                                  config["task"]["decision_dt"])

    def decide(self, raw, observed, current, age, phase, grasped_goal, rho):
        human = clamp_norm(self.kalman.step(np.asarray(raw, float), self.config["task"]["decision_dt"]), self.maximum_force)
        return {"human_force": human, "motion_force": human, "nominal_force": human,
                "autonomous_force": np.zeros(3), "autonomous_damping_gain": 0.,
                "alpha": 0., "lambda": 0., "authority_scale": 1.,
                "autonomy_confidence": 0., "human_confidence": 1. - rho}


def _scene_difference(a, b):
    if a.frame != b.frame or a.obstacle_ids != b.obstacle_ids:
        return float("inf")
    errors = []
    for name in ("goals", "start", "place", "centers", "radii"):
        x, y = getattr(a, name), getattr(b, name)
        if x.shape != y.shape:
            return float("inf")
        errors.append(float(np.max(np.abs(x - y), initial=0.)))
    return max(errors)


def run_workstation(config, hardware_file, models, output, execute=False, condition="C3",
                    skill_path=None, autonomous_goal=None, stage=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Measured recordings are immutable; select a new output file")
    if condition == "autonomous" and autonomous_goal is None:
        raise ValueError("A physical autonomous evaluation requires an assigned goal")
    if models is None and (condition != "C1" or skill_path is not None):
        raise ValueError("Model-free acquisition is available only for initial C1 demonstrations")
    hardware = load_hardware(hardware_file)
    scene = Scene.from_dict(hardware["scene"])
    c = copy.deepcopy(config)
    c["task"].update(decision_dt=hardware["loop_seconds"], max_seconds=hardware["maximum_seconds"])
    limits, period = hardware["limits"], hardware["loop_seconds"]
    session_id = uuid.uuid4().hex
    acquisition_started_utc = datetime.now(timezone.utc).isoformat()
    stage = stage or ("human_bootstrap" if models is None else
                      "autonomous_evaluation" if condition == "autonomous" else "human_shared_control")
    controller, hashes = ManualController(c), {}
    if models is not None:
        models = Path(models)
        skill_file = Path(skill_path) if skill_path is not None else models / "bootstrap_hsmm.npz"
        cvae, _ = load_models(c, models)
        controller = HierarchicalController(c, scene, TPHSMM.load(c, skill_file), cvae, condition)
        hashes = {"skill": hashlib.sha256(skill_file.read_bytes()).hexdigest(),
                  **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.glob("*.pt")}}
    engine = PhysicalPassivity(c, session_id)
    task = GraspPlaceTask(c, scene, autonomous_goal if condition == "autonomous" else None)
    robot = master = grip = stream = video = None
    address = None
    samples, decisions, startup, video_pairs = [], [], [], {}
    last_force, stop_reason, error_record = np.zeros(3), "initialization", None
    started = trial_start = None
    deadman_released = False
    cleanup_errors = []
    recorder = MasterRecorder(output.with_suffix(".samples"))
    try:
        robot = URRTDE(hardware["robot"]["host"], execute, limits["maximum_speed_m_s"],
                       hardware["robot"]["model"], c["task"]["force_limit"])
        tool, mode = hardware["tool"], hardware["force_mode"]
        robot.configure_tool(tool["tcp_pose"], tool["payload_mass_kg"], tool["payload_cog_m"],
                             mode["damping"], mode["gain"])
        if hardware["gripper"]["type"] != "robotiq_socket":
            raise ValueError("This task executor requires measured Robotiq object-contact feedback")
        gc = hardware["gripper"]
        device = RobotiqSocket(hardware["robot"]["host"], gc["port"], execute,
                                gc["speed"], gc["force"], gc["timeout_seconds"])
        status = device.status()
        if not status["active"] or status["activation_status"] != 3 or status["fault"]:
            raise RuntimeError("The physical gripper must be activated and fault-free before acquisition")
        if status["position"] > 10 or status["object_status"] != 3:
            raise RuntimeError("Open the physical gripper and remove retained objects before starting the trial")
        grip = GripperWorker(device).start()
        cc = hardware["camera"]
        stream = RGBDStream(RealSenseCamera(hardware["calibration_record"],
                              output.with_suffix(".rgbd"), cc["width"], cc["height"], cc["fps"],
                              cc.get("serial")), hardware["perception_record"], cc["scene_mode"],
                              max(cc["history_seconds"], hardware["feedback_delay_seconds"] + 2.)).start()
        deadline = time.monotonic() + 5.
        while True:
            try:
                measured_scene = Scene.from_dict(stream.snapshot()["scene"])
                break
            except CameraUnavailable:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        if _scene_difference(scene, measured_scene) > c.get("study", {}).get("hardware_scene_tolerance_m", .005):
            raise ValueError("Live calibrated geometry differs from the assigned measured layout")
        video = VideoServer(hardware["video"]["host"], hardware["video"]["port"],
                             cc["maximum_pairing_skew_seconds"]).start()
        master = MasterUDP(hardware["master"]["udp_port"], min(period, limits["watchdog_seconds"]))
        initial = robot.state()
        if np.linalg.norm(initial.position - scene.start) > limits["start_tolerance_m"]:
            raise ValueError("Actual TCP position is outside the assigned start tolerance")
        observations = DelayBuffer(np.r_[initial.position, initial.velocity, initial.wrench],
                                   hardware["feedback_delay_seconds"] + 2.)
        ledger = FeedbackLedger()
        started, last_robot_stamp = time.monotonic(), None
        robot.configure_watchdog(c["passivity"]["physical"]["robot_maximum_hold_seconds"])
        next_tick, stop_reason = started, "arming_timeout"
        previous_frame_sequence = -1
        while True:
            tick = time.monotonic()
            if trial_start is None and tick - started > hardware.get("arming_timeout_seconds", 30.):
                break
            if trial_start is not None and tick - trial_start >= hardware["maximum_seconds"]:
                stop_reason = "timeout"
                break
            state = robot.state()
            fresh_deadline = tick + min(period, .01)
            while last_robot_stamp is not None and state.timestamp <= last_robot_stamp:
                if time.monotonic() >= fresh_deadline:
                    raise RuntimeError("RTDE state did not advance within its sampling deadline")
                time.sleep(.0002)
                state = robot.state()
            sample_time = time.monotonic()
            last_robot_stamp = state.timestamp
            elapsed = sample_time - started
            observations.push(elapsed, np.r_[state.position, state.velocity, state.wrench])
            packet, address = master.read()
            received = packet["host_receive_monotonic"]
            if "energy_packet" not in packet:
                raise ValueError("A physical master command requires its PC1 energy ledger")
            epoch_ready = packet.get("robot_session_ack") == session_id
            engine.begin_sample(sample_time, state.wrench[:3], state.position, state.velocity,
                                packet["energy_packet"] if epoch_ready else None,
                                executed_command_force_N=last_force)
            delayed = observations.observe(elapsed, hardware["feedback_delay_seconds"])
            feedback = None
            if elapsed - delayed.source_time >= hardware["feedback_delay_seconds"] - 1e-9:
                image = stream.at(started + delayed.source_time, cc["maximum_pairing_skew_seconds"],
                                  cc["maximum_age_seconds"])
                if image.sequence != previous_frame_sequence:
                    frame = ledger.issue(delayed.state[:6], delayed.state[6:], delayed.source_time,
                                           started + delayed.source_time, time.monotonic())
                    reflection = clamp_norm(-delayed.state[6:9], limits["maximum_reflection_force_N"])
                    pairing = video.publish(image, frame.observation_sequence, frame.source_time,
                                              frame.host_source_time, frame.state, frame.wrench[:3], reflection)
                    feedback = {**frame.payload(), **pairing}
                    video_pairs[frame.observation_sequence] = pairing
                    previous_frame_sequence = image.sequence
            acknowledged = ledger.acknowledge(packet.get("observation_sequence"), received,
                                                limits["feedback_acknowledgement_seconds"])
            if acknowledged is not None:
                pairing = video_pairs.get(acknowledged.observation_sequence)
                if (pairing is None or packet.get("video_sequence") != pairing["video_sequence"]
                        or packet.get("video_session_id") != pairing["video_session_id"]):
                    acknowledged = None
            if not packet["enable"]:
                deadman_released = True
            if np.any(state.position < limits["workspace_min_m"]) or np.any(state.position > limits["workspace_max_m"]):
                raise RuntimeError("Actual TCP crossed its commissioned workspace boundary")
            if state.safety_mode != 1:
                raise RuntimeError("The UR controller is not in normal safety mode")
            if np.linalg.norm(state.velocity) > limits["maximum_speed_m_s"] + .002:
                raise RuntimeError("Measured TCP speed exceeded the commissioned bound")
            if np.linalg.svd(jacobian(state.joints, hardware["robot"]["model"]), compute_uv=False)[-1] < limits["minimum_jacobian_singular_value"]:
                raise RuntimeError("Measured joints approached the declared singularity limit")
            stream.latest(cc["maximum_age_seconds"])
            if cc["scene_mode"] == "online":
                current_scene = Scene.from_dict(stream.snapshot()["scene"])
                if isinstance(controller, HierarchicalController):
                    controller.update_scene(current_scene, freeze_task_frames=task.phase > 0)
                    scene = controller.scene
                    task.scene = scene
            clearance = float(scene.clearance(state.position, c["task"]["robot_radius"]).min())
            safety_distance = limits["maximum_speed_m_s"] * c["passivity"]["physical"]["robot_maximum_hold_seconds"]
            enabled = bool(packet["enable"] and deadman_released and epoch_ready and acknowledged is not None
                           and video.connected and clearance > limits["obstacle_margin_m"] + safety_distance)
            if trial_start is not None and not enabled:
                raise RuntimeError("An active trial lost deadman, displayed feedback or geometric clearance")
            observed = acknowledged.state if acknowledged else None
            source_time = acknowledged.source_time if acknowledged else None
            age = received - acknowledged.host_source_time if acknowledged else None
            raw = clamp_norm(packet["force_N"], limits["maximum_master_force_N"])
            decision, corrected, phase_at_decision = None, np.zeros(3), task.phase
            if enabled:
                if trial_start is None:
                    trial_start = sample_time
                    stop_reason = "active"
                actual_dt = engine.port.last_dt or period
                c["task"]["decision_dt"] = actual_dt
                port_audit = engine.diagnostics()
                removed_j = port_audit["rho_window_removed_work_j"] + float(packet.get("rho_window_removed_work_j", 0.))
                nominal_j = port_audit["rho_window_nominal_positive_work_j"] + float(packet.get("rho_window_nominal_positive_work_j", 0.))
                rho = float(np.clip(removed_j / max(nominal_j, 1e-12), 0., 1.))
                engine.set_transparency_ratio(rho)
                if isinstance(controller, HierarchicalController):
                    decision = controller.decide(raw, observed, np.r_[state.position, state.velocity],
                        age, task.phase, task.grasped_goal, rho,
                        autonomous_goal=autonomous_goal if condition == "autonomous" else None)
                else:
                    decision = controller.decide(raw, observed, np.r_[state.position, state.velocity],
                                                 age, task.phase, task.grasped_goal, rho)
                alpha, blend = decision["alpha"], decision["lambda"]
                original = decision["autonomous_force"]
                corrected_velocity = engine.limit_autonomy_velocity(original, state.velocity, alpha,
                                    blend, actual_dt, three_port=condition != "C1")
                autonomy = original + decision["autonomous_damping_gain"] * (state.velocity - corrected_velocity)
                nominal = decision["authority_scale"] * ((1 - alpha) * decision["motion_force"] + alpha * autonomy)
                nominal = clamp_norm(nominal, c["task"]["force_limit"])
                corrected = engine.limit_robot_force(nominal, state.velocity, alpha, blend, period,
                                                       three_port=condition != "C1")
                decision.update(autonomous_force_after_PC3=autonomy, nominal_force=nominal,
                                 executed_force=corrected, PC3_velocity=corrected_velocity, rho=rho)
            else:
                engine.limit_autonomy_velocity(np.zeros(3), state.velocity, 0., 0., period,
                                               three_port=condition != "C1")
                corrected = engine.limit_robot_force(np.zeros(3), state.velocity, 0., 0., period,
                                                       three_port=condition != "C1")
            executed = robot.force(corrected, period, bool(enabled and execute))
            last_force = corrected.copy() if executed else np.zeros(3)
            status = grip.status()
            request = task.update(elapsed, state.position, state.velocity, packet["grip"], status, enabled and execute)
            if execute and enabled:
                grip.request(request)
            reflected = clamp_norm(-delayed.state[6:9], limits["maximum_reflection_force_N"]) if enabled else np.zeros(3)
            master.reflect_force(reflected, address, feedback, engine.packet())
            record = {"time": elapsed, "robot_timestamp": state.timestamp, "position": state.position,
                "rotation_vector": state.rotation_vector, "velocity": state.velocity, "joints": state.joints,
                "raw_force": raw, "observed_state": observed, "source_time": source_time,
                "feedback_age": age, "feedback_acknowledged": acknowledged is not None,
                "acknowledged_observation_sequence": acknowledged.observation_sequence if acknowledged else None,
                "acknowledged_video_sequence": packet.get("video_sequence"),
                "acknowledged_video_session_id": packet.get("video_session_id"),
                "sent_observation_sequence": feedback["observation_sequence"] if feedback else None,
                "master_sequence": packet["sequence"], "master_skipped_packets": packet.get("skipped_packets", 0),
                "host_receive_time": received - started, "phase": phase_at_decision,
                "phase_after_action": task.phase, "measured_wrench": state.wrench,
                "reflected_force": reflected, "executed_force": last_force,
                "master_velocity": packet["velocity_m_s"], "master_energy": packet["energy_packet"],
                "robot_session_ack": packet.get("robot_session_ack"),
                "passivity": engine.diagnostics(), "decision": decision, "gripper_status": status,
                "minimum_clearance_m": clearance, "enabled": enabled, "executed": bool(executed),
                "scene": scene.to_dict() if cc["scene_mode"] == "online" else None,
                "actual_cadence_seconds": engine.port.last_dt}
            recorder.append(record)
            if trial_start is None:
                startup.append(record)
            else:
                samples.append(record)
                decisions.append(decision)
            if task.stop_reason:
                stop_reason = task.stop_reason
                break
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining > c["passivity"]["physical"]["robot_maximum_hold_seconds"]:
                raise RuntimeError("Robot command scheduling exceeded the reserved hold interval")
    except BaseException as exc:
        stop_reason = f"fault_{type(exc).__name__}"
        error_record = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        if robot is not None:
            try:
                robot.stop()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if master is not None and address is not None:
            try:
                master.reflect_force(np.zeros(3), address, None, engine.packet())
            except Exception as exc:
                cleanup_errors.append(str(exc))
        for device in (grip, video, stream, master, robot, recorder):
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
        write_json(output, {"schema_version": 3, "provenance": "UR10_workstation_measurement",
            "session_id": session_id, "execute": execute, "condition": condition, "stage": stage,
            "controller_mode": "manual_bootstrap" if models is None else "trained_physical_control",
            "scene": Scene.from_dict(hardware["scene"]).to_dict(), "model_hashes": hashes,
            "samples": samples, "commands": decisions, "startup_samples": startup,
            "robot_commands": robot.commands if robot is not None else [],
            "sample_journal": str(output.with_suffix(".samples") / "samples.jsonl"),
            "gripper_commands": grip.events if grip is not None else [], "task_events": task.events,
            "stop_reason": stop_reason, "error": error_record, "cleanup_errors": cleanup_errors,
            "acquisition_started_host_monotonic": started, "trial_started_host_monotonic": trial_start,
            "acquisition_started_utc": acquisition_started_utc,
            "task_outcome": "requires_independent_annotation", "passivity": engine.diagnostics(),
            "execution_contract": {"maximum_seconds": hardware["maximum_seconds"],
                "robot_radius_m": c["task"]["robot_radius"], "place_tolerance_m": c["task"]["place_tolerance"],
                "workspace_min_m": limits["workspace_min_m"], "workspace_max_m": limits["workspace_max_m"]},
            "timing": {"feedback_delay_seconds": hardware["feedback_delay_seconds"],
                "command_delay_seconds": 0., "feedback_channels": ["displayed_RGBD", "cartesian_state", "wrench"],
                "feedback_definition": "one_way_host_buffer_with_paired_video_presentation_acknowledgement",
                "nominal_cadence_seconds": period, "requested_master_period_seconds": hardware["master"]["servo_seconds"],
                "robot_model": hardware["robot"]["model"], "hard_realtime_asserted": False,
                "scene_mode": hardware["camera"]["scene_mode"],
                "camera_timestamp_definition": "host_frame_receipt_with_retained_sensor_clock_timestamps"}})
        if cleanup_errors and not unwinding:
            raise RuntimeError("Device cleanup reported errors; retain the recorded diagnostics")


def import_recording(recording: Path, annotations: Path, output: Path):
    data = json.loads(recording.read_text())
    metadata = json.loads(annotations.read_text())
    if metadata.get("source_recording_sha256") != hashlib.sha256(recording.read_bytes()).hexdigest():
        raise ValueError("The independently annotated recording has changed")
    if metadata.get("stage") != data.get("stage"):
        raise ValueError("Acquisition stage conflicts with its annotation")
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
    metadata["acquisition_started_utc"] = data["acquisition_started_utc"]
    metadata["observed_minimum_surface_clearance_m"] = float(min(s["minimum_clearance_m"] for s in samples))
    metadata["reference_definition"] = "unaltered_causal_LQT_reference_at_each_recorded_robot_decision"
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
        if all(isinstance(sample.get("decision"), dict) and "reference" in sample["decision"] for sample in samples):
            arrays["reference"] = np.asarray([sample["decision"]["reference"] for sample in samples], float)
            arrays["reference_velocity"] = np.asarray([sample["decision"]["reference_velocity"] for sample in samples], float)
            arrays["reference_time"] = np.asarray([sample["time"] for sample in samples], float)
        arrays["measured_wrench"] = np.asarray([sample["measured_wrench"] for sample in samples], float)
        arrays["executed_force"] = np.asarray([sample["executed_force"] for sample in samples], float)
        arrays["surface_clearance_m"] = np.asarray([sample["minimum_clearance_m"] for sample in samples], float)
        if any(sample.get("scene") is not None for sample in samples):
            arrays["scene_json"] = np.asarray([json.dumps(sample["scene"]) for sample in samples])
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
    finally:
        staging.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
