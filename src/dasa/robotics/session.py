"""Workstation loop and offline replay of the same nominal hierarchy.

The workstation performs translational admittance at a measured host cadence.
Its velocity interface is not the simulated force plant. Hardware energy traces
are observations and must not be reported as the simulation's passivity theorem.
"""

from __future__ import annotations

import copy
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
from dasa.robotics.interfaces import MasterUDP, RobotiqSocket, URRTDE
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
    observations = DelayBuffer(np.r_[initial.position, initial.velocity],
                               hardware["feedback_delay_seconds"] + 1)
    started, phase, grasped, grip_previous = time.monotonic(), 0, None, False
    samples, commands, reflection_work, stop_reason = [], [], 0.0, "duration"
    try:
        while time.monotonic() - started < hardware["maximum_seconds"]:
            tick = time.monotonic()
            state = robot.state()
            elapsed = tick - started
            observations.push(elapsed, np.r_[state.position, state.velocity])
            observed = observations.observe(elapsed, hardware["feedback_delay_seconds"])
            try:
                packet, address = master.read()
            except socket.timeout:
                robot.stop()
                stop_reason = "master_watchdog"
                break
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
            if packet["grip"] != grip_previous:
                if gripper:
                    gripper.command(packet["grip"])
                grip_previous = packet["grip"]
                if grip_previous:
                    grasped = int(np.argmin(np.linalg.norm(scene.goals - state.position, axis=1)))
                    phase = 1
                elif phase >= 1:
                    phase = 3
            if phase == 1 and state.position[2] >= c["simulation"]["lift_height"] - .05:
                phase = 2
            decision = controller.decide(raw, observed.state, np.r_[state.position, state.velocity],
                                         observed.age, phase, grasped, 0.0)
            desired = clamp_norm(state.velocity + hardware["loop_seconds"] * decision["nominal_force"] /
                                  c["simulation"]["mass"], hardware["limits"]["maximum_speed_m_s"])
            # Stop before entry into an inflated obstacle; this is a workstation interlock.
            next_position = state.position + desired * hardware["limits"]["watchdog_seconds"]
            clearance = scene.swept_clearance(state.position, next_position,
                                              c["simulation"]["robot_radius"] + hardware["limits"]["obstacle_margin_m"])
            enabled = packet["enable"] and clearance > 0
            robot.speed(desired, hardware["loop_seconds"], enabled)
            reflected = clamp_norm(-state.wrench[:3], hardware["limits"]["maximum_reflection_force_N"])
            if not enabled or not execute:
                reflected = np.zeros(3)
            master.reflect_force(reflected, address)
            reflection_work += hardware["loop_seconds"] * float(reflected @ np.asarray(packet["velocity_m_s"]))
            samples.append({"time": elapsed, "robot_timestamp": state.timestamp,
                            "position": state.position, "velocity": state.velocity,
                            "raw_force": raw, "observed_state": observed.state,
                            "source_time": observed.source_time, "phase": phase,
                            "measured_wrench": state.wrench,
                            "master_velocity": packet["velocity_m_s"],
                            "cumulative_reflection_work_J": reflection_work,
                            "actual_cadence_seconds": time.monotonic() - tick,
                            "enabled": enabled, "executed": execute})
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
        write_json(output, {"schema_version": 1, "provenance": "UR10_workstation_measurement",
                            "execute": execute, "stop_reason": stop_reason,
                            "scene": scene.to_dict(), "samples": samples, "decisions": commands,
                            "outcome": "requires_independent_task_annotation",
                            "passivity_scope": "measured trace only; no hardware passivity assertion"})


def import_recording(recording: Path, annotations: Path, output: Path):
    data = json.loads(recording.read_text())
    metadata = json.loads(annotations.read_text())
    for field in ("episode_id", "operator_id", "intended_goal", "success", "collision", "placed",
                  "bounds_violation", "grasped_goal", "approval_id"):
        if metadata.get(field) is None:
            raise ValueError(f"Missing independent task annotation: {field}")
    samples = data["samples"]
    if len(samples) < 3 or not data["execute"]:
        raise ValueError("Only executed, nonempty measured recordings may become hardware demonstrations")
    for key in ("success", "collision", "placed", "bounds_violation"):
        if not isinstance(metadata[key], bool):
            raise ValueError(f"Annotation {key} must be boolean")
    metadata["provenance"] = "UR10_measured_teleoperation"
    metadata["source_recording"] = str(recording)
    demo = Demonstration(np.asarray([s["time"] for s in samples]),
                          *[np.asarray([s[k] for s in samples]) for k in
                            ("position", "velocity", "raw_force", "observed_state", "phase")],
                          Scene.from_dict(data["scene"]), metadata.pop("intended_goal"), metadata)
    demo.save(output)
