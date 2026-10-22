import copy
import importlib.util
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import yaml

from dasa.data import Demonstration
from dasa.robotics.interfaces import FeedbackLedger, MasterUDP, RobotState
from dasa.robotics.session import ManualController, admittance_velocity, import_recording, run_workstation
from dasa.robotics.configuration import default_profile


def test_feedback_ledger_preserves_exact_sent_state_and_bounds_history():
    ledger = FeedbackLedger(maximum_frames=2)
    state, wrench = np.arange(6, dtype=float), np.arange(6, dtype=float) + 20
    first = ledger.issue(state, wrench, .1, 100.1, 100.9)
    state[:] = -1
    frame = ledger.acknowledge(first.observation_sequence, 100.92, .1)
    assert np.array_equal(frame.state, np.arange(6))
    assert frame.payload()["host_source_time_s"] == 100.1
    assert np.array_equal(frame.wrench, wrench)
    second = ledger.issue(np.zeros(6), np.ones(6), .2, 100.2, 101.)
    assert ledger.acknowledge(second.observation_sequence, 101.02, .1) is second
    assert ledger.acknowledge(first.observation_sequence, 101.02, .2) is None
    assert ledger.acknowledge(second.observation_sequence, 101.2, .1) is None
    assert ledger.acknowledge(None, 101.02, .1) is None
    assert ledger.acknowledge(400, 101.02, .1) is None
    ledger.issue(np.zeros(6), np.zeros(6), .3, 100.3, 101.1)
    assert len(ledger.frames) == 2 and first.observation_sequence not in ledger.frames


def test_udp_wire_carries_delayed_state_and_display_acknowledgement():
    master = MasterUDP(port=0, timeout=.2)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(.2)
    packet = {"sequence": 0, "frame": "base", "force_N": [1, 0, 0],
              "velocity_m_s": [0, 0, 0], "enable": True, "grip": False,
              "observation_sequence": None}
    try:
        target = master.socket.getsockname()
        client.sendto(json.dumps(packet).encode(), target)
        received, address = master.read()
        assert received["observation_sequence"] is None
        frame = FeedbackLedger().issue(np.arange(6), np.ones(6), 1., 5., 5.8)
        master.reflect_force([0, -1, 0], address, feedback=frame.payload())
        reply = json.loads(client.recvfrom(8192)[0])
        assert reply["feedback"]["position_m"] == [0., 1., 2.]
        assert reply["feedback"]["measured_wrench_N_Nm"] == [1.] * 6
        assert reply["reflected_force_N"] == [0., -1., 0.]
        packet.update(sequence=1, observation_sequence=reply["feedback"]["observation_sequence"])
        client.sendto(json.dumps(packet).encode(), target)
        assert master.read()[0]["observation_sequence"] == 0
    finally:
        client.close()
        master.close()


class _Clock:
    def __init__(self):
        self.value = 100.

    def monotonic(self):
        self.value += .001
        return self.value

    def sleep(self, duration):
        self.value += duration


@pytest.mark.parametrize("send_acknowledgement", [False, True])
@pytest.mark.parametrize("release_grip", [False, True])
@pytest.mark.parametrize("manual_bootstrap", [False, True])
def test_workstation_requires_real_feedback_acknowledgement_and_delays_wrench(
        config, tmp_path, monkeypatch, send_acknowledgement, release_grip, manual_bootstrap):
    from dasa.robotics import session
    hardware = default_profile(config)
    hardware["robot"]["host"] = "test.invalid"
    hardware["scene"] = _recording_and_annotation(config)[0]["scene"]
    hardware["gripper"]["type"] = "external"
    hardware.update(maximum_seconds=.24, feedback_delay_seconds=.04)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"source_frame": "camera_optical", "target_frame": "base",
                                       "transform": np.eye(4).tolist(), "rms_m": .001}))
    hardware["calibration_record"] = str(calibration)
    hardware_path = tmp_path / "hardware.yaml"
    hardware_path.write_text(yaml.safe_dump(hardware))
    clock = _Clock()
    monkeypatch.setattr(session.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(session.time, "sleep", clock.sleep)
    def forbidden_loader(*args):
        raise AssertionError("Manual acquisition must not access learned models")

    monkeypatch.setattr(session.TPHSMM, "load", forbidden_loader if manual_bootstrap else lambda *args: object())
    monkeypatch.setattr(session, "load_models", forbidden_loader if manual_bootstrap else lambda *args: (object(), None))
    monkeypatch.setattr(session, "jacobian", lambda *args: np.eye(6))

    class Controller:
        def __init__(self, *args):
            pass

        def decide(self, *args):
            return {"nominal_force": np.array([1., 0, 0])}

    class Robot:
        def __init__(self, *args):
            self.count = 0

        def state(self):
            self.count += 1
            return RobotState(self.count * .02, np.array([.16, -.1, .46]), np.zeros(3),
                              np.zeros(3), np.array([self.count, 0, 0, 0, 0, 0]),
                              np.zeros(6), 1)

        def speed(self, velocity, duration, enabled):
            return enabled

        def stop(self):
            pass

        def close(self):
            pass

    class Master:
        def __init__(self, *args):
            self.feedback, self.sequence = None, 0

        def read(self):
            acknowledgement = (self.feedback["observation_sequence"]
                               if send_acknowledgement and self.feedback else None)
            self.sequence += 1
            return {"sequence": self.sequence, "force_N": [1, 0, 0],
                    "velocity_m_s": [0, 0, 0], "enable": True,
                    "grip": release_grip and 4 <= self.sequence < 6,
                    "host_receive_monotonic": clock.monotonic(),
                    "observation_sequence": acknowledgement}, ("127.0.0.1", 1)

        def reflect_force(self, force, address, feedback=None):
            self.feedback = feedback

        def close(self):
            pass

    monkeypatch.setattr(session, "URRTDE", Robot)
    monkeypatch.setattr(session, "MasterUDP", Master)
    monkeypatch.setattr(session, "HierarchicalController", Controller)
    (tmp_path / "bootstrap_hsmm.npz").write_bytes(b"test fixture")
    refined = tmp_path / "refined_hsmm.npz"
    refined.write_bytes(b"refined fixture")
    output = tmp_path / "recording.json"
    run_workstation(config, hardware_path, None if manual_bootstrap else tmp_path, output,
                     execute=True, condition="C1" if manual_bootstrap else "C3",
                     skill_path=refined if release_grip and not manual_bootstrap else None)
    result = json.loads(output.read_text())
    samples = result["samples"]
    assert samples and samples[0]["observed_state"] is None
    assert all(not sample["executed"] for sample in samples if not sample["feedback_acknowledged"])
    assert "test.invalid" not in output.read_text()
    if manual_bootstrap:
        assert result["controller_mode"] == "manual_bootstrap" and result["model_hashes"] == {}
        assert "skill_file" not in result and "skill_sha256" not in result
        assert all("posterior" not in decision and "latent" not in decision
                   for decision in result["decisions"] if decision is not None)
    else:
        assert result["controller_mode"] == "trained_shared_control"
        assert result["skill_file"] == ("refined_hsmm.npz" if release_grip else "bootstrap_hsmm.npz")
    if send_acknowledgement:
        assert any(sample["executed"] for sample in samples)
        if release_grip:
            assert result["stop_reason"] == "gripper_release"
            assert samples[-1]["phase_after_action"] == 3
        for sample in samples:
            if sample["executed"] and sample["phase_after_action"] != 3:
                assert sample["feedback_age"] >= .04
                assert sample["delayed_feedback_wrench"][0] < sample["measured_wrench"][0]
                assert sample["reflected_force"][0] == -min(
                    sample["delayed_feedback_wrench"][0],
                    hardware["limits"]["maximum_reflection_force_N"])
    else:
        assert not any(sample["executed"] for sample in samples)
        assert all(sample["reflected_force"] == [0., 0., 0.] for sample in samples)


def _recording_and_annotation(config):
    scene = config["scene"]
    geometry = {"goals": scene["goals"], "start": scene["start"], "place": scene["place"],
                "centers": [obstacle["center"] for obstacle in scene["obstacles"]],
                "radii": [obstacle["radius"] for obstacle in scene["obstacles"]],
                "obstacle_ids": [obstacle["id"] for obstacle in scene["obstacles"]],
                "frame": "base", "provenance": "calibrated_scene"}
    samples = [{"time": i * .02 + 1., "position": [.2, -.1, .46], "velocity": [0, 0, 0],
                "raw_force": [1, 0, 0], "observed_state": [.2, -.1, .46, 0, 0, 0],
                "phase": 0, "executed": True, "feedback_acknowledged": True,
                "source_time": i * .02 + .2, "host_receive_time": i * .02 + 1.02,
                "feedback_age": .82, "acknowledged_observation_sequence": i,
                "master_sequence": i + 1} for i in range(4)]
    recording = {"schema_version": 2, "provenance": "UR10_workstation_measurement",
                 "execute": True, "condition": "C3", "scene": geometry, "samples": samples,
                 "timing": {"feedback_delay_seconds": .8, "robot_model": "UR10"}}
    annotation = {"episode_id": "actual-1", "operator_id": "P01", "intended_goal": 0,
                  "success": True, "collision": False, "placed": True,
                  "bounds_violation": False, "grasped_goal": 0,
                  "source_condition": "C3", "delay_seconds": .8}
    return recording, annotation


def test_hardware_import_keeps_feedback_audit_and_rejects_missing_ack(config, tmp_path):
    recording, annotation = _recording_and_annotation(config)
    source, annotations, output = [tmp_path / name for name in
                                  ["recording.json", "annotations.json", "demo.npz"]]
    source.write_text(json.dumps(recording))
    annotations.write_text(json.dumps(annotation))
    import_recording(source, annotations, output)
    demo = Demonstration.load(output)
    with np.load(output, allow_pickle=False) as archive:
        assert np.allclose(archive["observed_source_time"], [.2, .22, .24, .26])
        assert np.allclose(archive["feedback_age"], .82)
        assert np.array_equal(archive["acknowledged_observation_sequence"], np.arange(4))
        assert np.all(archive["executed"])
    assert "observed_source_time" in demo.metadata["feedback_timing_audit"]["timing_array_names"]
    assert len(demo.metadata["source_recording_sha256"]) == 64
    from dasa.recorded import _recorded_age
    measured_age, age_source = _recorded_age(output, demo, .8)
    assert np.allclose(measured_age, .82)
    assert "command_receive_time" in age_source
    failed_annotation = {**annotation, "success": False, "placed": False, "grasped_goal": None}
    annotations.write_text(json.dumps(failed_annotation))
    failed_output = tmp_path / "failed.npz"
    import_recording(source, annotations, failed_output)
    assert Demonstration.load(failed_output).metadata["grasped_goal"] is None
    annotations.write_text(json.dumps(annotation))
    bad = copy.deepcopy(recording)
    bad["samples"][1]["feedback_acknowledged"] = False
    source.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="acknowledged displayed feedback"):
        import_recording(source, annotations, tmp_path / "bad.npz")


def test_keyboard_echoes_only_rendered_feedback(capsys):
    path = Path(__file__).resolve().parents[1] / "scripts/master_keyboard.py"
    specification = importlib.util.spec_from_file_location("master_keyboard", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    frame = FeedbackLedger().issue(np.arange(6), np.ones(6), 1., 5., 5.8)

    class Socket:
        def __init__(self):
            self.payloads = [json.dumps({"feedback": frame.payload()}).encode()]

        def recvfrom(self, size):
            if not self.payloads:
                raise BlockingIOError()
            return self.payloads.pop(), ("127.0.0.1", 55001)

    assert module.receive_displayed_feedback(Socket()) == 0
    assert "delayed TCP" in capsys.readouterr().out


def test_udp_burst_selects_newest_valid_command_and_preserves_peer():
    master = MasterUDP(port=0, timeout=.2)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = {"frame": "base", "force_N": [1, 0, 0], "velocity_m_s": [0, 0, 0],
              "enable": True, "grip": False, "observation_sequence": 0}
    try:
        target = master.socket.getsockname()
        for sequence in (0, 2, 1, 4):
            client.sendto(json.dumps({**packet, "sequence": sequence}).encode(), target)
        client.sendto(b"invalid JSON", target)
        received, address = master.read()
        assert received["sequence"] == 4
        assert received["skipped_packets"] == 4 and received["invalid_packets"] == 1
        other.sendto(json.dumps({**packet, "sequence": 9000}).encode(), target)
        client.sendto(json.dumps({**packet, "sequence": 5}).encode(), target)
        received, current_address = master.read()
        assert received["sequence"] == 5 and received["foreign_peer_packets"] == 1
        assert current_address == address
        client.sendto(json.dumps({**packet, "sequence": 3}).encode(), target)
        with pytest.raises(ValueError, match="fresh master"):
            master.read()
    finally:
        client.close()
        other.close()
        master.close()


def test_manual_controller_has_filtered_bounded_human_force_only(config):
    selected = copy.deepcopy(config)
    selected["simulation"]["force_limit"] = .25
    controller = ManualController(selected)
    decision = controller.decide([10., 0, 0], np.zeros(6), np.zeros(6), 0., 0, None, 0.)
    assert 0 < decision["human_force"][0] < 10
    assert np.linalg.norm(decision["nominal_force"]) == pytest.approx(.25)
    assert set(decision) == {"human_force", "nominal_force", "alpha", "lambda"}
    assert decision["alpha"] == decision["lambda"] == 0


@pytest.mark.parametrize("condition,skill", [("C2", None), ("C3", None), ("C1", Path("skill.npz"))])
def test_nonmanual_model_free_acquisition_is_rejected_before_device_access(
        config, tmp_path, monkeypatch, condition, skill):
    from dasa.robotics import session

    def forbidden(*args):
        raise AssertionError("Invalid acquisition must fail before reading device configuration")

    monkeypatch.setattr(session, "load_hardware", forbidden)
    with pytest.raises(ValueError, match="manual C1"):
        run_workstation(config, tmp_path / "hardware.yaml", None,
                          tmp_path / "raw.json", condition=condition, skill_path=skill)


def test_admittance_midpoint_damping_work_balance(config):
    velocity = np.array([.10, -.08, .03])
    force = np.array([.5, .2, -.1])
    dt, mass, damping = .02, config["simulation"]["mass"], config["simulation"]["damping"]
    following = admittance_velocity(force, velocity, config, dt, maximum_speed=10.)
    midpoint = (following + velocity) / 2
    kinetic_change = .5 * mass * (following @ following - velocity @ velocity)
    assert kinetic_change == pytest.approx(dt * force @ midpoint - dt * damping * (midpoint @ midpoint))
    damped = admittance_velocity(np.zeros(3), velocity, config, dt, maximum_speed=10.)
    assert np.linalg.norm(damped) < np.linalg.norm(velocity)
    assert np.linalg.norm(admittance_velocity([100, 0, 0], np.zeros(3), config, dt)) == pytest.approx(.08)
