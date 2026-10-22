import json
import socket
import sys
import threading
import types

import numpy as np
import pytest

from dasa.geometry import Scene
from dasa.robotics.calibration import calibrate_file, load_calibration
from dasa.robotics.configuration import configure, load_hardware
from dasa.robotics.interfaces import RobotiqSocket, URRTDE
from dasa.robotics.perception import perceive_file


def _inputs(config, tmp_path):
    landmarks, calibration_path, snapshot_path = [tmp_path / name for name in
                                                 ["landmarks.json", "calibration.json", "snapshot.json"]]
    landmarks.write_text(json.dumps({"camera_points_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                     "base_points_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                     "camera_frame": "camera_optical"}))
    calibration = calibrate_file(landmarks, calibration_path)
    cfg = config["scene"]
    scene = Scene(np.asarray(cfg["goals"]), np.asarray(cfg["place"]), np.asarray(cfg["start"]),
                  np.asarray([item["center"] for item in cfg["obstacles"]]),
                  np.asarray([item["radius"] for item in cfg["obstacles"]]),
                  [item["id"] for item in cfg["obstacles"]], "calibrated_rgbd_geometry", "base")
    detections = [{"object_id": f"goal{i}", "role": "goal", "center_base_m": point.tolist(),
                   "radius_m": 0.} for i, point in enumerate(scene.goals)]
    detections += [{"object_id": name, "role": "obstacle", "center_base_m": center.tolist(),
                    "radius_m": float(radius)} for name, center, radius in
                   zip(scene.obstacle_ids, scene.centers, scene.radii)]
    snapshot = {"schema_version": 1, "provenance": "calibrated_rgbd_geometry",
                "target_frame": "base", "camera_frame": "camera_optical",
                "camera_to_base": calibration["transform"],
                "calibration_id": calibration["calibration_id"], "detections": detections,
                "scene": scene.to_dict()}
    snapshot_path.write_text(json.dumps(snapshot))
    return calibration_path, snapshot_path, calibration, snapshot


def test_hardware_configuration_links_native_perception_and_freezes_inputs(config, tmp_path):
    calibration, snapshot, _, _ = _inputs(config, tmp_path)
    output = tmp_path / "ur10.local.yaml"
    generated = configure(config, "192.0.2.10", calibration, snapshot, output)
    assert load_hardware(output) == generated
    assert generated["scene"]["provenance"] == "calibrated_rgbd_geometry"
    assert len(generated["calibration_sha256"]) == len(generated["snapshot_sha256"]) == 64
    snapshot.write_text(snapshot.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after"):
        load_hardware(output)


def test_hardware_inputs_reject_mismatched_calibration_and_invalid_limits(config, tmp_path):
    calibration, snapshot_path, _, snapshot = _inputs(config, tmp_path)
    snapshot["camera_to_base"][0][3] += .02
    snapshot_path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="same transform"):
        configure(config, "192.0.2.10", calibration, snapshot_path, tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="hostname"):
        configure(config, "invalid host", calibration, snapshot_path, tmp_path / "bad.yaml")


def test_calibration_rejects_nonrigid_transform_and_nonfinite_residual(config, tmp_path):
    calibration, _, record, _ = _inputs(config, tmp_path)
    record["transform"][0][0] = 2
    calibration.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="proper camera"):
        load_calibration(calibration)


def test_perception_reads_exact_calibration_record(tmp_path):
    landmarks, calibration = tmp_path / "landmarks.json", tmp_path / "calibration.json"
    landmarks.write_text(json.dumps({"camera_points_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                     "base_points_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                     "camera_frame": "camera_optical"}))
    actual = calibrate_file(landmarks, calibration)
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[3:14, 3:14] = [255, 0, 0]
    np.savez(tmp_path / "rgbd.npz", rgb=rgb, depth=np.full((20, 20), 1000))
    specification = {"intrinsics": {"width": 20, "height": 20, "fx": 50., "fy": 50.,
                                    "cx": 10., "cy": 10.},
                     "objects": [{"object_id": "goal0", "role": "goal",
                                  "rgb_lower": [255, 0, 0], "rgb_upper": [255, 0, 0]}],
                     "calibration_record": "calibration.json", "timestamp_s": 1., "sequence": 1}
    (tmp_path / "spec.json").write_text(json.dumps(specification))
    result = perceive_file(tmp_path / "rgbd.npz", tmp_path / "spec.json", tmp_path / "out.json")
    assert result["calibration_id"] == actual["calibration_id"]
    assert len(result["calibration_record_sha256"]) == 64


def test_rtde_connection_failure_closes_receive_and_commands_record_actual_status(monkeypatch):
    events = []

    class Receive:
        def __init__(self, host):
            pass

        def disconnect(self):
            events.append("receive_disconnected")

        def getActualTCPPose(self):
            return [0.] * 6

        getActualTCPSpeed = getActualTCPPose
        getActualTCPForce = getActualTCPPose
        getActualQ = getActualTCPPose

        def getTimestamp(self):
            return 1.

        def getSafetyMode(self):
            return 1

    def failed_control(host):
        raise ConnectionError("test failure")

    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=Receive))
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=failed_control))
    with pytest.raises(ConnectionError):
        URRTDE("test.invalid", execute=True)
    assert events == ["receive_disconnected"]

    class Control:
        result = False

        def __init__(self, host):
            pass

        def speedL(self, velocity, acceleration, duration):
            assert len(velocity) == 6 and velocity[3:] == [0., 0., 0.]
            return self.result

        def speedStop(self, acceleration):
            events.append("stop")

        def stopScript(self):
            events.append("stop_script")

        def disconnect(self):
            events.append("control_disconnected")

    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=Control))
    robot = URRTDE("test.invalid", execute=True)
    assert robot.speed([1, 0, 0], .02, True) is False
    assert robot.commands[-1]["executed"] is False
    robot.control.result = True
    assert robot.speed([1, 0, 0], .02, True) is True
    assert np.linalg.norm(robot.commands[-1]["velocity"]) <= .08
    robot.close()
    robot.close()
    assert events.count("control_disconnected") == 1


def test_robotiq_ascii_activation_commands_and_status():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(.05)
    stopped = threading.Event()
    state = {"ACT": 0, "STA": 0, "FLT": 0, "POS": 0, "OBJ": 3}
    commands = []

    def serve():
        while not stopped.is_set():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            with client:
                raw = b""
                while b"\n" not in raw:
                    raw += client.recv(1024)
                command = raw.decode().strip()
                commands.append(command)
                pieces = command.split()
                if pieces[0] == "GET":
                    client.sendall(f"{pieces[1]} {state[pieces[1]]}\n".encode())
                else:
                    for name, value in zip(pieces[1::2], pieces[2::2]):
                        state[name] = int(value)
                    state["STA"] = 3 if state["ACT"] else 0
                    client.sendall(b"ack")

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        gripper = RobotiqSocket("127.0.0.1", server.getsockname()[1], True, speed=90, force=70)
        assert gripper.activate()["executed"]
        assert gripper.command(True)["executed"]
        assert gripper.wait_until_stopped()["position"] == 255
        assert "SET ACT 0" in commands and "SET ACT 1" in commands
        assert "SET POS 255 SPE 90 FOR 70 GTO 1" in commands
        assert not gripper.activate()["executed"]
    finally:
        stopped.set()
        worker.join(timeout=1)
        server.close()
