import ctypes
import json

import numpy as np
import pytest

from dasa.robotics import master


class Function:
    def __init__(self, operation):
        self.operation = operation

    def __call__(self, *arguments):
        return self.operation(*arguments)


class VendorLibrary:
    def __init__(self):
        self.events = []
        self.position_status = 0
        self.dhdOpenID = Function(lambda identifier: self.record("open", identifier, result=identifier))
        self.dhdGetPosition = Function(lambda *arguments: self.vector("position", [0.01, .02, .03], arguments))
        self.dhdGetLinearVelocity = Function(lambda *arguments: self.vector("velocity", [.1, .2, .3], arguments))
        self.dhdGetButton = Function(lambda button, identifier: self.record("button", button, identifier,
                                                                           result=button == 0))
        self.dhdSetForce = Function(lambda *arguments: self.record("force", *arguments))
        self.dhdEnableForce = Function(lambda *arguments: self.record("enable", *arguments))
        self.dhdClose = Function(lambda *arguments: self.record("close", *arguments))
        self.dhdErrorGetLastStr = Function(lambda: b"SDK test status")

    def record(self, name, *arguments, result=0):
        self.events.append((name, *arguments))
        return result

    def vector(self, name, values, arguments):
        self.events.append((name, arguments[-1]))
        for pointer, value in zip(arguments[:3], values):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double))[0] = value
        return self.position_status if name == "position" else 0


def vendor(monkeypatch, tmp_path):
    library = VendorLibrary()
    path = tmp_path / "dhd-test-library.so"
    path.write_bytes(b"")
    monkeypatch.setattr(master.ctypes, "CDLL", lambda actual: library)
    return path, library


def test_dhd_signatures_measured_vectors_and_zero_force_cleanup(monkeypatch, tmp_path):
    path, library = vendor(monkeypatch, tmp_path)
    with master.DHDDevice(path, device_index=1) as device:
        assert np.allclose(device.getPosition(), [.01, .02, .03])
        assert np.allclose(device.getLinearVelocity(), [.1, .2, .3])
        assert device.getButton(0)
        assert not device.getButton(1)
        device.enableForce(True)
        device.setForce(.4, -.2, .1)
    assert library.dhdGetPosition.argtypes[-1] is ctypes.c_byte
    assert library.dhdSetForce.argtypes[:3] == [ctypes.c_double] * 3
    assert library.dhdEnableForce.argtypes[0] is ctypes.c_ubyte
    assert library.events[0] == ("open", 1)
    assert library.events[-3:] == [("force", 0.0, 0.0, 0.0, 1), ("enable", 0, 1), ("close", 1)]
    device.close()


@pytest.mark.parametrize("status", [-1, 1, 2])
def test_nonzero_sdk_status_disarms_before_raising(monkeypatch, tmp_path, status):
    path, library = vendor(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match=f"returned {status}"):
        with master.DHDDevice(path) as device:
            device.enableForce(True)
            library.position_status = status
            device.getPosition()
    assert library.events[-3:] == [("force", 0.0, 0.0, 0.0, 0), ("enable", 0, 0), ("close", 0)]


def test_invalid_master_rotation_rejected_before_device_or_socket(monkeypatch, tmp_path):
    path = tmp_path / "rotation.json"
    path.write_text(json.dumps({"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}))
    monkeypatch.setattr(master.socket, "socket", lambda *arguments: pytest.fail("Socket must not open"))
    with pytest.raises(ValueError, match="proper device-to-base rotation"):
        master.run_master("not-installed", path, maximum_samples=1)


def feedback(reply_to, observation):
    return json.dumps({"sequence": reply_to, "reflected_force_N": [8, 0, 0],
                       "feedback": {"observation_sequence": observation, "frame": "base",
                                    "source_time_s": 0.0, "host_source_time_s": 2.0,
                                    "host_send_time_s": 3.0, "position_m": [.4, 0, .3],
                                    "velocity_m_s": [0, 0, 0],
                                    "measured_wrench_N_Nm": [0] * 6}}).encode()


def bridge_fixture(monkeypatch, tmp_path):
    events, clock = [], [0.0]

    class Socket:
        def __init__(self):
            self.commands, self.delivered, self.pending = [], set(), []
            self.closed = False

        def connect(self, peer):
            self.peer = peer

        def setblocking(self, blocking):
            assert not blocking

        def recv(self, size):
            count = len(self.commands)
            if count not in self.delivered:
                self.delivered.add(count)
                if count == 1:
                    self.pending.extend([feedback(99, 0), feedback(0, 0)])
                elif count == 4:
                    self.pending.append(feedback(0, 1))
                elif count == 5:
                    self.pending.append(feedback(4, 1))
            if not self.pending:
                raise BlockingIOError
            return self.pending.pop(0)

        def send(self, raw):
            packet = json.loads(raw)
            self.commands.append(packet)
            events.append(("send", packet["sequence"], packet["observation_sequence"]))
            return len(raw)

        def close(self):
            self.closed = True

    class Device:
        def __init__(self):
            self.index, self.forces, self.closed = -1, [], False

        def open(self):
            return self

        def enableForce(self, enabled):
            pass

        def getPosition(self):
            self.index += 1
            return np.array([0, .01, .05, .08, .1, .12, .15])[self.index] * np.array([1., 0, 0])

        def getLinearVelocity(self):
            return np.array([.1, .2, .3])

        def getButton(self, button):
            return self.index != 3 if button == 0 else True

        def setForce(self, force):
            self.forces.append(np.asarray(force))

        def close(self):
            self.setForce(np.zeros(3))
            self.closed = True

    client, device = Socket(), Device()
    monkeypatch.setattr(master.socket, "socket", lambda *arguments: client)
    monkeypatch.setattr(master, "DHDDevice", lambda *arguments: device)
    monkeypatch.setattr(master.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(master.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    path = tmp_path / "rotation.json"
    path.write_text(json.dumps({"rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                               "source_frame": "device", "target_frame": "base"}))
    return path, client, device, events


def test_bridge_acknowledges_display_and_maps_virtual_commands_with_watchdog(monkeypatch, tmp_path):
    path, client, device, events = bridge_fixture(monkeypatch, tmp_path)
    result = master.run_master("test-only", path, frequency=500, stiffness=100, maxcommand=3,
                               maxfeedback=2, feedback_timeout=.005, maximum_samples=7,
                               display_callback=lambda frame: events.append(("display", frame["observation_sequence"])))
    packets = client.commands[:-1]
    assert result["samples_sent"] == 7
    assert result["feedback_packets_rejected"] == 2
    assert result["feedback_frames_displayed"] == 2
    assert events.index(("display", 0)) < events.index(("send", 1, 0))
    assert [packet["enable"] for packet in packets] == [False, True, True, False, False, True, True]
    assert np.allclose(packets[1]["force_N"], 0)
    assert np.allclose(packets[2]["force_N"], [0, 3, 0])
    assert np.allclose(packets[5]["force_N"], 0)
    assert np.allclose(packets[2]["velocity_m_s"], [-.2, .1, .3])
    assert np.allclose(device.forces[1], [0, -2, 0])
    assert np.allclose(device.forces[4], 0)
    assert packets[4]["observation_sequence"] is None
    assert packets[2]["command_source"] == "virtual_spring_from_measured_master_displacement"
    assert not client.commands[-1]["enable"]
    assert np.allclose(device.forces[-1], 0)
    assert device.closed and client.closed


def test_display_failure_closes_device_and_does_not_acknowledge(monkeypatch, tmp_path):
    path, client, device, events = bridge_fixture(monkeypatch, tmp_path)

    def failed_display(frame):
        raise RuntimeError("Display did not complete")

    with pytest.raises(RuntimeError, match="Display did not complete"):
        master.run_master("test-only", path, maximum_samples=3, display_callback=failed_display)
    assert all(packet["observation_sequence"] is None for packet in client.commands)
    assert np.allclose(device.forces[-1], 0)
    assert device.closed and client.closed
