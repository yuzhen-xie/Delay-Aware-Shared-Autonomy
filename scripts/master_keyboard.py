"""Windows keyboard bridge for bench checks: WASD/QE, hold Space, G toggles grip.

This bridge provides command inputs; it cannot render haptic forces. Device SDK
bridges should follow the same packet format and supply measured master velocity.
"""

import argparse
import json
import socket
import time

import numpy as np


def receive_displayed_feedback(client, last_observation_sequence=None):
    """Acknowledge only a feedback frame rendered by this diagnostic console."""
    current = last_observation_sequence
    while True:
        try:
            raw, _ = client.recvfrom(8192)
        except BlockingIOError:
            return current
        try:
            feedback = json.loads(raw.decode("utf-8")).get("feedback")
            if not feedback:
                continue
            sequence = feedback["observation_sequence"]
            position = np.asarray(feedback["position_m"], float)
            if (not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0
                    or (current is not None and sequence <= current)
                    or feedback["frame"] != "base" or position.shape != (3,)
                    or not np.isfinite(position).all()):
                continue
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            continue
        print(f"\rFeedback {sequence:06d} | delayed TCP [m]: "
              f"{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}", end="", flush=True)
        current = sequence


def main():
    import msvcrt
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=55001)
    args = parser.parse_args()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.setblocking(False)
    sequence, grip, enabled_until = 0, False, 0.0
    observation_sequence = None
    force, expiry = np.zeros(3), np.zeros(3)
    mapping = {"w": (0, 1), "s": (0, -1), "a": (1, -1), "d": (1, 1),
               "q": (2, 1), "e": (2, -1)}
    try:
        while True:
            now = time.monotonic()
            observation_sequence = receive_displayed_feedback(client, observation_sequence)
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "\x1b":
                    return
                if key == " ":
                    enabled_until = now + .15
                if key == "g":
                    grip = not grip
                if key in mapping:
                    axis, sign = mapping[key]
                    force[axis], expiry[axis] = 2 * sign, now + .15
            force[expiry < now] = 0
            packet = {"sequence": sequence, "device_timestamp": now, "frame": "base",
                      "force_N": force.tolist(), "velocity_m_s": [0, 0, 0],
                      "enable": now < enabled_until, "grip": grip,
                      "observation_sequence": observation_sequence}
            client.sendto(json.dumps(packet).encode(), ("127.0.0.1", args.port))
            sequence += 1
            time.sleep(.02)
    finally:
        client.close()


if __name__ == "__main__":
    main()
