"""Windows keyboard bridge for bench checks: WASD/QE, hold Space, G toggles grip.

This bridge provides command inputs; it cannot render haptic forces. Device SDK
bridges should follow the same packet format and supply measured master velocity.
"""

import argparse
import json
import socket
import time

import numpy as np


def main():
    import msvcrt
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=55001)
    args = parser.parse_args()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence, grip, enabled_until = 0, False, 0.0
    force, expiry = np.zeros(3), np.zeros(3)
    mapping = {"w": (0, 1), "s": (0, -1), "a": (1, -1), "d": (1, 1),
               "q": (2, 1), "e": (2, -1)}
    try:
        while True:
            now = time.monotonic()
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
                      "enable": now < enabled_until, "grip": grip}
            client.sendto(json.dumps(packet).encode(), ("127.0.0.1", args.port))
            sequence += 1
            time.sleep(.02)
    finally:
        client.close()


if __name__ == "__main__":
    main()
