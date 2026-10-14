"""UR10/UR10e nominal DH kinematics; calibration offsets are not inferred."""

import numpy as np


PARAMETERS = {
    "UR10": {"a": [0, -.612, -.5723, 0, 0, 0],
             "d": [.1273, 0, 0, .163941, .1157, .0922]},
    "UR10e": {"a": [0, -.6127, -.57155, 0, 0, 0],
              "d": [.1807, 0, 0, .17415, .11985, .11655]},
}


def dh(a, d, alpha, theta):
    c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s * ca, s * sa, a * c], [s, c * ca, -c * sa, a * s],
                     [0, sa, ca, d], [0, 0, 0, 1]])


def transforms(joints, model="UR10"):
    if len(joints) != 6 or model not in PARAMETERS:
        raise ValueError("Expected six joints and an explicit UR10/UR10e model")
    parameters = PARAMETERS[model]
    alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    t, result = np.eye(4), [np.eye(4)]
    for a, d, angle, joint in zip(parameters["a"], parameters["d"], alpha, joints):
        t = t @ dh(a, d, angle, joint)
        result.append(t.copy())
    return result


def forward(joints, model="UR10"):
    return transforms(joints, model)[-1]


def jacobian(joints, model="UR10"):
    frames = transforms(joints, model)
    endpoint = frames[-1][:3, 3]
    return np.stack([np.r_[np.cross(frame[:3, 2], endpoint - frame[:3, 3]), frame[:3, 2]]
                     for frame in frames[:-1]], axis=1)


def inverse(position, initial, model="UR10", iterations=200, tolerance=1e-5):
    q = np.asarray(initial, float).copy()
    for _ in range(iterations):
        error = np.asarray(position) - forward(q, model)[:3, 3]
        if np.linalg.norm(error) < tolerance:
            return q
        j = jacobian(q, model)[:3]
        q += j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), error)
        q = (q + np.pi) % (2 * np.pi) - np.pi
    raise ValueError("Position inverse kinematics did not converge from the provided seed")
