"""Repeatable delayed-energy bench for coordinated force/velocity regulation."""

from pathlib import Path

import numpy as np

from dasa.config import write_json
from dasa.control.ports import MultiportPassivity, Port
from dasa.delay import DelayBuffer


def run(config: dict, output: Path):
    dt = config["simulation"]["control_dt"]
    count = config["experiments"].get("port_bench_steps", 20000)
    observer = MultiportPassivity(.1, config["passivity"]["maximum_energy"])
    incoming = DelayBuffer(np.zeros(1), config["simulation"]["feedback_delay"] + 1)
    last_source = -1.0
    records = {k: [] for k in ["time", "energy", "input_energy", "outbound_work",
                               "removed_work", "residual", "effort", "flow"]}
    for sample in range(count):
        time = sample * dt
        incoming.push(time, np.array([dt * (.02 + .01 * np.sin(time))]))
        packet = incoming.observe(time, config["simulation"]["feedback_delay"])
        credit = float(packet.state[0]) if packet.source_time > last_source else 0.0
        last_source = packet.source_time
        alpha = .5 + .45 * np.sin(1.2 * time)
        blend = .5 + .45 * np.cos(.7 * time)
        effort = np.array([[2 * np.sin(time), 1, .2],
                           [4, 3 * np.cos(time), 1], [2, 1, 2 * np.sin(.5 * time)]])
        flow = np.array([[.02, -.01 * np.cos(time), .01],
                         [.04 * np.cos(.8 * time), .03, .02], [.05, .02, .01]])
        ports = [Port(effort[i], flow[i], "velocity" if i == 2 else "force")
                 for i in range(3)]
        nominal = dt * np.sum(effort * flow)
        corrected, removal = observer.regulate(ports, dt, credit,
                                               np.array([1 - alpha, alpha, 1 - blend]))
        delivered = dt * sum(p.effort @ p.flow for p in corrected)
        row = {"time": time, "energy": observer.energy, "input_energy": credit,
               "outbound_work": delivered, "removed_work": removal,
               "residual": observer.residual(),
               "effort": np.array([p.effort for p in corrected]),
               "flow": np.array([p.flow for p in corrected])}
        if not np.isclose(nominal - delivered, removal.sum(), atol=1e-10):
            raise ArithmeticError("Port damping does not match removed work")
        for key, value in row.items():
            records[key].append(value)
    arrays = {k: np.asarray(v) for k, v in records.items()}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "signals.npz", **arrays)
    write_json(output / "metrics.json", {"provenance": "synthetic_sampled_port_bench",
        "steps": count, "input_energy_j": arrays["input_energy"].sum(),
        "minimum_storage_j": arrays["energy"].min(),
        "maximum_budget_residual_j": np.abs(arrays["residual"]).max(),
        "removed_work_j": arrays["removed_work"].sum(),
        "duplicate_energy_packets_credited": 0})
