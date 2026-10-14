"""Standalone figures, audit tables and result summaries."""

import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dasa-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dasa.config import write_json
from dasa.geometry import Scene
from dasa.simulation import Trial


COLORS = {"C1": "#7b7b7b", "C2": "#d28c28", "C3": "#287bb5", "C4": "#388d65",
          "delay_br_shared": "#8658a3", "C3_guarded": "#c44d58"}


def load_rows(path: Path):
    return json.loads(path.read_text()) if path.exists() else []


def summary(rows, metrics):
    result = []
    cells = sorted({(r["condition"], r["delay_seconds"], r.get("scenario", "")) for r in rows})
    for method, delay, scenario in cells:
        selected = [r for r in rows if (r["condition"], r["delay_seconds"], r.get("scenario", ""))
                    == (method, delay, scenario)]
        item = {"condition": method, "delay_seconds": delay, "scenario": scenario,
                "n_trials": len(selected), "n_units": len({r["unit"] for r in selected})}
        for metric in metrics:
            values = [r[metric] for r in selected if r.get(metric) is not None]
            units = {}
            for r in selected:
                if r.get(metric) is not None:
                    units.setdefault(r["unit"], []).append(r[metric])
            item[metric] = float(np.mean([np.mean(v) for v in units.values()])) if units else None
            item[f"{metric}_n_observed"] = len(values)
            item[f"{metric}_n_units"] = len(units)
        result.append(item)
    return result


def metric_figure(rows, metric, ylabel, output: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method in sorted({r["condition"] for r in rows}):
        delays, means, errors = [], [], []
        for delay in sorted({r["delay_seconds"] for r in rows}):
            units = {}
            for row in rows:
                if row["condition"] == method and row["delay_seconds"] == delay and row.get(metric) is not None:
                    units.setdefault(row["unit"], []).append(row[metric])
            values = [np.mean(v) for v in units.values()]
            if not values:
                continue
            delays.append(delay * 1000)
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)
        ax.errorbar(delays, means, yerr=errors, marker="o", capsize=3, label=method,
                    color=COLORS.get(method))
    ax.set(xlabel="Feedback delay (ms)", ylabel=ylabel,
           title="Synthetic task-space simulation; mean ± unit-level SE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def sphere(ax, center, radius, color, alpha=0.25):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 12))
    ax.plot_surface(center[0] + radius * np.cos(u) * np.sin(v),
                    center[1] + radius * np.sin(u) * np.sin(v),
                    center[2] + radius * np.cos(v), color=color, alpha=alpha, linewidth=0)


def trajectory_figure(trial: Trial, output: Path):
    a, scene = trial.arrays, Scene.from_dict(trial.metadata["scene"])
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for i, (center, radius) in enumerate(zip(scene.centers, scene.radii)):
        sphere(ax, center, radius, "gray")
        ax.text(*center, scene.obstacle_ids[i], fontsize=8)
    ax.scatter(*scene.goals.T, c="#dc752f", s=65, label="Candidate objects")
    for i, goal in enumerate(scene.goals):
        ax.text(*goal, f" G{i}")
    ax.scatter(*scene.place, c="#38966c", s=75, marker="s", label="Placement zone")
    ax.scatter(*scene.start, c="black", s=35, label="Initial pose")
    ax.plot(*a["position"].T, c="#287bb5", linewidth=2, label="Robot trajectory")
    ax.plot(*a["reference"].T, c="#be4f62", linewidth=1.2, alpha=0.8, label="Autonomous reference")
    ax.plot([], [], [], color="gray", linewidth=6, alpha=0.4, label="Obstacle spheres")
    ax.set(xlabel="Base x (m)", ylabel="Base y (m)", zlabel="Base z (m)",
           title=f"{trial.metadata['condition']}, {trial.metadata['delay_seconds'] * 1000:.0f} ms; "
                 f"{trial.metadata['termination']}")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def signal_figures(trial: Trial, directory: Path):
    a, scene = trial.arrays, Scene.from_dict(trial.metadata["scene"])
    for name, keys, ylabel in [
        ("authority", ["alpha", "lambda", "autonomy_confidence", "human_confidence"], "Coefficient"),
        ("energy", ["tank_energy", "energy_residual"], "Energy (J)")]:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for key in keys:
            ax.plot(a["time"], a[key], label=key)
        ax.set(xlabel="Time (s)", ylabel=ylabel)
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.2)
        fig.savefig(directory / f"{name}.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for j, name in enumerate(scene.obstacle_ids):
        ax.plot(a["time"], a["clearances"][:, j], label=f"{name}: center={np.round(scene.centers[j], 2)} m")
    ax.axhline(0.08, color="#c94d4d", linestyle="--", label="0.08 m warning clearance")
    ax.axhline(0, color="black", linewidth=0.8, label="Contact boundary")
    for event in trial.metadata["events"]:
        ax.axvline(event["time"], color="gray", alpha=0.5)
        ax.text(event["time"], ax.get_ylim()[1], event["event"], rotation=90, va="top")
    ax.set(xlabel="Time (s)", ylabel="End-effector surface clearance (m)")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(directory / "clearances.png", dpi=180)
    plt.close(fig)


def video(trial_path: Path, output: Path):
    from matplotlib.animation import FuncAnimation, PillowWriter
    trial = Trial.load(trial_path)
    a, scene = trial.arrays, Scene.from_dict(trial.metadata["scene"])
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for center, radius in zip(scene.centers, scene.radii):
        sphere(ax, center, radius, "gray")
    ax.scatter(*scene.goals.T, color="#dc752f", label="Objects")
    ax.scatter(*scene.place, color="#38966c", marker="s", label="Placement")
    line, = ax.plot([], [], [], color="#287bb5", label="Robot")
    ax.set(xlim=(0, 1), ylim=(-0.5, 0.5), zlim=(0, 0.8), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.legend()
    stride = max(1, len(a["time"]) // 120)
    def update(index):
        p = a["position"][:index + 1]
        line.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        ax.set_title(f"Simulation replay, t={a['time'][index]:.2f}s; {trial.metadata['condition']}")
        return (line,)
    animation = FuncAnimation(fig, update, frames=range(0, len(a["time"]), stride), interval=80)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=12))
    plt.close(fig)


def html_table(rows):
    if not rows:
        return "<p>No observations.</p>"
    keys = [k for k in rows[0] if not k.endswith("_n_observed")]
    def format_cell(value):
        return "missing" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(format_cell(r[k]))}</td>" for k in keys) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def report(config: dict, root: Path):
    directory = root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    sharing = load_rows(root / "sharing/metrics.json")
    intent = load_rows(root / "intent/metrics.json")
    refinement = load_rows(root / "refinement/metrics.json")
    figures = []
    if sharing:
        for key, label in (("success_rate", "Task success fraction"),
                           ("completion_seconds", "Successful completion time (s)"),
                           ("minimum_clearance_m", "Minimum surface clearance (m)")):
            metric_figure(sharing, key, label, directory / f"sharing_{key}.png")
            figures.append(f"sharing_{key}.png")
        traces = list((root / "sharing/trials").glob("*d800_C3.npz"))
        if traces:
            trial = Trial.load(traces[0])
            trajectory_figure(trial, directory / "trajectory.png")
            signal_figures(trial, directory)
            figures.extend(["trajectory.png", "authority.png", "clearances.png", "energy.png"])
    if intent:
        metric_figure(intent, "ipa", "Correct and confident fraction", directory / "intent_ipa.png")
        figures.append("intent_ipa.png")
    tables = {
        "sharing": summary(sharing, ["success_rate", "completion_seconds", "minimum_clearance_m"]),
        "intent": summary(intent, ["accuracy", "ipa", "nll"]),
        "refinement": summary(refinement, ["success_rate", "reference_error_m"])}
    write_json(directory / "summary.json", tables)
    budgets = load_rows(root / "refinement/budgets.json")
    outstanding = [b for b in budgets if not b["quota_met"]]
    limitation = (f"{len(outstanding)} refinement cells did not meet the demonstration quota. "
                  "Their evaluations are diagnostic and cannot support the specified fixed-budget comparison.")
    images = "".join(f"<figure><img src='{name}' alt='{name}'><figcaption>{name}</figcaption></figure>"
                     for name in figures)
    sections = "".join(f"<h2>{name}</h2>{html_table(rows)}" for name, rows in tables.items())
    body = f"""<!doctype html><html lang='en'><meta charset='utf-8'><title>Delayed shared control: run report</title>
    <style>body{{font:15px system-ui;max-width:1200px;margin:32px auto;padding:16px;color:#263244}}
    img{{max-width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #ddd}}
    .notice{{background:#fff3d6;padding:16px}}figure{{display:inline-block;width:48%;vertical-align:top;margin:1%}}</style>
    <h1>Delayed shared control: run report</h1>
    <p class='notice'>Synthetic task-space simulation. Physical device measurements and questionnaire responses
    are separate data sources. {limitation}</p>
    <p>Completion times condition on success; failed trials retain elapsed time and failure outcomes.
    Figures show unit-level means and standard errors, without untested significance markers.
    Raw and Holm-adjusted p values, paired confidence intervals, effects and sample sizes are in the
    corresponding statistics.json files. Statistical units are simulated operator profiles or independent
    skill-training repetitions, never time samples or repeated test trials.</p>{images}{sections}</html>"""
    (directory / "index.html").write_text(body, encoding="utf-8")
    print(f"Report written to {directory / 'index.html'}", flush=True)
