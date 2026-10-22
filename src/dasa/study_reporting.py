import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dasa-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dasa.config import write_json


def render_workload(ratings, output: Path, delay=.8):
    selected = [r for r in ratings if np.isclose(r["delay_seconds"], delay)]
    if not selected:
        write_json(Path(output) / "workload_figure_status.json", {
            "status": "not_generated", "reason": "no_actual_ratings_for_requested_delay", "delay_seconds": delay})
        return
    if any(r.get("provenance") != "actual_participant_measurement" for r in selected):
        raise ValueError("Workload figure requires imported participant measurements")
    scales = ["rtl_x", "mental", "physical", "temporal", "performance", "effort", "frustration"]
    labels = ["RTLX", "Mental demand", "Physical demand", "Temporal demand", "Performance workload",
              "Effort", "Frustration"]
    conditions = [c for c in ["C1", "C2", "C3", "C4"] if any(r["condition"] == c for r in selected)]
    colors = ["#637381", "#d69b40", "#2478a8", "#459b79"][:len(conditions)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), layout="constrained", sharey=True)
    for ax, scale, label in zip(axes.flat, scales, labels):
        values = [[r[scale] for r in selected if r["condition"] == c] for c in conditions]
        boxes = ax.boxplot(values, positions=np.arange(len(conditions)), widths=.5,
                           patch_artist=True, showfliers=False, medianprops={"color": "#182735"})
        for box, color in zip(boxes["boxes"], colors):
            box.set(facecolor=color, alpha=.3, edgecolor=color)
        for i, condition in enumerate(conditions):
            group = sorted([r for r in selected if r["condition"] == condition], key=lambda r: r["unit"])
            jitter = np.linspace(-.12, .12, len(group)) if len(group) > 1 else [0.]
            ax.scatter(i + np.asarray(jitter), [r[scale] for r in group], s=14,
                       color=colors[i], alpha=.7, zorder=3)
        ax.set(title=label, xticks=np.arange(len(conditions)), xticklabels=conditions, ylim=(-3, 103))
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .95, "Observed participant ratings\nBoxes: median and IQR\nWhiskers: 1.5 IQR\nDots: individual participants\n\nHigher values = more workload\nAdjusted p values: tlx_statistics.csv",
                       va="top", transform=axes.flat[-1].transAxes, fontsize=9, linespacing=1.6)
    fig.suptitle(f"Workload at {delay * 1000:.0f} ms feedback delay", fontsize=15, fontweight="semibold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rating (0–100)")
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(folder / f"fig8_workload.{suffix}", dpi=200)
    plt.close(fig)
    write_json(Path(output) / "workload_figure_status.json", {
        "status": "generated_from_imported_measurements", "delay_seconds": delay,
        "n_participants": len({r["unit"] for r in selected}),
        "n_ratings": len(selected), "significance_source": "tlx_statistics.csv",
        "performance_direction": "higher_means_greater_workload", "source": "tlx.json"})
