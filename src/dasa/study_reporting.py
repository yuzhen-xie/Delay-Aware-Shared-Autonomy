import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dasa-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dasa.config import write_json


def _workload_figure(ratings, output, delay, lower, upper, name, score_label, statistics_file):
    selected = [row for row in ratings if np.isclose(row["delay_seconds"], delay)]
    scales = ["rtl_x", "mental", "physical", "temporal", "performance", "effort", "frustration"]
    labels = ["RTLX", "Mental demand", "Physical demand", "Temporal demand", "Performance burden",
              "Effort", "Frustration"]
    conditions = [condition for condition in ("C1", "C2", "C3", "C4")
                  if any(row["condition"] == condition for row in selected)]
    palette = {"C1": "#637381", "C2": "#d69b40", "C3": "#2478a8", "C4": "#459b79"}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), layout="constrained", sharey=True)
    margin = .03 * (upper - lower)
    for ax, scale, label in zip(axes.flat, scales, labels):
        values = [[row[scale] for row in selected if row["condition"] == condition] for condition in conditions]
        boxes = ax.boxplot(values, positions=np.arange(len(conditions)), widths=.5,
                           patch_artist=True, showfliers=False, medianprops={"color": "#182735"})
        for box, condition in zip(boxes["boxes"], conditions):
            box.set(facecolor=palette[condition], alpha=.3, edgecolor=palette[condition])
        for index, condition in enumerate(conditions):
            group = sorted([row for row in selected if row["condition"] == condition], key=lambda row: row["unit"])
            jitter = np.linspace(-.12, .12, len(group)) if len(group) > 1 else [0.]
            ax.scatter(index + np.asarray(jitter), [row[scale] for row in group], s=14,
                       color=palette[condition], alpha=.7, zorder=3)
        ax.set(title=label, xticks=np.arange(len(conditions)), xticklabels=conditions,
               ylim=(lower - margin, upper + margin))
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .95, "Observed participant ratings\nBoxes: median and IQR\nWhiskers: 1.5 IQR\nDots: individual participants\n\nHigher values = more workload\nAdjusted p values:\n" + statistics_file,
                       va="top", transform=axes.flat[-1].transAxes, fontsize=8.5, linespacing=1.5)
    fig.suptitle(f"Workload at {delay * 1000:.0f} ms feedback delay", fontsize=15, fontweight="semibold")
    for ax in axes[:, 0]:
        ax.set_ylabel(score_label)
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf", "svg"):
        target = folder / f"{name}.{suffix}"
        fig.savefig(target, dpi=200)
        files.append(str(target.relative_to(output)))
    plt.close(fig)
    return files


def render_workload(ratings, output: Path, delay=.8, specification=None, normalized=None):
    output = Path(output)
    if not isinstance(specification, dict):
        raise ValueError("A declared questionnaire scale and anchors are required")
    selected = [row for row in ratings if np.isclose(row["delay_seconds"], delay)]
    if not selected:
        write_json(output / "workload_figure_status.json", {
            "status": "not_generated", "reason": "no_actual_ratings_for_requested_delay", "delay_seconds": delay})
        return
    if any(row.get("provenance") not in {"actual_participant_measurement", "UR10_measured_teleoperation"}
           or row.get("score_scale") != "original_burden_oriented" for row in selected):
        raise ValueError("Workload figures require imported, explicitly oriented participant ratings")
    lower, upper = specification["scale_min"], specification["scale_max"]
    files = _workload_figure(ratings, output, delay, lower, upper, "fig8_workload",
                             f"Original burden-coded score ({lower:g}–{upper:g})", "tlx_original_statistics.csv")
    if normalized is not None:
        normalized_selected = [row for row in normalized if np.isclose(row["delay_seconds"], delay)]
        if any(row.get("provenance") not in {"actual_participant_measurement", "UR10_measured_teleoperation"}
               or row.get("score_scale") != "normalized_0_to_100_burden" for row in normalized_selected):
            raise ValueError("Normalized workload figures require the recorded scale-conversion contract")
        if normalized_selected:
            files += _workload_figure(normalized, output, delay, 0., 100., "fig8_workload_0_100",
                                      "Normalized workload score (0–100)", "tlx_statistics.csv")
    write_json(output / "workload_figure_status.json", {
        "status": "generated_from_imported_measurements", "delay_seconds": delay,
        "n_participants": len({row["unit"] for row in selected}), "n_ratings": len(selected),
        "ratings_per_condition": {condition: sum(row["condition"] == condition for row in selected)
                                 for condition in ("C1", "C2", "C3", "C4")},
        "original_statistics": "tlx_original_statistics.csv", "normalized_statistics": "tlx_statistics.csv",
        "performance_direction": "burden_oriented_after_explicit_contract",
        "literal_original_source": "tlx_original.csv", "original_burden_source": "tlx_original_burden.csv",
        "normalized_source": "tlx_normalized.csv", "questionnaire_contract": "questionnaire_contract.json",
        "analysis_artifacts": "analysis_artifacts.json", "files": files})
