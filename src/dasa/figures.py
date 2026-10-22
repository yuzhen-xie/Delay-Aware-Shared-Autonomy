from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dasa-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dasa.config import manifest, write_json


COLORS = {
    "C1": "#697586", "C2": "#ce852b", "C3": "#2479b2", "C4": "#32916c",
    "C3_guarded": "#b44968", "delay_br_shared": "#8b63ad", "full": "#2479b2",
    "no_latent": "#ce852b", "no_delay": "#8b63ad", "br": "#697586",
    "memory": "#32916c", "dmp": "#b44968", "lstm": "#a99a34",
    "transformer": "#4a9da8",
}

LABELS = {"C3_guarded": "C3 (guarded)", "delay_br_shared": "Delayed BR sharing",
          "full": "Full model", "no_latent": "Without latent", "no_delay": "Without delay alignment",
          "br": "BR", "memory": "Memory", "dmp": "DMP", "lstm": "LSTM", "transformer": "Transformer"}

METRICS = {
    "ipa": "Correct and confident fraction (IPA)",
    "early_prediction_seconds": "Early prediction time (s)",
    "completion_seconds": "Successful completion time (s)",
    "success_rate": "Task success fraction",
    "stability_index": "Stability index",
    "safety_index": "Safety index",
    "minimum_clearance_m": "Minimum TCP surface clearance (m)",
    "jerk_rms": "TCP jerk RMS (m/s³)",
    "reference_error_m": "Mean reference deviation (m)",
}

MEASURED_PROVENANCE = {"actual_participant_measurement", "UR10_measured_teleoperation",
                       "UR10_measured_autonomous", "physical_robot_measurement",
                       "measured_workstation_acquisition"}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
        raise ValueError(f"Expected a JSON list of records: {path}")
    if any(row.get("provenance") not in MEASURED_PROVENANCE for row in result):
        raise ValueError("Figures accept only explicitly identified physical measurement records")
    return result


def export_csv(path: Path, rows: list[dict]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def eligible(row: dict) -> bool:

    return all(row.get(key) is not False for key in
               ("refinement_quota_met", "confirmatory_eligible", "quota_met"))


def unit_summary(rows: list[dict], metrics: list[str], group_keys: tuple[str, ...],
                 seed: int, bootstrap: int = 5000) -> tuple[list[dict], list[dict]]:






    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("provenance") not in MEASURED_PROVENANCE:
            raise ValueError("Figure summaries require physical measurement provenance")
        if any(key not in row for key in group_keys):
            raise ValueError(f"Missing figure grouping field: {group_keys}")
        if "unit" not in row and "repeat" not in row:
            raise ValueError("Each figure record requires an independent unit or training repeat")
        identity = row.get("unit", row.get("repeat"))
        if identity is None or not str(identity).strip():
            raise ValueError("Each physical record requires a nonempty independent-unit identifier")
        cells.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    units, summaries = [], []
    for cell, selected in sorted(cells.items(), key=lambda item: repr(item[0])):
        group = dict(zip(group_keys, cell))
        all_units = {str(row.get("unit", row.get("repeat"))) for row in selected}
        for metric in metrics:
            values: dict[str, list[float]] = {}
            for row in selected:
                value = row.get(metric)
                if value is None:
                    continue
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite {metric}; use an explicit null for a missing value")
                unit = str(row.get("unit", row.get("repeat")))
                values.setdefault(unit, []).append(value)
            estimates = []
            for unit, observed in sorted(values.items()):
                estimate = float(np.mean(observed))
                estimates.append(estimate)
                units.append({**group, "metric": metric, "unit": unit, "value": estimate,
                              "n_observed_trials": len(observed)})
            ci = None
            if len(estimates) >= 2:
                encoded = json.dumps([seed, cell, metric], sort_keys=True).encode()
                derived_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
                rng = np.random.default_rng(derived_seed)
                sample = np.asarray(estimates)
                distribution = sample[rng.integers(0, len(sample), (bootstrap, len(sample)))].mean(1)
                ci = np.quantile(distribution, [0.025, 0.975]).tolist()
            summaries.append({**group, "metric": metric,
                              "mean": float(np.mean(estimates)) if estimates else None,
                              "sd": float(np.std(estimates, ddof=1)) if len(estimates) >= 2 else None,
                              "median": float(np.median(estimates)) if estimates else None,
                              "ci95": ci, "n_units": len(estimates),
                              "n_units_missing": len(all_units) - len(estimates),
                              "n_trials": len(selected),
                              "n_observed_trials": sum(map(len, values.values())),
                              "ci_method": "percentile_independent_unit_bootstrap" if ci else
                                           "insufficient_independent_units",
                              "bootstrap_samples": bootstrap if ci else 0})
    return units, summaries


def _provenance(rows: list[dict]) -> tuple[list[str], str]:
    values = sorted({row["provenance"] for row in rows})
    if any(value not in MEASURED_PROVENANCE for value in values):
        raise ValueError("Only physical measurement provenance is supported")
    return values, "Physical human–robot measurements"


def _draw_panel(ax, units, summaries, metric: str, x_key: str, x_values: list):
    methods = sorted({row["condition"] for row in summaries})
    width = min(0.65 / max(len(methods), 1), 0.14)
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        color = COLORS.get(method, plt.get_cmap("tab10")(index % 10))
        xs, means = [], []
        for x_index, value in enumerate(x_values):
            records = [row for row in summaries if row["condition"] == method and
                       row[x_key] == value and row["metric"] == metric]
            if not records or records[0]["mean"] is None:
                continue
            record = records[0]
            x = x_index + offset
            observed = sorted(row["value"] for row in units if row["condition"] == method and
                              row[x_key] == value and row["metric"] == metric)
            jitter = np.linspace(-width * .25, width * .25, len(observed))
            ax.scatter(x + jitter, observed, color=color, alpha=.35, s=15, linewidths=0)
            if record["ci95"] is not None:
                low, high = record["ci95"]
                ax.plot([x, x], [low, high], color=color, linewidth=1.7)
                ax.plot([x - width * .18, x + width * .18], [low, low], color=color)
                ax.plot([x - width * .18, x + width * .18], [high, high], color=color)
            ax.plot(x, record["mean"], "o", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=.6)
            xs.append(x)
            means.append(record["mean"])
        if xs:
            ax.plot(xs, means, color=color, linewidth=1.2, label=LABELS.get(method, method))
    labels = [f"{value * 1000:g}" if x_key == "delay_seconds" else str(value).replace("_", " ")
              for value in x_values]
    ax.set_xticks(range(len(x_values)), labels)
    ax.set_xlabel("One-way feedback delay (ms)" if x_key == "delay_seconds" else
                  "Accepted additional demonstrations" if x_key == "budget" else "Evaluation scenario")
    ax.set_ylabel(METRICS[metric])
    counts = [row["n_units"] for row in summaries if row["metric"] == metric]
    observed_counts = [count for count in counts if count]
    if observed_counts:
        lower, upper = min(observed_counts), max(observed_counts)
        span = str(lower) if lower == upper else f"{lower}–{upper}"
        missing = sum(count == 0 for count in counts)
        ax.set_title(f"Independent units/cell: {span}; unavailable cells: {missing} (exact counts: CSV)",
                     loc="left", fontsize=7.5, color="#546274", pad=8)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def _save_figure(fig, path: Path) -> list[str]:
    files = []
    for suffix in ("png", "pdf", "svg"):
        target = path.with_suffix(f".{suffix}")
        fig.savefig(target, dpi=220, bbox_inches="tight")
        files.append(target.name)
    plt.close(fig)
    return files


def _metric_set(config: dict, rows: list[dict], output: Path, name: str,
                metrics: list[str], x_key: str, unit_kind: str) -> dict:
    group_keys = ("condition", x_key)
    unit_rows, summaries = unit_summary(rows, metrics, group_keys, config["seed"],
                                        config["experiments"]["bootstrap_samples"])
    export_csv(output / f"{name}_units.csv", unit_rows)
    export_csv(output / f"{name}_summary.csv", summaries)
    write_json(output / f"{name}_summary.json", summaries)
    available = [metric for metric in metrics if any(row["metric"] == metric and row["n_units"]
                                                   for row in summaries)]
    provenance, subtitle = _provenance(rows)
    details = {"status": "available" if available else "no_observed_metrics", "unit_kind": unit_kind,
               "provenance": provenance, "metrics": available,
               "n_input_rows": len(rows), "files": [f"{name}_units.csv", f"{name}_summary.csv",
                                                    f"{name}_summary.json"]}
    if not available:
        return details
    columns = min(2, len(available))
    lines = (len(available) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(6 * columns, 4 * lines), squeeze=False)
    x_values = sorted({row[x_key] for row in rows}, key=lambda x: (isinstance(x, str), x))
    for ax, metric in zip(axes.flat, available):
        _draw_panel(ax, unit_rows, summaries, metric, x_key, x_values)
    for ax in list(axes.flat)[len(available):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(.5, .005), fontsize=8, columnspacing=1.8, handlelength=2.3)
    fig.suptitle(f"{subtitle}\nUnit means, individual units and 95% bootstrap intervals",
                 fontsize=11, y=.995)
    fig.tight_layout(rect=(0, .09 if len(labels) > 4 else .065, 1, .985), h_pad=2.1, w_pad=2.3)
    details["files"].extend(_save_figure(fig, output / name))
    return details


def build(config: dict, root: Path, output: Path | None = None) -> dict:

    root = Path(root)
    output = Path(output) if output is not None else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    inputs, result = [], {}
    specifications = [
        ("figure5", "intent", ["ipa", "early_prediction_seconds"], "delay_seconds",
         "held-out participant"),
        ("figure6", "sharing", ["completion_seconds", "success_rate", "stability_index", "safety_index"],
         "delay_seconds", "registered participant"),
        ("learning", "refinement", ["success_rate", "reference_error_m"], "scenario",
         "independent skill-training repeat"),
    ]
    for name, stage, metrics, x_key, unit_kind in specifications:
        source = root / stage / "metrics.json"
        if stage == "refinement" and not source.exists() and (root / "metrics.json").exists():
            standalone = load_rows(root / "metrics.json")
            if standalone and all(row.get("stage") == "autonomous_evaluation" for row in standalone):
                source = root / "metrics.json"
        if stage == "sharing" and (root / "trial_metrics.json").exists():
            source = root / "trial_metrics.json"
        if not source.exists():
            result[name] = {"status": "missing_input", "required": str(source.relative_to(root))}
            continue
        inputs.append(source)
        rows = load_rows(source)
        export_csv(output / f"{name}_all_trials.csv", rows)
        selected = [row for row in rows if eligible(row)]
        excluded = [row for row in rows if not eligible(row)]
        if stage == "refinement":
            selected = [{**row, "unit": str(row.get("unit") if row.get("unit") is not None else row["repeat"])}
                        for row in selected]
        result[name] = _metric_set(config, selected, output, name, metrics, x_key, unit_kind)
        result[name]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        result[name]["source_file"] = str(source.relative_to(root))
        if stage == "sharing":
            result["figure6_secondary"] = _metric_set(
                config, selected, output, "figure6_secondary", ["minimum_clearance_m", "jerk_rms"],
                x_key, unit_kind)
            success_rates = root / "success_rates.csv"
            if success_rates.exists():
                inputs.append(success_rates)
                (output / "figure6_success_denominators.csv").write_bytes(success_rates.read_bytes())
                result[name]["success_denominators"] = "figure6_success_denominators.csv"
        result[name]["n_incomplete_budget_rows_excluded"] = len(excluded)
        if excluded:
            export_csv(output / f"{name}_incomplete_budget_diagnostics.csv", excluded)
        statistics = root / stage / "statistics.json"
        if stage == "refinement" and source == root / "metrics.json":
            statistics = root / "statistics.json"
        if stage == "sharing" and source == root / "trial_metrics.json":
            statistics = root / "statistics.json"
        if statistics.exists():
            inputs.append(statistics)
            content = json.loads(statistics.read_text(encoding="utf-8"))
            if not isinstance(content, list):
                raise ValueError("Figure statistics must contain a complete contrast table")
            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if any(row.get("metric_source_sha256") is not None
                   and row["metric_source_sha256"] != source_digest for row in content):
                raise ValueError("Figure statistics do not identify the plotted measurement table")
            write_json(output / f"{name}_statistics.json", content)
            if isinstance(content, list):
                export_csv(output / f"{name}_statistics.csv", content)
            result[name]["statistics"] = f"{name}_statistics.json"
            result[name]["statistics_csv"] = f"{name}_statistics.csv"
            result[name]["statistics_sha256"] = hashlib.sha256(statistics.read_bytes()).hexdigest()
            result[name]["statistics_linkage"] = "verified_metric_source_sha256" if content and all(
                row.get("metric_source_sha256") == source_digest for row in content) else "source_hash_not_recorded"
            result[name]["p_value_columns"] = ["p_raw", "p_holm"]
            result[name]["effect_columns"] = ["difference_mean", "effect_dz"]
            result[name]["interval_columns"] = ["ci95", "effect_dz_ci95"]
        else:
            result[name]["statistics"] = None
    generalization = root / "generalization/metrics.json"
    if not generalization.exists() and (root / "metrics.json").exists():
        standalone = load_rows(root / "metrics.json")
        if standalone and all(row.get("stage") == "autonomous_evaluation" for row in standalone):
            generalization = root / "metrics.json"
    if generalization.exists():
        inputs.append(generalization)
        rows = load_rows(generalization)
        export_csv(output / "generalization_all_trials.csv", rows)
        selected = [row for row in rows if eligible(row)]
        selected = [{**row, "unit": str(row.get("unit") if row.get("unit") is not None else row["repeat"]),
                     "budget": row.get("budget", row.get("refinement_budget"))} for row in selected]
        if any(row["budget"] is None for row in selected):
            raise ValueError("Generalization records require budget or refinement_budget")
        result["generalization"] = {}
        for scenario in sorted({row["scenario"] for row in selected}):
            subset = [row for row in selected if row["scenario"] == scenario]
            safe_name = "".join(c if c.isalnum() else "_" for c in scenario)
            name = f"generalization_{safe_name}"
            result["generalization"][scenario] = _metric_set(
                config, subset, output, name, ["success_rate", "reference_error_m"], "budget",
                "independent skill-training repeat")
        statistics = root / "generalization/statistics.json"
        if generalization == root / "metrics.json":
            statistics = root / "statistics.json"
        if statistics.exists():
            inputs.append(statistics)
            content = json.loads(statistics.read_text(encoding="utf-8"))
            if not isinstance(content, list):
                raise ValueError("Physical generalization statistics require a complete contrast table")
            generalization_digest = hashlib.sha256(generalization.read_bytes()).hexdigest()
            if any(row.get("metric_source_sha256") is not None
                   and row["metric_source_sha256"] != generalization_digest for row in content):
                raise ValueError("Generalization statistics differ from their recorded metric source")
            write_json(output / "generalization_statistics.json", content)
            if isinstance(content, list):
                export_csv(output / "generalization_statistics.csv", content)
            result["generalization_statistics"] = {
                "json": "generalization_statistics.json", "csv": "generalization_statistics.csv",
                "metric_source_sha256": generalization_digest,
                "statistics_sha256": hashlib.sha256(statistics.read_bytes()).hexdigest()}
    else:
        result["generalization"] = {"status": "missing_input", "required": "generalization/metrics.json"}

    workload = root / "workload_figure_status.json"
    if workload.exists():
        result["figure8"] = json.loads(workload.read_text(encoding="utf-8"))
        inputs.append(workload)
        for name in ("questionnaire_contract.json", "tlx_original_statistics.json", "tlx_statistics.json",
                     "tlx_original.csv", "tlx_original_burden.csv", "tlx_normalized.csv", "analysis_artifacts.json"):
            if (root / name).exists():
                inputs.append(root / name)
    else:
        result["figure8"] = {"status": "no_collected_workload_analysis",
                             "required": "study_analyze_collected_original_questionnaire_ratings"}
    inputs.extend(path for path in (root / "manifest.json", root / "run_status.json") if path.exists())
    for directory in (root / "models", root / "scenes"):
        if directory.exists():
            inputs.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
    record = manifest(config, "figures", list(dict.fromkeys(inputs)))
    record["data_provenance"] = "derived_physical_measurement_records"
    source_manifest = root / "manifest.json"
    if source_manifest.exists():
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        record["source_run"] = {key: source.get(key) for key in
                                ("git_commit", "git_dirty", "config_sha256", "data_provenance")}
        record["source_configuration"] = source.get("config")
        record["export_configuration_matches_source"] = source.get("config") == config
    else:
        record["source_run"] = None
        record["source_configuration"] = None
        record["export_configuration_matches_source"] = None
    record["figure_map"] = result
    record["interval_scope"] = "pointwise condition means; paired comparisons in statistics exports"
    record["files"] = sorted(path.name for path in output.iterdir() if path.is_file() and
                             path.name != "manifest.json")
    write_json(output / "manifest.json", record)
    return record
