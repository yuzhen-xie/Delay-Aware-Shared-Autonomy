"""Paired unit-level inference with sign permutations, bootstrap CIs and Holm control."""

import itertools

import numpy as np


def holm(p_values: list[float | None]) -> list[float | None]:
    valid = [i for i, p in enumerate(p_values) if p is not None]
    order = sorted(valid, key=lambda i: p_values[i])
    result, previous = [None] * len(p_values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, p_values[index] * (len(order) - rank)))
        result[index] = previous
    return result


def paired_inference(left, right, seed: int, bootstrap=5000, permutations=20000):
    left, right = np.asarray(left, float), np.asarray(right, float)
    selected = np.isfinite(left) & np.isfinite(right)
    difference = left[selected] - right[selected]
    n = len(difference)
    if n < 2:
        return {"n_units": n, "difference_mean": float(difference.mean()) if n else None,
                "ci95": None, "p_raw": None, "effect_dz": None,
                "test": "insufficient_independent_units"}
    rng = np.random.default_rng(seed)
    if n <= 18:
        signs = np.array(list(itertools.product((-1, 1), repeat=n)))
        distribution = np.abs(signs @ difference / n)
        p = np.mean(distribution >= abs(difference.mean()) - 1e-12)
        test = "exact_paired_sign_permutation"
    else:
        signs = rng.choice((-1, 1), size=(permutations, n))
        distribution = np.abs(signs @ difference / n)
        p = (1 + np.sum(distribution >= abs(difference.mean()) - 1e-12)) / (permutations + 1)
        test = "monte_carlo_paired_sign_permutation"
    boot = difference[rng.integers(0, n, size=(bootstrap, n))].mean(1)
    std = difference.std(ddof=1)
    return {"n_units": n, "difference_mean": float(difference.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "p_raw": float(p),
            "effect_dz": float(difference.mean() / std) if std > 1e-12 else None,
            "test": test, "bootstrap_resampling_unit": "paired_independent_unit"}


def aggregate(rows: list[dict], method: str, delay: float, metric: str) -> dict:
    result = {}
    for row in rows:
        if row["condition"] == method and row["delay_seconds"] == delay and row.get(metric) is not None:
            result.setdefault(row["unit"], []).append(row[metric])
    return {key: float(np.mean(values)) for key, values in result.items()}


def comparisons(rows: list[dict], reference: str, methods: list[str], metrics: list[str], config: dict):
    results = []
    for delay in sorted({r["delay_seconds"] for r in rows}):
        for method in methods:
            for metric in metrics:
                a, b = aggregate(rows, reference, delay, metric), aggregate(rows, method, delay, metric)
                units = sorted(a.keys() & b.keys())
                inference = paired_inference([a[u] for u in units], [b[u] for u in units],
                                             config["seed"], config["experiments"]["bootstrap_samples"],
                                             config["experiments"]["permutation_samples"])
                results.append({"reference": reference, "comparison": method,
                                "delay_seconds": delay, "metric": metric, "units": units, **inference})
    corrected = holm([r["p_raw"] for r in results])
    for row, p in zip(results, corrected):
        row["p_holm"] = p
        row["family"] = "all_planned_comparisons_in_this_analysis_file"
    return results
