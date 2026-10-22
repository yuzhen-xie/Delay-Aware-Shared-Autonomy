import hashlib

import numpy as np


def holm(p_values: list[float | None], family_size: int | None = None) -> list[float | None]:

    valid = [i for i, p in enumerate(p_values) if p is not None]
    if any(not np.isfinite(p_values[i]) or not 0 <= p_values[i] <= 1 for i in valid):
        raise ValueError("p values must be finite probabilities")
    size = len(valid) if family_size is None else family_size
    if size < len(valid):
        raise ValueError("Family size cannot be smaller than the number of tested hypotheses")
    order = sorted(valid, key=lambda i: p_values[i])
    result, previous = [None] * len(p_values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, p_values[index] * (size - rank)))
        result[index] = previous
    return result


def paired_inference(left, right, seed: int, bootstrap=5000, permutations=20000):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Paired observations must be one-dimensional arrays of equal length")
    if bootstrap < 1 or permutations < 1:
        raise ValueError("Positive resampling counts are required")
    selected = np.isfinite(left) & np.isfinite(right)
    difference = left[selected] - right[selected]
    n = len(difference)
    common = {
        "n_units": n, "n_pairs_omitted_nonfinite": int((~selected).sum()),
        "difference_mean": float(difference.mean()) if n else None,
        "difference_direction": "reference_minus_comparison",
        "alternative": "two_sided", "null_hypothesis": "mean_paired_difference_zero",
        "assumptions": ["independent analysis units", "paired differences jointly sign-exchangeable under the null"],
        "seed": int(seed), "confidence_level": 0.95,
        "ci_method": "percentile_bootstrap_of_paired_unit_differences",
        "ci_scope": "individual_contrast_not_simultaneous", "bootstrap_samples": int(bootstrap),
        "effect_definition": "Cohen_dz_mean_difference_divided_by_sample_SD_of_paired_differences",
    }
    if n < 2:
        return {**common, "ci95": None, "p_raw": None, "effect_dz": None,
                "effect_dz_ci95": None, "test": "insufficient_independent_units",
                "permutations_used": 0, "monte_carlo_standard_error": None}
    rng = np.random.default_rng(seed)
    observed = abs(difference.mean())
    tolerance = 16 * np.finfo(float).eps * max(observed, float(np.max(np.abs(difference))))
    if n <= 20:
        permutations_used, exceedances = 1 << n, 0
        shifts = np.arange(n, dtype=np.uint64)
        for start in range(0, permutations_used, 4096):
            codes = np.arange(start, min(start + 4096, permutations_used), dtype=np.uint64)
            signs = 2 * ((codes[:, None] >> shifts) & 1).astype(np.int8) - 1
            distribution = np.abs(signs @ difference / n)
            exceedances += int(np.count_nonzero(distribution >= observed - tolerance))
        p = exceedances / permutations_used
        test = "exact_paired_sign_permutation"
        mc_se = None
    else:
        signs = rng.choice((-1, 1), size=(permutations, n))
        distribution = np.abs(signs @ difference / n)
        p = (1 + np.sum(distribution >= observed - tolerance)) / (permutations + 1)
        test = "monte_carlo_paired_sign_permutation"
        mc_se = float(np.sqrt(p * (1 - p) / (permutations + 1)))
        permutations_used = permutations
    sampled = difference[rng.integers(0, n, size=(bootstrap, n))]
    boot = sampled.mean(1)
    std = difference.std(ddof=1)
    boot_std = sampled.std(1, ddof=1)
    finite_effects = boot[boot_std > 1e-12] / boot_std[boot_std > 1e-12]
    effect_ci = np.quantile(finite_effects, [0.025, 0.975]).tolist() \
        if std > 1e-12 and len(finite_effects) >= bootstrap * .95 else None
    return {**common, "ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "p_raw": float(p),
            "effect_dz": float(difference.mean() / std) if std > 1e-12 else None,
            "effect_dz_ci95": effect_ci, "test": test,
            "effect_bootstrap_degenerate_samples": int(np.sum(boot_std <= 1e-12)),
            "permutations_used": permutations_used, "monte_carlo_standard_error": mc_se,
            "nonzero_paired_differences": int(np.count_nonzero(difference)),
            "minimum_attainable_raw_p": min(1., 2. / 2 ** int(np.count_nonzero(difference)))
            if n <= 20 else 1. / (permutations + 1),
            "bootstrap_resampling_unit": "paired_independent_unit"}


def aggregate(rows: list[dict], method: str, delay: float, metric: str) -> dict:
    result = {}
    for row in rows:
        if row["condition"] == method and row["delay_seconds"] == delay and row.get(metric) is not None:
            value = float(row[metric])
            if np.isfinite(value):
                result.setdefault(row["unit"], []).append(value)
    return {key: float(np.mean(values)) for key, values in result.items()}


def comparisons(rows: list[dict], reference: str, methods: list[str], metrics: list[str], config: dict,
                family="all_planned_comparisons_in_this_analysis_file", planned_delays=None):

    results = []
    delays = sorted(set(planned_delays) if planned_delays is not None
                    else {r["delay_seconds"] for r in rows})
    for delay in delays:
        for method in methods:
            for metric in metrics:
                a, b = aggregate(rows, reference, delay, metric), aggregate(rows, method, delay, metric)
                units = sorted(a.keys() & b.keys())
                name = f"{reference}|{method}|{delay:.12g}|{metric}|{family}"
                offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
                seed = (int(config["seed"]) + offset) % 2**32
                inference = paired_inference([a[u] for u in units], [b[u] for u in units], seed,
                                             config["experiments"]["bootstrap_samples"],
                                             config["experiments"]["permutation_samples"])
                results.append({"reference": reference, "comparison": method,
                                "delay_seconds": delay, "metric": metric, "units": units,
                                "reference_n_units_available": len(a), "comparison_n_units_available": len(b),
                                "reference_only_units": sorted(a.keys() - b.keys()),
                                "comparison_only_units": sorted(b.keys() - a.keys()), **inference})
    corrected = holm([r["p_raw"] for r in results], family_size=len(results))
    for row, p in zip(results, corrected):
        row.update(p_holm=p, family=family, family_size_planned=len(results),
                   family_size_estimable=sum(r["p_raw"] is not None for r in results),
                   correction="Holm_step_down_FWER", alpha=.05,
                   significant_holm=p < .05 if p is not None else None)
    return results
