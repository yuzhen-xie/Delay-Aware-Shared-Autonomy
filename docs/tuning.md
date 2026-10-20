# Validation-only sensitivity and parameter selection

The checked-in values are reproducible reference settings, not the outcome of
an undocumented optimization. `validation-sweep` provides an executable,
prospective procedure for investigating the runtime parameters listed in
`configs/experiments/sensitivity.yaml`. It must not be presented as a recovered
historical tuning record.

First prepare the bootstrap models and the operator-disjoint dataset splits:

```bash
dasa prepare --output artifacts/reference
dasa --profile configs/experiments/sensitivity.yaml validation-sweep artifacts/reference --output artifacts/sensitivity
```

For a shorter, explicitly partial sensitivity run:

```bash
dasa --profile configs/experiments/sensitivity.yaml validation-sweep artifacts/reference --output artifacts/sensitivity-beta --parameters intent.beta
```

The Python API is `sweep(config, prepared_root, output, parameters=None)` in
`dasa.tuning`. It returns per-trial metric records. `parameters`, when supplied,
is a list of exact dotted keys from the configured grids. The output directory
must be empty to preserve previous analyses.

## Design and information boundary

Each parameter is varied one at a time, keeping all other values at the supplied
base configuration. The grid is not a Cartesian product. The routine rejects
training hyperparameters and unknown paths; it supports the nine runtime
parameters in the supplied sensitivity grid. It validates every candidate
before opening model files or creating outputs. In particular, distance
thresholds must remain ordered, Kalman measurement variance must remain
positive, and all grid values must be distinct finite numbers. A configured
`tuning.split` other than `validation` is rejected.

The loader reads `dataset/splits.json` to reject duplicate or overlapping file
paths, then opens only the listed validation demonstrations. It does not open
training or test demonstrations, their signals or their outcome metrics.
Operator-level separation still depends on the prepared dataset's partition
audit; this routine cannot establish test-operator separation without inspecting
test records. Preserve the original partition audit with the prepared models.

Validation demonstrations supply the scene and intended goal. Their original
force traces and success labels are not replayed as observed human responses.
The routine runs C3 in the task-space simulator using a common random seed,
goal and simulated-operator noise profile for each validation scene across all
candidate values. Seeds are `run_seed + 900000 + validation_index`; odd indices
use the noisy simulated-operator profile and even indices use the expert
profile. These are simulated responses on held-out layouts, not a new human
experiment. The frozen scene list records the exact geometry, goal, source
episode, source operator group, random seed and noise assignment.

The prepared TP-HSMM and intent checkpoints remain fixed. No optimizer or model
fit is called. The intent loader checks its objective and architecture against
any supplied training audit. If an older checkpoint has no audit, its objective
is recorded as `unrecorded`; the sensitivity run does not establish its training
history. All model files are hashed before and after the sweep, and a mismatch
aborts completion.

## Ranking and interpretation

All collisions, bounds violations, wrong-object selections and timeouts remain
in the metric table. Within each source-validation-operator group, trial success
and minimum executed surface clearance are averaged over all its validation
scenes. Group means then receive equal weight in the candidate summary. These
groups describe the source partition; the closed-loop operators are simulated.

Candidates are ranked separately for each parameter by:

1. Mean group success rate, descending.
2. Mean group minimum surface clearance, descending, including failures and
   negative clearances.
3. Mean successful completion time, ascending. This mean is computed within
   each group with successes and then across those groups. It remains missing
   if there are no successes; a missing value ranks last at this tie breaker.
4. Distance to the base value, then original grid order, to resolve exact ties.

Successful completion time is conditional on success and can be affected by
which trials succeed. It is therefore a tertiary criterion rather than a
replacement for the failure-inclusive success criterion. Inspect all raw rows,
termination reasons and sample counts when comparing candidates. Sensitivity
rankings are descriptive validation results, with no significance claims or
held-out performance guarantee.

`selection.json` records a prospective validation winner for each parameter.
If a grid omits the base value, this is stated explicitly. Winners from separate
one-at-a-time grids must not be combined and called an optimized controller:
their interactions have not been assessed. A subsequent combined configuration
requires a separate prespecified validation procedure. After a final choice,
freeze its complete configuration and model hashes before running the test
pipeline once. This command does not change default files or evaluate the test
partition automatically.

## Outputs

| File | Contents |
| --- | --- |
| `validation-scenes.json` | Exact frozen validation layouts, goals, seeds and source grouping |
| `trials/*.npz` | Every raw trial, including all unsuccessful trials |
| `metrics.json` | One row per candidate and validation scene; termination and failure-retention flags |
| `ranking.json` | Candidate rankings, group summaries, success/clearance/time values and counts |
| `selection.json` | Declared ranking rule, per-parameter prospective choices, split-access audit and limitations |
| `manifest.json` | Full configuration, code version, input hashes, unchanged checkpoint hashes and training-objective status |

The integration test runs actual short simulator trials, guards demonstration
loading so that only the validation recording can be opened, verifies failure
retention and missing completion-time handling, and checks that the original
configuration and prepared model files remain unchanged. Additional tests reject
invalid grids before file access and verify equal weighting of source groups.
