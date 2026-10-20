# Experiment and figure recipes

These recipes generate figures from saved records. They do not reconstruct
historical measurements from image axes or published summary values. A new
simulation run, a proposed human protocol and an observed human study are separate
data sources. Record the Git commit and keep `manifest.json`, the resolved
configuration, split manifests and model checkpoints with each run.

Run commands from the repository root after activating the environment described
in the README. Every new training run uses a new output directory. `--profile`
merges a configuration override into `configs/default.yaml`.

| Export | Required records | Initial model and configuration | Commands | Output |
| --- | --- | --- | --- | --- |
| Fig. 5 | Operator-disjoint train/validation/test demonstrations; saved held-out predictions and `intent/metrics.json` | Bootstrap TP-HSMM, cVAE/classifier and sequence baselines in `models/`; `configs/experiments/intent.yaml` | `dasa --profile configs/experiments/intent.yaml prepare --output artifacts/intent`; then `dasa --profile configs/experiments/intent.yaml intent --output artifacts/intent`; then `dasa --profile configs/experiments/intent.yaml figures --output artifacts/intent` | `figures/figure5.{png,pdf,svg}`, unit/summary/trial CSV, caption, full `figure5_statistics.{json,csv}` |
| Fig. 6 | Matched condition/delay trial logs; `sharing/metrics.json` | Frozen bootstrap TP-HSMM and intent models; `configs/experiments/sharing.yaml` | `dasa --profile configs/experiments/sharing.yaml prepare --output artifacts/sharing`; then `dasa --profile configs/experiments/sharing.yaml sharing --output artifacts/sharing`; then `dasa --profile configs/experiments/sharing.yaml figures --output artifacts/sharing` | `figures/figure6.{png,pdf,svg}`, unit/summary/trial CSV and full `figure6_statistics.{json,csv}` |
| Fig. 8 | Pseudonymous analysis-unit IDs, task trials, recorded analysis plan and observed six-subscale NASA-TLX responses | No learned questionnaire model; the analysis plan specifies the conditions and delay | `dasa study-analyze study/local/trials.csv study/local/participants.csv study/local/protocol.yaml --tlx study/local/tlx.csv --output artifacts/human-study` | `figures/fig8_workload.{png,pdf,svg}`, `tlx.json`, `tlx_statistics.{json,csv}` and the analysis audit in the study output directory |
| Learning comparison | Original bootstrap demonstrations, all collection attempts, success-selection audit and independent training/evaluation repetitions | Original TP-HSMM plus full refits with accepted new demonstrations; `configs/experiments/learning.yaml` | `dasa --profile configs/experiments/learning.yaml prepare --output artifacts/learning`; then `dasa --profile configs/experiments/learning.yaml refinement --output artifacts/learning`; then `dasa --profile configs/experiments/learning.yaml figures --output artifacts/learning` | Raw collection/evaluation NPZ, accepted counts, fitted models, `refinement/statistics.json`; `figures/learning.{png,pdf,svg}` and tables |

The `prepare` examples create explicitly synthetic bootstrap data. The `intent`
and `sharing` stages generate simulator interactions even if their scene
coordinates originated from measured demonstrations. Use the separate recorded
route to evaluate measured input signals:

```bash
dasa --profile configs/experiments/intent.yaml train-data DATASET --output artifacts/recorded-models
dasa --profile configs/experiments/intent.yaml intent-recordings artifacts/recorded-models \
  --output artifacts/recorded-evaluation
dasa --profile configs/experiments/intent.yaml figures --output artifacts/recorded-evaluation
```

`intent-recordings` evaluates all eight methods on the same saved held-out pickup
samples. It preserves raw-force, current-state and supplied delayed-observation
alignment; it never generates operator commands or injects artificial delays.
Each recording must declare `delay_seconds`; only actually recorded delays
produce observations. Missing planned delays retain unavailable contrasts. Any
additional observed delays expand the correction family and are explicitly
marked exploratory. The goal labels evaluation outcomes without entering the
estimator. EPT uses elapsed recording duration only for successfully annotated
episodes, so recordings must terminate at the documented completion event.

Optional `observed_source_time`, `observation_source_time` or `source_time`
arrays must share the recording timestamp clock; their ages are `time-source`.
A saved `feedback_age` array is also accepted. Without these arrays, the age
argument uses nominal metadata and the output states that effective age was not
measured. `observed_state` is already aligned and is never delayed a second time.
Every prediction NPZ and the input inventory identify its source SHA-256,
provenance, timestamps and age source. Synthetic source records remain labelled
synthetic. Held-out records remain separate from model fitting and selection.
The human-data import path does not generate questionnaire answers.

## Layouts, randomness and timing

The configuration resolves all experiment counts and the master seed. Dataset
`splits.json` lists exact episodes. Each demonstration and trial NPZ contains its
full base-frame scene and its seed. With all indices starting at zero, the
simulation generation rules are:

| Stage | Seed and order |
| --- | --- |
| Bootstrap splits | `seed + 10000*split_index + episode_index`; train, validation, test in that order |
| Intent interactions | `seed + 200000 + test_episode_index`, shared across delay conditions |
| Shared control | `seed + 100000 + unit_index*100 + repeat_index`; condition order permuted with this seed plus integer delay in ms |
| Refinement bootstrap | `seed + 300000 + repeat_index*1000 + episode_index` |
| Refinement collection | `seed + 400000 + repeat_index*1000 + attempt_index`, matched across C1–C3 |
| Autonomous evaluation | `seed + 500000 + repeat_index*1000 + trial_index`, matched across collection conditions |

The constrained scene-manifest generator is described in
[perception and scenes](perception-scenes.md). `dasa audit-run --output RUN`
exports actual saved layouts to `RUN/audit/actual_scene_records.json`, including
records produced by the legacy sampler. Preserve the generated manifest
when evaluating a second method; a seed alone is insufficient if the generator
version or random-number consumption changes.

The supported intent and shared-control delay grid is 0, 400, 800 and 1500 ms in
the feedback channel, with the command delay specified separately. The figure
axis describes one-way feedback delay; it must not be relabelled as round-trip
latency. Refinement collection uses 800 ms feedback delay. Every recorded command
and observation carries its source timestamp; effective age includes the channel
configuration and causal sample-and-hold behavior.

The broader generalization pipeline uses accepted demonstration prefixes and
frozen evaluation scene manifests, including the zero-new-demonstration model.
Use the `generalization` command documented in [skill refinement](skill-refinement.md)
and `configs/experiments/generalization.yaml`:

```bash
dasa --profile configs/experiments/generalization.yaml generalization \
  artifacts/bootstrap data/collections --output artifacts/bootstrap/generalization
dasa --profile configs/experiments/generalization.yaml figures --output artifacts/bootstrap
```

The first positional argument is the prepared run containing `models/` and
`dataset/`; the second is the recorded collection directory. Keep output under
`RUN/generalization` so that `dasa figures --output RUN` discovers it. The exporter
writes one learning-curve figure per observed scenario. An unfilled demonstration
budget is recorded as unavailable; it is not replaced by a full-budget result.

## Quantities and statistical units

Fig. 5 exports IPA and EPT. IPA counts correct and confident pickup-stage
predictions. EPT requires a sustained correct/confident interval and a successful
task completion. Trials without an eligible prediction or completion retain a
missing EPT, and the table reports observed units and trials. The sustained
interval and confidence threshold come from the resolved configuration.

Fig. 6 exports successful completion time, success fraction, minimum TCP surface
clearance and Cartesian jerk RMS. These are operationally defined simulator or
recording metrics. An unspecified steering-speed signal, steering acceleration,
or index coefficient cannot be inferred from a translational trajectory. The
export therefore does not manufacture numerical safety/stability indices from
those missing definitions. Failures remain in success fractions, while
completion-time estimates condition on success. Analyze timeout-penalized time
separately when preregistered; do not interpret a faster survivor-only mean as
unconditional improvement.

Each panel first averages repeated trials within an operator, or evaluation
trials within an independent training repeat. It then gives each independent
unit equal weight. Small points are those unit means; large points are condition
means. Pointwise percentile bootstrap intervals resample independent units.
With one unit, an interval is unavailable. Time samples and the 100 evaluation
trials of one fitted policy do not create 100 independent model replicates.

Hypothesis tests and confidence intervals for paired differences are copied from
the analysis statistics files with their complete sample counts, paired units,
raw and adjusted p values, effects, confidence intervals and family definitions.
Condition-mean interval overlap is not a hypothesis test. Figures do not add
significance markers. Any later annotation must link to the predeclared
comparison and its Holm-adjusted p value. The analysis files state the
exchangeability assumptions and exclusion rules.

Incomplete refinement budgets are retained in `*_all_trials.csv` and
`*_incomplete_budget_diagnostics.csv`, and excluded from comparative learning
tables and plots. Read `refinement/budgets.json` and the protocol status before
interpreting a fixed-budget comparison.

## Export audit

`figures/manifest.json` records the resolved configuration, code revision, dirty
working-tree status, model and input hashes, available/missing figure families,
provenance and output file list. Each figure has CSV rows at trial, independent
unit and summary levels, plus a machine-readable summary and caption. Missing
metrics do not produce invented panels. The separate study pipeline saves its
own input audit and aggregate analysis metadata; private source records remain
outside the public repository.
