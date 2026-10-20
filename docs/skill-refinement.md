# Demonstration collection, batch refinement and generalization

This protocol describes the executable implementation. Synthetic recordings are
identified as such in every record. A successful simulator episode is not a
physical grasp validation or a participant observation. The default 20-success
quota, ten training repeats and 100 autonomous evaluations per scene family are
execution settings; their presence in a configuration does not establish that
those experiments have been completed.

## Recorded signals and annotations

Every processed demonstration is an NPZ file, readable without pickle, containing:

| Field | Shape and unit | Meaning |
|---|---|---|
| `time` | T, seconds | Strictly increasing timestamps; elapsed time is measured from the first sample |
| `position` | T × 3, m | Measured TCP position in the robot base frame |
| `velocity` | T × 3, m/s | TCP linear velocity in the same frame |
| `raw_force` | T × 3, N | Unfiltered master command after device axis calibration |
| `observed_state` | T × 6, m and m/s | Delayed robot position and velocity actually available when issuing each command |
| `phase` | T, integer | Approach/grasp = 0, lift = 1, transport/place = 2, complete = 3 |
| `metadata` | JSON string | Schema version, episode/operator IDs, provenance, source condition, delay, task outcome, intended/grasped object, placement and scene geometry |

The raw controller-session and simulation files additionally retain source and
command timestamps, posterior/latent estimates, filtered human force, nominal and
executed force, references, arbitration/confidence variables, obstacle clearances,
port allocations and all energy-accounting terms. Preserve these raw files before
conversion. Task success for measured recordings requires independent annotations;
a controller reaching a commanded pose does not certify a physical grasp.

`dasa import-recording` converts a device recording with its annotation JSON to the
processed contract. Do not infer missing success, collision or gripper evidence
from smooth TCP trajectories. Store object/gripper measurements and annotation
provenance with the experimental record. Never replace participant input with a
simulated master signal while retaining measured-data provenance.

## Eligibility and acquisition order

Collection is at **0.8 s robot-to-operator feedback delay**. The command-channel
delay is an independently configured quantity. Generalization input validation
checks the recorded feedback-delay metadata and source condition. Use zero-delay
bootstrap demonstrations before this collection.

A candidate is accepted only if all of these checks pass:

1. The annotated task is successful and placed; the grasped object equals the
   intended categorical goal.
2. Neither a collision nor a workspace violation is annotated.
3. Every recorded TCP segment has strictly positive swept clearance from all
   obstacle spheres after expanding by `simulation.robot_radius`.
4. The endpoint lies within `simulation.place_tolerance` of the placement center,
   and elapsed duration does not exceed `simulation.max_seconds`.
5. Neither its episode ID nor the hash of its numerical signals/geometry duplicates
   an original or previously encountered candidate.

These geometric checks cover the TCP sphere abstraction. Full-link collision,
contact forces, secure grasp and object slip require separate device measurements;
the simulator does not resolve them. The physical and simulation termination
criteria must therefore be recorded separately.

The first eligible successes are used. There is no ranking by downstream test
performance. `selection_audit.json` records every candidate, its acquisition index,
accepted-prefix index, reasons for rejection, IDs, operator, provenance, duration,
signal SHA-256 and annotation-sensitive record SHA-256. A renamed copy retains the
same signal hash and is rejected. Operator and signal overlap with validation/test
data causes an error before updating a measured model.

`refine-data` uses lexicographic filename order. Use zero-padded acquisition names
such as `0000.npz`. The generalization runner also accepts `order.json`, a JSON list
of NPZ filenames naming every recording exactly once; this freezes an explicit
acquisition order without renaming files. Original data are retained with their
original IDs. A rejected episode is never silently rewritten as a successful one.

The `refinement` experiment stage is a simulation-only collection workflow and
requires an explicitly synthetic bootstrap corpus. It raises an error for measured
bootstrap data rather than silently replacing those data with generated paths.
Use `refine-data` or `generalization` for recorded measured collections.

The simulation collection attempt limit is
`refinement_count * experiments.collection_attempt_multiplier` (default multiplier
5). Reaching this limit with fewer successes is an **incomplete quota**. Such a fit
is saved for diagnosis, marked `confirmatory_eligible: false`, and excluded from
confirmatory equal-budget tests. Program completion and protocol completion are
separate fields. A result based on zero new successes cannot be described as a
20-demonstration refinement.

## Initial TP-HSMM and update rule

The bootstrap training inventory lists every source file and partition, with
operator/episode-disjoint training, validation and test groups. No held-out episode
is used to train the TP-HSMM, cVAE, goal classifier or sequence baselines.
Synthetic bootstrap trajectories are quintic waypoint motions with explicitly
recorded seeds; they provide executable test inputs, not a substitute for the
original measured bootstrap corpus.

For each complete position trajectory:

1. Normalize its time to [0, 1] and interpolate to 140 equally spaced samples.
   No test-time outcome enters this transform. Mean duration in physical seconds
   is retained for reproduction timing.
2. Express the trajectory in three task frames centered at the start TCP,
   intended object and placement center. Axes remain parallel to the calibrated
   robot base; this model does not learn arbitrary orientation changes.
3. Initialize seven ordered states by equal normalized-time bins. Initialize
   each duration mean to 140/7 samples and standard deviation to 140/(3×7).
4. Fit framewise Gaussian emissions and explicit truncated discrete-Gaussian
   duration distributions with six EM iterations. The left-to-right topology
   visits every state once without skips; duration support is 1…140 samples.
   Covariance eigenvalues are floored at `1e-5 m²`. The duration standard-deviation
   lower bound is 1.5 resampled samples. Duration fitting uses bounded L-BFGS-B on
   the normalized truncated-duration likelihood.
5. At reproduction, transform each frame Gaussian to the task scene and combine
   the three experts by summed precision. Infer the maximum-probability duration
   sequence for the fixed horizon and apply six-state position/velocity LQT with
   double-integrator dynamics and `R = 0.01 I`.

The EM fit uses the **sum of frame log densities**, not the normalized world
Gaussian-product log density; the latter is used during task adaptation. These
are distinct operations. The fit log records its actual weighted likelihood.

Refinement is a **fresh weighted batch fit on the union**, rather than an
incremental parameter update:

`D_fit = D_original ∪ first_n_successful_unique(D_collection)`.

Each original demonstration has weight `skill.original_weight = 1`; each accepted
new demonstration has `skill.new_weight = 1` by default. These are per-episode
weights multiplying that episode's emission/duration sufficient statistics.
Because all position paths have equal resampled length, the default gives equal
empirical mass to each trajectory. The original/new *group* masses are therefore
proportional to their episode counts; they are not each normalized to 50%.

The model is initialized and trained from scratch on this union. Neither cVAE,
classifier nor intent baselines are updated by this TP-HSMM-only refinement.
Model NPZ files include the exact fit inventory, weights, initialization, effective
skill configuration and EM history. Legacy checkpoints without these fields are
reported as missing training records; metadata are not reconstructed from guesses.

## Running measured-data refinement

```bash
dasa validate-data data/processed
dasa train-data data/processed --output artifacts/bootstrap
dasa refine-data data/processed data/collections/C3 --output artifacts/refined_C3
```

`artifacts/refined_C3/` contains `refined_hsmm.npz`, `selection_audit.json`,
`training_inventory.json`, `refinement_metadata.json` and an input-checksummed
`manifest.json`. Read `quota_met` before using this model for a planned
20-success comparison. Validation/test operators may not appear in the collection.

## Learning curves and broader generalization

The separate `generalization` runner evaluates recorded successful prefixes at
budgets `[0, 5, 10, 20]` and supports a prespecified 40-demonstration extension.
Budget 0 is the single bootstrap-only reference per training repeat. It is not
copied into three nominally different condition groups. Every nonzero-budget
model is independently refitted on the same original corpus and the corresponding
first-success prefix. Insufficient budgets remain unavailable; they are not
filled by synthetic demonstrations or reused episodes.

Supply the following directory structure:

```text
collections/
  r0/
    C1/0000.npz ...
    C2/0000.npz ...
    C3/0000.npz ...
    bootstrap/splits.json   # optional independent bootstrap corpus and its files
  r1/ ...
```

A collection NPZ may be a processed demonstration or a raw simulator `Trial`.
Only the processed schema is appropriate for independently annotated device data.
Source-condition and 0.8 s delay metadata are mandatory. An episode cannot appear
in two repeats or conditions, even after being renamed. The simulator's `refinement/collection/r*/` layout is also accepted and uses the
corresponding saved `refinement/bootstrap/r*/` corpus, preserving the original
collection-time initial model. When no per-repeat bootstrap corpus is supplied,
all models use the prepared bootstrap dataset and
inference is explicitly conditional on that shared corpus. Supplying the same
recordings under different repeat names does not create independent evidence.

```bash
dasa --profile configs/experiments/generalization.yaml generalization artifacts/bootstrap \
  data/collections --output artifacts/bootstrap/generalization
dasa --profile configs/experiments/generalization.yaml figures --output artifacts/bootstrap
```

Evaluation scene families are demonstrated layouts, per-target x/y displacements
of 5–15 cm with one extra obstacle, displacements scaled to 8–24 cm, and layouts
with three extra obstacles. Displacements are clipped to the declared workspace,
so the saved actual coordinates define the realized perturbation. The constrained
sampler checks target/obstacle separation, object envelopes, workspace bounds and
nominal TCP paths, and records all rejection attempts. This is a static,
translational, known-scene benchmark. It does not test novel objects, visual
recognition under domain shift, dynamic obstacles, contact-rich manipulation or
arbitrary 6D orientations. Those require additional measured trials and cannot be
claimed from these results.

All budgets and methods replay the identical frozen scene list; training-repeat
seeds and all actual layouts are saved. For an externally frozen layout list,
pass `--scenes path/to/scene-manifest.json`. Each resulting model is evaluated in
100 trials per scene family under the full profile. The smaller validation
profile may override this count and is not a full-budget result.

Output files:

| Path | Contents |
|---|---|
| `scenes.json`, `scene_manifests/*.json` | Frozen layouts, geometry constraints, sampling seeds and rejection ledger |
| `models/r*/` | Bootstrap and complete-budget TP-HSMM models with fit inventories |
| `trials/r*/` | Every autonomous trajectory and complete controller signals |
| `metrics.json` | Per-trial outcomes, scenario, budget, repeat and eligibility |
| `repeat_summary.json` | Per-repeat success rate and trajectory error; failed partial trajectories remain labeled |
| `budgets.json`, `selection_audit.json` | Available successes, missing budgets and all selection decisions |
| `statistics.json` | Full raw/Holm-adjusted p values, effect sizes, confidence intervals and unit lists |
| `manifest.json`, `protocol_status.json` | Source/input/model/scene checksums and completeness status |

The independent unit for these policy comparisons is a **training repeat**, not
one of its 100 autonomous trials or one trajectory timestamp. First aggregate
within each repeat/scenario/budget. Two-sided paired sign-flip tests compare C3
with C1, C2 and the bootstrap-only reference. Holm correction spans all planned
method × budget × scenario × metric comparisons in this analysis. Confidence
intervals resample paired training repeats. With fewer than two matched complete
repeats, inferential outputs remain unavailable.

Independence/exchangeability of training repeats must be supported by collection
sessions and operator allocation. Shared operators and a shared bootstrap corpus
limit the population to which conditional repeat-level results generalize.
Trajectory reproduction error is the mean positional deviation from each model's
own reference; it measures tracking consistency and is not independent evidence
of task correctness or semantic policy quality. Report collision/success outcomes
alongside it and retain failed trials in the success denominator.
