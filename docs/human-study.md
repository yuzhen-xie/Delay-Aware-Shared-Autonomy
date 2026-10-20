# Participant trial protocol and analysis

The public workflow registers anonymous participant IDs and analysis-inclusion status. `study/planned-protocol.yaml` defines the prospective session procedure; `study/protocol.template.yaml` contains a complete analysis configuration. The default design has 20 participant IDs, four control conditions and four feedback delays. Generated schedules are plans. Measurements enter the analysis through recorded trial and workload CSV files.

## Run the workflow

Run from the repository root after installation:

```bash
dasa --config configs/default.yaml study-schedule --output artifacts/study/schedule.csv
dasa --config configs/default.yaml study-analyze \
  data/study/trials.csv data/study/participants.csv data/study/protocol.yaml \
  --tlx data/study/tlx.csv --output artifacts/study/analysis
```

The schedule command writes `schedule.csv`, `schedule_practice.csv` and `schedule.design.json`. It generates anonymous IDs, conditions, order and seeds. The CSV files under `study/` define data contracts. Create one registration row per ID, with `inclusion_status` equal to `included`, `excluded` or `withdrawn`. ID-only legacy registration files remain supported; additional registration columns are ignored. Trial and workload rows reference these IDs. Analysis uses included IDs and reports counts of registered, included, excluded and withdrawn IDs in `cohort_summary.json`.

Copy the protocol configuration alongside the measurements. Retain its study identifier, version date, scoring and exclusion definitions. Set `record_status: collected` when importing completed observations; this marks the input data stage and does not change any measurement. A prospective protocol and planned trial rows are rejected by the analyzer. The version date identifies the current public protocol. Use a separately versioned configuration to describe a different executed protocol.

## Session and practice

The prospective session allocates 10 minutes to task instructions and stop procedures. Explain candidate-goal selection, grasp and placement criteria, collision/failure handling, and delayed observations. Demonstrate the stop action with assistance disabled. Use the same scripted task explanation across sessions.

Eight practice trials follow in this order: C1 at 0, 0, 400 and 800 ms; C3 at 0, 400, 800 and 800 ms. Each uses a separate practice seed and a 60 s timeout. The readiness rule is understanding the goal/grasp/place/stop sequence and completing the final two practice trials without collision. Allow at most four additional practice trials, recording the reason, outcome and duration. If readiness is not reached, mark that ID excluded before experimental randomization. Store practice outcomes in `practice.template.csv`; practice trials are excluded from inferential analysis and retained in the audit.

Experimental trials comprise C1–C4 × feedback delays 0, 400, 800 and 1500 ms, one trial per cell by default. Command delay is 0 s in this protocol. Feedback values are one-way observation-channel delays. The participant-task timeout is 60 s; the simulator has a separate configured timeout. Stop on collision, workspace violation, wrong-object grasp, withdrawal, device fault or timeout, and record the first termination cause. Task failure is an outcome. Device-invalid observations can be excluded under the configured rule using `exclude_reason`; the audit preserves their IDs and reasons.

Give a 2 minute rest after every four experimental trials and allow additional recorded rest. Collect workload ratings after the relevant conditions at 800 ms. Budget approximately 1 minute per rating set and 5 minutes for debriefing. The protocol allocates 40–60 minutes overall, depending on trial lengths. Record executed trial order, practice count/duration, rest and task duration. This timing defines the current prospective workflow.

## Order, layouts and repeated exposure

The generator constructs Williams sequences over all 16 method/delay cells, permutes cell labels with seed 20261014, and assigns sequences reproducibly. A complete set of 16 sequences balances first-order carryover. With 20 IDs, four sequences occur twice and twelve once. The design JSON records the permutation and assignments, allowing position and carryover counts to be audited. Use the generated schedule before data collection and preserve its executed order in the trial record.

For each ID, delay and repetition, C1–C4 share the same scene seed and intended goal. Evaluation seeds are `seed + 1000000 + participant_index*10000 + repeat*100 + delay_index`, with all indices starting at zero. Intended goal is `(participant_index + repeat + delay_index) % 3`. Practice seeds use a disjoint range beginning at `seed + 9000000`. Save the actual scene coordinates, generator version, geometry constraints and calibration/perception timestamps with the scene manifest. Record manual layout changes. Practice layouts are excluded from test manifests. Freeze model weights and evaluation configuration before experimental trials.

The within-subject contrasts account for stable differences between IDs. Counterbalancing limits order confounding but does not establish that learning effects are absent. Preserve trial index, order and practice exposure. Treat trial-index interactions as exploratory unless separately specified before analysis. The 20-ID design is a feasibility cohort rather than a prospective power calculation. A new confirmatory design should specify its smallest effect, variance estimate, family-adjusted alpha, target power and attrition allowance.

## Scoring and failures

Success requires the correct goal object to be grasped, lifted and placed before timeout without collision or workspace violation. Completion time runs from the task start to valid placement for successful trials. Minimum clearance is the minimum TCP-to-obstacle surface distance over the entire trial. Jerk is the RMS norm of translational jerk in m/s³. The trial schema records the termination cause and elapsed duration, including failures. Preserve the raw time series used to derive these measures.

Failures remain in success-rate denominators. Their successful-completion time is absent. A separate timeout-penalized measure equals observed completion time on success and the declared timeout on failure. It is a penalty-based outcome. Successful-only completion time can show selection bias, so report it alongside success fraction and the penalty measure. Missing values are not imputed. A contrast uses only paired complete ID-level cells and reports IDs available on one side only.

## Hypotheses and complete statistics

The analysis unit is the participant ID. Repeated trials are averaged within ID × method × delay before comparison. The task family compares C3 with C1/C2/C4 across four delays and five outcomes: success fraction, successful completion time, minimum clearance, RMS jerk and timeout-penalized time. This gives 60 planned contrasts. The workload family has 21 contrasts at 800 ms: three method comparisons × RTLX and six subscales. Added outcomes or delays expand the declared family. Unestimable contrasts remain in the planned family size and produce no p value.

Each contrast uses a two-sided sign-permutation test of zero mean paired difference. Enumeration is exact through 18 independent units; larger samples use 20,000 sign draws with `(exceedances + 1)/(draws + 1)`. Assumptions are independent analysis units and joint sign-exchangeability of paired differences under the null. Each contrast has a stable SHA-256-derived seed. Output tables report the test name, number of pairs, raw p, Holm-adjusted p, family size, alpha, seed, Monte Carlo standard error, C3-minus-comparator mean difference, Cohen's dz and uncertainty intervals.

The 95% intervals use 5,000 paired-unit percentile bootstrap samples. They are pointwise intervals. Cohen's dz is undefined for zero sample SD of paired differences. Its bootstrap interval is omitted when over 5% of resamples have zero SD. An omitted p, effect or interval is not zero. Holm controls familywise error at 0.05 within each declared family; task and workload are separate prespecified families. Numeric tables accompany figures. Significance is never inferred from visual separation.

## Workload input and Fig. 8

Import six observed ratings from 0 to 100: mental, physical, temporal, performance, effort and frustration. All use higher values for greater burden. RTLX is their unweighted arithmetic mean. If the original performance scale uses higher values for better performance, convert it to `100 - original` in documented preprocessing and preserve the original rating. A scale already coded as failure/workload keeps its direction.

The analyzer produces:

- `trial_metrics.json`, `statistics.json` and `statistics.csv`: trial outcomes and complete task comparisons.
- `tlx.json`, `tlx_statistics.json` and `tlx_statistics.csv`: observed ratings and complete workload comparisons.
- `figures/fig8_workload.png`, `.pdf` and `.svg`: rating distributions and individual ID-level observations at the configured delay.
- `cohort_summary.json` and `study_metadata.json`: anonymous registration counts and the public analysis configuration.
- `analysis_plan.json` and `analysis_audit.json`: hypotheses, units, assumptions, multiplicity families, missingness, exclusions and input hashes.

If no ratings exist at the selected figure delay, `workload_figure_status.json` records that no figure was generated. Raw measurements, analysis tables and visualizations retain distinct provenance; the program never creates participant outcomes from a schedule.
