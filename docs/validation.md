# Implementation validation — 0.2.0

Validation date: 2026-10-20. Runtime: Linux, Python 3.12, CPU PyTorch 2.6.0.
Dependencies were installed from the hash-pinned lockfiles. RTDE native modules
imported without opening a device connection; dependency consistency passed.

## Executed checks

The automated suite passed **91 tests**. It covers explicit-duration model
inference, energy accounting, causal model inputs, exact training objective and
checkpoint audits, paired statistics, anonymous study import, perception and
scene geometry, recorded-signal evaluation, and feedback acknowledgement over a
real localhost UDP socket. Workstation tests use a controlled clock and robot
adapter fixtures; no physical robot was moved. Source lint and diff checks passed.

The end-to-end command was:

```bash
dasa --profile configs/smoke.yaml run --output artifacts/protocol-validation
```

This generated 32 synthetic demonstrations (20 training, 6 validation, 6 test),
24 held-out interaction traces, 192 method/episode/delay intent rows, 72 shared
control trials, 35 refinement collection attempts, 36 autonomous evaluations and
24 stress trials. These are saved executions, not measured participant outcomes.
Repeated predictions of the same trace are not independent experimental units.

The cVAE audit records `joint_elbo`, unit reconstruction/deviation/KL weights,
`p(z)=N(0,I)`, all epoch components and bound checkpoint hashes. The maximum
epoch-level loss-identity roundoff was `7.11e-9`. Across the saved simulation
traces, maximum absolute robot-work energy residual was `3.02e-13 J`; minimum
storage was zero. The sampled-port bench ran 5,000 steps, with maximum budget
residual `6.67e-16 J` and no duplicated input-energy credit.

## Outcomes and interpretation

| Shared condition | Successful / attempted |
| --- | ---: |
| C1 | 9 / 12 |
| C2 | 3 / 12 |
| C3 | 3 / 12 |
| C4 | 11 / 12 |
| Delay-BR sharing | 3 / 12 |
| C3 guarded | 3 / 12 |

The smoke refinement budget requests three successful demonstrations per
condition. C1 obtained 3 in 5 attempts. C2 and C3 each obtained 0 in 15 attempts.
Their 24 autonomous evaluation rows are diagnostic and excluded from fixed-budget
comparisons. `completed=true` reports program completion; `protocol_complete=false`
reports the unmet scientific collection budget. These observations do not support
a superiority claim for C3. Energy-budget compliance does not establish collision
avoidance.

Additional executed routes:

- Recorded-signal inference replayed six saved synthetic test demonstrations
  through eight methods, producing 48 rows while preserving synthetic provenance.
  It injected no new interactions or delays.
- A validation-only beta sweep evaluated three values on six validation scenes
  (18 actual simulation trials); it opened no test demonstration signals and
  did not change the default configuration.
- Budget generalization used the saved repeat bootstrap and collection attempts,
  budgets `0/1/3`, one training repeat and one frozen scene from each of four
  scenario families. It executed 12 autonomous trials, with 5 successes. Missing
  C2/C3 fits remained unavailable; 48 planned contrasts have no estimable p value.
  One training repeat cannot support model-level inference.
- The scene command generated and validated ten constrained extra-obstacle
  layouts. The study command generated 320 planned experimental rows and 160
  separate practice rows. Schedules contain no outcome measurements.

## Artifacts and scope

`report/index.html`, `figures/`, `models/training-audit.json`,
`refinement/{budgets,selection_audit,protocol_status,statistics}.json`,
`generalization/` and `audit/` contain the run outputs and evidence inventories.
Statistics were regenerated from unchanged saved metrics after the final
multiplicity-metadata corrections; the audit records that operation. The original
training source hashes and working-tree status remain in the run manifest.

These checks validate executable data paths and the stated discrete contracts.
The full 10-repeat/100-trial experimental budget, physical UR10 execution,
camera calibration and observed human-workload figures were not executed in this
cloud instance. They require their corresponding recorded input data and device
execution; simulation files are never substituted for those records.
