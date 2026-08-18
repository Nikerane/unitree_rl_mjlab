# Independent review request: Z1 impulse-CaT `p=0.2` bridge result

Review this completed screening experiment independently and bluntly. Distinguish the
preregistered result from the stronger thesis claim that would require independent PPO training
seeds. Do not infer actuator damage limits, hardware safety, or hard constraint enforcement.

## Research question and fixed protocol

Does a lower impulse-CaT training dose reduce native observed per-joint reaction-impulse exposure
without moving risk into true joint velocity or degrading hammering utility?

- Simulated six-joint Unitree Z1 variable-impedance hammering policy.
- Control: velocity CaT active, impulse CaT log-only (`imp_max_p=0`).
- Target: velocity CaT active, impulse CaT active (`imp_max_p=0.2`).
- Both used project-defined provisional caps `[0.82, 1.64, 0.82, 0.82, 0.82, 0.82] N.m.s`.
- The target trained for 500 PPO iterations with 4,096 environments and 24 rollout steps; the
  frozen control was reused only after exact log-only cap-vector identity passed.
- Frozen evaluation kept live impulse CaT log-only in both arms and used one descriptive fixed
  64-world population plus three separate matched stochastic 4,096-world populations.
- Whole paired environment IDs, not 50 Hz reads, were the resampling unit; each interval used
  10,000 bootstrap resamples with one preregistered seed. No pooled gate was permitted.

## Preregistered gates

Every stochastic population had to pass separately: provisional-risk difference upper 97.5% bound
`< -0.005`; provisional utilization p95 and p99 reductions each `>=10%`; no qualifying joint p99
increase `>=10%` and no new violating joint; true-velocity-risk difference upper bound `<=0.001`;
success and productive-strike differences each `>=-0.01`; delivered-impulse ratio lower bound
`>=0.90`; and identity, finiteness, and native-horizon claims valid.

## Exact preregistered results

Percentages are initial-episode risks. `RD upper` is the upper 97.5% paired bound for target minus
control. `rho` is maximum native observed per-joint Lambda divided by its provisional cap.

| Evaluation seed | Impulse risk C to T | Impulse RD upper | rho p95 reduction | rho p99 reduction | Velocity risk C to T | Velocity RD upper | Delivered ratio point / lower | Utility differences | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2` | 0.85449% to 0% | -0.58594 pp | 36.059% | 33.910% | 0.51270% to 0.17090% | -0.12207 pp | 0.99540 / 0.99483 | 0 / 0 | PASS |
| `2026081701` | 0.75684% to 0% | -0.51270 pp | 43.272% | 30.644% | 0.68359% to 0.19531% | -0.24414 pp | 0.99643 / 0.99571 | 0 / 0 | PASS |
| `2026081702` | 0.85449% to 0% | -0.58594 pp | 35.456% | 35.818% | 0.65918% to 0.24414% | -0.14648 pp | 0.99550 / 0.99498 | 0 / 0 | PASS |

All qualifying joint tails decreased, no target joint violated the provisional cap, all native
median episode durations were 140 ms in both arms, and the observed first-contact-prefix Lambda p99
also decreased in every stochastic population. The fixed-64 population had zero violations in both
arms and is descriptive only. Contact prefixes remained heavily right-censored: control
`4036/4103`, `4049/4112`, `4036/4103`; target `3888/4097`, `3894/4098`, `3890/4096`.

## Limitations

- This is one PPO training seed. The three evaluation seeds are not training replicates.
- The caps are project-defined simulation thresholds, not manufacturer damage limits.
- Soft-CaT is graded training pressure, not a runtime clamp.
- The result concerns native observed exposure, not complete contact or actuator loading.
- Zero observed target violations does not establish zero risk.
- `p=0.2` was predeclared as one bridge dose; this result does not show it is optimal or physically
  calibrated.
- First-event delivered hammer impulse is utility, not the constrained joint-reaction quantity.

## Required response

Return no more than 700 words:

1. `VERDICT: CONFIRM | STOP | REVISE`
2. What was actually established?
3. What is the strongest defensible thesis wording now?
4. What is the most serious remaining confound or unsupported claim?
5. Do the preregistered results justify matched independent-seed confirmation, or should this line
   stop? If confirmation is justified, state the minimal design without adding treatments.
6. Is any contact/window-completion sensitivity necessary before confirmation, or only before a
   complete-contact claim?
7. State the single most important reason this result could mislead us.
