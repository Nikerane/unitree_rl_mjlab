# Z1 diagnostic-0.9 impulse-CaT post-training evaluation

**Status:** banked native-result record, 2026-08-17. **Verdict:** the target policy learned a
large reduction in J3 reaction-impulse exposure while retaining 100% task success and productive
first strikes, but its true 500 Hz velocity-limit violation risk increased in all three stochastic
replicas. The preregistered native verdict is therefore branch 3: **redistribution/trade-off, not
clean enforcement**.

This is one training seed in simulation. Soft CaT is not a physical clamp, neither cap vector is a
manufacturer-certified limit, and this record makes no hardware-safety claim.

## Frozen evidence and analysis contract

- Vega job `41386821_[0-1]`: both array tasks `COMPLETED`, exit `0:0`, empty stderr; control leaf
  `41386821_0_diag90_control`, target leaf `41386821_1_diag90_target`.
- Evaluator revision `142098b8af4cbd4b52777012b27f477f0e9aa358`; asset revision
  `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Control checkpoint SHA-256
  `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3`; target
  `ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15`.
- Both five-row remote `SHA256SUMS` manifests were re-parsed in fixed order and all ten artifacts
  were rehashed successfully. Checkpoint roles, task, protocol, code/assets, exact stochastic seed
  tuple, matched initial-population hashes, live `imp_max_p=0`, threshold order, array shapes, and
  finiteness were validated before comparison.
- Primary unit: paired initial episode, clustered by whole environment ID with the same resampled
  IDs used in both arms. Each interval uses exactly 10,000 resamples and seed `2026081703`.
  Controller reads and overlapping windows are never inferential units.
- The fixed population is 64 mean-action worlds. Stochastic replicas are reported separately in
  the preregistered order `2`, `2026081701`, `2026081702`, each with 4,096 worlds. The 12,288-world
  concatenation is descriptive only—there is no pooled CI or p-value.
- Compact sufficient statistics and all raw-input digests are in
  `docs/results/assets/2026-08-17_z1_impulse_diag90_500_evaluation/analysis.json`. The deterministic
  source is `scripts/analyze_vic_impulse_diag90_500_evaluation.py`; both are bound by the adjacent
  `SHA256SUMS`. The 124+ MB raw traces/summaries remain outside Git.

`rho = max_t,j Lambda[t,j] / L[j]`. Risks below are the fraction of initial episodes with
`rho > 1`; RD is target minus control in percentage points. All CIs are paired environment-cluster
bootstrap intervals.

## Evidence: provisional project caps first

Caps: `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N.m.s`.

The fixed mean-action population had 0/64 violations in both arms. Control versus target rho was
p50 `0.771` vs `0.468`, p95 `0.797` vs `0.507`, p99 `0.837` vs `0.622`, and max `0.853` vs
`0.630`. Thus, for this fixed surveyed population, **"impulse constraint is empirically nonbinding
under the surveyed population."** This statement does not generalize beyond the sampled worlds.

| Stochastic replica | Control risk | Target risk | RD, pp (95% CI) | rho p95 C→T | rho p99 C→T | rho max C→T |
|---|---:|---:|---:|---:|---:|---:|
| `2` | 0.854% (35/4096) | 0% (0/4096) | −0.854 [−1.147, −0.586] | 0.807→0.643 | 0.974→0.708 | 1.300→0.938 |
| `2026081701` | 0.732% (30/4096) | 0.024% (1/4096) | −0.708 [−0.977, −0.464] | 0.807→0.648 | 0.908→0.713 | 1.239→1.059 |
| `2026081702` | 0.854% (35/4096) | 0% (0/4096) | −0.854 [−1.147, −0.586] | 0.807→0.638 | 0.987→0.701 | 1.198→0.893 |

Pooled descriptive counts were 100/12,288 (`0.814%`) control versus 1/12,288 (`0.008%`)
target; pooled rho p95 was `0.807→0.644` and p99 `0.963→0.708`. The first-contact-event-prefix
p95 values match these full-initial-episode values to within `0.0002` in control and exactly in
target; shorter target episodes therefore do not explain the reduction.

## Evidence: trained diagnostic caps

Caps: `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N.m.s`. These are the treatment-effect
endpoint, not hardware limits.

The fixed population again had 0/64 violations in both arms. Control versus target rho was p50
`0.857` vs `0.519`, p95 `0.885` vs `0.563`, p99 `0.930` vs `0.691`, and max `0.948` vs `0.700`.

| Stochastic replica | Control risk | Target risk | RD, pp (95% CI) | Risk ratio (95% CI) | rho p95 C→T | rho p99 C→T | rho max C→T |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2` | 1.416% (58/4096) | 0.098% (4/4096) | −1.318 [−1.685, −0.952] | 0.069 [0.015, 0.151] | 0.896→0.714 | 1.082→0.786 | 1.445→1.043 |
| `2026081701` | 1.025% (42/4096) | 0.073% (3/4096) | −0.952 [−1.270, −0.635] | 0.071 [0, 0.175] | 0.897→0.720 | 1.009→0.792 | 1.376→1.176 |
| `2026081702` | 1.343% (55/4096) | 0% (0/4096) | −1.343 [−1.685, −1.001] | 0 [0, 0] | 0.896→0.709 | 1.097→0.779 | 1.331→0.992 |

Pooled descriptive counts were 155/12,288 (`1.261%`) control versus 7/12,288 (`0.057%`)
target; pooled rho p95 was `0.896→0.715` and p99 `1.070→0.787`. The first-contact-event-prefix
comparison has the same direction and essentially the same magnitude.

### Per-joint redistribution

J3 caused every diagnostic violation. Across the three replicas, J3 p95 utilization fell from
`0.896–0.897` to `0.709–0.720`, J3 p99 from `1.009–1.097` to `0.779–0.792`, and violating
initial episodes from `58/42/55` to `4/3/0`. J1–J5 p95 and p99 utilization also fell. J6 p99 rose
slightly (`0.032–0.034` control to about `0.033–0.034` target) while remaining tiny, with no J6
impulse violation. Per-joint Lambda p50/p95/p99/max, utilization, positive margins, read counts,
responsible joints, and physical-event associations are retained in the compact JSON.

## Evidence: utility, true velocity, and VIC gains

| Population | Success / productive strike, C→T | First-event delivered mean ratio T/C | Duration p50, ms C→T | True-velocity risk C→T | RD, pp (95% CI) | True-speed p99 C→T, rad/s |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 64/64→64/64 | 1.020 | 140→100 | 0→0 | 0 [0, 0] | 3.105→3.105 |
| `2` | 4096/4096→4096/4096 | 1.017 | 140→100 | 0.513%→0.928% | +0.415 [+0.098, +0.733] | 3.125→3.140 |
| `2026081701` | 4096/4096→4096/4096 | 1.019 | 140→100 | 0.684%→1.123% | +0.439 [+0.049, +0.830] | 3.129→3.143 |
| `2026081702` | 4096/4096→4096/4096 | 1.017 | 140→100 | 0.659%→1.123% | +0.464 [+0.098, +0.830] | 3.131→3.142 |

The target clears every provisional utility margin: fixed success/productive strike is 64/64;
stochastic success is not below control; first-event delivered impulse exceeds rather than falls
below the `0.90` ratio floor. Nail depth and cumulative delivered impulse are also retained in the
compact JSON. The velocity margin fails: target velocity-violation risk is higher in every replica,
and every paired 95% RD interval is above zero. Descriptively pooled, true-velocity violations are
76/12,288 (`0.618%`) control versus 130/12,288 (`1.058%`) target.

Median first-contact VIC values were identical in fixed and all three stochastic populations for
both arms:

- normalized stiffness coordinate `[-1, -1, +1, +1, +1, -1]`;
- Kp `[800, 1200, 1250, 1250, 1250, 800]`;
- Kd `[89.44, 134.16, 111.80, 111.80, 111.80, 89.44]`.

The detailed precontact/contact p50/p95/p99/max summaries likewise show saturation at these values.
This evaluation therefore does not attribute the impulse reduction to a visible first-contact gain
shift; other learned action/trajectory behavior must account for it. Episode reward is explicitly
unavailable because the frozen evaluator did not record reward; no value is imputed.

## Evidence: impulse-versus-velocity attribution and event pressure

At provisional caps, control had `35/30/35` active reads by stochastic replica and target had
`0/1/0`. At diagnostic caps, control had `58/43/56` and target `4/3/0`. At offline `p=0.5`, impulse
won every one of these active reads; velocity masked zero and ties were zero. Thus the trained
constraint was active and independently winning where Lambda exceeded the cap, but the target
policy made such reads rare. `delta_velocity`, `delta_impulse`, exact
`max(delta_velocity, delta_impulse)`, read/window/event counts, and `1-product(1-delta_impulse_t)`
pressure are all preserved in the JSON.

The pressure split does not license a complete-event dose claim. For diagnostic control, completed
unambiguous versus physically right-censored/ambiguous activation windows were `51/7`, `35/7`, and
`51/4` across the three replicas; none reaches 95% completion. Target has only `4`, `3`, and `0`
activating windows. Across all physical contacts—not just activating ones—right-censoring is
`4036/4103`, `4050/4112`, `4036/4103` control and `3986/4100`, `3967/4096`, `3999/4097` target.
Completed, censored, unambiguous, and ambiguous event-pressure quantiles are separate in the JSON;
they are never pooled as independent reads.

## Inference: the five preregistered verdict branches

1. **Lower diagnostic violation probability and lower p95/p99 utilization, with retained task and
   velocity behavior: learned graded impulse reduction for this training seed.** Impulse and task
   conditions hold; retained velocity behavior does not, so branch 1 is not selected.
2. **Lower mean but unchanged violation rate or tail: behavioral softening, not boundary
   confinement.** Not selected: both diagnostic risk and tail fell sharply.
3. **Lower J3 exposure with higher exposure at another joint or in velocity:
   redistribution/trade-off, not clean enforcement.** **Selected.** J3 exposure fell, but true
   velocity-limit risk increased in every replica with paired intervals above zero.
4. **Active, winning impulse pressure with unchanged tail: calibration/treatment-strength
   problem.** Not selected: impulse pressure was active/winning, but the tail changed materially.
5. **Absent or too-rare provisional-cap violations: report exactly, "impulse constraint is
   empirically nonbinding under the surveyed population."** Applied only to the 64-world fixed
   population. It is not the stochastic verdict because control had 100 provisional violations
   across the three replicas.

The evidence supports learned policy shaping for this one training seed, but the clean graded
impulse-reduction claim is blocked by the velocity trade-off. Zero observed target violations in
one replica does not prove zero risk, a hard clamp, or hardware safety.

## Censoring decision and separate follow-up

Native termination remains authoritative for task behavior. It does not provide the required
complete, unambiguous event dose: physical contacts remain overwhelmingly terminal-censored and
the activating subsets fail the 50-event/95%-completion calibration gate. No release/window-flush
shadow was run and `imp_max_p` was not raised.

A separate, test-first, non-authorized follow-up is specified in
`docs/superpowers/plans/2026-08-17-z1-impulse-diag90-release-window-flush-shadow.md`. It removes only
`nail_driven` termination after exact native-boundary parity, observes five off-contact 2 ms samples
plus the 25-substep Lambda flush, stops at 250 ms, and keeps unresolved cases right-censored.

## Limitations

- One PPO training seed; three evaluation replicas measure inference/reset stochasticity, not
  independent training variability.
- Simulation only; no manufacturer limit, hardware validation, damage probability, or safety case.
- Soft CaT shapes learned behavior through graded continuation/credit; it is not a runtime clamp.
- The treatment used diagnostic caps. Provisional-cap and diagnostic-cap claims must remain
  separate.
- Native terminal censoring blocks complete-event-dose calibration; no inference uses controller
  reads as independent observations.
- Reward was not recorded. Task success, productive strike, delivered impulse, depth, duration,
  true velocity, and gains are available; reward is not reconstructed.
