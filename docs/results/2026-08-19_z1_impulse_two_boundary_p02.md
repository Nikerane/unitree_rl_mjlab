# Z1 two-boundary impulse-CaT behavior diagnostic

**Status:** banked one-training-seed simulation diagnostic, 2026-08-19. Two seed-2 VIC-TT policies were trained at the same active impulse-CaT dose (`imp_max_p=.2`) under different diagnostic cap geometries, then evaluated frozen with live `imp_max_p=0`. Both are task-complete, but each retains a small own-boundary tail: uniform-0.9 at J3 and joint-stress chiefly at J6. This is **partial compliance with a geometry-dependent bottleneck**, not hard enforcement, actuator safety, or an optimum-boundary result.

## Exact treatment and evidence identity

| Role | Training cap vector (`N.m.s`) | Final checkpoint SHA-256 | Train / frozen-eval array |
|---|---|---|---|
| `p02_uniform09` | `[.738,1.476,.738,.738,.738,.738]` | `a99593b263a74944d60ac412bb1da733a36a29a1cd9f4eeeaed89906372595df` | `41622386_0` / `41636489_0` |
| `p02_joint_stress` | `[.369,.246,.738,.369,.246,.0164]` | `509cc26a2e521a935bcbc8c342040c95c7d2e3d7fc105a52d5d9450518bd2ec7` | `41622386_1` / `41636489_1` |

Both used native VIC-TT, seed 2, 4,096 environments × 24 rollout steps, 500 PPO iterations, active velocity CaT, active impulse CaT at `.2`, delivered-impulse weight 4, and impact-progress weight 8. The only treatment difference was the cap vector. Both training and frozen-evaluation arrays completed once per arm (`COMPLETED`, `0:0`, no restarts, empty stderr). The evaluator revision was `7642a0558970fb6ef450efe981809a0d48873456`; the asset revision was `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.

All eight frozen traces are finite, use live `imp_max_p=0`, have identically zero live `delta_impulse`, and have exact `delta=max(delta_velocity, delta_impulse)`. Fixed-64 is descriptive only; the three matched stochastic 4,096-world populations are separate. Controller reads are descriptive measurements, not inferential units.

## Own-boundary observations

`reads` are 50 Hz controller reads. `P50` is counterfactual `.2` activation-window pressure; all active reads were impulse winners with zero velocity masking. Each observed activation window and each associated physical event had one read, but terminal censoring forbids a complete-event dose claim.

| Policy / stochastic seed | Own-cap violating reads (responsible joints) | Max own-cap utilization | `P50/P95` pressure | True velocity-risk | Success / productive | Delivered first-event p50 (`N.s`) |
|---|---:|---|---|---:|---:|---:|
| uniform-0.9 / 2 | 6 (J3) | J3 1.155 | .178/.200 | .0073 | 4096/4096 | .4490 |
| uniform-0.9 / 2026081701 | 5 (J3) | J3 1.067 | .038/.176 | .0073 | 4096/4096 | .4490 |
| uniform-0.9 / 2026081702 | 4 (J3) | J3 1.140 | .104/.188 | .0093 | 4096/4096 | .4492 |
| joint-stress / 2 | 13 (J1:1, J5:2, J6:11) | J6 2.016 | .006/.200 | .0044 | 4096/4096 | .3887 |
| joint-stress / 2026081701 | 11 (J6) | J6 1.655 | .008/.153 | .0032 | 4096/4096 | .3887 |
| joint-stress / 2026081702 | 11 (J5:1, J6:10) | J6 1.350 | .019/.200 | .0029 | 4096/4096 | .3888 |

Counterfactually applying the other arm's strict vector to uniform yields 4,322/4,299/4,309 violating reads, concentrated at J2 and J6. Joint-stress has zero violations under uniform-0.9 and provisional project caps in all three populations; uniform has one/zero/one provisional-cap read. Raw Lambda therefore accompanies utilization in the bank: the cap denominator changes which channels bind. The machine-readable result includes raw per-joint Lambda p50/p95/p99/max, every cap geometry, utilization/margin/segment endpoint, responsible/all-active attribution, event/read and censoring descriptors, velocity tails, task output, VIC gains, and actions.

Associated physical-contact prefixes have median observed duration 28–30 ms for uniform and 20 ms for joint-stress; censored prefixes remain. Neither duration nor the pressure survival product is a unique physical-event dose.

## Interpretation and boundary

The pre-training log-only replay predicted dense joint-stress activation: 98.75–99.12% of initial stochastic episodes violated at least one boundary; J2 supplied 75.7–79.7% and J6 13.7–14.5% of responsible reads; impulse won 99.68–99.85% of active reads. After training, the residual tail is sparse and chiefly J6 for joint-stress, while uniform leaves only J3. This is geometry-sensitive behavior, not identified causal transfer: no paired p=0 policy exists at each geometry.

This one-seed project-cap experiment does not establish hard constraint satisfaction, actuator damage limits, manufacturer or hardware safety, an optimal cap vector/dose, individual-joint causality, cross-seed reliability, or complete-contact behavior. Soft-CaT is pressure, not a clamp. No new curriculum, domain randomization, cap/dose, evaluation, seed, torque constraint, contact flush, or hardware treatment was launched after this bank.

## Artifact ledger

The deterministic [analysis.json](assets/2026-08-19_z1_impulse_two_boundary_p02/analysis.json) has SHA-256 `b7a10a1c5d309aa2f3c215a3d5367b05ee01cd1a48045ce5186c28e61fc72cfa`. Its adjacent [`SHA256SUMS`](assets/2026-08-19_z1_impulse_two_boundary_p02/SHA256SUMS) binds the analysis, exact analyzer source, and this record.
