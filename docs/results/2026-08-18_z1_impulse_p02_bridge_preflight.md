# Z1 impulse-CaT `p=0.2` bridge preflight

**Status:** banked zero-learning preflight, 2026-08-18 — **PASS**. At the provisional project caps,
the frozen seed-2 control has sparse but repeatable stochastic violations; offline `imp_max_p=0.2`
is finite, graded, and wins the max soft-OR on every violating read. The historical `p=0.5` target
also has lower native and observed-first-contact-prefix p95/p99 exposure in every population, so the
predeclared exposure-direction gate passes.

This is permission to continue the approved bridge workflow, not a learned-policy result. No new
simulation, training, optimizer update, or dose sweep was performed. The caps remain project-defined
simulation thresholds, not manufacturer limits; soft CaT is training pressure, not a runtime clamp.

## Evidence

Caps were `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N.m.s`. Pressure was replayed from the raw
control Lambda in its native cap-subtraction dtype and checked against the stored candidate. Live
evaluation remained log-only (`delta_impulse == 0`); the counterfactual combined pressure is exactly
`max(delta_velocity, delta_impulse)` in every population.

| Population | Active reads / distinct positive doses | Responsible joint | Reads per activation window | Winner reads I/V/tie | `P_window` p50/p95/max | Cap-associated duration p50/p95/max, ms | Physical contacts censored | Native rho p95/p99/max |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fixed mean, 64 | 0 / 0 | — | — | 0/0/0 | — | — | 64/64 | 0.797/0.837/0.853 |
| stochastic `2`, 4,096 | 35 / 35 | J3 | 1 each | 35/0/0 | 0.039/0.156/0.200 | 28/32/34 | 4,036/4,103 | 0.807/0.974/1.300 |
| stochastic `2026081701`, 4,096 | 30 / 29 | J3 | 1 each | 30/0/0 | 0.063/0.185/0.200 | 26/35/36 | 4,050/4,112 | 0.807/0.908/1.239 |
| stochastic `2026081702`, 4,096 | 35 / 35 | J3 | 1 each | 35/0/0 | 0.077/0.192/0.200 | 26/32/34 | 4,036/4,103 | 0.807/0.987/1.198 |

`P_window = 1 - product_t(1 - delta_impulse_t)` is reported for contiguous 50 Hz activation
windows. Every activating window here contains one controller read. It is not a unique physical-event
dose: one read in seed `2026081701` and two reads in seed `2026081702` have ambiguous multi-event
associations, and 98.37–98.49% of all observed physical contacts are terminal-censored.

All stochastic violations are at J3. The table below retains the per-joint cap-utilization context;
ranges are across the three stochastic populations.

| Joint | Cap, N.m.s | Fixed median / max utilization | Stochastic median range | Stochastic max range | Violating reads by stochastic population |
|---|---:|---:|---:|---:|---:|
| J1 | 0.820 | 0.473 / 0.546 | 0.466–0.466 | 0.674–0.842 | 0 / 0 / 0 |
| J2 | 1.640 | 0.181 / 0.210 | 0.179–0.179 | 0.290–0.319 | 0 / 0 / 0 |
| J3 | 0.820 | 0.771 / 0.853 | 0.753–0.755 | 1.198–1.300 | 35 / 30 / 35 |
| J4 | 0.820 | 0.457 / 0.473 | 0.429–0.430 | 0.639–0.706 | 0 / 0 / 0 |
| J5 | 0.820 | 0.305 / 0.346 | 0.301–0.301 | 0.434–0.519 | 0 / 0 / 0 |
| J6 | 0.820 | 0.017 / 0.019 | 0.018–0.018 | 0.062–0.070 | 0 / 0 / 0 |

The fixed population remains descriptively nonbinding: **“impulse constraint is empirically
nonbinding under the surveyed population.”** That does not override the 100 active stochastic reads.

## Exposure-direction gate

The following values are historical `p=0.5` target minus `p=0` control at the provisional caps.
Every required p95/p99 difference is strictly negative.

| Population | Native rho p95 / p99 difference | Observed-prefix rho p95 / p99 difference |
|---|---:|---:|
| fixed mean | −0.2904 / −0.2149 | −0.2904 / −0.2149 |
| `2` | −0.1639 / −0.2659 | −0.1639 / −0.2659 |
| `2026081701` | −0.1586 / −0.1945 | −0.1586 / −0.1853 |
| `2026081702` | −0.1689 / −0.2856 | −0.1688 / −0.2851 |

This same-direction sensitivity does not remove the unequal-horizon/contact-censoring limitation and
does not predict the velocity behavior of a policy trained at `p=0.2`.

## Frozen provenance and verification

- Source leaves: Vega job `41386821_[0-1]`, evaluator revision
  `142098b8af4cbd4b52777012b27f477f0e9aa358`, asset revision
  `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Control checkpoint SHA-256:
  `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3`; historical target:
  `ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15`.
- Copied seed-2 trace SHA-256 values independently matched
  `a2666a6f9cde2f5607f8d7bc8cde696578f3aab65247166d20b6c530359c1547` and
  `62967852edd9e3a3abaa3dbf9de5b77faae1e0cc4854e9b41ead92c010c6b54d`; both five-row manifests
  rehashed cleanly.
- Validator commit: `e0161adfa00012264e53e2f2dc85f6b6b91be598`. The generated compact JSON SHA-256 is
  `e006358c385aafbeb2b5478512c8c2d62ac71e2024eafdd33fa07c7bc668a3fc` and is stored under
  `docs/results/assets/2026-08-18_z1_impulse_p02_bridge_preflight/` with its checksum manifest.
- Independent post-generation checks confirmed JSON finiteness, exact `35/30/35` active reads,
  `35/29/35` distinct doses, `35/30/35` impulse winners, zero masks/ties, exact max combination,
  and all sixteen direction differences below zero.

## Verdict

The real project caps are nonbinding in the fixed-64 regression but binding in every representative
stochastic population. At `p=0.2`, impulse pressure is independently active and graded rather than
masked by velocity. The zero-learning gate for the exact one-target bridge therefore passes. This
does not select `0.2` as optimal and does not establish confinement, learned safety, or hardware
safety; those questions require the separately guarded training and frozen evaluation stages.
