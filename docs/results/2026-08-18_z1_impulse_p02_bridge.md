# Z1 provisional-cap impulse-CaT `p=0.2` bridge

**Status:** banked one-training-seed simulation screen, 2026-08-18. **Verdict: bridge PASS**:
all three stochastic populations passed every preregistered gate separately. Relative to the
log-only control, the trained `p=0.2` target had lower native observed provisional-cap impulse risk
and tails, lower—not higher—true 500 Hz velocity-limit risk, unchanged success/productive-strike
rates, and about 99.5% retained first-event delivered impulse.

This is native observed exposure from one PPO training seed. It is not a hard clamp, not a
hardware-safety result, not evidence that `p=0.2` is optimal, and not proof across training seeds.
Zero observed target violations does not mean zero risk. The caps are project-defined simulation
thresholds, not manufacturer damage limits.

## Frozen protocol and provenance

- Treatment: one VIC-TT seed-2 target, 4,096 environments x 24 steps, 500 PPO iterations,
  provisional caps `[0.82, 1.64, 0.82, 0.82, 0.82, 0.82] N.m.s`, `imp_max_p=0.2`. Velocity CaT and
  every other treatment choice were unchanged.
- Training job `41461595`: `COMPLETED`, `0:0`, 20:58, empty stderr, one A100, no retry/requeue;
  code revision `8ff0ccfb577d2218638a25170a11023255a78b9d`, asset revision
  `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Log-only control checkpoint SHA-256:
  `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3`.
- `p=0.2` target checkpoint SHA-256:
  `57000e958bbafa2c62929652d3b76fd6ed571c9867bee3735c14baf0ca57d8de`.
- Frozen evaluation array `41465643_[0-1]`: both arms `COMPLETED`, `0:0`, 1:55, empty stderr,
  one A100 each, no retry/requeue. Evaluator code revision
  `fa0c969ec2846ee6bbb8bd43558dd42eaf4c153b`; asset revision unchanged.
- Evaluation ran both policies with live `imp_max_p=0`: `delta_impulse` was identically zero and
  combined delta was the exact `max(delta_velocity, delta_impulse)`. The fixed 64-world population
  is descriptive only. Stochastic seeds `2`, `2026081701`, `2026081702` each contain 4,096 matched
  worlds. Whole paired environment IDs are the inferential unit; 50 Hz reads are not.
- Each paired interval uses exactly 10,000 bootstrap resamples and seed `2026081703`. No stochastic
  population is pooled for a verdict.

## Preregistered result

`rho = max_t,j Lambda[t,j]/cap[j]`. Risks are fractions of initial episodes. RD is target minus
control in percentage points; bracketed values are the paired 95% interval. The delivered ratio
column gives point estimate and lower 97.5% bound.

| Population | Real-cap binding C to T | Impulse RD, pp [95%] | rho p95 C to T | rho p99 C to T | Velocity risk C to T | Velocity RD, pp [95%] | Delivered ratio point / lower | Utility diff S/P | Gate verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed-64 | 0/64 to 0/64 | descriptive | 0.7972 to 0.4207 | 0.8373 to 0.4213 | descriptive | descriptive | descriptive | descriptive | not inferential |
| `2` | 35/4096 to 0/4096 | -0.8545 [-1.1475, -0.5859] | 0.8067 to 0.5158 | 0.9735 to 0.6434 | 0.5127% to 0.1709% | -0.3418 [-0.5859, -0.1221] | 0.99540 / 0.99483 | 0 / 0 | PASS |
| `2026081701` | 31/4096 to 0/4096 | -0.7568 [-1.0254, -0.5127] | 0.8071 to 0.4578 | 0.9174 to 0.6363 | 0.6836% to 0.1953% | -0.4883 [-0.7568, -0.2441] | 0.99643 / 0.99571 | 0 / 0 | PASS |
| `2026081702` | 35/4096 to 0/4096 | -0.8545 [-1.1475, -0.5859] | 0.8068 to 0.5208 | 0.9870 to 0.6335 | 0.6592% to 0.2441% | -0.4150 [-0.6836, -0.1465] | 0.99550 / 0.99498 | 0 / 0 | PASS |

Every qualifying joint p99 decreased and no new joint became violating. Per-joint provisional-cap
p99 utilization vectors (J1-J6) were:

| Population | Control p99 utilization | Target p99 utilization |
|---|---|---|
| fixed-64 | `[0.5401, 0.2087, 0.8373, 0.4697, 0.3432, 0.0190]` | `[0.3443, 0.1442, 0.4213, 0.2633, 0.2288, 0.0169]` |
| `2` | `[0.5416, 0.2224, 0.9735, 0.5294, 0.3456, 0.0284]` | `[0.4538, 0.1929, 0.6434, 0.3779, 0.2994, 0.0262]` |
| `2026081701` | `[0.5448, 0.2218, 0.9174, 0.5102, 0.3493, 0.0303]` | `[0.4595, 0.1927, 0.6363, 0.3775, 0.3039, 0.0268]` |
| `2026081702` | `[0.5364, 0.2256, 0.9870, 0.5371, 0.3446, 0.0293]` | `[0.4525, 0.1898, 0.6335, 0.3735, 0.2996, 0.0261]` |

## Duration, contact prefixes, reads, and learned behavior

Native median episode duration was 140 ms in both arms for fixed and all stochastic populations;
there is no shorter-episode explanation for this bridge direction. Physical-contact prefixes were
still overwhelmingly task-terminal and right-censored: control `4036/4103`, `4049/4112`,
`4036/4103`; target `3888/4097`, `3894/4098`, `3890/4096`. Therefore this remains a native-prefix
claim, not a complete-contact claim.

Control cap-violating reads were `35/31/35`; target had `0/0/0`. Every control activation window
had one associated 50 Hz read. Observed first-contact-prefix global Lambda p99 decreased
`0.7983 to 0.5276`, `0.7438 to 0.5218`, and `0.8090 to 0.5195 N.m.s`. Fixed-64 likewise had no
cap-active read.

The predeclared no-learning preflight had already established separate pressure attribution on the
control behavior: at offline `p=0.2`, active reads were `35/30/35`, impulse won `35/30/35`, velocity
masked `0/0/0`, ties were `0/0/0`, and combined delta was always the exact max. Each contiguous
activation window had one 50 Hz read. Candidate activation-window pressure median/p95/max was
`0.0393/0.1558/0.2`, `0.0633/0.1851/0.2`, and `0.0769/0.1916/0.2`. These are observed activation-
window prefixes, not unique complete-event doses.

All six precontact and at-contact VIC-gain coordinates and all 12 native action coordinates are
retained in `analysis.json`. Descriptively, at-contact median normalized stiffness was identical in
all stochastic populations: `[-1, -1, +1, +1, +1, -1]` in both arms. The target precontact J5
median was about `0.598-0.657` versus control `+1`; the other median gain coordinates matched.
Action medians differed materially in several coordinates and commonly saturated at `+/-1`, as
expected for distinct learned policies; no action-equality claim is made.

## Exact artifact ledger

| Arm | Artifact | SHA-256 |
|---|---|---|
| control | `fixed_trace.npz` | `a4c2cf530707feeddfc345d434bed147a8e6c711428475e631570be62af2a524` |
| control | `training_like_seed_2_trace.npz` | `fa354e9f01f027f5e6aaa21959348e8b674ed88b98e5c2c9f523acd2b059c957` |
| control | `training_like_seed_2026081701_trace.npz` | `0a755298a63855c3cab456f1c5e349cd34b658c1623d50143275dfca92027bd8` |
| control | `training_like_seed_2026081702_trace.npz` | `03a6fca3656bc585420497b109c331cf2f37e6477c2268c406cc5b9956db2dd0` |
| control | `summary.json` | `906ee66f78beedb817f5518c5c3364d9a93d30e35b5ee582ccc20a83112ccce7` |
| target | `fixed_trace.npz` | `45c8cbb105ad8013a80488337a6adb928b7d73a7c6ebeebd2eb3a7b08e8e2003` |
| target | `training_like_seed_2_trace.npz` | `9b816d22b5ef16d9d336c16ab5d8a5318c06a280d709b3716a3d98558373a03f` |
| target | `training_like_seed_2026081701_trace.npz` | `4a5465c756ba2b1d44eb6947c91ed981b75dc7b8fe9208558a396c94501b47d7` |
| target | `training_like_seed_2026081702_trace.npz` | `63a0e141bf6006ad7a1e12c170d1ec0e0a867f62a94e21e24041f3fc9e075ad0` |
| target | `summary.json` | `333ded1c094cdb2a993e1dbf99dc52f6c83f11b823ee40efdba345eeaf77d8b1` |

The compact sufficient statistics are in
`docs/results/assets/2026-08-18_z1_impulse_p02_bridge/analysis.json`; the adjacent `SHA256SUMS`
binds it, the deterministic analyzer, and the exact sanitized review packet. Raw traces and
checkpoints remain outside Git.

- Compact analysis SHA-256: `90c17761a73ff5b06598e86056109b6b6e8c9ec1c393e4edc3b2135c4d62abdd`.
- Analyzer SHA-256: `0d33a835bc6bb6b816ddfd39e903767616f3b3837b2303b5f16a447070624655`.
- Sanitized packet SHA-256: `fd8fc307bdef15ce17b6ebe157776083faba1a291422becbccca958ee1b72cb5`.

## Decision

The result is **binding, independently active, and graded** on the surveyed control behavior, and
the `p=0.2` learned-policy bridge passed the native impulse, velocity, joint-tail, utility,
delivered-impulse, identity, and finiteness screens. That supports proposing matched independent
training-seed confirmation at the same fixed treatment—not raising the dose or changing the caps.
No confirmation run, another dose, contact flush, torque constraint, or rescue experiment is
authorized by this record.
