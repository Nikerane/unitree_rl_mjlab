# First-strike reward-package CPU qualification

**Date:** 2026-07-25
**Scope:** Z1 fixed impedance, log-only impulse-CaT (`imp_max_p=0`)
**Qualified commit:** `5d986539d5ef7c9503df378467e3c5bf5c6875c4`
**Outcome:** **PASS — CPU pre-training gate only**

The production-faithful C/D-prime/F/E reward packages passed the frozen
force-bearing bank and the repository's full CPU reward/impulse pre-training
gate. This qualifies their implementation, event semantics, payout streams and
CPU simulator setup for the next clean CUDA instrumentation smoke. It does
**not** predict PPO ranking, establish that E will outperform D-prime, or
authorize the 4×8 training matrix.

## Package definitions and causal boundaries

All arms retain the configured maximize weights `impact=8` and `delivered=2`.

- **C:** shipped repeated-credit legacy readers and legacy normalizer.
- **D-prime:** the unchanged 50 Hz legacy readers, censored to the shared first
  event and paid once on its finalization boundary.
- **F:** shared 500 Hz first-event measurement with linear event impulse payout.
- **E:** the same event measurement as F, with delivered payout saturated at
  the event reference.

The comparisons are treatment packages:

- C versus D-prime changes repeated credit, censoring and payout timing.
- D-prime versus F changes the legacy measurement/gating/normalizer package to
  the 500 Hz first-event package.
- F versus E is the isolated linear-versus-saturated payout contrast.

Only F versus E is an atomic saturation comparison. The bank diagnostics below
must not be described as dose equivalence or expected policy performance.

## Frozen-bank qualification

The final schema-v5 replay completed 384/384 episodes across 12 complete
checkpoint groups and 32 paired seeds. It passed all 14 ordered gates, all 12
stochastic proofs, the 10/10 phase-invariance gate, physical equality across
reward arms, D-prime inner-reader fidelity, shared D-prime/F/E finalization
timing, at most one positive payout per component on the common finalization
boundary, and zero delayed payout. Zero is allowed: D-prime impact was positive
in 364/384 episodes and D-prime delivered was positive in 384/384. The
aggregate-only summary reports `valid=true`.

Artifact provenance:

| Artifact | Schema | Bytes | SHA-256 |
|---|---:|---:|---|
| `bank_manifest.json` | 3 | 27,923 | `69d5bc66f2127d0a5b29463b35763092d388bc0e71661b06fdd81b0eea8d661f` |
| `probe_raw.npz` | 5 | 4,009,467 | `ea4a82e007d95cf7ff962091ba0d6ff3e07e639f4cc1c2d0feab9fe368f47ac8` |
| `probe_summary.json` | 4 | 6,543 | `641e520c0cd35932175918d0bf48c7b6df07749ff1462b3d7da7310d76338b1b` |

The large raw NPZ and generated JSON artifacts remain local evidence and were
not added to this documentation commit.

### Return diagnostics

These are production reward-manager returns on the frozen physical bank, not
training predictions.

| Arm | Discounted | Undiscounted | Positive impact payouts | Positive delivered payouts |
|---|---:|---:|---:|---:|
| C | 0.3659260012 | 0.3893165645 | 364 | 384 |
| D-prime | 0.2194305203 | 0.2345366787 | 364 | 384 |
| F | 0.2368241319 | 0.2529904535 | 384 | 384 |
| E | 0.2355000119 | 0.2515744455 | 384 | 384 |

Per-stratum discounted / undiscounted returns:

| Stratum | C | D-prime | F | E |
|---|---:|---:|---:|---:|
| `dc_delonly` | 0.3481401990 / 0.3686995136 | 0.2311346482 / 0.2464265369 | 0.2351285162 / 0.2505869365 | 0.2337989300 / 0.2491709591 |
| `dc_imponly` | 0.4650784571 / 0.4942209462 | 0.2277864938 / 0.2439231163 | 0.2294580184 / 0.2456227849 | 0.2279346334 / 0.2439889282 |
| `mx_maxmax` | 0.4200421898 / 0.4514011756 | 0.2208939342 / 0.2378269872 | 0.2328150368 / 0.2506624204 | 0.2312248174 / 0.2489533093 |
| `mx_maxoff` | 0.2304431589 / 0.2429446227 | 0.1979070048 / 0.2099700745 | 0.2498949560 / 0.2650896720 | 0.2490416668 / 0.2641845851 |

### Normalizers

- Legacy configured reference: `0.6094 N·s`.
- Legacy reproduction: `0.6093726754188538 N·s`, within the frozen 1% gate.
- Event configured reference: `0.3088 N·s`.
- Eight repeated event derivations: mean
  `0.30883467197418213 N·s`, zero standard deviation and zero range.
- Event derivation used `min_windup_clearance=0.05`, 10/10 phase detections,
  12 stochastic proofs and zero delayed payout.

### Preserved failed D comparator

The original dose-matched D bank remains preserved under
`failed_dose_match_v2/` as negative evidence. It contained two distinct
findings:

1. the original legacy-reference derivation used the wrong horizon and observed
   `0.742736 N·s` instead of the shipped `0.6094 N·s`;
2. after separating that defect, a single dose multiplier still failed to
   transport: held-out discounted-return error was 9.17% overall, exceeded 10%
   in every stratum, and reached 43.67% in `mx_maxoff`.

The preserved artifact hashes are:

- manifest: `5796559eb9679361206a7b485d703eca2e3be516226113d1004af0e899bb62fd`;
- raw: `e910c7e8136626e23b68f0c55bff53857c217ff55fc077723c19fdf75f00e1cb`;
- summary: `7147bb2564d40641a7c609c5fedfbe9b9056ef3c8cb70df26d3af8c612852e91`.

This failed arm is not eligible for confirmatory training and must not be
retuned against the validation bank.

## Full CPU pre-training gate

An initial provenance preflight used the stale repository-root paths recorded
in the comparator plan:

```bash
shasum -a 256 validate_rewards.py verify_contact_sensor.py verify_reward_setup.py
```

It exited nonzero because those files do not exist at the repository root. No
gate had run and no source was patched. `docs/README.md` and `rg --files`
identified the authoritative scripts under
`docs/research/reward-design/`. After explicit approval, the gates were run
from commit `5d986539d5ef7c9503df378467e3c5bf5c6875c4` with those paths, and the
execution plan was corrected.

### Reward/impulse pytest set

```bash
/usr/bin/time -p env \
  PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=. \
  MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_first_strike_event.py \
  tests/test_first_strike_probe.py \
  tests/test_impact_progress_reward.py \
  tests/test_impulse_bound.py \
  tests/test_impulse_constraint.py \
  tests/test_delivered_impulse_reward.py \
  tests/test_cat_soft_hook.py \
  tests/test_configs.py
```

Exit `0`; 237 passed with 35 `torch.jit` deprecation warnings. Pytest time
5.03 s; `real 6.45`, `user 4.65`, `sys 0.58` seconds.

### Reward validation A–M

```bash
/usr/bin/time -p env \
  PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=. \
  MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py
```

Exit `0`; phases A–M passed. `real 8.31`, `user 6.43`, `sys 0.67` seconds.
The live baseline weights read dynamically from the manager were
`approach=0.1`, `nail_driven=0.5`, `nail_depth_delta=600`,
`impact_progress=8`, `completion=100`, `action_rate=-0.01`,
`joint_pos_limits=-10`, and `r_imit=0.1`. Phase I observed peak
`impact_progress=10.8980`. Phase M observed raw peak per-joint Λ
`[0.000, 0.185, 0.154, 0.118, 0.000, 0.000]`, zero soft-CaT delta under
`imp_max_p=0`, peak delivered reward `0.7803`, and a passing sign-aware Track-2
bound.

### Contact sensor validation

```bash
/usr/bin/time -p env \
  PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=. \
  MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
```

Exit `0`; `real 5.36`, `user 4.68`, `sys 0.45` seconds. Exactly one
`hammer_head_0` primary contact sensor was active with shape `(4, 1)`. All four
environments first contacted at step 5 with force `10.947008 N`.

### Random-policy reward setup validation

```bash
/usr/bin/time -p env \
  PATH=/Users/nikerane/miniconda3/envs/unitree_mjlab/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=. \
  MPLCONFIGDIR=/private/tmp/unitree_mpl_cache \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_reward_setup.py
```

Exit `0`; `real 43.17`, `user 42.31`, `sys 0.73` seconds. All five
non-exempt reward terms fired; completion and impact-progress silence was
expected under the random policy. The sweep produced five contact events,
mean head speed `1.7289 m/s` at step 5, air-time percentiles
`[0.118, 0.280, 0.360, 1.462, 1.806] s`, and impact-speed percentiles
`[0.484, 0.566, 1.140, 1.219, 1.265] m/s`. The script reported the setup safe
to begin training.

## Proven and open

Proven at the qualified commit:

- shared first-event state, censoring and one-shot payout semantics;
- production reward-manager stream capture and cross-arm physical equality;
- D-prime legacy-reader fidelity and D-prime/F/E payout-boundary equality;
- bank population, stochastic, phase, normalizer, digest and schema gates;
- full CPU reward/impulse pytest, reward validation, contact sensor and
  random-policy setup gates.

Still open:

- the preregistered primary hypothesis E versus D-prime and all PPO rankings;
- training-seed statistics, success guardrails and mechanism interpretation;
- `impossible_success_n==0`: structural/config tests pass, but this sampled
  evaluator sentinel was not produced by the CPU gate;
- `lambda_dead_n==0`: Phase M correctly shows zero soft-CaT delta in the
  log-only configuration, but this training liveness sentinel was not produced;
- the 500 Hz hardware-speed acceptance gate: the bank records joint speeds,
  but Task 3 has no approved acceptance-threshold evaluator;
- CUDA instrumentation, clean Vega provenance, hardware transfer and variable
  impedance.

The three campaign invalidation sentinels therefore remain **OPEN** until the
approved clean CUDA/sampled-evaluator smoke. Task 4 and Vega were not started.

## Source and environment provenance

Relevant-source tree: 59 files, SHA-256
`fa124e2629f71b43f1dc9b1bf340245a7b125342beac4d1cc0ba58171cec7ab3`.
Task configuration digest:
`42025e7983cc3d6d6194a1438c4532001d6465917ba3a6e1657f258bb3862543`.
The paired checkpoint-assets repository was at
`b58ccd2f81fd246f27c1e8d88cf86484cd888703`. Both repositories contained
unrelated dirty files; the gate binds the complete relevant-source set and each
checkpoint/action/trace tuple by recomputed digest.

| File | SHA-256 |
|---|---|
| `docs/results/assets/2026-07-25_first_strike_reward/probe_first_strike.py` | `084aff39a28dc2a0f6d3d9398b2a976685f06fb0e842e7ef5e8fcb42a258d8a7` |
| `src/tasks/hammer/config/z1/__init__.py` | `a25d0b98259ae8313e31c419c1f56f63d1f5042e501f58305c30ee81d7eb47ac` |
| `src/tasks/hammer/config/z1/env_cfgs.py` | `a6d9966b2d6a69e52c819c1902f21028de21aa0ef6420d17e1be60d37c728880` |
| `src/tasks/hammer/mdp/first_strike.py` | `7aeb3bb7b57959c293707eb137f91e8ab86c94b27be446cec7f263b4b6c77c86` |
| `src/tasks/hammer/mdp/rewards.py` | `28fc4b6a4daa58a69b3d69f747ad24f2b88935f36d41c7bd0d0597e3ad9372eb` |
| `docs/research/reward-design/validate_rewards.py` | `d96a2e87a186b8652ada44ca10e63f827a32c2e584213e078452b6b31fb299da` |
| `docs/research/reward-design/verify_contact_sensor.py` | `15d0df02638b6660fc68d74cfba9c42be0cdf918eb5515107e16cf5324774979` |
| `docs/research/reward-design/verify_reward_setup.py` | `746159284cebd182c2647425a4f02939dbbab424d88d1b2f3c9c4dcc2bb392c1` |

Test-file hashes:

| File | SHA-256 |
|---|---|
| `tests/test_first_strike_event.py` | `dc05395f2568d971c0e882bfa19241df63886badbf101e4726702faf273d2549` |
| `tests/test_first_strike_probe.py` | `bba21bd55ba3082e3c6f469326e9f343943697ac78099bd78ae2850da6d6f06d` |
| `tests/test_impact_progress_reward.py` | `f3d115e29faf41b90253ca2bf92ee17c44acf87e3e03734f65a340ddc7a47bd1` |
| `tests/test_impulse_bound.py` | `28331ac9b3b731216bbcc88b7e1f10184e21521a79589a38563e210b6cefc99f` |
| `tests/test_impulse_constraint.py` | `276ea0a576f05389f8dbec37f25faa950fbd2f973f441bf43bc5814b8f090fef` |
| `tests/test_delivered_impulse_reward.py` | `5634d62cf87882ca6459e7ed2240288a5d6492559912079565ca6003ceedeff9` |
| `tests/test_cat_soft_hook.py` | `764b5e43dcff590d89778f2bffa23d8fed2fabaaab98d852ff2131703e8aa9b3` |
| `tests/test_configs.py` | `caf1612de71cb47a552c536b4c384d11242aad190ff8c2467c5d35da47ab4e0e` |

Pinned environment: Python 3.10.20, PyTorch 2.12.0, NumPy 2.2.6, mjlab
1.4.0, MuJoCo 3.8.1, mujoco-warp 3.8.1 and rsl_rl 5.2.0.
