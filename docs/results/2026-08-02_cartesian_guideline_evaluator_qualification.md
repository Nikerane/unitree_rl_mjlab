# Cartesian guideline pilot evaluator qualification

**Date:** 2026-08-02

**Reviewed code revision:** `a54c7a1a212d4383afb1329ac1806399b7c3d14f`

**Branch:** `cartesian-guideline-fic`

**Outcome:** **PASS — local CPU evaluator pre-training gate only, after C1 repair**

This note supersedes its earlier qualification of revision `bf6fab9`. Final
whole-branch review found that revision's CSV-to-analysis boundary discarded
load-bearing identity and result columns. The corrected boundary at the
reviewed revision above is the only qualified one.

No guideline `model_499.pt` with clean accepted provenance existed locally.
Accordingly, this qualification covers evaluator plumbing and fail-closed
certification only. It does not authorize training, establish a learned-policy
result, or support a C-Gate-versus-C0 scientific claim.

## Qualified four-row contract

| Treatment | Registered task | Training seed | Checkpoint |
|---|---|---:|---|
| C0 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0` | 0 | `model_499.pt` |
| C0 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0` | 1 | `model_499.pt` |
| C-Gate | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate` | 0 | `model_499.pt` |
| C-Gate | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate` | 1 | `model_499.pt` |

Each row must retain and validate the evaluator's persisted identity rather
than reconstructing a lossy projection:

- literal task-to-treatment mapping and training seed 0 or 1;
- `model_499.pt`, with 64-hex actual and accepted checkpoint SHA-256 values
  equal per row;
- 64-hex accepted-manifest, campaign-config, treatment-config, reset,
  guideline-geometry, treatment-base, sampled-trace, and trace-artifact
  identities;
- 40-hex training/evaluation code and asset revisions, literal clean boolean
  dirty flags, and training asset revision equal to evaluation asset revision;
- one accepted manifest, training provenance, evaluation provenance, campaign
  config, reset identity, geometry identity, and treatment-base identity shared
  across all four rows;
- one treatment-config identity shared by the two seeds in each arm, with C0
  and C-Gate identities distinct, and unique sampled trace/artifact identities;
- the exact frozen RNG tuple: reset `2036072919`, observation `2046072933`, and
  action `2056072941`;
- exactly 256 environments, 2 completed episodes per environment, 512 sampled
  episodes, and the 4.0 s failure-mode horizon;
- finite primary episode-first q90 perpendicular error in `[0.0, 0.050]`, plus
  finite/range-valid all-six-gates, corridor-occupancy, backward-progress,
  gate-payout, success-rate, and qvel-sentinel results.

C0 still requires literal absence of `r_gate` and zero actual manager payout.
Each C-Gate seed requires `r_gate` at weight 8 and finite positive actual
manager payout. Both arms retain fixed reset `(0.0, 0.0)`, no wind-up, fixed
impedance, and log-only `imp_max_p=0.0`.

The `success_rate_sampled >= 0.25` continuation rule remains descriptive. With
two seeds per arm it cannot support a treatment-effect claim; a confirmatory
comparison still requires at least five seeds per arm.

## Final-fix verification

All Python commands used `PYTHONPATH=.` and
`/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python`.

- Strict RED: **25 failed**, all because the old direct or DictWriter boundary
  did not raise for the intended tamper/omission.
- Focused identity/CSV GREEN: **36 passed**, with **169 deselected**.
- Full focused evaluator and pure campaign suite: **205 passed**, exit 0.
- Documentation freshness/path suite: **2 passed**, exit 0.
- `git diff --check`: clean before commits.

The focused suite includes direct-row and actual evaluator-`FIELDNAMES`
DictWriter tests for frozen RNG drift, missing/non-boolean dirty flags,
missing/malformed hashes and revisions, wrong registered tasks, checkpoint SHA
mismatch, shared and per-arm config drift, missing/nonfinite/out-of-range q90,
episode-horizon drift, and invalid or duplicated trace identities.

The 35 warnings in the focused integration run are the existing third-party
`torch.jit.script` deprecation. They are not project warnings.

## Earlier runtime evidence and optional-suite limitation

The earlier qualification run exercised the real tracker/reward collector,
the broader guideline/config/environment set, mandatory reward/impulse tests,
reward phases A–M, and contact/reward setup scripts. C1 did not change runtime
collection or task configuration, but those commands were not rerun for this
final serialization-boundary repair; they remain historical supporting
evidence, not fresh verification for revision `a54c7a1`.

The optional full-repository pytest run was also not rerun. Its earlier attempt
was stopped after 14 minutes at 1,081 passed, one skipped, and 20 failures in
unchanged `test_fixed_reset_video_library.py`. Those failures traced to the
untracked, absent
`docs/results/assets/2026-07-29_lambda_feasibility_stage0/lambda_feasibility_stage0_inputs.json`
in the linked worktree. The unchanged file separately reproduced as 47 passed
and 20 failed. This incomplete optional run is non-authoritative and is not
presented as final-fix evidence.

## Constraint and compatibility audit

The C1 fix changes only the pure campaign validator/CSV loader and focused
tests. The complete evaluator feature remains persistence-only outside
`scripts/eval_impulse.py`; no task dynamics, reward definition, tracker, task
registration, action, gain, manufacturer cap, or enforcement source changed.
There is no evaluator call to `set_gains`. `IMP_J_LIMIT` remains exactly
`[1.640, 3.280, 1.640, 1.640, 1.640, 1.640] N·m·s`, and native guideline
configuration is still validated before evaluator mutations.

Legacy trace handling remains additive. The corrected loader retains exactly
the load-bearing guideline-pilot columns in a fixed typed schema and delegates
to the existing exact-four-row validator; it is not a generic manifest layer.

## Decision boundary

This corrected revision may be considered for owner-approved push/deploy and
the four exact PPO identities above. Stop here until the owner separately
authorizes those actions. Once final checkpoints exist, the full
256-environment, 512-episode sampled evaluation remains required with
unaltered checkpoint, manifest, provenance, configuration, and trace identity
gates.
