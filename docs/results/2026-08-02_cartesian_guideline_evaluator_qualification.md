# Cartesian guideline pilot evaluator qualification

**Date:** 2026-08-02

**Reviewed code revision:** `bf6fab930014cbc0357dcf44e96698a122d3e22f`

**Branch:** `cartesian-guideline-fic`

**Outcome:** **PASS — local CPU evaluator pre-training gate only**

The minimal evaluator is qualified to evaluate the excluded four-row
fixed-reset Cartesian-guideline pilot after its final checkpoints exist. This
does not authorize training, establish a C-Gate-versus-C0 result, or qualify a
learned policy: no guideline `model_499.pt` with clean provenance existed
locally, so no checkpoint rollout was invented and the required checkpoint
identity was not relaxed.

## Qualified contract

The evaluator accepts exactly these future final-checkpoint rows:

| Treatment | Registered task | Training seed | Checkpoint |
|---|---|---:|---|
| C0 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0` | 0 | `model_499.pt` |
| C0 | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-C0` | 1 | `model_499.pt` |
| C-Gate | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate` | 0 | `model_499.pt` |
| C-Gate | `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CGate` | 1 | `model_499.pt` |

All rows require the explicit `cartesian-guideline-pilot` campaign, final
checkpoint and manifest hashes, clean code and asset provenance, the frozen
evaluation RNG tuple, exactly 512 sampled episodes, zero physical-impossibility
and dead-Lambda sentinels, and zero nonfinite-qvel rate. C0 requires literal
absence of `r_gate` and zero stored gate payout. Each C-Gate seed independently
requires a finite positive total of the actual weighted manager payout.

The `success_rate_sampled >= 0.25` rule is only a descriptive decision for
whether each arm may continue beyond the excluded pilot. With two seeds per arm
it supports no C-Gate-versus-C0 scientific claim; a confirmatory comparison
still requires at least five seeds per arm.

## Fresh verification

All commands used `PYTHONPATH=.` and
`/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python`.

- Focused evaluator and pure campaign reducer: **175 passed**, exit 0.
- All six additional test files discovered by
  `rg -l "guideline|Guideline|GUIDELINE" tests/test_*.py` after excluding the
  focused pair — configs, direct-reference scientific gates, environment,
  guideline qualification, guideline mechanics, and first-strike
  instrumentation: **447 passed**, exit 0.
- Mandatory reward/impulse set
  (`impact_progress`, impulse bound/constraint, delivered impulse, soft-CaT):
  **122 passed**, exit 0.
- `validate_rewards.py`: phases **A through M all passed**, exit 0. Phase M
  retained `imp_max_p=0` with zero CaT delta and live substep impulse/delivered
  signals.
- `verify_contact_sensor.py`: exit 0; all four environments detected first
  contact at step 5.
- `verify_reward_setup.py`: exit 0; all four non-exempt terms fired. The random
  sweep made no contact in 200 steps, so its optional contact-threshold tuning
  was skipped.
- Explicit no-checkpoint fixture smoke: **7 passed**, exit 0. It covered both
  one-environment C0/C-Gate task executions, all six ordered gates before
  accepted contact, positive finite C-Gate manager payout, real tracker/reward
  collector capture, the typed CSV bridge, and exact four-row reduction.
- `git diff --check`: clean before documentation.

The recurring warnings were the existing TorchScript deprecation, Matplotlib
and font-cache fallbacks, and pytest's inability to create `.pytest_cache` in
the externally managed worktree. They did not change command exit status.

An optional full-repository pytest run was stopped after 14 minutes because it
was disproportionate to this gate. At interruption it reported 1,081 passed,
one skipped, and 20 failures, all in `test_fixed_reset_video_library.py`. The
unchanged file reproduced as 47 passed and 20 failed. Every failure traced to
the absent
`docs/results/assets/2026-07-29_lambda_feasibility_stage0/lambda_feasibility_stage0_inputs.json`:
the artifact is not tracked by Git and therefore is absent from this linked
worktree, although an untracked local copy exists in the main checkout. The
video-library source and test already depended on that path at revisions
`30570dc` and `dfc3c88` and were not changed by this evaluator work. No artifact
was copied and no unrelated source or test was changed. This incomplete optional
run is non-authoritative; the explicit task gates above are the qualification
evidence.

## Constraint and compatibility audit

The implementation delta from design revision `dfc3c88` through the reviewed
revision touches only:

- `scripts/eval_impulse.py`;
- `evaluation/analysis/guideline_campaign.py`;
- their two focused test files.

No task dynamics, reward definition, tracker, task registration, action,
actuator gain, manufacturer cap, or enforcement source was changed. There is
no evaluator call to `set_gains`; the native validator pins the fixed actuator
and action signatures. `IMP_J_LIMIT` remains exactly
`[1.640, 3.280, 1.640, 1.640, 1.640, 1.640] N·m·s`, and guideline evaluation
requires the imported and configured limits to match it. Native guideline
configuration is validated before evaluator mutations, `imp_max_p` remains
literal zero, and the physical reset remains `(0.0, 0.0)` with no wind-up.
C0 and C-Gate share a canonical base-configuration digest after removing only
`r_gate`; C-Gate pins `ordered_gate_progress_reward` at weight 8.

Legacy traces remain additive and unchanged: the guideline digest/schema is
activated only for registered guideline traces, and the focused suite retained
the frozen legacy campaign and treatment digest literals.

## Decision boundary

This revision may be considered for owner-approved push/deploy and the exact
four PPO identities above. Stop here until the owner separately authorizes
those actions. After checkpoints exist, the direct CPU evaluator smoke and the
full 256-environment, 512-episode sampled evaluation remain required with
unaltered final-checkpoint and provenance gates.
