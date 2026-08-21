# Task 2 report — guarded horizontal-route training launcher

Date: 2026-08-21

Branch: `codex/z1-horizontal-trajectory-pilot`

Starting revision: `9b493545` (`feat(hammer): add horizontal route pilot tasks`)

## RED → GREEN evidence

The launcher seam was exercised through a temporary fake code repository, canonical
fake sibling asset repository, fake Vega Python executable, and real shell execution.
No production launcher existed when the first behavior test was run.

### Required missing-launcher RED

Command:

```text
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_vic_horizontal_routes_p02_train_launcher.py::test_horizontal_route_arms_run_only_the_two_frozen_serialized_configs
```

Result: **1 failed**. Bash exited `127` because
`scripts/slurm/vega_vic_horizontal_routes_p02_train.sbatch` did not exist. This was
the intended failure and proved the test was observing the new launcher seam.

After adding the launcher, the same test passed **1/1** in 6.06 seconds.

### Explicit schedule-identity RED

A later self-review identified that the six anneal stages froze step `6000` but did
not explicitly serialize its training-iteration meaning under the frozen 24-step
rollout. The arm test was extended first to require `r_imit_zero_iteration`.

Result before implementation: **1 failed** because the serialized schedule omitted
`r_imit_zero_iteration: 250`.

After deriving the value as `6000 / 24`, the arm test and both live registered-config
preflight cases passed **3/3** in 43.42 seconds. Persistent serializes a null zero
iteration because it has no imitation curriculum.

## Implemented launcher contract

New launcher:
`scripts/slurm/vega_vic_horizontal_routes_p02_train.sbatch`

- Array index `0` maps only to
  `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-HorizontalRoutes-Annealed`.
- Array index `1` maps only to
  `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-JointPosition-VariableImpedance-TT-HorizontalRoutes-Persistent`.
- Both arms freeze seed `2`, `4096` environments, 24 steps/environment, 500
  iterations, save interval 50, impulse-CaT `p=0.2`, and caps
  `[0.369,0.246,0.738,0.369,0.246,0.0164]` N.m.s.
- The serialized preflight binds the two exact task IDs, joint-position plus
  joint-stiffness actions, reset-time route-sign sampler identity, `0.020 m` route
  amplitude across the event/observation/imitation readers, active velocity and
  impulse CaT, the caps and dose, maximize-reward weights, imitation sigma, arm
  weight, exact anneal stages, and zero-weight iteration. It loads and compares the
  selected registered config twice, emits canonical JSON and its SHA-256, and fails
  closed on any mismatch.
- The annealed arm binds `r_imit=0.1`, `sigma=0.05`, stages ending at step `6000`,
  and zero weight at iteration `250`. The persistent arm binds `r_imit=0.2`,
  `sigma=0.05`, and no imitation curriculum.
- The launcher retains the qualified A100-SXM4-40GB and
  `mjlab 1.4.0`/`mujoco 3.8.1`/`mujoco-warp 3.8.1` runtime guard.
- Required code/asset revisions, clean trees, canonical sibling assets, exact array
  shape and indices, numeric Slurm IDs, no arguments, no inherited treatment,
  distributed, resume, or optimized-Python environment, and fresh attempt/node-temp
  leaves are fail-closed.
- Training output must contain exactly models
  `0,50,100,150,200,250,300,350,400,450,499`. Every checkpoint must carry its exact
  integer iteration, contain finite tensor state, and produce a lowercase SHA-256.
  Code and asset provenance are rechecked after training before hashes/pass output.
- The campaign, result root, attempt leaf, run name, node-temp path, and Slurm output
  names are isolated under `z1-vic-horizontal-routes-p02-train`.
- The Slurm log parent remains a pre-submission contract: the launcher names the
  fixed `.../campaigns/z1-vic-horizontal-routes-p02-train/slurm/` parent and never
  attempts to create it after the job begins.
- `#SBATCH --no-requeue` is present; no submission, requeue, retry, resume, or
  distributed training path exists.

## Verification evidence

### Focused launcher suite

Command:

```text
PYTEST_ADDOPTS='-p no:cacheprovider' \
  /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q \
  tests/test_vic_horizontal_routes_p02_train_launcher.py
```

Result: **55 passed, 0 failed** in 83.20 seconds.

The suite runs both successful arms, runs the serializer against both actual Task 1
registered configs, and mutates route amplitude, route-sign sampler, imitation
weight/schedule, impulse caps, velocity CaT, repeated-load identity, array/execution
environment, code/assets, runtime/GPU, collisions, checkpoint set/iteration/
finiteness, postflight provenance, and SHA-256 handling.

### Historical launcher regressions

The qualified historical launcher plus generic Slurm suites passed **155/155** in
97.54 seconds.

The final comprehensive command covering every `tests/test_*launcher.py` plus
`tests/test_slurm_launchers.py` passed **592/592** in 405.16 seconds. No historical
production file changed.

### Static, historical-preservation, and scope checks

- `bash -n` passed for the new launcher and the qualified source launcher.
- The qualified historical launcher remains byte-identical with SHA-256
  `055d4a431e9483642a4b81dca9acb163fab6a2bc286a9cb2b09adeb83caf9444`.
- `git diff --check` reported no whitespace errors for the reviewable files.
- The implementation scope is one new launcher, one focused launcher test, and this
  required report. No historical launcher was edited.
- No Vega access, push, submission, training, evaluator change, model artifact, or
  result/document banking was performed.

## Self-review and residual concerns

- The new launcher is intentionally a one-seed diagnostic pilot; its guards do not
  turn the resulting policies into a generalization or hard-safety claim.
- Local verification exercises the real registered config serializer and all shell
  guards with fake repositories/Python. Task 3 still owns the real A100 runtime,
  clean detached Vega worktree, `sbatch --test-only`, one-shot submission, and
  monitoring evidence.
- The Slurm output/error parent must be created before submission because Slurm
  opens those files before the job script can run.
