# Task 6 report: guarded frozen-policy Vega evaluation launcher

Status: complete

Commit message: `feat(hammer): launch matched impulse policy evaluation`

## Implementation

- Added `scripts/slurm/vega_vic_impulse_diag90_500_eval.sbatch`, a two-arm, no-learning Slurm
  launcher with the exact `0 -> diag90_control` and `1 -> diag90_target` role mapping.
- The launcher requires explicit `RUN_ROOT`, `ASSET_REPO`, `EXPECTED_CODE_REVISION`,
  `EXPECTED_ASSET_REVISION`, `CONTROL_CHECKPOINT`, and `TARGET_CHECKPOINT`. It refuses missing
  variables, malformed revisions, array/parallel-shape drift, arguments, optimized Python, dirty
  repositories, noncanonical assets, unpinned asset revisions, unapproved evaluator lineage, and
  runtime/GPU drift.
- Both checkpoint variables are required before role selection. The selected checkpoint must be a
  regular, non-symlink `model_499.pt`; it is bound to the immutable role SHA-256 before evaluation
  and rehashed after evaluation.
- Each array task atomically claims a unique outer attempt leaf. It passes its fresh, nonexistent
  inner `evaluation/` child to `run_policy_evaluation` through the existing evaluator CLI, which
  preserves the evaluator's overwrite refusal. There is no retry or requeue path.
- The evaluator child must contain exactly five regular artifacts:
  `fixed_trace.npz`, the three prespecified stochastic trace files, and `summary.json`. A fixed
  order, relative-path `SHA256SUMS` manifest is written in the outer attempt leaf.
- The job does not create the configured Slurm-log parent; Task 7 retains that pre-submit concern.

## RED-GREEN evidence

### RED

Command:

```bash
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_vic_impulse_diag90_500_eval_launcher.py
```

Result before the launcher existed: `5 failed in 0.76s`. The behavioral fixtures all failed with
`bash: .../vega_vic_impulse_diag90_500_eval.sbatch: No such file or directory`; the static
resource guard likewise failed when attempting to read the absent launcher. This was the intended
missing-production-artifact failure.

### GREEN

Command:

```bash
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_vic_impulse_diag90_500_eval_launcher.py
```

Final result: `6 passed in 4.52s`.

The fake-interpreter exercise is part of the focused suite. It runs each arm through the real Bash
launcher, verifies the fake evaluator receives the exact role, immutable-role checkpoint, fresh
inner output path, and `cuda:0`, then creates the five expected artifacts. The test independently
recomputes the ordered relative-path SHA manifest. Negative fixtures prove collision refusal,
dirty-code refusal, missing checkpoint-variable refusal, symlink rejection, frozen hash mismatch,
and a checkpoint changed during evaluation.

## Verification

Commands run after implementation:

```bash
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_vic_impulse_diag90_500_eval_launcher.py
bash -n scripts/slurm/vega_vic_impulse_diag90_500_eval.sbatch
git diff --check
```

The focused suite passed as above; `bash -n` and `git diff --check` exited successfully. Staged
diff checks are performed immediately before the Task 6 commit.

## Scope and concerns

- No training, Slurm submission, or Vega runtime was started. The GPU/runtime check is guarded but
  has only been exercised through the fake interpreter; Task 7 owns the live smoke and mandatory
  reward/contact preflight gates.
- The fixture uses a narrow Git/SHA command shim solely to model the immutable remote checkpoint
  identities and approved-base ancestry without fabricating SHA-256 preimages or requiring the
  real Vega checkout. Real repository cleanliness, output collision behavior, Bash argument flow,
  and artifact hashing remain exercised by the launcher.

## Fix round 1: hidden evaluator artifacts

Review identified that the original `"$EVALUATION_DIR"/*` enumeration omitted dotfiles, allowing
the five required artifacts plus a hidden extra file to reach PASS. The regression fixture now has
the fake evaluator write `.extra` after producing the normal five files.

### RED

```bash
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_vic_impulse_diag90_500_eval_launcher.py -k hidden_extra
```

Result before the fix: `1 failed, 6 deselected in 1.98s`; the launcher returned `0` despite the
hidden sixth output.

### GREEN and verification

The launcher now enables Bash `nullglob` and `dotglob` before expanding the immediate output
directory into an array. This remains argument-safe and makes `.extra` count as a sixth artifact.

```bash
conda run -n unitree_mjlab python -m pytest -q \
  tests/test_vic_impulse_diag90_500_eval_launcher.py -k hidden_extra
# 1 passed, 6 deselected in 0.77s

conda run -n unitree_mjlab python -m pytest -q \
  tests/test_vic_impulse_diag90_500_eval_launcher.py
# 7 passed in 4.85s

bash -n scripts/slurm/vega_vic_impulse_diag90_500_eval.sbatch
git diff --check
```

Both syntax and diff checks exited successfully. No Vega submission or training was performed.
