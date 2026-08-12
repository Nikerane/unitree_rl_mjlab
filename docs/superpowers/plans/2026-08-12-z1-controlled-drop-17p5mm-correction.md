# Z1 Controlled-Drop 17.5 mm Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the primary controlled-drop striker to the owner-approved `17.5 mm` total height, prove that it maintains productive contact through successful nail penetration, and bank a fresh five-drop Vega A100 impulse reference.

**Architecture:** Retain the existing fail-closed controlled-drop fixture and change only its MuJoCo cylinder half-height from `0.004 m` to `0.00875 m`. Lock the correction at the existing public environment and one-drop seams, render the real corrected environment, then execute the unchanged five-release calibration from a clean pinned revision on Vega. Preserve the old result as superseded evidence and publish the new result separately.

**Tech Stack:** Python 3.10, pytest, mjlab 1.4.0, MuJoCo 3.8.1, mujoco-warp 3.8.1, Git, Slurm, Vega A100.

## Global Constraints

- Work only in `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal`.
- Preserve all six owner-owned untracked paths and do not stage or modify unrelated research notes or result directories.
- Total striker height is exactly `0.0175 m`; `PRIMARY_DROP_HALF_HEIGHT_M` is exactly `0.00875`.
- Keep mass `0.200 kg`, radius `0.012 m`, diameter `0.024 m`, and lower-face clearance `0.150 m` unchanged.
- Keep release velocity, slide axis, slide damping/friction, rotation constraint, contact properties, production nail, solver, timestep, tracker, window, progress epsilon, device semantics, and exactly-five-release protocol unchanged.
- Do not preserve effective density and do not change the contact footprint.
- Do not overwrite or rewrite `evaluation/results/2026-08-11_z1_controlled_drop_iref/primary.json`.
- The old `0.10635668784379959 N s` result is superseded evidence and must not be propagated into FIC.
- Do not change D4, FIC registrations, `I_ref`, `k_tt`, RTT, CaT, policy metadata, actuator gains, Cartesian tasks, curriculum, domain randomization, or VIC in this plan.
- CUDA remains authoritative. Do not hard-code the exploratory CPU impulse as the new reference.
- If the authoritative Python calibration starts and returns an invalid/partial/mixed-horizon result, preserve evidence and stop rather than tuning or retrying the experiment.
- Use `MPLCONFIGDIR=/private/tmp/z1-drop-17p5-mpl` and `WARP_CACHE_PATH=/private/tmp/z1-drop-17p5-warp` for local environment/test commands.

## Approved Public Test Seams

1. `make_controlled_drop_env_cfg() -> ManagerBasedRlEnvCfg`: compiled geometry, mass, placement, contact, and production configuration.
2. `run_one_primary_drop(device="cpu", trial_index=1) -> ControlledDropTrial`: observable contact-persistence outcome; the corrected primary fixture must finalize with `reason == "success"`.
3. `run_primary_calibration(...) -> ControlledDropResult` and `scripts/calibrate_controlled_drop.py`: unchanged fail-closed five-release execution and publication boundary.

## File Map

- Modify `src/tasks/hammer/calibration/controlled_drop_env.py`: change the single primary half-height constant.
- Modify `tests/test_controlled_drop_env.py`: freeze `0.00875 m` and require the corrected live drop to reach `success`.
- Modify `docs/superpowers/specs/2026-08-12-z1-controlled-drop-hammer-poll-height-design.md`: record the owner's rounded `17.5 mm` decision.
- Create `evaluation/results/2026-08-12_z1_controlled_drop_poll_height_iref/primary.json`: unmodified output of the new authoritative run.
- Create `docs/results/2026-08-12_z1_controlled_drop_poll_height_iref.md`: execution identity, all raw trials, arithmetic mean, interpretation, and boundary.
- Modify `docs/results/2026-08-11_z1_controlled_drop_iref.md`: add only a concise supersession banner pointing to the new result.
- Create no renderer, plotting pipeline, launcher framework, or calibration override API.

---

### Task 1: Correct and qualify the primary striker

**Files:**
- Modify: `tests/test_controlled_drop_env.py:172-178,352-368`
- Modify: `src/tasks/hammer/calibration/controlled_drop_env.py:21-26`

**Interfaces:**
- Consumes: `make_controlled_drop_env_cfg()` and `run_one_primary_drop(device: str, trial_index: int)`.
- Produces: the same public interfaces with the corrected fixed geometry and a successful primary-drop outcome.

- [ ] **Step 1: Write the failing public-seam assertions**

Change the fixed literal in `test_primary_fixture_compiles_exact_protocol`:

```python
assert calibration.PRIMARY_DROP_HALF_HEIGHT_M == 0.00875
```

Tighten the final assertion in `test_one_primary_drop_contacts_and_finalizes_with_finite_positive_measurements`:

```python
assert trial.reason == "success"
```

- [ ] **Step 2: Run both tests and witness RED on the old 8 mm fixture**

Run:

```bash
MPLCONFIGDIR=/private/tmp/z1-drop-17p5-mpl \
WARP_CACHE_PATH=/private/tmp/z1-drop-17p5-warp \
PYTHONPATH=. \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_controlled_drop_env.py::test_primary_fixture_compiles_exact_protocol \
  tests/test_controlled_drop_env.py::test_one_primary_drop_contacts_and_finalizes_with_finite_positive_measurements \
  -q
```

Expected: the static contract reports `0.004 != 0.00875`, and the live drop reports `"window" != "success"`.

- [ ] **Step 3: Make the minimum implementation change**

In `src/tasks/hammer/calibration/controlled_drop_env.py`, change only:

```python
PRIMARY_DROP_HALF_HEIGHT_M: float = 0.00875
```

The existing body-center calculation and face-site placement must remain expressed in terms of `PRIMARY_DROP_HALF_HEIGHT_M`; this preserves exactly `0.150 m` lower-face clearance.

- [ ] **Step 4: Run the two tests and witness GREEN**

Run the Step 2 command. Expected: `2 passed`.

- [ ] **Step 5: Run the complete controlled-drop focused suite**

Run:

```bash
MPLCONFIGDIR=/private/tmp/z1-drop-17p5-mpl \
WARP_CACHE_PATH=/private/tmp/z1-drop-17p5-warp \
PYTHONPATH=. \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_controlled_drop_contract.py \
  tests/test_controlled_drop_env.py \
  tests/test_calibrate_controlled_drop.py \
  -q
```

Expected: all controlled-drop tests pass.

- [ ] **Step 6: Self-review and commit only the code and tests**

```bash
git diff --check
git diff -- src/tasks/hammer/calibration/controlled_drop_env.py tests/test_controlled_drop_env.py
git add src/tasks/hammer/calibration/controlled_drop_env.py tests/test_controlled_drop_env.py
git diff --cached --check
git commit -m "fix(hammer): match controlled drop poll height"
```

---

### Task 2: Render and verify the corrected revision

**Files:**
- Do not modify tracked repository files.
- Produce visual evidence under `/Users/nikerane/.codex/visualizations/2026/08/11/019ff201-151e-7e20-b4e8-418647d4738e/controlled-drop-17p5mm-confirmation/`.

**Interfaces:**
- Consumes: the committed corrected fixture from Task 1.
- Produces: three raw simulator frames, a montage, capture metadata, and fresh CPU-suite evidence.

- [ ] **Step 1: Capture the real environment at three observable states**

Use the existing temporary capture driver at `/private/tmp/capture_z1_controlled_drop_visuals.py`, changing only its output directory to `/private/tmp/2026-08-12_z1_controlled_drop_17p5mm_visual_confirmation`. Run it outside the restricted graphics sandbox with the plan-owned caches. It must capture:

1. release at `0.150 m` face clearance;
2. first nail contact;
3. successful event finalization.

- [ ] **Step 2: Preserve and inspect the visual evidence**

Copy the three PNGs, montage, and metadata JSON to the plan-owned visualization directory. Inspect all frames at original resolution and confirm that the cylinder is centered, visibly taller, starts with the correct gap, contacts the nail face, and ends with successful nail penetration.

- [ ] **Step 3: Run the full CPU suite exactly once on the corrected commit**

Run:

```bash
MPLCONFIGDIR=/private/tmp/z1-drop-17p5-mpl \
WARP_CACHE_PATH=/private/tmp/z1-drop-17p5-warp \
PYTHONPATH=. \
/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest -q
```

Expected: exit code `0`. Record the exact passed/skipped totals and the tested Git revision in the SDD report. Do not rerun the full suite unless a later code fix changes tested behavior.

---

### Task 3: Execute and bank the authoritative Vega calibration

**Files:**
- Create: `evaluation/results/2026-08-12_z1_controlled_drop_poll_height_iref/primary.json`
- Create: `docs/results/2026-08-12_z1_controlled_drop_poll_height_iref.md`
- Modify: `docs/results/2026-08-11_z1_controlled_drop_iref.md`

**Interfaces:**
- Consumes: the reviewed, pushed Task-1 revision and asset revision `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Produces: one immutable authoritative JSON result and a human-readable audited result record.

- [ ] **Step 1: Push the exact corrected implementation revision**

Verify that the index and tracked worktree are clean, the only untracked paths are the six protected owner paths, and the branch is `z1-controlled-drop-iref`. Push the branch and record `CODE_SHA=$(git rev-parse HEAD)`.

- [ ] **Step 2: Create a fresh detached Vega checkout and submit one A100 job**

On Vega, fetch `origin`, create a new clean detached worktree at `CODE_SHA`, verify the asset repository is clean at `b58ccd2f81fd246f27c1e8d88cf86484cd888703`, load the same runtime modules used by successful job `41085760`, and run:

```bash
python scripts/calibrate_controlled_drop.py \
  --device cuda:0 \
  --output evaluation/results/2026-08-12_z1_controlled_drop_poll_height_iref/primary.json \
  --code-revision "$CODE_SHA" \
  --asset-revision b58ccd2f81fd246f27c1e8d88cf86484cd888703
```

The directory must exist and the output path must not exist before Python starts. Submit only one calibration job and monitor it to a terminal Slurm state.

- [ ] **Step 3: Audit the authoritative result before copying it into the branch**

Require all of the following:

- Slurm `COMPLETED` with exit `0:0` on an NVIDIA A100;
- exact code and asset revisions;
- device `cuda:0`, backend `mujoco-warp`;
- mjlab `1.4.0`, MuJoCo `3.8.1`, mujoco-warp `3.8.1`;
- `half_height_m == 0.00875`, `mass_kg == 0.2`, `radius_m == 0.012`, `h0_m == 0.15`;
- exactly five trials indexed `1..5`, all contacted, finalized, productive, and `reason == "success"`;
- finite positive pre-contact velocities and impulses;
- exact arithmetic mean recomputed with `math.fsum(impulses) / 5`;
- a SHA-256 digest of the JSON and the complete Slurm output log.

If any requirement fails after Python starts, preserve evidence and stop.

- [ ] **Step 4: Bank the unmodified JSON and result interpretation**

Copy the authoritative `primary.json` byte-for-byte into the new local result directory. Create the new Markdown record with the Slurm job, node, elapsed time, GPU, revisions, versions, JSON SHA-256, all five raw rows, exact unrounded arithmetic mean, observed spread, and explicit simulation-only limitations.

Add this banner immediately below the old result's title:

```markdown
> **Superseded on 2026-08-12:** This `8 mm` contact-coupon result is preserved as historical evidence but must not be used for FIC normalization. The corrected `17.5 mm` hammer-poll-height calibration is recorded in [`2026-08-12_z1_controlled_drop_poll_height_iref.md`](2026-08-12_z1_controlled_drop_poll_height_iref.md).
```

- [ ] **Step 5: Verify provenance and commit the result**

Independently parse the JSON, recompute its mean, compare every Markdown row and identity field with the JSON and Slurm record, run `git diff --check`, and confirm the old JSON is byte-identical to `fbde444`. Stage only the new JSON, new result document, and old-document banner, then commit:

```bash
git commit -m "docs(hammer): bank corrected controlled drop reference"
```

---

### Task 4: Final reviews, corrections, and branch completion

**Files:**
- Modify only files required by concrete review findings within this plan's scope.

**Interfaces:**
- Consumes: all Task 1–3 commits and evidence.
- Produces: a reviewed, verified, pushed branch with no unresolved correctness findings.

- [ ] **Step 1: Run independent reviews**

Obtain all of the following against the corrected plan base through `HEAD`:

1. Opus code/scientific review;
2. Gemini code/scientific review;
3. DeepSeek code/scientific review;
4. repository Standards review;
5. repository Spec review.

Each review must check the `17.5 mm` decision, unchanged invariants, contact-persistence regression, result identity/arithmetic, supersession boundary, and protected-path scope.

- [ ] **Step 2: Fix concrete correctness findings test-first**

Use one bounded fix wave. For every accepted code finding, add or tighten a public-seam test, witness RED, implement the minimum correction, and rerun its focused tests. Re-review the fix diff once. Do not expand into FIC propagation or training.

- [ ] **Step 3: Perform final verification**

Confirm focused tests still pass, the recorded full CPU run covered the final code or rerun it once if code changed after that gate, the authoritative CUDA result is bound to the exact implementation revision, all result hashes and arithmetic match, tracked files are clean, and only the six protected untracked paths remain.

- [ ] **Step 4: Push and stop at the FIC boundary**

Push `z1-controlled-drop-iref`, verify local and remote SHAs match, and report the corrected `I_ref`, test totals, Vega job evidence, review verdicts, pictures, commits, and protected-path status. Do not propagate the value or start FIC training inside this plan.
