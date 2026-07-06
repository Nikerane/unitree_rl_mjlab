# Impulse-CaT C2→C3 Two-Track Enablement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the impulse soft-CaT constraint (`imp_max_p > 0`) on the baseline-subtracted Λ_j with fixture-era measured caps, run the C3 three-arm results campaign on Vega, and in parallel build the rigorous contact-row impulse isolation as a logged validator (never enforced, never gating Track 1).

**Architecture:** Track 1 (Tasks 1–6) is the strictly-ordered critical path: commit → verify the adopted NEAR_NAIL windup re-solve → threshold re-derivation + config flip → gate re-green → C2 local enforcement gate → C3 Vega submission. Track 2 (Tasks 7–9) adds a `ContactRowImpulseAccumulator` that reads only the hammer↔nail efc rows from `mujoco_warp` (`env.sim.wp_data.efc` sparse `JᵀF`), validated by an exact-reconstruction invariant, wired as a log-only metric and a three-way gate cross-check. Task 10 collects C3 results into a dated record and runs the vacuous-check.

**Tech Stack:** mjlab 1.4.0 / mujoco 3.8.1 / mujoco_warp 3.8.1, torch, conda env `unitree_mjlab` (`~/miniconda3/envs/unitree_mjlab/bin/python`, aliased `$PY` below), Slurm on EuroHPC Vega. Sibling assets repo: `~/repos/safe_impact_manipulation` (branch `hammer-z1`).

**Spec:** `docs/superpowers/specs/2026-07-06-impulse-cat-c2c3-two-track-design.md` — read it for rationale, decision records, and the two 2026-07-06 amendments (adopted windup re-solve; C2 probe-limit machinery gate with the vacuous-check moved to C3 analysis).

## Global Constraints

- **One GPU per Vega run, never multi-GPU**: `#SBATCH --gres=gpu:1`; fan out arms × seeds with `sbatch --array` only.
- **Enforced quantity = baseline-subtracted Λ_j** (`subtract_baseline: True`); the contact-row Λ (Track 2) is a **logged metric only** — it must never feed `joint_impulse_excess` or any δ in this plan.
- **Track 2 must never gate Track 1**: a Track-2 failure blocks only Tasks 7–9, never C2/C3.
- **Track-2 training wiring is conditional**: `mujoco_warp`'s efc struct has NO `worldid` field (verified 2026-07-06); the contact-row metric enters the *training* config only after the `num_envs=2` attribution test (Task 7 Step 6) passes — otherwise it ships gate/eval-only.
- **`imp_seed` stays `1e-3`** (normalizer decay floor). Do NOT paste any p95 statistic into it — the "reference-strike p95" comments are stale pre-hardening remnants that Task 3 deletes.
- **J_limit values are measurements**: only numbers printed by `derive_impulse_thresholds.py` run on the post-re-solve tree may be pasted into config. Never reuse gripper-era numbers (J_limit [3.44, 6.88, …], seed 0.16) or any number from docs.
- **Never edit installed packages** (`rsl_rl`, `mjlab`, `mujoco_warp` under site-packages) — read-only.
- **Never penalize raw ‖Δq̇‖² / never clock-time velocity tracking near contact / impulse always accumulated at substep rate** (standing corrections, `tracking_impact_impulse_design_research.md`).
- **Pre-train gate must be green before any GPU submission**: `validate_rewards.py` phases A–M + `verify_contact_sensor.py` + `verify_reward_setup.py` + the 5-file pytest set + `test_single_strike.py` (mandatory after any EE/pose change).
- **Full pytest suite green before every commit**: `$PY -m pytest tests/ -q` is part of every task's commit step.
- **Commits stay local** — never push. Never commit `.env`, `.DS_Store`, `graphify-out/`, `.claude/settings.local.json`.
- **The working tree is being actively edited by the user** (z1_constants.py and sibling XMLs changed 2026-07-06 late afternoon): before editing any file this plan quotes, RE-READ it and re-anchor — line numbers marked `~` are approximate by design.
- Docs freshness guard (`tests/test_docs_current.py`) must pass after every docs edit.
- `$PY` = `~/miniconda3/envs/unitree_mjlab/bin/python`; run repo commands from `/Users/nikerane/repos/unitree_rl_mjlab` unless stated.

---

### Task 1: Commit the standing tree (both repos)

The working trees carry the entire docs consolidation + impulse machinery + the user's 2026-07-06 windup re-solve (this repo, ~95 modified/untracked paths including many `docs/` archive moves) and the fixture/grasp/mocap/scratchpad changes (sibling). Group them into reviewable commits so every later task has a clean diff. **Nothing is pushed.**

**Files:** no source changes — `.gitignore` additions + commits only.

**Interfaces:**
- Produces: clean `git status` in both repos; the adopted `NEAR_NAIL_JOINT_POS` (windup re-solve) is committed here, so Task 2 is verification-only.

- [ ] **Step 1: Take fresh inventories (do not trust this plan's counts — the tree moves)**

```bash
git status --short | tee /tmp/t1_inventory_main.txt | wc -l
cd ~/repos/safe_impact_manipulation && git status --short | tee /tmp/t1_inventory_sibling.txt
```

Expected (approximate, refresh from the actual output): main repo ~95 lines — `src/tasks/hammer/**`, `tests/**`, `docs/**` (including archive moves/deletions), `CLAUDE.md`, `thesis_synthesis.md`, `thesis_direction_update.md`, `thesis_handoff_brief_original.md`, `src/assets/robots/unitree_z1/z1_constants.py` (the user's re-solve), `.claude/settings.json`, plus junk (`.env`, `.DS_Store`, `docs/.DS_Store`, `docs/research/.DS_Store`, `graphify-out/`, `.claude/settings.local.json` if present). Sibling: `hammer_z1_env/assets/*` (2 XMLs + new mesh), `hammer_z1_env/README.md`, `hammer_z1_env/view.py`, `hammer_z1_env/solve_ik_oriented.py` (if touched), `hammer_z1_env/scratchpad/**` (the windup-sweep solver — provenance, commit it), `.gitignore`, and untracked `CLAUDE.md` + `.claude/`.

- [ ] **Step 2: Ignore the never-commit artifacts (main repo)**

Append to `.gitignore` (only lines not already present):

```
.env
.DS_Store
graphify-out/
.claude/settings.local.json
```

- [ ] **Step 3: Grouped commits in the main repo (four commits, each with its own `git commit`)**

```bash
$PY -m pytest tests/ -q   # suite must be green BEFORE committing; STOP on failures

git add src/tasks/hammer/ tests/
git commit -m "feat(impulse-cat): C0/C1/C5 machinery — substep accumulators, per-event pulse, soft-OR hook, guards + tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git add docs/ CLAUDE.md thesis_synthesis.md thesis_direction_update.md thesis_handoff_brief_original.md .claude/settings.json .gitignore
git commit -m "docs: consolidation — current-truth index, thesis digest+notation, freshness guard, archive-read hook, dated-record banners

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git add src/assets/robots/unitree_z1/z1_constants.py
git commit -m "feat(pose): NEAR_NAIL windup re-solve for the L6 grasp — in-plane vertical reset at (0.5,0,0.25); vertical-at-nail kinematically infeasible

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git status --short   # remaining lines must ALL be ignored junk; anything else → STOP and report it
```

If `git status` still shows a tracked/stageable file not covered above, do NOT `git add -A` — list it in the task report and stop for a decision.

- [ ] **Step 4: Commit the sibling assets repo**

```bash
cd ~/repos/safe_impact_manipulation
git add hammer_z1_env/ .gitignore
git commit -m "feat(fixture): L6 SimplifiedLink06 holder + −90°Z grasp reseat (pos −0.0506); mocap mirror; windup-sweep IK scratchpad; README title fix

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short   # expected leftovers: untracked CLAUDE.md and .claude/ ONLY — leave them, note them in the report
```

- [ ] **Step 5: Verify the suite is still green on the committed tree**

Run: `cd /Users/nikerane/repos/unitree_rl_mjlab && $PY -m pytest tests/ -q -m "not integration"`
Expected: all pass, 0 failures.

---

### Task 2: Verify the adopted NEAR_NAIL windup re-solve

**Do NOT re-run any IK.** The user re-solved NEAR_NAIL on 2026-07-06 (committed in Task 1): with the new grasp, a vertical strike AT the floor nail is kinematically impossible; the adopted pose poises the face at world (0.5, 0, 0.25), in-plane (joint1=0), dead-vertical, and the policy drives down to discover the (necessarily oblique near the floor) contact. This task verifies the task machinery still works from that reset. Read the provenance block at `src/assets/robots/unitree_z1/z1_constants.py` (~L175–197) before starting.

**Files:** none modified — verification only.

**Interfaces:**
- Consumes: the committed windup `NEAR_NAIL_JOINT_POS`.
- Produces: a PASS/FAIL verdict on strike feasibility from z=0.25 — the go/no-go for everything downstream.

- [ ] **Step 1: Reference playback gate**

Run: `$PY docs/research/reward-design/playback_reference.py 2>&1 | tee /tmp/t2_playback.txt`
Expected: the script's own gate — **best**-strike depth ≥ `NAIL_SUCCESS_THRESHOLD` (0.027 m) across the approach heights (historically only the lowest approach cleared it — that is a PASS). Record the press-probe depth as a **watchdog number** in the report (a sustained press CAN reach threshold per the 2026-06-10 measurement — that is expected, not a failure).

- [ ] **Step 2: Q1 re-confirm (mandatory after any EE/pose change)**

Run: `$PY docs/research/reward-design/test_single_strike.py 2>&1 | tail -5`
Expected: single strike clears the 0.027 m threshold from the new reset. If it does NOT: STOP — report the measured depths per approach; the threshold/pose trade-off (lower the 0.027 threshold vs revisit the reset height) is the **user's decision**, do not tune either yourself.

- [ ] **Step 3: Visual sanity render**

Run: `$PY scripts/render_reference.py --distance 0.85 --elevation -25`
Expected: frames + mp4 under `/tmp/hammer_ref/`; read 2–3 PNGs — the arm starts poised above the nail, descends, and the hammer face (not the claw) meets the nail head; no interpenetration at reset.

- [ ] **Step 4: Report**

No commit (nothing changed). Report: playback best-strike depth, press watchdog depth, single-strike verdict, and one rendered frame path. Any FAIL here blocks Tasks 3–10.

---

### Task 3: Threshold re-derivation + enforcement config flip

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py` (re-read first; the `cat_impulse` block: `substep_impulse` params ~L219–228, `cat_soft` params ~L242–257, `delivered_impulse` ~L261–269)
- Modify: `docs/research/reward-design/derive_impulse_thresholds.py` (module docstring only: stale p95 wording)
- Modify: `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` (Status line 3, line 6, Pinocchio lines ~128/188)
- Test: existing `tests/test_cat_soft_hook.py`, `tests/test_impulse_constraint.py`, `tests/test_impulse_bound.py` + full suite; `validate_rewards.py` Phase M

**Interfaces:**
- Consumes: verified pose from Task 2.
- Produces: `IMP_J_LIMIT` (list of 6 floats) and `i_ref` in `env_cfgs.py`, consumed by Tasks 5–6; enforcement stays OFF (`imp_max_p=0.0`) in the repo default.

- [ ] **Step 1: Run the quantity gate on the new tree**

```bash
$PY docs/research/reward-design/derive_impulse_thresholds.py | tee /tmp/derive_thresholds_fixture.txt
echo "exit: $?"
```

Expected: `exit: 0` (gate PASS) and five printed sections. Capture: section [4]'s six per-joint `J_limit` values (N·m·s) and section [3]'s **MEAN** object-side ∫F·dt (for `i_ref`; the gripper-era analogue was ≈0.107 N·s). On `exit: 1`: STOP, paste the failing section verbatim, do not touch config. Note: the oblique-contact geometry may shift the contact-window duration substantially vs the old face-on strike — that is exactly why these are measurements.

- [ ] **Step 2: Flip the enforced quantity to baseline-subtracted**

In `env_cfgs.py` `substep_impulse` params (~L226), change:

```python
        "subtract_baseline": False,  # C0 gate decides raw vs baseline-subtracted (friction removal)
```

to:

```python
        "subtract_baseline": True,  # C2 (2026-07-06): enforce the baseline-subtracted Λ_j — removes
        # the dominant dof-friction share; residual quantified by the Track-2 contact-row metric.
```

- [ ] **Step 3: Write the measured caps into the config**

**Placement pin:** inside the `if cat_impulse:` branch, directly above the `cfg.metrics["cat_soft"] = MetricsTermCfg(` assignment whose params include `imp_limit` (~L242) — NOT the velocity-only `cat_soft` branch earlier in the file (placing it there → `NameError` in the impulse branch).

```python
    # Fixture-era per-joint impulse caps, MEASURED by derive_impulse_thresholds.py on 2026-07-06
    # (windup NEAR_NAIL reset, oblique contact; gate PASS log: /tmp/derive_thresholds_fixture.txt →
    # dated record at C3). J_limit_j = τ_rated_j × 2 (HD Repeated-Peak) × Δt_impact.
    IMP_J_LIMIT = [<j1>, <j2>, <j3>, <j4>, <j5>, <j6>]  # ← paste section [4], N·m·s
```

and change the `cat_soft` params:

```python
        "imp_limit": hammer_mdp.Z1_JOINT_IMPULSE_LIMIT,  # placeholder; real per-joint J_limit at C0/C2
        "imp_max_p": 0.0,  # C0 LOG-ONLY
        "imp_seed": 1e-3,  # C0 gate replaces with the reference-strike p95 over-limit excess
```

to:

```python
        "imp_limit": IMP_J_LIMIT,  # measured fixture-era per-joint caps (see above)
        "imp_max_p": 0.0,  # still log-only HERE; C2/C3 raise it per-run via CLI override (Tasks 5–6)
        "imp_seed": 1e-3,  # normalizer DECAY FLOOR only — never a p95/excess statistic (hook.py)
```

- [ ] **Step 4: Set `i_ref` from the measurement**

In the `delivered_impulse` term, replace `"i_ref": 1.0,` with section [3]'s **MEAN** ∫F·dt and update the comment line to: `i_ref MEASURED 2026-07-06 (gate section [3] mean); weight tuned at C2.`

- [ ] **Step 5: Clean the stale p95 docstring in the derive script**

In `derive_impulse_thresholds.py`'s module docstring, rewrite the "emits imp_seed = p95(Λ_j)" sentence to: `section [5] prints p95(Λ_j) for reference only — imp_seed in config stays a small decay floor (1e-3); the hook self-seeds its normalizer from the first over-limit sample.`

- [ ] **Step 6: Record the discharged gates in the impl plan**

In `IMPULSE_CAT_IMPL_PLAN.md`: update the Status line («C2 … pend the Khadiv mechanism confirm + fixture-era threshold re-derivation») and line 6 («awaits Khadiv mechanism confirm + Pinocchio decision») to record: Khadiv verbal go-ahead 2026-07-06 (C3 results = formal confirmation at the next meeting); fixture-era re-derivation DONE 2026-07-06 (gate PASS, windup reset); **Pinocchio decision RECORDED**: the MuJoCo-native object-side ∫F·dt is the accepted ground truth for enforcement validation — Pinocchio `impulseDynamics` stays scaffolded/deferred indefinitely (mirrors visual plan plan-c00a0e0144274fe4 "Already decided (2026-06)"). Update the Pinocchio "revisit before max_p>0" lines (~128/188) to point at this record.

- [ ] **Step 7: Verify + commit**

```bash
$PY -m pytest tests/test_cat_soft_hook.py tests/test_impulse_constraint.py tests/test_impulse_bound.py -q
$PY docs/research/reward-design/validate_rewards.py 2>&1 | tail -5
$PY -m pytest tests/ -q
git add src/tasks/hammer/config/z1/env_cfgs.py docs/research/reward-design/derive_impulse_thresholds.py docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md
git commit -m "feat(impulse-cat): fixture-era measured J_limit + i_ref; enforce baseline-subtracted Λ; record gate decisions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: all green; validate_rewards ends `ALL PHASES PASS` (A–M). If Phase M hardcodes the raw accumulator mode, fix the phase to certify the shipped config value (`subtract_baseline=True`), not a hardcoded mode.

---

### Task 4: Full pre-train gate re-green + blocker-doc sync

**Files:**
- Modify: `docs/research/reward-design/OPEN_QUESTIONS.md` (~L168 Q1 note, ~L179 Remaining blockers)
- Modify: `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` (§6b, ~L178)
- Modify: `~/.claude/projects/-Users-nikerane-repos-unitree-rl-mjlab/memory/l6-hammer-fixture-ee.md` + `MEMORY.md` index line 16

**Interfaces:**
- Consumes: Tasks 2–3 complete.
- Produces: a green gate log — the precondition for any `imp_max_p > 0` run (Tasks 5–6).

- [ ] **Step 1: Run the entire gate**

```bash
$PY docs/research/reward-design/validate_rewards.py 2>&1 | tail -3
$PY docs/research/reward-design/verify_contact_sensor.py 2>&1 | tail -3
$PY docs/research/reward-design/verify_reward_setup.py 2>&1 | tail -5
$PY -m pytest tests/ -q
```

Expected: phases A–M pass; contact sensor verified; random-policy sweep sane; full pytest green. (`test_single_strike.py` already re-ran in Task 2.) Any failure: STOP, report verbatim.

- [ ] **Step 2: Sync the blocker docs**

- `OPEN_QUESTIONS.md` ~L168: Q1 note → `re-measured post-L6-fixture 2026-07-06 (windup re-solve, reset z=0.25 + test_single_strike re-run): strike feasibility re-confirmed on the adopted pose`.
- `OPEN_QUESTIONS.md` ~L179 Remaining blockers → `NEAR_NAIL windup re-solve adopted + gate re-green DONE 2026-07-06; no blockers before the C2/C3 campaign` (keep the deferred-Q list).
- `IMPULSE_CAT_IMPL_PLAN.md` §6b → mark the L6 re-solve done: adopted the user's 2026-07-06 windup-sweep pose (vertical-at-nail infeasible with the new grasp; reset (0.5,0,0.25); oblique contact discovered by the policy).
- Memory `l6-hammer-fixture-ee.md`: the ⚠️ RE-STALED bullet → resolved (2026-07-06 windup re-solve, note the z=0.25/oblique design + the "bracing must emerge" open thread); `MEMORY.md` line 16 → `NEAR_NAIL windup re-solve DONE 2026-07-06 (reset z=0.25, vertical-at-nail infeasible); gate re-green DONE`.

- [ ] **Step 3: Guard + commit**

```bash
$PY -m pytest tests/test_docs_current.py -q && $PY -m pytest tests/ -q
git add docs/research/reward-design/OPEN_QUESTIONS.md docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md
git commit -m "docs: blockers discharged — windup pose adopted, fixture-era gate re-green (A–M, sensors, single-strike, full suite)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: C2 — local enforcement gate + max_p mini-sweep

**Spec amendment applies (read it):** reference strikes measure ~5% of J_limit (C0: the cap binds at ~21× more violent), so the δ machinery would never fire against the real caps during this gate. The gate therefore runs **twice per max_p**: (A) a **probe pass** with the limit scaled to `0.05 × IMP_J_LIMIT`, forcing over-limit events to hard-gate the machinery (c_max bounded / not pinned at seed / not spiked >100× worst excess; δ graded, not saturated; strike physically unchanged); (B) a **report pass** at the real caps, printing the binding statistics (fraction of strikes in `[0.8·J_limit, J_limit]`, max Λ/J_limit) WITHOUT failing on them. The δ-attributable-reduction vacuous check runs at C3 analysis (Task 10).

**Files:**
- Create: `docs/research/reward-design/c2_enforcement_gate.py`
- Create: `docs/research/reward-design/reward_design_util.py` (shared strike-driving helper, **modeled on — not refactoring —** `derive_impulse_thresholds.py:main()`; the derive script is NOT edited in this task)
- Test: the script IS the gate (exits 1 on FAIL); plus the standing suites stay green.

**Interfaces:**
- Consumes: `IMP_J_LIMIT` in `env_cfgs.py` (Task 3); `z1_hammer_env_cfg(play=True, cat_impulse=True)`; hook extras `env.extras["cat_delta"]`; the reference-driving idiom of `derive_impulse_thresholds.py:59-133` (env/site setup, `SingleStrikeReference` playback loop).
- Produces: `run_reference_strikes(cfg, heights, limit_override=None) -> dict` in `reward_design_util.py` (keys: `max_excess: list[float]`, `max_delta: list[float]`, `cmax: list[float]`, `depth: list[float]`, `band_fraction: float`, `binding_ratio: float`) — reused by Track-2 tests; the chosen `imp_max_p` consumed by Task 6.

- [ ] **Step 1: Write the shared helper**

`docs/research/reward-design/reward_design_util.py`: extract the env-build + site-helper + playback-loop pattern from `derive_impulse_thresholds.py` `main()` (lines ~59–133) into a reusable function. The driving loop is EXACTLY this idiom (copied from the derive script — keep it verbatim):

```python
env.reset()
ref = SingleStrikeReference(1, env.device, approach_height=h)
ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
for k in range(1, n + HOLD_STEPS + 1):
    target = ref.playback_target(min(k, n))
    action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
    env.step(action)
    if int(env.episode_length_buf[0]) == 0:  # success auto-reset; strike captured
        break
```

Per strike, record: worst-joint excess over the (possibly overridden) limit, peak `env.extras["cat_delta"]`, final nail depth; after all strikes snapshot the hook's `_imp_cmax`. `limit_override` (a scalar factor) rescales the limit used for excess/δ by monkeypatching the hook's `_imp_limit` tensor for the run (restore after) — the config file is never edited.

- [ ] **Step 2: Write the gate script**

`docs/research/reward-design/c2_enforcement_gate.py`, complete content:

```python
"""C2 enforcement gate (IMPULSE_CAT_IMPL_PLAN.md C2 + spec amendment 2026-07-06).

Two passes per imp_max_p ∈ {0.25, 0.5}:
  PROBE pass (limit × 0.05 — forces over-limit events; machinery hard gates):
    [1] c_max bounded: finite, above the 1e-3 seed on ≥1 column, and ≤ 100× the worst observed
        excess (not single-event-spiked).
    [2] graded δ: across strikes, higher excess ⇒ strictly higher δ, and δ is not the saturated
        max_p on every over-limit strike.
    [4] strike survives δ: nail depth unchanged vs the log-only run (δ reweights learning only —
        physics must be bit-identical; catches hook wiring that mutates state).
  REPORT pass (real IMP_J_LIMIT — statistics only, no failure):
    [3] band_fraction = fraction of strikes with worst-joint Λ in [0.8·J_limit, J_limit];
        binding_ratio = max Λ_j/J_limit_j. Printed for the C3 record; the δ-attributable
        vacuous-check runs at C3 analysis (Task 10), not here.
Usage: $PY docs/research/reward-design/c2_enforcement_gate.py   (exits 1 on FAIL)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

# reward-design is hyphenated (not a package): import the shared helper by path, like its siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_design_util import run_reference_strikes  # noqa: E402

MAX_PS = (0.25, 0.5)
HEIGHTS = (0.06, 0.10, 0.15)
PROBE_SCALE = 0.05  # forces over-limit events on reference strikes (C0: real cap binds ~21× higher)


def build_cfg(imp_max_p: float):
    cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
    cfg.scene.num_envs = 1
    cfg.metrics["cat_soft"].params["imp_max_p"] = imp_max_p
    return cfg


def main() -> int:
    failures: list[str] = []
    baseline = run_reference_strikes(build_cfg(0.0), heights=HEIGHTS)  # log-only physics reference

    for mp in MAX_PS:
        probe = run_reference_strikes(build_cfg(mp), heights=HEIGHTS, limit_override=PROBE_SCALE)
        print(f"\n=== PROBE imp_max_p={mp} (limit × {PROBE_SCALE}) ===")
        print(f"excess: {probe['max_excess']}\nδ:      {probe['max_delta']}\nc_max:  {probe['cmax']}")
        cmax = torch.as_tensor(probe["cmax"])
        worst_excess = max(probe["max_excess"])
        if not torch.isfinite(cmax).all() or (cmax <= 1e-3).all():
            failures.append(f"[{mp}] c_max degenerate (pinned at seed / non-finite): {probe['cmax']}")
        if worst_excess > 0 and (cmax > 100 * worst_excess).any():
            failures.append(f"[{mp}] c_max single-event-spiked (> 100× worst excess {worst_excess:.3g})")
        over = sorted((e, d) for e, d in zip(probe["max_excess"], probe["max_delta"]) if e > 0)
        if len(over) < 2:
            failures.append(f"[{mp}] probe produced <2 over-limit strikes — probe scale too loose")
        else:
            es, ds = zip(*over)
            if not ds[-1] > ds[0]:
                failures.append(f"[{mp}] δ not graded: excess {es} → δ {ds}")
            if all(abs(d - mp) < 1e-6 for d in ds):
                failures.append(f"[{mp}] δ saturated at max_p on every over-limit strike")
        if not torch.allclose(
            torch.as_tensor(probe["depth"]), torch.as_tensor(baseline["depth"]), atol=1e-6
        ):
            failures.append(f"[{mp}] δ changed the PHYSICS: depths {probe['depth']} vs log-only {baseline['depth']}")

        report = run_reference_strikes(build_cfg(mp), heights=HEIGHTS)
        print(f"=== REPORT imp_max_p={mp} (real caps) ===")
        print(f"band_fraction [0.8·J,J]: {report['band_fraction']:.3f}")
        print(f"binding_ratio max Λ/J:   {report['binding_ratio']:.3f}  (reference strikes are gentle by design; the LEARNED policy under delivered-impulse maximization is what approaches the cap)")

    if failures:
        print("\nC2 GATE FAIL:\n  - " + "\n  - ".join(failures))
        return 1
    print(f"\nC2 GATE PASS for imp_max_p ∈ {MAX_PS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the gate**

```bash
$PY docs/research/reward-design/c2_enforcement_gate.py | tee /tmp/c2_gate.txt; echo "exit: $?"
```

Expected: `C2 GATE PASS`, `exit: 0`. On any FAIL: **STOP and report; propose a `tau`/robust-statistic change in the report — do not apply it, and never touch `IMP_J_LIMIT`** (it is a measurement).

- [ ] **Step 4: CPU smoke-train under enforcement**

First discover the exact tyro override path: `$PY scripts/train.py Unitree-Z1-Hammer-CaT-Impulse --help 2>&1 | grep -i "imp.max.p"` — then:

```bash
$PY scripts/train.py Unitree-Z1-Hammer-CaT-Impulse --gpu-ids '[]' --env.scene.num-envs 8 --agent.max-iterations 5 --agent.logger tensorboard <discovered-imp-max-p-flag> 0.5 2>&1 | tail -5
```

Expected: 5 iterations complete, no exception, reward not NaN. If tyro exposes no such override path, add `imp_max_p: float = 0.0` as a `z1_hammer_env_cfg` factory kwarg plumbed into the params dict (repo default unchanged), and note it in the report.

- [ ] **Step 5: Pick `imp_max_p` + commit**

Decision rule: prefer `0.5` if both pass (matches the velocity arm's `max_p=0.5` for comparability); `0.25` if 0.5 saturates δ in the probe. Record the choice at the top of `/tmp/c2_gate.txt` and in the Task-6 commands.

```bash
$PY -m pytest tests/ -q
git add docs/research/reward-design/c2_enforcement_gate.py docs/research/reward-design/reward_design_util.py
git commit -m "feat(impulse-cat): C2 enforcement gate — probe-limit machinery hard-gates + real-cap binding report, max_p mini-sweep

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: C3 — Vega submission (three arms × 3 seeds)

**Files:**
- Modify: `docs/VEGA_TRAINING_PLAN.md` (append the C3 campaign block)

**Interfaces:**
- Consumes: chosen `imp_max_p` + override flag path from Task 5; green gate from Task 4.
- Produces: three Slurm array jobs; run names `c3_imp_seed{0..2}`, `c3_track_seed{0..2}`, `c3_catsoft_seed{0..2}` consumed by Task 10.

- [ ] **Step 1: Append the C3 block to `docs/VEGA_TRAINING_PLAN.md`**

```markdown
## C3 campaign (2026-07-06) — impulse-CaT enforcement results
One GPU per run (`--gres=gpu:1`), 3 seeds via array, ITERS=5000, NENVS=4096.
    ITERS=5000 NENVS=4096 RUN=c3_imp     TASK=Unitree-Z1-Hammer-CaT-Impulse EXTRA="<imp-max-p flag from C2> <CHOSEN>" sbatch --array=0-2 scripts/slurm/train_array.sbatch
    ITERS=5000 NENVS=4096 RUN=c3_track   TASK=Unitree-Z1-Hammer-Track                                                 sbatch --array=0-2 scripts/slurm/train_array.sbatch
    ITERS=5000 NENVS=4096 RUN=c3_catsoft TASK=Unitree-Z1-Hammer-CaT-Soft                                              sbatch --array=0-2 scripts/slurm/train_array.sbatch
(-Track = r_imit-only baseline; -CaT-Soft = velocity CaT re-run on the fixture-era tree for a same-tree comparison. The repo default keeps imp_max_p=0 — only c3_imp's EXTRA raises it.)
```

- [ ] **Step 2: Sync + submit**

Both repos must be on Vega at the committed state (siblings, per `scripts/slurm/setup_vega.sh`). If `ssh vega true` succeeds from this machine: rsync both repos and submit the three lines from the login node. **If not: emit the exact rsync + sbatch commands into the task report and hand submission to the user** — do not simulate it.

- [ ] **Step 3: Commit**

```bash
$PY -m pytest tests/ -q
git add docs/VEGA_TRAINING_PLAN.md
git commit -m "docs(vega): C3 impulse-CaT campaign — 3 arms x 3 seeds, one GPU per run

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Track 2 — efc layout probe + exact-reconstruction invariant

Pin the sparse efc→joint mapping empirically and lock it with the invariant `Σ_all_rows (JᵀF) == qfrc_constraint`. **Known API facts (verified 2026-07-06, encode them — do not rediscover):** active contact count is `d.nacon` (NOT `ncon`; `contact.geom.shape[0]` is the ALLOCATED capacity `naconmax` — never use it as a count); the efc struct has NO `worldid` field (fields: `type,id,J_rownnz,J_rowadr,J_colind,J,pos,margin,D,vel,aref,frictionloss,force,state,Ma,Jqvel`); `d.contact` HAS `worldid`; geom names in the compiled model are **entity-prefixed** (`robot/hammer_head_0`, `robot/hammer_head_1`, `nail_block/nail_shaft`, `nail_block/nail_head`, `nail_block/block_geom`).

**Files:**
- Create: `src/tasks/hammer/mdp/contact_row_impulse.py` (helpers only in this task)
- Test: `tests/test_contact_row_impulse.py`

**Interfaces:**
- Consumes: `env.sim.wp_data`; `warp.to_torch`; the reference-driving idiom (same verbatim loop as Task 5 Step 1 — inline it in the tests; do not import from the hyphenated docs dir in tests).
- Produces (all importable from `src.tasks.hammer.mdp.contact_row_impulse`, consumed by Task 8):
  - `reconstruct_qfrc_from_efc(env) -> torch.Tensor  # (num_envs, nv)`
  - `contact_row_qfrc(env) -> torch.Tensor  # (num_envs, nv)` — one-arg; geom ids resolved internally
  - `arm_dof_cols(env) -> torch.Tensor  # (6,) long` — the robot arm joints' columns in nv space, validated by the invariant test

- [ ] **Step 1: Write the failing invariant test**

`tests/test_contact_row_impulse.py` — the env build follows `derive_impulse_thresholds.py:59-72` (`z1_hammer_env_cfg(play=True, cat_impulse=True)`, `cfg.scene.num_envs = 1`, `ManagerBasedRlEnv(cfg, device="cpu")`), and the strike is DRIVEN with the verbatim `SingleStrikeReference` playback loop from Task 5 Step 1 (zero actions never reach the nail from the z=0.25 reset — a zero-action test is vacuous). At every control step during the drive:

```python
recon = reconstruct_qfrc_from_efc(env)                                  # (1, nv)
qfrc = env.scene["robot"].data._joint_dof_field("qfrc_constraint")      # (1, nv_robot)
cols = arm_dof_cols(env)
torch.testing.assert_close(recon[:, cols], qfrc[:, arm_joint_ids], rtol=1e-4, atol=1e-6)
```

plus a final `assert saw_contact` (from the `hammer_nail_contact` sensor) so the test cannot pass vacuously. Mark `@pytest.mark.integration`.

- [ ] **Step 2: Run to fail** — `$PY -m pytest tests/test_contact_row_impulse.py -v` → FAIL with ImportError.

- [ ] **Step 3: Implement `reconstruct_qfrc_from_efc` + `arm_dof_cols`**

In `src/tasks/hammer/mdp/contact_row_impulse.py`. Row→world attribution for num_envs=1 is trivially world 0; the sparse gather per active row r (`nefc` rows) is:

```python
nnz, adr = int(rownnz[r]), int(rowadr[r])
cols = colind.reshape(-1)[adr : adr + nnz].long()
vals = J.reshape(-1)[adr : adr + nnz]
out[world_of_row, cols] += vals * force[r]
```

Probe the actual shapes/dtypes of `J`, `J_rownnz`, `J_rowadr`, `J_colind`, `force`, `nefc` on the live env in the failing test run and **document the measured layout in the module docstring** (e.g. whether `nefc` is scalar or per-world, whether rows are world-blocked). Iterate until the invariant passes at the stated tolerance — never weaken the tolerance.

- [ ] **Step 4: Add the contact-row filter + its test**

`contact_row_qfrc(env)`: resolve geom-id sets ONCE at first call using the **prefixed** names — hammer = geoms matching `robot/hammer_head_*`; nail = `{nail_block/nail_shaft, nail_block/nail_head}` (explicitly EXCLUDING `nail_block/block_geom`). Iterate contacts `0..int(_t(d.nacon)[...])`, keep those whose geom pair spans the two sets, gather their efc rows via `contact.efc_address` (+ per-contact row count — probe `efc.type`/`contact.dim` to pin pyramidal `2·(dim−1)` vs elliptic `dim` and document it), and accumulate `JᵀF` with the same gather as Step 3. Test (same driven-strike idiom): rows-signal is nonzero on some in-contact step, exactly zero on every off-contact step, and `saw_contact` holds.

- [ ] **Step 5: The `num_envs=2` attribution test (gates Task 8's training wiring)**

Same driven test with `cfg.scene.num_envs = 2`, driving BOTH envs (broadcast the same playback action): assert the invariant per world, and assert `contact_row_qfrc` attributes forces to the correct world via `contact.worldid` (e.g. perturb env 1's playback by one step so the contact windows differ, and check the rows-signal is nonzero only for the world whose sensor reports contact). If per-world efc attribution cannot be established from the exposed fields, mark this test `xfail` with the reason, and **Task 8 must then skip Step 4 (training wiring)** per the Global Constraint.

- [ ] **Step 6: Full suite + commit**

```bash
$PY -m pytest tests/test_contact_row_impulse.py -v && $PY -m pytest tests/ -q
git add src/tasks/hammer/mdp/contact_row_impulse.py tests/test_contact_row_impulse.py
git commit -m "feat(impulse-cat): Track-2 efc contact-row isolation — exact-reconstruction invariant, prefixed-geom filter, 2-env attribution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Track 2 — `ContactRowImpulseAccumulator` (log-only metric)

**Files:**
- Modify: `src/tasks/hammer/mdp/contact_row_impulse.py` (add the accumulator class)
- Modify: `src/tasks/hammer/mdp/__init__.py` (export it)
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py` (wire `cfg.metrics["substep_impulse_rows"]` — ONLY if Task 7 Step 5 passed)
- Test: `tests/test_contact_row_impulse.py` (extend)

**Interfaces:**
- Consumes: `contact_row_qfrc(env)` and `arm_dof_cols(env)` from Task 7; the pulse-semantics pattern of `SubstepImpulseAccumulator` (`src/tasks/hammer/mdp/impulse_bound.py:68-146`).
- Produces: `env._hammer_substep_impulse_rows` attr with `.impulse -> (B, J)` and `._episode_peak -> (B,)`; metric key `substep_impulse_rows`. NOTHING consumes it for control/δ.

- [ ] **Step 1: Write the failing tests** (extend `tests/test_contact_row_impulse.py`)

(a) **Driven-strike test** — same playback idiom; after the strike: `rows_acc._episode_peak > 0`, shape matches the shipped accumulator's, and `rows_acc._episode_peak ≤ raw-signal episode peak × 1.05` (contact rows exclude friction/limits/weld — compare against a RAW-mode shipped peak, not the baseline-subtracted one, since subtraction can undershoot).
(b) **Pulse-semantics unit test** — synthetic contact-edge sequence exercising rising-edge open, falling-edge max-combine, and control-step pulse clearing, mirroring the shipped accumulator's coverage in `tests/test_impulse_bound.py` (reuse that file's mocking/synthetic-edge pattern for the sensor and the qfrc source — monkeypatch `contact_row_qfrc` to a scripted tensor sequence so the semantics are tested independent of the physics).

- [ ] **Step 2: Run to fail** — `AttributeError: _hammer_substep_impulse_rows` / ImportError.

- [ ] **Step 3: Implement the accumulator**

Same window/pulse semantics as the shipped class — open on rising contact edge, accumulate `|contact_row_qfrc(env)[:, arm_dof_cols(env)]| · env.physics_dt` while in contact, max-combine on falling edge, pulse cleared at each control-step boundary, episode-peak returned for `reduce="last"` logging. Cache `arm_dof_cols(env)` at `__init__`. Mirror `SubstepImpulseAccumulator.__init__/reset/__call__` structure line-for-line where semantics are shared (sensor gate, `_dec`, branchless masks) so the window definition is IDENTICAL to the enforced quantity's. Performance: contacts between one geom pair are rare — vectorize the filtered gather with `index_add_`; if GPU profiling at wiring time shows a training slowdown, do not wire it (fall back to gate/eval-only and flag in the report).

- [ ] **Step 4: Wire the metric (log-only) — CONDITIONAL**

Only if Task 7 Step 5 (num_envs=2 attribution) passed. In `env_cfgs.py`, directly under `substep_impulse`:

```python
    # Track-2 RIGOROUS metric: contact-row-only Λ (JᵀF over hammer↔nail efc rows) — validates the
    # enforced baseline-subtracted Λ; LOG-ONLY, never feeds joint_impulse_excess/δ.
    cfg.metrics["substep_impulse_rows"] = MetricsTermCfg(
      func=hammer_mdp.ContactRowImpulseAccumulator,
      per_substep=True,
      reduce="last",
      params={"sensor_name": "hammer_nail_contact", "robot_cfg": vb_robot_cfg},
    )
```

If Step 5 xfailed: skip this, keep the accumulator importable for the gate script (Task 9 instantiates it manually on a 1-env gate env), and record the wiring deferral in the task report.

- [ ] **Step 5: Tests + full suite + commit**

```bash
$PY -m pytest tests/test_contact_row_impulse.py tests/test_impulse_bound.py tests/test_cat_soft_hook.py -v
$PY -m pytest tests/ -q
git add src/tasks/hammer/mdp/contact_row_impulse.py src/tasks/hammer/mdp/__init__.py src/tasks/hammer/config/z1/env_cfgs.py tests/test_contact_row_impulse.py
git commit -m "feat(impulse-cat): ContactRowImpulseAccumulator — rigorous log-only Λ metric with shared window semantics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Track 2 — three-way gate section + contamination figure

**Files:**
- Modify: `docs/research/reward-design/derive_impulse_thresholds.py` (new section [6])
- Create: `docs/research/reward-design/figures/impulse_contamination.png` (generated)
- Modify: `docs/research/reward-design/validate_rewards.py` (Phase M: certify the rows accumulator)
- Modify: `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` (record Track-2 shipped)

**Interfaces:**
- Consumes: both accumulators (Tasks 3+8); `SubstepDeliveredImpulse` (object-side ∫F·dt).
- Produces: section [6] printout `raw Λ | baseline-subtracted Λ | contact-row Λ | object-side ∫F·dt` per joint/strike + the saved figure (the methods-chapter artifact).

- [ ] **Step 1: Add section [6] to the derive script** — per reference strike print the four quantities and the derived shares (`friction share = (raw − rows)/raw`, `residual after subtraction = (subtracted − rows)/rows`), and `matplotlib`-save a grouped per-joint bar chart to `docs/research/reward-design/figures/impulse_contamination.png` (`matplotlib.use("Agg")`; create the dir). **Hard gate:** `rows ≤ raw` per joint per strike (5% tolerance). **Cross-check with failure path:** compare the worst-joint contact-row Λ against the object-side ∫F·dt scale — if they disagree beyond the expected friction-residue band, print `[TRACK-2 BUG]`, treat it as a Track-2 defect (blocks Tasks 7–9 conclusions + any promotion decision, NEVER Track 1), and record the disagreement in the Step-4 impl-plan note.
- [ ] **Step 2: Extend Phase M** in `validate_rewards.py`: assert the rows accumulator (wired, or manually instantiated if Task 8 deferred wiring) is zero before contact, positive after the scripted strike, and `≤ raw` (same tolerance) — mirroring the existing Phase-M shipped-accumulator certification.
- [ ] **Step 3: Run both gates**

```bash
$PY docs/research/reward-design/derive_impulse_thresholds.py | tee /tmp/derive_with_rows.txt; echo "exit: $?"
$PY docs/research/reward-design/validate_rewards.py 2>&1 | tail -3
```

Expected: exit 0, section [6] table printed, figure file exists, `ALL PHASES PASS`.
- [ ] **Step 4: Record + commit** — impl-plan note that Track 2 shipped log-only with the three-way validation (+ any recorded disagreement); `$PY -m pytest tests/test_docs_current.py -q && $PY -m pytest tests/ -q`; `git add` the four files; commit `"feat(impulse-cat): three-way quantity validation (raw/subtracted/rows/object-side) + contamination figure"` with the co-author trailer.

---

### Task 10: C3 results collection + dated record + vacuous-check

Runs only after the Vega arrays finish (queue latency: hours; runs ~15 min each).

**Files:**
- Create: `docs/results/2026-07-XX_impulse_cat_c3.md` (date = collection date)
- Modify: `docs/results/README.md` (add row), `docs/research/reward-design/OPEN_QUESTIONS.md` (Q3/Q11 notes if informed), `docs/thesis/README.md` (C2-contribution status update)

**Interfaces:**
- Consumes: `logs/rsl_rl/z1_hammer/` runs `c3_{imp,track,catsoft}_seed{0..2}`.

- [ ] **Step 1: Rsync logs back** (or receive them from the user if no SSH): `rsync -av vega:<remote>/logs/rsl_rl/ logs/rsl_rl/`.
- [ ] **Step 2: Extract per-arm metrics** — success rate, nail depth, delivered impulse (`substep_delivered`), enforced Λ peak (`substep_impulse`), rigorous Λ peak (`substep_impulse_rows`, if wired), episode length, mean δ. Per seed + aggregated.
- [ ] **Step 3: The binding-ness / vacuous-check (spec amendment — this is a hard conclusion-gate).** Same-seed comparison `c3_imp` vs `c3_track`: δ-attributable peak-impulse reduction + fraction of strikes in `[0.8·J_limit, J_limit]`. If the reduction ≈ 0 AND δ ≈ 0 throughout training, the impulse limit was **vacuous** on this task — the record must say so explicitly, and no "enforcement works" thesis claim may be drawn (the honest fallback narrative: the constraint machinery is validated and the cap is shown non-binding for this task's learned strikes at fixture-era limits).
- [ ] **Step 4: Write the dated record** — structure of `docs/results/2026-06-18_softcat_velocity.md`: provenance (job ids, ITERS/NENVS/max_p, commit hash), numbers table, findings (Λ tail bounded? delivered-impulse cost vs `-Track`? comparison vs velocity-CaT?), the vacuous-check verdict, and the contamination-figure cross-reference.
- [ ] **Step 5: Guard + commit** — freshness guard + full suite; `git add docs/results/ docs/research/reward-design/OPEN_QUESTIONS.md docs/thesis/README.md`; commit with trailer.

---

## Execution notes

- Tasks 1→6 are strictly ordered. Tasks 7–9 run after Task 6 submission (overlapping the queue wait) and require Task 3's `IMP_J_LIMIT` for comparison printing. Task 10 waits on the external queue.
- Every task's implementer reports the standard status contract (DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED) with test evidence.
- STOP-conditions are real user-decision points, not things to code around: a red derive gate (T3), a failed strike from the new reset (T2), a probe-gate failure (T5 — report + propose, never apply), an invariant needing tolerance-weakening (T7), a Track-2 disagreement (T9).
- The plan's quoted line numbers were verified 2026-07-06 but the user edits the tree concurrently — RE-READ every anchor before editing (Global Constraints).
