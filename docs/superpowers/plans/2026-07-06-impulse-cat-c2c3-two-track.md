# Impulse-CaT C2→C3 Two-Track Enablement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the impulse soft-CaT constraint (`imp_max_p > 0`) on the baseline-subtracted Λ_j with fixture-era measured caps, run the C3 four-arm results campaign on Vega with full observability (per-joint force-propagation + cross-run comparison), and in parallel build the rigorous contact-row impulse isolation as a logged validator (never enforced, never gating Track 1).

**Architecture:** Track 1 (Tasks 1–7) is the strictly-ordered critical path: commit → verify the adopted NEAR_NAIL windup re-solve → threshold re-derivation + config flip → gate re-green → C2 local enforcement gate → **observability instrumentation** → C3 Vega submission (4 arms). Track 2 (Tasks 8–10) adds the `ContactRowImpulseAccumulator` (hammer↔nail efc rows only) during the queue wait. Tasks 11–12 are the post-queue checkpoint eval (the impl plan's C4 analogue) and the dated results record with the vacuous-check.

**Tech Stack:** mjlab 1.4.0 / mujoco 3.8.1 / mujoco_warp 3.8.1, torch, matplotlib 3.10.9, conda env `unitree_mjlab` (`~/miniconda3/envs/unitree_mjlab/bin/python`, aliased `$PY`), Slurm on EuroHPC Vega. Sibling assets repo: `~/repos/safe_impact_manipulation` (branch `hammer-z1`).

**Spec:** `docs/superpowers/specs/2026-07-06-impulse-cat-c2c3-two-track-design.md` — read it, including BOTH amendment blocks (adopted windup re-solve + probe-limit C2; round-2: the `c3_imp0` baseline, observability track, eval protocol, weight-share, probe corrections, provenance gap).

## Global Constraints

- **One GPU per Vega run, never multi-GPU**: `#SBATCH --gres=gpu:1`; fan out arms × seeds with `sbatch --array` only.
- **Enforced quantity = baseline-subtracted Λ_j** (`subtract_baseline: True`); the contact-row Λ (Track 2) is a **logged metric only** — never feeds `joint_impulse_excess` or any δ.
- **Track 2 must never gate Track 1**: a Track-2 failure blocks only Tasks 8–10, never C2/C3.
- **Track-2 training wiring is conditional**: `mujoco_warp`'s efc struct has NO `worldid` field (verified); the contact-row metric enters the *training* config only after the `num_envs=2` attribution test (Task 8 Step 5) passes — otherwise gate/eval-only.
- **`imp_seed` stays `1e-3`** (normalizer decay floor). Never paste a p95 statistic into it.
- **J_limit values are measurements**: only numbers printed by `derive_impulse_thresholds.py` on the post-re-solve tree go into config. Never reuse gripper-era numbers.
- **The enforcement override flag is verified to exist**: `--env.metrics.cat-soft.params.imp-max-p FLOAT` on `scripts/train.py` (tyro). If it ever disappears, register a task VARIANT (e.g. `Unitree-Z1-Hammer-CaT-Impulse-Enforce`) — a bare factory kwarg is not reachable from `train.py`.
- **Never edit installed packages** (`rsl_rl`, `mjlab`, `mujoco_warp`) — read-only.
- **Never penalize raw ‖Δq̇‖² / never clock-time velocity tracking near contact / impulse always accumulated at substep rate.**
- **Pre-train gate green before any GPU submission**: `validate_rewards.py` A–M + `verify_contact_sensor.py` + `verify_reward_setup.py` + the 5-file pytest set + `test_single_strike.py`.
- **Full pytest suite green before every commit**: `$PY -m pytest tests/ -q` is part of every commit step.
- **Commits stay local** — never push. Never commit `.env`, `.DS_Store`, `graphify-out/`, `.claude/settings.local.json`.
- **The user edits the tree concurrently** — RE-READ every quoted file before editing; `~` line numbers are approximate by design.
- Docs freshness guard (`tests/test_docs_current.py`) must pass after every docs edit.
- `$PY` = `~/miniconda3/envs/unitree_mjlab/bin/python`; run from `/Users/nikerane/repos/unitree_rl_mjlab` unless stated.

---

### Task 1: Commit the standing tree (both repos)

The working trees carry the docs consolidation + impulse machinery + the user's 2026-07-06 windup re-solve (~86 dirty paths here) and the fixture/grasp/mocap changes (sibling). Group into reviewable commits. **Nothing is pushed.**

**Files:** `.gitignore` additions + commits only.

**Interfaces:**
- Produces: clean `git status` in both repos; the adopted windup `NEAR_NAIL_JOINT_POS` committed → Task 2 is verification-only.

- [ ] **Step 1: Fresh inventories (do not trust counts — the tree moves)**

```bash
git status --short | tee /tmp/t1_inventory_main.txt | wc -l
cd ~/repos/safe_impact_manipulation && git status --short | tee /tmp/t1_inventory_sibling.txt
```

Expected main groups: `src/tasks/hammer/**` + `tests/**`; `docs/**` + `CLAUDE.md` + `thesis_*.md` + `.claude/settings.json`; `src/assets/robots/unitree_z1/z1_constants.py`; plus ignorable junk. Sibling: `.gitignore`(M), `hammer_z1_env/README.md`(M), `hammer_z1_env/assets/z1_hammer_robot.xml`(M), `hammer_z1_env/assets/z1_mocap_hammer.xml`(M), `hammer_z1_env/view.py`(M), untracked mesh + `CLAUDE.md` + `.claude/`. **There is NO scratchpad directory** — the windup-sweep solver was not preserved (see Step 5).

- [ ] **Step 2: Ignore the never-commit artifacts (main repo)**

Append to `.gitignore` (only lines not already present):

```
.env
.DS_Store
graphify-out/
.claude/settings.local.json
```

- [ ] **Step 3: Grouped commits in the main repo (three commits)**

```bash
$PY -m pytest tests/ -q   # green BEFORE committing; STOP on failures

git add src/tasks/hammer/ tests/
git commit -m "feat(impulse-cat): C0/C1/C5 machinery — substep accumulators, per-event pulse, soft-OR hook, guards + tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git add docs/ CLAUDE.md thesis_synthesis.md thesis_direction_update.md thesis_handoff_brief_original.md .claude/settings.json .gitignore
git commit -m "docs: consolidation — current-truth index, thesis digest+notation, freshness guard, archive-read hook, dated-record banners

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git add src/assets/robots/unitree_z1/z1_constants.py
git commit -m "feat(pose): NEAR_NAIL windup re-solve for the L6 grasp — in-plane vertical reset at (0.5,0,0.25); vertical-at-nail kinematically infeasible

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git status --short   # remaining lines must ALL be ignored junk; anything else → STOP and report
```

Never `git add -A`. Anything unexplained → STOP for a decision.

- [ ] **Step 4: Commit the sibling assets repo**

```bash
cd ~/repos/safe_impact_manipulation
git add hammer_z1_env/ .gitignore
git commit -m "feat(fixture): L6 SimplifiedLink06 holder + −90°Z grasp reseat (pos −0.0506); mocap mirror; README title fix

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short   # expected leftovers: untracked CLAUDE.md and .claude/ ONLY — leave, note in report
```

- [ ] **Step 5: Provenance gap — report item (STOP/ask, non-blocking for later tasks)**

The windup-sweep solver that produced the new `NEAR_NAIL_JOINT_POS` exists in NEITHER repo. Put a question in the task report: the user either supplies the solver file to commit, or Task 4's doc pass amends the `z1_constants.py` provenance comment with "(sweep solver not preserved; pose verified by playback + single-strike gates)".

- [ ] **Step 6: Suite still green on the committed tree**

Run: `$PY -m pytest tests/ -q -m "not integration"` → all pass.

---

### Task 2: Verify the adopted NEAR_NAIL windup re-solve

**Do NOT re-run any IK.** The user's 2026-07-06 pose (committed in Task 1): face poised at world (0.5, 0, 0.25), in-plane (joint1=0), dead-vertical; vertical-at-nail is kinematically impossible with the new grasp; the policy discovers the (necessarily oblique near the floor) contact. Read the provenance block at `src/assets/robots/unitree_z1/z1_constants.py` (~L175–197) first.

**Files:** none — verification only.

**Interfaces:**
- Produces: PASS/FAIL on strike feasibility from z=0.25 — the go/no-go for everything downstream.

- [ ] **Step 1: Reference playback gate**

Run: `$PY docs/research/reward-design/playback_reference.py 2>&1 | tee /tmp/t2_playback.txt`
Expected: the script's own gate — **best**-strike depth ≥ `NAIL_SUCCESS_THRESHOLD` (0.027 m); historically only the lowest approach clears it (that is a PASS). Record the press-probe depth as a **watchdog number** (a sustained press CAN reach threshold — expected, not a failure).

- [ ] **Step 2: Q1 re-confirm (mandatory after any EE/pose change)**

Run: `$PY docs/research/reward-design/test_single_strike.py 2>&1 | tail -5`
Expected: single strike clears 0.027 m from the new reset. If NOT: STOP — report per-approach depths; the threshold-vs-pose trade-off is the **user's decision**.

- [ ] **Step 3: Visual sanity render**

Run: `$PY scripts/render_reference.py --distance 0.85 --elevation -25` → read 2–3 PNGs under `/tmp/hammer_ref/`: arm descends from above, face (not claw) meets the nail head, no reset interpenetration.

- [ ] **Step 4: Report** — playback best-strike depth, press watchdog, single-strike verdict, one frame path. Any FAIL blocks Tasks 3–12.

---

### Task 3: Threshold re-derivation + enforcement config flip

**Files:**
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py` (re-read first; `cat_impulse` block: `substep_impulse` ~L219–228, `cat_soft` ~L242–257, `delivered_impulse` ~L261–269)
- Modify: `docs/research/reward-design/derive_impulse_thresholds.py` (docstring stale-p95 wording only)
- Modify: `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` (Status L3, L6, Pinocchio ~L128/188)

**Interfaces:**
- Consumes: verified pose (Task 2).
- Produces: `IMP_J_LIMIT` (6 floats) + `i_ref` in config, consumed by Tasks 5–7 and 11; repo default stays `imp_max_p=0.0`.

- [ ] **Step 1: Run the quantity gate**

```bash
$PY docs/research/reward-design/derive_impulse_thresholds.py | tee /tmp/derive_thresholds_fixture.txt
echo "exit: $?"
```

Expected `exit: 0`. Capture: section [4] six per-joint `J_limit` values; section [3] **MEAN** object-side ∫F·dt (→ `i_ref`; gripper-era analogue ≈0.107 N·s). On `exit: 1`: STOP, paste the failing section, touch nothing. (The oblique contact may shift the window duration substantially — that's why these are measurements.)

- [ ] **Step 2: Flip to baseline-subtracted** — in `substep_impulse` params (~L226):

```python
        "subtract_baseline": True,  # C2 (2026-07-06): enforce the baseline-subtracted Λ_j — removes
        # the dominant dof-friction share; residual quantified by the Track-2 contact-row metric.
```

- [ ] **Step 3: Write the measured caps** — **inside the `if cat_impulse:` branch**, directly above the `cfg.metrics["cat_soft"]` assignment whose params include `imp_limit` (~L242; NOT the velocity-only branch — placing it there is a `NameError` in this branch):

```python
    # Fixture-era per-joint impulse caps, MEASURED by derive_impulse_thresholds.py on 2026-07-06
    # (windup NEAR_NAIL reset, oblique contact; gate log: /tmp/derive_thresholds_fixture.txt →
    # dated record at C3). J_limit_j = τ_rated_j × 2 (HD Repeated-Peak) × Δt_impact.
    IMP_J_LIMIT = [<j1>, <j2>, <j3>, <j4>, <j5>, <j6>]  # ← paste section [4], N·m·s
```

and in `cat_soft` params:

```python
        "imp_limit": IMP_J_LIMIT,  # measured fixture-era per-joint caps (see above)
        "imp_max_p": 0.0,  # log-only default; C2/C3 raise it per-run via
        #   --env.metrics.cat-soft.params.imp-max-p (verified tyro flag)
        "imp_seed": 1e-3,  # normalizer DECAY FLOOR only — never a p95/excess statistic (hook.py)
```

- [ ] **Step 4: Set `i_ref`** — replace `"i_ref": 1.0,` with section [3]'s MEAN and change the comment to `i_ref MEASURED 2026-07-06 (gate section [3] mean); delivered_impulse SHARE measured at C2 (see /tmp/c2_gate.txt) — weight decision recorded there.`

- [ ] **Step 5: Clean the stale p95 docstring** in `derive_impulse_thresholds.py`: → `section [5] prints p95(Λ_j) for reference only — imp_seed stays a small decay floor (1e-3); the hook self-seeds from the first over-limit sample.`

- [ ] **Step 6: Record the discharged gates** in `IMPULSE_CAT_IMPL_PLAN.md`: Khadiv verbal go-ahead 2026-07-06 (C3 = formal confirmation); fixture-era re-derivation DONE (gate PASS, windup reset); **Pinocchio decision RECORDED** — object-side ∫F·dt is the accepted ground truth; `impulseDynamics` deferred indefinitely (mirrors plan-c00a0e0144274fe4). Point the ~L128/188 "revisit" lines here.

- [ ] **Step 7: Verify + commit**

```bash
$PY -m pytest tests/test_cat_soft_hook.py tests/test_impulse_constraint.py tests/test_impulse_bound.py -q
$PY docs/research/reward-design/validate_rewards.py 2>&1 | tail -5
$PY -m pytest tests/ -q
git add src/tasks/hammer/config/z1/env_cfgs.py docs/research/reward-design/derive_impulse_thresholds.py docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md
git commit -m "feat(impulse-cat): fixture-era measured J_limit + i_ref; enforce baseline-subtracted Λ; record gate decisions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If Phase M hardcodes the raw mode, fix the phase to certify the shipped config value. Note: with `imp_limit` now a 6-float list, re-check the tyro `--help` output at Task 5 (list fields can change flag inference for `imp-limit` — `imp-max-p` is unaffected).

---

### Task 4: Full pre-train gate re-green + blocker-doc sync

**Files:**
- Modify: `docs/research/reward-design/OPEN_QUESTIONS.md` (~L168, ~L179), `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` (§6b ~L178)
- Modify: memory `l6-hammer-fixture-ee.md` + `MEMORY.md` L16
- Possibly modify: `src/assets/robots/unitree_z1/z1_constants.py` (provenance amendment per Task 1 Step 5's answer)

- [ ] **Step 1: Run the entire gate**

```bash
$PY docs/research/reward-design/validate_rewards.py 2>&1 | tail -3
$PY docs/research/reward-design/verify_contact_sensor.py 2>&1 | tail -3
$PY docs/research/reward-design/verify_reward_setup.py 2>&1 | tail -5
$PY -m pytest tests/ -q
```

All green (single-strike already re-ran in Task 2). Any failure: STOP, report verbatim.

- [ ] **Step 2: Sync the blocker docs** — OPEN_QUESTIONS Q1 note (`re-measured post-L6-fixture 2026-07-06 (windup re-solve, reset z=0.25): strike feasibility re-confirmed`), Remaining blockers (`adopted + gate re-green DONE 2026-07-06; no blockers before the C2/C3 campaign`), IMPULSE_CAT §6b (adopted the user's windup pose, vertical-at-nail infeasible), memory file + index line (re-solve DONE, z=0.25 design + "bracing must emerge" open thread). Apply the provenance amendment if the user didn't supply the solver.

- [ ] **Step 3: Guard + commit**

```bash
$PY -m pytest tests/test_docs_current.py -q && $PY -m pytest tests/ -q
git add docs/research/reward-design/OPEN_QUESTIONS.md docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md src/assets/robots/unitree_z1/z1_constants.py
git commit -m "docs: blockers discharged — windup pose adopted, fixture-era gate re-green

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: C2 — local enforcement gate + max_p mini-sweep + reward-share table

**Spec amendments apply (read both).** Reference strikes measure ~5% of J_limit, so the machinery hard-gates run against a **probe limit** (`0.05 × IMP_J_LIMIT`); real caps get a **statistics report** only. **Heights run DESCENDING (0.15, 0.10, 0.06)** — the hook seeds its normalizer from the FIRST over-limit sample, so ascending order saturates δ on the first strike and fails the graded-δ gate spuriously. The δ-attributable vacuous-check happens at Task 12 (`c3_imp` vs `c3_imp0`).

**Files:**
- Create: `docs/research/reward-design/c2_enforcement_gate.py`
- Create: `docs/research/reward-design/reward_design_util.py` (**modeled on — not refactoring —** `derive_impulse_thresholds.py:main()`; the derive script is not edited here)

**Interfaces:**
- Consumes: `IMP_J_LIMIT` (Task 3); `z1_hammer_env_cfg(play=True, cat_impulse=True)`; hook extras `env.extras["cat_delta"]`; **the live hook instance accessor** `env.metrics_manager.cfg["cat_soft"].func` (mjlab replaces the class in the manager's deepcopied cfg with the constructed instance — verified; never edit `cfg.metrics` after env build expecting effect).
- Produces: `run_reference_strikes(cfg, heights, limit_override=None) -> dict` (keys: `max_excess`, `max_delta`, `cmax`, `depth`, `band_fraction`, `binding_ratio`, `reward_terms: dict[str, float]` — per-term weighted episode return); chosen `imp_max_p` for Task 7.

- [ ] **Step 1: Write the shared helper** — env-build + site helpers + the verbatim playback loop from `derive_impulse_thresholds.py` (~L59–133):

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

Per strike record worst-joint excess (vs the possibly-overridden limit), peak `env.extras["cat_delta"]`, nail depth, and the per-term **weighted episode reward sums** (read the reward manager's episode sums the way `validate_rewards.py` prints its per-term table). After all strikes snapshot the hook's `_imp_cmax`. `limit_override` (scalar factor) scales the LIVE hook's limit: `hook = env.metrics_manager.cfg["cat_soft"].func; hook._imp_limit = hook._imp_limit * limit_override` (fresh env per arm — no restore needed).

- [ ] **Step 2: Write the gate script** — `docs/research/reward-design/c2_enforcement_gate.py`:

```python
"""C2 enforcement gate (IMPULSE_CAT_IMPL_PLAN.md C2 + spec amendments 2026-07-06).

Per imp_max_p ∈ {0.25, 0.5}:
  PROBE pass (limit × 0.05, heights DESCENDING — machinery hard gates):
    [1] c_max bounded: finite, above the 1e-3 seed on ≥1 column, ≤ 100× worst observed excess.
    [2] graded δ: ≥1 over-limit strike with 0 < δ < max_p (non-degenerate grading; the FIRST
        over-limit strike seeds c_max ⇒ δ=max_p there is EXPECTED, not a failure).
    [4a] regression tripwire: nail depths identical to the log-only baseline run (δ has no physics
        pathway in playback — this catches future wiring that mutates sim state).
    [4b] δ delivery: probe pass has cat_delta > 0 on ≥1 contact step; log-only baseline δ ≡ 0.
  REPORT pass (real IMP_J_LIMIT — statistics only, no failure):
    [3] band_fraction (strikes with worst-joint Λ in [0.8·J,J]), binding_ratio (max Λ_j/J_limit_j).
  REWARD-SHARE table (once, log-only cfg): per-term weighted episode return on the reference
  strikes incl. delivered_impulse's share of the positive total — fulfils the config's "C2 tunes
  the weight" promise. Hard-gate: delivered_impulse fires, positive, finite. The weight decision
  itself is the USER's — surface the share and stop short of changing weight.
Usage: $PY docs/research/reward-design/c2_enforcement_gate.py   (exits 1 on FAIL)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg

sys.path.insert(0, str(Path(__file__).resolve().parent))  # reward-design is not a package
from reward_design_util import run_reference_strikes  # noqa: E402

MAX_PS = (0.25, 0.5)
HEIGHTS = (0.15, 0.10, 0.06)  # DESCENDING — see module docstring
PROBE_SCALE = 0.05


def build_cfg(imp_max_p: float):
    cfg = z1_hammer_env_cfg(play=True, cat_impulse=True)
    cfg.scene.num_envs = 1
    cfg.metrics["cat_soft"].params["imp_max_p"] = imp_max_p
    return cfg


def main() -> int:
    failures: list[str] = []
    base = run_reference_strikes(build_cfg(0.0), heights=HEIGHTS)

    print("=== REWARD SHARE (log-only cfg, reference strikes) ===")
    total_pos = sum(v for v in base["reward_terms"].values() if v > 0)
    for name, v in sorted(base["reward_terms"].items(), key=lambda kv: -abs(kv[1])):
        share = (v / total_pos * 100) if total_pos > 0 else float("nan")
        print(f"  {name:24s} {v:12.4f}   {share:6.1f}% of positive")
    di = base["reward_terms"].get("delivered_impulse", 0.0)
    if not (di > 0 and torch.isfinite(torch.tensor(di))):
        failures.append(f"delivered_impulse term degenerate on reference strikes: {di}")
    if max(base["max_delta"]) != 0.0:
        failures.append(f"log-only baseline has nonzero δ: {base['max_delta']}")

    for mp in MAX_PS:
        probe = run_reference_strikes(build_cfg(mp), heights=HEIGHTS, limit_override=PROBE_SCALE)
        print(f"\n=== PROBE imp_max_p={mp} (limit × {PROBE_SCALE}, heights descending) ===")
        print(f"excess: {probe['max_excess']}\nδ:      {probe['max_delta']}\nc_max:  {probe['cmax']}")
        cmax = torch.as_tensor(probe["cmax"])
        worst_excess = max(probe["max_excess"])
        if not torch.isfinite(cmax).all() or (cmax <= 1e-3).all():
            failures.append(f"[{mp}] c_max degenerate (pinned at seed / non-finite): {probe['cmax']}")
        if worst_excess > 0 and (cmax > 100 * worst_excess).any():
            failures.append(f"[{mp}] c_max single-event-spiked (>100× worst excess {worst_excess:.3g})")
        over_deltas = [d for e, d in zip(probe["max_excess"], probe["max_delta"]) if e > 0]
        if len(over_deltas) < 2:
            failures.append(f"[{mp}] probe produced <2 over-limit strikes — probe scale too loose")
        elif not any(0.0 < d < mp - 1e-6 for d in over_deltas):
            failures.append(f"[{mp}] δ degenerate: no over-limit strike with 0 < δ < max_p: {over_deltas}")
        if not any(d > 0 for d in probe["max_delta"]):
            failures.append(f"[{mp}] probe never delivered δ>0 on contact — hook wiring broken")
        if not torch.allclose(torch.as_tensor(probe["depth"]), torch.as_tensor(base["depth"]), atol=1e-6):
            failures.append(f"[{mp}] enforcement branch mutated sim state: {probe['depth']} vs {base['depth']}")

        report = run_reference_strikes(build_cfg(mp), heights=HEIGHTS)
        print(f"=== REPORT imp_max_p={mp} (real caps) ===")
        print(f"band_fraction [0.8·J,J]: {report['band_fraction']:.3f}")
        print(f"binding_ratio max Λ/J:   {report['binding_ratio']:.3f}  (reference strikes are gentle by design; the LEARNED policy is what approaches the cap — vacuous-check at C3 analysis)")

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

Expected: reward-share table + `C2 GATE PASS`, exit 0. On FAIL: **STOP and report; propose a `tau`/robust-statistic change in the report — do not apply it, never touch `IMP_J_LIMIT`.** Surface the delivered_impulse share to the user as an explicit weight decision (record the answer at the top of `/tmp/c2_gate.txt`).

- [ ] **Step 4: CPU smoke-train under enforcement**

```bash
$PY scripts/train.py Unitree-Z1-Hammer-CaT-Impulse --gpu-ids '[]' --env.scene.num-envs 8 --agent.max-iterations 5 --agent.logger tensorboard --env.metrics.cat-soft.params.imp-max-p 0.5 2>&1 | tail -5
```

Expected: 5 iterations, no exception, reward not NaN. (Flag verified to exist; re-confirm with `--help` if tyro re-inferred fields after the `imp_limit` list change.)

- [ ] **Step 5: Pick `imp_max_p` + commit** — prefer `0.5` (matches the velocity arm); `0.25` if 0.5 degenerates in the probe.

```bash
$PY -m pytest tests/ -q
git add docs/research/reward-design/c2_enforcement_gate.py docs/research/reward-design/reward_design_util.py
git commit -m "feat(impulse-cat): C2 gate — descending-height probe, graded-δ + c_max + δ-delivery hard gates, reward-share table

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Observability — per-joint Λ metrics, δ-peak metric, strike diagnostics, cross-run comparison

**Must land BEFORE the Vega submission** — the training runs must carry the per-joint keys. Grounded facts: mjlab metric terms must return shape `(num_envs,)` (a `(B,J)` return raises in `_check_term_shape`); `Episode_Metrics/*` is the env-MEAN at reset; a **full-step** term with `reduce="last"` logs its exact compute-time value (the per-substep `reduce="last"` path logs a substep-mean — which under-reports peaks in the terminal step, one reason these reader terms are the authoritative peaks); `MetricsManager` evaluates terms in cfg insertion order.

**Files:**
- Modify: `src/tasks/hammer/mdp/impulse_bound.py` (per-joint peak buffer + reader; δ-peak reader)
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py` (register 7 new metric terms in the `cat_impulse` block)
- Create: `scripts/diag_impulse_trace.py`
- Create: `scripts/compare_runs.py`
- Test: `tests/test_impulse_bound.py` (extend)

**Interfaces:**
- Consumes: `SubstepImpulseAccumulator` internals (`_pulse`, `_running`, `reset`), `env.extras["cat_delta"]`, `_ENV_SUBSTEP_IMPULSE_ATTR`.
- Produces: TB keys `Episode_Metrics/imp_peak_joint1..joint6` and `Episode_Metrics/cat_delta_peak` (consumed by Tasks 11–12); `scripts/diag_impulse_trace.py --ckpt <pt> | --reference [--j-limit CSV] [--out DIR]`; `scripts/compare_runs.py --runs <glob> [--j-limit CSV] [--out DIR]`.

- [ ] **Step 1: Failing tests** (extend `tests/test_impulse_bound.py`, following its existing synthetic-edge pattern): (a) after a synthetic contact window, `acc._episode_peak_perjoint` equals the per-joint window Λ and survives until `reset()`; (b) `joint_impulse_peak(env, joint=k)` returns column k; (c) `cat_delta_peak(env)`'s running max follows a scripted `env.extras["cat_delta"]` sequence and zeroes on reset.

- [ ] **Step 2: Implement in `impulse_bound.py`**

In `SubstepImpulseAccumulator.__init__`: `self._episode_peak_perjoint = torch.zeros(env.num_envs, J, device=env.device)`; in `__call__` next to the existing peak update: `torch.maximum(self._episode_peak_perjoint, torch.maximum(self._pulse, self._running), out=self._episode_peak_perjoint)`; zero it in `reset()`. Then module-level readers:

```python
def joint_impulse_peak(env, joint: int) -> torch.Tensor:
    """Full-step metric: episode-peak Λ for ONE arm joint, from the stashed accumulator.
    reduce="last" on a FULL-STEP term logs the exact compute-time value of the monotone buffer —
    the authoritative per-joint episode peak (the per-substep worst-joint scalar is substep-mean
    diluted in the terminal control step)."""
    acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
    if acc is None:
        raise RuntimeError("joint_impulse_peak requires the SubstepImpulseAccumulator metric (cat_impulse).")
    return acc._episode_peak_perjoint[:, joint]


class CatDeltaPeak(ManagerTermBase):
    """Full-step metric: episode-peak δ from env.extras['cat_delta'] (episode-MEAN δ dilutes
    strike-time δ by ~episode length). Register AFTER cfg.metrics['cat_soft'] — the manager
    evaluates in insertion order, so the hook has written this step's δ."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._peak = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._peak[idx] = 0.0

    def __call__(self, env, **params) -> torch.Tensor:
        delta = env.extras.get("cat_delta")
        if delta is not None:
            torch.maximum(self._peak, delta.reshape(self._peak.shape), out=self._peak)
        return self._peak
```

- [ ] **Step 3: Register the metrics** in `env_cfgs.py`, inside the `cat_impulse` block AFTER the `cfg.metrics["cat_soft"]` assignment:

```python
    # Per-joint episode-peak Λ (TB: Episode_Metrics/imp_peak_joint1..6) — the authoritative peaks;
    # J_limit differs 2× across joints (joint2 τ_rated=60), so worst-joint alone can't be compared
    # to the cap vector. cat_delta_peak: peak binding pressure (episode-mean δ dilutes strikes).
    for _j, _jn in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
      cfg.metrics[f"imp_peak_{_jn}"] = MetricsTermCfg(
        func=hammer_mdp.joint_impulse_peak, per_substep=False, reduce="last", params={"joint": _j},
      )
    cfg.metrics["cat_delta_peak"] = MetricsTermCfg(
      func=hammer_mdp.CatDeltaPeak, per_substep=False, reduce="last", params={},
    )
```

Export the new symbols from `src/tasks/hammer/mdp/__init__.py`.

- [ ] **Step 4: `scripts/diag_impulse_trace.py`** — the impulse analogue of `diag_policy_trace.py` (the user's force-propagation figure). Args: `--ckpt <model.pt>` XOR `--reference`; `--task Unitree-Z1-Hammer-CaT-Impulse` (default — required so both accumulators + hook exist); `--num-envs 1 --env-idx 0 --device cpu --out /tmp/impulse_trace`; `--j-limit j1,...,j6` (CSV; REQUIRED to draw cap lines — never default to the 0.1 placeholder; omit → no lines). Mechanism: reuse the `compute_substep` monkeypatch pattern **verbatim** from `derive_impulse_thresholds.py:100-111` (hook AFTER the shipped accumulators), recording per substep: per-joint `|qfrc_constraint_j|`, object-side `F_axial`, the shipped accumulator's `impulse` (B,6), delivered total, contact flag, per-joint `|q̇_j|`, and (control-rate, forward-filled) `env.extras["cat_delta"]`. Checkpoint mode loads the policy exactly as `diag_policy_trace.py` does (`load_runner_cls` + `runner.get_inference_policy`); reference mode uses the Task-5 playback loop. Output: one matplotlib figure, 5 stacked substep-time panels — (1) per-joint |qfrc| (the propagation down the chain, one line per joint), (2) F_axial, (3) running Λ_j with dashed J_limit lines, (4) δ, (5) per-joint |q̇| — saved as `trace.png` + raw arrays as `trace.npz`.

- [ ] **Step 5: `scripts/compare_runs.py`** — cross-run training curves. Args: `--runs 'logs/rsl_rl/z1_hammer/*c3_*'` (glob or explicit dirs), `--out /tmp/compare_runs`, `--last-frac 0.2`, optional `--j-limit CSV`. Read TB event files with `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` (verified importable in the env). Arm label = dir basename minus leading `^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_` and trailing `_seed\d+` (the `train_array.sbatch --agent.run-name ${RUN}_seed${SEED}` contract). **Guard every key** with `key in ea.Tags()["scalars"]` — `c3_track`/`c3_catsoft` lack the impulse keys; omit an arm from a panel rather than crash. Panels (2×3, per-arm mean±std bands across seeds via `np.interp` onto a common iteration grid): mean reward; success rate; `Episode_Metrics/substep_delivered`; worst-joint peak = max over `Episode_Metrics/imp_peak_joint*` (with J_limit lines if given); `Episode_Metrics/cat_delta_peak` (+ note `Episode_Metrics/cat_soft` is the diluted episode-mean); episode length. Also emit a `summary.csv` of last-`frac` means per run.

- [ ] **Step 6: Verify + commit**

```bash
$PY -m pytest tests/test_impulse_bound.py -q
$PY docs/research/reward-design/validate_rewards.py 2>&1 | tail -3     # A–M still green with 7 new metrics
$PY scripts/diag_impulse_trace.py --reference --out /tmp/impulse_trace_ref   # produces trace.png on the open-loop strike
$PY -m pytest tests/ -q
git add src/tasks/hammer/mdp/impulse_bound.py src/tasks/hammer/mdp/__init__.py src/tasks/hammer/config/z1/env_cfgs.py scripts/diag_impulse_trace.py scripts/compare_runs.py tests/test_impulse_bound.py
git commit -m "feat(obs): per-joint Λ peaks + δ-peak metrics, impulse strike-trace figure, cross-run comparison tool

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Read `/tmp/impulse_trace_ref/trace.png` and confirm the five panels render with sane data (contact spike visible in panels 1–3).

---

### Task 7: C3 — Vega submission (FOUR arms × 3 seeds)

**Files:**
- Modify: `docs/VEGA_TRAINING_PLAN.md` (append the C3 campaign block)

**Interfaces:**
- Consumes: chosen `imp_max_p` (Task 5); observability metrics (Task 6) — they must be in these runs' logs; green gate (Task 4).
- Produces: run names `c3_imp_seed{0..2}`, `c3_imp0_seed{0..2}`, `c3_track_seed{0..2}`, `c3_catsoft_seed{0..2}` consumed by Tasks 11–12.

- [ ] **Step 1: Append the C3 block to `docs/VEGA_TRAINING_PLAN.md`**

```markdown
## C3 campaign (2026-07-06) — impulse-CaT enforcement results
One GPU per run (`--gres=gpu:1`), 3 seeds via array, ITERS=5000, NENVS=4096.
    ITERS=5000 NENVS=4096 RUN=c3_imp     TASK=Unitree-Z1-Hammer-CaT-Impulse EXTRA="--env.metrics.cat-soft.params.imp-max-p <CHOSEN>" sbatch --array=0-2 scripts/slurm/train_array.sbatch
    ITERS=5000 NENVS=4096 RUN=c3_imp0    TASK=Unitree-Z1-Hammer-CaT-Impulse                                                          sbatch --array=0-2 scripts/slurm/train_array.sbatch
    ITERS=5000 NENVS=4096 RUN=c3_track   TASK=Unitree-Z1-Hammer-Track                                                                sbatch --array=0-2 scripts/slurm/train_array.sbatch
    ITERS=5000 NENVS=4096 RUN=c3_catsoft TASK=Unitree-Z1-Hammer-CaT-Soft                                                             sbatch --array=0-2 scripts/slurm/train_array.sbatch
c3_imp0 = SAME task, repo-default imp_max_p=0 (CatPPO at δ≡0 ≡ stock PPO): the same-reward,
same-algorithm, same-seed UNCONSTRAINED baseline — the vacuous-check pairs c3_imp vs c3_imp0.
c3_track (r_imit-only) and c3_catsoft (velocity CaT, same-tree re-run) are secondary comparators.
```

- [ ] **Step 2: Sync + submit** — both repos to Vega at the committed state (siblings, `scripts/slurm/setup_vega.sh`). If `ssh vega true` succeeds: rsync + submit the four lines from the login node. **If not: emit the exact commands in the task report and hand submission to the user.**

- [ ] **Step 3: Commit**

```bash
$PY -m pytest tests/ -q
git add docs/VEGA_TRAINING_PLAN.md
git commit -m "docs(vega): C3 impulse-CaT campaign — 4 arms x 3 seeds incl. c3_imp0 unconstrained baseline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Track 2 — efc layout probe + exact-reconstruction invariant

Pin the sparse efc→joint mapping empirically; lock it with `Σ_all_rows (JᵀF) == qfrc_constraint`. **Known API facts (encode, do not rediscover):** active contact count is `d.nacon` (NOT `ncon`; `contact.geom.shape[0]` is allocated capacity `naconmax`); the efc struct has NO `worldid` (fields: `type,id,J_rownnz,J_rowadr,J_colind,J,pos,margin,D,vel,aref,frictionloss,force,state,Ma,Jqvel`); `d.contact` HAS `worldid`; compiled geom names are **entity-prefixed**: `robot/hammer_head_0`, `robot/hammer_head_1`, `nail_block/nail_shaft`, `nail_block/nail_head` (exclude `nail_block/block_geom`).

**Files:**
- Create: `src/tasks/hammer/mdp/contact_row_impulse.py` (helpers only)
- Test: `tests/test_contact_row_impulse.py`

**Interfaces:**
- Consumes: `env.sim.wp_data`; `warp.to_torch`; the Task-5 playback loop (inline it in tests — the hyphenated docs dir is not importable from tests; zero actions never reach the nail from z=0.25).
- Produces (imported by Task 9): `reconstruct_qfrc_from_efc(env) -> (num_envs, nv)`; `contact_row_qfrc(env) -> (num_envs, nv)` (one-arg; geom ids resolved internally); `arm_dof_cols(env) -> (6,) long` (validated by the invariant test).

- [ ] **Step 1: Failing invariant test** — env built as in `derive_impulse_thresholds.py:59-72` (1 env, cpu, `cat_impulse=True`), strike DRIVEN with the verbatim playback loop; at every control step:

```python
recon = reconstruct_qfrc_from_efc(env)
qfrc = env.scene["robot"].data._joint_dof_field("qfrc_constraint")
cols = arm_dof_cols(env)
torch.testing.assert_close(recon[:, cols], qfrc[:, arm_joint_ids], rtol=1e-4, atol=1e-6)
```

plus a final `assert saw_contact`. Mark `@pytest.mark.integration`.

- [ ] **Step 2: Run to fail** (ImportError).

- [ ] **Step 3: Implement `reconstruct_qfrc_from_efc` + `arm_dof_cols`** — per active row r (`nefc` rows):

```python
nnz, adr = int(rownnz[r]), int(rowadr[r])
cols = colind.reshape(-1)[adr : adr + nnz].long()
vals = J.reshape(-1)[adr : adr + nnz]
out[world_of_row, cols] += vals * force[r]
```

Probe actual shapes/dtypes on the live env in the failing run and **document the measured layout in the module docstring** (is `nefc` scalar or per-world; are rows world-blocked). Never weaken the tolerance.

- [ ] **Step 4: Contact-row filter + test** — `contact_row_qfrc(env)`: geom-id sets from the prefixed names above, contacts `0..int(d.nacon)`, keep pairs spanning the two sets, rows via `contact.efc_address` (+ per-contact row count — probe `efc.type`/`contact.dim` to pin pyramidal `2·(dim−1)` vs elliptic `dim`, document it), same gather as Step 3. Test: rows-signal nonzero on some in-contact step, exactly zero off-contact, `saw_contact` holds.

- [ ] **Step 5: `num_envs=2` attribution test (gates Task 9's training wiring)** — `SingleStrikeReference(2, env.device, ...)` is batched, but `playback_target(k)` takes one scalar k for ALL envs; give the two envs different phases by row-selecting two calls:

```python
t = ref.playback_target(min(k, n)); t_lag = ref.playback_target(min(max(k - 1, 1), n))
target = torch.cat([t[:1], t_lag[1:2]])
```

(actions `torch.zeros(2, …)`; run to fixed `n + HOLD_STEPS` and detect contact windows per world — no `[0]`-only break). Assert the invariant per world and that `contact_row_qfrc` attributes force to the world whose sensor reports contact (via `contact.worldid`). If per-world efc attribution cannot be established from the exposed fields: mark `xfail` with the reason — **Task 9 must then skip its training wiring step.**

- [ ] **Step 6: Full suite + commit**

```bash
$PY -m pytest tests/test_contact_row_impulse.py -v && $PY -m pytest tests/ -q
git add src/tasks/hammer/mdp/contact_row_impulse.py tests/test_contact_row_impulse.py
git commit -m "feat(impulse-cat): Track-2 efc contact-row isolation — exact-reconstruction invariant, prefixed-geom filter, 2-env attribution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Track 2 — `ContactRowImpulseAccumulator` (log-only metric)

**Files:**
- Modify: `src/tasks/hammer/mdp/contact_row_impulse.py` (accumulator class), `src/tasks/hammer/mdp/__init__.py` (export)
- Modify: `src/tasks/hammer/config/z1/env_cfgs.py` (wire `substep_impulse_rows` — ONLY if Task 8 Step 5 passed)
- Test: `tests/test_contact_row_impulse.py` (extend)

**Interfaces:**
- Consumes: `contact_row_qfrc(env)`, `arm_dof_cols(env)` (Task 8); the pulse-semantics pattern of `SubstepImpulseAccumulator` (`impulse_bound.py:68-146`).
- Produces: `env._hammer_substep_impulse_rows` with `.impulse -> (B, J)`, `._episode_peak -> (B,)`; metric key `substep_impulse_rows`. NOTHING consumes it for control/δ.

- [ ] **Step 1: Failing tests** — (a) driven-strike test: `rows_acc._episode_peak > 0`, shape matches, and ≤ 1.05 × a RAW-signal shipped peak (build the comparison env with `cfg.metrics["substep_impulse"].params["subtract_baseline"] = False` — subtraction can undershoot, raw is the physical upper bound); (b) pulse-semantics unit test: monkeypatch `contact_row_qfrc` to a scripted tensor sequence and assert rising-edge open, falling-edge max-combine, control-step pulse clearing (mirror `tests/test_impulse_bound.py`'s synthetic pattern).

- [ ] **Step 2: Run to fail.**

- [ ] **Step 3: Implement** — same window/pulse semantics as the shipped class, accumulating `|contact_row_qfrc(env)[:, arm_dof_cols(env)]| · env.physics_dt` (cache the cols at `__init__`); mirror `__init__/reset/__call__` structure line-for-line where shared (sensor gate, `_dec`, branchless masks) so the window definition is IDENTICAL to the enforced quantity's. Vectorize the filtered gather (`index_add_`); if GPU profiling at wiring time shows a slowdown, don't wire — gate/eval-only + flag in the report.

- [ ] **Step 4: Wire the metric — CONDITIONAL on Task 8 Step 5**

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

If Step 5 xfailed: skip wiring, keep the class importable (Task 10 instantiates it on a 1-env gate env), record the deferral.

- [ ] **Step 5: Tests + full suite + commit** (same shape as before; commit message `"feat(impulse-cat): ContactRowImpulseAccumulator — rigorous log-only Λ metric with shared window semantics"`).

---

### Task 10: Track 2 — three-way gate section + contamination figure

**Files:**
- Modify: `docs/research/reward-design/derive_impulse_thresholds.py` (new section [6]); `docs/research/reward-design/validate_rewards.py` (Phase M extension); `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` (record Track-2 shipped)
- Create: `docs/research/reward-design/figures/impulse_contamination.png` (generated)

- [ ] **Step 1: Section [6]** — per reference strike print `raw Λ | baseline-subtracted Λ | contact-row Λ | object-side ∫F·dt` per joint + derived shares (`friction share = (raw − rows)/raw`, `residual after subtraction = (subtracted − rows)/rows`); save a grouped per-joint bar chart (`matplotlib.use("Agg")`). **Hard gate:** `rows ≤ raw` (5% tolerance). **Failure path:** if contact-row Λ disagrees with the object-side ∫F·dt beyond the friction-residue band, print `[TRACK-2 BUG]` — blocks Tasks 8–10 conclusions + any promotion decision, NEVER Track 1; record it in Step 4's note.
- [ ] **Step 2: Phase M extension** — rows accumulator (wired or manually instantiated) zero before contact, positive after the scripted strike, ≤ raw.
- [ ] **Step 3: Run both** — derive script (exit 0, section [6] + figure) and validate_rewards (`ALL PHASES PASS`).
- [ ] **Step 4: Record + commit** — impl-plan Track-2 note (+ any disagreement); freshness guard + full suite; commit `"feat(impulse-cat): three-way quantity validation (raw/subtracted/rows/object-side) + contamination figure"`.

---

### Task 11: C3 checkpoint eval — per-joint peaks + delivered impulse (the C4 analogue)

Runs after the Vega arrays finish. Training logs alone cannot yield per-joint tails (`Episode_Metrics/*` are env-means) and the non-impulse arms lack the impulse keys entirely — this eval closes both gaps by rolling EVERY checkpoint in the SAME instrumented env.

**Files:**
- Create: `scripts/eval_impulse.sh` (thin driver over `scripts/diag_impulse_trace.py` + a summary aggregator; the June `eval_peak_qv.sh` stays frozen as the historical velocity protocol)

**Interfaces:**
- Consumes: rsync'd checkpoints `logs/rsl_rl/z1_hammer/*c3_*_seed*/model_4999.pt`; `diag_impulse_trace.py` (Task 6); `IMP_J_LIMIT` (Task 3).
- Produces: `/tmp/eval_impulse/summary.csv` + per-checkpoint trace figures, consumed by Task 12.

- [ ] **Step 1: Rsync logs+checkpoints back** (or receive from the user if no SSH): `rsync -av vega:<remote>/logs/rsl_rl/ logs/rsl_rl/`.
- [ ] **Step 2: Pinned protocol** (write it into the script header AND the record): eval env = `Unitree-Z1-Hammer-CaT-Impulse` play cfg with `imp_max_p=0` (log-only hook = pure instrumentation; obs space identical across arms — the velocity campaign's same-env cross-arm protocol); checkpoint = final `model_4999.pt` per run; **256 envs × enough steps for ≥2 episodes each (≥512 episodes per policy)**; mean-action rollout primary, one sampled-action repeat as robustness check; fixed eval seed (record it); eval job/host recorded.
- [ ] **Step 3: Extract per checkpoint** — per-joint Λ max & p95 (from the accumulator's `(B,6)` buffer), worst-joint Λ_j/J_limit_j, delivered impulse mean±std, success rate, nail depth, episode length → `summary.csv` (rows = 12 checkpoints), plus one `diag_impulse_trace.py --ckpt` figure per arm (seed 0) for the force-propagation panel.
- [ ] **Step 4: Commit** the script + a pointer note (figures/csv stay under /tmp until the record cites them; copy the keepers into `docs/results/assets/` in Task 12). Full suite + trailer.

---

### Task 12: C3 dated record + vacuous-check

**Files:**
- Create: `docs/results/2026-07-XX_impulse_cat_c3.md` (+ `docs/results/assets/` for the kept figures)
- Modify: `docs/results/README.md` (row), `docs/research/reward-design/OPEN_QUESTIONS.md` (Q3/Q11 if informed), `docs/thesis/README.md` (C2-contribution status)

- [ ] **Step 1: Cross-run curves** — `$PY scripts/compare_runs.py --runs 'logs/rsl_rl/z1_hammer/*c3_*' --j-limit <IMP_J_LIMIT CSV> --out /tmp/compare_c3`; keep the figure.
- [ ] **Step 2: The two mandated maximize-side artifacts** — (a) delivered-impulse training curve, `c3_imp0` overlaid on `c3_imp` (mean ± seed band): rising curve = "learned to strike harder"; the converged gap = the delivered-impulse **cost of the cap**; (b) eval frontier from Task 11's csv: per-checkpoint delivered impulse vs worst-joint Λ/Λ̄ scatter, arms color-coded.
- [ ] **Step 3: The vacuous-check (hard conclusion-gate)** — same-seed `c3_imp` vs `c3_imp0`: δ-attributable peak-impulse reduction (per-joint eval peaks) + band-fraction. If reduction ≈ 0 AND δ ≈ 0 throughout training (use `cat_delta_peak`, not the diluted mean): the limit was **vacuous** on this task — the record must say so and no "enforcement works" claim may be drawn (honest fallback: machinery validated; cap shown non-binding for fixture-era limits on this task). Note: mean δ is n/a for `c3_track` (no hook); `c3_imp0`'s δ≡0 log doubles as the no-op cross-check.
- [ ] **Step 4: Write the record** — structure of `docs/results/2026-06-18_softcat_velocity.md`: provenance (job ids, ITERS/NENVS/max_p, commit hash, eval protocol + seed), numbers table (per arm mean±std across seeds), the two artifacts, per-joint Λ table vs caps, vacuous-check verdict, contamination-figure cross-ref, secondary comparisons (`c3_track` strike quality, `c3_catsoft` mechanism contrast).
- [ ] **Step 5: Guard + commit** — freshness guard + full suite; `git add docs/results/ docs/research/reward-design/OPEN_QUESTIONS.md docs/thesis/README.md`; commit with trailer.

---

## Execution notes

- Order: Tasks 1→7 strict; Tasks 8–10 during the queue wait; Task 11 after the queue; Task 12 last.
- Every implementer reports DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED with test evidence.
- STOP-conditions are user-decision points, never code-arounds: red derive gate (T3), failed strike from the new reset (T2), probe-gate failure or the delivered_impulse weight-share decision (T5), tolerance-weakening temptation (T8), Track-2 disagreement (T10), vacuous verdict (T12).
- RE-READ every quoted anchor before editing — the user edits the tree concurrently.
