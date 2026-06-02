# Impact-Progress Reward — Implementation Spec (changes #1 + #2)

**Date:** 2026-06-02 · **Branch:** `hammer-z1` · **Status:** design approved, pre-implementation
**Source of truth for the design:** `hammering_reward_design_deep_dive_v2.md` §4.3, §4.7, §5.A.

## Scope

**In:**
- **#1** Rebalance: `nail_depth_delta` weight `2000 → 600`.
- **#2** New reward term: `impact_progress` (double-gated momentum reward).

**Out (deferred — not this change):** domain randomisation (#3), privileged/asymmetric critic (#4),
`action_rate` −0.01→−0.02, Adroit `time_penalty`, PBRS shaping, the reward machine, and anything
repetitive-strike (gated on `test_single_strike.py` / Q1).

## Change #1 — Rebalance `nail_depth_delta`

- **File:** `src/tasks/hammer/hammer_env_cfg.py` → `rewards["nail_depth_delta"].weight`: `2000.0 → 600.0`.
- **Code comment:** `# 2000 -> 600 (A1 rebalance); A0 baseline = 2000`.
- **Why:** at 2000 the cumulative dense reward ≈ `2000·(0.07−0.004)=132 > 100` completion, and a single
  ~66 mm strike spikes +132 in one step — inverting the intended *completion-is-dominant* hierarchy and
  raising value-target variance. At 600 the full-drive cumulative is ≈30–40, restoring `completion=100`
  as the dominant attractor.
- **Ablation:** A0 (2000) ↔ A1 (600) is a one-constant flip. `validate_rewards.py` reads the weight
  dynamically, so its phase-C/E/F expected values auto-rescale — no test edit needed for the weight change.

## Change #2 — `ImpactProgressTerm`

**File:** `src/tasks/hammer/mdp/rewards.py` — new `class ImpactProgressTerm(ManagerTermBase)`, modelled on
the existing `NailDepthDeltaTerm` (stateful `__init__`/`reset`/`__call__`).

**Reward:**

```
r_impact = (v_axial / v_expected) · 1[first_contact] · 1[Δdepth > ε]
```

- `v_axial = max(0, n̂ · ẋ_head)`, `n̂ = (0,0,-1)` — **verified** against `nail_block_scene.xml:26`
  (`<joint name="nail_slide" type="slide" axis="0 0 -1">`). Downward head motion → positive `v_axial`.
- `ẋ_head` via **finite difference** of `site_pos_w` (`(head − prev_head)/dt`), **not** `site_vel_w`
  (deliberate — Q10 lazy-velocity hazard). Zeroed on the first post-reset step via an `_init` mask so the
  reset teleport cannot spike velocity.
- `first_contact = sensor.compute_first_contact(env.step_dt).any(-1)` — true only on the control step a
  contact begins (`current_contact_time>0 & <dt+tol`).
- `Δdepth > ε` with `ε = 5e-4` m; `prev_depth` **max-tracked** (like `NailDepthDeltaTerm`) so a scrape
  after peak depth cannot re-fire the gate.
- `v_expected = 1.0` m/s — normalises the bonus to O(1) per strike before weighting (§4.7.4). Numerically
  identical to the raw skeleton at the default, but makes `weight=8.0` interpretable and the scale tunable.

**State / reset:** `_prev_head (B,3)`, `_prev_depth (B,)`, `_init (B,) bool`. `reset(env_ids)` handles
`env_ids is None` (full reset) and a subset; sets `_init[ids]=False`, `_prev_depth[ids]=0`.

**Config wiring** (`hammer_env_cfg.py` rewards dict):

```python
"impact_progress": RewardTermCfg(
    func=hammer_mdp.ImpactProgressTerm,
    weight=8.0,
    params={
        "sensor_name": "hammer_nail_contact",
        "robot_cfg": SceneEntityCfg("robot", site_names=()),   # head site — wired per-robot
        "nail_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
        "axis": (0.0, 0.0, -1.0),
        "eps": 5e-4,
        "v_expected": 1.0,
    },
),
```

**Per-robot wiring** (`config/z1/env_cfgs.py`), mirroring the `approach` reward:

```python
cfg.rewards["impact_progress"].params["robot_cfg"].site_names = (HAMMER_HEAD_SITE_NAME,)
```

## Verification (all local / CPU, in TDD order)

1. **`tests/test_impact_progress_reward.py`** (NEW, **no marker** → runs under `pytest -m "not integration"`).
   Stub env (dict `scene` + `SimpleNamespace` robot/nail with `.data` tensors + a sensor exposing
   `compute_first_contact`). Cases:
   - **scrape → 0:** `first_contact=True` but `Δdepth ≤ ε`; depth-at-peak; `v_axial=0` (horizontal motion).
   - **strike → >0:** `first_contact=True` + downward velocity + `Δdepth > ε`.
   - **reset** zeroes `_prev_depth`/`_init` (no cross-episode leak).
   Written **red first**, then implement to green.
2. **`validate_rewards.py` + new Phase I** for the double-gate (no-impact→0, first-impact→>0,
   repeat-accumulate, reset-zeroes-state), per `REWARD_VALIDATION_METHODOLOGY.md`. CPU-pinned; existing
   8 phases pass unchanged.
3. **`verify_contact_sensor.py`** — no reward dependency; passes unchanged.
4. **`verify_reward_setup.py`** — **exempt `impact_progress`** from the "every term fires under random
   policy in 200 steps" assertion (it is an *event* reward, designed to stay silent until a productive
   strike — a random policy will essentially never trigger it). Add to an exemption list with a comment.
5. **CPU smoke train (~5 min)** — confirm the loop runs with the 7-term reward, signal flows, checkpoint saves:
   ```bash
   python scripts/train.py Unitree-Z1-Hammer --gpu-ids '[]' \
     --agent.max-iterations 50 --agent.save-interval 25 --env.scene.num-envs 16
   ```

## Doc updates

- `README.md` §Reward design — **DONE** (rendered LaTeX, 7 terms; phantom `strike_vel` term removed).
- `CLAUDE.md` "Live weights" line — update `nail_depth_delta 2000→600`, add `impact_progress=8`
  (**after** implementation, so docs match code).

## Git + training handoff

- After **all** local verification passes: commit + push branch `hammer-z1`.
- Real **A0/A1/A3 ablation** training runs on **Lightning AI cloud (T4 GPU)**; default `--gpu-ids [0]`
  works there. A0=`{depth_delta:2000, impact:0}`, A1=`{600,0}`, A3=`{600,8}` — all reachable by editing
  two weights; mjlab logs per-term episode sums automatically.

## Acceptance criteria

- New unit test green; `validate_rewards.py` (8 + Phase I) green; `verify_contact_sensor.py` green;
  `verify_reward_setup.py` green (with `impact_progress` exemption); CPU smoke run completes without crash
  and logs non-degenerate per-term episode reward.
