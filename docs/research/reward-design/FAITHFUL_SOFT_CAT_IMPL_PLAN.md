# Faithful soft `γ(1−δ)` CaT — design, decisions & implementation plan

**Status:** pre-implementation (design approved, decisions recorded). 2026-06-17.
**Scope:** Z1 hammer task, fixed impedance. Replace the *naive sampled-hard* `CaTJointVelConstraint`
(`src/tasks/hammer/mdp/velocity_bound.py:109-160` — a `TerminationTermCfg` that does
`torch.rand_like(δ) < δ` and returns a hard boolean done) with the **faithful soft** mechanism
of Chane-Sane et al., *Constraints as Terminations for Legged Locomotion RL*, IROS 2024
(arXiv 2403.18765; reference code github.com/gepetto/constraints-as-terminations).

**Why faithful matters (thesis):** the supervisor wants the CaT paper *applied*, not approximated.
The naive version samples δ into a hard termination every step — that is closer to a stochastic
hard cap than to CaT. The paper's mechanism keeps δ as a *continuous* soft-termination that
discounts the value target; the policy learns to avoid violating trajectories because they yield
systematically lower returns, without the sim ever being reset on a soft violation.

> This document is the authoritative design + decision record for the port. The companion
> conceptual dive is `CAT_DEEP_DIVE.md` (why CaT is an incentive, not a brake; CaT-vs-VIC).

---

## 1. The CaT mechanism, as actually implemented in the reference

CaT splits across **env-side** (reward discount + soft-done emission) and **learner-side**
(GAE bootstrap consumes the float done). The δ math lives in a standalone `CaT`/`ConstraintManager`
helper. All line numbers below are from the cloned reference (`/tmp/cat_ref/.../tasks/utils/cat/`)
and were read verbatim.

### 1.1 The δ math — `constraint_manager.py`

For each constraint term, the per-env, per-column termination probability is

```
probs[mask] = min_p + clamp(c / c_max, 0, 1) · (max_p − min_p)      # constraint_manager.py:72
mask = (c > 0)                                                       # constraint_manager.py:67  (violations only; else 0)
```

- `c` is the **raw signed margin** returned by the constraint function (e.g. `|q̇| − limit`),
  *not* clamped and *not* a probability. The manager does the clamping.
- `c_max` is a **Polyak EMA of the batch-max margin** over envs:
  `c_max ← τ·c_max + (1−τ)·max_envs(c)`, `τ=0.95`, floored at `1e-6`, **seeded to the first
  batch max** (`constraint_manager.py:55-61`). This is what makes CaT **scale-free** — δ depends
  on the violation *relative to the current population*, not on hand-tuned units.
- The EMA **persists across episode resets** — `reset()` clears only `probs`/`raw_constraints`,
  never `running_maxes` (`constraint_manager.py:34-37`).
- Per-env aggregate δ = **max over all terms and all columns** — a soft logical-OR
  (`get_probs()`, `constraint_manager.py:82`).
- `max_p` is the per-term probability ceiling. `max_p = 1.0` ⇒ a true hard constraint;
  `max_p < 1` ⇒ soft. This is the soft→hard sweep knob.

### 1.2 Env-side application — `cat_env.py:99-110, 118-121, 147`

```python
cstr_prob = self.constraint_manager.compute()                              # δ ∈ [0, max_p], shape (N,)
self.reward_buf = torch.clip(
    self.reward_manager.compute(dt=self.step_dt) * (1.0 - cstr_prob),      # reward · (1 − δ)
    min=0.0, max=None,                                                     # ← the clip (see Decision 1)
)
dones = cstr_prob.clone()                                                  # the SOFT done IS the float δ
...
dones[reset_env_ids] = 1.0          # real terminations/timeouts overwrite δ with a hard 1.0
...
return self.obs_buf, self.reward_buf, dones, self.reset_time_outs, self.extras
```

- The `(1−δ)` reward discount happens **env-side**, before the reward leaves the env.
- A soft δ∈(0,1) **never resets the sim** — only the real `reset_buf` ids (terminations/timeouts)
  are reset (`cat_env.py:118-128`). The "termination" is realized **purely in the value target**.
- `dones` is returned as a **float** carrying δ (1.0 for real terminations).

### 1.3 Learner-side bootstrap — CleanRL fork `ppo.py:255-277`

The float done enters GAE as a continuation mask. The reference uses a **dual mask**:

```
nextnonterminal      = 1 − dones[t]        # the soft δ
true_nextnonterminal = 1 − true_dones[t]   # real terminations only
delta = reward + γ · V(s_{t+1}) · nextnonterminal · true_nextnonterminal − V(s_t)
adv   = delta + γ · λ · nextnonterminal · true_nextnonterminal · adv
```

So δ at step *t* discounts **both** the (already `(1−δ)`-scaled) immediate reward **and** the
bootstrapped future value `V(s_{t+1})`. A violating step is worth less *now* and cuts off the
*future* (including the `completion = +100` terminal bonus). The future-value cut is the real
teeth. The rl_games and skrl integrations achieve the identical effect by storing `dones` as
float32 and reusing the stock `1 − dones` GAE (`cat_experience.py:20-33`, `skrl/ppo.py:211,425`).

### 1.4 Soft→hard curriculum — `curriculums.py:21-42`

Ramps `max_p` (not `min_p`) by interpolating an expected lifetime `T: 20 → 1/init_max_p` linearly,
then setting `max_p = 1/T` (convex in probability), clocked by `common_step_counter / num_steps`.
`init_max_p` is the **final hard ceiling** (the name is misleading). For us this is the
"sweep p_max soft→hard" knob; v1 may pin `max_p` constant and add the ramp later.

---

## 2. How it maps onto our stack (mjlab 1.4.0 + rsl_rl) — the thin-adapter approach

**Principle (user's call, and the reference's own pattern): never edit the installed `rsl_rl` or
`mjlab` packages.** Editing site-packages is not version-controlled, breaks on reinstall, and would
have to be re-applied and kept in sync on the Vega cluster. The CaT authors never forked CleanRL,
rl_games, or skrl — for the two libraries with *encapsulated* rollout buffers (rl_games, skrl —
structurally just like rsl_rl) they wrote **thin subclass-adapters in their own repo** and injected
them at runtime. We mirror that.

| Library | Buffer encapsulated? | Approach | Injection seam |
|---|---|---|---|
| CleanRL | no (script-level) | inline dual-mask GAE in their own train script | n/a |
| rl_games | **yes** | **thin subclass-adapter** (`CaTExperienceBuffer`, `CaTA2CAgent`) — float `dones`, stock GAE reused | rl_games' algo registry (`register_builder`), selected by string |
| skrl | **yes** | fork-by-copy of `PPO`/`Trainer`/`Runner` (subclass the *base*, ~3 surgical edits) | a forked `Runner` string-map |
| **rsl_rl (ours)** | **yes** | **thin subclass-adapter** | **`cfg.algorithm.class_name` resolved by string** (`on_policy_runner.py:39`) — cleanest of the three |

**Our injection seam is the cleanest available.** `OnPolicyRunner` resolves its PPO class from a
config string (`resolve_callable(cfg["algorithm"]["class_name"])`, `on_policy_runner.py:39-40`) and
PPO builds its own `RolloutStorage` inside the `construct_algorithm` static method
(`ppo.py:408-440`). So we point `cfg.algorithm.class_name` at our own `CatPPO` subclass and inject a
float-dones storage from its `construct_algorithm`. The runner needs no change.

**The one structural difference from the reference (record this for the thesis).** In the
rl_games/skrl stacks the env class is selectable, so the reference does the `(1−δ)` discount and
float-δ emission inside a `CaTEnv.step()` override. **mjlab pins the env class** — `train.py:110`
hardcodes `env = ManagerBasedRlEnv(...)` and the task registry has no `env_class` field
(`registry.py:22-40`); mjlab's `step()` returns a *bool* `reset_terminated` as the done and applies
**no** reward scaling (`manager_based_rl_env.py:436-477`). We therefore **cannot** override `step()`.
Instead we apply the `(1−δ)` discount in `CatPPO.process_env_step`, where rsl_rl clones the reward
before storing it (`ppo.py:140`), and we surface δ to the learner via the env's `extras` dict.

> **Thesis framing of this point (precise):** *the CaT reward discount and soft-done bootstrap are
> mathematically identical to the paper; only the host method differs (learner `process_env_step`
> instead of env `step`), because the mjlab framework does not expose an env-class override seam.*
> This is a hosting detail, not a change to the algorithm.

---

## 3. Design decisions (thesis-facing)

### Decision 1 — Scale only the *positive* reward terms by `(1−δ)`; leave penalties unscaled (the "constrained-reward" form). **[the load-bearing decision — revised after adversarial review]**

**Decision.** Apply the CaT discount to the **positive task terms only**, and leave the negative
regularizers unscaled and additive:

```
reward = clip( r_pos · (1 − δ), min=0 ) + r_neg
       = r_pos · (1 − δ) + r_neg               # clip never fires here: r_pos ≥ 0  (exact paper fidelity)
       = r_total − δ · r_pos
```
with `r_pos = approach + nail_driven + nail_depth_delta + impact_progress + completion` (all ≥ 0)
and `r_neg = action_rate + joint_pos_limits` (≤ 0; `hammer_env_cfg.py:219,223`).

**This decision was *changed* by an adversarial review** (3 independent lenses — continuation-
probability theory, reward-hack red-team, faithfulness-to-cited-method — plus adjudication,
2026-06-17). The original draft shipped the *verbatim-clip-removed, full-reward* form
`reward = r_total · (1−δ)` (no clip). That form is **wrong**; the analysis below records why.
Keeping the overturned option visible is deliberate — it is the defensible thesis narrative
(we stress-tested the choice and found a concrete failure mode).

**What survives from the original reasoning — the *verbatim* clip is wrong for us.** CaT's reference
clips `reward·(1−δ)` to ≥ 0 (`cat_env.py:102-106`). Locomotion reward is non-negative, so that clip
**never fires** — it is the paper *implicitly enforcing a non-negative-reward precondition*. Our
reward is mixed-sign, so a verbatim clip floors **every net-negative step to 0, independent of δ**
(even at δ=0, `clip(−8, 0) = 0`), silently erasing the penalty signal across the board. So shipping
the verbatim clip is out; that part of the original diagnosis holds.

**Why NOT full-reward `(1−δ)` with no clip (the overturned option).** Under the dual-mask bootstrap
(§1.3), the per-step *incentive to violate* (moving a step from δ=0 to δ>0) is exactly

```
G(δ) − G(0)  =  −δ · [ r_t + γ·V(s_{t+1}) ]   ≡   −δ · B_t
```

This is **positive — i.e. the policy is *paid* to violate — whenever `B_t = r_t + γ·V(s_{t+1}) < 0`.**
Because `(1−δ)` multiplies *both* the immediate reward and the bootstrapped future value in the
*same* bracket, the immediate-penalty relief and the future-value cut **add; they do not trade off.**
So the original "future-value cut dominates the local relief" claim is **false as stated** — dominance
holds only where `γ·V(s_{t+1}) > |r_t|`, and fails in the regimes that dominate this task:
- **Cold critic / early training:** `V ≈ 0` for hundreds of iterations and `completion=+100` is not
  yet reachable, so no large positive `V(s_{t+1})` exists anywhere — the suppression term is ~0
  exactly when the immediate relief is at full strength.
- **Physical coupling (the damaging one):** high arm velocity (large δ) is *what drives joints into
  their position limits* — a fast swing overshoots (closed-loop peak `|q̇| ≈ 4.3–4.65` vs the `3.14`
  limit, [[z1-velocity-bound-finding]]). So the steps where `joint_pos_limits = −10` fires are *the
  same* steps where δ is largest. Full-reward `(1−δ)` then hands the policy a smooth,
  gradient-followable knob: **swing faster → δ up → halve the −10 position penalty.** No timing skill
  is required; it is the default correlation of the dynamics. This couples the velocity and
  position-limit safety signals so that *violating* the velocity limit becomes a *tool* for evading
  the position penalty — adversarial to the thesis's own impact-safety objective.

**Why scale-positives-only is the fix — and is *more* faithful to CaT, not less.** It restores the
paper's operating regime instead of deleting the line that assumed it. The incentive-to-violate
becomes `−δ · [ r_pos + γ·V(s_{t+1}) ]` with `r_pos ≥ 0`; the bracket can go negative only if
`V(s_{t+1}) < 0` (rare, and *uncoupled* from the −10 penalty), so the evasion knob is gone and the
penalty magnitude is no longer policy-controllable. CaT's `(1−δ)` is a "forfeit future *positive*
return" weight — only coherent on `r ≥ 0`; applying it to penalties (as full-reward does) is the
semantic error. This also implements our own companion analysis `CAT_DEEP_DIVE.md:19` ("keep
penalties as separate constraints or shift reward non-negative"), which the overturned option
silently contradicted.

**Why not "shift the whole reward non-negative" (the deep dive's other option).** Re-tuning the
validated 7-term reward to be non-negative is a large, risky change that conflicts with the repo's
augment-not-replace / trust-the-code discipline. Scale-positives-only achieves the same precondition
restoration without touching the reward design.

**Implementation note (it is not literally "one line").** Scale-positives-only needs `r_pos` (or
`r_neg`) surfaced to the learner. The `cat_hook` computes `r_neg` from the two known negative terms
(`action_rate`, `joint_pos_limits`) and stashes `r_pos = r_total − r_neg` into `extras` alongside δ;
`CatPPO.process_env_step` then applies `reward ← r_total − δ·r_pos`. Verified by test (iv) in C2.

### Decision 2 — Apply the `(1−δ)` discount in the learner, not the env `step()`.

Forced by the framework (mjlab pins the env class, §2). Implemented in `CatPPO.process_env_step`
where rsl_rl clones the reward before storage (`ppo.py:140`). **Mathematically identical** to
env-side scaling; only the host method differs. (See the thesis-framing note in §2.)

### Decision 3 — Explicit dual-mask GAE, not the algebraically-equivalent shortcut.

The single combined float done `dones = 1 − (1−δ)(1−true_done)` fed through stock single-mask GAE
reproduces `γ(1−δ)(1−true)` exactly, so a `compute_returns` override is not *strictly* required.
We nonetheless implement the **explicit dual mask** (separate soft-δ and hard-done buffers) because
it (a) mirrors the reference (`ppo.py:255-277`) line-for-line — cleaner provenance for the thesis,
and (b) avoids entangling δ with rsl_rl's timeout-bootstrap-as-reward path. We unit-test that it
equals the combined-float-done identity.

### Decision 4 — A CaT violation is a *true termination*, not a *timeout*.

It must enter the hard-done / soft-δ channel, **never** `extras["time_outs"]`. rsl_rl bootstraps
timeouts by injecting `γ·V` into the stored reward (`ppo.py:151-155`); routing a constraint
violation through that path would *add back* the future value we are trying to cut, destroying the
signal. We keep rsl_rl's standard timeout bootstrap for genuine episode-length timeouts and apply
the soft δ only on the termination channel. This is the correct CaT semantics (CaT terminations ≠
timeouts) even though it diverges from the CleanRL fork's treatment of timeouts as terminal.

### Decision 5 — Soft constraint is a *non-terminating* term (never resets the sim).

This is the core difference from our naive impl (which is a `TerminationTermCfg` and must
Bernoulli-sample δ→hard to drive a real reset — i.e. *hard* CaT, the thing we are replacing).
The soft constraint is computed by a non-terminating manager hook that stashes δ on the env and into
`extras`, and **does not feed `reset_buf`**. The sim keeps running on a soft violation; genuine
terminations/timeouts still reset through the normal `TerminationManager`.

### Decision 6 — Keep `min_p = 0` (reference default).

A barely-violating step contributes ≈0 probability (`constraint_manager.py:25`). Revisit only if
training shows a dead-zone near the limit.

---

## 4. Architecture — new files in our repo (zero installed-package edits)

| File | Responsibility |
|---|---|
| `src/tasks/hammer/cat/constraint_manager.py` | Port of the reference `CaT` helper: per-term EMA `running_maxes` (τ=0.95, persists across resets), δ = `min_p + clamp(c/c_max,0,1)·(max_p−min_p)`, max-combine over terms. Pure torch, CPU-testable. `compute(env) → δ (B,)`. |
| `src/tasks/hammer/cat/constraints.py` | Pure constraint funcs returning the **raw signed margin** `c = value − limit`. `joint_velocity_excess` reuses `_ARM_CFG` + `SubstepPeakJointVel` (substep-peak signal) from `velocity_bound.py:49-87`. Future substep-impulse constraint plugs in here. |
| `src/tasks/hammer/cat/cat_hook.py` | Non-terminating manager hook (Decision 5): calls the manager, stashes δ on `env` + into `extras["cat_delta"]`, never feeds `reset_buf`. Also computes `r_pos = r_total − r_neg` (Decision 1, from the two negative terms) and stashes `extras["cat_r_pos"]`. Exposes `max_p` for the curriculum. |
| `src/tasks/hammer/cat/curriculum.py` | Port of `modify_constraint_p` (`curriculums.py:21-42`): lifetime ramp `T:20→1/init_max_p`, `max_p=1/T`, clocked by `common_step_counter`. Optional for v1. |
| `src/tasks/hammer/rl/cat_storage.py` | `CatRolloutStorage(RolloutStorage)`: float32 `dones` buffer + a separate `soft_dones` (δ) float buffer and its `add_transition` write. Fixes the byte-truncation (`rollout_storage.py:149,180`). |
| `src/tasks/hammer/rl/cat_ppo.py` | `CatPPO(PPO)`: overrides `construct_algorithm` (inject `CatRolloutStorage`), `process_env_step` (read δ + `r_pos` from `extras`; apply **scale-positives-only** `reward ← r_total − δ·r_pos` per Decision 1; store δ), and `compute_returns` (dual-mask GAE per §1.3). |

**Wiring (existing files, additive):**
- `config/z1/rl_cfg.py`: CaT variant sets `algorithm.class_name = "src.tasks.hammer.rl.cat_ppo:CatPPO"`.
- `config/z1/env_cfgs.py`: new `cat_soft=True` flag wiring the hook (+ optional curriculum), replacing
  the naive `TerminationTermCfg` block (`env_cfgs.py:146-155`) for this arm.
- `config/z1/__init__.py`: `register_mjlab_task("Unitree-Z1-Hammer-CaT-Soft", ...)` — same pattern
  as the existing `-CaT` / `-CaT-Substep` / `-VelHardTerm` arms (`__init__.py:39-75`).

---

## 5. Phased implementation plan (TDD, each phase CPU-testable before any GPU run)

- **C0 — δ math (pure, no env).** Implement `constraint_manager.py` + `constraints.py`. Unit-test the
  δ formula against hand values, max-combine over 2 terms, EMA persistence across `reset()`,
  `c≤0 ⇒ δ=0`. `pytest tests/test_cat_constraint_manager.py`. Reuse `Z1_JOINT_VEL_LIMIT`/`_ARM_CFG`.
- **C1 — float-dones storage.** Implement `cat_storage.py`. Test δ=0.37 round-trips un-truncated vs
  the stock byte buffer truncating to 0. `pytest tests/test_cat_storage.py`.
- **C2 — dual-mask GAE + scale-positives reward discount (PPO subclass, mocked storage).** Implement
  `cat_ppo.py`. Tests: (i) δ=0 reproduces stock rsl_rl GAE exactly; (ii) dual-mask == combined-float-
  done identity (Decision 3 cross-check); (iii) scale-positives-only applied once
  (`reward = r_total − δ·r_pos`, Decision 1); **(iv) incentive-to-violate is non-positive** —
  assert `G(δ) − G(0) = −δ·[r_pos + γ·V_next] ≤ 0` across a grid of
  `r_neg ∈ {0,−8,−10}`, `V_next ∈ {0,+2,+60,−1}`, `δ ∈ {0,0.5,1.0}` (incl. a cold-start `V_next≈0`
  row). Test (iv) is the regression guard: it **fails** for full-reward no-clip and **passes** for
  scale-positives-only. `pytest tests/test_cat_ppo_gae.py` (CPU, no sim).
- **C3 — env hook + extras plumbing (sim, CPU).** Implement `cat_hook.py`; wire `cat_soft`. Test:
  forced over-limit qv → `extras["cat_delta"]` present, float, ∈[0,max_p]; episode length
  **unchanged** by a soft violation (Decision 5, no spurious reset). `scripts/verify_cat_soft.py`.
- **C4 — end-to-end short train (smoke).** Register `-CaT-Soft`; run ~20 iters; assert loop runs,
  δ logged, mean reward reflects the scale-positives discount, no NaN, checkpoints save. Re-run the
  CLAUDE.md pre-train gate (`validate_rewards.py` 9 phases + `verify_contact_sensor.py`) for the new
  arm. **Penalty-evasion trip-wire (Decision 1):** log per-env δ alongside `joint_pos_limits` /
  `action_rate` firing and assert their co-occurrence correlation does **not** rise over training,
  and that inter-strike peak `|q̇|` does not creep up — treat a rising correlation as a hard gate,
  not a footnote.
- **C5 — curriculum (optional, after C4 green).** Implement `curriculum.py`; test the lifetime ramp
  and `max_p` mutation by `common_step_counter`.

**Reuse map:** `velocity_bound.py::SubstepPeakJointVel` + `_ARM_CFG` + `Z1_JOINT_VEL_LIMIT` carry into
C0/C3. The naive `CaTJointVelConstraint` is the **reference behavior to diff against**, not reused.
The `-CaT*` registration pattern is the template for the `-CaT-Soft` arm.

---

## 6. Open / verify-at-implementation items

1. **`is_finite_horizon`.** mjlab populates `extras["time_outs"]` only when **not** finite-horizon;
   `ManagerBasedRlEnvCfg.is_finite_horizon` defaults `False`. Confirm the hammer `env_cfg` keeps it
   `False`, else timeout bootstrapping silently vanishes. (Not yet verified.)
2. **rsl_rl/mjlab line numbers are install-version-specific** — the anchors above
   (`ppo.py:140/151-155/174/408-440`, `rollout_storage.py:149/180`, `on_policy_runner.py:39`,
   `manager_based_rl_env.py:436-477`) were read from the current conda env; re-confirm at impl time.
3. **Logger episode counting** uses `(dones > 0)` (`logger.py:118-125`) — a threshold, not a bool.
   Keep the soft δ **out** of the wrapper's hard `dones` channel (carry it via `extras`) so episode
   stats stay correct. Verify in C3/C4.
