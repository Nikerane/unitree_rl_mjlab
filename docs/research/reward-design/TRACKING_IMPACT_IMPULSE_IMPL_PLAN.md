# Tracking + Impact + Impulse — Staged Implementation Plan (Z1 Phase-0 → G1)

**Date:** 2026-06-10 · **Branch:** `hammer-z1` · **Status:** plan approved-for-staging, **pre-implementation**. **Scope: Z1 only** — G1 deferred entirely (user decision, 2026-06-10; see stub at end).
**Design source of truth:** `docs/research/tracking_impact_impulse_design_research.md` (decisions D1–D6, open questions Q13–Q18).
**Goal (user directive):** the RL policy follows a given strike trajectory while maximizing delivered impact and bounding per-joint impulse. "Max force" is operationalized as **delivered axial impulse/momentum** — never commanded or instantaneous simulated force.

> **Decision changelog (2026-06-10, supersedes report D1 for the Z1):** primary tracking mechanism = **weak annealed reward-level prior (A-PRIOR)**, per the user's expectation that the policy should be free to deviate qualitatively from the reference, and because the reward-prior machinery is what the eventual thesis policy uses. The **residual action space (A-RES) is demoted to an optional ablation arm**. All anti-degenerate-tracking safeguards from the research report apply to A-PRIOR and are specified in Stage T2.

---

## Stage T0 — Prerequisites (gate: nothing below starts before this)

1. **Path A physics fixes** (`docs/FUTURE_UPDATES.md` items 1, 2, 5, 6): `clip_actions=1.0` in `src/tasks/hammer/config/z1/rl_cfg.py`; reset randomization in `hammer_env_cfg.py`; nail `frictionloss` 10→30 N + `damping` 1→0.5 **in BOTH `nail_block_scene.xml` AND `hammer_nail_scene.xml`** (the audit found them desynced — viz still has 0.3 N/8); nail geometry fix (6a quick or 6b proper). Also fix the stale 0.3 N references in `OPEN_QUESTIONS.md` Q1 and `validate_rewards.py:80`.
2. **Run `test_single_strike.py` (Q1)** post-fix. Output decides the reference shape: ≥0.07 m → single-strike reference; <0.03 m → cyclic multi-strike reference (`n_strikes > 1`). Record the result in `OPEN_QUESTIONS.md`.
3. **Commit the research corpus** (Path C) so this plan and its design report survive context loss.
4. Re-run the full existing gate: `validate_rewards.py` (9 phases), `verify_contact_sensor.py`, `verify_reward_setup.py`, `pytest tests/`.

**Acceptance:** all gates green on the recalibrated physics; Q1 answered; baseline A-BASE retrained once on fixed physics (this is Path A's own deliverable — press dissolves or persists, write it up either way).

---

## Stage T1 — Reference + phase infrastructure

**New:** `src/tasks/hammer/mdp/references.py`
- `StrikeReference`: piecewise reference for the hammer-head target — wind-up (time-indexed; no contact risk) → descent (**phase-indexed by hammer–nail distance**, monotone; D2/D5: never clock-time near contact) → optional retract segment (cyclic mode, per Q1). Built from the existing IK (`solve_ik.py` in the assets repo) as `p*(φ)` waypoints; `q*(φ)` optional later.
- Parameterized by `n_strikes`, approach height (reuse `test_single_strike.py` heights), strike depth target.
- v0 is hardcoded; the module's interface is the contract a Vu-style QP / Ti-style iLQR generator fills later (thesis upgrade, D2).

**Changes:**
- `src/tasks/hammer/mdp/observations.py`: add `strike_phase` (φ ∈ [0,1]) and `ref_next_waypoint` (next `p*` in robot frame) observation terms — without these the policy cannot anticipate the reference (DA flag #2 mitigation).
- `hammer_env_cfg.py` events: add optional **RSI** reset event — initialize the arm at a random phase of the reference (PGDM precedent; off by default until T2).

**Tests:** `tests/test_strike_reference.py` (stub, no MuJoCo): φ monotone on descent; waypoints continuous; cyclic mode loops; reachability asserted against joint limits via the IK check.
**Validation:** new `validate_rewards.py` **Phase J** — step the env scripted, assert φ advances monotonically during descent and resets cleanly.

---

## Stage T2 — Tracking enters: weak annealed reward prior (D1, revised — PRIMARY ARM)

**New reward term** in `src/tasks/hammer/mdp/rewards.py`:

```
r_imit(t) = anneal(iter) · exp( −‖p_head − p*(φ_t)‖² / σ² ) · 1[pre-contact phase]
```

- **Task-space, position-only.** Track the hammer-head site against the reference waypoint `p*(φ)` — no joint-space term (no `q*` needed in v0), and **no velocity imitation anywhere** (Biemond/TAC: ill-posed through contact; descent is distance-indexed so timing variation doesn't corrupt the error).
- `σ = 0.05 m` initial (wide enough to permit deviation, narrow enough to shape the swing); tune via the budget rule below, not by feel.
- **Ante-impact only (reference-spreading lesson):** the gate `1[pre-contact phase]` zeroes the term from `first_contact` onward — permanently for the single-strike reference; until the retract segment re-arms it in cyclic mode (Q1-dependent). Implemented as a stateful latch in the term (reset per episode), reusing the `hammer_nail_contact` sensor.
- **Anneal:** linear, `anneal(iter) = max(0, 1 − iter/T_anneal)`, `T_anneal ≈ 2500` of 5000 iterations (Freitag: schedule shape immaterial; do NOT change weights mid-run by hand — one schedule, fresh run per arm). Implementation: prefer a curriculum term that decays the manager weight (mjlab curriculum manager, pattern per `mjlab/envs/mdp/curriculums/`) so `validate_rewards.py`'s dynamic weight read stays truthful; fallback: internal factor inside the term with the factor logged. Discovery task, timeboxed.
- **Weight budget rule (the corpus's dense/sparse-balance lesson, change #1):** choose `w_I0` so the *maximum cumulative* imitation reward per episode stays ≲ 0.3 × completion: with ~1000 steps × mean `r_imit` ≈ 0.5 → `w_I0 ≈ 0.05–0.1`. **Start `w_I0 = 0.1`.** Never approach DeepMimic-class weights (no evidence supports them for exceed-the-reference tasks).
- **Wiring:** `hammer_env_cfg.py` rewards dict + per-robot site in `config/z1/env_cfgs.py` (mirrors `approach`).

**Observations** (`src/tasks/hammer/mdp/observations.py`): add `strike_phase` (φ) and `ref_error` (`p*(φ) − p_head`, robot frame) — the policy must *see* the reference for the prior to anchor rather than merely perturb.

**Degenerate-tracking guard (mandatory metrics, this stage onward):**
- *press-watchdog*: fraction of successes with exactly one contact event;
- *deviation norm*: episode-mean `‖p_head − p*(φ)‖` (tells whether the prior anchors or cages — the Q13 evidence);
- *beat-the-reference*: delivered windowed impulse vs. the scripted-playback baseline `I_ref` (handoff-brief pillar F diagnostic: "match everywhere → copying; beat → healthy").

**Tests:** `tests/test_imitation_reward.py` (stub env, pattern of `test_impact_progress_reward.py`): =1 at the reference point; Gaussian decay with distance; **zeroed after first_contact latch**; latch resets per episode; anneal factor decreases; per-env shape.
**Validation:** extend **Phase J** — state-inject head at/off the reference → expected values; force `first_contact` → term zeroed; **budget assertion**: `w_I0·Σ r_imit` over a scripted full episode < 0.35 × completion. **Phase M** (kept, reworked): a *scripted playback* of the reference (no policy) must strike the nail and log its `I_ref` — this gates reference quality for both arms.

**Optional ablation arm (A-RES, demoted):** residual action `Δp_total = Δp_ref(φ) + α·a_policy` injected at the DiffIK-target level (Ranjbar), `α = 0.5 × delta_pos_scale`. Build only if A-PRIOR shows the cage/press failure modes or if budget allows the Q13 comparison.

---

## Stage T3 — Impact term: windowed axial impulse (D3)

**New reward term** in `src/tasks/hammer/mdp/rewards.py`:
- `WindowedImpactImpulseTerm(ManagerTermBase)`: on `first_contact`, open a window of `W` control steps; accumulate axial impulse `I = Σ F_axial·h` from the contact sensor's force history (`ContactSensorCfg(history_length=10)` covers the 10 substeps of each control step — accumulate across the W steps in term state); at window close pay `(I / I_ref) · 1[Δdepth_window > ε]` — **one payout per contact event**.
- Keeps the double-gate philosophy (validated by RoboStriker/Ma/HITTER patterns). A sustained press = one contact event = one bounded payout.
- `W`: **derived, not hardcoded** (DA flag #1): new script `docs/research/reward-design/log_strike_window.py` logs force-pulse durations from scripted strikes post-Path-A; pick W ≈ p95 pulse duration. `I_ref`: median logged single-strike impulse.
- Config: replaces the `impact_progress` slot at weight 8.0 in the main arm; `impact_progress` retained verbatim as the **I-MOM ablation arm** (artifact-immune momentum proxy; the two should co-move — Q14).

**Tests:** `tests/test_windowed_impact_reward.py` (stub env feeding synthetic force histories): single payout per event; window closes; depth gate; reset clears state; press-pattern (long single contact) earns exactly one bounded payout.
**Validation:** **Phase K** — state-injected force history → expected payout; real-strike end-to-end (mirrors Phase I).
`verify_reward_setup.py`: add to `LIVENESS_EXEMPT`.

---

## Stage T4 — Per-joint impulse: measure first, then penalize excess (D4 scaffold)

**New:** `src/tasks/hammer/mdp/rewards.py` (or `costs.py`):
- `JointImpulseTerm`: per-joint windowed impulse `Λ_j = Σ|qfrc_constraint_j|·h` over the same contact-anchored window (substep access via the sensor-history pattern; if `qfrc_constraint` is not exposed per-substep in mjlab, fall back to windowed `Δq̇_j` per Aouaj and log the limitation). **Weight 0.0 initially — log-only** for ≥1 training run.
- `JointImpulseExcessTerm`: `Σ_j max(0, Λ_j − Λ_safe,j)²` — silent until limits approached; this is the penalty that does **not** fight `r_impact` (research §3, SQ5). Never ship raw ‖Δq̇‖² at meaningful weight.
- **Thresholds:** new script `docs/research/reward-design/derive_impulse_thresholds.py` — ratings-ladder arithmetic (Repeated-Peak ≈ 2× rated torque per joint × effective impact duration), Z1 joint specs as constants with sources; 10⁴-event fatigue budget printed as a per-run strike budget. Until hardware specs per joint are confirmed, constants marked TBD and the term stays log-only.

**Tests:** `tests/test_joint_impulse_cost.py` — injected velocity-jump/force history → expected Λ_j; excess form zero below threshold; reset.
**Validation:** **Phase L** — scripted strike produces nonzero logged Λ_j; excess term zero on a gentle strike, positive on an injected violent one.

---

## Stage T5 — Constraint escalation (D4 target form)

1. **Fixed-λ sweep** on `JointImpulseExcessTerm` (λ ∈ {0.5, 2, 8, 32}) — map the impact-vs-impulse Pareto (Spoor recipe) using T4's logged metrics. → Q15.
2. **PID-Lagrangian** (episodic-sum cost `Σ Λ_j`, per-joint or summed): port from OmniSafe defaults into the rsl-rl runner (`HammerOnPolicyRunner` extension). Training cost limit **stricter** than the deployment target. Only after (1) shows the penalty trades poorly.
3. Tail handling if needed: CVaR-on-cost or Saute-style budget-in-observation (env wrapper — cheap).
4. **G1/hardware only:** training-time CBF-style impulse filter (precedent deploys filter-free on a G1) or runtime QP shield (Wang IJRR 2023, corpus).

**Acceptance:** constraint satisfied within tolerance across ≥3 seeds while success rate ≥ the C-PEN best arm; multiplier non-oscillating (else revisit Spoor recipes).

---

## Ablation table (the research result; mjlab logs per-term episode sums natively)

| Arm | Tracking | Impact term | Impulse handling |
|---|---|---|---|
| A-BASE | none (current 7-term, post-Path-A) | `impact_progress` | none |
| **A-PRIOR** (primary) | weak annealed reward prior (T2) | I-WIN (T3) | log-only |
| A-RES (optional) | residual action over reference | I-WIN | log-only |
| C-PEN | A-PRIOR | I-WIN | excess penalty, λ sweep |
| C-LAG | A-PRIOR | I-WIN | PID-Lagrangian |
| I-MOM | A-PRIOR | `impact_progress` | best constraint arm |

Primary readouts: A-PRIOR vs A-BASE answers "does the reference help at all?"; the deviation-norm and beat-the-reference metrics answer "anchor or cage?" (Q13); A-RES runs only if A-PRIOR shows cage/press failure or budget allows.

**Metrics (all arms):** success rate · strikes/episode · delivered `I_axial` (windowed) · peak & episodic `Λ_j` · press-watchdog fraction · deviation-from-reference norm · action std. ≥3 seeds, mean±CI (peer-review R1-W5 requirement).

## G1 (DEFERRED — out of scope for this plan, user decision 2026-06-10)

Not planned here. When the thesis phase starts, the research report §4.1 D6 column + §6 Q17/Q18 hold the G1 design (variable-`kp` action space, flux objective, Pinocchio cross-validation, passivity filter). What this Z1 plan deliberately builds G1-compatible: the `r_imit` prior machinery, the windowed-impulse term, the threshold-derivation script, the constraint stack, and all validation-phase patterns. Re-confirm D1/D4 with Prof. Khadiv before any G1 work (CLAUDE.md guardrail).

## Do-not-start-GPU-training-until checklist

- [ ] T0 complete (physics fixed in BOTH XMLs, Q1 answered, corpus committed)
- [ ] New unit tests green (`pytest -m "not integration"`)
- [ ] `validate_rewards.py` phases A–I **+ J/K/L/M** green (J includes the imitation-budget assertion: cumulative `w_I·Σr_imit` < 0.35 × completion)
- [ ] `verify_contact_sensor.py`, `verify_reward_setup.py` (new exemptions) green
- [ ] Phase M scripted playback strikes the nail and logs `I_ref` (reference quality gate)
- [ ] `W`, `I_ref`, `Λ_safe,j` derived from logged data / spec sheets, not invented
- [ ] CPU smoke run (50 it, 16 envs) logs all new terms non-degenerate, press-watchdog + deviation-norm metrics flowing
