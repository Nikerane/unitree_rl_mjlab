# Docs index — current truth map (2026-07-14)

**Authority rule: code > this index > the living docs it lists.** `docs/archive/**` and dated
records are historical evidence — never act on them without checking here first. When a doc and
the code disagree, the code on branch `soft-cat` wins; fix the doc.

## 1. Current truth map (the living set)

| Doc | What it is |
|---|---|
| `CLAUDE.md` (repo root) | Agent context: thesis direction, live reward weights, pre-train gate, environment |
| `docs/thesis/README.md` | Curated thesis digest: confirmed contributions, decisions, defense points, citations |
| `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` | The impulse-CaT arm (the thesis headline): quantity Λ_j, substep accumulator, staged plan C0–C5, current status |
| `docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md` | Faithful soft `γ(1−δ)` CaT: design, Decisions 1–7, file map, phases + the CaT conceptual deep-dive appendix |
| `docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md` | Why soft-CaT over Lagrangian/CMDP (constraint-TYPE argument) + the joint-velocity-bound research appendix |
| `docs/research/reward-design/LITERATURE.md` | Merged annotated bibliography (reward design, hammering RL, impact/tracking RL) |
| `docs/research/reward-design/OPEN_QUESTIONS.md` | Open vs resolved experiment questions (Q1–Q12) |
| `docs/research/reward-design/NAIL_PRECISION_CURRICULUM.md` | Idea capture: shrink-the-nail-head precision curriculum (not specced) |
| `docs/research/reward-design/ORIENTATION_ROBUST_SIM2REAL_ARM.md` | Future arm: orientation-robust / Vicon sim-to-real striking (build after fixed-impedance results) |
| `docs/VEGA_TRAINING_PLAN.md` | GPU campaign plan on EuroHPC Vega + dated V1/b_strike result records |
| `docs/results/README.md` | Index of dated training-run records (records by design — they keep their old numbers) |
| `docs/archive/README.md` | Archive index: what each superseded doc was and what replaced it |

Dated research records kept in place (bannered, bodies frozen):
`docs/research/hammering_reward_design_deep_dive_v2.md` (2026-05-29 reward deep-dive),
`docs/research/tracking_impact_impulse_design_research.md` (2026-06-10 impulse design research, D1–D6),
`thesis_synthesis.md`, `thesis_direction_update.md`, `thesis_handoff_brief_original.md` (repo root).

## 2. Reading order (onboarding)

1. `CLAUDE.md` — direction + ground rules (10 min)
2. `docs/thesis/README.md` — what the thesis claims and why
3. `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` — the impulse constraint (headline contribution)
4. `docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md` — the soft-CaT machinery it runs on
5. `docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md` — why this mechanism and not CMDP/Lagrangian
6. `docs/research/reward-design/LITERATURE.md` — the cited field
7. `docs/research/reward-design/OPEN_QUESTIONS.md` — what is still open and what resolves it

## 3. How we got here (journey; each line links its evidence)

- 2026-05-22 — baseline reward built + audited; "trust the code, not the spec" established (`docs/archive/OPUS_AUDIT.md`, `docs/archive/BASELINE_AUDIT.md`, `docs/archive/RECOMMENDED_REWARD_SPEC.md`)
- 2026-05-29 — v2 reward deep-dive (external literature pass) → `impact_progress` + depth-delta rebalance (specced + shipped 2026-06-02) (`docs/research/hammering_reward_design_deep_dive_v2.md`, `docs/archive/IMPACT_PROGRESS_IMPL_SPEC.md`)
- 2026-06-02 — peer review + first impact/tracking lit review (`docs/archive/PEER_REVIEW_v2.md`, `docs/archive/impact_tracking_rl_litreview.md`)
- 2026-06-10 — track-reference + maximize-impact + bound-impulse architecture designed; generate-then-track walked back (`docs/research/tracking_impact_impulse_design_research.md`, `docs/archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`, `docs/archive/IMPACT_TRACKING_REWARD_SPEC.md`, `docs/archive/HANDOVER.md`)
- 2026-06-15/17 — real claw-hammer integrated + sim audit fixes; hammer-site correction + recalibration landed 06-17 (`docs/archive/REAL_HAMMER_PLAN.md`)
- 2026-06-17 — V1 + b_strike GPU campaigns: trained single strike works, press does not emerge (Q1/Q2 resolved); b_strike surfaces worst-case |q̇| over the joint-velocity limit (V1: `docs/VEGA_TRAINING_PLAN.md` → "V1 — Results"; b_strike: `docs/results/2026-06-17_b_strike.md`)
- 2026-06-17 — velocity-bound ablation A1–A4: worst-case overshoot is chain-coupled, no single bound fixes it → adopt CaT (`docs/results/2026-06-17_velocity_bound_ablation.md`, `docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md` appendix)
- 2026-06-17 — direction update (user decision): **Z1 is the primary platform**; full thesis machinery built here first (`CLAUDE.md` DIRECTION UPDATE block)
- 2026-06-17/18 — faithful soft `γ(1−δ)` CaT designed + shipped (CatPPO/CatSoftHook); GPU result: complies + preserves strike, worst-case tail remains (`docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md`, `docs/results/2026-06-18_softcat_velocity.md`)
- 2026-06-18 — impulse-CaT C0 shipped LOG-ONLY: substep Λ_j accumulator, object-side delivered ∫F·dt, quantity gate (`docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md`, `docs/research/reward-design/derive_impulse_thresholds.py`)
- 2026-06-22 — L6 fixture EE change (gripper → 3D-printed bracket in the sibling repo) → all gate numbers EE-dependent (re-derive per `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` C0 findings); the NEAR_NAIL re-solve blocker is tracked in `docs/research/reward-design/OPEN_QUESTIONS.md` ("Remaining blockers")
- 2026-07-05 — two deep code-review rounds hardened the impulse arm (per-event pulse Λ, episode-cumulative capped delivered, first-violation seeding, log-only invariant); docs consolidated into this index (`docs/superpowers/plans/2026-07-05-docs-consolidation.md`)
- 2026-07-06 — L6 NEAR_NAIL reset re-solved (windup pose; vertical-at-floor infeasible with the fixture grasp); fixture-era `IMP_J_LIMIT`/`i_ref` re-derived; Khadiv verbal go-ahead for soft-CaT as the impulse mechanism
- 2026-07-10 — Track-2 efc-row ground truth shipped (joint3 sign-cancellation finding); C2 enforcement gate PASS, `imp_max_p=0.5` chosen (`docs/results/2026-07-10_c2_enforcement_record.md`); feasibility gate re-greened via `playback_reference.py` (`test_single_strike.py` probe retired)
- 2026-07-12/13 — **vacuity finding**: the impulse constraint is vacuous for reachable ballistic impacts on fixed impedance (velocity effort-clamped); the *windowed press-through* reaction is what binds, conditional on the window/cap pairing → the Λ-quantity is Khadiv decision (e) (`docs/results/2026-07-12_impulse_vacuity.md`, `2026-07-12_state_of_everything.md`, `2026-07-12_khadiv_vic_addendum.md`). Λ re-semanticized as a time-based **sliding window** after an adversarial review falsified the prefix cap (masking bypass). Reference prior fixed (follow-through strike, `i_ref` 0.0811→0.6094)

## 4. Code entry map (`src/tasks/hammer/` — suggested read order)

| # | File | Role |
|---|---|---|
| 1 | `src/tasks/hammer/config/z1/env_cfgs.py` | Arm wiring: flags (`cat_soft`, `cat_impulse`, …) → env config; where the impulse metrics/hook are attached |
| 2 | `src/tasks/hammer/hammer_env_cfg.py` | Base task: scene, DiffIK action space, the 7-term reward, terminations, live weights |
| 3 | `src/tasks/hammer/nail_block.py` | Nail asset loader + goal-depth / success-threshold constants (the physics XML incl. frictionloss lives in the sibling-repo scene it loads) |
| 4 | `src/tasks/hammer/mdp/rewards.py` | Reward terms incl. `ImpactProgressTerm`, `DeliveredImpulseTerm`, imitation prior |
| 5 | `src/tasks/hammer/mdp/velocity_bound.py` | Substep peak-\|q̇\| metric + naive CaT ablation reference (A3) |
| 6 | `src/tasks/hammer/mdp/impulse_bound.py` | **The thesis quantity**: `SubstepImpulseAccumulator` (sliding-window Λ_j, 50 ms contact-masked) + `SubstepDeliveredImpulse` (episode-cumulative capped ∫F·dt, debounced re-arm) |
| 7 | `src/tasks/hammer/cat/constraints.py` | Raw margins: `joint_velocity_excess`, `joint_impulse_excess(env, limit)` |
| 8 | `src/tasks/hammer/cat/constraint_manager.py` | CaT δ-math: EMA-normalized margin → per-constraint termination probability between `min_p` and `max_p`, max-combine (exact formula in the docstring) |
| 9 | `src/tasks/hammer/cat/hook.py` | `CatSoftHook`: velocity ∪ impulse soft-OR, first-violation seeding, log-only short-circuit, guards |
| 10 | `src/tasks/hammer/rl/cat_storage.py` | Float-dones rollout storage (δ round-trips un-truncated) |
| 11 | `src/tasks/hammer/rl/cat_ppo.py` | `CatPPO`: scale-positives reward discount + dual-mask GAE (`γ(1−δ)` bootstrap) |
| 12 | `src/tasks/hammer/rl/runner.py` | Runner + ONNX export |
| — | `src/tasks/hammer/mdp/observations.py`, `src/tasks/hammer/mdp/references.py`, `src/tasks/hammer/mdp/terminations.py` | Obs terms; open-loop strike reference + phase machinery; termination terms |
| — | `src/tasks/hammer/config/z1/rl_cfg.py`, `src/tasks/hammer/config/z1/__init__.py` | PPO/CatPPO config; task registration (all `Unitree-Z1-Hammer-*` arms) |

**Gates (run before any GPU training** — see `CLAUDE.md` for the exact contract):
`docs/research/reward-design/validate_rewards.py` (phases A–M) ·
`docs/research/reward-design/verify_contact_sensor.py` ·
`docs/research/reward-design/verify_reward_setup.py` ·
`docs/research/reward-design/derive_impulse_thresholds.py` (impulse quantity gate; re-run after any EE change) ·
`docs/research/reward-design/playback_reference.py` (strike feasibility gate; the crude `test_single_strike.py` probe was retired 2026-07-10 — the script remains but is not a gate).

**Entry scripts:** `scripts/train.py` · `scripts/play.py` · `scripts/lightning_pair.sh` (**the GPU route** — Lightning.ai prior-vs-none pair: train both arms + auto-eval; Vega is retired) ·
`scripts/diag_policy_trace.py` (strike-vs-press classifier + peak-|q̇| eval) ·
`scripts/eval_impulse.py` + `scripts/eval_impulse.sh` (**standalone C3 checkpoint eval**: per-joint Λ max/p95, worst Λ/cap ratios, delivered impulse, success — one summary.csv row per checkpoint, same-env cross-arm protocol) ·
`scripts/diag_impulse_trace.py` (per-substep impulse trace) · `scripts/compare_runs.py` ·
`scripts/render_reference.py` (headless strike render) · `scripts/verify_cat_soft.py` + `scripts/smoke_cat_soft.py` (CaT plumbing).
**Tests:** `tests/` (unit suite incl. `tests/test_docs_current.py`, the freshness guard for this very index).
