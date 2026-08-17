# Docs index — current truth map (2026-08-17)

**Authority rule: code > this index > the living docs it lists.** `docs/archive/**` and dated
records are historical evidence — never act on them without checking here first. When a doc and
the code disagree, the checked-out code wins; fix the doc.

## 1. Current truth map (the living set)

| Doc | What it is |
|---|---|
| `CLAUDE.md` (repo root) | Agent context: thesis direction, live reward weights, pre-train gate, environment |
| `docs/thesis/README.md` | Curated thesis digest: confirmed contributions, decisions, defense points, citations |
| `docs/superpowers/specs/2026-08-12-z1-direct-reference-fic-design.md` | Approved direct-reference fixed-impedance FIC-0/FIC-TT task and treatment contract |
| `docs/superpowers/plans/2026-08-12-z1-direct-reference-fic.md` | Executable implementation, Vega qualification/training, and result-banking plan for that campaign |
| `docs/superpowers/specs/2026-08-13-z1-native-vic-design.md` | Approved native Z1 VIC-TT prototype contract; implementation and qualification are banked |
| `docs/superpowers/plans/2026-08-13-z1-native-vic.md` | Executed native VIC-TT prototype qualification plan |
| `docs/superpowers/specs/2026-08-13-z1-vic-canary-design.md` | Approved exact seed-2 VIC-TT engineering-canary contract; the one authorized run is complete |
| `docs/superpowers/plans/2026-08-13-z1-vic-canary.md` | Executed telemetry, guarded-launch, and one-canary assessment plan |
| `docs/results/2026-08-14_z1_vic_seed2_canary.md` | Banked one-seed native VIC-TT engineering result, gain evidence, video, checkpoint, and claim boundary |
| `docs/results/2026-08-15_z1_impulse_cat_step1.md` | Step-1 impulse-CaT qualification: provisional-cap binding, separate velocity/impulse attribution, terminal-censoring limits on event-pressure calibration, and diagnostic-canary evidence |
| `docs/superpowers/specs/2026-08-15-z1-impulse-diag90-500-design.md` | User-approved matched 500-iteration extension of the uniform-`0.9` diagnostic pair; engineering learning-pressure test, not a hardware or scientific threshold |
| `docs/superpowers/plans/2026-08-15-z1-impulse-diag90-500.md` | Exact guarded launcher, test-first verification, one-shot Vega submission, and matched-evaluation plan for the 500-iteration diagnostic pair |
| `docs/superpowers/specs/2026-08-17-z1-impulse-diag90-post-training-evaluation-design.md` | Approved no-learning matched evaluation contract for the completed diagnostic pair |
| `docs/superpowers/plans/2026-08-17-z1-impulse-diag90-post-training-evaluation.md` | Executed evaluator, paired-measurement, Vega-launch, and result-banking plan |
| `docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md` | Banked one-training-seed result: lower *native observed* J3 impulse exposure and retained task utility, but higher true velocity pressure makes the verdict redistribution/trade-off, not clean enforcement; unequal horizons leave causal/full-contact interpretation unresolved |
| `docs/superpowers/plans/2026-08-17-z1-impulse-diag90-release-window-flush-shadow.md` | Separate test-first follow-up plan for complete contact/window-flush dose; planned only, not launch-authorized |
| `docs/thesis/decisions/2026-08-13_vic_impulse_cat_and_trajectory_direction.md` | Supervisor direction: compare velocity+active-impulse CaT against the log-only control, then consider trajectory-source families |
| `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` | The impulse-CaT arm (the thesis headline): quantity Λ_j, substep accumulator, staged plan C0–C5, current status |
| `docs/research/reward-design/IMPULSE_CAP_PROVENANCE.md` | Current threshold provenance: why the historical `2x` conversion is retired; banked, provisional no-`2x`, and uniform-`0.9` diagnostic identities |
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
- 2026-07-10 — L6 asset correction pins the **complete rigid hammer body at 0.2 kg**
  and the separate printed fixture at **0.045 kg**; the older heavier gripper-era
  mass is historical only. Track-2 efc-row ground truth shipped (joint3
  sign-cancellation finding); C2 enforcement gate PASS, `imp_max_p=0.5` chosen
  (`docs/results/2026-07-10_c2_enforcement_record.md`); feasibility gate re-greened
  via `playback_reference.py` (`test_single_strike.py` probe retired)
- 2026-07-12/13 — **vacuity finding**: the impulse constraint is vacuous for reachable ballistic impacts on fixed impedance (velocity effort-clamped); the *windowed press-through* reaction is what binds, conditional on the window/cap pairing → the Λ-quantity is Khadiv decision (e) (`docs/results/2026-07-12_impulse_vacuity.md`, `2026-07-12_state_of_everything.md`, `2026-07-12_khadiv_vic_addendum.md`). Λ re-semanticized as a time-based **sliding window** after an adversarial review falsified the prefix cap (masking bypass). Reference prior fixed (follow-through strike, `i_ref` 0.0811→0.6094)
- 2026-08-12 — direct-reference fixed-impedance FIC-0/FIC-TT campaign completed at three matched seeds; FIC-TT reduced target RMSE in all three but did not establish safety (`docs/results/2026-08-12_z1_fic_direct_reference.md`)
- 2026-08-13/14 — native 12-output VIC-TT qualified and its one authorized seed-2 engineering canary completed; 64/64 deterministic evaluation worlds succeeded with joint-specific gains and impulse CaT log-only (`docs/results/2026-08-14_z1_vic_seed2_canary.md`)
- 2026-08-13 — supervisor direction records active per-joint impulse soft-CaT plus velocity soft-CaT as the target treatment, paired against the impulse-log-only control; trajectory-source comparisons come later (`docs/thesis/decisions/2026-08-13_vic_impulse_cat_and_trajectory_direction.md`)
- 2026-08-14 — threshold provenance corrected: Unitree publishes actuator maximum/effort values, not a Z1 reaction-impulse damage limit; the unsupported historical `2x` conversion is retired. Existing task identities retain their banked vector, while the no-`2x` and uniform-`0.9` vectors are opt-in Step-1 provisional/diagnostic thresholds only (`docs/research/reward-design/IMPULSE_CAP_PROVENANCE.md`)
- 2026-08-15 — Step 1 qualified the active impulse path without changing the registered treatment: the no-`2x` project boundary was nonbinding in 64 deterministic worlds but bound 593/98,304 training-like controller reads, almost entirely at J3; offline pressure was independently active and graded per read. Complete-event dose remains unresolved because the associated events were overwhelmingly terminal-censored. The authorized uniform-`0.9`, `p=0`/`p=0.5` training pair completed as diagnostic plumbing only (`docs/results/2026-08-15_z1_impulse_cat_step1.md`)
- 2026-08-17 — the completed 500-iteration diagnostic pair was evaluated on one fixed and three matched stochastic populations. The evaluation observed lower native J3 diagnostic violation risk and p95/p99 utilization with 100% task success, but true 500 Hz velocity-limit violation risk increased in every stochastic replica; verdict: redistribution/trade-off, not clean enforcement. Unequal native horizons and terminal censoring leave causal/full-contact interpretation unresolved and block complete-event dose (`docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md`)

## 4. Code entry map (`src/tasks/hammer/` — suggested read order)

| # | File | Role |
|---|---|---|
| 1 | `src/tasks/hammer/config/z1/env_cfgs.py` | Arm wiring: flags (`cat_soft`, `cat_impulse`, …) → env config; where the impulse metrics/hook are attached |
| — | `src/tasks/hammer/mdp/variable_impedance.py` | VIC-TT native gain map and ordered six-joint stiffness action; writes expanded MuJoCo gain/bias fields without changing force limits |
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

**Current GPU route:** Clean Vega A100 VIC-TT qualification is banked and complete, and the exact seed-2 engineering canary is also banked and complete from an exact pushed SHA in a clean detached worktree. Qualification ran the same nominal arm twice to freeze per-tensor parity tolerances, then VIC authority, `p=0` FIC–VIC parity, and a genuine one-iteration CatPPO smoke. Training job `41119011` completed 500 iterations and its frozen 64-world evaluation passed 64/64 success and productive-strike gates. This is a one-seed engineering result, not a VIC-superiority claim; impulse CaT was log-only. The direct-reference FIC launchers are banked baseline/reproducibility routes. The matched diagnostic-0.9 pair and its no-learning evaluation are now complete: lower native observed J3 impulse exposure and retained task behavior were observed, but true 500 Hz velocity-limit violation risk increased in every stochastic replica, so the preregistered verdict is redistribution/trade-off, not clean enforcement; unequal native episode/contact horizons prevent a causal or full-contact interpretation. Event-dose calibration is not complete because physical contacts remain overwhelmingly task-terminal and right-censored. The next scientific route is not yet launch-authorized: the release/window-flush shadow is planned but not launch-authorized; no provisional-boundary `imp_max_p` or canary is selected. See `docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md`. `scripts/lightning_pair.sh` is retained as a legacy Lightning.ai launcher, not the current route.
**Completed Step-1 diagnostic authorization:** The frozen-checkpoint no-learning survey (`41272370`) and the one authorized matched Vega activation canary (`41272362_[0-1]`) both completed without retry. The canary used the uniform `0.9` diagnostic boundary and `imp_max_p=0.0` versus `0.5`, with velocity CaT active and all other treatment choices fixed. It is engineering plumbing evidence, not the scientific comparison and not a hardware-limit claim; see `docs/results/2026-08-15_z1_impulse_cat_step1.md` and `docs/research/reward-design/IMPULSE_CAP_PROVENANCE.md`.
**Completed 500-iteration diagnostic authorization (2026-08-15 to 2026-08-17):** The one authorized uniform-`0.9` extension and its matched evaluation completed without retry/requeue. It establishes a one-training-seed simulation result, not a provisional-boundary dose, hard clamp, hardware-safety result, or independent-seed claim. The native verdict and compact evidence are in `docs/results/2026-08-17_z1_impulse_diag90_500_evaluation.md`; the separate release/window-flush shadow in `docs/superpowers/plans/2026-08-17-z1-impulse-diag90-release-window-flush-shadow.md` is not launch-authorized.
**Current VIC prototype route:** Native Z1 VIC-TT implementation and qualification are banked under `docs/superpowers/specs/2026-08-13-z1-native-vic-design.md` and `docs/superpowers/plans/2026-08-13-z1-native-vic.md`. The one seed-2 engineering canary authorized by `docs/superpowers/specs/2026-08-13-z1-vic-canary-design.md` and `docs/superpowers/plans/2026-08-13-z1-vic-canary.md` is complete and banked in `docs/results/2026-08-14_z1_vic_seed2_canary.md`; that plan authorizes no additional seeds or formal FIC–VIC comparison.
**Entry scripts:** `scripts/train.py` · `scripts/play.py` ·
`scripts/impulse_cat_activation_survey.py` (frozen-checkpoint Step-1 no-learning binding/attribution survey) ·
`scripts/analyze_vic_impulse_diag90_500_evaluation.py` (deterministic paired initial-episode bootstrap and compact result banker) ·
`scripts/slurm/vega_vic_impulse_diag90_canary.sbatch` (authorized two-arm, 50-iteration diagnostic activation canary) ·
`scripts/slurm/vega_vic_impulse_diag90_500.sbatch` (authorized matched 500-iteration diagnostic learning-pressure pair) ·
`scripts/slurm/vega_fic_direct_reference_smoke.sbatch` (banked direct-reference FIC CUDA baseline/reproducibility) ·
`scripts/slurm/vega_fic_direct_reference.sbatch` (banked direct-reference FIC training reproducibility) ·
`scripts/slurm/vega_vic_canary.sbatch` (completed guarded one-seed VIC-TT engineering-canary route) ·
`evaluation/joint_position/evaluate_fic_pilot.py` (compact direct-reference FIC evaluator) ·
`scripts/lightning_pair.sh` (legacy Lightning.ai prior-vs-none launcher) ·
`scripts/diag_policy_trace.py` (strike-vs-press classifier + peak-|q̇| eval) ·
`scripts/eval_impulse.py` + `scripts/eval_impulse.sh` (**standalone C3 checkpoint eval**: per-joint Λ max/p95, worst Λ/cap ratios, delivered impulse, success — one summary.csv row per checkpoint, same-env cross-arm protocol) ·
`scripts/diag_impulse_trace.py` (per-substep impulse trace) · `scripts/compare_runs.py` ·
`scripts/render_reference.py` (headless strike render) · `scripts/verify_cat_soft.py` + `scripts/smoke_cat_soft.py` (CaT plumbing).
**Tests:** `tests/` (unit suite incl. `tests/test_docs_current.py`, the freshness guard for this very index).
