# unitree_rl_mjlab — agent context

## ⚠️ Thesis context (read this first)

This repo is the development platform for a Master's thesis at TU Munich (ATARI Lab, Prof. Khadiv): *impact-safe contact-rich manipulation*, ultimately targeting the Unitree G1 humanoid with a **variable-impedance action space** and an **explicit per-joint impulse constraint**.

> **DIRECTION UPDATE (2026-06-17, user):** The **Z1 is now the primary platform** — build and validate the *full* thesis machinery here first: soft `γ(1−δ)` CaT, the substep-accumulated impulse constraint, and **variable impedance** (the policy commanding per-joint stiffness via `set_gains`). **The G1 is optional future replication, not the current focus** — do not hyperfixate on it. Sequencing: get the **fixed-impedance** results complete first (with real soft CaT), **then** add variable impedance at the end. This supersedes the earlier "Z1 is a fixed diagnostic; do variable impedance fresh on the G1; don't retrofit VIC into the Z1" framing.

**Authority contract:** Current truth = code > `docs/README.md` index > the living docs it lists.
`docs/archive/**` and dated records are historical evidence — never act on them without checking
the index.

**Docs index:** `docs/README.md` — the current-truth map, the onboarding reading order, the "how we got here" journey (each step linking its archived evidence), and the code entry map. Start there for any docs question. The three repo-root direction docs (`thesis_synthesis.md`, `thesis_direction_update.md`, `thesis_handoff_brief_original.md`) are bannered dated records; the current direction is the DIRECTION UPDATE above plus the curated thesis digest.

**Curated thesis material:** `docs/thesis/README.md` — the running, curated digest of confirmed thesis-bound contributions, decisions, defense points, and citations (distilled from `docs/research/`; exhaustive detail stays there). Add to it as material becomes thesis-ready. The constraint-mechanism justification + cited literature map lives in `docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md`; the faithful soft-CaT design + decisions in `docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md`. The plan to extend that machinery from joint-velocity to the **per-joint impact-impulse** constraint (the thesis's headline contribution) is `docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md` — note its top decision-to-confirm (soft-CaT vs the docs' CMDP/Lagrangian) and the qfrc_constraint-contamination blocker it resolves (the contaminant is dof-FRICTION, ~41–48% of the raw contact-window Λ_j and EE-dependent, spread across the load-bearing arm joints — NOT the weld, whose contribution is tiny; re-derive per-joint figures via `derive_impulse_thresholds.py`).

**Decisions that affect what code is worth writing:**
- The thesis is **single-policy, variable-impedance, online RL**. The "two-level SURE+RL" and "generate-then-track / DeepMimic" architectures were considered and **walked back by the supervisor**. Do not implement either without re-confirming with Khadiv.
- The Z1 is the primary development platform (see Direction Update above): the full program — soft CaT, impulse constraint, then variable impedance — is built and validated on the Z1 first. Variable impedance comes **after** the fixed-impedance results are complete. The G1 is optional future replication.

## Reward design (Z1 hammer task — Phase 0 only)

If working on rewards for the Z1 hammer task, **start by reading**:

- `docs/research/hammering_reward_design_deep_dive_v2.md` — **v2 deep dive (2026-05-29), the current entry point**: external-literature pass (29 net-new citations). Covers repetitive hammering (reward machine), the position-only DiffIK action-space constraint (impact lever is momentum `m_eff·v_axial`, not force), safety-constrained RL, and a reward-hacking checklist. Flags several v1 recommendations (orientation gate, torque/impedance terms) as not actionable on the current action space. (Supersedes the older `DEEP_RESEARCH_REPORT.md`, archived under `docs/archive/` on 2026-06-17.)
- `docs/research/tracking_impact_impulse_design_research.md` — **design research (2026-06-10)**: track-a-reference + maximize-impact + bound-joint-impulse architecture (6-axis verified lit sweep, decisions D1–D6). Its staged plan is archived (`docs/archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`) — the weak annealed reward prior `r_imit` shipped; the impulse machinery it staged was superseded by the shipped per-event-pulse soft-CaT (`docs/research/reward-design/IMPULSE_CAT_IMPL_PLAN.md`). Key corrections (still binding): never penalize raw `‖Δq̇‖²` (use excess-over-threshold or a CMDP constraint); never clock-time velocity tracking near contact (position-only, ante-impact, phase-indexed); impulse must be accumulated at substep rate.

Load-bearing facts that override the spec:

- **Trust the code, not the spec.** The implemented **7-term** baseline in `src/tasks/hammer/hammer_env_cfg.py` is the source of truth. `docs/archive/RECOMMENDED_REWARD_SPEC.md`'s 9-term "complete config" is aspirational (archived 2026-06-17); the user's strategy is **augment-not-replace** (add a term only when a specific failure mode is observed) — with one deliberate, literature-backed exception: changes **#1 + #2** (2026-06-02) rebalanced `nail_depth_delta` 2000→600 and added the double-gated `impact_progress` term up front. See `docs/archive/IMPACT_PROGRESS_IMPL_SPEC.md` (term shipped; its design rationale is preserved there).
- **Live weights** (read these from the code, not the spec): `approach=0.1`, `nail_driven=2.0`, `nail_depth_delta=600`, `impact_progress=8`, `completion=100`, `action_rate=−0.01`, `joint_pos_limits=−10`. `validate_rewards.py` reads them dynamically via `reward_manager.get_term_cfg(name).weight` — no need to keep its constants in sync.
- **Pre-training gate:** before any GPU training, run `validate_rewards.py` (**all phases A–M** must pass — Phase I covers `impact_progress`, Phase M the impulse-CaT arm) and `verify_contact_sensor.py`. Also `verify_reward_setup.py` (random-policy sweep; refreshed to current API 2026-06-02). Unit tests: `pytest tests/test_impact_progress_reward.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_delivered_impulse_reward.py tests/test_cat_soft_hook.py`.
- **Q1/Q2 are resolved** (V1 training, 2026-06-17): the single strike is feasible and `air_time_bonus` is not load-bearing (NOT added). After any EE/pose change (e.g. the pending L6 NEAR_NAIL re-solve), re-run `docs/research/reward-design/test_single_strike.py` to re-confirm strike feasibility.
- Open questions (Q3, Q6, Q7, Q9, Q11, Q12) require either GPU training or hardware — see `docs/research/reward-design/OPEN_QUESTIONS.md`.

## Environment

- Conda env: `unitree_mjlab` — **mjlab 1.4.0**, mujoco 3.8.1, mujoco_warp 3.8.1.
- Z1 + hammer assets live in a sibling repo: `~/repos/safe_impact_manipulation/hammer_z1_env/assets/`.
- View scene: `python hammer_z1_env/view.py` (normal mode, weld active) or `mjpython hammer_z1_env/view.py --no-weld` (sliders mode, gravity disabled, requires `mjpython` on macOS).
- **Visual inspection of the task (works headless on macOS — MuJoCo offscreen GL needs no display/MUJOCO_GL):** `python scripts/render_reference.py --distance 0.85 --elevation -25` renders the open-loop T1 strike to PNG frames + montage + mp4 under `/tmp/hammer_ref/` (agents can then read the PNGs). Interactive/browser view of the same: `python scripts/play_reference.py` (viser, prints a URL). Both drive the scripted reference, not a trained policy.
