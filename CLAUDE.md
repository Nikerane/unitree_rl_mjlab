# unitree_rl_mjlab — agent context

## ⚠️ Thesis context (read this first)

This repo is the development platform for a Master's thesis at TU Munich (ATARI Lab, Prof. Khadiv): *impact-safe contact-rich manipulation*, ultimately targeting the Unitree G1 humanoid with a **variable-impedance action space** and an **explicit per-joint impulse constraint**.

> **DIRECTION UPDATE (2026-06-17, user):** The **Z1 is now the primary platform** — build and validate the *full* thesis machinery here first: soft `γ(1−δ)` CaT, the substep-accumulated impulse constraint, and **variable impedance** (the policy commanding per-joint stiffness via `set_gains`). **The G1 is optional future replication, not the current focus** — do not hyperfixate on it. Sequencing: get the **fixed-impedance** results complete first (with real soft CaT), **then** add variable impedance at the end. This supersedes the earlier "Z1 is a fixed diagnostic; do variable impedance fresh on the G1; don't retrofit VIC into the Z1" framing.

**Authoritative direction docs (read in this order):**
1. `thesis_synthesis.md` — deep-research synthesis over the two source docs below; current architectural truth.
2. `thesis_direction_update.md` — the supervisor-conversation update; the *current* architecture (single policy, variable impedance, online RL).
3. `thesis_handoff_brief_original.md` — the original brief; conceptual pillars (impulse-not-energy, preparation-not-reaction, hitting flux) survive intact. Architecture sections are superseded.

**Curated thesis material:** `docs/thesis/README.md` — the running, curated digest of confirmed thesis-bound contributions, decisions, defense points, and citations (distilled from `docs/research/`; exhaustive detail stays there). Add to it as material becomes thesis-ready. The constraint-mechanism justification + cited literature map lives in `docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md`; the faithful soft-CaT design + decisions in `docs/research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md`.

**Decisions that affect what code is worth writing:**
- The thesis is **single-policy, variable-impedance, online RL**. The "two-level SURE+RL" and "generate-then-track / DeepMimic" architectures were considered and **walked back by the supervisor**. Do not implement either without re-confirming with Khadiv.
- The Z1 is the primary development platform (see Direction Update above): the full program — soft CaT, impulse constraint, then variable impedance — is built and validated on the Z1 first. Variable impedance comes **after** the fixed-impedance results are complete. The G1 is optional future replication.

## Reward design (Z1 hammer task — Phase 0 only)

If working on rewards for the `hammer-z1` branch, **start by reading**:

- `docs/research/hammering_reward_design_deep_dive_v2.md` — **v2 deep dive (2026-05-29), the current entry point**: external-literature pass (29 net-new citations). Covers repetitive hammering (reward machine), the position-only DiffIK action-space constraint (impact lever is momentum `m_eff·v_axial`, not force), safety-constrained RL, and a reward-hacking checklist. Flags several v1 recommendations (orientation gate, torque/impedance terms) as not actionable on the current action space. (Supersedes the older `DEEP_RESEARCH_REPORT.md`, archived under `docs/archive/` on 2026-06-17.)
- `docs/research/tracking_impact_impulse_design_research.md` — **design research (2026-06-10)**: track-a-reference + maximize-impact + bound-joint-impulse architecture (6-axis verified lit sweep, decisions D1–D6). Staged plan: `docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (stages T0–T5, **pre-implementation, Z1-only — G1 deferred**; T0 = Path A physics fixes + Q1; primary arm = weak annealed reward prior `r_imit`, per plan changelog). Key corrections: never penalize raw `‖Δq̇‖²` (use excess-over-threshold or a CMDP constraint); never clock-time velocity tracking near contact (position-only, ante-impact, phase-indexed); impulse must be accumulated at substep rate.

Load-bearing facts that override the spec:

- **Trust the code, not the spec.** The implemented **7-term** baseline in `src/tasks/hammer/hammer_env_cfg.py` is the source of truth. `docs/archive/RECOMMENDED_REWARD_SPEC.md`'s 9-term "complete config" is aspirational (archived 2026-06-17); the user's strategy is **augment-not-replace** (add a term only when a specific failure mode is observed) — with one deliberate, literature-backed exception: changes **#1 + #2** (2026-06-02) rebalanced `nail_depth_delta` 2000→600 and added the double-gated `impact_progress` term up front. See `docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md`.
- **Live weights** (read these from the code, not the spec): `approach=0.1`, `nail_driven=2.0`, `nail_depth_delta=600`, `impact_progress=8`, `completion=100`, `action_rate=−0.01`, `joint_pos_limits=−10`. `validate_rewards.py` reads them dynamically via `reward_manager.get_term_cfg(name).weight` — no need to keep its constants in sync.
- **Pre-training gate:** before any GPU training, run `validate_rewards.py` (**9 phases** must pass — Phase I covers `impact_progress`) and `verify_contact_sensor.py`. Also `verify_reward_setup.py` (random-policy sweep; refreshed to current API 2026-06-02). Unit tests: `pytest tests/test_impact_progress_reward.py`.
- **Before deciding `air_time_bonus` is essential:** run `docs/research/reward-design/test_single_strike.py` to resolve Q1 (single-strike feasibility).
- Open questions (Q2, Q3, Q6, Q7, Q9, Q11, Q12) require either GPU training or hardware — see `OPEN_QUESTIONS.md`.

## Environment

- Conda env: `unitree_mjlab` — **mjlab 1.4.0**, mujoco 3.8.1, mujoco_warp 3.8.1.
- Z1 + hammer assets live in a sibling repo: `~/repos/safe_impact_manipulation/hammer_z1_env/assets/`.
- View scene: `python hammer_z1_env/view.py` (normal mode, weld active) or `mjpython hammer_z1_env/view.py --no-weld` (sliders mode, gravity disabled, requires `mjpython` on macOS).
- **Visual inspection of the task (works headless on macOS — MuJoCo offscreen GL needs no display/MUJOCO_GL):** `python scripts/render_reference.py --distance 0.85 --elevation -25` renders the open-loop T1 strike to PNG frames + montage + mp4 under `/tmp/hammer_ref/` (agents can then read the PNGs). Interactive/browser view of the same: `python scripts/play_reference.py` (viser, prints a URL). Both drive the scripted reference, not a trained policy.
