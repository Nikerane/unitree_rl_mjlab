> ⚠️ **ARCHIVED 2026-07-05** — superseded by `docs/thesis/README.md` + `CLAUDE.md` (current direction).
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Z1 Hammer Project — Handover

**Audience:** an agent or engineer picking up this branch cold.
**Date written:** 2026-06-09. **Revised:** 2026-06-10 with thesis-context reframing.
**Status:** Phase 0 (diagnostic baseline) for a TU Munich thesis on the G1. See §0 below.

---

## 0. Thesis context — read first (added 2026-06-10)

This Z1 work is **Phase 0** of a Master's thesis at TU Munich, ATARI Lab, Prof. Khadiv. The thesis topic is *impact-safe contact-rich manipulation on the Unitree G1 humanoid*. The full direction docs live at the repo root: `thesis_synthesis.md` (synthesis), `thesis_direction_update.md` (current architecture, post-supervisor), `thesis_handoff_brief_original.md` (original conceptual pillars; architecture superseded).

**Two consequences for anyone working on this branch:**

1. **The thesis architecture is single-policy, variable-impedance, online RL** — *not* two-level SURE+RL, *not* DeepMimic-style generate-then-track. Section 5 of this document describes a "generate-then-track" architecture as a candidate v2 for the Z1 task. **The supervisor walked that direction back for the thesis.** It is preserved below for historical context and as a reasonable candidate for the standalone Z1 task only; **do not implement it as the thesis architecture without re-confirming with Khadiv.**

2. **The Z1 task is preparation, not the thesis itself.** The thesis runs on the G1 with a different action space (joint position + time-varying stiffness, per Bogdanovic, Khadiv, Righetti 2020) and an explicit per-joint impulse constraint computed via the Pinocchio impact map. Retrofitting variable impedance into the current Z1 task probably wastes engineering time the thesis needs elsewhere. The right move on this branch is to finish its diagnostic value (the press-exploit finding + cheap physics fixes), then start fresh on the G1.

The press-exploit finding from §4 below has direct thesis value: it is empirical evidence that a position-only stiff-PD action space allows quasi-static solutions to a striking task, which is exactly what variable impedance is designed to fix. Write this up.

---

## 1. What we are trying to do

We are training a Unitree Z1 6-DOF arm with a rigidly-mounted hammer to drive a nail into a wooden block in simulation (mjlab / MuJoCo-Warp). The arm is controlled by a Differential Inverse Kinematics action layer that exposes a **3-D position delta of the hammer head** — the policy commands "move the hammer this much in x/y/z each step," and an underlying solver turns that into joint position targets driven by stiff PD controllers. The arm has **no torque or impedance command**, only position. This action-space choice is the single most important constraint in the whole project — it shapes everything downstream.

The success criterion is that the nail (a `slide` joint with range 0–7.5 cm) reaches a depth threshold of about 7 cm. The episode terminates on success or after a 20 s timeout. Sim runs at 500 Hz with `decimation=10` (a 50 Hz control step).

The work lives in two repos that must move in lockstep:

- **`unitree_rl_mjlab`** (this repo) — the RL side: env config, reward terms, scripts, tests, research notes. The active branch is `hammer-z1`.
- **`~/repos/safe_impact_manipulation`** (`main` branch) — the assets side: MuJoCo XMLs for the scene, the standalone viewer, and the IK solver. Two scene XMLs (`hammer_nail_scene.xml` for visualisation, `nail_block_scene.xml` for mjlab training) must stay synced; both files now have warning comments about this.

---

## 2. Why this task is hard (the architectural constraint)

A hammer drives a nail via **impulse** — a short, high-velocity contact that transfers momentum into the nail. On a real arm, you would deliver that impulse by controlling either contact force directly (force control) or end-effector impedance (compliant control). We have neither — we have position commands going through stiff PD servos. That means:

- We cannot command "hit hard" in any direct way. We can only command the hammer head to be in a slightly lower position next step.
- During contact, the stiff position servo behaves as a **composite rigid body**: the effective mass at the hammer head is determined by arm configuration `q`, not by the commanded action. The controllable impact lever is `m_eff(q) · v_axial` — the product of pose-determined effective mass and pre-impact axial velocity. (Reference: Wang, Dehio & Kheddar 2022, RA-L.)
- The discrete-time saturation of the IK action and the contact solver together create a regime where the policy can solve the task by **pressing slowly** rather than **striking impulsively**. Empirically we have observed this: a constant downward action drives the nail from 1.3 mm to 66 mm in one sustained push to success, which is not what we want.

This is the central problem the project keeps running into. Two distinct architectural ideas have been tried; the second is now in the middle of being committed to.

---

## 3. Architecture v1 — augmented reward shaping (May)

The first approach (committed and live as of `baf586b`) is a **7-term reward function** that tries to make impulsive striking the highest-return policy without changing the action space. The terms are:

1. `approach` (+0.1) — Gaussian on hammer-to-nail distance, narrow std (0.08 m). Keeps the arm near the nail; low weight to prevent hovering being optimal.
2. `nail_driven` (+2.0) — Gaussian on absolute nail depth, std 0.03. Provides a near-goal gradient.
3. `nail_depth_delta` (+600) — stateful progress reward. Pays `max(0, current_depth − max_depth_so_far)` each step. Originally weight 2000, recently rebalanced down to 600 so the dense total over a full drive (~30–40 reward units) no longer dwarfs the +100 completion bonus. Has a 4 mm dead-zone on the tracked depth to absorb a gravity-settling artifact in the constraint solver.
4. **`impact_progress` (+8)** — the term added in early June. A *double-gated* momentum reward: `(v_axial / v_expected) × indicator[first_contact] × indicator[Δdepth > eps]`. Velocity is finite-differenced from the hammer head site position (robust against lazy site-vel evaluation). Pays the policy for axial speed at the moment of a fresh contact, *only if* that contact actually advances the nail. The two gates were intended to make the term unfarmable by scraping or tapping.
5. `completion` (+100) — sparse success bonus, fires once on the step the nail crosses the success threshold (termination fires on the same step, so the bonus cannot re-fire).
6. `action_rate` (−0.01) — standard smoothness penalty on consecutive-action deltas.
7. `joint_pos_limits` (−10.0) — soft penalty for joint-limit violations; weight was raised from −1 to make safety dominate over task reward in extended configurations.

All seven terms are validated end-to-end by `validate_rewards.py`, which runs nine phases of state injection + reward assertion against the live env. The script reads weights dynamically from the reward manager via `get_term_cfg(name).weight`, so future weight drift will not break the test. There are also unit tests for the stateful terms (`test_impact_progress_reward.py`, 11 stub-env cases) and for the nail physics (`test_nail_physics.py`, `test_hammer_physics.py`). A CPU smoke training run (50 iterations, 16 envs) ran clean.

The supporting research that produced this architecture is in `docs/research/reward-design/` — six markdown files plus the validation scripts, indexed by `DEEP_RESEARCH_REPORT.md`. That report was produced by a deep-research pipeline and audited by an independent model pass; it carries an explicit verdict of **"trust the code, not the spec"** because the implemented baseline diverges intentionally from the spec's aspirational nine-term design.

---

## 4. What broke this architecture

Two pieces of work in late May / early June converged on a critical finding.

**The v2 deep dive** (`docs/research/hammering_reward_design_deep_dive_v2.md`, 2026-05-29) was a second-pass external literature sweep — six parallel literature-search agents and 18 self-verified citations. It argued, drawing on Wang/Dehio/Kheddar (2022) and Ti et al. (2024), that on a position-only DiffIK action space, the entire "swing vs press" framing is the wrong abstraction. The controllable quantity is `m_eff(q) · v_axial`, not contact force, and rewarding force-like quantities or even raw velocity bonuses without addressing the action-space limitation is treating a symptom rather than the disease. The deep dive recommended either (a) recalibrating nail physics so pressing is physically infeasible, or (b) restructuring the architecture so the strike is generated upstream and the RL only tracks it.

**The v2 peer review** (`docs/research/reward-design/PEER_REVIEW_v2.md`, 2026-06-02) was a five-reviewer simulated panel plus a Devil's Advocate pass. The reviewers, working blind to each other, all independently identified the same flaw: **the `impact_progress` term does not actually do what it claims to do.** Because `first_contact` fires on exactly one step (the moment a fresh contact begins), during a sustained press the term pays exactly once and then zeroes out — while `nail_depth_delta` continues to reward the press all the way to completion. The Δdepth gate prevents *scraping* (touching without driving), but it does not prevent *pressing* (a single contact that drives the nail continuously). The Devil's Advocate flagged this as a CRITICAL falsifiability failure: the term was designed to prevent the press, but the sim solves the task by pressing, with the term contributing only ~3.88 reward to a successful press episode. Verdict: unanimous MAJOR REVISION, CRITICAL block on Accept.

The takeaway is that v1 — at least as currently implemented — does not solve the problem we set out to solve. It may train a working policy in sim, but the policy will exploit the press regime that the design was supposed to eliminate.

---

## 5. Architecture v2 — generate-then-track (the new direction)

The design that emerged from the peer review and the v2 lit review is **generate-then-track**, spelled out in `docs/research/reward-design/IMPACT_TRACKING_REWARD_SPEC.md`. The architecture has two stages:

**Stage-1 (offline, not RL):** generate an optimal strike trajectory `q*(φ), q̇*(φ), p*(φ)` parameterised by a phase variable `φ`. "Optimal" means the trajectory co-optimises delivered impact momentum (`m_eff · v` along the nail axis) against joint-velocity recoil (`‖Δq̇‖`). Three independent papers converge on this objective: Vu et al. (2026) co-optimise `m_eff · v` in a QP on HRP-5P; Ti et al. (2024) maximise directional velocity manipulability for real nail hammering on a 7-DoF arm; Wang/Kheddar (2019/2023) embed impact dynamics as QP constraints. We start with a hardcoded back-swing → strike pattern as the v0 reference and replace it later with a real trajectory optimiser.

**Stage-2 (online, PPO):** the policy *tracks* the Stage-1 reference and is allowed to refine it. The reward has the imitation + task structure that the motion-imitation literature shows is required for striking — pure tracking never learns the strike on its own (DeepMimic ablations confirm this), and pure RL on this action space falls into the press as we saw. The reward becomes:

`R = w_I · r_imit + w_G · (r_impact − λ · r_recoil) + r_safety`

with:

- **`r_imit`** — DeepMimic-style pose + velocity + end-effector tracking against the reference, weighted `0.7 · r_pose + 0.1 · r_vel + 0.2 · r_ee`. CoM term dropped because the base is fixed.
- **`r_impact`** — windowed axial impulse `∫ F_axial dt` over the contact window, gated on nail-depth progress (the geometric anti-artifact cross-check inherited from the v1 work). This requires either extending the existing `ContactSensor(found, force)` with `history_length > 1`, or adding a wrist force/torque sensor.
- **`r_recoil`** — `‖Δq̇‖²`, the squared joint-velocity jump across the contact step, measured directly from `joint_vel`. The analytic `Δq̇ = M⁻¹Jᵀι` form (with CRB-corrected inertia per Wang 2022) is reserved for an optional predictive critic later, because computing `J` and `M` every step may be expensive.
- **`r_safety`** — the existing `action_rate`, `joint_pos_limits`, and `completion` survive unchanged; later we escalate `r_recoil` to a hard CMDP constraint via PID-Lagrangian (Stooke et al. 2020).

Starting weights are `w_I ≈ 0.6, w_G ≈ 0.4` (HMAMP) or `0.7 / 0.3` (DeepMimic), with `λ` small and ramping. The training recipe is Reference State Initialisation + Early Termination, which DeepMimic shows is critical for dynamic skills.

In this architecture, the press is dissolved by **two independent guards**: the impact term has `v_axial ≈ 0 → I_axial ≈ 0 → r_impact ≈ 0` during a press, and the imitation term structurally penalises departing from the back-swing → strike reference. The v1 terms `approach`, `nail_driven`, and `impact_progress` are subsumed and removed. `nail_depth_delta`, `completion`, `action_rate`, and `joint_pos_limits` are kept.

---

## 6. Where we are right now

The v1 architecture is fully implemented, committed, and validated. The v2 design is documented but not yet implemented. The decision to commit to v2 has been made on paper but not in code, and there are reasonable arguments for trying a cheap v1 physics fix first to see whether the press exploit is genuinely an architectural problem or just a poorly-calibrated nail. The relevant cheap experiments are listed in `docs/FUTURE_UPDATES.md`:

- **Clip policy actions to [−1, 1]** — currently the Gaussian PPO policy outputs actions up to ±7.8, which the IK then saturates. This breaks the smoothness penalty's intended scale and is hostile to sim-to-real transfer.
- **Add reset randomisation** — currently the arm and nail reset to identical positions every episode, so the trained policy memorised one fixed 5-step trajectory rather than learning a closed-loop strike. Action std at convergence was 0.12, which is consistent with this.
- **Add a starting-pose curriculum** — start the arm near the nail, end at the neutral pose 26 cm away.
- **Raise nail `frictionloss` from 10 N to ~30 N** — at 10 N a hammer with 0.25 kg head can drive the nail 51 mm in a single step's worth of contact. At 30 N (with `clip_actions=1.0`), a realistic strike should only drive about 10 mm. This is the most likely single fix that could rule out the architectural change.
- **Redesign nail geometry** — currently the nail head passes through the block during driving. Either shrink the goal depth to 3.2 cm (matches current geometry) or raise the nail body so a full 7.5 cm drive ends with the head flush with the block surface.

If the cheap physics fixes (especially `clip_actions=1.0` + `frictionloss=30`) make pressing actually infeasible, then v1 may still be viable. If pressing persists, the v2 generate-then-track architecture is the right move.

There is also a substantial body of untracked-but-on-disk research that should be committed regardless of architectural choice: six files in `docs/research/` (the lit sweep runbook, the v2 deep dive, the new lit review, the impact-tracking reward spec, the peer review, and a hammering literature notes file). These are the load-bearing context for whichever direction is chosen, and they should not live only in the working tree.

Note also that `docs/research/reward-design/DEEP_RESEARCH_REPORT.md` is modified but uncommitted — it predates the architectural shift, so it should be either updated with a pointer to the new direction or left as a historical snapshot of v1's rationale.

---

## 7. What to do next

**Updated 2026-06-10 with thesis reframing.** Path B (generate-then-track) was deprioritised once the thesis direction settled on single-policy variable impedance — it's a reasonable architecture for the standalone Z1 task but it's the *wrong* preparation for the thesis. The current recommendation is Path A + Path C, then move to G1.

**Path A — finish the diagnostic value of the Z1 task (~1 week).** Apply the cheap fixes from `docs/FUTURE_UPDATES.md` (`clip_actions=1.0`, reset randomisation, curriculum optional, raise `frictionloss` to ~30 N, fix nail geometry). Retrain. Two outcomes, both useful:
- Press exploit dissolves → write up "fixed-PD striking task is feasible with calibrated physics" as the Phase-0 baseline result for the thesis introduction.
- Press exploit persists → write up "position-only stiff-PD action space allows quasi-static solutions even at calibrated friction" as direct empirical justification for the thesis's variable-impedance architecture.

Either outcome supports the thesis. This is the highest-priority work on this branch.

**Path C — commit the research corpus (immediate).** The untracked research docs (six files in `docs/research/`, the modified `DEEP_RESEARCH_REPORT.md`, the two thesis docs at the repo root, plus `thesis_synthesis.md`) should be committed so the planning context survives agent context loss. Free, and load-bearing for any future work.

**Path B — generate-then-track (deprioritised).** *Historical note for posterity.* When the Z1 work surfaced the press-exploit problem, the natural fix on a position-only action space looked like DeepMimic-style generate-then-track: a hardcoded back-swing → strike reference, an imitation reward + windowed impact impulse, recoil minimisation. The specification for this lives in `docs/research/reward-design/IMPACT_TRACKING_REWARD_SPEC.md`. The supervisor walked it back for the thesis on the grounds that TO and RL minimise the same objective and the redundancy was solving it twice. The single-policy variable-impedance design owns the same physics objectives directly. **Do not implement Path B as the thesis architecture; re-confirm with Khadiv if you think it's worth pursuing on the Z1 task standalone.**

**Path D (added) — start the G1 work.** Once Path A is finished and written up, the next move is to set up an arm-only fixed-base G1 task in mjlab, port the validated reward-design discipline (validate_rewards.py pattern, contact-sensor wiring) over, and add the variable-impedance action interface. The actual thesis starts here. See `thesis_synthesis.md` §6 for the suggested phasing.

A reasonable sequence: Path C immediately (free), Path A this week, Path D once Path A is written up.

---

## 8. Where to look first

The fastest way to verify any claim in this document is to read these files in order:

- `CLAUDE.md` (repo root) — agent-context pointer, lists live reward weights and pre-training gates.
- `src/tasks/hammer/hammer_env_cfg.py` — env config including the live reward dictionary. **This is the ground truth for what is actually running.**
- `src/tasks/hammer/mdp/rewards.py` — the implementation of `NailDepthDeltaTerm` and `ImpactProgressTerm` (the two stateful terms).
- `docs/research/reward-design/IMPACT_PROGRESS_IMPL_SPEC.md` — what the v1 impact term was meant to do.
- `docs/research/reward-design/PEER_REVIEW_v2.md` — the review that found `impact_progress` does not actually do what it claims.
- `docs/research/reward-design/IMPACT_TRACKING_REWARD_SPEC.md` — the proposed v2 reward design (UNTRACKED — read from the working tree).
- `docs/research/hammering_reward_design_deep_dive_v2.md` — the broader literature argument for v2 (UNTRACKED).
- `docs/FUTURE_UPDATES.md` — the cheap-fix list for Path A.
- `docs/research/reward-design/validate_rewards.py` — the pre-training gate; run this before any training.

Environment: conda env `unitree_mjlab` (mjlab 1.4.0, mujoco 3.8.1, mujoco_warp 3.8.1). To visualise the scene: `python ~/repos/safe_impact_manipulation/hammer_z1_env/view.py` (or `mjpython` with `--no-weld` for slider-based tuning on macOS).
