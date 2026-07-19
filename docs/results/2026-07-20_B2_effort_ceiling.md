# B.2 — effort-limit ablation: the ceiling is CONTROLLER-bandwidth, not actuator torque (2026-07-20)

**Question:** is the fixed-impedance ~0.87× i_ref / ~1.4 m/s delivered-impulse ceiling actuator-limited
or reward-limited? **Answer: NEITHER — it is CONTROLLER-BANDWIDTH-limited.**

CPU diagnostic (`docs/results/assets/2026-07-17_fixed_impedance_diag/probes/effort_sweep.py`): scale ONLY
the arm `effort_limit` (30/60 → ×1/1.5/2/3), gains/armature/action/nail FIXED, drive the scripted strike
at nominal (speed 1) and aggressive (speed 6) tracking:

| speed | effort×1 | ×1.5 | ×2 | ×3 | reading |
|---|---|---|---|---|---|
| nominal (1) | v_touch 1.334, deliv 1.000× i_ref | 1.334 / 0.986× | 1.334 / 0.986× | 1.334 / 0.986× | flat |
| aggressive (6) | v_touch 1.408, deliv 0.678× | 1.408 / 0.678× | 1.408 / 0.678× | 1.408 / 0.678× | flat |

**Contact speed and delivered impulse are IDENTICAL to 4 decimals across effort ×1→×3.** Tripling the
motor torque budget (30→90 N·m) moves the impact ceiling by exactly zero. 30 N·m already saturates what
the controller demands; the arm is NOT torque-starved.

**So the ~1.4 m/s ceiling is set by the DiffIK controller bandwidth** — `delta_pos_scale=0.15` (head-step
per control step) + kp/kd tracking — NOT by actuator torque. (Aggressive commanding raises v_touch a
little, 1.334→1.408, because it demands the controller track faster; but more TORQUE at fixed commanding
does nothing.) Consistent with kp∉M(q): neither higher motor torque nor torque-based VIC would raise the
BALLISTIC impact ceiling.

**Implications:**
- The Vega effort-ablation A/B (baseline vs 1.5× effort on a trained policy) is now LOW-VALUE — the CPU
  physics predicts a flat delivered result (effort is a trajectory-independent no-op on contact speed).
- The actual controller-bandwidth levers the result points to are **delta_pos_scale** (bigger position
  step → faster head, but coarser strike placement — a real sim2real accuracy tradeoff) and **kp/kd**
  (stiffer tracking → faster convergence). These, not effort/torque, gate the fixed-impedance impact
  ceiling. Sweep delta_pos_scale next (CPU-first).
- Caveat: single scripted-strike deliver numbers at speed 6 are one-shot noisy (0.34–0.75 across earlier
  runs); the ROBUST signal is v_touch (contact-onset speed) + the effort-invariance, which is exact.

Protocol: `docs/results/2026-07-20_eval_protocol.md`. Menu: `docs/results/2026-07-19_next_experiments_codex.md`.

---

## delta_pos_scale sweep — the ceiling is the real Z1 JOINT-VELOCITY limit (2026-07-20)

B.2 showed effort (torque) is a no-op and pointed at the DiffIK controller bandwidth (`delta_pos_scale`)
as the lever. `dps_sweep.py` sweeps it (0.15 baseline → 0.3/0.5/1.0), saturated descent, and tracks
max arm |q̇| vs the real Z1 limit 3.1415 rad/s:

| delta_pos_scale | v_touch (m/s) | deliv ×i_ref | max\|q̇\| (rad/s) | reachable on real Z1? |
|---|---|---|---|---|
| **0.15 (shipped)** | 1.267 | 1.01 | **2.556** | **YES** (< 3.1415) |
| 0.30 | 2.745 | 1.17 | 4.945 | NO |
| 0.50 | 3.519 | 0.16* | 5.122 | NO |
| 1.00 | 4.963 | 0.47* | 5.223 | NO |
(*delivered at dps≥0.5 is single-strike chaos — v_touch + max\|q̇\| are the robust signals.)

**Contact speed IS liftable via controller bandwidth (1.3 → 5 m/s), so the ceiling is not a physics wall
— BUT every faster scale demands joint velocities the real Z1 cannot produce.** The shipped
delta_pos_scale=0.15 already runs the dominant joint to 2.56 rad/s, ~81% of the 3.1415 rad/s hardware
ceiling; dps=0.30 needs 4.95 rad/s (unreachable). So the reachable contact-speed ceiling (~1.4 m/s) is set
by the **real Z1 per-joint velocity limit**, and 0.15 is already calibrated to it.

## ⚠ SUPERSEDED VERDICT — kept for the record, CORRECTED below
"Why does delivered impulse plateau at ~0.87× i_ref / ~1.4 m/s?" — I originally answered:
- NOT reward-limited (Phase 2); NOT torque-limited (effort ×1→×3 "exact no-op"); controller-bandwidth
  capped by the real Z1 3.1415 rad/s joint-velocity limit ⟹ "a HARDWARE JOINT-VELOCITY limit."

**This verdict was WRONG on two counts, found by an independent Codex re-evaluation (2026-07-20). See below.**

## ✅ CORRECTED VERDICT (independent Codex re-evaluation, 2026-07-20)
Two errors in the superseded verdict:

1. **The effort "exact no-op" was a PROBE BUG, not physics.** `effort_sweep.build()` mutated a
   **module-level SHARED articulation** (`z1_hammer_env_cfg()` returns cfg instances that share the SAME
   robot articulation object — verified: `articulation shared identity == True`), so the sweep loop
   cumulatively contaminated its own effort settings. Re-run in **isolated fresh processes**, effort ×2
   raises aggressive contact speed **+7.5%** (1.308→1.407 m/s) then plateaus — small but NONZERO. And the
   baseline straight-down strike IS genuinely torque-saturated: peak joint torques ≈ [1.07, **60, 30, 30**,
   0.53, 0.08] N·m (j2/j3/j4 AT their clamps), ≥1 joint at the clamp for **87/117 pre-contact substeps**.
   So torque is a (weak) contributor, not a proven non-factor. `effort_sweep.py` is now fixed (deepcopy).

2. **The ~1.4 m/s ceiling is TRAJECTORY-specific, NOT a hardware wall.** Head speed = J(q)·q̇, so a
   coordinated multi-joint whip reaches higher end-effector speed than the straight-down strike at the SAME
   per-joint |q̇|. At the hardware rail (max_j|q̇| ≤ 3.1415 rad/s) the **global downward head-speed box
   optimum is ≈ 4.22 m/s** (cf. `ceiling.py`'s 3.97 m/s kinematic ceiling); the shipped straight-down
   strike exploits only **~45%** of it (1.4 of ~4.2 m/s). So the fixed-impedance CONTACT-SPEED ceiling is a
   **trajectory/controller-shaping limit a trained policy could beat**, not a hardware velocity wall. My
   "0.15 is calibrated to the hardware cap" reasoning conflated the straight-down |q̇| (2.56 rad/s) with the
   reachable envelope.

**What SURVIVES the correction (and is now the real story):** the DELIVERED-IMPULSE / constraint ceiling is
dominated by the **soft yielding 7 g target**, not the velocity ceiling. Codex reproduced Phase-0 exactly:
solref×2 → Λ −76.3% + delivered −76.55% (contact/yield/press-dependent, not pure pre-impact velocity); at
the 3.1415-box velocity max the worst **inelastic (e≈0) Λ/cap is still only ≈0.616**; the inelastic
cap-crossing speed is 6.95 m/s (above even the 3.97 m/s kinematic box). So **even a maximal whip would not
make the realistic e≈0 ballistic constraint bind** under the current 7 g nail — the constraint vacuity is a
TARGET-physics result, robust to the velocity correction.

**Net corrected picture:**
- Constraint vacuity (Phase 0 / A3) — **holds**, driven by the soft yielding target (not velocity).
- "Reward can't maximize impulse" (Phase 2) — **partially reopened**: the Phase-2 policy strikes
  straight-down; a policy able/incentivized to **whip** could reach higher contact speed (toward ~4 m/s)
  and thus more delivered impulse, WITHIN the joint-velocity budget. The maximization question is NOT as
  closed as claimed — the straight-down trajectory, not a hardware wall, is the binding limit.
- VIC is still motivated (decouples contact speed from steady-state joint velocity), but the fixed-impedance
  headroom is larger than I claimed — a whip-capable fixed-impedance policy should be tested FIRST.

**DECISIVE NEXT EXPERIMENT (Codex's recommendation, CPU):** direct-trajectory optimization over the actual
Cartesian DiffIK action interface — optimize 10–20 pre-contact actions a_t∈[−1,1]³ under the unchanged
fixed PD + 30/60 N·m effort limits, enforce max|q̇|≤3.1415 rad/s at every 2 ms substep + joint limits +
terminal hammer-face alignment, maximize pre-contact downward head speed; replay winners in fresh processes.
Settles whether the ~4 m/s kinematic opportunity is DYNAMICALLY reachable through the real controller/actuator
envelope, or collapses back toward ~1.6–1.8 m/s. If reachable → train a whip-capable policy and re-test
impulse maximization before concluding VIC is required.

[[softcat-velocity-result]] [[z1-velocity-bound-finding]] — the trained policy's |q̇|>π was for a specific
learned trajectory, NOT proof the straight-down velocity is the global reachable max.
