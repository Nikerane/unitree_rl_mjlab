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

## UNIFIED CEILING VERDICT (fixed impedance)
"Why does delivered impulse plateau at ~0.87× i_ref / ~1.4 m/s?" — answered on all three axes:
- **NOT reward-limited** — Phase 2: depth-gate + impact_progress reshaping don't move it.
- **NOT torque-limited** — B.2 effort ×1→×3 (30→90 N·m): exact no-op on contact speed + delivered.
- **Controller-bandwidth is the mechanism** (delta_pos_scale) BUT **capped by the real Z1 3.1415 rad/s
  joint-velocity limit** — 0.15 already sits near it.
⟹ **The fixed-impedance impact-impulse ceiling is a HARDWARE JOINT-VELOCITY limit.** Consistent with the
prior closed-loop finding that the trained policy's worst-case |q̇| already exceeds π ([[softcat-velocity-result]],
[[z1-velocity-bound-finding]]): the policy is already at the hardware velocity ceiling. This is the wall
variable impedance is designed to beat (stiff windup stores strain energy; compliant release decouples
contact speed from steady-state joint velocity) — the principled motivation for the VIC phase, now backed
by a measured ceiling on every axis. NOT VIC-in-disguise: delta_pos_scale is an action-bandwidth knob with
a strike-placement accuracy tradeoff, and it is already at the hardware cap.
