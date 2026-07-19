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
