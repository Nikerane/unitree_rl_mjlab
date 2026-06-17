# Velocity-bound ablation — A1–A4 (2026-06-17)

> **DECISION (accepted 2026-06-17):** Accept this as the Phase-0 finding and **adopt CaT (A3)** as the velocity-honesty mechanism. The worst-case overshoot is **chain-coupled** and no single soft/plant/kinematic bound removes it on a position-only fixed-PD action space — which is itself the result that motivates the thesis's variable-impedance + explicit-constraint direction. Carry CaT forward to the G1, where the per-joint *impulse* constraint actually matters.
>
> **FOLLOW-UP (2026-06-17):** Not chasing the Z1 number for its own sake, but a **targeted CaT-knob ablation** is worthwhile to *learn the tool* (it transfers to the G1) and to test the **cause-agnostic** hypothesis — since CaT terminates on the violation regardless of cause, **hard CaT (p_max→1.0) + substep-rate detection** may bound the worst-case where the per-joint plant (A4) could not. Axes: p_max (soft 0.5 → hard 1.0), detection (control-rate → substep), curriculum (on/off). Design to be finalized from the CaT deep-dive (`../research/reward-design/`), then run as a focused ablation. The action-level Jacobian projection remains the known hard-bound fix, better rehearsed on the G1.

**Question:** b_strike gave a real ~1.2 m/s single strike but drove arm joints to 4.3–4.65 rad/s worst-case, over the real Z1 limit of 3.1415. Which way of bounding velocity keeps the strike *and* gets honest? (Decision context: `2026-06-17_b_strike.md`; research: `../research/reward-design/JOINT_VELOCITY_BOUND_RESEARCH.md`.)

**Harness (all arms):** A-BASE 7-term reward, 3 seeds × 500 iters, Vega. Eval: `diag_policy_trace` (64 envs × 80 steps) on the trained `model_499`. A1 evaluated at its trained delta=0.10; A2/A3 on the base task at delta=0.15 (measures the learned policy with no constraint masking its raw velocity); A4 on the DcMotor task (the envelope IS the plant).

## Results (peak joint speed, π = 3.1415 rad/s; lower worst-case = honester)

| Arm | mechanism | success | ep len | impact m/s | peak q̇ mean | **worst** | under π? |
|---|---|---|---|---|---|---|---|
| b_strike | none (delta 0.15) | 100% | 4.0 | 1.20 | 2.7–3.7 | **4.29–4.65** | ✗ |
| **A1** | delta→0.10 (kinematic) | 100% | 5.1 | 0.82 | 2.5–2.6 | **3.33–3.61** | ✗ (closest) |
| **A2** | vel-excess penalty (w=0.5) | 100% | 4.0 | 1.20 | 2.7–3.6 | **4.14–4.49** | ✗ (≈ baseline) |
| **A3** | CaT termination (p_max=0.5) | 100% | 4.4 | 1.12 | 2.6–2.8 | **3.61–4.00** | ✗ |
| **A4** | DcMotor torque-speed envelope | 100% | 4.4 | 0.93 | 3.0 | **4.27–4.36** | ✗ (≈ baseline) |

## Reading

- **All four arms keep 100% success and a single-contact strike.** The strike is robust to every bounding method tried — it does not collapse. The task is not infeasible under a velocity bound; the question is purely how *honest* we can get.
- **None gets worst-case under π.** This is the decisive empirical result: on a position-only fixed-PD action space, no single soft / kinematic / plant bound makes a hard strike fully velocity-honest at the *worst case*. The **mean** is easily bounded (A1/A3/A4 all ≈2.5–3.0), but the worst-case windup **tail** leaks to 3.3–4.4 on every arm.
- **A4 (honest plant) is the punchline.** It barely moved the worst-case (4.27–4.36 vs baseline 4.29–4.65) *and* lowered the impact (0.93 m/s). The torque-speed curve removes the bounded joint's *own motor* contribution, but the overshoot is **chain-coupled** — momentum delivered to a joint through the linkage by other joints' motion, plus contact rebound — which no per-joint motor curve can brake. So the residual is *not* motor-driven; it is kinematic coupling.
- **A2 (fixed penalty, w=0.5) effectively failed** — worst-case ≈ baseline. A strike's squared-excess penalty (~1.4/step) is negligible against `completion=+100`, so the policy eats it. A fixed-λ penalty is hackable; it needs a far larger weight (risking the strike) or a Lagrangian that auto-scales.
- **A3 (CaT) is the best impact-preserving reducer** — keeps 1.12 m/s while pulling the worst-case to 3.6–4.0 and the *mean* solidly under π. The termination threat (losing +100) is a much stronger, scale-free deterrent than the weak penalty. Still soft: control-rate detection + the chain-coupled tail leak through.
  - **CAVEAT (added 2026-06-17, post CaT deep-dive):** A3 used the **naive sampled-Bernoulli HARD-termination** variant (`velocity_bound.py:100`), not the paper's low-variance soft `γ(1−δ)` value-discount. The *core ablation conclusion is robust* to this — no termination (naive or real) physically brakes the spike; both are soft-in-expectation, so real CaT would also leave a worst-case tail (the chain-coupled residual is a physics fact). What may refine with real CaT: A3's exact numbers + training stability, and the bigger confound — **control-rate vs substep detection** (A3 aliased the 500 Hz peak). The planned substep-CaT run de-confounds the detection axis; the soft `γ(1−δ)` rebuild is the foundational fix carried to the G1. See `../research/reward-design/CAT_DEEP_DIVE.md`.
- **A1 (scale-down) has the best worst-case (3.3–3.6)** but the slowest strike (0.82) — and still ~6–15% over π. Pure kinematic scaling is necessary-but-not-sufficient.

## Final ranking

1. **A3 (CaT)** — best honesty-per-impact, scale-free, and the *thesis-transferable* mechanism (same code → G1 impulse constraint). Soft, not hard.
2. **A1 (delta→0.10)** — cheapest, best worst-case, but ~⅓ less impact and still not under π.
3. **A4 (DcMotor)** — physically honest but ineffective against the chain-coupled tail and blunts the typical strike; only worth it for plant realism, not for the bound.
4. **A2 (penalty w=0.5)** — inert at this weight; needs a Lagrangian or a much larger weight.

## Decision / what this means

The chain-coupled residual is the binding fact. To get a **hard** worst-case bound you need an **action-level Jacobian velocity projection** (scale the commanded Cartesian delta so the induced per-joint velocity ≤ π) — the only mechanism that targets the *command* worst-case rather than one joint's motor. That is the v2 candidate (combine with A1 or A3). Two honest paths:

- **Push to a hard bound (v2):** hard-CaT (p_max→1.0) + substep-rate detection, and/or the Jacobian action-projection on top of A3.
- **Accept the Phase-0 finding:** *position-only fixed-PD striking is velocity-limited; the residual overshoot is chain-coupled, which an action-level projection or — the thesis's answer — variable impedance addresses.* This is itself the result that motivates the thesis direction, so over-investing in the last 0.5 rad/s on the Z1 is the rabbit hole the research flagged.

**Recommendation:** adopt **A3 (CaT)** as the velocity-honesty mechanism (transferable, best impact), record the chain-coupled residual as the decisive Phase-0 finding, and only pursue the hard bound (Jacobian projection) if a velocity-certified Z1 baseline is specifically needed — otherwise carry CaT forward to the G1 where the constraint actually matters.

## Artefacts
- Checkpoints (Vega): `logs/rsl_rl/z1_hammer/2026-06-17_*_{c_a1_delta010,c_a2_vpenalty,c_a3_cat,c_a4_dcmotor}_seed{0,1,2}/model_499.pt`.
- Local pulls: `/tmp/ckpt/{c_a1,c_a2,c_a3,c_a4}/seed{0,1,2}_model_499.pt`.
- Runs: A1 `36526179`, A2 `36526270`, A3 `36526271`, A4 `36526318`.
- `diag_policy_trace.py` gained a `--delta-scale` override (eval a policy at its trained scale, e.g. A1 at 0.10).

## Artefacts
- Checkpoints (Vega): `logs/rsl_rl/z1_hammer/2026-06-17_*_{c_a1_delta010,c_a2_vpenalty,c_a3_cat,c_a4_dcmotor}_seed{0,1,2}/model_499.pt`.
- Local pulls: `/tmp/ckpt/{c_a1,c_a2,c_a3}/seed{0,1,2}_model_499.pt` (A4 pending).
- Runs: A1 `36526179`, A2 `36526270`, A3 `36526271`, A4 `36526318`.
- `diag_policy_trace.py` gained a `--delta-scale` override (eval a policy at its trained scale, e.g. A1 at 0.10).
