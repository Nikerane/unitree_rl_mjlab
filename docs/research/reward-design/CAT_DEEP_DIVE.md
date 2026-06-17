# CaT deep dive — knobs, the impulse constraint, and what it changes (2026-06-17)

Multi-lens deep dive (workflow `wf_01a029a0-59f`: mechanism / extensions / CaT-for-impulse / humanoid-G1 / internal-fit → synthesis → adversarial critique) into Constraints-as-Terminations ([[cat-constraints-as-terminations]]), to find what helps us more and how to carry it to the G1 impulse constraint. The critique substantially sharpened (and partly downgraded) the synthesis — both are captured.

## The big finding: our A3 is NOT real CaT

`src/tasks/hammer/mdp/velocity_bound.py:100` returns `torch.rand_like(delta) < delta` — a **sampled-Bernoulli HARD episode cut** (wired `time_out=False`). The CaT paper's actual contribution is the **analytic soft discount**: `reward ← reward·(1−δ)`, `done ← δ` fed into GAE as a per-state `γ·(1−δ)` bootstrap (never a sampled reset). Same *expected* objective, but the soft version is **lower-variance** and is what makes "mid-strike termination is benign" true (an over-budget strike has its *future* return discounted smoothly while the already-delivered impact reward is retained). Our naive version worked for the *easy* velocity case (soft p=0.5) but is the variant the paper improves on — and a hard mid-strike cut on a harder/impulse constraint is the **"refuses to act" trap** (learn to strike softly so the cut never fires).

**Implication:** before crediting any impulse-CaT design, implement (or confirm) the real `γ(1−δ)` rollout. The env-side termination we ship is the crude approximation.

## CaT's right role (critique's reframing)

CaT fits **instantaneous** limits (velocity, contact force — the paper's demos). An **accumulated-window budget** (`Σ|qfrc|·h ≤ Λ_max`) is a **stretch**: encoding it as a per-step ceiling on a monotone accumulator converts "spend-when-you-like budget" into "hard instantaneous ceiling," which for a single impulsive strike = terminate at the impact peak — it inherits mid-strike-termination pathology and buys nothing over an instantaneous-rate constraint. **Budget-native tools are better-matched:** Saute-RL (remaining budget in the observation → policy shapes the whole strike, almost-sure by construction) and PID/dual-ascent Lagrangian on the episodic sum (multiplier auto-scales). 

So: **adopt CaT as the scale-free SHAPER of the ante-impact instantaneous quantity** (velocity; on the G1, ante-impact `m_eff·v_axial` read at contact onset) — which on the position-only Z1 *is* essentially what A3 already does — and do **not** delete the PID-Lagrangian arm (T5.2) on the strength of A2(fixed-λ)-vs-A3: that's a category error (fixed-λ ≠ adaptive dual).

## Knobs A3 left unused (use these)

1. **Real `γ(1−δ)` value-bootstrap** (we sampled Bernoulli) — the low-variance core; "copy exactly." Hazard for us: the `clip(reward·(1−δ), min=0)` assumes mostly-positive reward; our task has penalty terms → keep penalties as separate constraints or shift reward non-negative.
2. **Time-to-death curriculum** (we used fixed p) — ramp expected time-to-death `T` (set `p_max=1/T`), and for a hard term **disable it for the first ~20–30% of training** then ramp **both** `p_max` and the threshold loose→tight. Prevents the documented "all-hard fails completely" collapse.
3. **Substep detection** (we read post-decimation `joint_vel`, aliasing the 500 Hz peak — implicated in the residual overshoot) — peak-hold / accumulate inside the decimation loop (the ContactSensor history+max pattern is the in-repo template).
4. **Per-constraint, contact-active EMA normalizers** (we used one shared scalar, no floor) — on ~29 G1 joints a shared EMA lets the largest-unit joint dominate the `max`-aggregation; compute the EMA over contact-active samples only (a bursty contact-sparse signal diluted by free-flight zeros corrupts `c_max`).
5. **Multi-constraint max-aggregation** (`δ=max_i`, not sum) — compose {velocity-CaT, impulse-CaT, commanded-stiffness-CaT}; tune so the impulse term can *win the max* during contact or it is silently masked.

## CaT-for-impulse (the thesis target) — recipe + caveats

Stateful per-joint `ConstraintTerm`: open a window on `first_contact` (latch from `ImpactProgressTerm`), accumulate `Λ[env,j] += Σ_substeps |qfrc_constraint_j|·h` (h=2 ms) inside the decimation loop, return `c_j = Λ_j − Λ_max,j` every control step on the running total; close/reset at window end (`reset()` must actually clear state — A3's is a no-op). The `air_time` term is the API precedent. **Net-new: CaT-on-substep-accumulated-impulse is unpublished** (prior force-terminations are crude instantaneous aborts). Caveats:
- **Measurement is the gating unknown:** mjlab 1.4.0 may not expose per-substep `qfrc_constraint` on the Entity (it doesn't, per `velocity_bound.py`); fallback = the windowed `Δq̇` velocity-jump proxy (changes what `Λ_max` means).
- **Only benign under the real soft bootstrap** (see above) — under our shipped hard cut it's the refuses-to-act trap.
- **Variable-impedance × CaT-on-impulse is a foundational risk** (unpublished): the policy can game the *measurement* (go compliant at the read instant → lower `qfrc` while delivering the same momentum), and `m_eff` itself depends on commanded K → the constraint surface `c=Λ−Λ_max` is **non-stationary** under the policy's own action, which CaT's stationary-tuned EMA will lag. Validate on a toy variable-K Z1 before the thesis depends on it.
- **No clean almost-sure cap** even at p_max=1.0 (per-step chance-constraint; sim-to-real torque leaked 3.17 vs 3.0 N·m for 0.05 s). The CBF fallback is likely **ill-posed for impulse** (integral through a contact discontinuity, relative-degree issue) — so the practical hard lever loops back to ante-impact velocity/`m_eff` shaping (what A3 does). SDH (arXiv:2602.04599) is the citeable off-policy successor of `γ(1−δ)` and proves the survival-weighted objective under-penalizes the rare-deep tail (= our worst-case).

## De-risked by precedent

A CaT co-author (Leziart, Chane-Sane et al., 2026) ported CaT to the **Unitree H1 humanoid** + a box loco-manipulation task with a **soft, contact-phase-gated hand contact-force** constraint (`c = 1_contact·F`) — the direct, less-physical analogue of our impulse term. G1 recipe: joint-limits + **falling** = hard/separate terminations; velocity/torque/action-rate/impulse + commanded-stiffness = soft tier with curriculum; couple `K_d=2√(M·K_p)`.

## Revised plan / next actions (critique-ordered)

1. **(done here)** Audit the shipped mechanism → it's naive sampled-hard, not `γ(1−δ)`.
2. **Cheap Z1 confirmation, reframed honestly** as a *velocity-CaT detector/pressure test* (NOT an impulse rehearsal): de-confound A3's two axes — **substep vs control-rate detection**, with **curriculum-ramped p_max** — but do **NOT** set `p_max=1.0` AND substep simultaneously (a 500 Hz Bernoulli at p=1 fires on one-substep transients → "strike≈die"). Pre-register: it will *reduce* the worst case but not *provably bound* it < π. Log `Λ_j`/`Δq̇` in parallel.
3. **De-risk the `Λ_j` measurement** independently (does mjlab expose substep `qfrc_constraint`? else commit to the `Δq̇` proxy).
4. **For the thesis budget:** primary instrument = **Saute-RL** (budget-in-obs) and/or **PID-Lagrangian** (keep T5.2); evaluate CaT-on-accumulator as a *third* arm, not the sole method. Use CaT for the ante-impact instantaneous shaping it actually fits, with the **real `γ(1−δ)`** mechanism.
5. **Honest defense scope:** CaT delivers strong-adherence-*in-expectation*, not an almost-sure cap; set `Λ_max` with margin; the hard ceiling (if redeemable) comes from a separate layer.

**Confidence: medium** (critique). The single highest-priority correction is the mechanism audit/fix; the single most useful cheap experiment is the reframed substep-vs-control-rate velocity-CaT run.
