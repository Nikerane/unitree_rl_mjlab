> ⚠️ **ARCHIVED 2026-07-05** — superseded by nothing — executed 2026-06-17 design record.
> Facts below may contradict the current code. Do not act on them; check `docs/README.md`.

# Z1 Hammer — "Make it actually strike, kill pressing" redesign (design + decision record)

**Date:** 2026-06-17 · **Branch:** `hammer-z1` · **Status:** DESIGN DECIDED — **start single-strike** (Option S); multi-strike (Option M) deferred but documented (§6). Diagnostics done. **Scope: Z1 Phase-0 only** (G1 explicitly out of scope per user, 2026-06-17).

## TL;DR / decision so far

V1 (the A-BASE/A-TRACK campaign, arrays `36472565`/`36472566`) showed the task is solved by a **gentle ~0.45 m/s single contact**. A multi-lens critique + four diagnostics established that:

1. **The current "strike" is really a position-servo *push*, not a hammer blow** (energy: head KE ≈ 0.05 J vs ≈ 0.81 J of work to drive the nail → the motor does ~94%).
2. **Do NOT build the planned T3–T5 reward machinery on the current task** — no observed failure (violates augment-not-replace), the windowed force-impulse term is *press-friendly*, and T4/T5 have no measurand (joint loads are tiny, the constraint never binds).
3. **The fix is the TASK, not new reward terms.** The hammer hits gently only because the action-step size (`delta_pos_scale=0.05`) is small — *not* a hardware limit. The real Z1 can honestly strike at ~1.3–1.8 m/s.
4. **Two parameters control honest strike speed:** `delta_pos_scale` (the speed command) **and** `max_dq` (the velocity rail, currently ~80× too loose so the sim has *no* realistic velocity limit).

**Direction:** fix the task so the *existing* reward elicits a genuine strike — add the `max_dq` velocity rail, raise `delta_pos_scale` to target ~1.5 m/s, fix a raw-nail-position clamp bug. **Start single-strike** (user, 2026-06-17); keep friction modest and *verify* the press stays excluded. Multi-strike (raise friction to physically kill the press) is deferred but its trigger conditions and design are recorded (§6). No new reward terms.

> **CORRECTION (during implementation, 2026-06-17) — supersedes the `max_dq = 3.1415·dt` recipe in TL;DR pt 4 and §3/§5 below.** Probe calibration (`scripts/diag_strike_probe.py --mode max_vel`, peak-joint-speed readout) found two things:
> 1. **`delta_pos_scale=0.15` is self-limiting.** The hardest straight-down command peaks at **2.41 rad/s** joint speed — under the real 3.1415 rad/s limit — so a velocity rail is **redundant at this scale** (a rail only matters if the scale is pushed toward ~2.5 m/s). Caveat: 2.41 is the open-loop straight-down max; confirm the trained policy's peak qvel in the rollout (wind-up / reversals could differ).
> 2. **`max_dq` is NOT a clean velocity limit and is NOT `3.1415·dt`.** It clamps the per-substep IK *increment*; the position-PD then settles at `qvel ≈ (kp/kd)·max_dq`, and for the Z1 `kp/kd = 10` (stiffness 1000 / damping 100). So the `0.0063` value caps joint speed at ~0.06 rad/s and **craters the arm** (measured 0.055 rad/s crawl). Empirical fit across `max_dq ∈ {0.0063, 0.314, 0.5}` → qvel ≈ {0.055, 3.32, 5.20} ≈ `10·max_dq`. A real rail, if ever needed, is `max_dq ≈ 0.29–0.31` (0.314 → 3.32 rad/s peak).
>
> The `max_dq` line was **reverted** in `env_cfgs.py` (replaced with a comment); `delta_pos_scale=0.15` stays. Whether to add an insurance rail vs rely on self-limiting + rollout monitoring is an **open decision** (not yet trained). Durable summary in the `diffik-maxdq-velocity` and `z1-hardware-limits` auto-memories.

> **TRAINING RESULT (2026-06-17 — `b_strike`, run `36517248`, 3 seeds × 500 iters, all COMPLETED). The strike redesign worked; Option A (self-limiting, no rail) is FALSIFIED.** All seeds: 100% success, **single ballistic contact** (~1 contact/ep vs V1's ~3-step drive), **~1.2 m/s impact** (2.7× V1's 0.45), ~4 control steps/ep (vs ~8.6), press still excluded (press-basin → LIFT-AND-STRIKE). **But** the trained policy's peak arm joint speed amplifies to **4.3–4.65 rad/s worst-case (≈1.9× the 2.41 open-loop ceiling)** on every seed — seed 1's *mean* (3.75) already exceeds the real 3.1415 rad/s limit. The rollout-caveat in pt 1 above was correct: closed-loop wind-up / off-axis Jacobian breaks the "self-limiting" assumption. **Open decision before the next retrain:** (1) lower `delta_pos_scale`→~0.10; (2) keep 0.15 + add a joint-velocity-excess penalty (thesis-aligned, augment-on-observed-failure); (3) verified `max_dq≈0.29` rail; (4) accept as the Phase-0 diagnostic finding ("the binding limit for a hard position-only strike is joint velocity"). Full results in `docs/VEGA_TRAINING_PLAN.md` → "b_strike" section.

---

## 1. Background — why we're here

- **Thesis context:** Z1 hammer is the Phase-0 fixed-base diagnostic. (G1 deliberately ignored in this analysis.)
- **Reframed goal (user, 2026-06-17):** make the arm *genuinely strike* the nail, and *eliminate pressing entirely* — not just "exclude the press in reward."
- **The planned ladder was** T0✅ T1✅ T2✅ (r_imit) → **T3** windowed axial-impulse impact reward → **T4** per-joint impulse measure/penalize → **T5** constraint escalation (Lagrangian). This record pivots away from T3–T5 on the Z1.

## 2. Multi-lens critique outcome (workflow `wf_2a07574c-52a`, 5 lenses + synthesis + adversarial)

Unanimous, high-confidence: **don't build T3–T5 on the current task.**
- **Augment-not-replace violated:** V1 observed *no* failure mode (100% success, single clean strike, press never emerges, r_imit no effect), so adding "hit harder" (T3) and "limit joint stress" (T4/T5) terms has no trigger.
- **T3 is press-friendly here:** on a position-servo driving a friction-resisted nail, a windowed *force*-impulse `I=∫F·dt` measures *how hard/long the servo pushed* — exactly what a press maximizes — and replacing `impact_progress` drops the **velocity gate**, the one thing that disfavours the press (a press has v≈0). Keep `impact_progress` (momentum proxy); the design corpus's own SQ4 already says the velocity/momentum proxy suffices on the Z1.
- **T4/T5 have no measurand:** gentle strike + gravity-compensated arm → per-joint impulses far below ratings; the excess penalty is ~0 and the Lagrangian multiplier stays ~0 (flat Pareto). Can't validate a safety constraint that never binds.
- **Adversarial correction (kept):** there is **no physical "press-stall invariant"** — at 30 N friction a scripted press *slow-succeeds* (~90 steps); the press is excluded only by reward + time cost, so "harder reset is safe because friction is untouched" is **false**. Hardening must re-verify press exclusion, not assume it.

## 3. Diagnostics & verified facts

Scripts (new, uncommitted): `scripts/diag_policy_trace.py` (kinematic strike/press/slam classifier), `scripts/diag_strike_probe.py` (`--mode max_vel` ceiling probe, `--mode press_basin` lift-vs-press eval, `--delta-scale`/`--max-dq`/`--no-term` overrides). Run on the 6 V1 checkpoints (CPU).

### ① Press-basin eval — *the policy does NOT press*
Started in the press basin (hammer in contact, v≈0, nail at ~2.4 mm), handed to the trained policy: it drives the nail in and then **retracts ~7 cm and re-strikes** (a learned swing cycle), identical for A-BASE and A-TRACK. With terminations on it just drives through to success (the "+45 mm lift" in the raw verdict was a post-termination reset teleport, not a real lift — corrected). **Conclusion: pressing is not a behaviour the trained policy adopts; the issue is the swing is *gentle*, not that it presses.**

### ② Max-velocity probe + `delta_pos_scale` sweep — *speed is a tunable config artifact, not a hardware limit*
Open-loop max-down command:
- At `delta_pos_scale=0.05`, head reaches a **steady ~0.45 m/s regardless of runway** (13 cm vs 24 cm → same 0.45). It does *not* accelerate over distance — it's a terminal tracking velocity.
- Sweep (linear): `delta_pos_scale` 0.05→**0.45**, 0.10→**0.90**, 0.20→**1.81**, 0.40→**3.55** m/s. So `v ≈ 0.18 × delta_pos_scale / dt` (the IK closes ~18% of the commanded step per control cycle).

### Mechanism (from `differential_ik.py`)
`target = current_pos + action × delta_pos_scale` (`:180`); each substep a DLS-IK solve (`:248`) computes a joint step (clamped to `max_dq=0.5 rad`, `:249`) and a stiff PD actuator (kp 1000/1500, kd 100/150) chases it. Net = a proportional tracker → terminal head speed **linear in `delta_pos_scale`**.

### Real-vs-sim velocity limit (the load-bearing check)
- **Real Z1 (official `z1_description` URDF):** every joint velocity limit = **3.1415 rad/s (π = 180°/s)**; torque 60 N·m (joint2), 30 N·m (others).
- **Sim:** reproduces torque + position limits *exactly*, but **enforces no velocity limit**. The only stand-in is the IK clamp `max_dq = 0.5 rad/substep ≈ 250 rad/s` — **~80× the real limit**, i.e. effectively absent. (Confirmed: speed scaled linearly to 3.5 m/s with no saturation → pure kinematic-tracking regime; torque isn't even binding.)
- **Honest speed ceiling (Jacobian at the strike/NEAR_NAIL posture):** joint2 (−0.496 m/rad) + joint3 (−0.380) dominate vertical motion. Theoretical max downward head speed (all joints at 3.1415 rad/s) = **3.44 m/s**; realistic min-norm solutions need max joint speed 1.5–2.2 rad/s for 1.3 m/s and 2.1–3.1 rad/s for 1.8 m/s. **The current 0.45 m/s uses only ~20% of capacity** (max joint ~0.5–0.8 rad/s).

| head speed | max joint speed | honest? | head KE | vs 0.81 J to drive nail |
|---|---|---|---|---|
| 0.45 (current) | 0.5–0.8 rad/s | uses ~20% | 0.05 J | 6% (must push) |
| 0.90 | 1.1–1.5 rad/s | ✅ | 0.20 J | 25% |
| 1.30 | 1.5–2.2 rad/s | ✅ | 0.42 J | 52% |
| 1.81 | 2.1–3.1 rad/s | ✅ edge | 0.81 J | **100% (ballistic)** |

**→ A genuine ballistic strike (~1.5–1.8 m/s) is hardware-honest and ~3–4× the current speed.**

## 4. Root cause

On a position-servo action space driving a friction-resisted nail, "drive the nail" and "press the nail" are the **same mechanism** (the motor supplies the force). The policy converges to the minimum-effort version — a slow ~0.45 m/s drive-through — because (a) it works, (b) nothing rewards real impact, and (c) `delta_pos_scale=0.05` caps the head speed at ~0.45 m/s so a real momentum-driven blow is impossible. The press isn't *physically* excluded; the trained policy just happens to swing rather than hold.

## 5. Design direction (the fix — task, not reward)

1. **Add the velocity rail:** set `max_dq = 3.1415 × physics_dt = 0.0063 rad/substep` (enforce the real joint-velocity limit, currently missing). Makes any trained speed hardware-faithful.
2. **Raise `delta_pos_scale`** (0.05 → ~0.15) so the policy can command a ~1.3–1.8 m/s strike (under the rail). Verify the policy can still aim at the coarser action scale (retrain).
3. **No-regret clamp fix (do first):** reward terms read *raw* nail `joint_pos`, which transiently overshoots the 0.032 m soft limit to ~63 mm (`completion`, `NailDepthDeltaTerm`, `nail_driven`, `ImpactProgressTerm` at `rewards.py:33/80/125/191`; only the *observation* is clamped, `observations.py:119-129`). Harmless in training today (episode ends at 27 mm) but would corrupt any impulse integral. Clamp at the reward source `clamp(0, NAIL_GOAL_DEPTH)` + add a validate_rewards phase asserting no reward fires above 0.032.
4. **Keep the existing reward**, recalibrate `impact_progress`'s `v_expected` (currently 1.0) to the new achievable speed. **No windowed force-impulse term** (it's press-friendly; velocity gate is what excludes the press).

### Shelved (documented, not deleted)
- **T3 windowed axial-impulse**, **T4 per-joint impulse penalty**, **T5 Lagrangian** — no measurand / wrong problem on the Z1 position-only task. Keep `impact_progress` as the primary impact term (momentum, velocity-gated). Revisit only if a *harder* task produces an observed failure of the velocity reward.

## 6. DECISION: start single-strike (Option S); multi-strike (Option M) deferred — findings recorded

**Decision (user, 2026-06-17): start single-strike.** Keep nail friction modest (do not raise it to block the press by physics yet); rely on `delta_pos_scale`↑ + the existing velocity-gated `impact_progress` for a genuine fast strike, and **verify** (not assume) the press stays excluded via the press-basin / scripted-press probes. This is the cheapest path and keeps the current single-strike reference and reward intact.

### Multi-strike (Option M) — when it becomes necessary, and how (deferred, but the homework is done)

Captured now so the transition is cheap when the time comes:

- **Trigger condition (quantitative):** multi-strike becomes *mandatory* when the work to finish the nail exceeds one achievable impact's kinetic energy. Work ≈ `frictionloss × depth`; one honest impact KE ≈ ½·m·v² with m≈0.5 kg and v ≤ ~1.8 m/s → **≈ 0.8 J ceiling per strike**. So a single strike suffices only while `frictionloss × depth ≲ 0.8 J` (≈30 N × 27 mm = 0.81 J today — right at the edge). Raise friction, raise required depth, or use a heavier/stiffer nail past that line and one strike can no longer finish → multi-strike.
- **Pressing-dead and multi-strike are the SAME regime:** the only way to make the press *physically* fail is friction above the arm's sustainable quasi-static push (~50–80 N). But that same friction makes one 0.8 J strike insufficient (needs >2 J of work) → ~3+ strikes. **So "pressing physically impossible" ⇒ multi-strike.** If a future requirement is "press can never succeed," that decision automatically selects Option M.
- **Good news — the behaviour already emerges:** the `--no-term` press-basin/rollout probes show the trained policy *already* strikes → **retracts ~7 cm → re-strikes** with no rhythm reward. This resolves the old **Q2** (does retract-and-restrike emerge naturally? → **yes, observed**) and matches the literature (Liu 2025, Karbasi 2024, Robot Drummer 2025: repetitive striking emerges from the task, not an explicit rhythm term). So Option M likely needs *no new reward terms* — just a higher success bar (deeper/harder nail) and the reference/latch changes below.
- **Scope when we switch to M:** make the reference cyclic (`SingleStrikeReference` → re-arming phase; the monotone latch at `references.py:150` must reset per strike) and re-arm the r_imit ante-impact latch (`rewards.py:255-261`, currently permanent — but r_imit is annealed off and had no V1 effect, so low risk). Re-validate that the press fails at the new friction (`test_single_strike.py` + scripted press). Literature precedent for the regime: **ARMADA** (15 N dry friction, ~10 strikes for 20 mm) and **Adroit hammer** (15 N, full nail length, staged insertion bonuses).
- **Cheapest experiment to confirm M is viable when wanted:** raise `frictionloss` above the push force + re-pin threshold, retrain, and check the policy converges to repeated strikes (not a stall) and the scripted press stalls below threshold.

## 7. Next steps / verification plan

Small sweep (after the clamp fix), each retrain ~12 min on one A100, verified with the probes:
- Set `max_dq=0.0063`; sweep `delta_pos_scale ∈ {0.10, 0.15}` (± a friction setting if Option M).
- Per config confirm: (a) trained impact speed rises toward target, (b) a scripted slow push *fails* to drive the nail if friction raised (press dead), (c) the policy succeeds via a fast strike, (d) max joint speed stays ≤ 3.1415 rad/s (the rail holds), (e) success threshold re-pinned and `test_single_strike` / press re-checked if friction changed.
- Full local gate (`validate_rewards.py` incl. the new clamp phase, `verify_contact_sensor.py`, `pytest`) before any GPU submit.

## 8. Artifacts & pointers

- V1 results & numbers: `docs/VEGA_TRAINING_PLAN.md` §"V1 — Results"; Path-A resolution: `docs/research/reward-design/OPEN_QUESTIONS.md` Q1.
- New diag scripts: `scripts/diag_policy_trace.py`, `scripts/diag_strike_probe.py` (uncommitted).
- Superseded plan: `docs/research/reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (T3–T5 shelved per §5).
- Controller mechanics: `mjlab .../mdp/actions/differential_ik.py`; Z1 actuator/limits: `src/assets/robots/unitree_z1/z1_constants.py`; real limits: Unitree `z1_description` URDF.
