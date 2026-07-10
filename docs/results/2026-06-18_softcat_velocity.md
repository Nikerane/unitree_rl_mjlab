# 2026-06-18 — `cat_soft` velocity: first GPU result for the faithful soft `γ(1−δ)` CaT

> **Provenance:** record reconstructed 2026-07-05 (docs consolidation) from the session notes of the
> 2026-06-18 runs; numbers are as logged then. Training job `36557770`, eval job `36558685`.

## 1. Context

First GPU test of the faithful soft `γ(1−δ)` CaT mechanism (`CatPPO` + `CatSoftHook`) on the Z1
joint-velocity bound. Question: does soft-CaT actually reduce peak |q̇| toward the 3.1415 rad/s
hardware limit *without* destroying the strike — i.e., does it beat the A1–A4 ablation arms
([2026-06-17_velocity_bound_ablation.md](2026-06-17_velocity_bound_ablation.md)) and the naive sampled-hard CaT it replaces,
while staying episode-preserving?

## 2. Method

- Task `Unitree-Z1-Hammer-CaT-Soft`, 3 seeds × 500 iters / 4096 envs (500 iters to match the
  comparison matrix `a_base` / `c_a3_cat` / `c_hardterm`, which all ran 500; learning plateaus by ~250).
- Peak-|q̇| eval: `scripts/eval_peak_qv.sh` → `scripts/diag_policy_trace.py`, all policies rolled out
  in the same neutral base env (eval job `36558685`).

## 3. Results

**Training:** reward ≈ 2.78; δ engaged early (≈ 0.035) then relaxed to ≈ 0.0005;
`joint_pos_limits` trip-wire flat at 0.0000 across all seeds → no penalty-evasion
(the Decision-1 scale-positives guard holds empirically).

**Peak-|q̇| eval** (mean peak per strike, control-rate):

| Arm | mean peak \|q̇\| (rad/s) | note |
|---|---|---|
| soft-CaT | **2.48–2.65** (±0.25) | under the 3.1415 limit |
| `a_base` (unconstrained) | 3.54 (±0.60) | the average strike exceeds the limit |
| `c_a3_cat` (naive sampled-hard) | 2.78 | |
| `c_hardterm` | 2.48 | episode-truncating |

- soft-CaT keeps the **highest, tightest impact speed** (1.21 m/s, *above* `a_base`'s 0.88) at
  **100% success** — it did NOT comply by hitting softly; it learned a more controlled strike.
  On par with hard-termination's mean reduction but **episode-preserving**, and better than the
  naive sampled-hard CaT it replaces.
- **BUT every arm's global-max |q̇| still exceeds the limit** (soft-CaT 3.9–4.2, even hardterm 4.1).
  A fixed-impedance, position-only policy cannot give a hard per-joint guarantee — the residual
  worst-case is chain-coupled (same finding as the A1–A4 ablation). δ was not masking violations:
  the eval is a direct measurement, so the mean compliance is genuine.
- Caveat: these are control-rate |q̇| numbers; the true substep peak is higher — one more reason the
  impulse constraint must accumulate at substep rate.

## 4. Conclusion

Soft-CaT is validated as a *mechanism*: a real per-step incentive of the correct constraint type,
compliant in the mean, strike-preserving, episode-preserving. The residual worst-case overshoot is
the strike-vs-safety tension that motivates the next two thesis steps: **variable impedance** and the
substep **impulse** constraint (`../research/reward-design/IMPULSE_CAT_IMPL_PLAN.md`).

Design + decisions behind the mechanism: `../research/reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md`.
