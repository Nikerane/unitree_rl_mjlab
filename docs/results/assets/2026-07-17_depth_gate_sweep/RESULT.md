# Depth-gate sweep (dg) — result (2026-07-17)

**Verdict: removing the `delivered_impulse` depth-gate is a NO-OP for delivered impulse on the
terminating single-strike task.** Phase-2 Change-1 tested; confirms the Phase-0 physics story.

Sweep: `Unitree-Z1-Hammer-CaT-Impulse`, 500 iters, 4096 envs, `imp_max_p=0`, 3 seeds/arm.
`dg1` = gate-ON (`depth_gate=True`, shipped control); `dg0` = gate-OFF (`depth_gate=False`, treatment).
JIDs train 39626161/39626162, eval 39627853. Summary: `dg_eval_summary.csv`.

| arm (n=3) | delivered/i_ref det. | delivered/i_ref sampled | worst Λ/cap sampled | worst Λ/cap det. | success | ep_len | invariants |
|---|---|---|---|---|---|---|---|
| dg1 gate-ON | 0.778 (0.58–1.10) | 0.819 | 0.533 | 0.174 | 1.00 | 7.3 | clean |
| dg0 gate-OFF | 0.713 (0.50–1.08) | 0.698 | 0.661 | 0.157 | 1.00 | 7.3 | clean |

**Read:** delivered impulse is statistically indistinguishable between arms (gate-off if anything
slightly lower); success 1.00, depth 32 mm, single strike (ep_len 7.3) both ways; no press-farming
(`impossible_success=0`, `lambda_dead=0` all rows). The depth-gate removal had no effect.

**Why (confirms Phase 0):** in the terminating config the episode ends the instant the nail seats
(0.030 m), so there is no post-seating phase during which the ungated reward could earn extra — the
"seated-nail impulse" the gate-removal pays for does not exist here (reward-impl dig predicted this:
gate-removal surface = "essentially the single terminal strike"). Underneath, the effort-clamped
~1.4 m/s contact-speed ceiling caps delivered impulse regardless of the reward (Phase-0 A1/A5).

**Implication:** the depth-gate is NOT the maximization lever. The lever that targets the actual
ceiling variable is **ante-impact velocity** — the existing `impact_progress` term. Tested next (2b).

---

## Phase 2b — `impact_progress` weight sweep (ante-impact velocity)

Sweep: `impact_progress.weight` 8 (control = dg1) / 24 (ip24) / 48 (ip48), 3 seeds, same task/imp_max_p=0.
JIDs 39631659/39631660. **ip48 destabilized training: 3/3 diverged ~iter 50** (only `model_50` saved,
exit 120 on teardown) — the launcher failure-propagation fix (Codex P1) correctly surfaced these as
FAILED. ip24: 2/3 converged (seed0 diverged early). Read from training TensorBoard `delivered_total`
(the eval sbatch flaked on the ip24 single-glob campaign — exit 53, unrelated bug to fix later; data
taken directly from TB instead):

| arm | impact_progress.w | delivered_total / i_ref (converged seeds) | imp_peak_joint2 |
|---|---|---|---|
| dg1 (control) | 8 | 0.87 (0.84/0.89/0.87) | ~0.17 |
| ip24 | 24 | **0.86** (0.86/0.86) — FLAT | 0.21–0.27 |
| ip48 | 48 | training DIVERGED (3/3) | — |

**Tripling the ante-impact velocity reward did not raise delivered impulse (0.86 vs 0.87× i_ref);
cranking it to 48× broke training instead.**

---

## Unified conclusion — fixed-impedance maximization is closed
Across **both** reward levers — depth-gate removal (2a) and ante-impact velocity weight (2b) — **reward
reshaping cannot push delivered impulse past ~0.87× i_ref**, the effort-clamped ~1.4 m/s contact-speed
ceiling (Phase 0 A1/A5). The policy already strikes at that ceiling; more reward pressure either does
nothing (dg, ip24) or destabilizes training (ip48). This is the empirical close of the Phase-0
prediction: on fixed impedance the impulse is physics-limited, not reward-limited. **The only remaining
lever to raise impact impulse is variable impedance (VIC)** — deferred. Keep `depth_gate=True` and
`impact_progress.weight=8` (shipped). The excess-over-i_ref bonus (Change 2) stays code-unwritten +
enforcement-gated. TODO: fix the eval sbatch's ip-campaign exit-53 failure. Plan:
`docs/results/2026-07-17_fixed_impedance_execution_plan.md`.
