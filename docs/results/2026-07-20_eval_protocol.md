# Comparison-experiment eval protocol (2026-07-20)

The eval INSTRUMENT (`eval_impulse.py`) is sound — pre-reset buffer reads, self-certifying invariants
(`impossible_success_n`/`lambda_dead_n`), ≥512 episodes/policy, full provenance (now with `-dirty` guard).
What we lacked is a rigorous **comparison** protocol. This pins it, so B.1/B.2 and every future A/B is
citable rather than eyeballed. Motivated directly by af1: at n=3 the seed spread was 0.56–1.25× i_ref —
means are meaningless against that variance.

## Rules for any arm-vs-arm comparison

1. **≥5 seeds per arm** (was 3). Cheap (~19 min/run, they parallelize on Vega). More for a subtle effect.
2. **Use the SAMPLED columns, not the deterministic ones.** Play cfg has zero reset noise + mean action ⇒
   the deterministic columns are n=1-effective (all 256 envs identical, `depth_std≈0`). Only the
   `_sampled` columns carry a distribution. Every distributional claim reads `*_sampled`.
3. **Pre-register the primary metric + decision rule** per experiment BEFORE running (write it in the
   experiment's plan section). One primary metric; others are secondary/diagnostic.
4. **Report seed-level mean ± std + min/max**, and a **Mann–Whitney U across seeds** (n₁=n₂=5, two-sided)
   for the headline claim. Treat p<0.05 AND a non-overlapping practical threshold as "real"; anything else
   is "not distinguishable at this n" (state it, don't over-claim). MWU (rank-based) not t-test — seed
   distributions are small and non-normal.
5. **One held-out robustness eval** with training-matched reset-pose noise (±0.05 rad) so we test pose
   generalization, not one fixed pose. Report success + delivered under noise alongside the nominal.
6. **Instrument gate:** any row with `impossible_success_n>0` or `lambda_dead_n>0` FAILS the whole arm —
   do not analyze a run with a tripped sentinel.
7. **Provenance:** every row must carry a clean `git_hash` (no `-dirty`). A `-dirty` row is quarantined
   until the tree is committed and the run repeated.

## Aggregation
Use `scripts/compare_runs.py` (per-arm mean±std bands across seeds, strips `_seedN`) instead of hand-reading
CSVs; add the MWU as a small post-step over the banked `summary.csv` `*_sampled` columns.

## What this does NOT fix (known, accepted)
- CUDA run-to-run non-determinism (atomic-reduction ordering) — a single eval per checkpoint; we do not
  average over eval repeats. Effect is small vs the seed spread.
- Cross-policy causal claims still rest on the effect being larger than seed variance — which is exactly
  why n≥5 + MWU + a pre-registered practical threshold.

## Per-experiment template (fill before running)
```
Experiment: <id>
Hypothesis: <one line>
Primary metric: <e.g. sampled delivered_mean / i_ref>
Decision rule: <arm B > arm A by ≥X, MWU p<0.05 over 5 seeds> ⇒ <conclusion>; else <null conclusion>
Secondary: <contact-window len, peak/mean force, success, worst Λ/cap sampled>
Arms: <A (control) vs B (treatment)>, 5 seeds each, imp_max_p=0, 500 iters, 4096 envs
Robustness: held-out eval with ±0.05 rad reset noise
```
