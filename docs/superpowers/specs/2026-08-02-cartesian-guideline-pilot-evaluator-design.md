# Cartesian guideline pilot evaluator design

**Status:** owner-approved for implementation on 2026-08-02. Training remains a
separate approval gate.

## Question

Can the ordered-gate reward make a learned fixed-impedance hammer strike follow
the intended direct path without destroying task success, impact speed, impulse
performance, or hardware-qvel legality?

The excluded pilot contains exactly four final checkpoints:

- `C0`, training seeds 0 and 1: guideline observations, no `r_gate` key.
- `C-Gate`, training seeds 0 and 1: identical configuration plus
  `ordered_gate_progress_reward` at weight 8.

Both arms use the same fixed physical reset `(0.0, 0.0)`, direct no-wind-up
reference, six ordered gates, fixed impedance, unchanged manufacturer impulse
caps, and `imp_max_p=0.0`.

## Minimal architecture

Extend the existing sampled path in `scripts/eval_impulse.py`; do not create a
second evaluator or launcher. Reuse its immutable checkpoint loading, isolated
RNG streams, exactly two completed episodes per 256 environments, 500 Hz head,
contact and qvel traces, terminal Lambda/success snapshots, reset digests, and
code/asset/checkpoint provenance.

Add only the guideline state that cannot be reconstructed safely later:

- frozen entry and nail positions;
- the 500 Hz ordered `next_gate` state;
- the 500 Hz perpendicular error produced by the production geometry;
- the 500 Hz `disarmed` state that stops gate credit at accepted contact;
- the actual weighted manager `r_gate` payout at control rate, represented as
  zeros with `gate_reward_present=false` for C0.

These fields are additive and hash-bound only for guideline traces. Legacy
trace schemas and treatment contracts remain unchanged.

## Episode-first endpoint

The primary straightness value is computed once per episode:

1. Start at the first 500 Hz sample where `next_gate` advances from 0.
2. End inclusively at accepted contact; if contact never occurs, use episode end.
3. If gate 1 was never reached, or accepted contact occurred first, assign the
   fixed failure value `0.050 m`.
4. Otherwise compute q90 of `min(perpendicular_error_m, 0.050)` in that window.
5. A seed's endpoint is q90 across all 512 episode values. Never pool substeps
   and never condition on success.

Secondary descriptive outputs are: success rate, useful precontact speed,
all-six-gates rate, fraction of the scored window inside the 5 mm corridor,
backward-progress count, actual gate return, Lambda/cap ratios, and finite-qvel
rail qualification. Raw trajectories remain in the existing content-addressed
NPZ; no new video or plot format is part of this implementation.

## Fail-closed pilot contract

- Guideline tasks are accepted only with `--campaign cartesian-guideline-pilot`.
- Rows must be exactly `(C0,0)`, `(C0,1)`, `(C-Gate,0)`, `(C-Gate,1)` and final
  `model_499.pt` checkpoints; no replacement or performance-selected seed.
- All four use the same evaluator RNG tuple, fixed reset/geometry identity,
  clean code/assets, and exactly 512 sampled episodes.
- Every row requires `impossible_success_n=0`, `lambda_dead_n=0`, and no
  nonfinite qvel. Finite qvel exceedance is retained and removes a hardware-safe
  claim; it never removes a seed.
- C0 requires literal `r_gate` absence and exactly zero stored gate payout.
- Both C-Gate seeds must contain finite positive total actual gate payout.
- Each arm continues past the excluded pilot only if at least one of its two
  seeds has sampled success rate at least 0.25.
- The pilot makes no C-Gate-vs-C0 scientific claim and cannot retune gate radius
  or weight. Confirmatory comparison still requires at least five seeds per arm.

## Files and non-goals

Modify only:

- `scripts/eval_impulse.py`
- `tests/test_eval_impulse_hook.py`
- `evaluation/analysis/guideline_campaign.py` (new)
- `tests/test_guideline_campaign.py` (new)

Do not modify task dynamics, rewards, tracker logic, task registration, training
launcher, smoke script, action space, gains, caps, or enforcement. Do not add a
manifest builder, database, dashboard, renderer, or confirmatory statistics.

