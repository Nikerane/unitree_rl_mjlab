# Results & analysis log

Empirical analyses of training runs and diagnostic experiments — **one Markdown file per campaign/run**. This is the *canonical* home for "what a run showed." The plan docs (`../VEGA_TRAINING_PLAN.md`) stay focused on *what to run* and link here for findings.

**File convention:** `YYYY-MM-DD_<run-name>.md`. Suggested structure for each:

1. **Context** — what question the run answers; what changed since the last run.
2. **What changed** — exact config/code deltas (with commit hashes).
3. **Method** — exact scripts + commands (so it reproduces).
4. **Results** — tables; per-seed where relevant.
5. **Conclusion** — the two-sided read (what worked, what broke).
6. **Decision / next** — options + recommendation.
7. **Reproduce / artefacts** — Vega checkpoint paths, local pulls, TB scalars.

## Index

| Date | Run | Headline | File |
|---|---|---|---|
| 2026-06-17 | `b_strike` (`36517248`) | Honest-scale single strike (delta=0.15): real ~1.2 m/s single ballistic strike, 100% success, press excluded — **but** the trained policy exceeds the 3.1415 rad/s joint-velocity limit (worst 4.3–4.65). | [2026-06-17_b_strike.md](2026-06-17_b_strike.md) |
| 2026-06-17 | velocity-bound ablation A1–A4 (`36526179`/`270`/`271`/`318`) | 4 ways to bound joint velocity (delta→0.10 / penalty / CaT / DcMotor envelope). All keep 100% + the strike; **none gets worst-case under π** — the residual overshoot is chain-coupled. CaT best impact-preserving; DcMotor ≈ baseline (confirms chain-coupling). | [2026-06-17_velocity_bound_ablation.md](2026-06-17_velocity_bound_ablation.md) |
| 2026-06-17 | `a_base` / `a_track` V1 (`36472565`/`36472566`) | Press exploit dissolves under training; both arms reach 100% and converge to the same fast strike-drive; tracking prior has no measurable effect. | analysis in [`../VEGA_TRAINING_PLAN.md`](../VEGA_TRAINING_PLAN.md) → "V1 — Results" |
