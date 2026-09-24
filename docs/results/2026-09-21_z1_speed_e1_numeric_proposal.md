# E1 numeric qualification settings

## Material Passport

- Origin: academic-research-suite, experiment-agent plan mode
- Date: 2026-09-21
- Status: proposed experiment settings; not an executed training result or physical certification
- Evidence: [128-episode baseline replay](2026-09-21_z1_speed_baseline_recovery.md)

## Question and scope

Can the existing joint-target/impedance controller learn to achieve different requested impact speeds along the same quintic path, while retaining hits and tracking? E1 supplies the desired speed at reset. E2 subsequently enables policy-selected speed. There is still one controller, not a planner plus tracking policy.

## Numeric proposal

| Setting | Value |
|---|---|
| Experimental task | `Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed` |
| Desired-speed interval | **[1.5, 2.2] m/s**, uniform external goals during E1 |
| Evaluation goals | **1.5, 1.85, 2.2 m/s** |
| Candidate event-cost coefficients | **0.2831225, 0.8493676, 2.8312253** |
| Initialization | Fresh 13-output actor for each candidate; no 12-output checkpoint loading |
| Training seed | 2 for screening; one training seed supports feasibility only |
| Budget | 50 iterations × 1,024 worlds × 24 steps = 1,228,800 transitions per candidate |
| Checkpoints | Every25 iterations and final checkpoint |
| First hardware/runtime check | Separate2-iteration,256-world GPU smoke; discard its weights |
| Screening evaluation | Seed2026092101;128 first episodes per goal |
| Confirmation evaluation | Seed2026192101;128 first episodes per goal |
| Bootstrap | Existing evaluator:2,000 paired resamples, RNG seed equals evaluation seed |
| Time limits | Smoke30minutes; each screening candidate2GPU-hours; no automatic retries |

### Why these values

The baseline achieved2.135–2.314m/s, median2.216m/s. The2.2m/s upper target stays near demonstrated baseline behavior rather than extending into faster unexplored motion. The1.5m/s lower target is an **experimental choice**, approximately32% below the median, to test meaningful slowing rather than only predict natural speed variation. Existing evidence does not prove that lower target reachable. These are desired-speed bounds, **not bounds on actual velocity** and not a hardware-safe interval.

The baseline median positive weighted event-return scale before CatPPO transformation was2.8312253. The proposed costs apply the previously specified0.1/0.3/1.0 multipliers. Other rewards remain unchanged. Do not reward the magnitude of a requested speed or equate speed with delivered impulse.

## Acceptance and failure reporting

- At each goal, hit rate at least90% of the matched baseline; with128/128 baseline hits this requires at least116/128 hits.
- Use only `episode == 0` for every gate, matching the evaluator's paired contrasts. Later autoreset episodes emitted while waiting for other worlds are excluded from gate denominators and reported separately.
- Median absolute measured-minus-requested speed error among valid hits at most **0.07m/s** (10% of interval width); report misses/incomplete events and tails separately.
- Primary paired contrast: high minus low achieved speed has positive95% bootstrap interval lower bound. Other two contrasts are secondary diagnostics, not additional independent discoveries.
- Tracking RMS at most10% worse than a genuinely matched baseline using the identical metric and aggregation. Existing19.6605mm median implies a provisional21.6266mm median threshold only for the existing sequential protocol; do not transfer it blindly to a different reset protocol.
- Preserve impulse CaT p=.2, velocity CaT p=.5, caps, gain factor1.25 and all original reward weights. Report per-joint exceedance, maximum utilization and CaT alongside performance. Baseline J6 exceeded its cap in128/128 episodes; passing speed responsiveness is not passing a hard safety constraint.
- Choose the lowest coefficient meeting all E1 gates, then confirm on the held-out seed. If none qualifies, report all results;50iterations is a screen, not proof the architecture cannot learn. Any extension toward250iterations requires a reviewed decision.

## Corrections before launch

1. The earlier confirmation seed2026092102 overlapped127 of128 seeds under the sequential replay protocol. Use2026192101 instead. Different seeds still do not create different physical initial poses in this fixed-start task.
2. The baseline calibration used128 sequential single-world resets. The experimental evaluator uses128 parallel first episodes and samples external goals during reset, consuming additional RNG. Seed equality alone does not guarantee matched observations/reset randomness across baseline and E1. Before making baseline-relative promotion claims, verify actual reset evidence and use a protocol-matched baseline; do not describe the existing calibration as a matched128-world comparator.
3. The existing evaluator uses its evaluation seed for bootstrap, not the separately proposed2026092103. Use the existing behavior explicitly rather than adding a new flag solely for this pilot.
4. Freeze the exact clean source SHA, remote paths, config/asset manifests and existing-CLI commands after approval. Re-run applicable native pre-training gates and smoke on that exact remote runtime. The684local software tests and baseline replay do not establish the new13D GPU training path.
5. Use native episode end as the common tracking window, matching both current evaluators; report the first-event-finalization time separately. The successful baseline finalizes at episode end, whereas unsuccessful E1 contacts may finalize earlier. Do not relabel the whole-episode metric as pre-contact or first-event-only tracking.

Independent read-only agent review supported the range/costs as exploratory choices and identified the reset protocol, bootstrap seed, first-episode filtering and tracking-window qualifications recorded above. Review did not certify learning success or hardware safety. No production code changes were made for this numeric proposal.

## Launch boundary

Update: [GPU preflight and protocol-matched baseline](2026-09-21_z1_speed_gpu_preflight.md) are complete. Reward/contact checks and198focusedtests passed; a two-iteration smoke saved/reloaded correctly. The matched128-world comparator now establishes21.622820mm as the proposed median tracking gate with verified initial-state/RNG parity. Smoke evaluation missed all6strikes, so only software execution—not learned responsiveness—passed. No50-iteration screening run has launched.

These choices finalize a concrete proposal, not evidence that the interval is safe or reachable. No training submitted while preparing this record. Review the numeric settings and reset-comparison correction before approving the bounded E1 stage; E2 and three-curve training remain later gates.
