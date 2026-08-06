<!-- wayfinder-meta
{
  "schema": 1,
  "id": "wf-z1-executable-experiments",
  "kind": "map",
  "title": "Z1 executable experiment specification",
  "status": "open",
  "labels": ["wayfinder:map"],
  "assignee": null,
  "created_at": "2026-08-06T22:32:04+02:00",
  "closed_at": null
}
-->

# Z1 executable experiment specification

## Destination

Produce a supervisor-defensible, executable Z1 simulation experiment
specification that causally separates direct-joint control, fixed-impedance
command trackability, active impulse CaT, curriculum/domain randomization, and
variable impedance. Freeze treatments, promotion gates, evaluation rules, and
compute budgets so implementation sessions do not invent scientific choices.

## Notes

- Planning is the default. Implementation and training appear only as future task
  tickets when a measurement is genuinely required to resolve a decision.
- Current authority remains code, `docs/README.md`, the current living documents,
  and then dated evidence. Start from the
  [post-meeting handoff](../../results/2026-08-06_PRESENTATION_HANDOFF.md), the
  [Stage-1 implementation plan](../../superpowers/plans/2026-08-06-z1-joint-position-fixed-stage1.md),
  and the [FIC `r_tt` paper audit](../../research/khadiv_fic_rtt_audit.md).
- Preserve existing Cartesian registrations and presentation evidence. Do not
  reinterpret historical Cartesian policies as randomized contemporaneous
  controls.
- Research normally uses throwaway branches, but the owner's workspace boundary
  restricts this effort to the existing worktree. Research tickets therefore
  write isolated notes under `docs/research/wayfinder/` and do not switch branches
  or commit while unrelated work is dirty.
- The current Stage-1 plan treats the no-`r_tt` arm as engineering-only. That text
  must be amended before GPU work because the map now includes a trained matched
  FIC `r_tt` ablation.

## Decisions so far

- [Fix the Z1 simulation experiment ladder](tickets/fix-z1-simulation-experiment-ladder.md) — stage direct-joint FIC, controlled drop, diagnostic active CaT, curriculum/DR, then VIC.
- [Freeze the direct-joint fixed-gain baseline contract](tickets/freeze-direct-joint-fixed-gain-baseline-contract.md) — preserve the qualified six-joint fixed-gain P+V+D4 treatment and its safety/evidence gates.
- [Define the causal command-trackability term](tickets/define-causal-command-trackability-term.md) — use the calibrated one-step nonnegative cost with weight `-1.0`, unscaled by CaT, and matched across FIC/VIC.
- [Establish the paper-aligned FIC RTT ablation](tickets/establish-paper-aligned-fic-rtt-ablation.md) — train fixed-gain joint policies both without and with `r_tt`; the exact campaign depth remains open.
- [Freeze the controlled-drop reference setup](tickets/freeze-controlled-drop-reference-setup.md) — use the centered simulation-only 0.200 kg cylindrical primary drop from rest at `h0`.
- [Use a diagnostic windowed reaction-impulse constraint](tickets/use-diagnostic-windowed-reaction-impulse-constraint.md) — validate CaT with lowered diagnostic per-joint thresholds and defer a rigorous ballistic estimator.
- [Freeze the high-level VIC architecture](tickets/freeze-high-level-vic-architecture.md) — add bounded per-joint stiffness and coupled damping to the same joint targets, guidance, and `r_tt`, with VIC last.

## Not yet specified

- Failure-specific rescue experiments after a Stage-1 or active-CaT gate fails;
  the causal failure is not yet known.
- Optional drop heights or masses beyond the 0.200 kg, `h0` primary run; these
  graduate only if the primary result exposes repeatability or nonlinearity concerns.
- Additional curriculum stages required by an observed learning collapse,
  saturation pattern, or domain-randomization edge failure.
- VIC rescue actions if learned gains pin to bounds, fail nominal reproduction,
  or destabilize contact.
- Whether more than the preregistered checkpoint count is justified by observed
  between-checkpoint uncertainty.

## Out of scope

- Final thesis analysis, writing, presentation, or defense production.
- Implementing the experiments, GPU training, and result collection; this map
  ends at the executable specification.
- Hardware drops, physical sim-to-real validation, G1 replication, or
  manufacturer certification.
- A rigorous ballistic estimator or a split ballistic/press constraint in this
  map; those can become a later destination.
- Trajectory optimization, generate-then-track control, or replacement of P
  task-space guidance.
- New jitter or ad-hoc reward penalties without a measured failure.
- Removing or retroactively changing Cartesian tasks or historical evidence.
