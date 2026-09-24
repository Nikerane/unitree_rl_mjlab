# Impact-speed work: cross-review synthesis

Date: 2026-09-22. Analysis only. No production edits or new simulation/training launches.

## Evidence and agreement

Three independent reviewers examined [mechanism](2026-09-22_speed_mechanism_review.md), [objective](2026-09-22_speed_objective_review.md), and feasibility/measurement. They then challenged the proposed ordering. Primary experimental evidence is the [fresh matched dose pair](../../results/2026-09-22_z1_speed_impulse_dose_pair.md), not an assumption that longer training fixes the remaining problems.

The stronger impulse-CaT setting reduces wrist impulse severity, but not the near-universal exceedance count. It retains only 90.20% of control delivered impulse at the middle requested speed, close to the frozen 90% utility guard. Neither policy passes the existing full E1 gate. The speed selector remains inactive in E1: these results establish imperfect externally requested-speed control, not autonomous speed selection.

Root inspection of both raw `evaluation/trace.jsonl` files grouped by condition/world finds every episode's largest native substep J6 speed at control step zero: 4.609762 rad/s for dose .2, 4.580195 for dose .3, identical across all 128 worlds at each of three goals. Contact has not started then. Median maximum post-contact J6 speeds are approximately 2.65–2.67 rad/s, below the 3.1415 limit; this median does not establish universal post-contact compliance.

Independent source review finds zero reset velocity and J6 reset position matching the frozen default. J6's action scale is about .05516 rad; the two policies saturate its first position command in opposite directions, both with Kp6=800. Thus initial commanded motion/dynamics is a candidate explanation, not an established reset bug, infeasible cap, or impact-generated velocity problem. Native peaks include physics substeps, not just control endpoints.

The safety code does charge terminal positive rewards. Its zero-floor smooth excess penalty trades severity against task utility; it does not guarantee a violation-frequency target. Fresh evaluation normalizers do not reconstruct historical training pressure. These facts argue against blindly increasing penalties or changing caps.

## Proposed next steps, smallest first

1. **Startup attribution, no learning.** Matched state/controller state/randomness: compare normal first position command against a current-position hold for only that decision, retaining the policy stiffness command. Record reset state, actual targets, gains/torques and native substep velocities; verify no contact. Observe subsequent steps too: a hold may merely postpone the spike. This is an intervention, not an approved permanent fix. Whole-arm coupling means it is not automatically evidence against J6 alone.
2. **Command authority, no learning.** Using identical unmodified first motion, intervene on the actually consumed latched speed goal at the following decision with low/middle/high requests. Compare subsequent joint targets/stiffness, achieved impact speed, impulse, tracking and all load tails. Preserve strict checkpoint phase/metadata handling. This tests downstream goal conditioning with selection latency, not whether autonomous selection is beneficial.
3. **Separate impact-load feasibility.** Use existing traces to align contact, cap crossings, delivered-impulse milestones and terminal reward. Seek low-load useful-strike witnesses; failure to find one in these policies is not proof of impossibility. Inventory saved normalizer state before proposing historical-pressure conclusions. A later bounded physical probe would require a separately fixed protocol.
4. **Then a separately approved selector pilot, if step 2 supports it.** Keep the original failed gates and explicitly label any E2 run unqualified research, not promotion. Use the existing strict transfer and fresh optimizer, compare against an appropriate matched fixed-speed control, and repeat causal overrides afterward. Report actual delivered impulse, tracking and loads as well as chosen-versus-achieved speed. Correlation alone could mean the speed output labels motion instead of controlling it.

## Discussion resolution

The objective reviewer challenged indefinite E1 safety gating: complete safety qualification need not precede every scientific test of the selector. The mechanism reviewer agreed, conditional on separate approval and truthful failure reporting. Both then prioritized startup attribution after seeing the precontact spike. All agree the two load mechanisms must not be conflated, and no new infrastructure, spline timing architecture, reward redesign, cap relaxation or additional training is justified by this analysis alone.

Decisions still needed before execution: approve the bounded no-learning interventions; later choose whether to continue strict E1 qualification or authorize an explicitly unqualified E2 feasibility pilot. No hardware-safety or constrained-optimality claim is supported yet.
