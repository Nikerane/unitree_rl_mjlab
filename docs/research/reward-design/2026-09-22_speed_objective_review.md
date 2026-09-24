# Speed objective review: command meaning before further dose tuning

Date: 2026-09-22. Independent code-grounded design critique. Research only: no source changes, jobs, gate relaxation, or promotion authorized. Local source and approved design are the primary sources; recommendations below are deductions from this implementation, not new literature claims.

## Bottom line

The system is already learning continuous joint motion and variable impedance, but **not yet choosing its own impact speed**. The fresh pair addresses the quality and safety trade-off of an externally conditioned controller. It cannot establish the professor's full objective. The next useful question is whether the speed coordinate is a causal command under the actual one-shot selection timing—not whether another penalty increment can reduce one more load statistic.

The unchanged E1 accuracy gate failed, and the dose intervention did not reduce J6 exceedance frequency. Neither result should be renamed a pass. However, an explicitly approved, separately named **unqualified E2 feasibility diagnostic** need not logically wait for deployment-grade load compliance or perfect command accuracy. It must not be presented as ordinary Stage-5 promotion under the existing approved sequence. Its purpose would be to test selector credit and command semantics, with safety failure retained as a first-class result.

## What the code actually makes possible

1. In external mode, reset samples the goal and sets pending to zero. Later speed outputs cannot change it: [`impact_speed.py`](../../../src/tasks/hammer/mdp/impact_speed.py), lines 54–78. The actor masks the thirteenth coordinate out of likelihood, entropy and KL, using pending from recorded observations: [`impact_speed_policy.py`](../../../src/tasks/hammer/rl/impact_speed_policy.py), lines 31–61 and 95–104. Thus the selector has no direct selection learning in E1; the goal is not fixed globally, but is externally fixed **per episode**.
2. The speed action only changes stored state; its actuator application is empty: [`impact_speed.py`](../../../src/tasks/hammer/mdp/impact_speed.py), lines 109–115. The execution connection is learned goal conditioning, not a velocity servo or boundary-value solver. First motion occurs simultaneously with selection, so cannot condition on that newly sampled value. The approved design explicitly acknowledges this in sections 4 and 5 of [the approved design](../../superpowers/specs/2026-09-21-policy-selected-impact-speed-design.md).
3. Motion and impedance continue to be commanded every decision. Stiffness is a bounded multiplier with coupled square-root damping, and native gains are applied during simulation: [`variable_impedance.py`](../../../src/tasks/hammer/mdp/variable_impedance.py), lines 124–131 and 305–326. The inherited experimental config uses C=1.25: [`config/z1/__init__.py`](../../../src/tasks/hammer/config/z1/__init__.py), lines 419–424. Therefore “choose speed once” does **not** mean “choose the whole motion or impedance once.” Nor are six independent damping choices present.
4. The physical objective remains actual productive first-window delivered impulse, linear above its normalizer in this arm: [`rewards.py`](../../../src/tasks/hammer/mdp/rewards.py), lines 449–479; [`config/z1/__init__.py`](../../../src/tasks/hammer/config/z1/__init__.py), lines 392–407. The speed consistency term is an auxiliary once-per-event squared-error cost, not the impulse objective: [`impact_speed.py`](../../../src/tasks/hammer/mdp/impact_speed.py), `consume_speed_cost` and `ImpactSpeedCost`.
5. The curve is geometric guidance, not an exact prescribed trajectory. The pilot changes reference consumers to quintic geometry with 20 mm followthrough, while the inherited tracking prior has weight 0.2 and no annealing: [`config/z1/__init__.py`](../../../src/tasks/hammer/config/z1/__init__.py), lines 296–308 and 528–533. If “given trajectory” means exact geometric constraint, the present architecture does not provide that guarantee.

## Are we overgating?

The [fresh dose-pair result](../../results/2026-09-22_z1_speed_impulse_dose_pair.md) reports 128/128 hits at every goal, positive paired high-minus-low contrasts for both policies, and passing utility guards. Treatment median absolute speed errors are 0.0969/0.0609/0.0527 m/s; the low endpoint fails 0.07. That establishes useful but imperfect external-goal response. The 0.07 rule was a frozen engineering criterion (10% of interval width), not a demonstrated physical discontinuity below which selector learning becomes possible.

Conversely, J6 impulse exceedance remains 128/128, 128/128 and 127/128, with J6 velocity exceedance in all episodes. This is not a minor accuracy issue. No tested goal demonstrates load-feasible behavior in this evidence. Autonomous speed choice might alter that, but cannot be promised to solve it. The reported middle-goal impulse retention is only 90.2%, and other tails worsen. Further dose changes cannot be called monotonic safety improvements.

My strongest disagreement is with **requiring full safety success before any diagnostic of the actual selector**. That confounds “safe solution achieved” with “can this architecture represent a meaningful decision?” A converse mistake would be to remove load reporting or lower the frozen accuracy threshold merely to unlock E2. Keep both failed gates; request a different, bounded scientific question if the user wants progress toward autonomy now.

## Minimal honest bridge, ranked

### 1. Freeze the failure and choose the claim

Preserve both complete pair results. Obtain explicit approval either to continue E1 qualification or to depart from its promotion sequence for a small **unqualified selector-feasibility** study. Freeze one checkpoint, dose, bounds, coefficient, budget and stopping criteria prospectively. Dose 0.3 is a defensible diagnostic candidate because most impulse severities decreased while utility guards held—not a demonstrated optimum or safe controller. Do not search doses indefinitely.

### 2. Establish post-latch command authority before training a selector

Use matched reset/randomness, identical first executed motion and then low/mid/high stored-goal interventions before contact. Verify first-action identity, subsequent position-command and stiffness differences, achieved onset-speed contrasts, contact timing, tracking, delivered impulse, and every load tail. The evaluator already intervenes at decision 1 and records first action and second commands: [`evaluate_impact_speed.py`](../../../scripts/evaluate_impact_speed.py), lines 184–196 and 286–342. This diagnostic includes the first-motion conditioning latency that external E1 does not.

This is a proposed evaluation, not a claim it was run. Runtime mode/checkpoint transfer must remain strict and explicitly recorded. Do not silently load external metadata as policy mode. If one-shot timing destroys the response, stop and resolve that scoped transfer defect before autonomous training. If goals only change displayed labels without changing commands and achieved outcomes, the explicit-speed representation has failed regardless of impulse.

### 3. Only then request a bounded selector-learning pilot

The existing transfer initializes only the selector row/bias and distribution coordinate, preserves learned control, checks zero pending columns, and requires a fresh optimizer: [`impact_speed_policy.py`](../../../src/tasks/hammer/rl/impact_speed_policy.py), lines 107–142; [`runner.py`](../../../src/tasks/hammer/rl/runner.py), lines 774–798. Verify these contracts rather than inventing a second controller or timing planner.

Report selector participation/gradient evidence at selection transitions, number of actual selections, saturation and raw-versus-clipped choices, selected-versus-achieved values, actual impulse and load distributions. Compare with the freshly initialized selector and a matched fixed-midpoint intervention to distinguish selector benefit from further low-level controller training. Post-training repeat the overrides: correlation alone cannot exclude a speed head that merely predicts the motion its shared network would perform anyway.

The maximum possible reduction in speed cost by relabeling an unchanged motion is an obvious alternate explanation for better return. Physical command changes and achieved-speed intervention effects are therefore necessary, while improved actual delivered impulse at a declared load trade-off is the task evidence. A constant selected speed may be sensible on one fixed curve/start; diverse speed outputs are not a success criterion.

### 4. Leave load feasibility as an explicit parallel unresolved requirement

Inspect existing terminal-inclusive traces for when J6 velocity and impulse accrue and how they covary with commanded stiffness, contact geometry and delivered impulse. Do not assume speed alone controls joint load, or that increased configured CaT ceiling uniformly increases realized pressure—the pair explicitly shows otherwise. No hardware/generalization/safe-constrained-optimum claim is defensible until load compliance is established. Changing caps, geometry, gain range or safety mechanism is a new treatment requiring approval, not this bridge.

## Choice once versus ongoing

Keep one speed choice per first strike for this diagnostic. Allowing the target to update throughout the stroke would let the policy chase its realized speed and cheaply shrink the auxiliary cost, obscuring the intended terminal-speed commitment. Ongoing joint targets and stiffness already supply feedback adaptation. Repeated strikes, continuous speed replanning, a spline boundary solver or a second policy are different questions and are not needed to test whether the approved one-shot architecture works.

## What must be resolved first versus later

**Before any diagnostic launch:** explicit authority to depart from failed E1 promotion; frozen budget/checkpoint/treatment; strict phase transition and first-step timing validation; intact event measurement/reward accounting; causal intervention protocol and load-reporting stop rules.

**Before claiming professor-level success:** trained selector with retained causal command authority, improvement judged by actual impulse jointly with tracking and loads, demonstrated load-feasible operation, and independent training seeds. Neither precise 0.07 compliance nor command responsiveness alone proves this complete objective.

No experiment or implementation was executed for this review.
