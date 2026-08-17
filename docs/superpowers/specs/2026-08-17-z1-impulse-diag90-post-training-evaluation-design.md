# Z1 diagnostic-0.9 impulse-CaT post-training evaluation

**Status:** user-approved on 2026-08-17. This is a no-learning evaluation of the completed
500-iteration matched pair. It does not authorize another training treatment, a cap change, or a
hardware-safety claim.

## Question

For the exact seed-2 matched training pair, did training with `imp_max_p=0.5` at the uniform
diagnostic-0.9 boundary shift the learned VIC policy toward lower per-joint reaction-impulse
exposure without degrading hammering or moving pressure into the velocity constraint?

Soft-CaT is a graded PPO credit/continuation treatment, not a physical clamp. The primary claim is
therefore learned policy shaping. Near-zero diagnostic-boundary violations are a stronger secondary
compliance outcome, not an assumed property of the algorithm.

## Frozen inputs

| Role | Training `imp_max_p` | Final checkpoint SHA-256 |
|---|---:|---|
| `diag90_control` | `0.0` | `f4f86cfd81fdc78824b85a059735b2f605c6761c624be59e0e778ef3fbd681c3` |
| `diag90_target` | `0.5` | `ddd7ac4c855160bff1db2af52e642d960dab2bef34e41dd2532002185eb36d15` |

- Code revision: `030942f34ac4a79131c1b70206d3c4acd58da79a` plus the reviewed evaluator-only
  descendant produced by this plan.
- Asset revision: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Task: `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Track-Vel-Delivered4-`
  `JointPosition-VariableImpedance-TT`.
- Training identity: seed 2, 4,096 environments, 24 steps per rollout, 500 PPO iterations,
  velocity CaT active, diagnostic caps `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N.m.s`.

The evaluator must reject a role/checkpoint hash mismatch before constructing an environment.

## Matched measurement protocol

Both learned policies run in the same measurement-only environment with live `imp_max_p=0`. The
velocity arm stays active and the existing impulse measurement stays enabled. Counterfactual
`delta_impulse` at `imp_max_p=0.5` is replayed offline from the recorded chronological margins.
This preserves identical inference physics, actions, rewards, observations, and native task
termination across arms.

Two thresholds are computed from each raw Lambda trace:

1. provisional project caps `[0.820, 1.640, 0.820, 0.820, 0.820, 0.820] N.m.s`, reported first;
2. trained diagnostic caps `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738] N.m.s`, used for the
   treatment-effect endpoint.

Neither vector is a manufacturer-certified damage limit.

### Fixed population

- 64 environments;
- deterministic mean actions;
- seed `2026081202`;
- exact banked initial-population hash;
- first episode only, ordinary success termination, no auto-reset continuation.

### Training-like stochastic populations

Run three prespecified replicas, each with 4,096 environments and 24 control steps:

- `2`;
- `2026081701`;
- `2026081702`.

Reset, observation, and policy-action RNG streams are isolated. Each role receives the same reset
states and latent action-noise stream for a given replica. The paired primary analysis uses episode
zero in each environment. Later auto-reset segments remain labelled secondary stress evidence
because termination timing can make them unmatched.

## Outcomes

### Primary impulse endpoint

For every initial episode, compute

`rho = max_t,j Lambda[t,j] / L[j]`.

Report, by arm and replica:

- fraction of initial episodes with `rho > 1`;
- p50, p95, p99, and observed maximum of `rho`;
- target-minus-control paired violation-risk difference and target/control risk ratio;
- per-joint Lambda p50/p95/p99/max, utilization, positive margin, and violating-segment rate;
- first-contact-event utilization so episode-length differences cannot masquerade as lower impulse;
- violating reads, activation windows, physical-event associations, responsible joint, and reads per
  event, with ambiguity and censoring retained.

### CaT attribution

For actual velocity pressure and offline impulse pressure at `p=0.5`, report:

- `delta_velocity`, `delta_impulse`, and exact `max(delta_velocity, delta_impulse)`;
- impulse-win, velocity-mask, and tie shares;
- event pressure `1 - product_t(1 - delta_impulse_t)`;
- completed and right-censored event summaries separately.

### Utility and trade-off endpoints

Capture before auto-reset and summarize:

- success and timeout;
- productive first strike and first-event delivered impulse;
- nail depth and cumulative delivered impulse;
- episode duration and reward as descriptive quantities;
- per-joint 500 Hz velocity peaks and velocity-limit violations;
- VIC stiffness coordinate, Kp, and Kd immediately before and through first contact.

The comparison must flag a lower impulse result accompanied by worse task behavior, a higher
velocity tail, or load redistribution to another joint.

## Statistical unit and analysis

Controller reads are not independent: the 50 ms windows overlap, and reads share episodes and
physical events. Paired bootstrap resampling therefore uses whole environment IDs, preserving all
reads/events and both arms together. Use 10,000 resamples for 95% intervals and report all three
stochastic replicas separately before pooling.

Evaluation replicas quantify inference/reset stochasticity for these two learned policies. They do
not replace independent training seeds. No p-value over PPO iterations or controller reads is
permitted.

Provisional engineering utility margins, to be reported rather than silently optimized, are:

- fixed success and productive strike at least `58/64`;
- stochastic success no more than five percentage points below control;
- first-event delivered-impulse target/control ratio at least `0.90`;
- no clear increase in true velocity-limit violations.

These margins are not existing hardware requirements.

## Native-result verdicts

1. Lower diagnostic violation probability and lower p95/p99 utilization, with retained task and
   velocity behavior: learned graded impulse reduction for this training seed.
2. Lower mean but unchanged violation rate or tail: behavioral softening, not boundary confinement.
3. Lower J3 exposure with higher exposure at another joint or in velocity: redistribution/trade-off,
   not clean enforcement.
4. Active, winning impulse pressure with unchanged tail: calibration/treatment-strength problem.
5. Absent or too-rare provisional-cap violations: report exactly, "impulse constraint is empirically
   nonbinding under the surveyed population."

Zero observed violations never proves zero risk, a hard clamp, or hardware safety.

## Complete-contact calibration follow-up

Native termination remains authoritative for task behavior. If native traces remain materially
terminal-censored, run a separate measurement-only shadow continuation using the same survey seam:

- remove only the `nail_driven` termination after verifying parity through native success;
- leave physics, policy, actions, gains, observations, rewards, caps, and CaT equations unchanged;
- continue until five consecutive off-contact 2 ms samples and the 25-substep Lambda window has
  flushed;
- stop at 250 ms after native success and retain unresolved cases as right-censored;
- compute event pressure only for completed, unambiguous contact associations.

At least 50 completed unambiguous activating events and at least 95% completion are required to call
the complete-event dose calibrated. Residual violations do not automatically justify increasing
`imp_max_p`: three saturated `p=0.5` reads already give event pressure `0.875`, and four give
`0.9375`.

## Implementation boundary

Reuse `scripts/impulse_cat_activation_survey.py`, its recorder, `_run_population`,
`summarize_population`, `shadow_impulse_cat`, and contact/event association code. Add only safe
checkpoint-role selection, matched RNG streams, missing episode/utility telemetry, paired summaries,
and a guarded two-arm Vega launcher. Do not adapt the historical fixed-impedance evaluator or create
a new RL algorithm/environment/evaluator framework.

Use test-first development, a 2-environment x 8-step live smoke, mandatory reward/contact gates,
clean pinned code/assets, collision-safe output leaves, `--no-requeue`, and no automatic retry.
