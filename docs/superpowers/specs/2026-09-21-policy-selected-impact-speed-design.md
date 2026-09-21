# Policy-selected impact speed: direct-control design

Date: 2026-09-21. Status: written design for user review, with staged task breakdown; not an executable implementation plan or authorization to train.

## 1. Agreed purpose

Keep one online policy. At stroke start it chooses a desired hammer impact speed; afterwards the same policy observes that choice and continues commanding joint targets and impedance. The supplied quintic curve remains geometric guidance. The policy learns acceleration, deceleration and arrival time. Optimize actual delivered impulse while retaining curve tracking and joint-load protection.

This is option A, explicitly selected by the user. It is not a trajectory generator followed by a second learned controller. There is no fixed timing polynomial, prescribed intermediate acceleration, or claim that endpoint speed alone determines duration. Exact tracking and exact requested impact speed are learned outcomes, not guaranteed boundary conditions. Higher speed can help impulse, but impulse also depends on configuration, effective mass and contact quality: retain the physical impulse objective.

## 2. Code baseline and preserved contracts

Implementation anchor: `/Users/nikerane/repos/unitree_rl_mjlab-worktrees/z1-step1-evidence-cleanup`, branch `z1-horizontal-trajectory-pilot`, commit `3ca89a297a58d8e59d74c4797caeb8629499f1aa`. Clean when inspected. Main checkout contains unrelated work and must not substitute for this baseline.

Paths in the following table are relative to that anchor.

| Existing component | Verified behavior | Planned treatment |
|---|---|---|
| `src/tasks/hammer/config/z1/env_cfgs.py` | Six joint-position plus six stiffness actions | Preserve the first twelve action meanings/order |
| `src/tasks/hammer/mdp/variable_impedance.py` | Bounded stiffness multiplier; damping linked by square-root scaling | Unchanged initially |
| `src/tasks/hammer/config/z1/__init__.py` | Quintic pilot uses geometric reference consumers and 20 mm vertical followthrough | Preserve geometry and followthrough during first comparison |
| `src/tasks/hammer/mdp/references.py`, `observations.py`, `rewards.py` | Geometric progress/error and tracking prior, not a motor-reference generator | Preserve; append goal observations and one isolated objective |
| `src/tasks/hammer/mdp/first_strike.py` | Shared 500 Hz first-contact/event tracker | Reuse event identity and speed measurement after regression checks |
| `src/tasks/hammer/rl/cat_ppo.py` | Soft-CaT reward/return handling; delegates actor creation and PPO updates | Preserve CaT mathematics and storage |
| `src/tasks/hammer/rl/runner.py` | Strict 12-action/40-observation direct-VIC metadata | Add a distinct experimental schema, do not relax legacy checks |
| `src/tasks/hammer/mdp/trackability.py` | Validates supported action signatures | Explicitly recognize the new signature; continue measuring joint-target tracking |

All old task registrations, checkpoints, reward weights, reset distributions and safety settings remain unchanged. New behavior belongs to one separately named experimental configuration. No actuator rewrite, dependency patch, broad cleanup or new training framework.

## 3. What the literature actually contributes

- **Tebbe et al., Sample-efficient Reinforcement Learning in Robotic Table Tennis:** explicit bounded impact-speed selection is a legitimate action representation. Their action includes racket speed in [0, 2] m/s and is executed through Reflexxes. We borrow the representation, not that bound or planner architecture. No official implementation was located; do not describe unrelated repositories as their code. [Paper, sections III–IV](https://arxiv.org/html/2011.03275).
- **CNP3O / spline_rl:** inspected official code at commit `013d31ccc24ddfecc38fdac3e37ae0d165a9c79e`. It adjusts an optimizer-proposed hitting velocity, converts endpoint derivatives into spline boundary parameters, and selects whole-stroke parameters. Its useful lesson is that a velocity choice needs an effective execution connection. We do **not** copy its spline boundary solver, episodic optimizer, timing network or trajectory playback into option A. Our connection is learned goal conditioning, not their analytical boundary construction. [Pinned implementation](https://github.com/pkicki/spline_rl/blob/013d31ccc24ddfecc38fdac3e37ae0d165a9c79e/spline_rl/policy/bsmp_policy_stop.py).
- **Goal-conditioned robotics implementations:** FetchReach exposes desired and achieved goals to a conditional controller. This supports the observation-conditioning pattern, not the claim that it already learns self-selected impact speed or impedance. Do not add HER/replay to our on-policy PPO. [Official environment](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/reach.py).

The full combination is our adaptation, not an exact reproduction of any of these papers. No third-party source copying is proposed; license verification would precede any later copying.

## 4. Action and state contract

Action order: `[six joint targets, six stiffness values, one speed-selection scalar]`.

On the first policy decision after reset:

1. `goal_pending = 1`; stored normalized goal has a neutral sentinel.
2. All thirteen outputs are active. Existing motion outputs execute normally; the last scalar is clipped by the existing action wrapper and mapped affinely from [-1, 1] to the experiment's positive speed interval `[v_min, v_max]`.
3. Store this speed once; clear `goal_pending` for the next observation.

On subsequent decisions, only the first twelve outputs execute. The speed output cannot overwrite the latched goal, including near contact or after the first strike. No extra selection-only hold step: that would alter the initial motion and require new hold semantics. Consequently, **the first motion command cannot depend on the newly sampled speed; conditioning begins on the following decision**. This is an explicit limitation of the simple shared actor, not hidden autoregression.

Append normalized stored goal and pending flag to both actor and critic observations (40 → 42 for this exact baseline). Reset only the selected vectorized environments' goal state. Existing action-history observation must retain its existing six-position-action meaning; do not silently expand it. Non-finite speed requests must fail validation rather than produce invalid gains or stored state.

Physical speed means hammer-head approach speed projected onto the configured nail axis at the first accepted contact onset, in m/s. Use the existing tracker's convention, not Euclidean speed or post-impact speed. Its implementation uses the onset/previous-free-flight position difference; confirm the documented callback order with existing instrumentation before using it as training truth.

Initially one selection covers the first strike of an episode. Repeated strikes with fresh selections are out of scope. Existing post-contact simulation can continue, but cannot earn repeated goal-achievement payouts.

## 5. PPO integration without a new algorithm

Use a narrowly scoped actor/distribution extension through the installed `RslRlModelCfg.class_name` and distribution configuration. Installed `rsl_rl.models.mlp_model.MLPModel` and `GaussianDistribution` are the actual interfaces; do not implement against an assumed older ActorCritic API.

Effective action masks are thirteen ones at selection, then twelve ones plus zero. Derive the mask from the **raw recorded observation**, before normalization, both during rollout and minibatch updates. Apply it consistently to log-probability, entropy and KL; include the mask with saved distribution parameters. Keep positive variances, and check old/new masks agree for each sampled transition. Never consult the live environment to reconstruct a training minibatch mask.

The installed PPO/storage interfaces already carry arbitrary distribution-parameter tuples and delegate these computations. Preserve `CatPPO`, soft-done returns and ordinary delayed-return credit for the initial selection. Do not repeat its likelihood across the stroke. Existing Gaussian raw-sample likelihood remains paired with the environment's clipped-action mapping; do not substitute clipped actions into that density.

An unmasked 13D Gaussian would be mathematically valid with ignored actions, but would add irrelevant entropy/KL and PPO-ratio noise. The mask is an efficiency/semantic choice, not a claim that ordinary PPO is invalid.

## 6. Learning a command, not a prediction

Conditioning alone does not force the controller to use the speed. Self-chosen goals can also collapse into predictions of whatever motion the policy already makes. Therefore qualify responsiveness before claiming success.

**Proposed training addition requiring approval with this design:** a short externally sampled-goal qualification stage using the same actor/controller, followed by policy-selected goals. This is training coverage, not a second policy or deployment-time external choice. Externally assigned episodes reset with `goal_pending=0` and the sampled goal already populated: the mask is twelve ones plus zero from the first decision onward. Infer masks from the pending flag, never episode length. During self-selected episodes, use section 4 exactly.

Use one event-based speed-consistency cost, not clock-time velocity tracking. Proposed normalized cost at first-strike finalization is `min(((v_measured - v_goal)/(v_max - v_min))^2, 1)`. A no-contact episode receives the maximum cost once on termination/timeout. A contacted but unfinalized event at termination also receives the maximum cost, labeled incomplete rather than successful speed achievement. A consumed-event latch prevents double charging. Safety termination retains its existing semantics. Report failures separately; never label a miss as measured zero-speed achievement.

This cost supplements, not replaces, actual delivered impulse, geometric guidance and CaT. Do not reward the magnitude of the requested speed itself. Preserve all existing reward weights initially. The new coefficient is an experimental calibration parameter, not justified by a paper's number. Determine it in a small qualification sweep against the measured event-return scale, report every attempted value, and freeze it before the main comparison. Existing reward-manager time scaling and negative-term classification in `cat/hook.py` must be checked so the intended once-per-event cost has the intended magnitude. Do not weaken the global reward-sign guards.

If self-selected training loses responsiveness, stop and report it. A continuing mixture of externally assigned goals is a possible follow-up, not an automatic expansion of this experiment. Ordinary PPO/GAE may give weak credit to the initial selector; diagnose that before proposing a second algorithm.

## 7. Staged task breakdown and go/no-go checks

### Stage 0 — Freeze evidence and experimental numbers

Record the baseline commit, task/checkpoint, existing training seed, reset library, reward configuration, curve and control timestep. Replay baseline speed, tracking, impulse and joint-load distributions. Choose a conservative positive speed interval based on observed Z1 behavior and physical limits, not the table-tennis paper. This is a candidate interval, not proof every goal is reachable. Publish the numeric interval and cost-coefficient sweep before GPU qualification.

### Stage 1 — Isolated state/action plumbing

Add a small goal-state/action module under `mdp/`, append observations, register the experimental configuration, and implement the actor/distribution seam under `rl/`. Preserve legacy configuration paths. Tests cover clipping/range endpoints, once-only latching, asynchronous subset resets, first-step semantics, inactive outputs, non-finite input, and physical axis/sign/units.

### Stage 2 — Event objective and compatibility

Connect to the existing first-strike event, with exactly-once success/miss/timeout handling. Check same-control-step finalization/reset ordering, event cost versus reward-manager dt scaling, and CaT positive/negative classification. Update the narrow action validators, versioned checkpoint/export metadata and supported evaluation entry point. No silent old-checkpoint loading into a wider actor; initial experimental training is fresh unless a separate explicit weight-migration test is approved.

### Stage 3 — Regression before training

Test masked likelihood/entropy/KL against hand calculations; replay saved observations through a PPO minibatch; verify unchanged CaT returns and correct delayed credit; serialize/reload and check deterministic outputs. Run legacy targeted tests and repository pre-training gates: `validate_rewards.py` phases A–M, `verify_contact_sensor.py`, `verify_reward_setup.py`, and the existing impact-progress, impulse-bound/constraint, delivered-impulse and soft-CaT tests. An unavailable runtime is a reported blocker, not an assumed pass.

### Stage 4 — One-curve responsiveness qualification

Train the same controller on externally sampled goals in the candidate interval. Evaluate low/middle/high goals using paired initial states and common randomness. Provisional promotion criteria, to be frozen with numeric interval before running: at least 90% of baseline hit rate; median absolute speed error no more than 10% of interval width among hits; paired high-minus-low achieved speed positive with a 95% confidence interval excluding zero. Report tails and failures, not only successful hits. Tracking RMS must not degrade more than 10% relative to a nonzero matched baseline, and report absolute millimetres. Keep safety thresholds fixed and report exceedance frequencies, CaT rates and maximum loads; no claim of hardware safety follows from this gate.

### Stage 5 — Self-selected speed plus impedance

Continue with all goals selected by the actor and unchanged bounds/coefficient. The selector received no direct PPO training in qualification: it is an unqualified head, even though its shared features may have changed. Begin a distinctly recorded training phase, explicitly initializing the selector output row/distribution state and fresh optimizer state while preserving the learned controller/critic and normalization. Test the transition; do not call the selector already trained. Qualification provided goals before first motion, whereas autonomous selection conditions only subsequent motion: this is a deliberate transfer limitation, not a directly matched E1/E2 comparison.

Re-run the goal-override diagnostic as a separately labeled post-latch intervention, as well as normal autonomous evaluation. Test whether changing only the stored goal changes commands before contact and achieved speed afterwards. Let the learned stiffness outputs respond too; report their changes. This measures total goal-conditioned policy response, not an isolated mechanical speed effect with fixed impedance. Report selector saturation, selected-versus-achieved scatter and first-step selection statistics. A constant best speed can be a valid optimum; variation across curves is not mandatory. Ignoring overridden goals is a failure of the explicit-command claim.

Compare against unchanged baseline under matched initial states, training budget and evaluation conditions. Use multiple training seeds before strong claims; the first seed is a plumbing/feasibility pilot, not a scientific conclusion. Compare actual impulse jointly with tracking, hit rate and joint-load distributions. Do not select a winner from training reward alone.

### Stage 6 — Three curves and visual evidence

Only after the one-curve gates, expand to left/centre/right starting points with the existing three quintic curves. Audit whether existing curve observations identify the relevant geometry before a shared-policy claim; if not, return for a scoped observation-design decision. Do not silently add a general spline encoder. Provide red-reference videos and white-background plots with nail, commanded curve and actual head trace. Overlay selected and measured speed in the video/summary. Fixed-versus-variable impedance is a later matched ablation, not needed to debug the speed channel.

### Stage 7 — Consolidation

Review the entire change and remove only new dead/duplicate code proven unused. Preserve tests for action meaning, speed causality, event accounting and safety. Do not delete research evidence or unrelated user changes. No large refactor is prerequisite to these experiments.

## 8. Deployment and reporting contract

A PT/ONNX actor alone does not contain the environment's goal latch. Deployment must reproduce goal selection/reset, bounds, normalization, observations and ignored-output handling. Export the raw deterministic thirteen outputs and versioned metadata; verify a short parity rollout through the deployment wrapper. No hidden controller-side velocity command exists in option A.

Every result includes requested speed, measured onset speed, contact validity, miss/timeout rate, actual delivered impulse, per-joint impulse/CaT statistics, path error and joint-target error. Keep simulated feasibility separate from hardware safety. Speed compliance is not a replacement for contact quality or impulse measurements.

## 9. Review and approval boundary

Three independent review lanes examined installed PPO/controller integration, scientific identifiability/training, and literature/public code. Their findings are reconciled here: retain direct control; admit first-step conditioning latency; avoid unused-action statistics; require externally varied-goal qualification; distinguish literature precedent from our adaptation.

This design intentionally gates physical bounds and cost calibration on baseline evidence rather than inventing robot-specific numbers. It is not ready for a large training launch until those recorded gates pass. Review cannot guarantee zero errors; regression, small qualification runs and causal interventions provide successive checks.

Next: user reviews this written design, particularly the short externally sampled-goal qualification stage. Then produce the executable file-by-file implementation plan with concrete test commands and subagent assignments. Subagent-driven execution is already the user's chosen method; do not ask them to choose it again. No production implementation or training was performed for this document.
