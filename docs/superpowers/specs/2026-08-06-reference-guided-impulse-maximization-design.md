# Reference-Guided Impulse-Maximization Campaign

**Date:** 2026-08-06

**Status:** owner-approved exploratory design; implementation not started

**Scope:** fixed-impedance Cartesian DiffIK Z1 hammering, with straight waypoint
guidance and 500 Hz joint-velocity CaT retained as mandatory final behavior

## Objective

Determine whether the existing productive-first-strike delivered-impulse reward can
increase useful axial impulse while preserving the learned straight approach and
joint-velocity legality. Distinguish a genuine harder strike from longer pressing,
repeat contact, lateral impulse, or reward-scale changes that do not alter behavior.

This is intentionally an experiment campaign, not a new analysis framework. Existing
reward readers, trackers, fixed-reset renderer, and CPU impulse diagnostic are reused.
The campaign may add only the task registrations, exact launch guards, tests, and
result-local summarization needed to run the declared treatments safely.

## Non-negotiable behavior and controller identity

The main scientific arms retain:

- the straight six-waypoint progress treatment at weight 8;
- the 500 Hz substep joint-velocity CaT signal at the real limit of 3.1415 rad/s;
- fixed Cartesian-delta DiffIK actions with scale 0.15;
- fixed impedance, with no `set_gains` and no variable-impedance action;
- the existing task geometry, reset, PPO settings, 4096 environments, and 200
  iterations;
- unchanged project-defined impulse caps `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]`.

The controller gains are frozen and reported explicitly:

| Joint | Kp | Kd | Effort limit |
|---|---:|---:|---:|
| Z1 joints 1, 3, 4, 5, 6 | 1000 | 100 | 30 N m |
| Z1 joint 2 | 1500 | 150 | 60 N m |
| Gripper, not policy-controlled | 100 | 20 | 30 N m |

MJCF passive damping (`1`, or `2` for joint 2) and DiffIK numerical damping (`0.05`)
are separate quantities and must not be reported as actuator derivative gains.

## Reward interpretation

The performance quantity is the nail-axis impulse delivered during the first
productive strike event. The existing event-linear reader is retained because it is:

- accumulated at 500 Hz;
- limited to one immutable first-contact event;
- paid once after event finalization;
- gated by productive nail-depth advance; and
- separated from later episode contact.

The primary outcome is the same first-event quantity, with failures assigned zero.
Episode-cumulative impulse is secondary only. A dose that raises cumulative impulse
through pressing or later contact is not an improved strike.

The estimand has one important limitation: task success can finalize or terminate
the event at the nail-depth threshold before the nominal 50 ms window ends. A faster
or harder successful strike may therefore be integrated over fewer substeps than a
slower press. Report the result as the implementation's **success-censored productive
first-event impulse**, not as an unrestricted fixed-horizon collision impulse. Event
duration and termination reason are mandatory companion fields.

## Stage 1: exploratory impulse-reward screen

Use a gated two-factor descriptive screen while keeping waypoint progress and
velocity CaT fixed:

| Factor | Levels |
|---|---|
| Existing pre-contact/impact-progress term `S` | 0, 8 |
| Event-linear delivered-impulse reward `D` | 0, 4, 16 |

This produces six cells. Train seeds 2--7 for every cell. Reuse a prior policy only
after proving exact equality of task identity, reward configuration, controller,
action scale, iterations, environment count, seed, code/asset provenance, and CaT
settings. The current `S8/D4` P+V policies are expected reuse candidates. Existing
`S8/D2` remains descriptive incumbent evidence but is not inserted as an unbalanced
cell in the primary matrix. All other cells are trained fresh. A completed weak seed
is never replaced.

`D8` is a conditional follow-up, not a seventh initial cell. Add it only if `D4`
shows a credible useful improvement while `D16` degrades or produces press-like
behavior, such that the missing intermediate dose can resolve the mechanism.

The screen is exploratory. It may identify at most two candidates and may describe
dose-response patterns, but it makes no population-level significance or definitive
"best reward" claim from these same seeds.

### Promotion rule

A candidate is eligible only if its six matched policies preserve the required
behavior:

- 6/6 productive successes;
- at least 5/6 complete all six waypoints no later than first contact onset;
- 6/6 have 500 Hz peak joint velocity no greater than 3.1415 rad/s;
- no contact interval lasting 50 ms or longer;
- no second contact onset; and
- finite, hash-bound evidence with the intended reward payout present.

Reject a cell if, versus its paired `D0` control, median contact dwell grows by more
than 25%, the transverse-to-axial impulse ratio worsens by more than 10%, or useful
speed/progress falls by more than 10%. Among eligible cells, choose the lowest dose
whose median productive first-strike axial impulse improves by at least 10% and whose
paired impulse improves in at least 5/6 seeds. If none passes, retain the lower-dose
incumbent by parsimony and record a reward/interface ceiling. Report all cells
regardless of eligibility. These margins are practical gates, not significance tests.

## Stage 1B: reward-stack simplification

For the selected `S,D` recipe, compare the complete reward stack with a lean version
that disables both low-contribution, behaviorally overlapping terms:

- `approach = 0`;
- `nail_driven = 0`.

The waypoint-progress reward, completion, nail-depth delta, action-rate term, joint
limit term, and selected impulse-related terms remain unchanged. This asks whether the
older proximity/occupancy shaping is useful or merely adds competing incentives.
It must not be described as removing all reference information.

If a waypoint-removal diagnostic is later run, it removes the newer waypoint package
only; the older base observations `strike_phase` and `strike_ref_error` remain. Its
correct label is therefore **waypoint removal**, not **all-reference removal**.

## Presentation-only comparison arms

The presentation requests four recognizable configurations:

| Label | Configuration | Evidence status |
|---|---|---|
| `M` | waypoint guidance + impulse-maximization reward, no velocity or impulse CaT | reuse the configuration-identical existing P+D4 policies where valid |
| `V` | waypoint guidance + velocity CaT | frozen P+V control |
| `V+I` | waypoint guidance + velocity and impulse CaT | conditional exploratory arm |
| `V+I+M` | waypoint guidance + both CaT constraints + selected impulse reward | conditional exploratory arm |

`M` is a diagnostic comparison, not a candidate final controller, because velocity
legality is mandatory. `V+I` and `V+I+M` are launched only if the unchanged impulse
caps genuinely bind in productive, qvel-legal, impact-dominated evidence. The branch
gate requires at least 5% eligible cap crossings, a lower confidence bound that does
not collapse to zero, p99 maximum utilization above 1, at least 95% productive
success, and at least 70% of the binding-joint exposure accumulated before first
release. If this gate does not pass, the presentation reports that the existing caps
are non-binding and no active-impulse treatment is scientifically identifiable.

If the branch opens, the current multi-read impulse signal and treatment fields are
re-qualified and the CaT dose is calibrated for those current semantics. Historical
`imp_max_p=0.5` is not reused merely because it exists; it predates the sliding-window
multi-read behavior. These arms remain exploratory and carry no hardware-safety
claim. Caps must never be lowered merely to manufacture an effect.

The conditional active-impulse arms are analysed separately from the reward-dose
curve. They do not decide the maximizing dose and are not pooled with log-only arms.

## Mechanism and anti-farming outcomes

For every policy record:

1. productive first-strike axial delivered impulse, primary;
2. 500 Hz joint-velocity legality;
3. first-event contact dwell;
4. first-event peak and mean axial force;
5. transverse delivered impulse;
6. number of contact onsets and post-event contact;
7. pre-contact axial hammer speed;
8. productive nail-depth advance and clamped final depth;
9. six-waypoint completion by contact onset;
10. finite-segment pre-contact perpendicular max/RMS and path ratio;
11. all six generalized constraint-reaction exposures and maximum cap utilization;
12. downward-action saturation frequency.

Interpretation is frozen before inspection:

- impulse rises with useful progress and without dwell/recontact growth: candidate
  stronger strike;
- impulse rises mainly with dwell, force, or later contact: press/recontact mechanism;
- dose changes nothing while downward action is saturated: action/controller ceiling;
- success, guidance, or velocity legality degrades: rejected dose;
- impulse CaT remains non-binding: honest null, not a failed run.

The robot-side `qfrc_constraint` quantity is described as a project-defined,
simulation-side generalized constraint-reaction exposure. It is not called a
manufacturer impulse limit, gearbox torque, or hardware-safety certificate.

## Evaluation and visual evidence

Use the existing canonical fixed-reset pipeline for every checkpoint:

1. run the trace-only 500 Hz renderer with auto-reset disabled;
2. run the CPU-only impulse diagnostic on the identical digest-bound reset;
3. require shared head-position, contact, and post-integration qvel channels to agree;
4. validate every leaf before aggregation;
5. render every policy in the exploratory screen without outcome filtering.

The existing tools do not support an honest held-out reset bank for these P tasks.
Repeated canonical evaluation must never be called held-out reset evaluation.
Confirmation instead uses untouched training seeds.

Every guided trajectory figure shows only:

- a dashed black entry-to-nail reference segment;
- six numbered open-diamond waypoints; and
- the realized hammer-head trace.

It must not show gate disks, a tube, or corridor shading. A waypoint-off diagnostic
shows only the reference/trajectory elements actually observed or rewarded by that
task. Figure titles state the arm, seed, waypoint reward, velocity-CaT dose, impulse-
CaT state, `S`/`D` weights, and measured first/cumulative impulse, gates, qvel, and
constraint utilization.

The slide video seed is declared before outcome inspection and reused across arms.
An optional representative video may be added only if clearly labelled as selected.

## Stage 2: held-out-seed confirmation

After Stage 1, freeze one winner and one exact incumbent/control before inspecting
confirmation outcomes. Train both on six untouched seeds, provisionally 24--29 after
confirming that none were used in development. Arms use matched seeds, exact
configuration/provenance, and no checkpoint selection or seed substitution.

The paired primary comparison is success-weighted productive first-strike impulse.
Call the candidate replicated only if it remains eligible and improves impulse in at
least 5/6 paired seeds without crossing the anti-press margins. Guidance, qvel
legality, success, and constraint exposure remain mandatory secondary evidence. This
confirmation may continue beyond the initial six-hour operating window.

## Minimal implementation boundary

Permitted changes are limited to:

- exact task/config registrations or a small existing-factory extension;
- fail-closed launch allowlists and provenance assertions;
- focused tests for treatment isolation and live construction;
- faithful renderer task mappings/titles; and
- one result-local table/figure generator.

Do not build a new evaluator, manifest framework, optimizer, curriculum system,
effective-mass reward, tube reward, or generic campaign framework. Do not edit
installed `mjlab`, `rsl_rl`, MuJoCo, MuJoCo-Warp, or the asset checkout.

## Qualification and operations

Before GPU launch:

- use focused TDD for every new treatment identity;
- run the affected tests and the mandatory reward/impulse test set;
- run `validate_rewards.py` A--M, `verify_contact_sensor.py`, and
  `verify_reward_setup.py`;
- re-run scripted-reference qualification;
- prove frozen comparator reward/config digests are unchanged;
- commit and push named source files only;
- deploy a clean detached revision-specific Vega checkout using Git, never `scp` for
  tracked source;
- run live CUDA config assertions and a treatment-faithful smoke before training.

Training runs in arrays with one GPU per policy. First launch a small operational
canary subset, inspect only task/config/provenance/finite execution, and then submit
the remaining predetermined seeds regardless of canary policy quality. Retry only a
documented infrastructure failure at the identical seed/task/config/revision.

The Mac currently has less than the established free-space floor. Bulk checkpoints,
traces, and videos remain on Vega; copy back only validated small tables, figures, and
presentation videos. Never delete frozen evidence to make space.

## Expected six-hour sequence

1. **0:00--1:15:** minimal TDD implementation, preregistration, focused gates.
2. **1:15--2:00:** serial regression/review, named commit and push.
3. **2:00--2:30:** clean Vega deploy, CUDA qualification, operational canaries.
4. **2:30--3:15:** complete the 30 new primary-screen policies; reuse six matched
   `S8/D4` policies only after identity verification.
5. **3:15--4:45:** fixed-reset traces, impulse diagnostics, validation.
6. **4:45--5:30:** tables, trajectory grids, and presentation videos.
7. **5:30--6:00:** independent raw-artifact verification, select/freeze the winner
   and incumbent, then launch the 12-policy fresh-seed confirmation if warranted.

Queueing, failures, or evidence volume may extend the campaign. Scientific gates are
not skipped to meet the nominal time.

## Explicitly deferred

- variable impedance and `set_gains`;
- peak-force, cumulative-episode impulse, energy, or superlinear rewards;
- reward curriculum or automatic curriculum;
- effective-mass/hitting-flux implementation;
- action-scale changes unless the dose curve is null and saturation evidence makes a
  separately qualified scale experiment necessary;
- waypoint-density, gate-radius, tube, or geometry tuning;
- formal hardware-safety claims from simulation-only constraint exposure.
