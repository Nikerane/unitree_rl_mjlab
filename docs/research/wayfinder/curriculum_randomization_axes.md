# Defensible curriculum and randomization axes for Z1 hammering

**Date:** 2026-08-06
**Purpose:** executable evidence boundary for the experiment sequence after the joint-position fixed-impedance baseline and active impulse-CaT mechanism check, but before the main VIC comparison.

## Decision in one paragraph

The first defensible randomization treatment is **robot reset joint-position variation**, using the repository's existing `[-0.05, +0.05] rad` per-joint envelope only as a *candidate outer boundary*. That envelope has local evidence from older Cartesian-action multi-reset campaigns, but it must first pass a short feasibility qualification with the new joint-position action/reference contract. The actor's current observation corruption is already part of the training baseline and must remain identical in every arm; it is not a new treatment. No numerical uncertainty band is currently defensible for nail/contact properties, actuator dynamics, latency, link inertias, encoder bias, or task-fixture placement. Those axes need measurements, and they should not enter the causal FIC-versus-VIC campaign. Contact-solver and half-timestep variants are useful only as labeled final simulator-sensitivity evaluations. If active impulse CaT is hard to learn from scratch, a ramp in **CaT termination pressure** can be tested as a separate curriculum; the impulse threshold itself must not be annealed.

This gives a deliberately narrow order:

1. qualify and train the nominal joint-position FIC baseline;
2. calibrate the impulse reference with the controlled drop;
3. validate active impulse CaT at fixed reset and fixed pressure;
4. introduce reset-pose generalization, and introduce a CaT-pressure curriculum only if the fixed-pressure run shows a learning failure;
5. integrate VIC under the same nominal task;
6. compare FIC and VIC with exactly the same accepted curriculum/randomization configuration;
7. run solver, timestep, and later hardware-calibrated perturbations as a held-out robustness panel.

That ordering implements the professor-meeting sequence recorded in the [presentation handoff](../../results/2026-08-06_PRESENTATION_HANDOFF.md) without allowing domain randomization to obscure whether joint actions, impulse CaT, or VIC caused an observed change.

## What the current baseline actually contains

"Fixed reset" does **not** mean fully deterministic training.

- The Z1 guideline configuration overrides the base reset event and fixes the robot joint reset at its nominal pose. The P/P+V/P+D4/P+V+D4 presentation campaign also preregistered no initial-condition randomization. See [env_cfgs.py](../../../src/tasks/hammer/config/z1/env_cfgs.py) and the [2x2 preregistration](../../results/2026-08-05_presentation3_2x2_prereg.md).
- Actor observations are already corrupted during training in [hammer_env_cfg.py](../../../src/tasks/hammer/hammer_env_cfg.py): joint position `+-0.01 rad`, joint velocity `+-1.5 rad/s`, EE and hammer-head position `+-0.005 m`, EE and hammer-head velocity `+-0.01 m/s`, nail-top position `+-0.002 m`, and nail depth `+-0.001 m`. Play/evaluation disables this corruption.
- Physics are nominal and fixed: `0.002 s` simulation timestep, decimation 10, hence one policy action every `0.02 s`. There is no active physics, latency, mass, friction, gain, or external-disturbance randomization in the hammer task.
- The matched presentation P+V arm uses fixed velocity-CaT pressure (`max_p=0.5`, `tau=0.95`) and log-only impulse CaT (`imp_max_p=0`). Those values do not constitute evidence that the same pressure is appropriate for the sparse impulse event.

Therefore every new arm below must preserve the current observation corruption unless observation noise itself is explicitly preregistered as the sole factor. Calling a nominal-versus-reset experiment "no DR versus DR" would otherwise be inaccurate; the precise comparison is **observation corruption only** versus **the same observation corruption plus reset-pose variation**.

## Evidence classification by axis

| Axis | Current nominal / local evidence | Numerical range status | Earliest legitimate use | Disposition |
|---|---|---|---|---|
| Actor observation corruption | Exact ranges listed above are live in code and were present in the training baseline | **Locally fixed**; no evidence for widening them | From the first joint-action baseline | Keep identical in all causal arms; do not treat as a new factor |
| Robot reset joint position | Guideline training currently fixes reset. Base event and older multi-reset evaluations used independent offsets within `[-0.05,+0.05] rad`; older Cartesian policies remained highly successful, but the new action/reference geometry is different | **Locally defensible candidate outer envelope**, conditional on new joint-action qualification | After nominal FIC and active impulse-CaT validation, before VIC | First actual generalization treatment; describe as simulation task generalization, not a hardware uncertainty model |
| Robot reset joint velocity | Reset is exactly zero; no local randomized campaign or hardware distribution | **Unsupported** | Only after a measured or deliberately specified task requirement | Keep zero |
| Nail/block/fixture initial pose | Fixed in the scene; no measured assembly, vision, or placement tolerance | **Requires preliminary measurement** | Final robustness or a later hardware-transfer campaign | Exclude from present training matrix |
| Nail mass, slide damping, friction loss, block/hammer friction | Live XML has nail mass `0.007 kg`, slide damping `0.5`, slide friction loss `30.0`, block friction `1.5 0.01 0.001`, and hammer contact friction `1.5 0.02 0.002`; prior proposed bands are stale, and an older friction override did not actually apply | **No defensible uncertainty band** | After physical parameter identification or a clearly labeled simulation-only study | Freeze nominal |
| Contact compliance / solver parameters | Live contact uses `solref="0.008 1"` and `solimp="0.9 0.95 0.001 0.5 2"`; local `solref x2` probes materially changed impact results | `x2` is an evidenced **stress condition**, not an uncertainty distribution | Final held-out simulator sensitivity | Do not train on it or fold it into nominal statistics |
| Simulation timestep | Nominal `0.002 s`; half-timestep probes exist | Half timestep is an evidenced **numerical convergence check**, not physical DR | Final numerical-sensitivity panel | Never randomize timestep inside the main campaign |
| Hammer, fixture, robot-link mass/COM/inertia | Live hammer head is `0.2 kg` and fixture is a separate `0.045 kg`; no uncertainty distribution exists. Older armature sweeps were sensitivity probes, not identification | **Requires weighing/CAD/system identification** | Final robustness after calibration | Do not reinterpret `0.2` to `0.245 kg` as a training interval; those are different bodies/experimental meanings |
| Actuator gains, damping, effort limits, armature, friction | Simulator has exact nominal parameters in [z1_constants.py](../../../src/assets/robots/unitree_z1/z1_constants.py); gain and armature variants were analytic probes. Unitree publishes capabilities, not the end-to-end plant uncertainty distribution needed here | **Requires hardware system identification** | After VIC integration, in a matched final robustness study | Freeze for primary FIC/VIC comparison |
| Observation/action latency | Simulator uses a 50 Hz policy ZOH. Mjlab can delay observations in integer policy steps, but the hammer action path has no qualified action-delay model; actual hardware pipeline latency is unmeasured | **Requires timestamped hardware measurement and action-delay implementation** | Final robustness after calibration | Do not assume one policy step (`20 ms`) is the hardware delay |
| Encoder bias / calibration offset | Current observation corruption is zero-mean noise, not persistent bias; no Z1 homing-repeatability dataset exists | **Requires hardware measurement** | Final robustness after calibration | Exclude now |
| Impulse threshold | Must be tied to the controlled drop and the chosen impulse semantics; the temporary lower diagnostic threshold is only a mechanism test | **Calibration output, not a DR axis** | Active-CaT validation | Freeze within each causal campaign; never randomize or anneal it |
| VIC-commanded stiffness | This is the policy action whose value is being tested, not environmental uncertainty | **Not DR** | VIC stage | Do not perturb or rescale it during the primary FIC/VIC efficacy comparison |

The live nail/contact values above come from the canonical asset scene under `/Users/nikerane/repos/safe_impact_manipulation/hammer_z1_env/assets/`. The local [impulse-vacuity](../../results/2026-07-12_impulse_vacuity.md), [solver-sensitivity](../../results/2026-07-10_solver_sensitivity.md), and [phase-0 diagnostic](../../results/2026-07-17_phase0_diagnostics.md) records explain why the old proposed parameter sweeps are not uncertainty calibrations.

## Curriculum: what to vary and what not to vary

### 1. Constraint-exposure curriculum

The first active impulse-CaT experiment should **not** use a curriculum. It should compare log-only versus active CaT at fixed reset, with a single frozen diagnostic threshold and frozen termination-pressure parameters. This answers the mechanism question: does active impulse CaT change the sampled return/behavior under otherwise identical conditions?

Only if the fixed-pressure active arm shows a clear exploration or learning collapse should a pressure curriculum be added as a separate recovery experiment:

- ramp `imp_max_p`, not the impulse threshold;
- hold the calibrated event definition, normalizer, `tau`, rewards, reset distribution, and training budget fixed;
- end at the same final pressure as the non-curriculum active arm;
- choose schedule endpoints and duration from a preliminary short sweep; the repository does not yet justify numerical values;
- if the curriculum is retained for the thesis comparison, apply the exact frozen schedule to every relevant FIC and VIC arm.

The CaT paper ramps soft-termination probability from `0.05` to `0.25` and shows the exploration/conservatism tradeoff, but that is evidence for the **method**, not a transferable Z1 impulse range. The paper studies locomotion constraints with different violation frequency and semantics. See [Constraints as Terminations](https://arxiv.org/pdf/2403.18765) and the local [faithful soft-CaT plan](../reward-design/FAITHFUL_SOFT_CAT_IMPL_PLAN.md).

The lower impulse threshold requested for early validation is likewise not a curriculum endpoint. It is a diagnostic intervention. After the controlled drop and active-mechanism check, establish and freeze the experiment threshold before making controller claims. The current calibration caveats and normalizer checks are documented in the [impulse-CaT plan](../reward-design/IMPULSE_CAT_IMPL_PLAN.md).

### 2. Reset-pose curriculum

Reset-pose variation is the only new axis with a local candidate envelope. Use it in two steps.

**Qualification, no RL:** Sample the nominal pose, boundary poses, and interior poses from the existing `+-0.05 rad` envelope under the new joint-position action/reference implementation. Reject or shrink the envelope if any required waypoint becomes unreachable, joint/action limits bind systematically, the head misses the intended approach corridor, impact ordering changes, or the reference can no longer produce a valid strike. The older multi-reset result only proves that the envelope was workable for an older Cartesian-action policy; it does not waive this gate.

**Training ablation:** Once qualified, compare:

- `R0`: fixed nominal reset plus the existing observation corruption;
- `R1`: the same configuration plus qualified reset joint-position variation.

Use matched seeds, training budgets, CaT configuration, rewards, evaluation checkpoints, and nominal evaluation episodes. Add a held-out reset panel with explicit nominal, interior, and boundary strata. Report nominal and randomized-panel outcomes separately so increased average robustness cannot conceal a nominal-performance or safety regression.

A performance-gated widening schedule is reasonable only after `R1` at the qualified envelope has been shown to be a learning problem. If used, advancement must require both task performance and constraint legality on nominal and current-boundary evaluations. The next width and advancement thresholds are owner decisions after the qualification data; no local source justifies inventing them now. This follows the general ADR principle that training can begin narrow and expand after competence, while avoiding importing Rubik's-Cube ranges into this task. See OpenAI's [ADR description](https://openai.com/index/solving-rubiks-cube/) and [paper](https://arxiv.org/abs/1910.07113).

Do not turn on a reset-width curriculum and a CaT-pressure curriculum in the same first campaign. If both are eventually needed, establish each against the same frozen baseline, then run a small interaction check. Otherwise a success or failure cannot be assigned to either intervention.

## Executable experiment order

### Stage A — nominal FIC contract

1. Finish the joint-position action implementation and reference qualification.
2. Run the fixed-impedance `r_tt` ablation under fixed reset. Keep current observation corruption in both arms.
3. Select the nominal FIC baseline without adding physics DR, reset DR, or CaT curriculum.

**Promotion requirement:** the new action/reference path must reproduce a valid strike and satisfy the existing kinematic, corridor, velocity, and contact-ordering qualification gates. This stage is a dependency for interpreting every later result.

### Stage B — impulse reference and active-CaT mechanism

1. Run the controlled `0.2 kg` drop at the frozen nominal solver/contact configuration.
2. Validate the impulse event, normalizer, and graded excess signal.
3. At a preregistered lower diagnostic threshold, compare impulse CaT off versus on at fixed reset and fixed pressure.
4. Do not add reset DR or pressure curriculum here.

**Promotion requirement:** active CaT must be observably engaged without invalidating the task, and logging must distinguish nail-delivered impulse, joint reaction impulse, success, and the violation/termination process.

### Stage C — curriculum and reset generalization before VIC

1. Qualify the `+-0.05 rad` candidate reset envelope for the new joint-action/reference path.
2. Run `R0` versus `R1` with the accepted active-CaT configuration frozen.
3. If fixed-pressure CaT—not reset variation—caused a learning collapse, run the pressure-curriculum ablation separately.
4. Freeze a named training-distribution version and schedule after these ablations. Record the exact reset sampler, observation corruption, threshold, pressure schedule, and evaluation panel.

**Promotion requirement:** demonstrate retained nominal task performance, improved held-out reset performance, and no unacceptable regression in the impulse constraint metrics. Success alone is insufficient.

### Stage D — VIC efficacy

1. Integrate bidirectional per-joint stiffness commands while retaining joint-position targets and task-space reference guidance.
2. First qualify and pilot VIC at the nominal reset with no new uncertainty axes.
3. Compare FIC and VIC using the exact same frozen reward, reference, action timing, impulse semantics, threshold, observation corruption, accepted reset distribution, and accepted CaT schedule.

Do not randomize actuator gains, scale the commanded stiffness behind the policy, or introduce latency during this primary comparison. Those changes directly alter what "VIC versus FIC" means.

### Stage E — final robustness, not controller selection

Evaluate the already-selected FIC and VIC policies on a preregistered held-out panel:

- nominal physics and fixed reset;
- qualified reset strata;
- `solref x2` as an explicitly labeled contact-solver stress condition;
- half simulation timestep as a numerical-convergence condition;
- later, one-factor-at-a-time hardware-calibrated bands for placement, actuator dynamics, latency, mass/inertia, encoder bias, and nail/contact properties.

Keep each condition separate in plots/tables. Do not pool solver/timestep sensitivity with physical-domain robustness, and do not retrain on a stress condition after looking at test results unless it is declared as a new experiment generation.

## Measurements needed before any broader DR campaign

These are blockers for numerical ranges, not optional polish:

1. **Reset feasibility map:** realized head/nail geometry, waypoint error, limit margin, action saturation, contact order, strike success, and impulse statistics over nominal/interior/boundary joint-reset samples with the new action space.
2. **Fixture and nail placement:** repeated physical measurements of nail/block pose in the robot frame, including the complete reset/replacement procedure. A CAD tolerance is not a distribution unless the setup actually realizes it.
3. **Nail/contact identification:** physical drop/impact traces sufficient to identify or bound slide resistance and contact compliance. MuJoCo `solref`/`solimp` jointly specify constraint behavior; they should not be treated as independent material knobs. See the official [MuJoCo modeling documentation](https://mujoco.readthedocs.io/en/stable/modeling.html).
4. **Actuator plant:** per-joint step/chirp or other safe identification for effective stiffness, damping, friction, saturation, and inertia under the intended controller. Published peak torque/speed do not provide uncertainty bands.
5. **End-to-end latency:** timestamp command generation, transport, application, sensing, and observation availability. Observation-only integer delay in a 50 Hz policy is not a substitute for this measurement.
6. **Encoder/homing repeatability:** repeated zeroing and pose measurements to separate persistent per-episode bias from sample noise.
7. **Mass/COM audit:** weigh the complete hammer assembly and reconcile it with the separate `0.2 kg` head and `0.045 kg` fixture bodies before specifying any assembly uncertainty.

Peng et al. provide primary evidence that per-episode dynamics randomization, observation noise, and timing variation can matter, but their Fetch-specific parameter ranges are not Z1 evidence: [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://xbpeng.github.io/projects/SimToReal/SimToReal_2018.pdf). Tan et al. likewise show that actuator modeling and latency can dominate transfer and warn about the robustness/optimality tradeoff, supporting measurement before broad randomization rather than supplying Z1 bands: [Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://arxiv.org/abs/1804.10332). Unitree's [Z1 specifications](https://www.unitree.com/cn/z1/) establish nominal capabilities and controller support, but not the uncertainty distributions or end-to-end latency needed above.

## Causal safeguards to preregister

- Change one axis per first-order ablation; use a factorial interaction check only after each factor works alone.
- Match seeds, total environment transitions, initialization, checkpoints, evaluation episodes, and model-selection rule.
- Keep the controlled-drop calibration and diagnostic-threshold experiment distinct from the final constrained-controller experiment.
- Freeze the impulse threshold within a comparison. If it changes, start a new experiment generation and rerun all relevant controls.
- Evaluate every randomization-trained policy on nominal conditions and every nominal policy on the same held-out randomization panel.
- Report nail impulse/success and joint impulse/constraint exposure together; randomization must not improve one by silently sacrificing the other.
- Treat current actor observation corruption as an invariant unless its own ablation is the registered question.
- Use identical accepted randomization and curriculum settings for FIC and VIC. A distribution selected only for VIC cannot support a causal claim that VIC itself is better.
- Label hardware-calibrated DR, simulation task generalization, solver sensitivity, and numerical convergence as four different categories.

## Bottom line

There is enough local evidence to proceed with **one controlled randomization axis—joint reset pose—after a new-action-space qualification**, and enough methodological evidence to add **termination-pressure curriculum only as a diagnosed learning aid**. There is not yet evidence for numerical Z1 hammer-task ranges for contact, nail, actuator, latency, inertial, placement, or sensor-bias DR. Freezing those axes now is not a lack of robustness work; it is what preserves the interpretability of the fixed-impedance, impulse-CaT, and VIC experiments. Their inclusion becomes defensible after the listed measurements and belongs primarily in the final matched robustness campaign.
