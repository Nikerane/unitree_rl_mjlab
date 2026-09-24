# Startup torque and timestep diagnostic

2026-09-22. Six open-loop startup replays; no learning or production changes. This is not checkpoint qualification at an altered timestep.

## Protocol

Replayed the first three saved 20-ms commands from each fresh dose-policy at the middle speed request (1.85 m/s), across 128 worlds, at physics dt=2,1,.5 ms. Position and stiffness actions are frozen from the normal startup trace; the policy is not re-run at altered timesteps. Native action processing, gain writes, force limits and physics are retained. Each condition covers exactly 60 ms. Initial full qpos/qvel hashes agree across timestep arms, and initial arm positions/velocities match the source trace exactly. Compiled integrator enum=3 (implicitfast); arm force ranges remain ±30 N·m except J2 ±60.

Recorded pre-step joint position/velocity, target/gains, reconstructed requested PD torque and native actuator force retained after the physics step (computed from its pre-integration state), plus post-step velocities. Native force equals clipped reconstructed J6 demand within .000112 N·m. No accepted hammer-nail contact occurs in these traces.

## Results

| Policy | Physics dt | Maximum J6 speed, all worlds | World-zero saturated samples | World-zero velocity sign changes |
|---|---:|---:|---:|---:|
| dose .2 | 2 ms | 4.609762 rad/s | 30/30 | 29/29 adjacent pairs |
| dose .2 | 1 ms | 2.504313 | 60/60 | 59/59 |
| dose .2 | .5 ms | 1.308782 | 49/120 | 48/119 |
| dose .3 | 2 ms | 4.580195 | 30/30 | 29/29 |
| dose .3 | 1 ms | 2.488044 | 60/60 | 59/59 |
| dose .3 | .5 ms | 1.300222 | 31/120 | 30/119 |

Both policies initially request magnitude44.131660 N·m, while the actual force is exactly30 N·m. Signs are opposite between policies. At2ms, velocity alternates sign every physics sample and torque is saturated throughout the60ms window. The unchanged2ms replay reproduces the earlier startup world-zero six-joint velocity trace to maximum absolute differences1.073e-6 (.2) and1.193e-7 (.3) rad/s across30samples.

## Interpretation and limits

Strong timestep dependence and sample-to-sample alternating velocity under saturation support numerical chattering in the actuator/integration response, rather than an unavoidable physical speed required for impact. This agrees with the [source review](../../research/reward-design/2026-09-22_startup_online_synthesis.md): saturated native actuators contribute zero local actuator-velocity derivative to the implicit update.

The experiment does not independently isolate that derivative branch from other coupled dynamics; no simulator bug or exclusive cause is proven. .5ms still contains sign reversals, and the peak continues to change substantially with refinement: **convergence has not been established**. Being below3.1415rad/s at1ms is not a sufficient criterion for adopting1ms. These are precontact frozen-command replays, not full-strike performance, contact impulse, or hardware-safety evidence. Gain calibration remains the separate [open question](../../research/reward-design/2026-09-22_z1_gain_range_provenance.md).

## Recommended continuation

Keep original training and checkpoint semantics unchanged. Establish finer-step convergence for this short replay, then separately test lower, justified gains with damping considered jointly; do not simultaneously change gains and timestep. A permanent change requires re-evaluation of full strikes, tracking, delivered impulse and load metrics. Do not merely increase torque limits, add arbitrary inertia, or retrain the policy to compensate for this transient.

## Reproducibility

Vega job45732348 completed exit0 in23s. Initial job45732308 failed during import because Git was not loaded; adding the existing cluster Git module resolved the launcher issue, with no simulation changes. No active jobs remained at completion.

Artifacts: [substeps and initial-state identities](assets/2026-09-22_torque_timestep/). Scratch scripts: `.superpowers/sdd/2026-09-21-vega-speed-qualification/torque_timestep_probe.py` and `analyze_torque_timestep.py`. Analyzer passed matched hashes, sample counts, no-contact check and force-clamp reconstruction for all six arms. A separate comparison passed the2ms baseline reproduction tolerance1e-4rad/s. No checkpoint loading restrictions were bypassed: the harness replays saved actions without loading a policy or asserting altered-timestep qualification.
