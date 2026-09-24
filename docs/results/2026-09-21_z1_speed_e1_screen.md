# E1 three-coefficient screening campaign

User approved the bounded screening campaign after the successful GPU preflight and matched-baseline checks. This record is not a completed result until scheduler and evaluation artifacts are recorded below.

## Frozen treatment

- Source: `cc46d9998a3c9cd5d8813e89b57e44248353e03e`, separate clean detached Vega checkout; assets `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Campaign directory: `/cephhome/eunikhilr/campaigns/z1-speed-e1-screen-20260921.QKg3yq`.
- Task: `Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed`, external uniform speed goals[1.5,2.2]m/s, unchanged quintic-M20 geometry and20mm followthrough.
- Three fresh actors, seed2 each,1024worlds,24steps/iteration,50iterations,1,228,800 transitions per candidate. No baseline/smoke checkpoint initialization; no resume.
- Coefficients: arm0=0.2831225; arm1=0.8493676; arm2=2.8312253. All original reward weights, controller mappings and safety settings held fixed.
- Save every25iterations and final model49. Evaluate finalcheckpoint at1.5,1.85,2.2m/s,128parallelfirstepisodes/goal,seed2026092101.
- OneGPU,4CPUs,24GiB,2hour hard limit per arm including evaluation; no automatic retries, requeue or budget extension.

## Predeclared analysis

Filter all comparisons to episode0. Verify matched reset hashes and CPU/CUDA post-reset RNG against the banked parallel baseline before applying relative gates. At each goal require at least116/128hits, median absolute speed error≤0.07m/s and median native whole-episode tracking RMS≤21.622820mm. Primary paired high-minus-low speed contrast must have95%bootstrap lower bound>0 (2000resamples,seed2026092101). Report invalid/incomplete events, misses, timing, impulse, per-joint loads/exceedance and CaT; no claim of hardware safety or optimal impedance.

Select the lowest coefficient passing every gate, then propose the held-out-seed confirmation. If none passes, report all arms and stop;50iterations is a feasibility screen, not a scientific rejection of the architecture. E2 autonomy,250iteration extensions and three-curve training are not authorized by this launch.

## Evidence and status

Preflight evidence: [GPU preflight](2026-09-21_z1_speed_gpu_preflight.md). Training command uses the verified Tyro tuple form `--speed-bounds '(1.5,2.2)'`; evaluation uses two separate floats. The script enforces source/assets and refuses reuse of an existing arm directory. Submission and outcomes will be appended after verification.

Submitted once as array **45616754_[0–2]**, after all three full argument sets parsed and an independent read-only launch check approved the script. Initial scheduler snapshot: arm0running ongn26,arms1–2pending resources. No completed performance result yet.

## Completed result — no candidate qualifies

All three arms completed50iterations and evaluation, exit0:0, without retries: arm0 in2m45s,arm1 in2m40s,arm2 in3m07s (scheduler elapsed including startup/evaluation). Total training:3fresh policies,3,686,400world transitions. No extension or autonomous-speed training was launched.

All nine requested-speed conditions had **128/128 first-episode hits, productive finalized events and valid speed measurements**. All initial reset hashes and CPU/CUDA post-reset RNG hashes matched the frozen parallel comparator. These repeated fixed-start/observation-noise trials are not independent training seeds or physical-start generalization.

| Cost coefficient | Requested speed (m/s) | Median achieved (m/s) | Median absolute error (m/s) | Tracking RMS median (mm) | Delivered impulse median (N·s) |
|---|---:|---:|---:|---:|---:|
| 0.2831225 | 1.50 | 1.706 | 0.227 | 14.94 | 0.258 |
| 0.2831225 | 1.85 | 1.822 | **0.067** | 14.46 | 0.327 |
| 0.2831225 | 2.20 | 1.483 | 0.717 | 15.09 | 0.381 |
| 0.8493676 | 1.50 | 1.322 | 0.249 | 9.94 | 0.356 |
| 0.8493676 | 1.85 | 1.862 | 0.122 | 11.04 | 0.414 |
| 0.8493676 | 2.20 | 2.394 | 0.196 | 14.00 | 0.403 |
| 2.8312253 | 1.50 | 1.515 | 0.144 | 4.84 | 0.311 |
| 2.8312253 | 1.85 | 2.104 | 0.256 | 7.73 | 0.382 |
| 2.8312253 | 2.20 | 2.395 | 0.198 | 11.62 | 0.430 |

The required median absolute error is≤0.07m/s **at every goal**. Only the marked middle-goal condition passes; no policy passes all three. A median achieved speed close to target does not imply low median absolute error: individual trajectories can scatter above/below target. All tracking medians satisfy21.622820mm, but this is a whole-episode sampled metric; varying contact timing/episode length prevents interpreting it as a controlled pre-contact improvement alone.

Primary paired high-minus-low mean speed contrasts (95%paired bootstrap intervals):

- Arm0: **−0.196m/s**, [−0.250,−0.141] — opposite response.
- Arm1: **+1.051m/s**, [+0.996,+1.109] — responsive, insufficiently accurate.
- Arm2: **+0.893m/s**, [+0.859,+0.926] — responsive, insufficiently accurate.

Every contrast has128common valid pairs. These intervals describe this one policy instance's paired evaluation variability, not training-seed uncertainty. All costs retain `selected_arm=null`; no held-out promotion is justified.

### Joint-load concern

Joint impulses are **N·m·s**, distinct from delivered linear impulse **N·s**. Reported load maxima are episode maxima of the native50ms sliding-window reaction-impulse metric, not total episode-integrated impulse.

Matched baseline J6 maximum was1.058×cap (128/128slightly over). Arm1/high reached **0.0732337N·m·s =4.465×the0.0164cap**, versus baseline maximum0.0173443N·m·s. New J1,J2,J5 exceedances also occurred. This is a worsening tail, not a uniform increase at every episode: arm1/high J6 exceeded in48/128episodes and its median utilization was0.843. Strongest-cost arm2 has J6 median utilizations1.733,1.519,1.328 across the three goals, with115/128,88/128,99/128over-cap episodes. Soft-CaT remained active and unchanged; it is not a hard clamp. No safety-improvement claim is supported.

### Complete reporting and artifacts

Evaluation emitted later-autoreset rows while waiting for slower worlds. Excluded later rows per low/mid/high condition: arm0=112/123/17;arm1=121/108/124;arm2=127/108/89. Only episode0 enters gates. The [machine-readable analysis](assets/2026-09-21_speed_e1_screen/analysis.json) includes95th-percentile absolute errors, timing, costs, caps/exceedances, conditional contrasts and these exclusions. All attempted arms and failures are retained.

Final checkpoint hashes (downloaded copies match evaluator-recorded hashes; all model/optimizer tensors finite,iteration49,zero active selector samples as expected in external mode):

- Arm0: `32d568d787814d068428cb8369f1dbd878387e3a1583cfe0bd18bef8c7ff895b`
- Arm1: `3840578cdfb4613a6b9869d6de9ec60b41e1e9d8325f969d839ac1b4f9110d7f`
- Arm2: `7b4359f571655973fc557fd9d64a4d5f5db1a66ad7f573b7c6f70f193741eb62`

Local artifacts under `docs/results/assets/2026-09-21_speed_e1_screen/arm{0,1,2}/`: finalcheckpoint,traininglog,evaluation summary and episode rows. Remote paths remain in the frozen campaign directory above. Independent review checked raw rows, gate aggregation and J6units/conversion, agreeing that no candidate qualifies.

## Next decision

Stop at the declared screen boundary. Two policies learned meaningful goal responsiveness, but precision and load tails need investigation before E2. Inspect time-aligned trajectories, speed-error scatter and per-joint load events before proposing one bounded extension or a treatment change. Do not silently retune safety, loosen accuracy gates or continue to250iterations. Higher impulse with lower onset speed in some conditions is further reason to retain impulse as the actual objective rather than assume speed alone determines it.
