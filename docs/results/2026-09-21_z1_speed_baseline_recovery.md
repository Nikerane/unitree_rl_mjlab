# Impact-speed baseline recovery and first replay attempt

## Result

The exact 12-output quintic-M20 comparator was recovered and hash-verified locally and on Vega. Initial job `45613192` failed before Python because the launcher omitted Vega's Git module. After explicit user approval, retry `45613460` completed in **1m35s**, exit `0:0`: **128/128 hits, successes and finalized first events**, zero timeouts and zero learning updates. The fix added `module load git` before the unchanged identity checks, following existing working Vega launchers.

## Fresh GPU replay measurements

| Measurement | Median | Observed range |
|---|---:|---:|
| First-event axial impact speed | 2.21644 m/s | 2.13541–2.31352 m/s |
| Delivered first-event impulse | 0.359681 N·s | 0.338043–0.377335 N·s |
| Native phase-indexed tracking RMS, all sampled steps | 19.6605 mm | 19.5326–19.8329 mm |
| Maximum sampled tracking error per episode | 28.1635 mm | 28.0320–28.3363 mm |
| Positive weighted reward terms, dt-scaled through finalization | 2.83123 | 2.81204–2.85193 |
| Raw environment return through finalization | 2.79704 | 2.77865–2.81709 |

All 128 episodes exceeded the provisional J6 impulse cap; largest utilization was **1.05856**. No other joint exceeded its impulse cap in this replay. Maximum combined CaT delta was0.5 in every episode; this is not an isolated impulse-only dose measurement. Soft-CaT is not a hard clamp and no hardware safety claim follows.

There was **one unique full qpos/qvel reset**, but128 unique reset hashes including actor/critic observations. Thus these are repeated physical fixed-start replays, not128 different physical starting conditions or independent trained policies. The speed spread does not prove commanded-speed responsiveness or establish a safe/reachable continuous interval.

Applying the predeclared coefficient-sweep rule to the positive reward-term median yields candidate costs **0.2831225,0.8493676,2.8312253** in event-return units. These are proposed screening values, not empirically tuned weights; the return excludes CatPPO's later training-time return transformation. No speed interval or E1 launch has been approved.

Local evidence: [episodes](assets/2026-09-21_speed_baseline_replay/results/episodes.json), [summary](assets/2026-09-21_speed_baseline_replay/results/summary.json), [reset arrays](assets/2026-09-21_speed_baseline_replay/results/resets.npz), [step traces](assets/2026-09-21_speed_baseline_replay/results/traces.json).

Remote/local matching SHA-256 values:

- episodes.json: `1c45f26af362ac7aadbea6e762cefdeeccdc92457438e79f036d17a7c1821b26`
- resets.npz: `cf872b23fdc8d4592e3abd55b8fe22ac04c9c2553b497bada79f666988981168`
- summary.json: `7b67886505af26a1c51c1a58ba625128f8235e5ae9375480aaf90864168d190e`
- traces.json: `dd29c166999db6be217f2e1b4a6251bb734c1aea9b724cfcedd877b9639bc784`

## Verified identity

- Training job `45513585`: COMPLETED, exit `0:0`, 9m47s; fresh seed 2, 250 iterations, 4096 worlds, 24-step rollouts.
- Task: `Unitree-Z1-Hammer-Quintic-M20-Pilot`.
- Source: `3ca89a297a58d8e59d74c4797caeb8629499f1aa`.
- Assets: `b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Checkpoint SHA-256: `fb526aeca10c47d10def0e5f32ee545e56a180378271e65b8a4972ea5e73525d`.
- Local checkpoint: `/Users/nikerane/repos/unitree_rl_mjlab/docs/results/quintic-m20-pilot/iter250/model_249.pt`.
- Remote checkpoint: `/cephhome/eunikhilr/campaigns/z1-quintic-m20-pilot/runs/3ca89a2-seed2-250iter-retry1/logs/rsl_rl/z1_hammer/2026-09-20_15-14-39_quintic_m20_seed2_baseline250/model_249.pt`.
- Login preflight verified clean tracked baseline source and assets. Local asset checkout has unrelated untracked documentation, not modified here.
- Local and remote package versions: mjlab1.4.0, MuJoCo3.8.1, mujoco-warp3.8.1, PyTorch2.12.0, rsl-rl-lib5.2.0.

## Replay preparation

An evaluation-only artifact preserves the training configuration (`play=False`), native 1000-step horizon and controller/rewards. It uses one world with autoreset disabled, records first-finalized-event speed/impulse, native reward-time phase tracking, joint-load/CaT metrics, dt-scaled positive reward terms separately from raw returned reward, reset arrays/hashes and complete step reward traces. No extra sensor, production edit or training framework was added.

Two local CPU episodes passed before submission; output `/tmp/z1-baseline-replay-final-smoke-20260921`. The smoke imported the current isolated `cc46d99` worktree with the unchanged baseline task, not the remote baseline checkout; the script's declared source constants do not themselves verify runtime identity. Independent read-only review found no blocking issue for the bounded replay and prompted clearer raw-return wording. The launch script checked exact source/asset identity, but failed on its first Git command.

Remote campaign: `/cephhome/eunikhilr/campaigns/z1-speed-baseline-20260921.GtvAqJ`.
Replay script SHA-256: `07db8ae4e04ee5d7c1997009b37d549e21fca777efac34d04d3501d053a149d8`.
Attempt: 128 sequential seeds beginning2026092101, one GPU,4 CPUs,16GiB,20-minute limit, zero learning updates.
Initial scheduler result: FAILED, elapsed00:00:00, exit1:0; stderr `git: command not found`. No simulation executed in that attempt. The approved retry loaded module Git2.45.1 from the shared module filesystem, passed provenance checks and produced the results above; original failed-attempt logs and launcher remain on Vega.

## Decision and next step

Recovery and128-episode replay succeeded. Next review an exploratory speed interval against these measurements and existing joint-load exceedance, and freeze the remaining qualification run record. The evidence does not determine a uniquely justified interval or prove slower commanded speeds are reachable; those remain experimental choices. No E1/E2 training is authorized.

Ruling retained: use a bounded standalone evaluator instead of the old eight-episode playback script because playback removes the native timeout and reports a different tracking metric. Cost if wrong: evaluator instrumentation could bias evidence; the CPU smoke and independent review reduce but do not eliminate that risk. Repeated fixed-start replays are not independent training seeds or a robustness distribution.
