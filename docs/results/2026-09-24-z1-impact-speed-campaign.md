# z1-impact-speed: paired Vega pilot

Status: prepared, not yet submitted. This is an exploratory simulation campaign, not hardware qualification.

## Frozen experiment

Question: does learned sampled joint impedance improve useful impacts while respecting the existing per-joint load limits, compared with fixed sampled impedance?

Six fresh runs: array indices 0/1 = fixed/variable seed 10, 2/3 = fixed/variable seed 11, 4/5 = fixed/variable seed 12. LEFT task `Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed`; no timed spline, no Cartesian impedance. External speed goals in [1.5, 2.2] m/s, speed-cost coefficient 0.8493676. All inherited safety, reward and guidance settings unchanged. Each run: 1,024 worlds, 50 PPO iterations, 24 steps/world/iteration (1,228,800 transitions), TensorBoard logging, no external logging service. Sampled controller physics 0.25 ms, torque update 2 ms, policy update 20 ms.

Resource ceiling: one 30-minute GPU preflight; six one-GPU training/evaluation jobs, maximum three concurrent, two-hour wall limit each. Maximum allocated GPU time 12.5 hours. No automatic retries, continuation, hyperparameter expansion or changes to old campaigns/jobs.

Preflight: exact source/asset hashes, GPU availability, reward/contact/setup gates, scoped regression tests, both gain modes' actual held torque/timing, two-iteration 32-world training, strict checkpoint reload, bounded evaluator smoke under both measurement protocols. Training array depends on successful preflight and checks its PASS marker. Each job gets a fresh Warp cache and refuses to overwrite run outputs.

Final checkpoint evaluation: seed 20260924, 128 initial episodes per goal at 1.5/1.85/2.2 m/s, both native and common-window protocols. Compare paired training seeds; examine all attempts, hit/miss coverage, speed error, geometric path error and approach coverage, per-joint load exceedance, and complete-window useful axial impulse. Missing impact windows remain censored, never interpreted as zero or silently discarded. Native and common-window outcomes must stay separately labeled. Fifty iterations may be insufficient for competence; no winner/safety claim from reward alone. Three training seeds are exploratory evidence, not strong statistical qualification.

## Reproduction and provenance

Harness: `scripts/vega_impact_campaign.py`; scheduler wrapper: `scripts/slurm/vega_impact_campaign.sbatch`. Immutable source snapshot comes from branch `z1-impact-speed`; per-file source hashes and archive digest will be retained in the isolated campaign directory. External assets are read-only and checked against the existing 32-file ledger. Runtime uses Vega's existing venv (Warp 1.14.0 versus local 1.13.0), with an isolated copy of existing test-only dependencies. GPU preflight is required because CPU tests do not establish CUDA compatibility.

Material passport: source=local committed implementation and launcher; reference=approved fixed/VIC paired pilot; license=repository/dependency licenses unchanged; verification=local launcher/routing tests and pending GPU gates; destination=isolated Vega campaign only; limitations=simulation, short exploratory training, no empirical result yet.

Next decisions after evidence: longer training if learning is still progressing; autonomous terminal-speed selector and three-start generalization only after speed-conditioned behavior and measurement coverage are understood. No such extension is submitted by this pilot.
