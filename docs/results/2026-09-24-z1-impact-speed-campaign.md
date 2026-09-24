# z1-impact-speed: paired Vega pilot

Status: submitted 2026-09-24, preflight running, training array waiting for its successful completion. This is an exploratory simulation campaign, not hardware qualification.

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

## Launch evidence

- Branch: `z1-impact-speed` (no `codex/` prefix).
- Source commit: `e3fda7e639a47df3be039dae4f59045cbbb30dc0`.
- Archive SHA-256: `03f8fddd6affbf52f7345673fab8c0ebe301e85156eacc87a4ef8d7e79d86e5b`, matched on Vega before extraction.
- Isolated remote directory: `/cephhome/eunikhilr/campaigns/z1-impact-speed-20260924.Fk9TSe`.
- Remote `source.sha256` covers the extracted source; `runtime-packages.txt` records installed runtime distribution versions. No Git repository is fabricated inside the archive; runner Git logging can skip that path. This record and archive/manifest supply source identity instead.
- Existing external asset revision `b58ccd2f81fd246f27c1e8d88cf86484cd888703`; all 32 asset file hashes passed before launch and again inside the preflight.
- Local launcher/routing verification: **12 passed**, 35 installed TorchScript deprecation warnings, 6.42 s. Shell syntax and `git diff --check` passed. Initial test-first collection failed for the absent launcher, before implementation.
- Independent bounded launcher review: no P1/P2 issue. GPU capacity and full-size train-to-evaluation memory remain empirical checks; the small smoke cannot establish full-run memory headroom.
- Preflight job **45885607**, submitted 19:08:09 Vega scheduler time; observed RUNNING on `gn29`, NVIDIA A100-SXM4-40GB, Torch `2.12.0+cu130`, one GPU, 30-minute limit, requeue disabled.
- Training array **45885608**, indices `0-5%3`; observed PENDING (Dependency), `afterok:45885607`, two-hour limit, one GPU/task, requeue disabled, invalid dependency cancellation enabled. No training result exists yet.

Actual scheduler submission arguments (campaign path abbreviated as C here only):

```bash
sbatch --parsable --time=00:30:00 --output="$C/logs/preflight-%j.log" "$C/source/scripts/slurm/vega_impact_campaign.sbatch" "$C" preflight
sbatch --parsable --array=0-5%3 --dependency=afterok:45885607 --kill-on-invalid-dep=yes --output="$C/logs/train-%A_%a.log" "$C/source/scripts/slurm/vega_impact_campaign.sbatch" "$C" train
```

Only these new job IDs were queried. The original source and old jobs 45860324/45860325 remain untouched. This post-submission record is not part of the already frozen source archive.
