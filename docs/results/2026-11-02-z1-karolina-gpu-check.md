# Z1 Karolina GPU compatibility check — 2 November 2026

**The existing pipeline passed one GPU update/save/reload and the three-step
CPU evaluator binding check.** Job5322192 completed with exit0:0 on acn24 in
4min21s, 17:12:18–17:16:39UTC (18:12:18–18:16:39Berlin).

| Check | Observed result |
|---|---|
| Resources | One NVIDIA A100-SXM4-40GB, four CPUs; 20-minute scheduler limit |
| Source | Clean `z1-safe-striking` checkout at `722f67ffd5c79a2417d8813975882f5dc2dfb3af`, verified before and after |
| Selected recipe | Reward037 cell0, initial position std0.3, seed2026110201; all three task rewards ON |
| Actual learning | One PPO update, 1024 worlds ×24 policy steps =24,576 rollout transitions |
| Saved clock | model0, completed_updates1, common_step_counter24 |
| GPU reload | Actor/critic parameters and buffers were overwritten, reloaded and compared with saved tensors; environment clock restored |
| CPU binding | 32 matched contexts, maximum three policy steps; `PASS_BINDING_ONLY` |
| Checkpoint identity | Actual file SHA matches the GPU result and CPU policy record |
| Scientific trials | Zero; no full-bank policy evaluation or training campaign |

The GPU result is `PASS_UPDATE_SAVE_FULL_RELOAD_ONLY`. Its checkpoint SHA256 is
`81533c16c64badde87867771c5e66a065e905400a21719dee5c97ebf06de1a14`.
PyTorch reports2.12.0+cu130; the allocated GPU was visible as cuda0.

The existing checker/evaluator and unchanged036 Warp bootstrap were used.
No rewards, thresholds, constraints, controller, reset bank or scientific
settings were changed. No retry, additional campaign or goal was created.

The CPU process deliberately masked CUDA. Warp's no-CUDA message is retained
in stderr; CPU execution subsequently completed successfully.

## Scope

This establishes technical execution for the selected base recipe. It does not
measure learned task performance, certify safety, establish cross-cluster
trajectory equivalence, test optimizer resume, or qualify every optional
configuration. The height-randomized and gSDE paths were not exercised by this
one-job check. Actor/critic reload is distinct from optimizer restoration.

## Preserved evidence

Karolina:
`/scratch/project/eu-26-66/nikhil/z1/z1-safe-striking/checks/gpu-20261102T171017Z`.

Local analysis records:
`.superpowers/sdd/2026-11-02-z1-karolina-gpu-check/`.

Independent SSD:
`/Volumes/NikeraneSSD/Z1-campaign-backups/2026-11-02-karolina-gpu-check/`.

All35 regular evidence files reconcile between remote hashes, archive members
and the independently hashed SSD copy. The archive is1,545,664bytes; its full
SHA256 is recorded in `backup.json`. Rebuildable Warp/CUDA caches were excluded from the
archive and remain on Karolina. Logs, configuration, checkpoint and raw
technical outputs are preserved.

See the [migration review](2026-11-02-z1-karolina-migration-review.md) and
[workspace handoff](../handoffs/2026-11-02-z1-karolina-code-handoff.md).
