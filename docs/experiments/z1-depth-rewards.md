# Z1 depth rewards: retained code and current workflow

Current update: the nine-run all-on reward/exploration implementation is described in [the current proposal](z1-reward-exploration-proposal.md). It delegates to the existing trainer and has not been launched. The four-arm031code below remains retained for provenance; it is not the current experiment selection.

The rewards remain available. The custom 031 campaign framework is retired. The user cleared the training goal. This cleanup launches no jobs. The new study still has 0 trained policies and 0 evaluated cases.

## Reward settings

| Arm | Bounded depth progress | Once-only 30 mm completion |
|---|---|---|
| A0 | Off | Off |
| A1 | On | Off |
| A2 | Off | On |
| A3 | On | On |

Select the arm through the existing 031 configuration/adapter. It sets the reward weights and ledger switches consistently. Completion becomes payable once at the genuine 200 ms contact horizon. Do not enable the legacy repeating completion term.

Existing constraints and control stay unchanged: velocity/load/impulse checks, NaN guard, variable impedance, the common 200 ms aftermath and split 50/200 ms payout. The baseline uses the current critic and ordinary learned global Gaussian exploration, with initial position/gain std 0.3 and speed std 1. No recovery motion or reward was added.

## Existing active code

- `src/tasks/hammer/config/z1/spline_depth_reward031_cfg.py`: four opt-in settings.
- `src/tasks/hammer/mdp/depth_credit031.py` and `depth_reward031.py`: bounded ledger, observer and reward readers.
- `src/tasks/hammer/rl/depth_reward031_contract.py`: checkpoint/configuration identity, used by the existing contract hook.
- `src/tasks/hammer/rl/depth_reward031_trace.py`: reward-accounting diagnostics.
- `scripts/z1_depth_reward_training031.py`: twelve-cell configuration and existing initial-state/checkpoint hooks.
- `scripts/train.py`: existing training runner. Shared 020/021/pilot helpers remain intact.

For a future authorized run, use `build_train_config(index)`, `selected_checkpoint_saves(...)` and `matched_initial_state_gate(...)` around the existing `run_train`. Index is seed-block × 4 plus arm index. A0 supplies that seed's initial actor/critic/normalizer reference for A1–A3. Keep the 500-update target and checkpoints 99/499 (100/500 updates). The adapter is a Python API, not a standalone CLI. This cleanup deliberately adds no new launcher.

No short smoke entry is provided here. The existing matched-study hook requires 500 updates, so do not pass a one-update run through it or label a smoke as a completed study.

Reuse the existing 32-context reset bank/evaluator for future evaluation. The 031 checkpoint/configuration seam still needs validation before use; this cleanup does not implement or qualify a replacement evaluator. Keep productive strikes, full depth, whole-episode velocity/load and contact impulse outcomes separate. Contact safety without contact is UNKNOWN. Seeds are replication units.

## Retired code and preserved evidence

`python -m scripts.z1_depth_campaign031` now prints a retirement notice only. Its former bootstrap, manifests, import/compiler checks, admission chain, clock/controller, claim/grant machinery and proof/join pipeline are no longer active code. Old plans/specs and command recipes are historical records, not instructions for future jobs.

The exact former package, nine orchestration tests and superseded navigation documents are preserved in:

- Local archive: `.superpowers/sdd/2026-11-01-z1-simplification-review/retired-campaign031.tar`.
- Independent SSD archive: `/Volumes/NikeraneSSD/Z1-campaign-backups/2026-11-01-depth-reward031/retired-campaign031-cleanup.tar`.
- Archive SHA256: `6394a68b62507a74702e2c4c344dc7fcf605c3a859b0802c6ba4eccc3f3790a2`.
- Original tracked source: Git revision `10c5bfe79c59ecfda16dfffae420d5e278d00977`.

The local worktree archive is not the sole preserved copy. Per-member verification is recorded in `cleanup-before.json` and `retirement-receipt.json` beside it. Frozen historical source remains independently readable; retrospective archive tools must use their own frozen version.

Historical outputs, failed runs, frozen packages, reset banks and backups remain untouched. Historical 029 combined closure jobs remain FAILED without closure.json. No cleanup or test result is a safety claim.
