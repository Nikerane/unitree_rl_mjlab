# Task 3 implementation report

Status: DONE_WITH_CONCERNS (software behavior verified; no empirical generalization claim).

## Changes

- Added a shared quintic config builder with explicit `left` / `three_starts` route mode and finite guidance weight/width validation. Registered `Unitree-Z1-Hammer-Quintic-ThreeStarts-ImpactSpeed` separately. The left registration retains its fixed `-1` route, persistent `0.2` imitation weight, `0.05 m` width, `0.02 m` vertical follow-through, and inherited CaT caps/dose.
- Added `SingleStrikeReference.diagnose_current_height(head_w) -> (phase, waypoint_w, error_m)`. Input is a `(num_envs, 3)` world-space head tensor after anchoring. Outputs have shapes `(num_envs,)`, `(num_envs, 3)`, `(num_envs,)`. It computes geometry from current height and never changes the monotone policy phase latch, including during rebound and off-curve motion.
- Narrowed sampled VIC metadata validation to accept either left fixed route or independently sampled three-start route, and valid configurable persistent guidance. The serialized identity includes route table, route mode, path, guidance weight, and width; per-episode draws are excluded. Existing physics/load checks remain intact.

## TDD evidence

- RED config: `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_spline_impact_config.py -q` -> `9 failed` (missing three-start registration and builder). Initial import error was corrected in the test before the recorded 9-failure RED run.
- GREEN config: same command -> `9 passed, 35 warnings`.
- RED diagnostic: `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_quintic_reference.py -q` -> `2 failed, 4 passed`, both missing `diagnose_current_height`.
- GREEN diagnostic: same command -> `6 passed, 35 warnings`.
- RED metadata: `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_impact_speed_contract.py::test_three_start_sampled_vic_identity_tracks_route_and_guidance_not_draws -q` -> `1 failed`, `ValueError: sampled VIC requires the left-only route`.
- GREEN metadata: same command -> `1 passed, 35 warnings`.
- Final focused integration command: `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_quintic_reference.py tests/test_imitation_reward.py tests/test_spline_impact_config.py tests/test_configs.py tests/test_impact_speed_contract.py -q` -> `213 passed, 54 warnings in 19.66s` (exit 0). An intermediate run had 1 failure from the task registration allowlist in `test_configs.py`; it was updated and the same set reran green.

The warnings are dependency deprecations from TorchScript and ONNX export, not test failures.

## Geometry and observation audit

A 24-environment live reset with fixed seed included all three signs. Each selected the corresponding qualified reset joint row; realized head X positions were 20 mm apart. Every quintic began at its realized head, crossed the nail, had a nearly vertical terminal tangent, and continued 20 mm vertically. A partial route/joint reset changed only selected environments. The registered actor observation shape remained 42 and action width 13.

Paired start observations at a shared nail position differed across every route pair. A second paired-state check used the same current head and nail position with three start anchors: `strike_phase` matched, while the signed X component of `strike_ref_error` differed in route order. This establishes distinguishability for those supplied precontact states, but not policy competence or generalization. No observation field was added.

## Self-review and limits

Reviewed scoped diff and ran `git diff --check` without whitespace errors. Config construction and geometry are software verified. No policy training, physical qualification, or reward calibration was attempted. The live tests use explicit test-only speed bounds and cost because the experimental task intentionally leaves them unset. The equal-probability route mechanism is the existing independent `torch.randint(0, 3)` per resetting environment; exact batch counts are intentionally unconstrained.

## Review fix 1: full left-config snapshot

The review identified that the original tests covered selected invariants but did not compare the whole left configuration to a prechange artifact. I added a full canonical digest check for both train and play defaults. The digest traverses every dataclass field, preserves mapping order, and normalizes arrays, tensors and callables. It compares the registered left `ImpactSpeed` config to SHA-256 values generated from committed pre-Task-3 source at `b60ddc29c0de46eaedf15781759c0c958295cf05` using Python 3.10. The same left config was present at Task-3 base `3b91419`. Baseline source was extracted to a temporary sibling checkout for read-only config construction; no original source or assets were edited. The generated train/play digests matched the current left config exactly. The test also mutates route selection, imitation weight, physics timestep, and impulse CaT dose individually and confirms each changes the full digest.

RED (placeholder digests before recording baseline): `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_spline_impact_config.py::test_left_default_full_config_matches_prechange_snapshot -q` -> `2 failed, 35 warnings`; observed train `def45332b8d4bbc40b0a84195a97ead911991a30a26194f9094a760546626c93`, play `d024b1d5e1dd971d942756e6b8c2f44bafbffc81f88caf49f803541b77e7c254` against placeholders. GREEN: `PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_spline_impact_config.py -q` -> `14 passed, 35 warnings in 6.18s` (exit 0). The warnings remain upstream TorchScript deprecations.
