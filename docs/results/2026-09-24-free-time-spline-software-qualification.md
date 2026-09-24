# Free-time spline impact: software qualification only (2026-09-24)

## Scope and provenance

This is a no-learning CPU integration record for the implementation through `aeb8fe5` plus the Task 5 test commit. It is not a policy or hardware result. The implementation is an isolated snapshot on `codex/free-time-spline-impact`, copied from experimental worktree `/Users/nikerane/.codex/worktrees/z1-impact-speed/unitree_rl_mjlab` at `cc46d9998a3c9cd5d8813e89b57e44248353e03e`. The [source manifest](2026-09-24-free-time-spline-source-manifest.json) records the exact tracked patch SHA-256 `a16bad2dbea42b271dc84a68b0fe63adbcf634999f477b3d052152ee2561ed2f`, 673 untracked source paths, 384 included paths, 289 excluded historical result assets, and read-only external asset hashes. The source-before/source-after inventories matched. The external asset ledger covers 32 files; its SHA-256 is `571bec311767567475176530012cf71c478d93ba9f27594f80aa551ba952e453`. The original source and jobs `45860324`/`45860325` were not changed or queried.

Development decisions retained from the scratch ledger: the already isolated `a370` checkout was reused; the missing sibling simulator-assets symlink was recreated read-only to match the source; the focused regression scope was approved instead of a repository-wide suite; sampled gain calibration remains independent of native nominal gains; the VIC checkpoint contract fingerprints homogeneous *live* model loads and the live clock; the three-route/guidance treatment is serialized; load evidence uses `max(current rolling Lambda, current control-step peak)` including retained pre-onset history, so peak-held exceedance duration is not instantaneous rolling-Lambda duration. These decisions and their limitations govern the interpretation below.

## Execution

macOS Darwin CPU; Python 3.10.20, pytest 9.0.3, PyTorch 2.12.0, MuJoCo 3.8.1, MuJoCo Warp 3.8.1, Warp 1.13.0, mjlab 1.4.0 (`importlib.metadata.version("mjlab")`; its module has no `__version__` attribute). Simulator construction and live stepping succeeded on CPU. No CUDA, training, cluster, or hardware execution occurred.

| Check | Result | Software evidence |
|---|---:|---|
| New live sampled smoke | 2/2 gain modes | Twelve CPU worlds in the three-start task showed actual R−/R0/R+ reset joint poses, finite 42D actor/critic observations, finite applied gains, a 13D synthetic action, one asynchronous timeout/reset, and an 80-sample pre-reset evaluation record. Nonreset route assignments stayed unchanged. |
| Native and sampled diagnostic parity | Pass for tested finite sequences in native, sampled fixed, and sampled variable modes | With evaluation metrics absent/present, three staggered world resets per mode produced exactly equal initial and subsequent actor/critic observations, rewards, done flags, speed goals, robot joint positions/velocities, route signs, speed-pending state, pre-reset applied gains, sampled held torques, and PyTorch RNG state. The same seeds and synthetic actions were used under the native measurement protocol. This does not cover numerical failure paths. |
| Deterministic intervention | Pass | An untrained checkpoint fixture plus deterministic test actor gave identical first actions across 1.5/1.85/2.2 m/s goal conditions; second motor actions were respectively −1/0/+1 (floating tolerance), with exactly one speed selection per world per condition. The harness still reported physical responsiveness `false`. |
| Existing checkpoint/export and masked-policy tests | Pass | Covered by the focused regression, including checkpoint round-trip/transfer, masked distribution, PT/ONNX parity, and native/common-window evaluation. Test-generated fixtures are untrained. |
| Optional trained fixed checkpoint | Not exercised | No compatible checkpoint was supplied in this implementation checkout. Historical bulky checkpoints were intentionally excluded from the source snapshot; no unrelated directories or original jobs were searched. Fixed replay construction is covered by synthetic checkpoint tests, without calling it trained VIC performance. |

The exact approved 18-file command was:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_sampled_pd500.py tests/test_variable_impedance.py tests/test_impact_speed.py tests/test_impact_speed_policy.py tests/test_impact_speed_contract.py tests/test_impact_speed_wrapper.py tests/test_impact_speed_train_routing.py tests/test_impact_speed_evaluation.py tests/test_quintic_reference.py tests/test_imitation_reward.py tests/test_spline_impact_config.py tests/test_strike_evaluation.py tests/test_first_strike_event.py tests/test_impulse_bound.py tests/test_impulse_constraint.py tests/test_cat_soft_hook.py tests/test_cat_ppo_gae.py tests/test_cat_storage.py -q
```

Exit 0: **340 passed, 0 failed, 0 skipped, 70 warnings in 55.59 s** at the original Task 5 commit. The new integration file exited 0 after the review fix: **6 passed, 0 failed, 0 skipped, 37 warnings in 13.63 s**. The original four-case file had exited 0 with 4 passed, 37 warnings in 7.38 s. A combined run before the final pose assertion exited through a shell bookkeeping error after pytest had reported **344 passed, 72 warnings**; the exact approved command and final new-file runs have clean process exit codes. Warnings were installed TorchScript/ONNX deprecations and one Torch profiler warning; no new functional failure or pre-existing failure appeared in the focused scope. The full repository suite was not run. The 18-file regression was not repeated after this test-and-doc-only review fix.

## Qualification boundary

| Software-testable and checked | Requires later empirical work |
|---|---|
| Applied sampled gain authority; 2 ms held-torque timing under 0.25 ms physics and 20 ms policy steps; asynchronous reset isolation; fixed/VIC checkpoint identities and transfer; free-time route geometry and persistent guidance; exact 50 ms common window; per-joint load reporting; goal latch and action recomputation. | Learned path quality, physical speed response/compliance at 1.5–2.2 m/s, useful impact improvement, joint-load compliance rates, and competence on all three curves. The intervention actor was deterministic test code, not a learned policy. |

The inherited effort limits `[30,60,30,30,30,30] N m`, per-joint impulse caps `[0.369,0.246,0.738,0.369,0.246,0.0164] N m s`, impulse dose `0.2`, velocity limit `3.1415 rad/s`, and velocity maximum probability `0.5` remain unchanged. These are software contract values, not a hardware safety guarantee. Hardware safety requires a separately validated load model and limits. Independent whole-change review is pending in the parent handoff; this document should receive its final review status there.

## Final-review findings and corrections

The independent whole-feature review identified two important measurement/identity gaps at `e9226fe`. Before these fixes, the root combined 19-file verification (the 18-file command above plus `tests/test_free_time_software_qualification.py`) exited 0 with **346 passed, 72 upstream warnings in 73.01 s**. That is a pre-fix result, not the final tree's test count.

**Executed actuator identity.** Host-template checks alone missed live actuator changes. A CPU regression changes only the executed J1 force upper limit to 1 Nm: the sampled controller holds 2.56 Nm while MuJoCo applies 1 Nm, with the host template untouched. Metadata validation now rejects this drift. It also checks executed transmission/type/gear/force-limit fields and joint-address mappings against the already validated actuator contract. Shared arrays and homogeneous per-world arrays are accepted; heterogeneous worlds are rejected. VIC load metadata now includes executed joint damping, so a homogeneous live-only damping change invalidates checkpoint matching. Homogeneous one/two-world identities and expanded mechanics/damping arrays are covered. Native and legacy fixed serialized identities remain unchanged. Controller-owned gain/bias parameters remain mutable and are not plant fingerprints; this is not an audit of every MuJoCo parameter.

**Approach coverage.** Full-path phase includes 20 mm of follow-through, so its last bin was unreachable in a complete precontact approach. The reference now supplies a separate clamped start-to-nail vertical coordinate for the 16 precontact bins. Output labels this coordinate and reports `approach_height_extent` instead of the misleading `phase_extent`. Full-path waypoint/error computation and the monotone policy phase latch are unchanged. Tests feed the real reference-to-metric adapter: a dense ideal approach has full coverage and negligible geometric error; sparse samples retain missing bins; rebound/off-curve samples retain current-height error without advancing policy phase. Coverage still reflects observed samples, not interpolation, exact pacing invariance, or learned competence.

The final-fix focused command was:

```bash
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest tests/test_sampled_pd500.py tests/test_impact_speed_contract.py tests/test_strike_evaluation.py tests/test_quintic_reference.py tests/test_impact_speed_evaluation.py tests/test_free_time_software_qualification.py -q --show-capture=no
```

Exit 0: **123 passed, 0 failed, 0 skipped, 70 upstream TorchScript/ONNX deprecation warnings in 64.03 s**. Sixteen new targeted regressions first failed for the missing validation/incorrect binning. This is the scoped final-fix suite, not a repository-wide regression. The final review remains **pending scoped re-review** of these fixes.

These corrections improve software identity and evidence quality. They establish no learned speed response, useful-impact improvement, joint-load compliance rate, three-route competence, CUDA/numerical-failure parity, or hardware safety. No training, cluster/job interaction, hardware action, or original-source edit was performed.
