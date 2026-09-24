# MJWarp repeatability: official guidance and deterministic-mode version check

Research note, 2026-09-22. Source review plus read-only inspection of Vega's installed packages; no installs, code changes, or simulations.

## Finding

The installed Vega environment reports `mujoco-warp 3.8.1`, `mujoco 3.8.1`, and `warp-lang 1.14.0`. Read-only Python introspection of that Warp install found no `warp.DeterministicMode`, no `warp.config.deterministic`, and no deterministic config fields. The official Warp 1.15.0 changelog dates deterministic execution for supported atomic patterns to 2026-07-07, after 1.14.0. Therefore Warp's process-wide deterministic-atomics switch is not available in the installed 1.14.0; do not attempt to enable it by assigning a made-up flag or assume an upgrade makes the full simulator safe.

MJWarp's current FAQ says GPU stepping is not deterministic: execution ordering can produce small numeric differences due to GPU atomics, and recommends `wp.set_device("cpu")` for deterministic results. Its still-open [MJWarp determinism issue #562](https://github.com/google-deepmind/mujoco_warp/issues/562) originally records a request to investigate an exact-repeat option. This is consistent with treating small same-GPU drift as a known possibility, not evidence that the observed impact divergence has been causally explained.

## Upgrade caveat: Warp determinism is not yet a safe drop-in for contacts

Warp 1.15+ does provide `RUN_TO_RUN` determinism for supported atomic patterns, but this is a Warp feature, not by itself a verified MJWarp-wide deterministic mode. In September 2026, MJWarp issue [#1635](https://github.com/google-deepmind/mujoco_warp/issues/1635) reported silent loss of every convex-narrowphase contact under Warp's deterministic mode; it reproduced on Warp 1.15/1.16 and MJWarp 3.11.0. The linked [fix PR #1636](https://github.com/google-deepmind/mujoco_warp/pull/1636) was still open and explicitly “not deployed” when inspected. Its notes also report record-buffer undercount risk during CUDA-graph capture unless configured, and substantial captured-run overhead. These reports postdate and do **not** directly establish behavior for the installed Warp 1.14.0; rather, they rule out casually upgrading and flipping the global flag as a safe diagnostic shortcut.

## Minimal diagnostic route

Keep the existing GPU/software stack fixed for the repeatability comparison. Start from identical serialized state and command, and record the first differing substep across all repeats: pre-contact `qpos/qvel` and applied controls first, then candidate/contact counts and contact data at the event, alongside overflow status. This can show whether a measurable pre-contact difference precedes the contact transition or whether divergence first appears at collision handling; it still would not, alone, prove causation. The currently reported pattern—small pre-contact ULP changes followed by sparse contact-event changes—is evidence of sequence, not yet proof that atomics caused the eventual event divergence.

If a deterministic reference is needed later, CPU execution is the official MJWarp-documented comparison path; treat CPU-versus-GPU as a different-backend diagnostic, not as a bitwise substitute for the GPU trajectory. No simulator or backend change is recommended by this source review.

## Primary sources

- [MuJoCo Warp documentation FAQ](https://mujoco.readthedocs.io/en/latest/mjwarp/), “Is MJWarp on GPU deterministic?” and “Why are numerical results from MJWarp and MuJoCo different?” (GPU atomic ordering; CPU recommendation; Warp uses float32 versus MuJoCo's usual double).
- [Warp 1.15.0 changelog](https://nvidia.github.io/warp/v1.15/user_guide/changelog.html#1-15-0-2026-07-07), which marks deterministic execution for supported atomics as newly added in 1.15.0.
- [MJWarp issue #562](https://github.com/google-deepmind/mujoco_warp/issues/562), upstream GPU determinism request/tracking issue.
- [MJWarp issue #1635](https://github.com/google-deepmind/mujoco_warp/issues/1635) and [unmerged fix PR #1636](https://github.com/google-deepmind/mujoco_warp/pull/1636), version-bounded evidence about later Warp deterministic-mode contact failure and proposed repair.

## Local version evidence

Inspected remotely, read-only, through `/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python` on Vega on 2026-09-22. The package metadata returned the three versions above; importing `warp` showed `hasattr(wp, "DeterministicMode") == False`, `hasattr(wp.config, "deterministic") == False`, and no config names containing “determin”. Targeted source inspection found many atomic operations in the installed MJWarp kernels, consistent with why run-to-run ordering is relevant, but does not attribute the measured trajectory drift to any specific atomic site.
