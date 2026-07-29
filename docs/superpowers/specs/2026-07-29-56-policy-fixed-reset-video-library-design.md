# 56-policy fixed-reset video library — design

**Date:** 2026-07-29  
**Status:** owner-approved design  
**Purpose:** apples-to-apples visual comparison of all accepted FQ4x8 and
FQ3x8 policies before further impulse/reward experiments.

## Scope

Render all 56 accepted `model_499.pt` checkpoints:

- FQ4x8: F8, F0, D0, FQ-min; seeds 8–15 (32 policies).
- FQ3x8: F8, B8, FQ; seeds 16–23 (24 policies).

This phase is qualitative. It does not select policies, test Lambda
feasibility, change rewards, or train anything.

## Comparison contract

Every policy must use:

- the same explicitly frozen realized robot `qpos/qvel` and nail state;
- the same reset-state digest;
- no reset randomization and no best-episode selection;
- one environment and online policy inference;
- the same camera, resolution, rollout horizon, frame rate, and slow-motion
  playback rate;
- only the first episode, stopping before an automatic reset can contaminate
  the trace.

The existing 24-policy media is rerendered because its metadata does not bind
the realized reset state strongly enough for a cross-campaign 56-policy
comparison.

## Minimal outputs

For each policy:

1. `policy.mp4` — the fixed-reset first strike;
2. `montage.png` — six evenly spaced frames;
3. `trajectory.png` — aligned x-z side and x-y top views of the hammer head,
   with start, time progression, first contact, and nail axis marked;
4. `metadata.json` — campaign, arm, seed, task, checkpoint path/hash, code and
   asset revision, reset contract/digest, camera, timing, and output hashes.

Campaign-level outputs:

- one FQ4x8 trajectory grid;
- one FQ3x8 trajectory grid;
- one Markdown library index linking all 56 videos, montages, trajectory
  plots, and metadata.

Axes and visual conventions are shared within and across grids so curvature
and terminal alignment can be compared directly.

## Implementation constraints

- Reuse `scripts/render_policy.py` and the existing FQ3x8 media conventions.
- Make only the smallest changes needed to restore a frozen reset, stop at the
  first episode boundary, record hammer-head positions/contact, and write
  metadata.
- Do not build an interactive site, video-ranking UI, computer-vision model,
  or best-episode selector.
- Do not infer reward or impulse causality from the videos.
- Keep all tracked source deployment to Vega commit/push/pull only; checkpoint
  and generated-media transfer may use artifact-copy tooling.

## Fail-closed checks

- exactly 56 unique checkpoint identities and expected arm/seed membership;
- all checkpoint hashes match the accepted manifests;
- every metadata row has the identical reset-state digest;
- no output contains frames from a second episode;
- every policy has a readable MP4, montage, trajectory plot, and metadata
  record;
- an independent reviewer checks the renderer and a sample from every arm
  before the library is called complete.

## Deferred work

After the owner inspects the fixed-reset library:

- optionally render best representative episodes only for selected policies;
- run the separate strict 56-policy Lambda/qvel evaluation;
- decide whether a final-approach reference or impulse-term experiment is
  justified.
