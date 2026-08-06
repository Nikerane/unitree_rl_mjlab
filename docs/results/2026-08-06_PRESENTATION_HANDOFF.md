# Presentation handoff — Z1 impact-safe hammering

## Start here in the new chat

Work only in:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/overnight-impulse-minimal`

The presentation must be designed collaboratively with the owner. **Do not edit the
PowerPoint until the narrative, slide sequence, plots, and videos have been discussed
and explicitly approved.** The existing deck is:

`/Users/nikerane/Desktop/_Munich_hammering/Thesis_updates_ppts/Julty_updates.pptx`

Its untouched SHA-256 at handoff is:

`619557de0714b3c512b48cee299b52a1d88962f0f56b66f1397611681db54754`

Use the `presentations` skill when slide work begins. First inspect and render the
existing deck, then propose a narrative and slide-by-slide plan. Do not silently
rewrite the deck.

## Current repository state

- Branch: `overnight-impulse-minimal`
- HEAD and remote: `8232d1ef82631ccd6def221669a68fd742fe6142`
- The branch was created on 2026-08-05 from `origin/cartesian-guideline-fic` to
  isolate the minimal overnight experiment from the broader campaign. It was not
  created for the final audit.
- The result generator and tests are committed and pushed.
- The two result directories below remain untracked intentionally, pending
  collaborative presentation design and a final freeze.
- No Critical or Important engineering defect remains in the accepted numerical
  evidence.

## Owner's immediate objective

Create a clear progress presentation that tells this experimental story:

1. Baseline hammering can solve the nail-driving task but discovers curved or
   inconsistent hammer-head approaches.
2. Ordered waypoint/reference guidance makes the pre-contact path substantially
   straighter for policies that learn the guidance.
3. The resulting guided policies violate the joint-velocity limit unless velocity
   CaT is enabled.
4. At delivered-reward dose D4, velocity CaT produces a complete observed legality
   contrast while retaining similar descriptive first-event impulse values.
5. Within the velocity-CaT arms, the sampled CUDA evaluation shows a descriptive
   D0→D2→D4 first-event impulse ordering, but the one-fixed-reset CPU comparison does
   not reproduce D4>D2.
6. The unchanged project impulse caps are not crossed, so active impulse CaT is not
   scientifically identifiable in this operating regime. No active impulse-CaT arm
   was trained.

The owner cares especially about understandable trajectories and videos. Keep the
presentation visual and concise; do not turn it into a provenance lecture. Put the
necessary scientific limitations in short captions or speaker notes.

## Experiment A — waypoint/reference guidance

Authoritative result note:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/docs/results/2026-08-03_wave2_waypoint_replication_result.md`

Failure atlas:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/docs/results/2026-08-03_wave2_guided_failure_atlas.md`

Frozen Wave-1 evidence/videos:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-02_wave1_waypoint/`

Pooled endpoints across six policies per arm:

| Arm | Meaning | All six gates by contact | Predeclared straight |
| --- | --- | ---: | ---: |
| C0 | task reward, no guidance treatment | 0/6 | 0/6 |
| G | one-shot ordered gate reward | 4/6 | 3/6 |
| P | dense unseen waypoint-progress reward | 5/6 | 3/6 |

Interpretation:

- Guided arms produced nine gate-completing policies versus zero controls.
- Six of twelve guided policies met the strict straight label.
- Three guided policies ignored guidance entirely while still driving the nail.
- Progress-only was selected pragmatically for later experiments because it was
  simpler and marginally ahead, not because P was proven superior to G.
- All 18 policies were qvel-illegal at the fixed reset. Maximum observed
  `Lambda/cap` was only about `0.128`.

Visual rule: only show waypoint/reference geometry for policies trained with that
guidance. For the later Presentation3 arms this geometry is appropriate because all
four retain waypoint-progress weight 8. Do not draw the old gate disks; use the
straight dashed guideline and ordered waypoint markers.

## Experiment B — velocity CaT and delivered-impulse reward

### Exact arms

All four arms retain waypoint-progress guidance, fixed impedance, and the same
DiffIK action interface. Impulse CaT is log-only in every arm (`imp_max_p=0`).

| Slide label | Exact treatment | Delivered reward | 500 Hz velocity CaT |
| --- | --- | ---: | --- |
| M | P+D4 | D4 (`4.0`) | no |
| V+D0 | P+V+D0 | D0 (`0.0`) | yes |
| V | P+V | D2 (`2.0`) | yes |
| V+M | P+V+D4 | D4 (`4.0`) | yes |

Prefer plain-language audience labels such as “D4 · no velocity CaT” rather than
unexplained M/V codes.

### Evidence population

- 24 trained checkpoints: four arms × seeds 2–7.
- Strict sampled evaluation: 512 CUDA episodes/checkpoint, 12,288 total episodes.
- Fixed comparison: one identical digest-bound CPU rollout/checkpoint.
- Scientific replicate is the trained checkpoint: `n=6` per arm. Do not describe
  12,288 episodes as 12,288 independent policies.

### Headline results

Velocity legality at D4:

- M, without velocity CaT: `0/3,072` sampled episodes legal.
- V+M, with velocity CaT: `3,072/3,072` sampled episodes legal.
- Sampled first-event checkpoint medians: M `0.340551 N·s`, V+M `0.338672 N·s`.
- Matched V+M−M median: `+0.005132 N·s`, 4/6 positive.

Across all velocity-CaT arms, legality was `9,213/9,216`; the three violations all
occurred in V/D2 seed 2. Therefore velocity CaT shows a strong observed contrast but
is not a hard guarantee and does not establish hardware safety.

Delivered-reward dose within velocity-CaT arms:

| Dose | Sampled CUDA first-event median | Fixed CPU first-event median |
| ---: | ---: | ---: |
| D0 | `0.317080 N·s` | `0.324832 N·s` |
| D2 | `0.326769 N·s` | `0.361221 N·s` |
| D4 | `0.338672 N·s` | `0.349387 N·s` |

- Sampled matched D2−D0 median: `+0.011350 N·s`, 6/6 positive.
- Sampled matched D4−D2 median: `+0.014680 N·s`, 5/6 positive.
- Fixed matched D4−D2 median: `−0.008917 N·s`, only 2/6 positive.

Permitted conclusion: the sampled CUDA evaluation shows a descriptive checkpoint-
mean ordering. Do not claim a proven, significant, causal, universal, or backend-
independent monotonic dose response.

Impulse-cap result:

- Strict crossings: `0/12,288` sampled episodes and `0/24` fixed rollouts.
- Maximum sampled `Lambda/cap = 0.760081`.
- Every arm used `imp_max_p=0`; impulse CaT was not active.
- Correct conclusion: active impulse CaT was **not identifiable and was not trained**.
- Caps are project-defined engineering thresholds, not manufacturer-certified
  safety limits.

Retained exception:

- M seed 6 completed all six waypoints in `0/512` sampled episodes, while still
  succeeding at the task.
- The bank contains 12,287 success-finalized first events and one productive
  window-finalized event, also from M seed 6.
- Keep that checkpoint visible; never filter or replace it.

### Metric language that must remain exact

- Primary impact endpoint: **finalized productive first-event nail-axis impulse**.
- Episode cumulative impulse = first event + post-event tail. It is not the same as
  first-strike impulse and can move in the opposite direction.
- The legacy `recontact` implementation means any contact after finalization; release
  is not required. Audience label: **post-finalization contact**.
- Sampled CUDA and fixed CPU results must remain visibly separate and must not be
  pooled.

## Current Presentation3 assets — inspect, do not freeze yet

Current directory:

`evaluation/results/2026-08-06_presentation3_final/`

It contains a table, four figures, one 2×2 video, and a poster. These assets were
intentionally left untouched for owner collaboration. Their current hashes are:

- table: `fe9ddd100103d3f6a5fcf05c2837b945917c2026d5a052162a3457a75e93d9a7`
- dose figure: `9470e3db5e318e8bb9f765ac50c848dcb45b9832c4d1d7d4d878decef8f71baf`
- velocity figure: `223874955688e15be9f01d4c9e7bc238e0ce965605d1325de144c50df2cdba9a`
- cap figure: `3509a612d5c1c168ff65ce1847e8c5d4f4f4b3df315a0f1e9865f98a55cf1d67`
- trajectory grid: `c52c8175ddcf6f5c2696473292bcd58b074c2006fb4f3f8181eb2b6bccd5c154`
- composite MP4: `1817ecae4ecc28c164a249b2869fbe806b621f5348d1af91307ccfefecc3010f`
- poster: `8b128e678d219293d0ae61932bc9cde80063b5420c3676958654958fb499af1d`

Important: these are pre-presentation-review assets, not final deliverables. The
current cap title sounds like an active impulse-CaT result and must be replaced. The
committed hardened generator also renames the legacy exported contact column; a
fresh hardened table has SHA-256
`4df0ccaaf0afb7bda055c0c3e569130cf2aecfab8c899065d45ca6bfaf3d9094`.
All numerical cells are unchanged by that rename, and all four regenerated figures
were byte-identical before presentation-specific redesign.

### Required visual corrections to discuss with the owner

1. Retitle the cap panel as log-only measurement, explicitly state 0/12,288 and
   “project-defined caps; not manufacturer-certified safety limits.”
2. Put protocol labels on every plot: sampled CUDA or identical fixed CPU reset.
3. State `n=6 checkpoints/arm`; explain thin seed traces versus thick median.
4. Use plain-language treatment labels rather than M/V shorthand alone.
5. Give the two dose panels one shared y scale if they invite direct comparison.
6. Keep first-event and cumulative impulse visibly distinct.
7. Define the nail target marker in trajectory plots.
8. Review the seed-2 composite with the owner. If retained, document that seed 2 is
   the numerically lowest frozen seed and record the terminal-frame padding rule.

## Raw input locations

Existing-arm sampled r4:

`evaluation/results/2026-08-06_presentation3_vd0/sampled_existing_r4/`

V+D0 sampled:

`evaluation/results/2026-08-06_presentation3_vd0/sampled/`

Existing-arm fixed evidence:

`/Users/nikerane/repos/unitree_rl_mjlab-worktrees/straight-waypoint-tube/evaluation/results/2026-08-05_presentation3/`

V+D0 fixed evidence/checkpoints:

`evaluation/results/2026-08-06_presentation3_vd0/`

Frozen manifests:

- `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/binding_first_18_r4.tsv`
- `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/pvd0_6.tsv`

## Audit sources

- Raw science and permitted claims:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/fresh-raw-science-audit.md`
- Final scientific wording review:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/final-scientific-review.md`
- Visual/reproducibility findings:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/final-visual-repro-review.md`
- Final generator review — APPROVED:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/fresh-generator-contract-rereview.md`
- Independent integrity review — APPROVED:
  `.superpowers/sdd/2026-08-06-minimal-overnight-binding-first/independent-integrity-fix-review.md`

## Next-chat workflow

1. Read this handoff and inspect the existing deck with the `presentations` skill.
2. Render the current deck for visual review without editing it.
3. Ask the owner about presentation length, audience, and which existing slides must
   remain. Then propose a concise narrative and slide sequence.
4. Review the available figures and videos together. Do not assume the current
   generated assets are the final visual design.
5. After explicit owner approval, edit a copy of the PowerPoint, render every slide,
   and visually verify layout, labels, equations, figures, and video links.
6. Only after the owner approves the final deck: regenerate/finalize assets, create
   one hash inventory with video-selection/composition provenance, back it up, and
   mirror it read-only. Do not freeze intermediate presentation drafts.

No new training, evaluator work, reward machinery, or infrastructure is needed for
the presentation task.
