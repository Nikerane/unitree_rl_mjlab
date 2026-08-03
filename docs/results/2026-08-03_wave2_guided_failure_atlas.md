# Wave-2 guided-policy failure atlas (2026-08-03)

Read-only navigation aid over the **frozen** Wave-1/Wave-2 evidence. No simulation was
run to build it: every number is read from the frozen pooled table
`57ff1ba7419b99cc…` and every link points at an already-rendered artifact. It changes
no result and is deliberately kept **outside** the frozen evidence tree so the 233-file
inventory root `fbec51b1…` stays valid.

Scope: the **12 guided policies** (G and P, seeds 2–7). The 6 control policies are not
included — C0 never completes the gates and has no treatment term.

Partition by the predeclared straight label (six gates by contact onset **and**
pre-contact max ≤ 15 mm **and** pre-contact path ratio ≤ 1.15):

| Outcome | Count | Arms |
| --- | --- | --- |
| Straight — all three clauses | **6/12** | G×3, P×3 |
| Gates completed, another clause failed | **3/12** | G×1, P×2 |
| Guidance ignored entirely | **3/12** | G×2, P×1 |

### 1. Straight — all three clauses met (6 of 12)

Both arms contribute three. These are the intended behaviour: the head tracks the entry->nail guideline through all six gates before touching the nail, at a path barely longer than the straight chord.

| policy | wave | gates@onset | payout | pre max (mm) | pre RMS (mm) | path ratio | video | x-z / x-y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **G/3** | 1 | 6 | 0.16 | 10.88 | 6.0 | 1.0392 | [`policy.mp4`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_g_seed3/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_g_seed3/trajectory.png) |
| **G/5** | 2 | 6 | 0.16 | 13.83 | 5.92 | 1.0398 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed5/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed5/trajectory.png) |
| **G/7** | 2 | 6 | 0.16 | 14.87 | 6.42 | 1.0356 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed7/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed7/trajectory.png) |
| **P/2** | 1 | 6 | 0.16 | 11.88 | 7.92 | 1.0971 | [`policy.mp4`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_p_seed2/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_p_seed2/trajectory.png) |
| **P/3** | 1 | 6 | 0.16 | 11.5 | 5.48 | 1.0347 | [`policy.mp4`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_p_seed3/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_p_seed3/trajectory.png) |
| **P/7** | 2 | 6 | 0.16 | 13.78 | 5.85 | 1.0298 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed7/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed7/trajectory.png) |

### 2. Gates completed, straightness clause failed (3 of 12)

All three cross every gate and all three fail **only** the 15 mm perpendicular clause — never the 1.15 path-ratio clause, which they pass comfortably (1.0344-1.0413). They bow slightly wider than the gate disks allow while still threading them, and all three sit within 1.5 mm of the bound. This is the group the predeclared threshold was decisive for: had it been set after seeing Wave 2, these would have been counted as successes.

| policy | wave | gates@onset | payout | pre max (mm) | pre RMS (mm) | path ratio | video | x-z / x-y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **G/4** | 2 | 6 | 0.16 | 15.86 | 6.3 | 1.0371 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed4/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed4/trajectory.png) |
| **P/4** | 2 | 6 | 0.16 | 16.46 | 6.41 | 1.0344 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed4/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed4/trajectory.png) |
| **P/5** | 2 | 6 | 0.16 | 16.17 | 6.7 | 1.0413 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed5/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed5/trajectory.png) |

### 3. Guidance ignored entirely (3 of 12)

Zero gates crossed, and all three still drive the nail to the 32 mm stop. The treatment term paid essentially nothing: G/2 and G/6 earned **exactly zero**, P/6 earned 0.001071 of a possible 0.16. These are task successes and guidance failures simultaneously - the task reward alone was sufficient, so the guidance term never became binding for these seeds.

| policy | wave | gates@onset | payout | pre max (mm) | pre RMS (mm) | path ratio | video | x-z / x-y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **G/2** | 1 | 0 | 0.0 | 54.0 | 32.59 | 1.4115 | [`policy.mp4`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_g_seed2/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-02_wave1_waypoint/videos/wave1_g_seed2/trajectory.png) |
| **G/6** | 2 | 0 | 0.0 | 54.16 | 37.6 | 1.5194 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed6/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_g_seed6/trajectory.png) |
| **P/6** | 2 | 0 | 0.001071 | 50.41 | 39.11 | 1.6372 | [`policy.mp4`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed6/policy.mp4) | [`trajectory.png`](evaluation/results/2026-08-03_wave2_waypoint/videos/wave2_p_seed6/trajectory.png) |

## Grids

All 18 policies on one shared scale, straight policies outlined in green:

- `evaluation/results/2026-08-03_wave2_waypoint/figures/wave2_eighteen_panel_xz.png` — x-z side view
- `evaluation/results/2026-08-03_wave2_waypoint/figures/wave2_eighteen_panel_xy.png` — x-y top view
- `evaluation/results/2026-08-03_wave2_waypoint/figures/wave2_per_arm_summary.png` — per-arm counts, straightness, qvel legality

## What the partition shows

The 12 guided policies fall into three clean groups with **nothing between them**:
straight policies run 10.88–14.87 mm, the near-misses 15.86–16.46 mm, and the guidance
failures 50.41–54.16 mm. The gap between "threads the gates" and "ignores the guideline"
is a factor of ~3.5 with no intermediate case — a guided policy either learned to track
the reference or it did not, and no seed partially learned it.

The near-miss group is narrow and one-sided: three policies, all failing the same single
clause, all within 1.5 mm of it, all passing the ratio clause easily. That is a threshold
sitting inside the completer distribution, not a distinct behaviour.

Every artifact linked here is also mirrored read-only on Vega at
`~/evidence/frozen/wave2_waypoint_train-e1a0282_analysis-bf498ff_result-cc125c0`.
