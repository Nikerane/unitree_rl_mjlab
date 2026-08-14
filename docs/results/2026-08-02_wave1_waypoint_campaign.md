# Wave-1 waypoint-guidance campaign ledger (2026-08-02)

> **Impulse-threshold provenance correction (2026-08-14):** The table label
> “Manufacturer caps” and `[1.64, 3.28, ...]` below record the historical
> registered-task boundary, not a validated Z1 reaction-impulse or damage limit. The
> campaign ledger remains frozen; see `../research/reward-design/IMPULSE_CAP_PROVENANCE.md`.

Launch record for Task 4 of `docs/superpowers/plans/2026-08-02-wave1-waypoint-guidance.md`.
This file records what was launched and how it qualified. **No analysis** — that is Task 5.

## Frozen identity

| Item | Value |
|---|---|
| Code revision | `a6a9c970ea9eca00b34e8e9be806d22d396a964a` (branch `cartesian-guideline-fic`) |
| Asset revision | `b58ccd2f81fd246f27c1e8d88cf86484cd888703` (`safe_impact_manipulation`, `hammer-z1`) |
| Vega code checkout | `~/repos/wave1_a6a9c97/unitree_rl_mjlab` (detached, 0 dirty) |
| Vega asset checkout | `~/repos/wave1_a6a9c97/safe_impact_manipulation` (detached, 0 dirty) |
| Seeds | 2, 3 |
| Environments | 4096 |
| Iterations | 200 |
| GPUs | 1 per run (`gres/gpu:1`, A100-SXM4-40GB) |
| Reward-weight CLI overrides | none (`IMPACT_W`/`DELIVERED_W`/`NAIL_DRIVEN_W` unset via `env -u`) |
| `imp_max_p` | 0.0 (log-only) |
| Manufacturer caps | `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]` — unchanged |

A revision-specific deploy directory was required: the pre-existing
`~/repos/unitree_rl_mjlab` carries 128 untracked files (would trip the launcher's
`-dirty` guard) and an editable install whose `MAPPING` pins `src` to its own tree.
The fresh checkout was verified to resolve both `src` and `z1_hammer_robot.xml` to
the new revision-specific paths.

## Gate results

**Reference qualification** (`evaluation/guideline/qualify_reference.py`, Vega clean checkout) —
exit 0, `qualified_resets=16/16`, `failures: []`.
Per reset: `gates_crossed=6`, `contact_seen=True`, `success_reached=True`, `finite=True`,
`multi_gate_crossings=0`, all six `target_start_distances_m` positive (0.0187–0.0211 m),
`qvel_peak_rad_s = 2.5308 <= 3.1415`, `max_commanded_rise_m = 0.0`.
Provenance recorded clean (`git_dirty=False`, asset `tracked_dirty=False`).

Artifacts: `~/wave1_a6a9c97_qual/`
- `per_reset.csv` `322085b01bff52a2bafaf1e7029fc936499dfee399fba22727bd65ecaf95ce90`
- `reference_xz_xy.png` `0b60cccc6c5cd410e5fab737f096c72fe5b93c15c5b487110be15e9160b843d9`
- `qualification.json` `ae44223388f700cd8a7ee8331f97f194c7fffccd1a032ebb973541f1db52cbd3`

**Final-revision CUDA smokes** (job 40529703, 256 envs, `cuda:0`) — all three
`cuda_qualification_pass: true`, `failed_predicates: []`.

| Predicate | C0 | C-Gate | P |
|---|---|---|---|
| `impossible_success_n` | 0 | 0 | 0 |
| `lambda_dead_n` | 0 | 0 | 0 |
| contact / success | 256/256 | 256/256 | 256/256 |
| `lambda_live` | 256/256 | 256/256 | 256/256 |
| `guideline_all_gates_crossed` | 256/256 | 256/256 | 256/256 |
| `multi_gate_crossings` | 0 | 0 | 0 |
| peak qvel (rad/s) | 2.5308 | 2.5308 | 2.5308 |
| `terminal_manager_gate` | 0.0 | 0.160 | 0.0 |
| `terminal_manager_progress` | 0.0 | 0.0 | 0.160 |

`0.160 = 8 (weight) x 1.0 (raw dose) x 0.02 (dt)`, confirming both treatment weights are
exactly 8 with a full scripted dose. `r_gate` and `r_waypoint_progress` are each created
only under their own config flag.

Artifacts: `~/wave1_smoke/out/*.json`, log `~/wave1_smoke/logs/z1-wave1-smoke-40529703.out`.

## Launch matrix

Submitted from `~/repos/wave1_a6a9c97/unitree_rl_mjlab` via
`scripts/slurm/vega_train.sbatch` `SINGLE_TASK` mode, `CAMPAIGN=wave1`, `--array=0-1`.
The launcher was not modified. Run names carry a doubled prefix because the launcher
composes `RUN=${CAMPAIGN}_${SINGLE_SHORT}_seed${seed}` and the approved
`SINGLE_SHORT` values already begin with `wave1_`.

| Job | Arm | Task | Seed | Run identity | Node | State |
|---|---|---|---|---|---|---|
| 40529882_0 | wave1_c0 | `...Guideline-C0` | 2 | `wave1_wave1_c0_seed2` | gn01 | COMPLETED 0:0 |
| 40529882_1 | wave1_c0 | `...Guideline-C0` | 3 | `wave1_wave1_c0_seed3` | gn01 | COMPLETED 0:0 |
| 40529890_0 | wave1_g | `...Guideline-CGate` | 2 | `wave1_wave1_g_seed2` | gn01 | COMPLETED 0:0 |
| 40529890_1 | wave1_g | `...Guideline-CGate` | 3 | `wave1_wave1_g_seed3` | gn02 | COMPLETED 0:0 |
| 40529893_0 | wave1_p | `...Guideline-CProgress` | 2 | `wave1_wave1_p_seed2` | gn04 | COMPLETED 0:0 |
| 40529893_1 | wave1_p | `...Guideline-CProgress` | 3 | `wave1_wave1_p_seed3` | gn04 | COMPLETED 0:0 |

Task prefix elided above is `Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-`.

## Startup validation (all six)

- `### TRAIN` header: correct arm + seed, `git=a6a9c970ea9eca00b34e8e9be806d22d396a964a`
  with no `-dirty` suffix.
- `### TRAIN_ASSET git=b58ccd2f81fd246f27c1e8d88cf86484cd888703` from the
  revision-specific sibling.
- `A100 ok: NVIDIA A100-SXM4-40GB`; `TresPerNode=gres/gpu:1`.
- `Number of environments 4096`, `Environment device cuda:0`, correct `Environment seed`.
- `Learning iteration 0/200` … `199/200`; no NaN/Inf anywhere in any log.
- `Episode_Metrics/impossible_success: 0.0000` at first iteration on all six.
- Reward manager, confirming treatment isolation and no leaked overrides:
  - C0: 8 terms, no `r_gate`, no `r_waypoint_progress`
  - G: 9 terms, `r_gate = 8.0`, no `r_waypoint_progress`
  - P: 9 terms, `r_waypoint_progress = 8.0`, no `r_gate`
  - all arms identical on the baked weights: `approach 0.1`, `nail_driven 0.5`,
    `impact_progress 8.0`, `completion 100.0`, `action_rate -0.01`,
    `joint_pos_limits -10.0`, `delivered_impulse 2.0`
- Both repositories still report 0 dirty lines after training.

No job was cancelled. No job was replaced.

## Outputs

Checkpoints (`model_0/50/100/150/199.pt` in each; **all six final checkpoints enter Task-5
analysis, including any zero-dose or weak seed — none may be filtered or replaced**):

```
~/repos/wave1_a6a9c97/unitree_rl_mjlab/logs/rsl_rl/z1_hammer/
  2026-08-02_22-37-05_wave1_wave1_c0_seed2/model_199.pt
  2026-08-02_22-37-05_wave1_wave1_c0_seed3/model_199.pt
  2026-08-02_22-37-05_wave1_wave1_g_seed2/model_199.pt
  2026-08-02_22-37-07_wave1_wave1_g_seed3/model_199.pt
  2026-08-02_22-37-07_wave1_wave1_p_seed2/model_199.pt
  2026-08-02_22-37-07_wave1_wave1_p_seed3/model_199.pt
```

Training logs: `~/repos/wave1_a6a9c97/unitree_rl_mjlab/logs/z1-train-{40529882,40529890,40529893}_{0,1}.out`

Elapsed 10:26–11:44 per run.
