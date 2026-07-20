# Maximization ablation — is the forward-swing caused by impact-maximization, or is it kinematic? (2026-07-20)

**Question (from the trajectory analysis, `assets/2026-07-20_policy_videos/`):** every competent July
policy strikes with a **forward-swing arc** (head arcs ~4–7 cm past the nail-x, then back down). Is that
swing a **learned impact-maximization maneuver** (a wind-up to build contact speed), or an **incidental
kinematic reaching signature** of this arm striking from directly above?

**Early partial evidence (existing policies, single fixed reset):** the swing does **not** buy contact
speed — the two straightest successful strikers (`ip24` +0.5 cm, `g1_track` +1.0 cm) reach the *highest*
v_touch (1.41, 1.39 m/s); the ~4.6 cm group spans v_touch 0.99–1.61. And `g2_delivoff` (delivered reward
off) still swings 4.6 cm. This *leans* kinematic — but is **not decisive**, because every existing policy
still has the `impact_progress` speed reward ON. This experiment removes that confound.

## Design — a 3-point dose–response

One knob: the **impact-maximization reward mass** = `impact_progress` (ante-impact contact speed) +
`delivered_impulse` (object-side ∫F·dt). Everything else identical (fixed impedance, same scene, same
seeds, same 500 iters / 4096 envs, `imp_max_p=0` log-only, no VIC). The task keeps `approach`,
`nail_depth_delta` (600), and `completion` (100) so it still learns to **drive the nail** with zero
speed/impulse incentive.

| arm | `impact_progress` | `delivered_impulse` | how |
|---|---|---|---|
| **max-off** | **0** | **0** | `IMPACT_W=0 DELIVERED_W=0` — train (3 seeds) |
| **max-on** | 8 | 2 | = **`af1`** (already trained, `6b447bf`; training code byte-identical to HEAD `e7d9e6b` — verified `git diff --stat 6b447bf HEAD -- src/tasks/hammer src/assets` is empty) → **no retrain needed** |
| **max-max** | 24 | 4 | `IMPACT_W=24 DELIVERED_W=4` — train (3 seeds) |

Task = `Unitree-Z1-Hammer-CaT-Impulse` (the **none** arm — no imitation prior; the `-Track` prior would
*confound* by teaching the reference's apex→descent swing shape). Matches how `af1` was trained.

### Predictions (pre-registered)
- **If the swing is FOR impact-maximization** → pre-contact forward-swing **increases with dose**:
  max-off ≈ straight (≪4 cm), max-max ≫ 4 cm. (And max-off may strike slower / shallower.)
- **If the swing is kinematic** → forward-swing is **~constant (~4 cm) across all three doses**; max-off
  still swings despite zero speed incentive.

Secondary read: does removing the maximization reward **hurt task success or contact speed**? (If max-off
still succeeds at ~1.4 m/s, the maximization reward isn't even buying impact on fixed impedance — consistent
with the Phase-0 velocity ceiling.)

## Pre-training gates — all GREEN (2026-07-20, local CPU)
- `pytest` impulse/reward suite (5 files): **80 passed**.
- `validate_rewards.py`: **all phases A–M PASSED** (incl. Phase I `impact_progress`, Phase M impulse-CaT).
- `verify_contact_sensor.py`: **verified**, first contact step 5.
- Override sanity: `train.py … --env.rewards.impact-progress.weight 0 --env.rewards.delivered-impulse.weight 0`
  parses cleanly (tyro accepts the paths; fails only at GPU-select on the CPU Mac, as expected). The sbatch
  already wires `IMPACT_W`/`DELIVERED_W` (the `g2_delivoff` precedent).

## Launch (Vega — submit from an authenticated session; `git pull` first)

```bash
# max-off (3 seeds)
CAMPAIGN=mx SEEDS="0 1 2" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=maxoff IMPACT_W=0 DELIVERED_W=0 \
  sbatch --array=0-2 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W \
  scripts/slurm/vega_train.sbatch

# max-max (3 seeds)
CAMPAIGN=mx SEEDS="0 1 2" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=maxmax IMPACT_W=24 DELIVERED_W=4 \
  sbatch --array=0-2 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W \
  scripts/slurm/vega_train.sbatch
```

Runs: `mx_maxoff_seed{0,1,2}`, `mx_maxmax_seed{0,1,2}` under `logs/rsl_rl/z1_hammer/`. ~1.5 h each,
1 GPU per task, `--exclude=gn03,gn34` + `--requeue` baked in. Optional rigor: retrain **max-on** in the
same campaign (`SINGLE_SHORT=maxon`, no IMPACT_W/DELIVERED_W) for perfect commit-parity — otherwise `af1`
serves as max-on.

## Analysis (after training, local)
1. `rsync` the `mx_maxoff_*` / `mx_maxmax_*` `model_499.pt` back into `logs/rsl_rl/z1_hammer/`.
2. Add them to `ARMS` in `assets/2026-07-20_policy_videos/trace_head_trajectories.py`, re-run the trace +
   `plot_head_trajectories.py` + `swing_vs_speed.py` → compare **pre-contact swing** and **v_touch** across
   the max-off / af1 / max-max dose axis (same fixed reset, apples-to-apples).
3. Verdict: monotone swing↑ with dose ⇒ swing is a maximization maneuver; flat ⇒ kinematic.

## Guardrails honored
`imp_max_p=0` (log-only, set in the sbatch), `IMP_J_LIMIT` unchanged, caps untouched, no VIC/`set_gains`,
no superlinear excess-over-i_ref reward (the max-max `delivered_impulse=4` is the same **linear** ΔI/I_ref
term, just weighted). Provenance: git-based deploy, `-dirty` guard, per-run `git/unitree_rl_mjlab.diff`.
