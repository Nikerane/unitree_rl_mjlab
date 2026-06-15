# Vega GPU Training Plan — Z1 Hammer ablation campaign

**Date:** 2026-06-15 · **Cluster:** EuroHPC Vega (`~/repos/VEGA_GPU.md` is the access guide).
**Why:** every "needs GPU training" gate in this repo (Path-A press-survival, the T2–T5
ablation arms, OPEN_QUESTIONS Q2/Q3/Q6/Q11/Q13–Q15) has been blocked on hardware. Vega
(4× A100-40GB/node, Slurm) unblocks them. CPU here only ever smoke-tested.

## What we learned during bring-up planning (2026-06-15)

- `ssh vega "<cmd>"` works non-interactively (live 8 h master) — scriptable.
- Both repos are on GitHub under `Nikerane/` → clone as **siblings** on Vega so the
  hard-coded asset path (`z1_constants.py` / `nail_block.py` use `parents[5]/safe_impact_manipulation`)
  resolves. Layout: `$REPО_ROOT/unitree_rl_mjlab` + `$REPO_ROOT/safe_impact_manipulation`.
- `scripts/train.py` is already GPU-ready: sets `CUDA_VISIBLE_DEVICES`, `MUJOCO_GL=egl`,
  `device=cuda:<rank>`, torchrun workers for multi-GPU. **No code changes to train on GPU.**
- Deps are pip packages (`mjlab>=1.2`, `mujoco-warp>=3.5`); local known-good = mjlab 1.4.0,
  mujoco 3.8.1, mujoco_warp 3.8.1, torch 2.12. Pin these on Vega for reproducibility.
- Training does **not** render (no camera obs), so EGL is set but not exercised — lower risk
  than the LIBERO render path in the access guide. Only `--video` would touch the renderer.

## Account / resources (from VEGA_GPU.md)

| Item | Value |
|---|---|
| `--account` | `d2026d06-166-users` |
| partitions | `dev` (short sanity), `gpu` (4× A100-40GB) |
| `--gres` | `gpu:1` per task — **one GPU per run, fan out with arrays** (never one giant multi-GPU job) |
| walltime cap | 48 h. A Z1 run is tiny (see below) → no checkpoint-resubmit needed for a single run. |

## Cost estimate (why this is cheap)

The scene is a 6-DOF arm + 1-DOF nail — trivial for MuJoCo-Warp. A run = `num_envs × num_steps_per_env × max_iterations`
= 4096 × 24 × 5000 ≈ 0.5 B env-steps. On one A100 this is expected to finish in well
under an hour (verify in V0). So: 1 GPU per run, all seeds/arms in parallel as an array,
each finishing fast — a full 6-arm × 3-seed campaign is ~18 short 1-GPU tasks.

---

## Stages

### V0 — Bring-up & sanity (the only real risk: the CUDA warp wheel)
One-time setup via `scripts/slurm/setup_vega.sh` (run on the **login** node — it has internet):
clone both repos as siblings, `uv venv --python 3.12`, install pinned deps + `-e .`.
Then `scripts/slurm/sanity.sbatch` on `dev`/`gpu`: 50 iters, `num_envs=2048`, asserts the
env constructs, GPU is used, a checkpoint is written, and the asset path resolved.
**Gate:** sanity job prints final iteration + saves `model_*.pt` without crashing.

### V1 — A-BASE: the Path-A deliverable (trainable NOW, no new code)
The current 7-term reward on the recalibrated T0 physics, 3 seeds (`scripts/slurm/train_array.sbatch`,
`--array=0-2`). Answers the open Path-A question with a **trained** policy, not a scripted probe:
*does the press exploit survive training at 30 N friction?* Logs success rate, episode length,
and (eyeball from a rollout video pulled back) press-vs-strike behavior. Either outcome is a
written result (press dissolves → fixed-PD striking viable; persists → variable-impedance
justification). Baseline numbers for every later arm.

### V2 — Full ablation campaign (after T2–T5 are implemented)
Array over arms × seeds: A-BASE, A-PRIOR (primary), A-RES (opt), C-PEN, C-LAG, I-MOM
(see `reward-design/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md`). One GPU per (arm, seed).
Metrics per the plan's ablation table (success, delivered impulse, peak/episodic Λ_j,
press-watchdog, deviation-from-reference, action std), ≥3 seeds, mean±CI.

### V3 — Analysis
`rsync` `logs/rsl_rl/` back here; tensorboard locally; fold results into the plan/HANDOVER
and resolve the corresponding OPEN_QUESTIONS.

---

## Operating notes
- Submit from the repo dir; logs land in `logs/<jobname>-<jobid>.out` and TB events in
  `logs/rsl_rl/z1_hammer/<timestamp>/`.
- `squeue -u $USER` to watch; `tail -f logs/...out` for live training output.
- Pre-warm any HF caches on the **login** node; GPU nodes have egress but offline is cleaner.
- Pin deps; if the warp wheel fails to build, fall back to the exact local versions and, if
  needed, a known-good CUDA index for torch (discovered live in V0).
