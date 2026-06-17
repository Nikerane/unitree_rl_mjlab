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

---

## V1 — Results (completed 2026-06-17)

V0 sanity PASSED (A100, GPU stack healthy). V1 ran **two arms in parallel** on the recalibrated
env (commit `f74a11d`, `NAIL_SUCCESS_THRESHOLD=0.027`), 500 iters, 3 seeds each, 4096 envs,
**1 A100 per task** (6 one-GPU array tasks). Logs under `logs/rsl_rl/z1_hammer/2026-06-17_10-50-13_*`.

| Arm | Gym task | Slurm array | Reward |
|---|---|---|---|
| A-BASE | `Unitree-Z1-Hammer` | **`36472565_[0-2]`** | current 7-term |
| A-TRACK | `Unitree-Z1-Hammer-Track` | **`36472566_[0-2]`** | 7-term + weak-annealed `r_imit` (T2) |

**Submit cmd (re-runs):** `ITERS=500 RUN=<a_base|a_track> TASK=<task-id> sbatch --array=0-2 scripts/slurm/train_array.sbatch`
Each run ≈ 12 min wall on one A100 (~62k FPS); all 6 finished cleanly (no crash, no NaN — incl. A-TRACK, the only arm with new GPU code).

### Headline: both arms reach 100% success and converge to the *same* clean single strike

All 6 seeds converge by **~iter 25** to **100% success** (terminate on `nail_driven`, **0% timeout**),
mean episode length **~8.6 control steps (~0.17 s)**. A-BASE and A-TRACK are **statistically
indistinguishable** on every metric (final TB + a 64-env × 80-step rollout per checkpoint via
`scripts/diag_policy_trace.py`):

| metric (mean±std, 3 seeds) | A-BASE | A-TRACK |
|---|---|---|
| success rate | 100% | 100% |
| episode length (steps) | 8.63 ± 0.04 | 8.65 ± 0.10 |
| # distinct contacts / episode | **1.04** | **1.06** |
| peak axial speed @ contact | **0.447 ± 0.003 m/s** | **0.448 ± 0.004 m/s** |
| ante-impact deviation from ref | 49.5 ± 1.2 mm | 51.0 ± 1.4 mm |
| TB per-term `impact_progress` | 0.0133 | 0.0133 |

### Path-A deliverable — *does the press exploit survive training?* → **No. The policy strikes.**

The trained policy is a genuine **single-strike hammer**, not a press: **one contact event** (~1.05),
**~0.45 m/s downward axial speed at contact**, nail driven 0→27 mm in ~3 contact steps, episode ends
in ~8.6 steps. A press (scripted) needs **~89 steps at v≈0** with sustained contact — none of that
appears. With terminations disabled the policy **strikes → retracts (head climbs back ~0.11 m) →
re-strikes** (env-0 no-term trace), confirming a learned swing rather than a quasi-static push. →
**Path-A outcome #1: the press dissolves under training; fixed-PD position-control striking is viable
on the Z1.**

*Honest caveat (matters for the thesis framing):* the reset places the head ~13 cm above the nail, so
a direct descent-strike is the obvious, trivially-reachable solution (converged by iter 25, both arms
identical). The reward (impact_progress + completion + implicit time cost) does disfavour the press,
but the easy reset means the press is never seriously explored. The behaviour is a **controlled
~0.45 m/s drive-through**, not a max-velocity slam (IK max ≈ 2.5 m/s) nor a press — on the position-only
DiffIK action space the policy modulates approach speed and lands on the minimum sufficient value. This
supports "striking viable" but does **not** prove the reward would beat a press on a harder reset / the G1.

### A-TRACK vs A-BASE — does the tracking prior help? → **No measurable effect on this task.**

- `r_imit` anneal verified end-to-end: live weight stepped **0.10→0.08→0.06→0.04→0.02→0.00** exactly on
  schedule (iters 0/50/100/150/200/250, keyed to the 24-steps/iter counter), and `Episode_Reward/r_imit`
  was non-zero through iter ~250 then 0 — so the prior *was* active early; the ablation is real, not a no-op.
- It neither **helped** (no faster/cleaner convergence — A-TRACK was if anything marginally slower at iter 25)
  nor **caged** (anchor-or-cage answer: **neither** — ante-impact deviation ~51 mm ≈ the prior's σ=50 mm and
  is *not lower* than A-BASE's ~49.5 mm, so the policy never hugged the reference). On a task this easy the
  prior leaves no fingerprint; its discriminating value would need a harder reset / the G1.

### Watch-item (r_imit hover-near-apex farming) — **did NOT occur; no guard added.**

Hover-farming would show as inflated episode length while the prior is active and deviation ≪ σ. Instead:
episodes are ~8.6 steps (converged by iter 25, *during* the active-prior phase), 0% timeout, all terminate
on success, and deviation ≈ σ. The early-training episode-length bump (iters 5–10) is identical in A-BASE
(no prior) → it is exploration, not r_imit farming. Per augment-not-replace, **no per-episode cap or
φ-descent gate added.**

### Reproduce / artefacts

- Behavioural rollout + metrics: `python scripts/diag_policy_trace.py --task <id> --ckpt <model_499.pt> --num-envs 64 --nsteps 80 --device cpu` (add `--no-term` for the full strike-retract trajectory + depth ceiling).
- TB scalars: `logs/rsl_rl/z1_hammer/2026-06-17_10-50-13_{a_base,a_track}_seed{0,1,2}/`. Checkpoints: `model_499.pt` in each.
- Resolves **Path-A** and feeds **Q1** in `docs/research/reward-design/OPEN_QUESTIONS.md`.
