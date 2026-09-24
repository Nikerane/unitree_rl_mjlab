# Speed-conditioned impulse-safety dose pair implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Test whether a higher impulse-CaT probability ceiling reduces joint-load violations without materially sacrificing useful hammering.

**Architecture:** Two fresh, matched training runs with the existing registered speed task. Change only impulse-CaT ceiling0.2versus0.3 through the existing configuration interface. Preserve strict checkpoint identity: do not resume the old0.2checkpoint under a new0.3contract. Use a small evaluator-only configuration adapter rather than changing production controller/reward logic.

**Tech Stack:** Existing pinned mjlab/MuJoCo/rsl-rl on Vega, existing training/evaluation scripts, Python observational adapter.

**Spec:** `docs/results/2026-09-22_z1_speed_safety_audit150.md`, especially the next-proposal and interpretation limits; user approved a stronger-dose comparison, with the fresh-run initialization decision presented for confirmation.

## Global constraints

- Worktree `/Users/nikerane/.codex/worktrees/z1-impact-speed/unitree_rl_mjlab`, source`cc46d9998a3c9cd5d8813e89b57e44248353e03e`, assets`b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Task`Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed`, external uniform goals[1.5,2.2]m/s, speed cost0.8493676.
- Two fresh models, seed2,1024worlds,24steps/iteration,150uninterruptedupdates each;3,686,400transitions each. Save every25updates and finalmodel149. No resume, autonomous phase, or additional candidate.
- Only independent variable: impulse probability ceiling0.2or0.3. A50%higher ceiling is not necessarily50%greater realized safety pressure.
- Preserve velocityCaT0.5, all joint caps (including J6=0.0164N·m·s),50mswindow/baseline subtraction, splines/follow-through/reset, joint/action/gain mapping and every reward weight.
- Historical resumed150-update policies are context, NOT the matched control for this fresh uninterrupted pair.
- OneGPU,4CPUs,24GiB,45minute hard limit per arm including evaluation; no retries or automatic extension. Reward/contact/setup gates and198focusedtests before learning. Stop on native nonfinite/process/gate failures.

## Review focus

1. Strict checkpoint safety metadata must match each dose; no bypass or edited checkpoint metadata.
2. Training and evaluation must use the same declared dose; all other live metadata equal.
3. Both actors start with identical initialization protocol and matched first physical resets; changing dose must not consume RNG.
4. Evaluation evidence must include true terminal steps before autoreset; only episode0/world enters primary comparisons.
5. Report load frequency and severity across every joint and velocity, actual delivered impulse, and effective constraint pressure—not aggregate reward alone.

## Task1 — Qualify the bounded launch/evaluation adapter

**Files:** Create scratch` .superpowers/sdd/2026-09-21-vega-speed-qualification/dose_eval.py`, `dose_pair.sbatch`; result record`docs/results/2026-09-22_z1_speed_impulse_dose_pair.md`. No production edits.

- [x] Verify both training overrides parse through the actual repository Tyro flags. Confirmed existing flag`--env.metrics.cat-soft.params.imp-max-p 0.2`and`0.3`; no new training CLI needed.
- [ ] Build evaluator-only adapter accepting a dose argument, set `cfg.metrics['cat_soft'].params['imp_max_p']` before constructing the environment, then call the existing evaluator. Keep its native strict checkpoint load and metadata validation. Reuse the existing snapshot seam to record velocity/impulse probabilities, immediate reward attenuation and actual substep joint-velocity maxima.
- [ ] Run a64transitionCPU observational smoke per dose. Assert no learning; `delta=max(delta_velocity,delta_impulse)`; reward-term sum equals native reward; metadata dose equals requested dose. Compare configuration dictionaries after removing only`metrics.cat_soft.params.imp_max_p` and assert equality.
- [ ] Verify a0.2checkpoint is rejected by the0.3evaluation environment. Verify an ordinary0.2reload still succeeds. Do not change the validator to make the mismatch pass.
- [ ] Freeze result paths and exact commands; independent reviewer checks budget, initialization, metadata, evaluator and non-overwrite guards before submission.

Training command for each dose (run in separate fresh directory):

```sh
python scripts/train.py Unitree-Z1-Hammer-Quintic-M20-ImpactSpeed \
 --speed-bounds '(1.5,2.2)' --speed-cost 0.8493676 --speed-mode external \
 --env.metrics.cat-soft.params.imp-max-p 0.2 \
 --env.scene.num-envs 1024 --agent.seed 2 --agent.max-iterations 150 \
 --agent.save-interval 25 --agent.logger tensorboard --agent.run-name dose02_fresh150
```

Treatment command changes only the dose to0.3 and run/output labels to`dose03_fresh150`. Both commands omit resume. Existing mandatory pretraining gate commands are those banked in`extend150.sbatch`; reuse their exact source-pinned invocations.

## Task2 — Run once, evaluate and adjudicate

- [ ] Submit exactly two fresh jobs after the initialization choice is confirmed; record IDs immediately. Monitor process state and timeout, do not auto-retry.
- [ ] Verify150trainingupdates, finaliteration149, original initial model/reset identity and final checkpoint hashes/finiteness.
- [ ] Evaluate both finalcheckpoints at1.5/1.85/2.2m/s,128firstepisodes/goal, seed2026092101. Check paired reset/RNG evidence; report any replay discrepancy rather than relaxing tolerance silently.
- [ ] Record J6and all-joint over-cap episode counts, p95 and maximum window/cap ratios; per-joint substep velocity maxima and over-limit frequency; hits, median delivered impulse, median absolute onset-speed error, geometric tracking error; separateCaTprobabilities and impulse's incremental attenuation above velocity-only.
- [ ] Compare0.3against the newly trained0.2control. Primary question: do both violation frequency and severity decrease? Any worsened joint/velocity tail is reported as a trade-off, not an overall safety improvement. A null realized-pressure difference means the dose manipulation may have been masked; it is not proof stronger impulse pressure cannot help.
- [ ] Prospective utility guard for choosing a follow-up:≥116/128hits at everygoal, median delivered impulse≥90%of matchedcontrol at everygoal, trackingmedian≤21.622820mm. The90%retention rule is a NEW proposed screening threshold, not a prior scientific/hardware requirement. Report all metrics even if the guard fails.
- [ ] Preserve original E1accuracy≤0.07m/s at everygoal as a separate unchanged qualification gate. Neither a near pass nor improved utility authorizes autonomous training. No dose or policy is certified safe by this screen.
- [ ] Bank both outcomes and independent result review; stop. No further training, best-checkpoint selection across intermediate iterations, cap loosening or broad code cleanup.

## Planning review and current status

Astra supports the fresh matched0.2/0.3pair, strict metadata preservation and manipulation check. Its cautions are incorporated: unproven cap feasibility, nonmonotonic dose response, single-seed scope, new90%utility threshold, and distinction between configured ceiling and realized pressure. Full CLI parsing passed both doses. No implementation adapter or training job has been launched. Await confirmation of fresh-pair initialization before executing the plan.

**Completion update:** User confirmed the freshpair. Both150-update jobs45693109_[2–3]completed, initialization/evaluationparity and checkpoint checks passed, independent resultsreview complete. See`docs/results/2026-09-22_z1_speed_impulse_dose_pair.md`. J6severityimproves but frequencydoesnot; bothfail fullE1qualification. All planned experiment steps complete; historical unchecked boxes above preserve the original planned procedure. No furtherlaunch.
