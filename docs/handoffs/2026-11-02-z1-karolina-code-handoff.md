# Z1 code preservation and Karolina handoff — 2 November 2026

Branch: `z1-full-depth-reward-learning`.

This branch preserves the current Z1 reward/training/evaluation implementation
and compact scientific reports while Vega is unavailable. It does not submit
jobs, transfer campaigns to Karolina, or establish that the latest code has
been qualified on Karolina.

## Current completed experiment

Reward037 completed eight policies at 500 PPO updates and sixteen CPU evaluation
cases at checkpoints 99/499 (100/500 updates), with 32 nominal reset contexts
each. The six main policies use initial position std 0.3 or 0.1, three training
seeds each. Two exploratory one-seed pilots randomize board and nail height
together by ±5 mm during training; their evaluations use nominal height.

All use split accepted-strike payout at 50/200 ms, actor43/event critic47,
ordinary learned state-independent Gaussian exploration, variable impedance,
and the existing constraints and 200 ms aftermath. Delivered impulse,
bounded depth progress, and once-only completion rewards are ON. The legacy
repeating completion term is OFF. No gSDE experiment has been run in this batch.

The six main policies reached 30 mm in all matched contexts at both checkpoints,
with no recorded velocity/load/contact-impulse violations. The std0.3 height
pilot at 500 updates has one whole-attempt load violation. These observations
do not certify safety or varied-height robustness.

## Existing pipeline entry points

- `scripts/z1_reward_training037.py`: eight-cell reward-enabled configuration
  and the existing PPO training runner.
- `scripts/z1_event_critic036.py`: the earlier twelve-cell event-critic study.
- `scripts/z1_event_critic036_check.py`: one-update/save/reload technical check;
  select `--study reward037` for the current recipe.
- `scripts/z1_event_critic036_eval.py`: strict checkpoint identity checks and
  the preserved matched-bank CPU evaluator; select `--study reward037`.
- `src/tasks/hammer/config/z1/spline_reward037_cfg.py`: current reward recipe.
- `src/tasks/hammer/mdp/depth_credit031.py`: bounded depth/once-only completion.
- `scripts/z1_thirty_evaluation023/vendor`: unchanged checksum-pinned evaluator
  and reset-bank dependencies. Keep their bytes intact.

The custom `scripts.z1_depth_campaign031` launcher stays retired. Some older
helpers are retained because the existing pipeline imports them; their dated
campaign commands are not a current launch plan.

## Data and backups

Checkpoints, raw evaluation arrays, training traces, frozen source archives,
failed-job evidence and detailed model-review transcripts remain in the
verified campaign backups, rather than being added to Git.

Independent SSD base: `/Volumes/NikeraneSSD/Z1-campaign-backups/`.

- Reward037: `z1-reward037-20261102T101509Z/`.
- Critic036: `z1-critic036-20261101T220701Z/`.
- Original thirty-policy study: `2026-10-31-training-evaluation023/`.

On 2 November, all eleven selected Reward037 source/checkpoint/raw archives
matched local, SSD and saved receipt hashes; all 75 final-analysis/deep-dive
files matched their SSD receipts. Critic036 completed raw archives and base
source were also rechecked. The original thirty-policy study has prior
independent backup verification; its entire archive set was not rehashed in
this preservation pass. Both original029 combined closure jobs remain failed
without `closure.json`.

## Karolina preparation still required

1. Check out this branch in a fresh source directory, preserving the existing
   Karolina release and all historical outputs.
2. Restore the matching external Z1 assets and selected checkpoints from the
   verified backups. `src/assets/robots/unitree_z1/z1_constants.py` expects the
   sibling `safe_impact_manipulation/hammer_z1_env/assets` directory; cloning
   this repository alone does not provide that external asset tree.
3. Confirm the software environment and local account/partition/cache paths.
   The [October Karolina environment record](../results/2026-10-01_z1-karolina-environment.md)
   documents an earlier qualification, not qualification of this branch.
4. Run focused CPU checks, then a separately authorized current-source GPU
   technical check and checkpoint reload before scientific training.

No new scheduler wrapper is needed. Reuse the existing pipeline and adapt only
the cluster-specific execution settings after checking the current environment.

## Preservation verification

An isolated checkout assembled from the staged Git tree passed 239 focused CPU
checks, with one historical-evidence-dependent check skipped. The unchanged
evaluator frontend imported successfully and verified all 25 pinned vendor
files without constructing an environment or running a rollout.

The broader configuration suite passed 144 checks and failed its older exact
global task-list assertion, which predates the subsequently registered studies
(`TestCartesianGuidelineStudy.test_new_registration_adds_only_the_approved_study_task_ids`).
That assertion remains visible; this preservation commit does not claim a
full-suite pass or change the experiment implementation to satisfy it.

Results: [Reward037](../results/2026-11-02-z1-reward037-analysis.md),
[mechanism deep dive](../results/2026-11-02-z1-reward037-mechanism-deep-dive.md),
and [Critic036](../results/2026-11-02-z1-event-critic036.md).
