# New-chat handoff: spline guidance, policy-selected speed and learned impedance

## Purpose and ownership

User requests a NEW chat to consolidate relevant papers/code, obtain independent
agent reviews, and make a proper plan for the professor's objective. This is
planning/review first, not authorization to launch new variants or rewrite control.
The original chat retains the two running/queued fixed-controller repetitions
and unresolved controller/load-measurement/test issues. Do not duplicate jobs.

Professor's stated objective: follow a given trajectory while finding its optimal
impedance; start/end points and a spline, with terminal velocity a decision
variable; later one policy across a range of trajectories or human demonstrations.

User-selected interpretation: ONE online RL policy selects impact speed, then
commands joint targets and impedance while following geometric spline guidance.
It learns timing. No imposed timing polynomial, second learned tracking policy,
or generate-then-track architecture. Keep changes incremental and consolidate
code after establishing the method. Z1 is primary; G1 is optional later.

## Authoritative local source — important

Current experimental source and uncommitted evidence are at:
`/Users/nikerane/.codex/worktrees/z1-impact-speed/unitree_rl_mjlab`
branch `codex/z1-impact-speed`, base `cc46d9998a3c9cd5d8813e89b57e44248353e03e`.
Read its `docs/README.md` first. Main checkout and a newly created worktree may
NOT include these uncommitted changes. Inspect this path read-only; do not assume
the new chat's HEAD is the experiment source. Do not reset, commit, or alter the
original chat's worktree/jobs. If later implementation needs a snapshot, plan an
explicit isolated transfer including untracked files; never silently drop them.

## Paper/code reading pack

These are previously recorded leads, not a new live literature verification.
Re-open primary papers/code before relying on precise claims. All additional
sources remain in the linked comprehensive reviews; do not discard them merely
because the short priority list is smaller. Distinguish official code from an
adjacent implementation and literature fact from our adaptation.

### Priority sources

1. Bogdanovic, Khadiv, Righetti, **Learning Variable Impedance Control for
   Contact Sensitive Tasks**: https://arxiv.org/abs/1907.07500,
   https://doi.org/10.1109/LRA.2020.3011379.
   Local audits: `docs/research/khadiv_fic_rtt_audit.md` and
   `docs/research/wayfinder/vic_gain_contract.md`. Joint-target/gain actions,
   coupled damping, fixed-gain comparator, one-step target regularizer.
   Do not conflate that regularizer with Cartesian spline guidance. Paper's
   gains are not Z1 hardware bounds; official paper-code provenance needs checking.
2. **CNP3O / spline_rl**, pinned inspected code:
   https://github.com/pkicki/spline_rl/tree/013d31ccc24ddfecc38fdac3e37ae0d165a9c79e
   and `spline_rl/policy/bsmp_policy_stop.py` in that tree. Prior review says it
   modifies optimizer-proposed hitting velocity and constructs spline endpoint
   derivatives/whole-stroke parameters. Borrow the effective speed-to-motion
   connection lesson, not its planner architecture. Verify exact paper identity
   from repository metadata; do not invent an arXiv identifier.
3. Tebbe et al., **Sample-efficient Reinforcement Learning in Robotic Table
   Tennis**: https://arxiv.org/abs/2011.03275. Prior review records bounded racket
   impact-speed action and Reflexxes execution. No official implementation was
   located in the prior search. Do not borrow its 2m/s bound as a Z1 limit.
4. Vu et al., **QP-based impact momentum maximization for a hammering task by
   a humanoid robot**: https://doi.org/10.1109/AMC67705.2026.11435814.
   User PDF exists at `/Users/nikerane/Desktop/_Munich_hammering/QP-based_impact_momentum_maximization_for_a_hammering_task_by_a_humanoid_robot.pdf`.
   See trajectory review for fifth-order Bézier/end-state construction and
   independently rederive its printed endpoint equations before copying.
5. Martín-Martín et al., **Variable Impedance Control in End-Effector Space**:
   https://arxiv.org/abs/1906.08880; https://stanfordvl.github.io/vices/.
   Adjacent action-space comparison, not a per-joint hammer controller recipe.
6. Goal conditioning code: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/reach.py.
   Supports desired/achieved-goal interface, not self-selected impact speed or
   on-policy impedance; do not add HER/replay merely because Fetch uses goals.
7. User-proposed https://arxiv.org/pdf/2609.13516: no corresponding inspected
   result was found in the current indexed notes. Resolve title/relevance in the
   new chat before treating it as evidence; do not silently call it CNP3O.

### Complete prior cross-domain source collection

`docs/research/reward-design/TRAJECTORY_GENERATION_LANDSCAPE.md` contains the
full source matrix (section8/Primary source matrix), source links and limitations:
Vu; Flash/Hogan; Piazzi/Visioli; Gasparetto/Zanotto; Jia/Gardner/Mu; Kober;
Ijspeert; Ude; Paraschos ProMP; Gómez-González striking; Dragan; Calinon;
Huang KMP; Koç free-time striking; Stouraitis impact/compliance; Ti hammering;
Konno; Teramae; Biemond; Rijnen; van Steen; Sidiropoulos/Doulgeri; Werling;
Fan Apollo; Pivtoraiko; McNaughton; Mellinger/Kumar; Richter/Bry/Roy; Usenko;
Gao; Kunz/Stilman; Pham TOPP; Berscheid/Kroeger Ruckig; Wang GCOPTER;
Zucker CHOMP; Schulman TrajOpt.
Its 2026-08-22 descriptions of then-current code are HISTORICAL. Use it as a
literature map, not current implementation authority or authorization for a planner.

Supporting collections:
- `docs/research/tracking_impact_impulse_design_research.md`
- `docs/research/hammering_reward_design_deep_dive_v2.md`
- `docs/research/reward-design/CONSTRAINED_RL_LANDSCAPE.md`
- `docs/research/reward-design/2026-09-22_fullstrike_direction_literature.md`
- `docs/research/reward-design/2026-09-22_z1_gain_range_provenance.md`
- `docs/thesis/README.md`

## Current implementation map

All paths below relative to the authoritative worktree above.

| Component | Inspect |
|---|---|
| Approved design/limitations | `docs/superpowers/specs/2026-09-21-policy-selected-impact-speed-design.md` |
| Speed latch and observation | `src/tasks/hammer/mdp/impact_speed.py` |
| Mode, wrapper, event cost integration | `src/tasks/hammer/rl/impact_speed_wrapper.py` |
| Masked policy/distribution | `src/tasks/hammer/rl/impact_speed_policy.py` |
| Checkpoint/export contract | `src/tasks/hammer/rl/impact_speed_contract.py` |
| External→policy qualification transfer | `src/tasks/hammer/rl/runner.py`, `load_qualification_checkpoint` |
| Actual fixed500Hz torque hold | `src/tasks/hammer/mdp/sampled_pd500.py` |
| Existing native gain mapping | `src/tasks/hammer/mdp/variable_impedance.py` |
| Spline geometry/phase | `src/tasks/hammer/mdp/references.py` |
| Task registrations/reset/observations | `src/tasks/hammer/config/z1/__init__.py`, `env_cfgs.py`, `mdp/observations.py` |
| Tracking/reward and first strike | `src/tasks/hammer/mdp/rewards.py`, `trackability.py`, `first_strike.py` |
| Matched evaluation/interventions | `scripts/evaluate_impact_speed.py` |
| Campaign recipe/evidence | `docs/results/assets/2026-09-24_fresh_pd500/` |
| Controller regression | `tests/test_sampled_pd500.py` and impact-speed policy/contract tests |

Key distinctions already checked:
- External mode does not train the selector. Policy mode requires its own
  recorded training phase; transfer resets selector/optimizer appropriately.
- Speed action is a latched goal observed by the same policy, NOT a direct
  actuator velocity command. It must causally influence subsequent commands.
  First motion cannot depend on the newly sampled speed; next50Hz decision can.
- Fixed sampled controller overrides impedance actions AND uses its own cloned
  fixed gains. Removing one overwrite is insufficient to implement sampled VIC.
- Physical torque is sampled500Hz/held8 physics substeps; physics4kHz, policy50Hz.
- Current geometric guidance does not impose clock-timed position/velocity.

## Established result and live ownership

One fresh seed2 policy,250iterations/1024envs, fixed Kp=[256,384,384,256,192,128],
Kd=[6.4]*6; unchanged speed3.1415rad/s and effort[30,60,30,30,30,30]Nm limits.
External speed range1.5–2.2m/s. 768 evaluated first episodes over two held-out
reset seeds×three goals: all productive, zero observed joint-speed violations.
J2max3.01409,J6max1.95007rad/s. Not a universal safety guarantee.
Measured medians1.615,1.873,2.205m/s; low-goal bias+0.115m/s means prior
qualification accuracy gate must not be assumed passed.
Tracker impulse is success/window-censored: no physical-optimum ranking from it.

Read `docs/results/2026-09-24_z1_fresh_pd500_pilot.md` for full caveats and hashes.
Two user-approved repetitions now belong to ORIGINAL CHAT:
`docs/results/2026-09-24_z1_pd500_repeats.md`, Vega jobs45860324(seed3),
45860325(seed4 afterok). Last checked seed3 RUNNING,seed4 PENDING dependency.
Do not launch duplicates, cancel, modify their source, or take over monitoring.

## Requested independent review and plan

Use three bounded read-only agents: (1) primary-paper/public-code mechanisms and
what actually transfers; (2) current controller/speed/impedance implementation;
(3) adversarial experiment design, observability and claim validity. Reconcile
disagreements; do not treat agent agreement as empirical validation.

Produce a concise staged plan with exact smallest code changes, experimental
comparators, go/no-go checks, videos/white-background trace plots, what is deferred,
and approval boundary. Consider fixed repeatability→causal selected speed→sampled
VIC→three curves, but critically compare whether VIC with external speed first is
better aligned with the professor. Do not presuppose the proposed order is final.

Nondeferrable for corresponding claims: actual gain authority for learned VIC;
post-latch interventions for commanded speed; common uncensored impulse window
for optimality comparisons; validated joint-load signal/thresholds for safety.
Unrelated PNG-order and native float equality test cleanup can remain here.
No new infrastructure, broad refactor, new training variants or hardware action
until user reviews the new chat's plan.
