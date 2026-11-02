# Contact-rich learning: Z1 striking to Kyber grasping — implementation and proposed studies

The research goal is a reusable learning approach for contact-rich tasks: PPO with variable impedance, task-appropriate rewards and physical constraints, and exploration that can adapt to the state. Z1 nail driving is the first test case; Kyber hand grasping is a second, substantially different contact task. Stronger Z1 strikes are an immediate outcome, not the entire research claim.

Research question: can the same learning mechanisms support useful contact interaction across a brief impact and sustained multi-contact grasping, after adapting the robot/task interfaces? State-dependent std and gSDE are hypotheses to test, not established improvements or safety mechanisms.

Delivered-impulse maximization remains required for Z1. The three selected Z1 task rewards stay ON in every comparison. That requirement does not mean maximizing collision impulse in grasping: the hand needs grasp/lift/retention objectives and task-appropriate load/slip measurements. Historical 031 records and the old ablation proposal remain preserved evidence.

The first claim to test is reuse of a learning method with separately trained policies. Direct transfer of Z1 weights, zero-shot control of a different robot, and a single generalist policy are separate research goals not demonstrated by this proposal. Generalization to unseen starts/routes or objects is also separate from learning multiple trained routes.

## What transfers, and what changes

| Part | Z1 striking | Kyber grasping | Transfer rule |
|---|---|---|---|
| Learning | Existing PPO/CaT and separate position/gain action groups | PPO with the hand's existing action/reference interface | Reuse probability, exploration and constraint-learning mechanisms; audit hand integration rather than assuming CaT already exists there |
| Control | Six position targets, six gain requests, one desired-speed selector | Hand joint/residual commands and hand-specific gain groups | Retain each robot's action meaning; do not copy 13 outputs or force SBTO reference following into Z1 |
| Task reward | Delivered impulse, bounded penetration and once-only full depth, plus retained path/strike terms | Task-appropriate grasp acquisition, object lift and sustained retention | Share task-reward separation/accounting; substitute the physical objective, not its numerical constants |
| Constraints | Existing velocity/load/reaction-impulse checks | Calibrated hand/joint/object limits, with slip/contact-load measures as supported by the hand setup | Reuse constraint handling; thresholds, sensing and aggregation need hand-specific validation |
| Event/outcome | First strike plus 200 ms aftermath; 30 mm attainment is not retained depth | Contact acquisition, lift and defined hold interval; slipping/dropping must remain visible | Reuse pre-reset/event accounting, not the nail's horizon or success threshold |
| Exploration | Position-only state-independent std, state-dependent std or gSDE | Equivalent command-group comparison in hand action units | Transfer the method; calibrate induced command/gain perturbations and time scales separately |

Pinned October Kyber audits describe bounded PPO residuals around SBTO replay and different gain maps. Zero residual follows that reference; zero Z1 action has different physical meaning. Those audits are code/design evidence, not a current Kyber execution or proof of successful transfer. This worktree has no inspected Kyber environment implementation to modify. Neither raw std 0.3 nor Z1 gain scales/limits should be copied into hand control.

## First experiment: Z1 exploration comparison

The nine opt-in configurations below are now implemented for CPU component checks, not launched or qualified by physical rollouts. Exploration columns concern the six position actions only. A separate critic study is proposed first, as described below.

| Runs | Delivered impulse | Bounded nail-depth progress | Once-only 30 mm bonus | State-independent std | State-dependent std | gSDE |
|---|---|---|---|---|---|---|
| 1–3 | ON | ON | ON | ON | OFF | OFF |
| 4–6 | ON | ON | ON | OFF | ON | OFF |
| 7–9 | ON | ON | ON | OFF | OFF | ON |

Each group uses the same three fresh training seeds: one run per seed. State-independent std remains learnable over training; it does not vary with the current observation at a given checkpoint. Both the network-predicted state-dependent std and gSDE provide state-dependent exploration; gSDE is a separate noise construction, not an additional std head. The prior 0.3/0.1 comparison did not test either method.

Every run keeps accepted-strike credit split half at 50 ms/half at 200 ms, existing path progress and costs, current critic, variable impedance, all existing velocity/load/reaction-impulse constraints, and the common 200 ms aftermath. Gain and speed exploration remain ordinary Gaussian. No prescribed recovery motion/reward. No other disabled historical shaping terms are implicitly restored.

The intended common Z1 recipe is explicit below. Coefficients shown as retained values are from the inspected current configuration; the normalized delivered-impulse coefficient has been restored to its historical value in this opt-in recipe. Reward-manager dt semantics remain unchanged.

| Actual reward key | Proposed status in every run | Coefficient |
|---|---|---|
| delivered_impulse | ON | Historical coefficient 4; existing reference 0.2799950838088989 N·s, productive first 50 ms window, linear/unsaturated, existing manager-dt scaling |
| native_depth_progress031 | ON | 1 |
| native_depth_completion031 | ON, once only | 1 |
| accepted_spline_strike | ON, half at 50 ms/half at 200 ms | 1 |
| spline_progress | ON | 1 |
| spline_path_cost | ON | -0.2 |
| speed_goal | ON | -0.8493676 |
| action_rate | ON | -0.01 |
| joint_pos_limits | ON | -10 |
| r_tt | ON | -1 |
| approach, nail_driven, nail_depth_delta, impact_progress, completion, r_imit | OFF | 0 each |

The legacy nail_depth_delta and completion keys stay OFF because the two native_depth terms supply the selected bounded/once-only behavior. This avoids paying both old and new versions. Existing CaT constraint handling is additional to this table and stays unchanged. No reward ON/OFF status here authorizes an actual switch change.

Use 500 PPO updates per policy, evaluating checkpoints 100 and 500 with the same 32 reset contexts: 9 policies, 18 checkpoint cases, 576 nominal trials. Training seeds are replication units. Historical policies supply context, not a matched control for this new combined objective.

Initial position perturbation should match the current 0.3 scale across methods; match induced action-noise RMS for gSDE rather than its matrix parameter name. Initial gain std remains 0.3 and speed std 1. Match shared initial mean/critic/normalizer components; extra exploration parameters differ by method. No reward tuning separately for each exploration row.

## Reuse through the existing pipeline

Reuse scripts/train.py and the retained reward/configuration/initialization/checkpoint helpers. Combine the implemented bounded-depth and once-only completion functions with the existing delivered-impulse term in an explicitly identified combined recipe. Do not relabel old A3 checkpoints or edit frozen identities. No new controller, admission framework, or launch package.

The user selected the historical normalized linear impulse reward without event-unit rescaling. The existing first-strike reader/window/reference and coefficient 4 are reused. gSDE resampling is every 4 policy steps (80 ms), plus resets, using per-world matrices and unit-L2 normalized actor inputs plus a constant as the explicit detached feature map. Historical A3 still has delivered_impulse weight zero; new RewardExploration032 task identities enable it. The actual configured reader is FirstStrikeDeliveredRewardTerm, a one-shot productive first-window reader; it is distinct from the generic depth-advance-gated DeliveredImpulseTerm. Check its saturation setting so an intended maximization objective is not silently flattened above a reference. Keep reward-manager dt scaling and existing CaT treatment unchanged; no positive term bypasses constraints.

Reuse the bounded depth reward (4–30 mm peak progress, maximum raw unit budget 1) and completion term (maximum 1 once, payable at the genuine 200 ms contact horizon if 30 mm was attained during that episode). The bonus is geometric attainment, not maintained final depth or settling. Do not activate the repeating legacy completion helper during aftermath.

The state-independent std actor already exists. The installed RSL-RL already supplies HeteroscedasticGaussianDistribution (rsl_rl/modules/distribution.py:243): its network outputs mean and state-dependent std, with initialization helpers. Reuse this for the position group, retaining the ordinary gain/speed law. The historical custom actor asserts 13 scalar std parameters; the new opt-in adapter adds position-only composition, masks, entropy/KL, initialization and model/checkpoint validation without weakening the old class checks.

SB3 already implements PPO + gSDE: use_sde/sde_sample_freq, StateDependentNoiseDistribution, and rollout noise resampling. Use its distribution/probability and resampling logic as the reference, retaining our existing PPO/CaT trainer and action semantics. No SAC migration or implicit tanh action transformation. Define independent vector-environment noise matrices, feature choice, resampling/reset behavior and cadence; match induced initial position-noise magnitude rather than a matrix parameter name. At a 20 ms policy interval, 4/8/24 steps mean 80/160/480 ms; the first interval is 4 steps (80 ms), selected by the user. Preserve original sampled actions for PPO likelihood accounting while clipping environment commands separately. If code is reused, retain the applicable license notice.

For both alternatives, preserve behavior-versus-update likelihood/input consistency, selector activity masks and meaningful entropy/KL over the active action groups. Add only the needed policy/distribution integration and explicit configuration/checkpoint fields in the existing pipeline. No new trainer, generic framework or launch machinery. Validate the combined Gaussian recipe with a small separate complete-learning-update/save/reload/evaluation check first, retaining the study's 500-update setting. The existing evaluator's new recipe/metadata compatibility still needs validation. This is technical validation, not a requirement that a favorable policy outcome precede the planned exploration comparison.

Record actual per-run reward flags, coefficients, normalizers and windows in saved configuration, alongside exploration settings. Report delivered impulse, productive strikes, full 30 mm attainment, whole-episode velocity/load and contact velocity/impulse separately. Contact safety without contact remains UNKNOWN. Higher impulse or a technical test PASS alone does not establish safety.

## Separate first stage: critic combinations with added rewards OFF

Current user decision: all four event-critic rows,12 new policies, historical controls only. A new active goal now covers their implementation/training/evaluation; see [overnight plan](../superpowers/plans/2026-11-01-z1-critic-overnight.md). The earlier six-policy fresh-control suggestion below is superseded as the selected scope and retained only as a design alternative. The nine-run all-on study stays separate and unlaunched.

User explicitly chose to investigate these before launching the reward/exploration study. Here OFF means the added delivered-impulse, nail-depth-progress and 30 mm attainment terms; accepted-strike/path terms and existing constraints remain active. This limited first-stage exception does not remove the impulse objective from the subsequent all-on study.

| Priority | Accepted-strike payout | Critic | Initial position std | Delivered impulse | Nail-depth progress | 30 mm bonus |
|---|---|---|---|---|---|---|
| 1 | Half at 50 ms, half at 200 ms | Event-aware | 0.3 | OFF | OFF | OFF |
| 2 | All at 200 ms | Event-aware | 0.3 | OFF | OFF | OFF |
| 3 | Half at 50 ms, half at 200 ms | Event-aware | 0.1 | OFF | OFF | OFF |
| 4 | All at 50 ms | Event-aware | 0.3 | OFF | OFF | OFF |

All use the ordinary state-independent std law, variable impedance, the same 200 ms aftermath, constraints and 500-update target, with 100/500 checkpoints. These four rows are investigated/documented here, not implemented or allocated by the nine-run entry.

Recommended smallest first comparison: priority 1 plus an otherwise identical current-critic control, three matched fresh seeds each (six policies). Expanding all four event rows is 12 policies before any fresh controls; all four matched current/event pairs would be 24 policies. Do not quietly append that expansion to the nine-run batch. Old 30-policy results are useful references, not automatically interchangeable fresh controls. Source/initialization/protocol equivalence must be demonstrated before claiming a paired comparison against historical policies.

Why this ordering: at 500 updates split/std0.3 and 100 ms/event/std0.3 each had 32/32 productive strikes per seed and zero recorded whole-episode velocity/load violations. That does not prove their combination will improve learning. The split/std0.3 whole-episode velocity counts at 100 updates were 17/32,32/32,0/32 and load counts 8/32,7/32,16/32, so priority 1 asks whether critic information improves earlier learning. Full-depth attainment at 500 was 8/32,0/32,30/32 for split/std0.3 versus 1/32,0/32,0/32 for 100 ms/event/std0.3; productive contact and full depth remain separate.

The 200 ms/current/std0.3 policies had 0/32 productive strikes and 32/32 whole-episode velocity violations in each seed at 500 updates, with almost all contact windows UNKNOWN. Event-critic improvement at 100 ms does not establish the cause of that 200 ms failure; priority 2 tests a value-information hypothesis. Split/std0.1 retained a seed with 32/32 whole-episode velocity violations at 500 despite zero contact-window velocity violations; priority 3 must not hide that whole-episode failure with a J4 exception. At 50 ms/current/std0.3, whole-episode velocity violations were 19/32,20/32,2/32 at 500; priority 4 tests information under early payout, not a guaranteed correction.

Implementation inspection: 50/200 ms event-aware factories and the 47-input critic already exist. Split+event is not a critic-class switch: the old event observation reader requires a single-deadline credit owner and rejects the split owner. A correct future adapter must expose both remaining split balances through truthful unpaid-credit/next-payout observations, preserve the actor's 43 inputs, initialize added critic columns to zero, and identify those new semantics in checkpoints. No fake 50 ms deadline marker or legacy single-payout identity should be used.

Source: .superpowers/sdd/2026-10-31-z1-recovery029-preparation/results029/cases60.json. The original combined closure failures remain FAILED without closure.json; these descriptive results do not certify safety.

## Implemented entry and limits

Use the existing pipeline through scripts/z1_reward_exploration032.py: indices 0–2 are state-independent std, 3–5 state-dependent std, 6–8 gSDE. Within each group, seed indices 0/1/2 use candidate seeds 2026110101/02/03. No training was launched; history/availability must be checked before any later authorized Vega run.

The entry calls scripts.train.run_train once and reuses the existing six-checkpoint save filter (0,24,49,99,249,499). It has no scheduler, worker controller, submission/retry, admission, backup or closure machinery. Existing training/evaluation CLIs discover the three new opt-in task IDs through the normal registry. Frozen campaign packages remain retired. The candidate study uses fresh training only; mid-training resume of gSDE noise matrices is not implemented.

Fable and Kimi3.1 review evidence and CPU logs: .superpowers/sdd/2026-11-01-z1-reward-exploration032/. No runtime competence or safety claim follows from component tests. One overly broad legacy test selection was interrupted after 12 CPU contract checks; it constructed a CPU environment fixture but did not reach rollout tests. This is preserved as an interrupted check, not a full-suite pass or an authorized physical smoke.

## How this tests the broader research goal

The nine Z1 policies compare exploration methods within a fixed variable-impedance constrained-learning setup. They cannot establish cross-task transfer alone, nor isolate the benefit of variable impedance or CaT, because both are held fixed. Claims that either is necessary or superior would need separately justified controls; those controls are not added to this batch.

A later Kyber test is needed for cross-task evidence. The smallest proposed follow-up is state-independent std versus the alternative selected from Z1, three matched fresh training seeds per method (six hand policies), under one common grasping recipe. This is a conditional, unscheduled proposal, not an allocation. If no alternative merits selection on the predeclared Z1 utility/violation criteria, do not invent a winner or silently choose on hand test outcomes. If selection is ambiguous, resolve the comparison before hand outcomes are seen. Disclose selection on Z1; this small follow-up tests applicability of that candidate, not a complete three-method ranking. For a full cross-task exploration comparison, use all three methods within Kyber (nine hand policies) only after separate scope approval. No hand seeds, horizon, reward coefficients, threshold changes or jobs are released here.

Keep the hand's existing reference/residual interface if that is the selected task; port only the needed exploration/constraint components rather than demand a trainer rewrite. Importantly, the pinned Kyber audit did not establish a working CaT implementation. Reusing exploration with its existing unconstrained PPO would establish only exploration portability. A claim about the same constrained-learning framework requires verifying the hand's reward discount, continuation/GAE, constraint measurements and terminal/truncation semantics against the intended shared method. Neither identical source files nor identical hyperparameter numbers across robots are prerequisites; the actual method and necessary task adaptations must be documented. An identical method need not win on both tasks: an interaction between task and exploration is also informative.

Use task-specific physical outcomes, never raw reward totals across tasks. For Z1: useful delivered impulse, productive strikes, peak/full-depth attainment and velocity/load outcomes. For Kyber: lift-and-hold success, retention duration, slips/drops and joint/object loading. For both: violation frequency and severity, action clipping, applied gains, learning efficiency, variability across training seeds, and observed behavior through the relevant post-contact interval. Unobserved/censored contact outcomes remain UNKNOWN. Proposed grasp criteria require the actual hand setup and sensor availability; no success numbers are assumed.

Broader generalization claims require predeclared held-out conditions: starts/routes for Z1 and object poses/properties or unseen objects for Kyber. The existing 32 matched Z1 contexts support a controlled comparison; they are not automatically an unseen-task test. Separate development/selection conditions from final held-out evaluation, and report any selection made on Z1 before hand testing. No additional generalization sweep is scheduled by this note.

## Minimal scope and unresolved choices

Implemented: the current PPO/CaT trainer and variable-impedance Z1 stack; existing bounded-depth/once-only attainment and impulse readers; opt-in all-on configuration; position-only exploration adapters for all three methods; new checkpoint identity/model dispatch; shared mean/critic initialization; and a one-policy entry that delegates to scripts.train.run_train. State-dependent std reuses RSL-RL's distribution; gSDE uses SB3's mathematical construction with the explicitly stated feature map.

The new combined Z1 recipe has its own identity; old A3 identities and constraint treatment are preserved. CPU tests cover distribution probabilities/masks, gradients, temporal sampling, shared initialization, serialization, deterministic export, config equality and entry wiring. Actual simulator/training/save/load/evaluation integration remains unvalidated because no physical or smoke rollout is authorized.

Remaining research choices concern whether to use a critic selected from the separate first-stage study in a later reward batch, and later Kyber observations/action groups, measurable grasp/hold criteria and calibrated load/velocity limits. The currently implemented nine-run recipe remains current-critic/split/std0.3; it is not silently changed based on future critic outcomes. Keep all such choices fixed across methods within a task. No arbitrary hand thresholds or exact effort estimate is justified before inspecting its selected code/runtime.

Any future complete-update/save/reload/evaluation check requires a subsequent explicit instruction. None runs as part of this document update. The matched-study hook remains 500 updates; no short check is silently passed through it.

## Why the earlier rewards were off

The September 22 saved audit explicitly reports delivered-impulse and completion reward in the evaluated policies. It does not establish every reward's activation in every September policy.

The local frozen September 30 source archive (.superpowers/sdd/spline-speed-20260930/source.tar.gz) contains source/src/tasks/hammer/mdp/spline_curriculum.py lines 7–8 listing approach, nail_driven, nail_depth_delta, impact_progress, completion, delivered_impulse and r_imit as disabled; source/src/tasks/hammer/config/z1/__init__.py lines 879–880 set their weights to zero. This was an isolated spline-profile change, not deletion of the old rewards globally. The dated September 30 curriculum report explicitly prioritizes path/speed and treats the nail as a secondary diagnostic. The completed 30-policy study inherited that profile. Current A3 config inspection confirms those legacy weights remain zero while the two new native-depth terms are enabled. The current combined all-on proposal deliberately changes that objective toward nail-driving performance.

Sources: docs/results/2026-09-22_z1_speed_safety_audit150.md; docs/results/2026-09-30_z1-spline-speed-curriculum-pilot.md; src/tasks/hammer/mdp/spline_curriculum.py; src/tasks/hammer/config/z1/spline_depth_reward031_cfg.py.

## Current Fable/Kimi discussion and source checks

Claude Fable 5.1 and Kimi K3.1 Pro completed read-only reviews of the full handoff plus a compact packet of local code/audit evidence. Both support framework applicability as the first defensible research direction, task-specific physical rewards/limits, and a future baseline-versus-Z1-selected exploration comparison as the smallest conditional hand screen. Both distinguish that six-policy screen from a complete three-method comparison. This revision adopts those distinctions without authorizing either hand batch. Their agreement is design advice, not experimental evidence.

Root qualification: retaining Kyber's existing PPO trainer is compatible with code reuse, but constraints cannot be claimed shared merely because both trainers are called PPO. The pinned hand audit records constraint RL as not yet implemented, so full PPO/CaT transfer remains a concrete integration and validation question. We have not inspected a current Kyber checkout or its sensors in this turn.

The original proposal is preserved byte-for-byte at .superpowers/sdd/2026-11-01-z1-simplification-review/reward-exploration-proposal-before-kyber-transfer.md. The full supplied handoff, requested model identities and evidence summary are in z1-kyber-transfer-discussion-prompt.txt in that directory. Raw actual model replies are z1-kyber-transfer-fable-raw.jsonl and z1-kyber-transfer-kimi31-raw.jsonl. No tools were enabled for those model replies.

Locally inspected components: installed RSL-RL modules/distribution.py HeteroscedasticGaussianDistribution (line 243), models/mlp_model.py network/distribution path, and this worktree's src/tasks/hammer/rl/impact_speed_policy.py masks, likelihood, KL and scalar-std assertions. Dependencies were read only; no installation or runtime probe was executed.

Official SB3 source checked for reuse: [PPO use_sde and sde_sample_freq](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html), [StateDependentNoiseDistribution](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html), and [on-policy noise resampling](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html). These verify that PPO+gSDE exists; they do not establish benefit for either of our tasks. A future implementation must pin the reused source/version. The handoff's detailed paper benchmark numbers were not re-audited or used as an effect-size prediction here.

Existing pinned Kyber evidence: docs/research/2026-10-21-dex-sbto-transfer-synthesis.md, docs/research/2026-10-21-dex-sbto-training-audit.md and docs/research/2026-10-29-z1-kyber-transfer-training-tests.md. These establish prior audited design facts, not current hand experiments or successful Z1-to-hand transfer.

## Earlier reward-only discussion (preserved)

Claude Fable 5.1, Claude Opus 5.5 and Kimi K3.1 Pro reviewed a compact evidence packet read-only. All supported a common all-on reward recipe for this user goal; they highlighted reward-scale/constraint interactions and the implementation needed for the exploration alternatives. Opus suggested deferring exploration based on E1 outcomes; that optional restriction is not adopted here. All three repeated an overbroad statement about depth-gated impulse from the packet; the actual configured FirstStrikeDeliveredRewardTerm was checked directly and is correctly distinguished above. Fable also suggested changing completion/CaT ordering; that is not adopted, because existing constraint treatment must remain unchanged. Their suggestions are reviews, not independent runtime evidence.

Raw discussion outputs are preserved under .superpowers/sdd/2026-11-01-z1-simplification-review/reward-exploration-{fable,opus,kimi31}-raw.jsonl. The Opus response includes textual plan-tool tags; tools were disabled and no such operation was executed.

No training, evaluation, scheduler action, reward switch, training-code change, or native goal creation is authorized or performed by this proposal.
