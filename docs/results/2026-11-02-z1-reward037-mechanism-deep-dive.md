# Why the Z1 reward bundle changed the learned policy

## Material Passport

Status: ANALYZED, not a simulator reproduction or isolated causal mechanism test. This deep dive reads saved036/037 training events,84 sparse saved checkpoints onCPU, source/configuration and24 matched base-policy evaluation JSON cases. It performs no new training, evaluation, resets, code/threshold changes or goal creation. Three training seeds per std setting are replication units;32matched resets and nested100/500checkpoints are repeated observations. The two height pilots remain exploratory nominal evaluations.

The strongest explanation is that the old objective rewarded reaching and accepting contact, while the new bundle rewards accomplishing physical work after contact. This explanation is supported by the exact reward definitions, observed training component budgets, seed-repeated full-depth improvement, and changed contact strategy. It cannot assign the improvement to impulse alone: three rewards changed together.

## 1. Previously the policy could earn its main reward without finishing the task

Old precontact spline progress pays only new approach progress, bounded at1. Accepted strike pays another1 for a downward first contact within10mm of the reference endpoint. That acceptance does not depend on insertion depth, delivered impulse magnitude, or the separate productive flag. Splitting the accepted unit half50ms/half200ms changes its timing, not its amount for deeper driving. On native reset the remaining split balances flush once; it is not literally a reward for surviving without a reset until200ms.

This is why productive-but-incomplete strikes are compatible with the old objective. Both depth-weak and depth-strong accepted strikes could receive the same accepted reward. Existing speed tracking/path/action/effort costs still mattered, but there was no direct reward preference for greater nail advancement after acceptance. A policy failing to reach30mm was not necessarily failing to optimize what it was trained on.

Observed outcome: old036 split/event policies were productive32/32 per case; native30mm counts at500 were0/30/0(std0.3) and0/0/2(std0.1). New037 reaches32/32 for every one of the six base policies at both100and500updates. Saved model/normalizer/RNG/initialinput/std fingerprints match all six corresponding starting policies; the actor,critic,controller,physical limits and PPO settings were retained. This is strong evidence for the changed objective bundle, not an impulse-only attribution or a population guarantee.

## 2. What each addition makes worth learning

| Reward | Actual pre-CaT episode contribution | Incentive |
|---|---|---|
| Delivered impulse | 0.08×I_snapshot/0.2799950838 | More axial impulse in the finalized productive first event |
| Depth progress | New increments of clip((peak_depth−4mm)/26mm,0,1), total≤1 | Even incomplete new-best insertion between4and30mm earns graded feedback |
| Completion | 1 once at genuine200ms contact horizon if historical peak≥30mm | Full geometric attainment is worth an additional distinct payment |

Depth terms divide by policy dt before the manager multiplies bydt, so a unit remains1. The historical impulse reader does not cancel dt:weight4×20ms=.08. This preserves the selected historical scaling. The reward labelled4 is therefore much smaller numerically than a depth reward labelled1.

Actual late training logs, after undoing their configured20s normalization, show combined depth contributions≈2 versus≈0.074–0.094 from impulse in the six base policies:21.24–26.87times as much. Old accepted reward remains≈1 and path progress≈0.9. The large change is adding about two return units contingent on useful depth, not merely making the existing hit bonus larger.

Reward size is not gradient attribution. PPO globally normalizes advantages and learns from differences between sampled returns; a small term can still distinguish otherwise tied strategies. Neither simulator forces nor depth are differentiated through the physics engine. Impulse is a graded one-shot event reward, not a force reward delivered densely at every step.

A plausible joint learning sequence is: graded progress makes partially successful contact worth improving; completion adds a large reason to cross the final threshold; impulse distinguishes productive strike outcomes. The code makes that sequence possible. The training logs show progress budgets stabilize before completion budgets in all six base runs, but simultaneous reward activation prevents proving that each term was necessary.

## 3. What physically changed: several successful strategies

Means below average the same32contexts for each trained policy at500updates. OFF→ON compares036→037. Impulse in this table is the independently accumulated full first50ms axial integral, not the canonical reward snapshot.

| Initial std | Seed suffix | Impact speed m/s OFF→ON | Physical first50ms impulse N·s OFF→ON | Current depth at50ms mm OFF→ON |
|---|---|---|---|---|
| 0.3 | 01 | 0.992→1.236 | 0.160→0.346 | 21.62→32.00 |
| 0.3 | 02 | 0.989→0.987 | 0.223→0.336 | 29.10→32.00 |
| 0.3 | 03 | 0.993→0.987 | 0.184→0.354 | 24.57→32.00 |
| 0.1 | 01 | 0.993→1.477 | 0.149→0.371 | 20.22→32.00 |
| 0.1 | 02 | 0.988→1.148 | 0.196→0.258 | 26.15→31.79 |
| 0.1 | 03 | 0.988→1.490 | 0.202→0.363 | 26.90→32.00 |

For std0.3seeds02and03, onset speeds are almost unchanged/slightly lower, yet physical50ms impulse rises≈51%and92%. Canonical snapshots also rise, by≈32%and69%. These are direct counterexamples to a universal “just hit faster” explanation. Other policies do choose higher speed, especially std0.1seeds01and03 near1.5m/s. A useful policy can exploit the speed selector instead of always choosing the old1m/sminimum.

Nor is uniform stiffening the explanation. At contact,std0.3seed02 changes actual Kp(J4,J5,J6)≈(218.65,205.80,156.94)→(204.82,159.20,102.40); seed03≈(262.66,196.49,102.40)→(205.03,163.25,102.40). These distal gains decrease while delivery increases. J2/J3remain at307.2 in both. Stiffness and damping are coupled by the existing gain map, not independently optimized free trajectories.

Different position targets, force-time profiles, effective contact inertia, joint coordination and impedance distribution are compatible explanations. These summaries cannot identify their individual mechanical contributions. A single scalar onset speed does not characterize the full subsequent contact dynamics. For the same two seeds, integral divided by recorded contact duration gives mean axial force5.27→7.92N and4.70→9.84N; contact occupancy is0.846→0.851 and0.785→0.719. Their greater impulse is therefore not simply longer contact either. These are contact-duration averages, not force peaks. Higher impulse alone also does not prove better axial alignment or energy efficiency: transverse impulses and stop reactions can change.

All new base policies reach≥30mm already at the literal50ms measurement in32/32contexts and remain≥30mm at200ms and native end. Thus the result is not just a brief30mm peak in these recorded evaluations. Later recontacts are not required to explain the initial full-depth gain; contacts and recontacts inside the first50ms are not ruled out. This does not establish long-horizon retention beyond the measured episode.

## 4. Less clipping, different exploration; not simply more noise

Saved evaluator `clipping_count` counts raw Gaussian components outside±1 before execution over all13actions. Dividing by13×policy steps gives an actual evaluation action-clip fraction. These per-world-mean fractions decrease in all six pairs at500:

| Initial std | OFF fractions by seed | ON fractions by seed |
|---|---|---|
| 0.3 | 41.7%,34.1%,35.6% | 19.9%,31.3%,27.1% |
| 0.1 | 15.3%,12.1%,15.6% | 12.0%,9.9%,9.5% |

Clipping remains substantial. This is action-bound clipping, not PPO likelihood-ratio clipping, Gaussian std clamping, or actuator torque saturation. The count includes the speed output even after its one-shot selection becomes inactive; it is not a fraction of physical interventions. Position-versus-gain clipping shares cannot be recovered from aggregate counts. Reduced clipping may accompany more effective commands, but this is an observed strategy difference rather than an isolated mediator.

Sparse checkpoint inspection finds learned stds stay positive without hitting their clamp floor. At100updates,std0.3mean position stds are0.216/0.224/0.214 with rewardsON versus0.347/0.287/0.306 OFF; mean gain-request stds0.472/0.487/0.476 versus0.694/0.652/0.654. These are learned latent exploration scales, not actual applied stiffness noise. Critical position channels narrow while some gain scales grow; a single mean across all13outputs can mislead. The successful bundle did not require a new exploration method: both studies used ordinary learned state-independent Gaussian exploration, not gSDE.

## 5. Why performance can rise without observed velocity violations—and the load trade-off

The constraint mechanism was retained. CatPPO applies `r_used=(1−delta)r_positive+r_negative` and multiplies future continuation by `(1−delta)`. The new rewards are included in positive return. For an identical trajectory, a newly added positive payment b loses an additional delta×b on a violating payment step; earlier constraint pressure discounts access to later rewards. There is more valuable task return to protect, without changing a threshold. This mathematical coupling is direct code evidence. Whether it caused the observed improvement is a hypothesis: policy-dependent CaT normalization, reward timing and other strategy changes coexist. A later violation does not retroactively erase already-paid progress or impulse.

Training meanCaTdelta overupdates50–99 is lower in all six paired runs. Std0.3OFF0.08535/0.00766/0.01961 becomesON0.00120/0.00142/0.00149; std0.1OFF0.01728/0.00160/0.00483 becomesON0.00374/0.00121/0.00246. This supports earlier learning of lower-constraint-pressure behaviour. Delta is a soft training metric, not an exceedance frequency or safety certificate.

Zero base evaluation exceedances must not hide reduced load margin. At500updates,the mean of each trial’s maximum enforced joint-load/limit ratio rises in all six policies:

| Initial std | OFF ratios by seed | ON ratios by seed |
|---|---|---|
| 0.3 | 0.322,0.477,0.375 | 0.728,0.724,0.774 |
| 0.1 | 0.401,0.375,0.393 | 0.741,0.644,0.726 |

These averages are not maximum-population safety bounds; one high-load world can be hidden by a mean. Velocity utilization falls clearly in four pairs, rises in one, and is almost tied in one. Therefore the policies obtain better depth while spending more of their available joint-load allowance; they are not uniformly farther from every limit. The one heightDRstd0.3final policy has the known joint6/world22whole-attempt exceedance,42samples/10.5ms,peak0.018727891against0.0164N·m·s. No contact-window exceedance occurred. Object-side axial impulseN·s and joint reaction impulseN·m·s are different quantities; maximizing one need not require violating the other, but it can tighten margins.

## 6. Critic learning is not a demonstrated explanation

The event critic existed in both matched studies. It cannot itself be the newly changed factor. Its four extra fields describe contact age/accepted-credit obligations, not every historical depth maximum or earned completion state after rebound. That hidden reward memory remains a possible value-estimation limitation; successful nominal outcomes do not prove the critic input is Markov sufficient.

Value loss overupdates50–99 is lower forON in all three std0.3seeds but higher in all three std0.1seeds. Late value loss is higher forON in every pair. Because the return target changed substantially, raw MSE is not a common-scale critic-quality test. There is no saved explained variance, event-aligned TD/advantage trace, component gradient attribution, trainingKL orPPOclip fraction. Fine-grained training credit traces were explicitly disabled. The saved events support the reward-accounting and optimization observations above, but cannot prove that impulse improved value learning or eliminated a particular PPO failure.

The fixed nominal evaluation bank already gives32/32depth at100 for every main policy, while stochastic training completion can remain below its full budget in one std0.1seed until aroundupdate124 by a descriptive98%-for10logged-updates criterion. Do not equate deterministic evaluation success at100 with universal saturation of stochastic training success. The32contexts are reused and do not establish held-out-domain generalization.

## 7. Impulse interpretation and mechanical-stop caveats

The canonical reward snapshot ends at first-event30mm success or50ms, whichever occurs first. It is not necessarily a fixed50ms maximization objective. Literal50ms impulse and whole-native impulse are independently accumulated physical quantities; latter may include recontact. A high whole-native sum does not prove a higher rewarded impulse. Greater full50ms impulse can partly include force after the nail reaches the32mmmechanical stop, so not every extra N·s should be labelled useful driving work. Canonical success-triggered finalization changes the integration duration as performance changes.

Implications: the restored normalized impulse term supplies graded reward for productive first-event transfer, but is much smaller numerically than the two depth terms at the selected scale. It could help distinguish productive strikes or resolve ties near the depth plateau. It could also interact with the success-triggered finalization window. The present bundle and traces cannot measure its necessity or marginal causal contribution. Calling it irrelevant is as unsupported as crediting it with the entire gain. No reader/window/coefficient was changed in this deep dive.

## 8. What transfers toward a general contact-learning framework

The demonstrated useful principle is to reward the physical outcome the task needs, while allowing the policy to choose among speed/impedance/position strategies under constraints. For Z1 that outcome includes productive axial transfer and insertion. For Kyber grasping it would be task-appropriate acquisition,lift,retention and loading criteria; maximizing collision impulse is not a universal grasp objective. Finite new-best progress accounting and once-only outcomes can be reused conceptually, with task-specific measurements and value-state memory.

The new terms did not prescribe a recovery movement or require a particular wrist trajectory. However the complete Z1 recipe still uses inherited precontact spline/path and speed shaping. These results do not demonstrate a motion-prior-free generalist or transferred Kyber policy. They support the broader research direction without substituting for a second-task experiment.

## Statistical integrity

11/11fallacy types reviewed: seed-level replication, per-seed vectors and within-policy repeated observations retained(Simpson/ecological); no contact-only selection or collider adjustment; denominators/UNKNOWN explicit; prior baseline selection disclosed(regression-to-mean); allselectedpolicies/checkpoints included(survivorship); exploratory mechanism comparisons/no significance tests(look-elsewhere/forking paths); bundle-level causal limits and temporal ordering stated(causation/reversal). No p-values, population risk confidence bounds, component causality or universal safety claims. Both029failed combined operationalclosures remainFAILED; technicalpass and archivehashes are not safety.

## Evidence and reproducibility

- `deep-dive/training-traces.py`, full500pointseries, exacttaginventory, perseedwindows and84checkpointstd audits.
- `deep-dive/behavior_analysis.py`,24caseJSONstatistics and12matchedcomparisons, archive/memberhashes, noNPZpayloaddecode needed.
- `deep-dive/ultra-mechanism-review.md`, sourcehash comparisons and independent mechanism critique.
- `deep-dive/panel/`, actual namedexternalmodel initialreviews and latestbehavior supplement responses, providerreceipts and any failures preserved.
- Plots: `reward-mechanisms037.png/.svg` and `training-traces-curves.png/.svg`.


## Named reviewer discussion and root adjudication

Fable5.1, KimiK3.1Pro(opencode-go/kimi-k3.1-pro), Gemini4Argon(opencode-go/gemini-4-argon), andGPT-6AstraUltra provided completed independent reviews. Fable/Kimi/Gemini received the same compact source/results/training packet with tools disabled; nativeUltra independently inspected saved source/evidence. Later observed behavior was supplied to all three external reviewers: Fable andGemini completed supplemental responses; Kimi’s supplemental call failed with providerquota402(insufficient account funds). Its original completedreview remains valid input; no retry or invented follow-up was used. Exact rawresponses/failure are preserved. Provider model labels are actual configured routes; “Ultra” was explicitly interpreted asGPT-6Astraultrareasoning.

Common supported interpretation: old accepted-contact proxy omitted a direct preference for successful driving; the reward bundle introduces it. The matched-init outcome jump and training budgets support this explanation. Reviewers treat impulse contribution as unidentified rather than an isolated cause. Fable’s initial assertion that impulse could not matter and that the depthtermsalonecausedsuccess is strongerthan the design supports and is not adopted. Gemini’s initial “dense impulse gradient” wording is rejected: this is a one-shot event reward, with policy-likelihood gradients, not differentiable physics. Fable’s initial accepted-reward-at-onset statement is rejected: credit is earned at onset but paid at50/200ms. Gemini’s initial entire-reward-forfeiture suggestion is rejected; onlycurrentpaymentandfuturecontinuationareweighted,withoutretroactiveclawback.

Behavior evidence refutes universal faster-onset, uniformstiffening, longercontact, later-after50msrecontact and blanketlargermargins explanations. It does not prove that impedance alone or reducedclippingalone mediatedthegain. Models’ useof “sixworlds” for six trainedpolicymeans is corrected to sixseed/policycomparisons; eachmean summarizes32contexts. Model claims ofstatisticalsignificance, universalconstraintmarginimprovement or optimalpolicystructure are not adopted. The loadheadroomtrade-off, mechanicalstopimpulseconfound, hiddencriticmemory and rewardwindowdefinitionsremain explicit.

No large ablation study is required to retain the successful all-on objective as the user’s working baseline. Any later component-specific causal claim would need additional evidence. Existing-data margin/time/force/impedance diagnostics and separately defined held-out height evaluations are discussion items, not launched work.
