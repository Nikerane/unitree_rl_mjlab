# Fresh speed-conditioned impulse-CaT0.2/0.3 pair

User explicitly approved the fresh matched pair in`docs/superpowers/plans/2026-09-22-speed-impulse-dose-pair.md`.

## Frozen treatment and decision

Fresh seed2 models,150uninterruptedupdates,1024worlds×24steps/update; impulse probability ceiling0.2control versus0.3treatment. Middle speed-error coefficient0.8493676, external uniform speed goals[1.5,2.2]m/s; controller, reference, rewards, velocityCaT0.5, joint caps and all other settings fixed. Sourcecc46d99/assetsb58ccd2. Historical resumed150policies are context only. Exactly two runs, no automatic retry/extension/autonomy.

Each finalmodel149 evaluated at1.5/1.85/2.2m/s,128firstepisodes/goal,seed2026092101. Compare J6/all-joint frequency,p95,maximum and velocity tails; actual delivered impulse and hit rate; unchanged accuracy/tracking criteria. Prospective utility guard:≥116hits/128,≥90%control median delivered impulse,trackingmedian≤21.622820mm at everygoal. Original speedqualification≤0.07m/s at everygoal remains unchanged. A worsened safety tail is a trade-off, not an overall safety improvement.

## Preflight

Existing CLI dose override parses at both0.2and0.3; no productionchanges. New evaluator adapter uses the existing registry loader to configure dose before strict load, preserving metadata validation. Config test confirmed onlythe dose changes and noRNGis consumed.64transitionCPU .3smoke and .2strictcheckpointreload passed with zero learning; sums of rewardterms/delta arithmetic/native per-joint velocitypeak shape verified. .3load of a.2checkpoint correctly rejected with safety-metadata mismatch. Training observer zero-updateCPU probe passed and wrote initialactor/physical/RNGhashes; this wrapper only observes runner initialization then delegates to native train.

Campaign:`/cephhome/eunikhilr/campaigns/z1-speed-dose-pair-20260922.28P0cS`. Scripts:`.superpowers/sdd/2026-09-21-vega-speed-qualification/{dose_train.py,dose_eval.py,dose_pair.sbatch,test_dose_adapter.py}`, observational dependencies`trace150.py,trace_screen.py`.

Ruling: added a read-only initialization wrapper to bank initialactor/physical/RNGidentities required by the plan. No controller/trainingalgorithm alteration; if hash parity fails, do not claim a matched-initialization comparison. Scratch artifacts remain beside the existing observational helpers named by the approved plan; no unrelated source edits or cleanup.

Status: preflightpassed; launchreview/submission/results recorded below.

Independent launch review approved after the zero-update initialization probe passed. No blockers or productionchanges.

Submitted once as array **45693109_[2–3]**. Per-arm45minute hard limit includes gates/training/evaluation; no automatic requeue.

## Completed result — severity improves, frequency does not

Both jobs completed exit0:0 without retry:0.2in8m14s,0.3in8m11s including gates and evaluation. Each passed198tests/51upstreamwarnings and native reward/contact/setup gates, then completed exactly150updates(labels0–149). Total7,372,800training transitions. Initialactor/physical/CPU-and-CUDA-RNGhashes match exactly. All evaluation reset/RNGidentities match, and livecontracts differ only in impulse ceiling. Finalcheckpoint hashes and all-tensor finiteness verified.

All six goalconditions hit in128/128firstepisodes. Both remain goal-responsive, but neither passes the unchanged complete E1accuracy requirement.

| Goal m/s | Median absolute speed error0.2→0.3(m/s) | Median delivered impulse0.2→0.3(N·s) | Retention | J6p95/cap0.2→0.3 | J6maximum/cap0.2→0.3 | Overcapepisodes0.2→0.3 |
|---|---:|---:|---:|---:|---:|---:|
|1.50|0.1957→0.0969|0.3006→0.2946|97.985%|1.644→1.350|2.160→1.770|128→128|
|1.85|0.0802→0.0609|0.3532→0.3186|90.200%|2.002→1.330|2.284→1.519|128→128|
|2.20|0.0792→0.0527|0.3618→0.3635|100.459%|2.239→1.566|2.412→1.599|127→127|

All prospective utility guards pass, but middle-goal impulse retention is only narrowly above90%. Treatment tracking medians12.13/11.24/10.14mm remain below21.622820mm, though higher thancontrol10.76/10.03/10.07mm. The0.3low-goal speed error0.09686still exceeds0.07m/s. High-minus-low speed contrasts remain positive:0.2mean0.5761,95%CI[0.5581,0.5932];0.3mean0.5665,[0.5519,0.5806]. These intervals describe paired evaluations of single policy instances, not training-seed uncertainty.

**Decision:** no overall safety improvement or E1qualification. J6severity decreases but its exceedance frequency is unchanged, so the intended frequency-and-severity improvement is not established. Keep0.3as a promising diagnostic result, not a qualified deployment policy. No automatic further dose increase, extension, or autonomous-speed training.

### Other tails and effective pressure

Every joint's maximum impulse decreased at all goals. However, low-goal J4impulse p95increases0.50351→0.51032×cap. Low-goal J2/J3velocity maxima rise2.9238→2.9610and2.6257→2.7453rad/s (both below3.1415); high-goal J4velocity p95rises2.77168→2.89910rad/s. These trade-offs are retained, not hidden by lower global maxima. J6velocity exceeds3.1415rad/s in128/128episodes at everygoal for bothdoses; its maximum decreases4.60976→4.58020rad/s but remains overlimit. Fullper-joint maxima,p95and counts are inanalysis.json.

Mean impulse delta on J6violating evaluation reads0.2→0.3:low0.0658→0.1679,mid0.1580→0.1531,high0.0736→0.1696. Thus the configured50%higher ceiling is **not uniformly greater realized evaluation pressure**. These probabilities use fresh evaluation normalizers and different learned trajectories; they do not reconstruct historical PPOgradients or prove that increased pressure alone caused a particular behavioral change.

Independent review recomputed raw metrics and confirmed utilitypasses, unchangedfrequency, E1failure and the additional p95/velocity trade-offs. Analyzer explicitly checks actual mode/seed/goals rather than assigning goal labels only by listposition; fullE1status additionally includes the positive high-low response criterion.

## Banked artifacts and stopping point

`docs/results/assets/2026-09-22_speed_dose_pair/analysis.json`;`dose{2,3}/` contains initialidentity,train/testlogs,finalmodel149, evaluationepisodes/summary and terminal-inclusive trace. FinalcheckpointSHA256:

-0.2:`b58f539c1439c550e7a5663a5b0f9f82dbabc73a1c9b55a5fb0db0210d01a14f`.
-0.3:`427aa85a325b79b7fc345e0f79941e98e5fa27b31b49a7d3a15654e29e66b69c`.

Production source remains unchanged. Single trainingseed/fixedphysicalstart; no hardware/generalization/optimal-dose claim. Completed the authorized pair and stopped with no further jobs.
