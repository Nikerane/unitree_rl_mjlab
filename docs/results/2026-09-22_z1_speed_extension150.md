# Two-candidate E1 extension to150 total updates

User authorized extending the two responsive policies. No autonomous selector training, new seed, reward/controller change, safety retuning, or automatic follow-on is authorized here.

## Frozen execution

- Same clean Vega source`cc46d9998a3c9cd5d8813e89b57e44248353e03e`, assets`b58ccd2f81fd246f27c1e8d88cf86484cd888703`.
- Resume arm1(coefficient0.8493676) and arm2(2.8312253) finalmodel49, verifying their original SHA256 identities; load actor, critic, normalizers and optimizer with native resume. No12D transfer or fresh initialization.
-100additional updates each;1024worlds×24steps, seed2, external uniform goals[1.5,2.2]m/s. Additional2,457,600transitions/candidate;150actualupdates and3,686,400totaltransitions/candidate.
- Upstream runner stores completed iteration49 and resumes its loop from49. Consequently100new updates have labels49–148 and finalcheckpoint`model_148.pt`; this is150actualupdates, not149. Training log must contain exactly100new iteration entries.
- Native resume restarts environment/RNG; it is not a bitwise uninterrupted rollout. Saved optimizer state is restored through the native loader; no custom scheduler/resume patch.
- Reward/contact/setup gates and198targetedtests rerun before each arm. Any gate/process failure stops that arm; no automatic retry. NaN checking remains enabled.45minute hard limit per GPU including evaluation.
- Evaluate finalcheckpoint at1.5/1.85/2.2m/s with128firstepisodes each, seed2026092101. Original speed/hit/tracking gates unchanged. Compare all results with own50-update checkpoint and matchedbaseline, retaining all misses and failures.

## Safety/decision boundary

Report per-joint maxima and exceedance counts for every condition, alongside impulse and speed accuracy. Do not call a candidate improved in safety if its maximum ratio or exceedance count worsens relative to its own50-update matched condition; report trade-offs explicitly. Existing provisional-cap exceedances remain safety failures regardless of speed accuracy. No claim of hardware safety, hard enforcement, optimal impedance, or E2qualification follows from longer training.

Stop after these two evaluations. Worsening safety tails or persistent accuracy failure requires a new decision; no additional training or controller edits. This is a bounded simulation diagnostic, not an online hard safety stop.

## Artifacts/status

Launcher: `.superpowers/sdd/2026-09-21-vega-speed-qualification/extend150.sbatch`. Uses original arm directories for native checkpoint discovery, unique`extension150`evidence subdirectories and new`*_extend150`run names; original runs/checkpoints remain untouched. Scheduler/results to be appended after submission.

Submitted once as array **45687007_[1–2]**, after complete local argument parsing, shell syntax checks, remote destination checks and independent launch approval. Parsing-only checks caught the need for explicit`--agent.resume True` before any submission; no GPU attempt failed for that issue.

## Completed results

Both jobs completed exit0:0 without retry: arm1 in5m42s, arm2 in5m51s including gates and evaluation. Reward/contact/setup gates and198tests passed per arm (51upstreamwarnings). Training logs contain exactly100new updates, labels49–148. Downloaded checkpoints match evaluator SHA256, all tensors finite, savediteration148. Native resume paths recorded in logs point to the intended model49.

All six conditions achieved128/128first-episode hits and complete valid measurements; reset/RNG comparisons pass. Every median absolute speed error improved relative to50updates, but **neither policy qualifies**. The original gate remains≤0.07m/s at every speed, not a rounded approximation.

| Policy cost | Requested m/s | Median achieved m/s | Median absolute error:50→150updates m/s | Tracking RMS mm | Delivered impulse N·s |
|---|---:|---:|---:|---:|---:|
| Middle0.8493676 |1.50|1.6210|0.2488→0.1210|15.54|0.2846|
| Middle0.8493676 |1.85|1.9202|0.1223→0.0701543|15.95|0.3119|
| Middle0.8493676 |2.20|2.2115|0.1957→0.0304|16.25|0.3495|
| High2.8312253 |1.50|1.5313|0.1442→0.0772|10.35|0.3292|
| High2.8312253 |1.85|1.9024|0.2559→0.0806|10.55|0.3606|
| High2.8312253 |2.20|2.1860|0.1979→0.0854|18.32|0.3811|

Only middle-cost/high-goal passes the per-goal speed-accuracy requirement. All tracking medians pass the frozen21.622820mm whole-episode gate, though all are higher than their own50-update values. Middle-cost/high-low paired mean speed difference0.5915m/s,95%CI[0.5823,0.6005]; high-cost0.6968,[0.6622,0.7296]. Both remain responsive. These are within-policy evaluation intervals, not training-seed uncertainty.

### Mixed safety result — no safety improvement claim

J6 worst peak ratios fell in every condition, but exceedance counts rose in every condition:

| Policy | Goal m/s | J6 maximum/cap:50→150 | Over-cap episodes:50→150(out of128) |
|---|---:|---:|---:|
| Middle |1.50|3.336→2.026|29→122|
| Middle |1.85|4.021→2.071|58→128|
| Middle |2.20|4.465→2.235|48→128|
| High |1.50|3.411→2.829|115→128|
| High |1.85|3.254→2.725|88→127|
| High |2.20|3.477→3.290|99→121|

Middle-cost has no J1–J5exceedances now. High-cost retains some J1/J2/J5exceedances, and its high-goal J3maximum increases0.897→0.967×cap while remaining belowcap. J6median ratios are1.566/1.593/1.539(middle) and1.751/1.877/2.036(high). Thus rare larger peaks became smaller but more persistent violations; the predeclared mixed-safety reporting rule is triggered. These remain50ms window reaction impulses, not instantaneous forces or hardware damage limits.

Delivered impulse fell in five of six conditions (high-cost/low-goal increased). Improved speed accuracy therefore does not imply improved impulse maximization. No autonomous selector training, held-out promotion, reward change, or further extension launched. Stop at the approved boundary; next decision should address accuracy/safety/impulse trade-offs rather than automatically add iterations.

### Banked evidence

Local: `docs/results/assets/2026-09-22_speed_extension150/analysis.json` and`arm{1,2}/` containing fulltraininglog, gates/tests, evaluationrows/summary andmodel148. Later-autoreset rows excluded from primary analysis: middle101/0/0,high123/121/125 for low/mid/highgoals.

Checkpoint SHA256:
- Arm1:`0a613a46e43e6c73ce6cbcb8633e9deae30d471225d1fad53c9d81d8a955ed0d`.
- Arm2:`7989f97da445cc5ab43ec1bea0936163013699514c8375995a1a00f3fd46a232`.

Source/controller/reward implementation unchanged. All original50-update checkpoints preserved.

Independent final review recomputed all six accuracy results from raw first-episode rows, verified reset/RNG parity against the old screen and baseline, and confirmed the mixed J6 result and`selected_arm=null`. No policy promotion is supported.
