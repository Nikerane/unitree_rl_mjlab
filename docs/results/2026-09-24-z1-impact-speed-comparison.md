## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-24
- Verification Status: ANALYZED (saved results analyzed; training not reproduced)
- Version Label: paired_pilot_analysis_v1
- Source: Vega campaign `/cephhome/eunikhilr/campaigns/z1-impact-speed-retry-20260924.IGsdQz`, jobs `45890150_0` through `_5`, source `eafb7959c11c35a6963d9418cc7c825153813d1c`.

## Outcome

All six runs completed 50 iterations and both evaluation protocols, with strict checkpoint reload passing. Learned impedance has better mean impact, speed error and geometric tracking in this exploratory pilot. Neither treatment achieves the prescribed joint-load safety objective. No hardware safety or optimal-impedance claim is supported.

## Fair comparison and measurement

Use each run's `completion.json` and `eval-common-window/episodes.jsonl`. Keep only `episode == 0`; sort by `(condition, world)`. Assert 384 unique records/run, 128 per goal (conditions 0/1/2 = 1.5/1.85/2.2 m/s). Initial `reset_hash` sequences match across all six policies, checked separately under each measurement protocol. This avoids overweighting extra asynchronous reset episodes, especially run 0. Group goals by condition index, not exact equality with decimal-valued float32 goal tensors.

Three paired training seeds (10, 11, 12), 384 first attempts/seed, 1,152 per treatment. All selected attempts registered contact; all selected common-window records contain the complete 200-sample / 50 ms impact window. All have full 16-bin approach coverage. These are three independent training replicates, not 1,152 independently trained policies; goals reuse initial states.

Statistics below are equally weighted attempt means, equivalent to equal seed/goal means because counts match. No p-values, confidence intervals, or population-superiority claim. Common-window protocol defers success resets to observe the impact; it is not identical to native deployment termination. Joint-load exceedance is any recorded episode peak above its prescribed cap under this protocol, not a full safety certificate. The load signal is the authoritative peak-held signal `max(current rolling Lambda, current control-step peak)`, including relevant pre-onset history. It is not instantaneous torque. Other constraints, such as joint velocity, are not certified by this table.

## Main comparison

| Metric | Fixed impedance | Learned impedance |
|---|---:|---:|
| Mean positive axial impulse, common 50 ms window (N s) | 0.37967 | 0.43202 |
| Mean measured contact-onset speed (m/s) | 1.09926 | 1.39133 |
| Mean absolute target-speed error (m/s) | 0.75250 | 0.47799 |
| Mean precontact observed-bin RMS path error (mm) | 13.24735 | 10.91511 |
| Mean per-attempt maximum precontact path error (mm) | 22.32609 | 17.84576 |
| Attempts exceeding any joint-load cap | 544/1152 (47.22%) | 533/1152 (46.27%) |
| Contact / complete impact-window coverage | 100% / 100% | 100% / 100% |

Descriptively, learned impedance has 13.8% greater mean axial impulse, 36.5% lower speed MAE and 17.6% lower observed-bin path RMS. Overall cap exceedance differs by only 0.95 percentage points; this is not evidence of a reliable safety improvement.

Path error is `measurement.precontact_path.observed_bin_rms_m * 1000`, averaged over attempts. This balances sampled approach bins within each attempt; it is not an exact continuous geometric distance or guaranteed timing-invariant metric. Complete bin coverage establishes sampled traversal, not close tracking by itself.

## Seed sensitivity

| Training seed | Fixed / learned impulse (N s) | Fixed / learned speed MAE (m/s) | Fixed / learned any-cap exceedance |
|---|---:|---:|---:|
| 10 | 0.32880 / 0.46470 | 0.65391 / 0.40866 | 69.79% / 55.47% |
| 11 | 0.38849 / 0.42651 | 0.87592 / 0.61091 | 44.53% / 40.89% |
| 12 | 0.42173 / 0.40487 | 0.72768 / 0.41440 | 27.34% / 42.45% |

Learned impedance improves speed error and path error in all three paired seeds, but impulse improves in only two; seed 12 has lower impulse and more cap violations than fixed impedance. Do not select only the best seed or claim consistent dominance.

## Terminal-speed response

| Requested speed (m/s) | Fixed measured mean (m/s) | Learned measured mean (m/s) |
|---|---:|---:|
| 1.50 | 1.19321 | 1.09032 |
| 1.85 | 0.96186 | 1.24610 |
| 2.20 | 1.14270 | 1.83758 |

The learned treatment's pooled mean speed rises with the requested goal, but seeds 11 and 12 still have lower speed at the middle goal than the low goal. Goal following is incomplete, not a qualified terminal-speed controller. These runs use externally supplied speed goals: they do not yet demonstrate autonomous terminal-speed selection. They cover only the LEFT spline, not all starts or human demonstrations.

## Joint-load detail

| Joint | Cap (N m s) | Fixed exceedance | Learned exceedance | Fixed / learned maximum observed peak (N m s) |
|---|---:|---:|---:|---:|
| 1 | 0.369 | 0% | 0% | 0.29996 / 0.23516 |
| 2 | 0.246 | 11.02% | 19.18% | 0.42147 / 0.45172 |
| 3 | 0.738 | 0% | 0% | 0.68827 / 0.67889 |
| 4 | 0.369 | 0% | 0% | 0.36770 / 0.33910 |
| 5 | 0.246 | 0% | 0% | 0.21311 / 0.13056 |
| 6 | 0.0164 | 40.19% | 39.15% | 0.08874 / 0.07274 |

Joints 2 and 6 dominate observed violations. The learned treatment's J2 violation rate is higher, while J6 changes little in frequency. Maximum observed J6 load is approximately 5.41 times cap (fixed) and 4.44 times cap (learned). Zero observed violations on another joint are not a guarantee.

There are 608 fixed and 619 learned attempts without any recorded joint-load exceedance. Their conditional mean impulses are 0.36208 and 0.37648 N s respectively. This selected-subset diagnostic is not an unconditional treatment effect: conditioning on load compliance changes the populations being compared, and it does not establish compliance with other safety constraints.

## Why native-only impact comparison would mislead

Separate native/common-window evaluations are not exact paired physical prefixes, even when initial reset hashes match. Their violation-count difference is a protocol contrast, not a count of proven post-success-only violations. Same-rollout success markers and load traces are required for that timing attribution.

Native termination truncates the impact window in 746/1152 fixed records (64.76%) and 933/1152 learned records (80.99%). Complete-window native-only means would compare differently selected subsets. The common-window evaluation yields 1152/1152 complete windows per treatment, so it is used for the primary impact comparison. Missing native windows are not treated as zero or silently discarded. Native any-load exceedance is 477/1152 (fixed) and 349/1152 (learned), but those shorter observation horizons cannot substitute for the complete-window safety comparison.

## Interpretation and next decision

This short pilot supports continuing to investigate learned joint impedance, not declaring the research question solved. The immediate bottleneck is load compliance, especially J2/J6; simply increasing speed would not address it. Before a larger campaign, inspect the learned gain profiles around contact and the safety-cost/impact trade-off while keeping prescribed caps unchanged. Longer matched-seed training can then test whether these failures reflect incomplete learning. Autonomous speed selection and three-start transfer remain separate later tests. No new jobs, controller edits, or safety-limit changes were made for this analysis.

## Validation cautions and fallacy scan

Overall confidence: CAUTION; descriptive exploratory result, three training seeds, one evaluation reset seed, one path, 50 iterations. No stochastic rerun was performed; reproducibility beyond saved-result calculation is not assessed.

11/11 fallacy types checked:

1. Simpson's paradox: seed and goal strata examined; seed 12 reverses the pooled impulse/safety direction, explicitly reported.
2. Ecological fallacy: treatment means are not claimed to apply to every attempt or seed.
3. Berkson selection: load-compliant subset is conditional and not the primary comparison.
4. Collider bias: no regression adjustment; conditioning on compliance is labeled potentially selected.
5. Base-rate neglect: contact, completeness and violation denominators all reported.
6. Regression to mean: no selection of extreme checkpoint/seed; fixed final iteration used.
7. Survivorship: all first attempts retained; native censoring and complete-window coverage explicit.
8. Look-elsewhere: all six runs and all three goals considered; no significance testing or best-seed promotion.
9. Forking paths: exploratory analysis; first-episode filter and metric definitions explicit, not claimed preregistered confirmatory testing.
10. Causal overreach: controlled simulator treatment comparison only; no human/hardware or general-policy causal claim.
11. Reverse causality: treatment assigned before training/evaluation; not inferred from observed load/impact correlation.
