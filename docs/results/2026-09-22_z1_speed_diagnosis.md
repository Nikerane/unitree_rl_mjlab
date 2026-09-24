# Frozen-policy speed and load diagnosis

## Outcome

The three-policy E1 screen remains failed: none meets the predeclared speed-accuracy gate. The new replay localizes concerns without establishing a controller defect. No production code, reward, safety setting, or training checkpoint was changed; no new learning occurred.

Replay job45617041 completed exit0:0. All three frozen model49 checkpoints were evaluated at1.5/1.85/2.2m/s with128 first episodes per condition, same seed2026092101. Initial-state, CPU/CUDA RNG and first-action hashes match the original screen. All1152 first episodes hit and completed with exactly one speed-cost charge. Maximum paired onset-speed discrepancy was0.000105m/s, consistent with small numerical replay variation rather than a changed treatment. This is a repeat, not independent confirmation/generalization.

## What the traces establish

### Speed: both bias and scatter

Middle-cost policy at1.85m/s has median1.862m/s, but only37/128 episodes are within±0.07m/s; its10th–90th percentile spans1.646–2.099m/s. Strongest-cost policy at1.5m/s has median1.515m/s but only22/128 episodes within tolerance. At2.2m/s both responsive policies overshoot to median≈2.394m/s. A good median alone is misleading.

The weakest-cost policy's2.2m/s condition has69/128 errors≥0.7m/s, saturating the normalized squared penalty at1. Larger errors receive no additional speed-error cost there. This can limit discrimination of poor outcomes but does not establish causality or explain the other policies.

![Requested versus achieved speed](assets/2026-09-21_speed_e1_screen/trace-diagnostic/speed_scatter.png)

### Loads: accumulation continues beyond the first contact-containing step

Worst J6 replay is middle-cost/high-goal/world55. First contact is146ms. At the160ms control read, J6 window peak is1.462×cap and delivered impulse0.3149N·s. At180ms, J6 reaches4.465×cap while delivered impulse is0.3313N·s: approximately3.05×the earlier joint measure for only5.2%more delivered impulse. That episode then terminates successfully.

This focuses the next investigation on contact/follow-through, not merely the selected onset speed. However, J6 is the maximum baseline-subtracted, contact-gated50ms sliding-window joint reaction impulse observed inside a20ms control interval. This is not an instantaneous force peak, and its timing cannot identify exact substep forces or prove harmful pushing, stiffness causality, or mechanical damage. Values are simulation diagnostics against provisional caps.

![Worst J6 trajectory](assets/2026-09-21_speed_e1_screen/trace-diagnostic/worst_j6_timeline.png)

### Action clipping is real, but not yet a demonstrated bottleneck

Across all first-episode control-step/action-coordinate samples, position-target outputs exceed normalized±1 in27–40% of entries; stiffness outputs in5–18%. J4 target clips95–99% in the weak-cost policy and94% in middle-cost/high-goal. The inactive selector clips at most0.29%, so it does not account for the aggregate motor saturation. These are step-weighted frequencies, not fractions of episodes.

J6 position-target output never exceeds±1 in these traces; middle-cost/high-goal J6 stiffness output also never does. Thus the worst J6 load is not explained by direct J6 output clipping alone. Whole-arm coupling and post-contact control remain hypotheses. The observer does not record applied joint targets after physical clipping or per-substep actuator forces, so do not infer the true mechanical limit from normalized output clipping.

### Reference tracking

Separating control samples before the first-contact latch gives median pre-contact geometric error3.9–13.3mm across conditions, versus7.1–24.9mm for samples after contact has begun. The latter includes the contact-containing control interval. These are different lengths and phase distributions, not equal-time or causal comparisons. Geometry remains guidance; no timed target was introduced.

## Interpretation and smallest next experiment

No evidence here justifies changing the action mapping, spline, safety settings, or removing the speed cost. Plausible remaining explanations include incomplete learning, competing objective incentives, noisy observations and saturation; the traces do not isolate them.

The two responsive policies were still improving in logged training reward/speed-cost aggregates late in the50-iteration run. These training metrics do not prove convergence or deterministic evaluation improvement. The old baseline had250iterations×4096worlds, versus50×1024 here: **20×more training transitions**. Its lower J6 tail therefore cannot be attributed solely to absence of the new speed mechanism.

Recommended next proposal: resume the middle/high-cost candidates unchanged for100 additional iterations each (150 total), preserving seed, optimizer state, external goals, original gates and all safety settings. Compare both50- and150-iteration checkpoints on the same evaluation cohort; report every candidate, with held-out evaluation only after qualification. This is an optimization-duration check, not proof of safety or policy-selected-speed success. Do not extend the nonresponsive low-cost arm automatically. **Not launched; requires user decision.**

If speed accuracy and load tails fail to improve, use one bounded mechanistic intervention rather than silently adjusting multiple rewards or limits. Autonomous speed selection remains unqualified.

## Evidence and reproducibility

- Raw replay: `assets/2026-09-21_speed_e1_screen/trace-diagnostic/arm{0,1,2}/{trace.jsonl,episodes.jsonl,summary.json}`.
- Descriptive results: `assets/2026-09-21_speed_e1_screen/trace-diagnostic/diagnosis.json`.
- Observer/analysis scripts: `.superpowers/sdd/2026-09-21-vega-speed-qualification/{trace_screen.py,trace.sbatch,analyze_diagnostic.py}`.
- Observer CPU64-transition smoke passed before launch. Analysis run verifies checkpoint/reset/RNG/action identities, first-episode completeness, cost-charge counts, replay speed agreement and trace/episode peak agreement. Figures visually inspected.
- Original screen: `2026-09-21_z1_speed_e1_screen.md`. Independent agent checked speed-cost/load semantics, observer limitations, raw worst-world trace, all clipping fractions and timing tolerance; numerical conclusions agree. Review recommends a separately approved bounded extension rather than a controller change, with explicit safety-tail stop/report criteria frozen before launch. No claim that extra training will solve the failures.
