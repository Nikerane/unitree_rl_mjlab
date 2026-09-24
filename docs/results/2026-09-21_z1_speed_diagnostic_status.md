# Speed diagnostic handoff

**Update2026-09-22:** Download and analysis are complete; see `2026-09-22_z1_speed_diagnosis.md`. Historical handoff below preserves the earlier stopping point. No further training launched.

User requested a concise summary before tokens run low; diagnosis paused at this evidence boundary.

- Three 50-iteration E1 policies and nine 128-world goal evaluations are complete; see `2026-09-21_z1_speed_e1_screen.md`. None qualifies; two respond to goals, all hit, precision and joint-load tails remain unresolved.
- Independent read-only audit confirmed onset-speed/cost semantics and observer validity. Saturated speed cost occurs in 69/128 low-coefficient/high-goal episodes. Action-clipping aggregates cannot yet distinguish motor and inactive selector coordinates; no causal clipping claim.
- Diagnostic observer CPU smoke passed 64 transitions. Evaluation-only Vega job **45617041** completed exit0:0 for all three frozen checkpoints, three goals each,128 worlds; **zero learning**, no controller/reward changes.
- Remote evidence: `/cephhome/eunikhilr/campaigns/z1-speed-e1-screen-20260921.QKg3yq/trace-diagnostic/arm{0,1,2}/` contains trace/episodes/summary; logs adjacent. Not yet downloaded or analyzed.
- Observer and proposed analysis: `.superpowers/sdd/2026-09-21-vega-speed-qualification/{trace_screen.py,trace.sbatch,analyze_diagnostic.py}`. Analysis script is drafted but **not run or verified**. Next: download evidence into `docs/results/assets/2026-09-21_speed_e1_screen/trace-diagnostic/`, run/check analysis, inspect figures, then bank reviewed conclusions.
- Trace J6 values are 50ms-window maxima observed in each20ms control interval, not exact force-peak timestamps. Onset speed/time are separately latched. `nail_depth` observer field is peak depth, not instantaneous depth. Clipping is not proof of a mechanical limit.
- No longer training, safety retuning, or autonomous E2 is underway. No automatic extension authorized. User wants summary now.
