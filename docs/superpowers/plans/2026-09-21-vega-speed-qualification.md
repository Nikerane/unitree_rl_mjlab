# Vega Impact-Speed Qualification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish that desired impact speed is an effective policy command, then test autonomous speed selection with learned impedance.

**Architecture:** Preserve the approved single-policy direct joint-target and impedance controller. First qualify externally supplied speed goals (E1); then transfer that controller to autonomous once-per-stroke speed selection (E2). No spline timing controller or new training infrastructure.

**Tech Stack:** Existing mjlab/MuJoCo/PyTorch/CatPPO stack, existing Vega job tools.

**Spec:** `docs/superpowers/specs/2026-09-21-policy-selected-impact-speed-design.md`; launch requirements and promotion criteria in `docs/results/2026-09-21_z1_impact_speed_qualification_proposal.md`.

## Material Passport

- Origin Skill: experiment-agent; writing-plans
- Origin Mode: plan
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED experiment proposal; no remote availability check or job submission in this planning turn
- Version Label: code_plan_v1

## Global Constraints

- Preserve the original twelve action meanings, original reward weights and safety settings.
- Keep impulse soft-CaT `imp_max_p=0.2`, velocity soft-CaT `max_p=0.5`, existing joint impulse caps and VIC factor `C=1.25`.
- Start with the existing single left-start quintic-M20 geometry and 20 mm followthrough.
- Speed bounds and cost coefficient must come from matched baseline evidence; `(0.2, 0.8)` and `0.4` are software test fixtures, not physical calibration.
- E1 starts a fresh 13-output actor. The existing 12-output policy is a comparator, never its initialization checkpoint.
- GPU availability is not approval of an unfrozen campaign. This document proposes work, not launch commands.
- No main-worktree changes, broad refactor, silent retry, automatic budget extension or safety retuning.

## Review Focus

1. Wrong geometry/checkpoint: verify code, asset, task, reset and checkpoint identities together.
2. Goal merely predicts motion: require paired goal interventions and achieved-speed response, not only network-output changes.
3. Misses hidden by conditional statistics: report hit coverage and all failures alongside paired valid-hit estimates.
4. Faster hammer but worse impact: evaluate delivered impulse, tracking and joint loads jointly.
5. E1-to-E2 transfer mismatch: explicitly test autonomous first-step timing and fresh selector initialization.

## Task 1 — Freeze baseline and calibrated launch record

**Files:** update the existing qualification proposal and software result record; use existing evaluation tools. No production changes planned.

- [ ] Record the completed whole-branch review at `cc46d99` and combined 684-test acceptance in the software evidence record. Preserve the incomplete full historical-suite caveat.
- [ ] Check Vega connection, free storage, GPU allocation and installed versions without submitting training. Verify transferred source and assets against recorded hashes; do not overwrite an unrelated remote checkout.
- [ ] Locate and verify the geometry/recipe-matched 12-output comparator. If unavailable, report that fact and propose a matched baseline run rather than using an unrelated checkpoint.
- [ ] Replay 128 initial episodes with evaluation seed 2026092101. Save the actual reset population and identity, valid onset-speed distribution, contact coverage, event reward scale, impulse, tracking and joint loads. Repeated identical states do not count as independent training evidence.
- [ ] Derive a conservative candidate positive speed interval from those observations and physical limits; publish low/middle/high goals. If available evidence cannot support an informative interval, stop for a bounded reachability investigation.
- [ ] Set a proposed coefficient sweep at 0.1, 0.3 and 1.0 times the median positive task return accumulated through first-event finalization among valid baseline strikes. Publish the resulting numeric coefficients and exact return definition before training. If that reference scale is zero or undefined, do not apply this rule.
- [ ] Freeze exact command lines using the installed CLI, source/config/asset hashes, output directories and job limits in the qualification proposal. Commands cannot be executable before checkpoint and physical parameters are established; that is an explicit launch gate, not permission to substitute test values.

**Deliverable:** reviewed numeric run record and a short baseline video. Request approval of this record before job submission.

## Task 2 — Native Vega smoke and E1 responsiveness pilot

**Files/tools:** existing `scripts/train.py`, `scripts/evaluate_impact_speed.py`, qualification proposal and a new dated result record; no new launcher.

- [ ] Re-run repository pre-training gates on the exact remote runtime: reward phases A–M, contact sensor and reward setup, plus experimental contract/policy tests. Existing local passes do not establish remote compatibility.
- [ ] Run a two-iteration GPU smoke with 256 worlds; check finite updates, action masks, saved checkpoint reload and event telemetry. Discard smoke weights from the main pilot.
- [ ] Proposed E1 screening budget: three coefficients, training seed 2, 1,024 worlds, 50 iterations each; save every 25 iterations. Hold all other settings fixed. Record transitions and elapsed GPU time, not only iterations.
- [ ] Evaluate low/middle/high goals on paired initial episodes: 128 per goal, evaluation seed 2026092101, bootstrap seed 2026092103 with 2,000 paired resamples. Choose the lowest coefficient meeting all existing E1 gates; if none passes, report all three outcomes rather than silently enlarging the sweep.
- [ ] If a candidate passes, confirm it on a separate 128-episode population with seed 2026092102. If screening shows promising but incomplete learning, propose continuing one selected run to a maximum of 250 total iterations; review that decision first. No automatic 500-iteration run.

**E1 gates:** hit rate at least 90% of matched baseline; median absolute speed error among hits at most 10% of interval width; paired high-minus-low achieved-speed 95% confidence interval above zero; tracking RMS at most 10% worse than nonzero baseline, with absolute millimetres reported. Report unchanged safety treatment and all load/exceedance statistics. These are provisional qualification criteria, not proof of hardware safety.

**Monitoring proposal:** smoke cap 30 minutes; each screening job cap 2 GPU-hours; any approved extension cap 8 GPU-hours. Check process state, iteration advancement and checkpoint/log freshness using existing job tooling. Report failures or non-finite values immediately; do not automatically retry or change settings. Hard time limits end jobs; lack of progress triggers inspection. These caps are proposals to freeze with the launch record.

## Task 3 — E2 autonomous speed selection

- [ ] Only after E1 passes, transfer its qualified 13-output checkpoint through the existing explicit qualification-transfer path. Preserve learned controller/critic/normalizers; initialize the selector and fresh optimizer as designed.
- [ ] Proposed initial E2 budget: seed 2, 1,024 worlds, 50 iterations; save every 25. Evaluate before approving continuation up to 250 total iterations and an 8 GPU-hour cap.
- [ ] Measure selected versus achieved speed, selector saturation, actual impulse, hit rate, curve tracking, stiffness and joint loads. Do not reward the requested speed itself or equate velocity with impulse.
- [ ] Run separately labeled post-latch goal interventions on paired initial states. Verify both later commands and achieved speed respond; the first action remains unchanged by design.
- [ ] If responsiveness is lost or waiting/missing dominates, stop and diagnose. Do not introduce a planner, dense velocity reward or indefinite external-goal mixture.

**Deliverable:** autonomous-policy checkpoint plus evidence of effective speed conditioning; no optimality claim from one seed.

## Task 4 — Visual review, then three curves

- [ ] Produce videos with the thin red reference, and white-background diagrams showing nail, reference and actual hammer-head trace. Include requested and achieved speed, with misses explicitly marked.
- [ ] Present one-page baseline/E1/E2 comparison before extending scope. E1 and E2 have different first-step goal availability; they are not a perfectly controlled ablation.
- [ ] After one-curve success, inspect whether existing observations distinguish the three left/centre/right curves. Propose a separate three-curve run record using the existing curves; do not silently add a spline encoder or three new controllers.
- [ ] Use independent training seeds and budget-matched freshly trained baselines before scientific improvement claims. Reusing an old comparator is adequate for pilot diagnostics, not a fair learning-efficiency claim.

## Handoff

Recommended next execution is Task 1 only: baseline recovery/replay and a completed numeric run record. Vega is useful for that replay and subsequent GPU qualification. The user should review this staged plan before execution; no remote job was launched while writing it.
