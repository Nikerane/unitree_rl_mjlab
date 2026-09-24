# Impact-speed GPU preflight and matched baseline

## Scope

User-approved checks only: protocol-matched frozen baseline, native pre-training gates, two-iteration GPU smoke and strict checkpoint reload. No50-iteration screening campaign, E2 training, merge or push. Source `cc46d9998a3c9cd5d8813e89b57e44248353e03e`, assets `b58ccd2f81fd246f27c1e8d88cf86484cd888703`; separate clean detached Vega checkout. Evaluation helpers are artifacts, not controller changes.

Remote evidence root: `/cephhome/eunikhilr/campaigns/z1-speed-preflight-20260921.O9UyYJ`.

## Native gates

Job45613873 passed reward phasesA–M, GPU contact verification (first contact step5), reward setup, and **198 focused tests**,51warnings,15.76s. Test files: impact-progress, impulse-bound, impulse-constraint, delivered-impulse, soft-CaT hook, impact-speed policy and contract. Warnings were upstream TorchScript/ONNX warnings; no full-suite claim. Random-policy rollout produced no contact in200steps, so that diagnostic did not calibrate thresholds.

The job then failed before learning because training's Tyro parser requires one tuple argument `--speed-bounds '(1.5,2.2)'`, not two separate arguments. The evaluator's argparse interface does accept separate numbers. Full corrected training arguments were parsed locally without constructing an environment. The user explicitly approved one smoke-only retry; passed gates were not rerun.

## Protocol-matched baseline

Job45614079 COMPLETED in26seconds, exit0:0. The verified12D model249 checkpoint was evaluated for the first episode of each of128parallel worlds, seed2026092101. All128 completed, hit and succeeded.

| Metric | Median | Observed range |
|---|---:|---:|
| Impact speed | 2.212232m/s | 2.100386–2.309665m/s |
| Delivered impulse | 0.3595285N·s | 0.339802–0.374631N·s |
| Whole-episode native phase-tracking RMS | 19.657109mm | 19.497407–19.828570mm |

J6 exceeded its provisional cap in128/128 episodes; maximum utilization1.057581. Other joints did not exceed their impulse caps. The resulting proposed median tracking limit for this matched protocol is **21.622820mm** (baseline median×1.10), replacing the sequential replay's provisional threshold.

The evaluator-only adapter consumes an external-goal-sized random draw at action-manager reset and recomputes actor/critic observations in the external evaluator's order. It adds no action or goal to the12D controller. Independent scoped review and CPU two-world checks passed. A separate untrained13D reset-only GPU probe proved exact agreement for all128 physical/common-first40-observation hashes before and after goal override, plus CPU/CUDA RNG hashes at both points. The probe took zero simulation steps and zero learning updates. Matching initial state/RNG does not guarantee identical later noise streams after policies terminate at different times, and does not establish robustness to different physical starts.

Evidence: [matched episodes](assets/2026-09-21_speed_preflight/matched-baseline/episodes.jsonl), [matched summary](assets/2026-09-21_speed_preflight/matched-baseline/summary.json), [exact GPU reset comparison](assets/2026-09-21_speed_preflight/reset-match.json).

## Smoke retry

Job45614091 **COMPLETED in3m17s, exit0:0**. It used external uniform goals[1.5,2.2]m/s,cost0.8493676,seed2,256worlds,24steps/iteration and2iterations (12,288 training transitions). It started fresh and retained existing controller/reward/safety settings. These smoke weights must not initialize the qualification screen.

The final `model_1.pt` contains the correct13D/42D external-mode contract, bounds,cost,caps and gain factor. Recursive inspection found zero nonfinite tensors across315,988 tensor elements (including optimizer/model state). Saved rollout telemetry reports6,144 samples for the final24-step rollout and **zero active speed selections**, as required for external-goal training. This checks the inactive selector semantics; it is not autonomous selector training.

The saved actor strictly reloaded and completed6evaluation episodes across1.5/1.85/2.2m/s,two worlds each,6,000 evaluation transitions,zero additional learning updates. All6timed out without contact and each charged the missed-event cost exactly once. All three responsiveness diagnostics were false. Thus this is a **software smoke pass, not an E1 performance pass**; there is no evidence of achieved speed control after only two iterations.

Checkpoint local/remote SHA-256: `13e195bf6361e64c7269b736aea7a7ca172f07e7e29d3d5053e1ff025427372f`. Local checkpoint is retained as `docs/results/assets/2026-09-21_speed_preflight/smoke_model_1.pt` for debugging only. Reload [summary](assets/2026-09-21_speed_preflight/reload/summary.json) and [episodes](assets/2026-09-21_speed_preflight/reload/episodes.jsonl) preserve the failed task outcomes rather than labeling them qualification success.

The final scoped reviewer checked the additional GPU post-override RNG field and reset probe, resolving the earlier CPU-only parity caveat. No production code was changed in these checks; CLI correction and reset-alignment instrumentation are confined to campaign artifacts.

## Remaining gate

These checks do not establish learned speed responsiveness, optimal impedance or safety. No E1 promotion follows from a two-iteration smoke. The three50-iteration coefficient candidates remain unlaunched.
