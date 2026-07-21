# Maximization ablation — is the forward-swing caused by impact-maximization, or is it kinematic? (2026-07-20)

**Question (from the trajectory analysis; scripts `evaluation/trajectory/`, figures `evaluation/results/2026-07-20_fic_maximization/`):** every competent July
policy strikes with a **forward-swing arc** (head arcs ~4–7 cm past the nail-x, then back down). Is that
swing a **learned impact-maximization maneuver** (a wind-up to build contact speed), or an **incidental
kinematic reaching signature** of this arm striking from directly above?

**Early partial evidence (existing policies, single fixed reset):** the swing does **not** buy contact
speed — the two straightest successful strikers (`ip24` +0.5 cm, `g1_track` +1.0 cm) reach the *highest*
v_touch (1.41, 1.39 m/s); the ~4.6 cm group spans v_touch 0.99–1.61. And `g2_delivoff` (delivered reward
off) still swings 4.6 cm. This *leans* kinematic — but is **not decisive**, because every existing policy
still has the `impact_progress` speed reward ON. This experiment removes that confound.

## Design — a 3-point dose–response

One knob: the **impact-maximization reward mass** = `impact_progress` (ante-impact contact speed) +
`delivered_impulse` (object-side ∫F·dt). Everything else identical (fixed impedance, same scene, same
seeds, same 500 iters / 4096 envs, `imp_max_p=0` log-only, no VIC). The task keeps `approach`,
`nail_depth_delta` (600), and `completion` (100) so it still learns to **drive the nail** with zero
speed/impulse incentive.

| arm | `impact_progress` | `delivered_impulse` | how |
|---|---|---|---|
| **max-off** | **0** | **0** | `IMPACT_W=0 DELIVERED_W=0` — train (3 seeds) |
| **max-on** | 8 | 2 | = **`af1`** (already trained, `6b447bf`; training code byte-identical to HEAD `e7d9e6b` — verified `git diff --stat 6b447bf HEAD -- src/tasks/hammer src/assets` is empty) → **no retrain needed** |
| **max-max** | 24 | 4 | `IMPACT_W=24 DELIVERED_W=4` — train (3 seeds) |

Task = `Unitree-Z1-Hammer-CaT-Impulse` (the **none** arm — no imitation prior; the `-Track` prior would
*confound* by teaching the reference's apex→descent swing shape). Matches how `af1` was trained.

### Predictions (pre-registered)
- **If the swing is FOR impact-maximization** → pre-contact forward-swing **increases with dose**:
  max-off ≈ straight (≪4 cm), max-max ≫ 4 cm. (And max-off may strike slower / shallower.)
- **If the swing is kinematic** → forward-swing is **~constant (~4 cm) across all three doses**; max-off
  still swings despite zero speed incentive.

Secondary read: does removing the maximization reward **hurt task success or contact speed**? (If max-off
still succeeds at ~1.4 m/s, the maximization reward isn't even buying impact on fixed impedance — consistent
with the Phase-0 velocity ceiling.)

## Pre-training gates — all GREEN (2026-07-20, local CPU)
- `pytest` impulse/reward suite (5 files): **80 passed**.
- `validate_rewards.py`: **all phases A–M PASSED** (incl. Phase I `impact_progress`, Phase M impulse-CaT).
- `verify_contact_sensor.py`: **verified**, first contact step 5.
- Override sanity: `train.py … --env.rewards.impact-progress.weight 0 --env.rewards.delivered-impulse.weight 0`
  parses cleanly (tyro accepts the paths; fails only at GPU-select on the CPU Mac, as expected). The sbatch
  already wires `IMPACT_W`/`DELIVERED_W` (the `g2_delivoff` precedent).

## Launch (Vega — submit from an authenticated session; `git pull` first)

```bash
# max-off (3 seeds)
CAMPAIGN=mx SEEDS="0 1 2" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=maxoff IMPACT_W=0 DELIVERED_W=0 \
  sbatch --array=0-2 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W \
  scripts/slurm/vega_train.sbatch

# max-max (3 seeds)
CAMPAIGN=mx SEEDS="0 1 2" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=maxmax IMPACT_W=24 DELIVERED_W=4 \
  sbatch --array=0-2 \
  --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT,IMPACT_W,DELIVERED_W \
  scripts/slurm/vega_train.sbatch
```

Runs: `mx_maxoff_seed{0,1,2}`, `mx_maxmax_seed{0,1,2}` under `logs/rsl_rl/z1_hammer/`. ~1.5 h each,
1 GPU per task, `--exclude=gn03,gn34` + `--requeue` baked in. Optional rigor: retrain **max-on** in the
same campaign (`SINGLE_SHORT=maxon`, no IMPACT_W/DELIVERED_W) for perfect commit-parity — otherwise `af1`
serves as max-on.

## Analysis (after training, local)
1. `rsync` the `mx_maxoff_*` / `mx_maxmax_*` `model_499.pt` back into `logs/rsl_rl/z1_hammer/`.
2. Add them to `ARMS` in `evaluation/trajectory/trace_head_trajectories.py` (all analysis scripts now live
   under `evaluation/` — see `evaluation/README.md`), re-run the trace +
   `plot_head_trajectories.py` + `swing_vs_speed.py` → compare **pre-contact swing** and **v_touch** across
   the max-off / af1 / max-max dose axis (same fixed reset, apples-to-apples).
3. Verdict: monotone swing↑ with dose ⇒ swing is a maximization maneuver; flat ⇒ kinematic.

## LAUNCHED — 2026-07-20 (Vega, commit `d4dc5d8`, training code == af1 verified)
12 jobs, all RUNNING immediately (nodes gn06/31/33/35/39/41), submitted via `ssh vega` (key-only, no OTP):

| job array | arm | task | impact / deliv | runs |
|---|---|---|---|---|
| 39857924 | **maxoff** | CaT-Impulse (none) | 0 / 0 | seeds 0,1,2 |
| 39857925 | **maxon** | CaT-Impulse (none) | 8 / 2 | seeds 0,1,2 |
| 39857926 | **maxmax** | CaT-Impulse (none) | 24 / 4 | seeds 0,1,2 |
| 39857927 | **maxofftrk** | CaT-Impulse-**Track** | 0 / 0 + imitation prior | seeds 0,1,2 |

`maxon` retrained in-campaign for commit-parity (af1 is the same config at identical code). `maxofftrk`
added as the **prior control**: if maxoff(none) strikes straight but maxofftrk swings, the swing is
taught by the imitation prior, not maximization. Runs: `mx_<arm>_seed{0,1,2}`. Local monitor polls to
completion + rsyncs checkpoints back, then the swing dose–response is re-plotted via the trace pipeline.

## RESULT — 2026-07-20 (all 12 COMPLETED ~28 min; traced from the same fixed reset, seed 12345)

**The forward-swing is a LEARNED impact-maximization maneuver — not kinematic, not the imitation prior.**
(mean ± sd over 3 seeds, `mx_trajectory_grid.png` + `mx_dose_response.png`):

| arm (impact/deliv) | pre-contact swing (cm) | v_touch (m/s) | delivered (× i_ref) | contact dwell (ms) |
|---|---|---|---|---|
| **maxoff** (0/0) | **0.10 ± 0.03** (dead straight) | 1.43 ± 0.01 | 0.62 ± 0.05 | ~16 |
| maxon (8/2) | 3.27 ± 2.11 | 1.24 ± 0.18 | 0.92 ± 0.23 | ~27 |
| **maxmax** (24/4) | **5.46 ± 2.35** | 1.35 ± 0.00 | **1.16 ± 0.09** | ~29 |
| maxofftrk (0/0 + prior) | **0.10 ± 0.01** (straight) | 1.45 ± 0.02 | 0.67 ± 0.04 | ~16 |

Three findings (graded/adversarially reviewed by a 4-lens analysis, `2026-07-20` — the corrections below):
1. **Swing is caused by the impact-max reward — SOLID (defend verbatim).** ON vs OFF is a non-overlapping,
   matched-reset, pre-registered separation: all 6 reward-OFF seeds ≤ 0.13 cm, 5 of 6 reward-ON seeds ≥ 2.1 cm.
   *(Overturns the earlier "kinematic" lean — controlled ablation beat correlation.)* **HEDGE:** the *graded*
   8→24 step is **underpowered** at n=3 (maxon is bimodal — one seed swings 0.28 cm, like reward-OFF; maxon/maxmax
   overlap). Report it as "directionally consistent, not a clean graded curve." Rest no claim on the graded magnitude.
2. **The imitation prior does NOT cause it — SOLID.** maxofftrk (prior, no impact reward) is straight (0.10 cm).
3. **The drive cashes out as contact-TIME, not contact-SPEED — SOLID, and the sharper finding.** v_touch is
   **dose-insensitive ~1.35–1.48 m/s** (maxoff is actually *fastest*; a soft effort clamp, not a hard wall),
   but delivered rises 0.62 → 1.16 (~1.9×). **Mechanism (corrected):** the extra impulse is **longer dwell +
   one extra tap at FLAT force** — mean force/contact-substep is flat across all 12 policies (swing-vs-force
   r≈0.21), while contact duration drives delivered (duration-vs-delivered r=0.91), dwell +~80% (16→29 ms),
   contact events 2→3. NOT a "harder" press. And all 12 nails seat to an **identical 32.0 mm** — the extra
   force·dt does *zero* additional task work: impulse without work, the press-against-a-stop signature.

**Interpretation (ties to Phase-0):** on **fixed impedance** an impact-maximization objective has only one
physically reachable outlet that RL found — **contact time** — so under this reward + action space it
**degenerates to press-maximization** (longer dwell at flat force), reproducing the open-loop Phase-0
press-integral finding *inside a trained policy*.

**⚠ CAUSALITY CAVEAT (Fable deep-review, 2026-07-21) — do NOT claim "VIC is necessary" yet.** The honest
statement is **"fixed-impedance RL never found the momentum channel," NOT "fixed-impedance forecloses it."**
Two facts force the weaker claim: (1) B2 (`2026-07-20_B2_effort_ceiling.md`) showed a coordinated whip
reaches **~4.22 m/s at the same fixed PD gains + 30/60 N·m limits** (kp doesn't enter M(q)), so the speed
channel is *not* physically closed by fixed stiffness — the trained straight-down strike just uses ~45% of
it. (2) That 4.22 m/s is a **kinematic upper bound from injecting joint velocities** (`coord_whip2.py`
bypasses the DiffIK controller); whether it's reachable *closed-loop through the real `a_t∈[−1,1]³` action
interface* is **B2's specified-but-UNRUN "decisive next experiment."** Across all 1,140 rollouts (0–24×
reward, 500–1500 iters) the fastest contact ever was **1.81 m/s** — RL didn't get close. So the defensible
VIC motivation is **landscape-shaping, not physical necessity**: *the fixed-impedance reward has a wide cheap
dwell ridge that dominates the narrow momentum peak, and RL never stepped off it even under 24× incentive.*
The **prerequisite** to any "necessity" claim is running the closed-loop whip search (or a whip curriculum)
on fixed impedance first — if that still caps ≪4.22 m/s, *then* necessity is earned; until then it's
"RL didn't find it," not "it can't be found." *(Also corrected earlier: "harder press" → dwell at flat force;
"velocity ceiling" → dose-insensitive soft clamp.)*

**Follow-ups to reach publication-solid** (ranked value-per-effort): (1) **[cheap, CPU, no GPU]** re-evaluate
the 12 checkpoints over 50–100 randomized resets → per-policy distributions, closing the single-fixed-reset
gap (every number is currently one rollout from one IC); (2) **[cheap, CPU]** log peak axial force +
per-event delivered → kills "harder press" at the physics level, separates one-long-press from extra-taps;
(3) **[GPU]** the **VIC arm** (policy commands `set_gains`) — the only experiment that converts "necessity"
into "efficacy"; plus decomposition arms (impact-only 8/0,24/0 vs delivered-only 0/2,0/4) to break the
impact/delivered co-scaling confound.

## VALIDATION — multi-reset eval, 2026-07-21 (follow-ups #1 + #2 DONE, CPU, `mx_multireset_box.png`)

Re-evaluated all 12 checkpoints over **50 randomized resets each** (`play=False` → reset noise active,
head ±14 mm; obs noise also on = real deployment conditions; 150 rollouts/arm). This closes the
single-fixed-reset gap and directly measures peak force + contact events. Means ± sd over the reset dist:

| arm | swing (cm) | % straight (<1cm) | v_touch | delivered ×i_ref | dwell (ms) | MEAN force (N) | PEAK force (N) | events |
|---|---|---|---|---|---|---|---|---|
| maxoff | 0.08 ± 0.69 | **90%** | 1.42 | 0.67 | 18.5 | 22.2 | 30.7 | 1.7 |
| maxon | 3.26 ± 2.22 | 29% | 1.36 | 0.86 | 25.4 | 20.6 | 32.8 | 2.2 |
| maxmax | 5.07 ± 2.30 | **7%** (93% swung) | 1.36 | 0.95 | 27.8 | 20.9 | 36.4 | 2.5 |
| maxofftrk | 0.05 ± 0.70 | **90%** | 1.44 | 0.69 | 18.7 | 31.2* | — | 1.8 |

**What strengthened (now robust, not a single draw):**
- **The ON/OFF swing separation GENERALIZES across ICs.** reward-OFF arms strike straight in **90%** of
  rollouts; maxmax swings in **93%**. The swing is a real, IC-robust learned behavior — the core causal
  claim survives IC + obs noise. (Success stays ≥99% everywhere.)
- **Speed stays flat** (~1.36–1.44 m/s) across the full reset distribution — the "paid 24× for speed, went
  no faster" result is robust.
- **maxon bimodality is a real training-seed basin, not undersampling:** maxon_s0 strikes straight in all 50
  resets while s1/s2 swing — the 8/2 dose induces the maneuver in ~2/3 of seeds. Confirms the "graded step is
  underpowered/bistable" hedge.

**Two things the multi-reset eval CORRECTED (single-reset draw had overstated them):**
- **Delivered gain is ~1.4×, not 1.9×** (robust mean 0.67 → 0.95). The 1.9× was a lucky single reset.
- **"Flat force" was too strong.** MEAN force per contact IS flat (22 → 21 N — the impulse gain is *not*
  from higher average force), **but PEAK axial force rises ~18%** (30.7 → 36.4 N): the force *profile
  sharpens modestly*. Honest mechanism: **the ~1.4× impulse is bought by longer dwell (+50%, 18.5 → 27.8 ms)
  + more contact events (1.7 → 2.5 taps) at flat MEAN force, with a modestly sharper peak** — dwell/taps
  dominate, peak-force is secondary. The "press not momentum" core (dwell-driven, speed flat) holds; "purely
  flat force" does not. (*maxofftrk mean-force N/A: its low delivered ÷ dwell is not a clean press.)

Net: the **binary causal claim and the press-not-speed dissociation are now IC-robust**; the delivered
magnitude and the "flat force" wording are corrected downward. Remaining open item for "efficacy": the VIC
arm (follow-up #3, GPU).

## FOLLOW-UP CAMPAIGNS — LAUNCHED 2026-07-21 (Vega, commit `7d3316f`, clean)

Two more campaigns (12 jobs, `SINGLE_TASK` none arm, 3 seeds each; sbatch gained an `ITERS` env-var hook):

**Decomposition (`dc`, 500 iters) — breaks the impact/delivered co-scaling confound:**
| job | arm | impact / deliv | question |
|---|---|---|---|
| 39861197 | `dc_imponly` | **24 / 0** | does the *speed* reward alone drive the swing? |
| 39861274 | `dc_delonly` | **0 / 4** | does the *impulse* reward alone drive it? |

Prediction: `g2_delivoff` (July, 8/0) already swung ~4.6cm, so impact_progress alone seems to induce the swing
*despite contact speed being capped* — completing the split (delivered-only) pins the mechanism. maxoff (0/0)
is the existing straight baseline.

**Longer training (`lg`, 1500 iters, `--time=02:30:00`) — does behavior change with 3× training?**
| job | arm | impact / deliv | question |
|---|---|---|---|
| 39861277 | `lg_maxmax1500` | 24 / 4 | does more training break the ~1.4 m/s ceiling or intensify the press? |
| 39861278 | `lg_maxoff1500` | 0 / 0 | does the straight striker stay straight with 3× training? |

Analysis: trace `dc_*`/`lg_*` via `trace_mx.py` (add to ARMS) + `eval_mx_multireset.py`; compare swing / v_touch /
delivered / dwell / peak-force vs the 500-iter mx arms. Provenance clean `7d3316f` (ITERS hook committed +
pushed; Vega fast-forwarded). *(Note: git-from-Vega login node hangs on pull sometimes; the run still recorded
clean `7d3316f` — a stray kill during the first submit left only `dc_imponly` from the initial batch, remaining
3 arrays submitted separately.)*

## DECOMPOSITION RESULT — 2026-07-21 (dc arms done; 30-reset validation, `dc_decomposition_box.png`)

Isolating the two rewards (30 randomized resets/policy, play=False) reveals they drive **dissociable
strategies** — a cleaner mechanism than the co-scaled mx arms could show:

| arm | reward | swing (cm) | v_touch | delivered | dwell (ms) | **PEAK F (N)** | **MEAN F (N)** | taps |
|---|---|---|---|---|---|---|---|---|
| maxoff | neither | 0.18 | 1.42 | 0.67 | 18.0 | 31.2 | 22.6 | 1.7 |
| **imponly** | **speed** | **2.08 (SWING)** | 1.32 | 0.92 | 26.6 | 33.2 | 21.1 | 2.7 |
| **delonly** | **impulse** | **0.16 (STRAIGHT)** | 1.34 | 1.00 | 24.9 | **43.1** | **24.4** | 1.9 |
| maxmax | both | 5.14 | 1.36 | 0.96 | 27.8 | 36.9 | 21.0 | 2.4 |

**The dissociation (robust):**
- **`impact_progress` (speed reward) → the wind-up SWING** (+ longer dwell + more taps), at **flat force**.
- **`delivered_impulse` (impulse reward) → a straight HARD PRESS** — no swing, but the **highest peak force
  (43 N) and mean force (24.4 N)**. Force is the impulse reward's lever, not the speed reward's.
- Both raise delivered (~0.9–1.0×) but by **different mechanisms** (swing/dwell vs force). maxmax (both) is a
  blend where the swing dominates the *look* but suppresses the force the impulse reward alone achieves.

**Two honest corrections this forced:**
1. **Retract the single-reset "the speed reward buys speed" revision** (from the trace draft, imponly hit
   1.62). Across 30 resets **imponly v_touch = 1.32 ≤ maxoff 1.42** — the speed reward does **NOT** raise
   contact speed. The original synthesis claim ("paid for speed, went no faster") **stands**; the 1.62 was a
   single-reset fluke. (The multi-reset caught it — exactly why it was run.)
2. **Refine the mx "flat force" story:** mean force is flat *in the combined arms*, but the **impulse reward
   ISOLATED (delonly) drives higher force** (peak 43 N, mean 24.4 N) in a straight press. "Flat force" is a
   property of the both-reward blend, not of the impulse reward's intrinsic effect.

**Constraint-relevant new angle:** delonly (pure impulse-maximization) produces the **highest joint-loading
peak force (43 N)** — so the delivered-impulse objective, isolated, is exactly what would drive the per-joint
reaction Λ hardest. This ties the maximization reward directly to the constraint the thesis bounds.

## LONGER-TRAINING RESULT — 2026-07-21 (lg arms, 1500 vs 500 iters, 30-reset validation)

> ⚠ **CONFIG BUG — caught 2026-07-21 by `evaluation/list_policies.py`.** `lg_maxoff1500` was trained with the
> **DEFAULT reward (impact 8 / delivered 2), NOT 0/0** — I omitted the `IMPACT_W=0 DELIVERED_W=0` override at
> submit. So the "maxoff@1500" row is really **maxon-config @1500**, and the earlier *"maxoff drifted into a
> swing / null-space wander"* reading is **RETRACTED** (an 8/2 policy swinging is expected, not drift). The
> `maxmax` (24/4) 500-vs-1500 comparison is same-config both ways and **stands**.

| arm (actual reward) | swing (cm) | v_touch | delivered | peak F | **success** |
|---|---|---|---|---|---|
| maxmax @500 (24/4) | 5.14 | 1.36 | 0.96 | 36.9 | 99% |
| **maxmax @1500 (24/4)** | 0.92 | 1.33 | 0.71 | 25.6 | **67%** |
| ~~maxoff~~ **maxon @1500 (8/2, mislabeled)** | 4.16 | 1.37 | 0.85 | 34.2 | 99% |

**What stands — maxmax (24/4), a clean same-config 500→1500 comparison: 3× training DESTABILIZES, doesn't help.**
- **A seed collapsed:** `maxmax1500_s0` fails on **0/30** resets (delivered 0.18, peak force 12 N) — overtraining
  killed a working policy; maxmax @1500 success drops 99% → 67%, survivors swing less and deliver less.
- **The ~1.4 m/s ceiling held:** max v_touch over all 1500-iter rollouts = **1.81 m/s** — more training finds no
  faster strike (a fixed-impedance structural limit, not a training-budget one).
- **Not answered:** whether a *real* maxoff (0/0) stays straight at 1500 iters — the arm meant to test that was
  mislabeled. A clean re-run (`IMPACT_W=0 DELIVERED_W=0 ITERS=1500`) would close it.

Takeaway: **on this bare fixed-env recipe, more iters over-optimize one MDP → the maxmax collapse; 500 is enough.**
The swing and speed ceiling are decided by the fixed-impedance physics + reward, not by how long you train.

**⚠ SCOPE CAVEAT (user, 2026-07-21): "500 is the sweet spot" is specific to the CURRENT recipe — NO domain
randomization, NO curriculum.** On a *fixed* environment the policy converges fast (~500) and then has nothing
left to learn, so 1500 iters is pure over-optimization on one MDP → collapse (the `maxmax1500_s0` 0/30 collapse
is the overfitting-to-a-fixed-env signature). **With DR + curriculum
the effective task is harder and non-stationary, so more iterations would be needed and productive, not
destabilizing** — the varied data would very likely *stabilize* longer training rather than let it wander.
So do NOT carry "1500 hurts" into the DR/curriculum regime (or into VIC, which adds an action dimension); the
right reading is "on this bare fixed-env recipe, extra iters only overfit." This makes Fable's canary
early-stopping / stability-monitoring suggestion (below) the thing to build *before* DR+curriculum+VIC training,
where longer horizons return.

## DEEP-REVIEW (Fable, 2026-07-21) — the essential fact + open experiments

**Essential fact (all findings are one thing):** under fixed impedance, delivered impulse is `∫F·dt` against
a target that's still there, so the reward can only select **contact duration**. Recomputed over 1,140 unique
rollouts (23 policies, 0–24× reward, 500–1500 iters): `delivered ≈ 0.027·dwell_ms + 0.014·peak_force − 0.28`,
**R²=0.842**; adding v_touch → +0.0002, adding swing → +0.003. Two scalars (how long in contact, how hard)
predict delivered impulse; **speed and the dramatic swing add ~nothing**. `corr(v_touch, delivered) = −0.26`
(negative). `corr(dwell, delivered) = 0.84` — the strongest, most universal relation, present in every arm.
The swing/press "dissociation" is two reward-lenses on the same lever (stay longer / push harder), not two
physical channels.

**Load-bearing open experiments (ranked):**
1. **Closed-loop whip search (fixed impedance)** — B2's specified-but-unrun decisive test: optimize/curriculum
   toward a coordinated whip through the *real* DiffIK `a_t∈[−1,1]³` interface (not velocity injection). This is
   the prerequisite for any "VIC necessary" claim. → **SPECCED + RUN (Stage 1, 2026-07-21):**
   `2026-07-21_whip_search_spec.md`. CEM open-loop shooting (RL-seeded, converged) through the real DiffIK
   interface. **RESULT:** at the shipped δ=0.15, an exhaustive legal search beats RL by only +6% (v\*=1.928,
   hardware-legal v\*_hw=**1.72**) → **~1.8 m/s IS the ceiling and RL already found it — no hidden whip channel.**
   BUT it is the **action-scale** ceiling, not impedance: raising `delta_pos_scale` 0.15→0.30 (PD unchanged, still
   FIC) lifts the legal ceiling to **2.83 m/s (+64%)** → **VIC is NOT required to exceed 1.8 m/s** (a FIC knob does
   it). Efficiency drops 0.26→0.19 (PD-bandwidth onset); δ=0.45 pending. Caveat holds: this is SPEED, not the
   press-integral Λ (`corr(v_touch,delivered)=−0.26`). (Superseded plan: the "retrain imponly with wind-up
   curriculum, anneal `delta_pos_scale` up" idea below is now moot for the *speed* question — the shooter already
   answered it.)
2. **Swing-is-a-controller-artifact check (cheap, CPU)** — replay a maxmax rollout with the head arc smoothed
   out but dwell/approach preserved; the regression predicts delivered/reward won't change. If confirmed: the
   swing is kinematic residue of DiffIK re-servoing a compliant target, not a chosen maneuver — a clean
   reward-hacking-diagnosis result (dramatic behavior, ~zero causal weight).
3. **Per-joint load of the pure-press arm** — Phase-0 A5: press loads **j1**, ballistic loads **j2–j4**. `delonly`
   (pure impulse-max) is a press → likely loads j1, but the constraint/ballistic framing targets j2–j4. Log
   per-joint qfrc during a `delonly` rollout: if the deployed-reward regime loads a *different* joint than the
   constraint is calibrated for, that's a real risk to the impulse-CaT going into VIC.
4. **Canary early-stopping** — the `lg` collapse happened between iter 500–1500 unexamined; log peak-force /
   delivered / return-variance every ~100 iters to find a pre-collapse precursor. Build this *before*
   DR+curriculum+VIC (where longer horizons return — see scope caveat above).

## Guardrails honored
`imp_max_p=0` (log-only, set in the sbatch), `IMP_J_LIMIT` unchanged, caps untouched, no VIC/`set_gains`,
no superlinear excess-over-i_ref reward (the max-max `delivered_impulse=4` is the same **linear** ΔI/I_ref
term, just weighted). Provenance: git-based deploy, `-dirty` guard, per-run `git/unitree_rl_mjlab.diff`.

## DEEP-CHECK RESULTS — 2026-07-21 (`check_deep.py`, CPU, play=False)

**Check 2 (per-joint peak Λ/cap, 15 resets × 3 seeds):** every arm loads **j1 (shoulder) most**;
`delonly` (pure impulse-max) loads it hardest (0.268× cap). Per-joint Λ/cap:
maxoff [0.11,0.06,0.05,0.05,0.07,0.01] · imponly [0.23,0.09,0.15,0.11,0.14,0.01] ·
delonly [0.27,0.10,0.14,0.09,0.17,0.02] · maxmax [0.23,0.09,0.14,0.10,0.14,0.01].
→ **Confirms the deployed (press-regime) policies load j1**, not the ballistic j2–j4 the constraint framing
emphasizes (Phase-0 A5). All < 0.27× cap (non-binding, log-only), but **enforcement would primarily bind j1**
— a calibration note for the impulse-CaT going into VIC. The impulse-max reward (delonly) is exactly what
drives j1 load, tying the reward to the constraint.

**Check 1 (is the swing causal? — INCONCLUSIVE / confounded):** replaying maxmax's actions with the lateral
(x,y) deltas zeroed dropped delivered **0.93 → 0.60 (−35%)** and missed the nail on 1/12 resets — so the swing
is **NOT a pure non-causal artifact** (leans against Fable's hypothesis). BUT the counterfactual is confounded:
zeroing x/y also changes the *approach* (Fable's test said preserve it), so this conflates "the arc" with "the
lateral approach" and does not cleanly isolate the arc. The clean statement stays the between-policy one:
**`delonly` (straight, trained) delivers 1.0 > maxmax (swing) 0.93** — the swing is causal within its own
policy but a *suboptimal* delivery strategy; a straight press does better. A clean arc-isolation test remains
open (hard to vary arc independent of approach in a closed-loop policy).
