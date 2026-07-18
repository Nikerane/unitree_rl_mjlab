# Vega neck-fix face-strike campaign — plan + live log (2026-07-15)

Consolidated from a Gemini × Codex (gpt-5.6-sol) design panel + the live Vega state.
Goal: rerun **prior-vs-none** on the **fixed (neck-off) geometry** to get the first *valid*
face-strike numbers, a re-checked vacuity, and a better-powered NULL test.

## Locked decisions
- **500 iters** (not 250): `r_imit` anneals to 0 at step 6000 = exactly 250 iters, so 250 gives
  ~zero post-handoff free-RL; the treatment (prior → free RL, path-dependent) needs 500. Canonical
  pair uses 500. (Codex; overrides Gemini/earlier "250 sufficient" which was about *strike* convergence.)
- **4096 envs, log-only** (`imp_max_p=0`; enforcement blocked on Khadiv).
- **Staged & fail-closed** (not one-shot): guard → canary → validity(n=3) gate → expansion(n=8).
- **1-GPU-per-task Slurm array** (`--gpu-ids [0]`), interleaved track/none, unique campaign run names.
- **`substep_impulse_rows` OFF** in production (`--env.metrics.substep-impulse-rows.params.enabled False`) — diagnostic-only perf sink.

## Vega facts
- `ssh vega` (login.vega.izum.si, user eunikhilr), 8h persistent master. Account `d2026d06-166-users`,
  partition `gpu` = **59× 4×A100-40GB nodes**, QoS `normal`, no queue wait observed.
- Repos at `~/repos/{unitree_rl_mjlab,safe_impact_manipulation}` (siblings). `.venv` = uv venv
  (no `pip`; torch 2.12.0+cu130, mjlab/rsl_rl/mujoco 3.8.1 import OK). **Gotchas:** no `pip`/`pytest`
  in venv (run guards inline); `python` is system 3.6.8 (use `.venv/bin/python`); login node has no
  CUDA driver (Warp falls back to CPU — fine for guards).
- Pins: unitree **b2ed6ee** (neck-fix + threshold), assets **b58ccd2** (neck-off scene). Both reset on Vega.

## Fail-closed guard (Codex's "one thing" — DONE, all PASS)
1. pin-ancestry: both repos contain the neck-fix pins ✓
2. scene: `hammer_head_1 = 1/1` (neck off nail channel) ✓
3. reachability (inline, compiled model): nail-reachable = `{hammer_head_0}` only ✓
4. `playback_reference` 30 mm gate: `PHASE M GATE: PASS`, face-strike reaches 30 mm all heights ✓

## Scripts (in `scripts/slurm/`, copied to Vega)
- `vega_canary.sbatch` — A100 assert + instrument gate + 50-iter production canary + **production-path**
  TB metric check (delivered_total>0, impossible_success==0).
- `vega_train.sbatch` — the training array. `CAMPAIGN=nf1 SEEDS="..." sbatch --array=0-N --export=ALL,CAMPAIGN,SEEDS`.
  task t → seed=SEEDS[t/2], arm = track (t even) / none (t odd).

## Timeline (staged)
| phase | action | gate |
|---|---|---|
| 0 bring-up | reset repos to pins | HEADs == pins ✓ |
| 1 guard | pins+scene+reachability+playback | all PASS ✓ |
| 2 canary | `sbatch vega_canary.sbatch` | A100+instrument+production metrics live |
| 3 validity | `CAMPAIGN=nf1 SEEDS="0 1 2" sbatch --array=0-5 ...` | 6× model_499, invariants 0, 30mm success, clean face renders |
| 4 expansion | `CAMPAIGN=nf1 SEEDS="3 4 5 6 7" sbatch --array=0-9 ...` → n=8/arm (stretch n=10-15, cluster is wide open) | — |
| 5 eval+closeout | `eval_impulse` w/ Tier-1 invariants (dependent CPU job); record BOTH repo hashes; analyze paired-by-seed; use `delivered_total` not `substep_delivered` | invariants 0 |

## Watch-items (from the panel)
- Edge/shoulder contact within `hammer_head_0` is invisible without contact-position logging (deferred, needs code) → proxy via Tier-1 invariants + close-up renders + Λ_j spread.
- Powered NULL: the observed d≈0.62 needs ~41 seeds/arm for 80% power (doesn't fit) → n=8-15 lifts power to ~35-50% + gives valid numbers; full resolution may need a follow-up.
- Eval provenance: unique names + require model_499 from the NEW dir + record the assets hash (eval only logs the main hash).

## LIVE LOG
- 19:xx bring-up + guard: all PASS. Canary `JID 39397581` RUNNING on gn31, zero queue.
- 19:2x canary v1 `JID 39397581` FAILED ~40s: `ImportError: Bad git executable` (git not on compute node PATH; GitPython chokes on import). Fix: `module load git` + `export GIT_PYTHON_REFRESH=quiet` in both sbatch scripts; re-scp'd.
- 19:33 canary v2 `JID 39397683` **CANARY_ALL_PASS**: A100 assert ✓, instrument gate ✓, 50-iter track+none production run ✓, production TB metrics live — `delivered_total` track=0.405 none=0.368 (>0 ✓), `contact_seen`≈1.0, `impossible_success`=0.0 both arms ✓.
- 19:xx **validity tranche submitted** `JID 39397954` (`CAMPAIGN=nf1 SEEDS="0 1 2" --array=0-5`): all 6 tasks RUNNING instantly on gn31/gn32/gn35, zero queue. seeds 0/1/2 × {track,none}, 500 iters, 4096 envs, log-only. ETA ~12 min → model_499.
- 20:05 **validity DRAINED — all 6 COMPLETED** (~17:30–17:50 wall each; stdout was block-buffered, TB events confirmed live throughout). `model_499.pt` present in all 6 dirs.
  - **Invariant CLEAN:** `Episode_Metrics/impossible_success = 0` on all 6; `contact_seen = 1.0` (face hits nail).
  - **⚠ SURPRISE — delivered impulse peaks early then REGRESSES.** `delivered_total` rises to ~0.36–0.40 (×i_ref) at it25–50, then DECAYS to ~0.25 and plateaus for the rest of training; `Episode_Reward/completion` peaks ~0.10 at it50, declines to ~0.05. `Episode_Termination/nail_driven` spikes at it25–50 (count 21–131) then drops ~10–50× to steady low. So the strike appears strongest EARLY (during the r_imit scaffold) and the policy drifts to a partial-impulse / gentle-contact optimum. **Both arms show it** (track didn't obviously retain the strike).
  - **Success rate is NOT resolvable from TB** (Episode_Termination units ambiguous; no `success` scalar exists — earlier "succ" col was a substring collision with `impossible_success`). Resolving via eval.
- 20:1x **eval submitted** `JID 39400679` (`scripts/slurm/vega_eval.sbatch`, NEW): scores all 6 `model_499` on 1 GPU, log-only, Tier-1 self-certifying → real success rate + worst Λ_j/J_limit (vacuity re-check) + delivered mean + nail depth. **n=8 expansion HELD** pending eval (don't expand a possibly-regressed setup).
- 20:47 **EVAL DONE** rc=0, failures=0, **all 6 rows `impossible_success_n=0 lambda_dead_n=0`** (instrument sound on CUDA at scale). `eval/nf1/summary.csv`. Numbers (mean-action primary | sampled robustness):

| run | succ(mean) | succ(sampled) | Λ/cap(mean) | Λ/cap(sampled) | delivered | ep_len(mean) |
|---|---|---|---|---|---|---|
| nf1_track_s0 | 0.00 | 0.49 | 0.087 | 0.96 | 0.211 | 200 (cap) |
| nf1_none_s0  | 0.00 | 0.60 | 0.093 | 0.91 | 0.203 | 200 |
| nf1_none_s1  | 1.00 | 0.69 | 0.104 | **1.00** | 0.231 | **9** |
| nf1_none_s2  | 0.00 | 0.28 | 0.178 | 0.92 | 0.198 | 200 |
| nf1_track_s1 | 0.00 | 0.47 | 0.080 | 0.93 | 0.190 | 200 |
| nf1_track_s2 | 0.00 | 0.71 | 0.082 | **1.12** | 0.202 | 200 |

  **Interpretation (3 load-bearing points):**
  1. **Setup WORKS — the v2 NULL was neck-confounded, this is valid.** Per-policy SAMPLED success = **28–71%** across all 6. The face-only policy drives the nail.
  2. **Mean-action succ is bimodal (0/1) = an EVAL ARTIFACT, not 0% failure.** Play-cfg uses deterministic reset (`position_range=(0,0)`), so under mean action all 256 envs are the SAME rollout → success is necessarily ~0 or ~1. The meaningful per-policy rate is the SAMPLED one (noise decorrelates envs). Training itself uses ±0.05 rad reset noise. So "5/6 mean-succ=0" overstates; real success is 28–71%. (Under-commitment of the mean is still real — TB showed delivered peak 0.40→plateau 0.25 — but not a dead setup.)
  3. **Vacuity is ACTION-MODE DEPENDENT — constraint BINDS on real strikes.** Mean (gentle) Λ/cap = 0.08–0.18 (vacuous). SAMPLED (committed) Λ/cap = **0.91–1.12** — reaches & slightly exceeds the cap. So the per-joint impulse constraint is NOT globally vacuous: it activates exactly when the policy commits to a completing strike. This refines the fixed-impedance vacuity finding (rigid-target ~0.6× cap was for the SCRIPTED slow reference; a trained committed strike hits ~1.0× cap).

## ROOT CAUSE of the under-commitment (design panel `wf_c9ee055c` + empirical guard) — 2026-07-15

**The `nail_driven` reward is a FARM.** It is a per-step Gaussian (weight **2.0**) on nail **depth**, centered at goal_depth **0.032 m**, but success terminates at **0.030 m**. The nail **ratchets** (nail_slide frictionloss=30, damping=0.5, **no spring**, gravity-compensated body) — driven to depth d, it holds at d for free. So the policy drives the nail to just **below 0.030** and **holds**, collecting `2.0·exp(−(0.032−d)²/0.013²) ≈ 1.95/step` **in perpetuity without terminating**. Completing (crossing 0.030) **ends the episode and forfeits that stream** for a one-time completion +100.

**Return arithmetic (γ=0.99 ⇒ effective horizon ~100 steps):** hold-at-threshold value ≈ `1.95/(1−γ)` ≈ **195–205**, vs complete-and-terminate ≈ **~130**. **Completing pays LESS.** Break-even at nail_driven weight ≈ 0.9–1.0. This is the classic "non-terminal state pays positive dense reward" bug — `nail_driven` is a potential-shaping term that isn't potential-based, re-paying a one-time (ratcheted) achievement forever.

**EMPIRICAL GUARD (existing eval `eval/nf1/summary.csv`, mandatory check both adversarial reviewers demanded) — CONFIRMED:** the 5 under-committing policies hold `nail_depth_mean` = **26.7 / 27.8 / 28.4 / 28.5 / 29.5 mm** (all ≥23 mm, all <30 mm, `depth_std=0.00`) — parked in the farm sweet-spot. The 1 success holds 31.9 mm (crosses 30 → terminates at 9 steps). Farm mechanism verified, not hypothesized.

**Why the other terms are innocent:** `nail_depth_delta` (600·Δdepth), `impact_progress` (gated first-contact∧advance), `delivered_impulse` (monotone delta-credit), `completion` (one-shot) all pay **0/step for a static hold** — correctly non-farmable. `nail_driven` is the SOLE dense term that pays for a static state. **Prior is a red herring** (r_imit perpetuity 0.1/(1−γ)=10 ≪ farm 200; strengthening it is the walked-back DeepMimic and wouldn't fix a terminal-return inversion).

### RECOMMENDED FIX (survived adversarial review; augment-not-replace, one term)
- **Cut `nail_driven` weight `2.0 → 0.5`** (hammer_env_cfg.py:172). At 0.5 the hold-at-threshold value drops ~205 → **~49 < completion 100**, so completing strictly dominates every sub-threshold hold; 0.5 keeps the "pull once the nail moves" nudge (break-even ~0.9, so 2× margin). **Nothing else changes.**
- **Pilot:** CLI flag `--env.rewards.nail-driven.weight 0.5` (weights read dynamically — no code/test churn). 6 runs (2 arms × seeds 0/1/2), 500 iters, 4096 envs, imp_max_p=0. Gate first: `validate_rewards.py` + `verify_reward_setup.py`.
- **GO:** mean-action ep_len drops off the 200-cap (<~30 steps) + mean success rises off the 0/1 artifact into the 28–71% band + delivered curve no longer peaks-then-regresses. **NO-GO:** mean still pinned at 200-cap (→ fallback).
- **Fallback (if commitment doesn't lift = exploration collapse, not farm):** entropy_coef 0.02→0.04 (rl_cfg.py). Heavier (gated loiter cost) MUST also add its term to `_NEG_TERMS` in `cat/hook.py:36` or it crashes the soft-CaT positives-only guard.
- **Thesis defense:** strike stays EMERGENT (raising completion's relative return, not baking a trajectory); prior untouched (still anneals to 0 by it250); sanctioned augment-not-replace retune of ONE term for an OBSERVED failure. Bonus: as the mean commits, the log-only Λ/cap climbs out of the vacuous band on the **mode** (not just sampled noise) → cleaner "constraint binds on real strikes" figure, no enforcement needed.
- **Bake in only AFTER GO:** hammer_env_cfg.py:172 `2.0→0.5` for the definitive n=8. **HELD** pending: user approval.

### External research corroboration (deep-research `wf_dfafcf6e`, 101 agents / 22 verified claims; synth cut by session limit)
- **Theory backs the fix exactly.** PBRS/terminal-incentive theory (arxiv 2502.01307): "the incentive to terminate in a goal state must originate from the original reward … PBRS cannot create it," formally requiring **goal reward r_g > on-step reward r_∞**. Our bug IS this ordering violated: on-step `nail_driven` (r_∞≈1.95) > the goal advantage. Cutting it to 0.5 restores r_g > r_∞. Non-potential dense shaping → lingering is the classic Ng/Harada/Russell result (bicycle-in-circles). Minimum-time sparse beats dense for goal-reaching (arxiv 2407.00324).
- **The ANCESTOR (DAPG hammer-v0, `mj_envs/hammer_v0`) sidesteps it by the OPPOSITE structural choice — a strong independent check.** It (a) **never terminates on success** (`done` hardcoded False, fixed 200-step horizon) so completing never forfeits future reward; (b) success is a **per-step** condition and the completion incentive is a **large recurring per-step bonus** (+25 for <20mm, +75 for <10mm) on top of `-10·dist` — so **strike-AND-HOLD-at-full-depth strictly dominates** a gentle partial tap (the bonus is biggest AT full depth, not below it); (c) uses **20 N nail static friction** as a physics lever to kill slow-push. Our farm exists precisely because we do the inverse: terminate-on-success AND the Gaussian peaks (0.032) *past* the success line (0.030), creating a profitable sub-threshold plateau.
- **⇒ Two valid fixes; we take the minimal thesis-faithful one.** (A, CHOSEN) cut `nail_driven` 2.0→0.5 so the sub-threshold hold no longer out-returns completing — one term, strike stays a single decisive terminating event. (B, documented alternative if A underdelivers) DAPG-style: don't terminate on success + make the per-step depth bonus peak AT/BEYOND 0.030 (no sub-threshold plateau) — bigger change, shifts to a "drive-and-hold" framing, less aligned with the single-impulsive-strike thesis story.
- **On the prior (secondary):** research confirms anneal-to-0 demo priors trigger regression (AWAC/IBRL/DTG-IRRL retain skill by NOT annealing; pure-sparse regresses 36% vs 80% imitation-init). Consistent with the panel: the anneal removes the only counter-nudge, but the farm out-returns the strike regardless — so removing the farm (A) is the root fix, not a persistent prior (which the thesis forbids anyway).

### Codex confirmation (task-mrmh3rlj-mcijcv, verified from commit-pinned files, NOT memory)
DAPG `hammer_v0.py`: `reward -= 10*dist; reward += 25 (<20mm); reward += 75 (<10mm); return ob, reward, False, ...` (done=False, `max_episode_steps=200`, force sensor OBSERVED only). DAPG demo gradient decays geometrically (`lam_0=1e-2, lam_1=0.95`) with NO retention (precedent for our regression, not a fix). CIMER never anneals (`task_ratio=0, tracking_ratio=1.0`, persistent tracking; horizon 31).
**THESIS-CRITICAL CATCH:** DAPG's structural fix (never terminate + biggest bonus AT full depth) rewards **drive-AND-HOLD = a sustained PRESS** — exactly the behavior the impulse thesis must avoid/distinguish from an impulsive strike. So option B is *wrong for us* despite being the ancestor's choice: it would re-introduce press incentives and pollute the Λ-is-impulse-vs-press question. Our terminate-on-success + single-impulsive-strike semantics are correct; **option A (cut the farm weight) removes the farm while preserving them.** The research sharpens, not overturns, the panel's pick.

### ⚠ CORRECTION — the "sustained press / B contaminates the instrument" catch was PARTIAL→WRONG (deep-dive `wf_36a53ecb`, 7 agents, code-grounded) — 2026-07-16
The above "THESIS-CRITICAL CATCH" is **overstated**. Corrected picture (all code-verified):
- **"B = sustained press": WRONG.** The nail RATCHETS (frictionloss=30, no spring, gravcomp) → holding the driven state needs ~ZERO force. B's recurring bonus reads nail DEPTH (a world state); a policy can strike ONCE impulsively, fully retract, and still collect it. B does not force/reward/prefer a press.
- **"B contaminates the impulse instrument": mostly a NON-ISSUE on shipped code.** Three distinct quantities: (1) ENFORCED robot-side Λ = 25-substep **sliding window** + episode-peak-max (impulse_bound.py:179-181,190) → a long tail CANNOT integrate; not contaminated. (2) OBJECT-side delivered normalizer = per-event 25-substep cap + re-arm debounce (:271-276) → a tail can't grow it; not contaminated. (3) LOG-ONLY diagnostic `ContactRowImpulseAccumulator` (contact_row_impulse.py:317) is the ONLY uncapped one — it WOULD inflate under a real lingering press and trip its unit test, but **feeds no training signal**. `i_ref=0.6094` is a FROZEN constant (env_cfgs.py:347); runtime terminate/no-terminate can't move it. The "0.6094→2.14" precedent is **unverifiable in the tree** (was a reference-DERIVATION-window artifact, different subsystem). My "~190-step tail" was DAPG's number — ours is ~970 (20 s × 50 Hz).
- **impulsive-vs-press is PHYSICS, orthogonal to A/B.** Yielding 7 g nail + effort-limited ~1.3 m/s reach ⇒ every reachable strike is a drive-through press (peak/mean 1.1-3.25 ≪ ballistic 5-20×). Neither reward option lifts that ceiling — **variable impedance is the lever** (vacuity reframe). And with imp_max_p=0, δ≡0 (inert on behavior). So **A-vs-B is a REWARD-BALANCE-vs-MEASUREMENT-HYGIENE + implementation-cost trade, NOT an impulse-contamination question.**
- **NEW defect found:** `completion_bonus` is **UN-LATCHED** (rewards.py:94-95 `return (depth>=success_depth)` every step) — one-time ONLY because the termination ends the episode. So "minimal B = delete the nail_driven termination" turns completion into **100/step** (~1e5 return, destabilizes training). **B REQUIRES retuning completion weight 100→~1.0.** This un-latched term (not the razor-thin nail_driven Gaussian) is the actual structural attractor that makes B robust.
- **B safety caveat:** `impossible_success` sentinel keys on `env.reset_terminated` (impulse_bound.py:345) → goes DARK under B (no success termination). Live `contact_seen` narrows but doesn't close it; B eval must add a Λ-vs-delivered agreement check.
- **STALE-DOC FLAG:** docs/results/2026-07-12_impulse_vacuity.md §4 ("no press-length cap") describes PRE-sliding-window code, stale since commit 581578d (2026-07-13) for the enforced quantity.

### RUN-BOTH comparative campaign (user request) — design
- **Variant A** (CLI-only): `--env.rewards.nail-driven.weight 0.5`, keep terminate-on-success. Clean single-strike measurement, `impossible_success` LIVE.
- **Variant B** (code edit): add `no_terminate: bool` to the cfg factory → `terminations.pop('nail_driven')` + `rewards['completion'].weight=1.0`; register `Unitree-Z1-Hammer-CaT-Impulse-NoTerm`. Structural fix (deletes the forfeiture cliff) at cost of ~970-step tail + dark sentinel.
- **Variant B-guard** (conditional, only if B eval shows contamination): mirror the 25-substep window into the log-only diagnostic + swap eval sentinel to contact_seen + Λ-vs-delivered.
- **Train:** all on the SAME instrumented arm, seeds 0/1/2, 4096 envs, imp_max_p=0. **Eval:** existing `eval_impulse.sh` pinned protocol (same env for all arms → comparable). **Plots:** final-depth histogram (parking band vs 30mm), success+mean-commit, **ep_len dist** (A short ~8-15 steps / B ~1000 = strongest behavioral discriminator), **delivered_total vs i_ref** (contamination discriminator), Λ_j worst-joint (must match if uncontaminated), re-armed contact events/contact duration, peak |q̇|, Tier-1 invariants.
- **Decision rule:** WINNER = fixes under-commitment (empties 26-29mm parking band) + preserves clean measurement (median delivered ∈ [0.8,1.3]×i_ref, Λ_j indistinguishable, ~1 contact event) + invariants clean. **Default A** if it empties the parking band (smaller, measurement-safer, live sentinel); **choose B only if A's ~40-return flip margin proves fragile**; **disqualify B** if delivered>1.5×i_ref / multi-i_ref tail / re-armed events routinely >1 / log-only residual >25%.

### Fable ×2 cross-check + consensus (`wf_2ede6576`, 5 fable agents, one EXECUTED the tyro parse) — 2026-07-16
Verdict: **agree-with-minor-caveats.** Both Fables independently confirmed every claim against source (un-latched completion TRUE rewards.py:94-95; dark sentinel TRUE impulse_bound.py:345; enforced Λ/delivered/i_ref uncontaminated; nail ratchets = zero holding force). Corrections + vetted spec:
- **A invocation was WRONG in my plan** (caught by both, parse-executed): task id is **POSITIONAL**, `--task` doesn't exist. Correct: `python scripts/train.py Unitree-Z1-Hammer-CaT-Impulse --env.rewards.nail-driven.weight 0.5`. Arithmetic re-confirmed (w=0.5 ⇒ hold ≈48-58 < 100, ~1.7-2× margin). **A launcher rider:** CLI-only ⇒ slurm/lightning launchers don't carry the flag — add it there OR promote 0.5 into hammer_env_cfg.py:172, else the next launch silently reproduces the parking config.
- **B = GO-WITH-FIXES.** Vetted diff (unanimous): (1) add `no_terminate: bool` param to `z1_hammer_env_cfg` → `cfg.terminations.pop("nail_driven"); cfg.rewards["completion"].weight = 1.0`; (2) **MANDATORY sentinel rekey** impulse_bound.py:345 → OR in a depth-success test `(terminated | (depth>=NAIL_SUCCESS_THRESHOLD)) & (lam_worst<=0)` (else `impossible_success` goes dark — and the CUDA-Λ blocker is why it's load-bearing); (3) register `Unitree-Z1-Hammer-CaT-Impulse-NoTerm` with a **terminating PLAY cfg** (eval keys success on the termination); (4) tests: add wiring test + **extend `_stub_env` in test_signal_safety_net.py** (its scene lacks `nail_block` → the rekey KeyErrors otherwise). **completion weight = 1.0** unanimous (1/(1−γ)=100 reproduces the one-time +100 in discounted return).
- **Sequencing (unanimous): run A FIRST as the zero-diff incentive probe; ship B only if A under-tilts.**
- **HUMAN DOUBLE-CHECK — ability-cap confound:** is the ~28mm parking INCENTIVE or a physical depth ceiling (nail_block.py:60-66 warns face-only policies may cap <30mm)? **Resolved by existing data:** sampled success 28-71% + none_s1 mean-action 31.9mm prove ≥30mm IS reachable from the deterministic reset pose ⇒ parking is incentive, not ability. (nf1 checkpoints are already the neck-fixed face-only ones, not the invalidated v2 set.)
- **CUDA Λ≡0/delivered≡0 blocker (Fables flagged from stale memory): already CLEARED** by today's Vega canary (diag_cuda_substep_probe --gate PASS; delivered_total 0.37-0.41 on 3 CUDA runs).

### Impulse-CaT enforcement check status (user asked 2026-07-16) — the parking fix is its PREREQUISITE
- **Velocity soft-CaT: PROPERLY checked** (2026-06-18_softcat_velocity.md, trained GPU with-vs-without): soft-CaT mean peak |q̇| 2.48-2.65 < 3.1415 vs a_base 3.54, kept 1.21 m/s @ 100% success, positives-only guard held. Mechanism validated; worst-case tail (chain-coupled) → motivates VIC.
- **Impulse-CaT enforcement: NOT properly checked** — only the C2 CPU GATE (2026-07-10_c2_enforcement_record.md): graded δ [0.5,0.369,0.112]@max_p=0.5, c_max self-seed, no log-only physics perturbation, δ>0 on contact. That is PLUMBING verification, NOT a trained with-enforcement-vs-without campaign. **Impulse enforcement has never shaped a GPU training run.**
- **Why it can't be done meaningfully until parking is fixed:** enforcement is vacuous on the parked policy (Λ/cap 0.08-0.18 ⇒ δ rarely fires); it only bites once the policy commits to strikes at Λ/cap ~0.9-1.1. So **the A/B parking fix is the enabler for a real impulse-CaT behavioral test.** Also blocked on: recalibrating imp_max_p=0.5 for the sliding-window (~3× pressure, survival≈(1−δ)³, C2 predates 581578d) + Khadiv (e)/(f).
- **Sequence:** fix parking (A/B) → recalibrate imp_max_p → Khadiv (e)/(f) → THEN impulse-CaT with-vs-without. Flagged as a future arm in the Fable execution plan; imp_max_p stays 0 until Khadiv unblocks.

## A/B PARKING-FIX CAMPAIGN — LAUNCHED 2026-07-16 (both training)
- **Option A** `a05` (JID 39528692 + resubmit 39529240 for track_s0): task `Unitree-Z1-Hammer-CaT-Impulse` with **CLI `--env.rewards.nail-driven.weight 0.5`** (override verified in env.yaml: nail_driven=0.5, completion=100, imp_max_p=0), 3 seeds × {track,none}, 500 iters/4096 envs. Correct POSITIONAL task-id invocation. Slow warp start (~10 min to dirs); track_s0 died on gn03 (no GPU) → hardened the A100 gate (`|| requeue+exit`) + resubmitted.
- **Option B** `b1` (JID 39529713): NEW task `Unitree-Z1-Hammer-CaT-Impulse-NoTerm`, 3 seeds (none-equivalent, no prior). **B implemented** (no_terminate param pops nail_driven termination + completion 100→1.0; impossible_success rekeyed to fire on depth-success so it stays live; `_stub_env` extended). **Gated LOCALLY: pytest 156/156, validate_rewards A–M PASS, verify_reward_setup PASS.** scp'd to Vega (repo dirty, uncommitted — provenance = b2ed6ee + B diff; Vega smoke confirmed nail_driven popped + completion=1.0). sbatch got a `SINGLE_TASK` mode.
- **EARLY LIVE SIGNALS** (iter ~400 A / ~110 B; bad-behaviour watch `~/watch_badbehavior.py`): **A looks like it WORKS** — delivered_total ≈0.51-0.53 (vs parked nf1 ~0.25, ~2×), nail_driven terminations ≈527-557 (vs nf1 ~2.5-5, ~100×) ⇒ completing not parking; invariants clean (imp_succ=0, contact=1.0, δ=0). **B: seed0 delivered_total=1.03 (1.69×i_ref) FLAGGED** — per-event-capped normalizer >i_ref ⇒ MULTIPLE re-armed contact events = B may MULTI-STRIKE within its non-terminating episode (seeds1,2 normal ~0.5). The "does B stay one clean strike" question, live.
- **POST-RUN PIPELINE (user directive 2026-07-16):** when A+B drain → (1) eval both via `vega_eval.sbatch` (extend glob to a05+b1; Tier-1 self-certifying) → (2) **deep-dive results with Codex analyzing + Fable checking** → (3) produce a NEW training plan for the next round. Decision rule: winner empties the 26-29mm parking band (final-depth histogram) + keeps delivered ∈[0.8,1.3]×i_ref + Λ_j indistinguishable + ~1 contact event; disqualify B if multi-strike (delivered>1.5×i_ref, re-armed events routinely>1).

### TRAINING RESULT (both DONE iter 499, 2026-07-16) — A clean, B disqualified
- **Option A (nail_driven=0.5): CLEAN, consistent across all 6.** reward ~2.8, `delivered_total` ≈**0.52-0.53** (one strike ~0.85×i_ref; vs parked nf1 ~0.25), `Episode_Termination/nail_driven` ≈**525-557** (vs nf1 ~2.5-5, ~100× more completions ⇒ COMPLETING not parking), imp_succ=0, contact=1.0, δ=0. The farm-cut worked.
- **Option B (NoTerm): MULTI-STRIKE BLOW-UP — disqualified.** `delivered_total` exploded to **51× / 31× / 1.9× i_ref** (seeds 0/1/2) and GREW over training (s0: 1.0→0.8→51). The non-terminating episode lets the policy hammer the nail dozens of times ⇒ delivered-impulse METRIC massively contaminated (the exact decision-rule disqualifier). reward ~60 (recurring-completion scale, not comparable). invariants still clean (imp_succ=0, contact=1.0). **Confirms the deep-dive:** terminate-on-success is the correct single-impulsive-strike discipline; B's structural fix re-introduces repetitive hammering — thesis-wrong.
- Bad-behaviour watch (`~/watch_badbehavior.py`) flagged B's multi-strike live from iter ~110. **PROVISIONAL WINNER: Option A** (pending the deterministic-mean eval `JID 39530747` for the parking-band / success / Λ / delivered-vs-i_ref confirmation).

### EVAL RESULT (deterministic mean policy, `eval/ab1/summary.csv`, JID 39530941) — A SOLVES PARKING
Eval TIMED OUT at 25 min with 7/9 rows (6 A + 1 B; B multi-strike → ~13k episodes/ckpt → slow). Node gn03 excluded (bad GPU); GPU fail-fast guard added to `vega_eval.sbatch`.
- **Option A — parking SOLVED, all 6 seeds:** mean-action success **1.00** (vs baseline 0/1 bimodal, 5/6=0), nail_depth **31.7-32.0 mm** (full depth, past the 30 mm line — parking band EMPTY), ep_len **7-8** (single strike then terminate, vs baseline 200-cap), delivered_mean 0.28-0.75 (≈ one i_ref), Λ/cap mean **0.13-0.23** / sampled 0.50-0.78, invariants 0/0. Clean single impulsive strike.
- **Option B — 1/3 eval rows (b1_noterm_s0: succ 1.0, delivered 0.607, ep_len 9, depth 32) looks clean BUT the eval MASKS the multi-strike:** B's play_cfg KEEPS the success termination (so eval can score success) ⇒ eval terminates B at its FIRST strike and never observes the multi-strike. B's real behaviour (delivered 51× i_ref) exists only in its native NON-terminating training env. **B disqualified on native-env behaviour, not on the terminating-eval row.**
- **CONCLUSION: Option A is the winner** — solves parking, single-strike-faithful, invariants clean. Deep-dive + next-plan: Codex analysing → Fable checking (user pipeline, 2026-07-16).

### CODEX × FABLE ANALYSIS (2026-07-16) — corrections + vetted next plan
**Codex analysed, Fable checked all 6 claims against source (GO-WITH-FIXES). Two corrections to the record above:**
- **UNITS:** B delivered = **51.08/31.45/1.149 N·s = 83.8×/51.6×/1.9× i_ref** (raw N·s, not ratios). Multi-strike is NOT a delivered-reward farm (DeliveredImpulseTerm is depth-gated, pays ~0 post-bottom, rewards.py:224) — it's unpenalized post-success behaviour contaminating the METRIC. Disqualification stands (stronger).
- **⚠ THE "CONSTRAINT NOW BINDS" CLAIM WAS WRONG.** Fixing parking moved the policy AWAY from the binding regime: nf1 sampled Λ/cap 0.91-1.12 → a05 sampled **0.50-0.78**, mean 0.13-0.23. Over-cap fraction on a05 = **0** across ~13k episodes/policy. **The healthy policy strikes MORE efficiently = LESS joint reaction = MORE vacuous, not less.** The enforcement branch (E0-vs-E1) is a **guaranteed no-op at the shipped window/cap.** Whether the constraint can bind at all routes through **Khadiv decision (e)** (the window/cap pairing — rescaled caps read 0.61-0.64× even on rigid press-through), NOT more training. (Also: eval Λ is aggregate, not success-conditioned; the mean-rollout is degenerate — deterministic reset + mean action = identical episodes, std~0 — so only the SAMPLED rollout carries a distribution.)

**FINAL VETTED NEXT PLAN (Codex ranked, Fable-corrected):**
- **RANK 1 — bake + bank + 8-seed fixed-impedance campaign (do now, no Khadiv needed).**
  - **Bake `nail_driven` 2.0→0.5 in BOTH** `hammer_env_cfg.py:172` AND `viz/reward_explorer/reward_formulas.py:26` (else `test_reward_viz_parity.py:32` fails — the test Codex missed). Update stale CLAUDE.md "Live weights" + v2 deep-dive listings. Retire the now-dead `NAIL_DRIVEN_W` knob in vega_train.sbatch. validate_rewards safe (reads dynamically).
  - **Bank evidence** (fix N·s units first): a05/b1/nf1 CSVs, checkpoints, TB, env.yaml/agent.yaml, git hash + dirty diff, assets hash.
  - **Add anti-farming config test**: at 26.7-29.5mm the discounted static nail_driven stream < completion incentive.
  - **Re-gate**: pytest (incl. test_reward_viz_parity), validate_rewards A-M, verify_contact_sensor, verify_reward_setup, playback_reference.
  - **Campaign**: F0 (A-none, delivered w=2), F1 (A-track, weak prior, paired), optional **F2 (A-delivered-off, w=0** — resolves A's 0.28-0.75 delivered spread), **8 seeds each**, 500 iters/4096 envs, terminate-on-success, imp_max_p=0. **`--agent.save-interval 50`** → checkpoints model_50/250/499 (Codex's "49/249" don't exist; interval was 250). n=8 for SELECTION (equivalence needs ~41).
- **RANK 2 — enforcement: BLOCKED, and now empirically a no-op.** Take the binding/window-cap question to **Khadiv decision (e)** FIRST. Only if the quantity is re-specified so it CAN bind: extend eval_impulse.py for **sampled per-episode worst-ratio quantiles + over-cap fraction**, evaluate on **model_250** (mid-training, the only place exposure could hide), and run E0-vs-E1 only if AND(Khadiv unblocks (e)/(f)+armature+solver, exposure gate passes). Recalibrate imp_max_p via event-level `1−∏(1−δ_t)` (0.206 is a start but UNDER-penalizes the ~1-read completing strike).
- **RANK 3 — VIC** log-only V0 (fixed-gain) vs V1 (commanded stiffness) after fixed-impedance banked; VIC+CaT only if both show leverage. (Caveat: kp ∉ M(q) → little ballistic leverage; authority is over the windowed reaction, gated on decision (e).)

### RANK 1 EXECUTED — 2026-07-16 (baked + gated + 8-seed campaign launched)
- **Baked `nail_driven` 2.0→0.5** in `hammer_env_cfg.py:172` AND `viz/reward_explorer/reward_formulas.py:26` (parity test green) + CLAUDE.md live-weights. Retired the `NAIL_DRIVEN_W` sbatch knob → `EXTRA_FLAGS` passthrough (adds `DELIVERED_W` for F2). `--agent.save-interval 250→50` (checkpoints 50/250/499).
- **Anti-farming test added** (`test_configs.py::test_nail_driven_not_farmable`): asserts the max discounted sub-threshold hold value < completion; would have caught the weight-2.0 bug (hold 205 > 100), passes at 0.5 (hold 58.8 < 100).
- **Gated LOCALLY:** pytest **170/170**, validate_rewards **ALL PHASES**, verify_contact_sensor OK, verify_reward_setup OK, playback_reference **PHASE M GATE PASS**. Baked file confirmed on Vega (`weight=0.5`).
- **Evidence banked:** `docs/results/assets/2026-07-16_parking_fix/{nf1,ab1}_eval_summary.csv`. Provenance: unitree **b2ed6ee** + baked nail_driven=0.5 diff (UNCOMMITTED — commit pending user OK), assets **b58ccd2**.
- **Campaign v1 (f1/f2, JID 39533117/8) TIMED OUT** — 24 concurrent runs (vs a05's 6) hit warp-cache/cephfs contention: ~8 min startup + ~5s/iter ⇒ 500 iters needs ~50 min > the 40-min `--time`. All 16 F0/F1 CANCELLED at iter ~250-350 (`DUE TO TIME LIMIT`); runs were HEALTHY (delivered ~0.5, ~520 completions, invariants clean) — just clocked out. Only 6 old nf1 checkpoints have model_499; f1/f2 saved model_250-350.
- **RESUBMITTED as g1/g2 with `--time=01:30:00`** (JID 39533758 F0/F1 + 39533759 F2), gn03 excluded. 24/24 completed (warm cache; ~46 min).

### F0/F1/F2 RESULT (g1/g2 eval, model_499, JID 39534996; eval `--time` bumped 25→60 min) — 2026-07-16/17
Eval timed out at 22/24 (~2.5 min/ckpt); 2 F2 seeds (s4,s5) backfilled FRESH=0 (JID 39565824).
| Arm | seeds@succ 1.0 | depth | delivered | Λ/cap(sampled) | invariants |
|---|---|---|---|---|---|
| **F0 (A-none)** | **8/8** | 31.7mm | 0.467 | 0.596 | clean |
| **F1 (A-track, prior)** | **8/8** | 31.8mm | 0.410 | 0.798 | clean |
| **F2 (delivered-off)** | **5/6** ⚠ | 26.4mm | 0.349 | 0.443 | clean |

**Findings:** (1) **A's parking fix is ROBUST at n=8** — F0+F1 = 16/16 seeds succeed, depth ~32mm, ep_len ~7-8 (single strike), invariants clean on every one. The 2.0→0.5 farm-cut is definitively validated. (2) **Prior (F1) doesn't help success** (both 8/8 ceiling; re-confirms Q2) but leaves a strike-dynamics signature (Λ/cap 0.80 vs F0 0.60) → prior is DROPPABLE. (3) **Delivered reward MATTERS for robustness (F2):** off ⇒ 1 seed (s3) COLLAPSED to no-contact (succ 0, contact 0) + survivors strike gentler (dlv 0.35, Λ/cap 0.44) → **KEEP delivered_impulse**. Over-cap rare (one F1 seed grazes 1.004; rest sub-cap) — constraint largely vacuous on the healthy policy (consistent w/ Fable). **BANKED as the canonical fixed-impedance baseline.**
