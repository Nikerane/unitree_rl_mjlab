# CODEX HANDOVER — fixed-impedance phase, Z1 hammer impact-safe RL (2026-07-20)

**Audience:** Codex, an AI coding agent that will independently read code, run CPU probes on this
Mac, and launch/evaluate GPU training on Vega. **Repo:** `/Users/nikerane/repos/unitree_rl_mjlab`,
branch `soft-cat` (HEAD at writing: `5dac687`). **Authority contract:** code > `docs/README.md`
index > living docs; `docs/archive/**` is historical evidence only. This document is a working
handover, not marketing — every number below is quoted from the banked results docs, and every
file:line was verified against the tree on 2026-07-20.

> **LIVE CAVEAT (§3 B.2 ceiling verdict):** the "impact ceiling is a hardware joint-velocity limit"
> conclusion is under an independent adversarial re-check (does a coordinated multi-joint *whip* reach
> higher end-effector speed than the ~1.4 m/s scripted straight-down strike, within the per-joint
> 3.1415 rad/s budget? `ceiling.py` reports a 3.97 m/s *kinematic* head-speed ceiling). If that
> re-check finds real trajectory headroom, the ceiling is a controller/trajectory-shaping limit a
> trained policy could beat — treat the "hardware wall" claim as PROVISIONAL until it resolves, and see
> whether `docs/results/2026-07-20_B2_effort_ceiling.md` has been updated with the verdict.

Primary sources (read these before trusting any secondary summary, including this one):

| doc | what it holds |
|---|---|
| `docs/results/2026-07-17_phase0_diagnostics.md` | Phase-0 CPU battery A1/A2/A3/A5/A6 — the press-vs-ballistic proof |
| `docs/results/2026-07-17_fixed_impedance_deep_dive.md` | system model (the two spine questions Q1/Q2) |
| `docs/results/2026-07-17_fixed_impedance_execution_plan.md` | plan of record (Phases 0–3 + guardrails) |
| `docs/results/assets/2026-07-17_depth_gate_sweep/RESULT.md` | Phase-2 GPU sweeps (dg, ip24/ip48) + af1 audit-fix validation |
| `docs/results/2026-07-20_B2_effort_ceiling.md` | B.2 effort ablation + delta_pos_scale sweep — the unified ceiling verdict (LATEST) |
| `docs/results/2026-07-19_reward_code_fault_audit.md` + `2026-07-19_fault_fix_plan.md` | the 11 audit findings F1–F11 + fix tracks |
| `docs/results/2026-07-19_next_experiments_codex.md` | prior ranked experiment menu (List A/B) — updated by §7 below |
| `docs/results/2026-07-20_eval_protocol.md` | the arm-vs-arm comparison protocol (binding for all A/Bs) |

---

## 1. TL;DR / STATE OF PLAY

- **The fixed-impedance program is essentially complete and defensible.** Phase 0 (CPU physics),
  Phase 2 (GPU reward-lever sweeps), and B.2 (CPU causal ablations) converge on one story with no
  loose empirical threads on the main line.
- **Headline verdict (2026-07-20, `2026-07-20_B2_effort_ceiling.md`):** the ~0.87× i_ref /
  ~1.4 m/s delivered-impulse ceiling is a **HARDWARE JOINT-VELOCITY limit** (real Z1: 3.1415 rad/s
  per joint). Proven on all three axes: **not reward-limited** (depth-gate removal and
  impact_progress ×3 both flat), **not torque-limited** (effort ×1→×3 is an exact no-op on contact
  speed and delivered impulse), **controller-bandwidth is the mechanism** (`delta_pos_scale` lifts
  v_touch 1.27→4.96 m/s) **but every faster scale demands joint speeds the real Z1 cannot produce**
  (shipped 0.15 already runs the dominant joint at 2.556 rad/s ≈ 81% of the cap).
- **The enforced constraint quantity Λ is a press integral, not ballistic momentum** (A2: −76%
  under solref×2, in lockstep with delivered), and the **ballistic quantity is structurally
  vacuous** across the reachable envelope (0.19–0.39× cap; ≤0.50× even at 30× armature). The
  constraint "binds" (1.184×) only under rigid target + press-through + the shipped 27.3 ms-cap /
  50 ms-window mismatch — bindability is a pairing choice (Khadiv decision (e)), not a training
  result.
- **The constraint is LOG-ONLY** (`imp_max_p=0`, `env_cfgs.py:305`) and stays that way until
  Khadiv decision (e). Enforcement day additionally requires the F2 (cap Δt-basis) + F3 (δ
  multi-read) recalibrations — neither is landed.
- **VIC (variable impedance) is the deferred next chapter**, now motivated by a measured ceiling
  on every axis rather than assertion. Do not touch it (guardrail §5).
- **Code state is clean and audited:** 6 of 11 audit findings fixed and GPU-validated
  behavior-neutral (af1, git `6b447bf`); the F1 delivered-impulse escrow/press-farm is closed with
  a discriminating unit test (locked-nail press: old code would bank 12.9× i_ref, new code pays 0).
  Provenance discipline is enforced (eval CSV `git_hash` gets `-dirty` on an uncommitted tree).
- **What's live for you:** delta_pos_scale-at-the-hardware-cap (f), harder-target-to-bind (b),
  the F7 latch A/B (d), C1 training-time Λ instrumentation (e), and the matched window/cap
  characterization (a). The Vega effort ablation from the old menu is now LOW-VALUE — drop it.

---

## 2. THE SYSTEM (understand this before touching anything)

### 2.1 The task
The Unitree Z1 arm (6 DoF, hammer rigidly fixtured to link06 via a 3D-printed bracket — no
gripper) drives a nail into a wooden block. Success = nail slide depth ≥ **0.030 m**
(`NAIL_SUCCESS_THRESHOLD`, `src/tasks/hammer/nail_block.py:66`); physical stop at **0.032 m**
(`NAIL_GOAL_DEPTH`, `nail_block.py:50`). The episode **TERMINATES on success** (`nail_driven`
termination, `src/tasks/hammer/hammer_env_cfg.py:236-239`; 20 s timeout otherwise,
`hammer_env_cfg.py:310`). The nail **ratchets**: `frictionloss="30.0"` on the slide joint
(sibling-repo asset `~/repos/safe_impact_manipulation/hammer_z1_env/assets/nail_block_scene.xml:38`),
total nail mass 7 g (shaft 5 g + head 2 g, `nail_block_scene.xml:40-43`), `gravcomp="1"` on the
nail body (`:32`) so it holds position without a spring and never returns. Consequence: the
*minimum* impulse reaching 0.030 m suffices; a healthy policy strikes once (eval ep_len ≈ 7.3
control steps) and the episode ends. Control: 50 Hz (decimation 10 × physics dt 0.002 s,
`hammer_env_cfg.py:309`).

### 2.2 The action space (why this is "fixed impedance")
Position-only Differential IK on the hammer-head site: 3-D delta-position action, scaled by
`Z1_HAMMER_DELTA_POS_SCALE = 0.15` (`src/assets/robots/unitree_z1/z1_constants.py:229`, wired at
`src/tasks/hammer/config/z1/env_cfgs.py:121`). PD gains are FIXED constants: kp=1000/kd=100 on
joints 1,3–6 with `effort_limit=30` N·m, kp=1500/kd=150 on joint2 with `effort_limit=60` N·m
(`z1_constants.py:83-94`). The policy never commands stiffness — that is what "fixed impedance"
means here; VIC (policy-commanded `set_gains`) is the deferred next phase. Head speed scales as
~`delta_pos_scale/dt` through the controller; actions clamp at ±1 (`clip_actions=1.0`). Key
physics fact: **kp enters τ, NOT the mass matrix M(q)** — commanded stiffness has zero lever on
ballistic impulse (`m_eff·v`), which is both why fixed impedance saturates and why *static* VIC
would too (§6h).

### 2.3 The THREE impulse quantities (never conflate them)
1. **Enforced robot-side Λ_j** — `SubstepImpulseAccumulator`
   (`src/tasks/hammer/mdp/impulse_bound.py:72-194`): Λ_j = contact-masked, baseline-subtracted
   Σ|qfrc_constraint_j|·dt over a **25-substep (50 ms) TIME-based sliding window**
   (`event_window_substeps=25`, wired `env_cfgs.py:258`, ring buffer `impulse_bound.py:134-139`),
   accumulated at 500 Hz, gated by the `hammer_nail_contact` sensor (face-geom `hammer_head_0` vs
   nail body only, `env_cfgs.py:101-113`). The 50 Hz constraint read is
   `max(pulse latch, current window sum)` (`impulse_bound.py:151-154`). This is what
   `joint_impulse_excess` (`src/tasks/hammer/cat/constraints.py:35`) compares against the caps.
   It **deliberately counts press reaction** (up to one window's worth) — whether that is the
   right bounded quantity is Khadiv decision (e) (`impulse_bound.py:105-109`).
2. **Object-side delivered ∫F·dt** — `SubstepDeliveredImpulse` (`impulse_bound.py:197-279`):
   episode-cumulative axial impulse from the separate `hammer_nail_impulse` netforce sensor
   (`env_cfgs.py:235-242`), **per-event capped** at 25 substeps with a 25-substep re-arm debounce
   (`env_cfgs.py:280-281`, cap/debounce logic `impulse_bound.py:269-274`) so a press pays at most
   one window per genuine contact event (≤50% duty even under flicker). This feeds the
   `delivered_impulse` reward and is the friction/weld-immune ground truth for the C0 gate.
3. **Log-only ContactRow Λ** — `ContactRowImpulseAccumulator`
   (`src/tasks/hammer/mdp/contact_row_impulse.py`, wired `env_cfgs.py:266-273`): rigorous
   Jᵀ·F-over-efc-rows diagnostic that validates the enforced approximation. Feeds nothing;
   disabled on Vega runs for perf (`--env.metrics.substep-impulse-rows.params.enabled False`,
   baked into `vega_train.sbatch:60`).

Context on the qfrc contamination this design answers: raw `qfrc_constraint` is ~41–48%
dof-FRICTION (not a weld — the training scene has neq=0); the contact-mask + frozen pre-contact
baseline subtraction (`subtract_baseline=True`, `env_cfgs.py:253`, mechanism
`impulse_bound.py:177-186`) removes the dominant share.

### 2.4 The soft-CaT constraint (currently log-only)
`CatSoftHook` (`src/tasks/hammer/cat/hook.py:39-250`), a full-step MetricsTerm registered at
`env_cfgs.py:298-314`: computes raw margin c = Λ_j − cap_j, maps to
δ = min_p + clamp(c/c_max)·(max_p − min_p) (`constraint_manager.py:49-57`), soft-ORs across
joints/constraints, writes `env.extras["cat_delta"]` + `cat_r_pos` for `CatPPO`, which discounts
**positive reward only** (r_total − δ·r_pos + (1−δ) value bootstrap; negative terms
`action_rate`/`joint_pos_limits` ride through undiscounted — `_NEG_TERMS`, `hook.py:36`, with
per-call polarity re-validation `hook.py:165-177` = audit F4 fix).

- **`imp_max_p = 0.0` (`env_cfgs.py:305`) ⇒ δ_imp ≡ 0, a TRUE no-op** — hard short-circuit at
  `hook.py:218-220` (probs zeroed, normalizer untouched). Byte-identical to stock PPO. LOG-ONLY.
- **Caps `IMP_J_LIMIT = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]` N·m·s**
  (`env_cfgs.py:35`, joint2 doubled because τ_rated=60): hardware-derived — τ_rated × 2
  (Harmonic-Drive Repeated-Peak) × the 2026-07-06 measured Δt ≈ 27.3 ms
  (`derive_impulse_thresholds.py`; record `docs/results/2026-07-10_c2_enforcement_record.md`).
  Note the Δt-basis mismatch vs the 50 ms window = audit F2, open.
- Construction guards (audit F5, `hook.py:43-101`): full probability/EMA/limit domain validation,
  plus a guard that refuses enforcement against the `Z1_JOINT_IMPULSE_LIMIT=0.1` placeholder
  (`impulse_bound.py:58`, guard `hook.py:95-101`).
- Sentinels (both wired, `env_cfgs.py:330-336`): `contact_seen` ≈ 1.0 healthy;
  `impossible_success` (success with Λ ≡ 0 ⇒ dead instrument — keyed on the depth predicate
  alone since audit F9, `impulse_bound.py:346-355`) must stay flat 0.0.

### 2.5 The 8-term reward (live weights — read from code, not specs)
All in `hammer_env_cfg.py` except `delivered_impulse` (added by the cat_impulse arm in
`env_cfgs.py:350-358`):

| term | weight | where | notes |
|---|---|---|---|
| `approach` | 0.1 | `hammer_env_cfg.py:157` | Gaussian head→nail distance |
| `nail_driven` | 0.5 | `hammer_env_cfg.py:177` | Gaussian on depth; cut 2.0→0.5 2026-07-16 (parking farm) |
| `nail_depth_delta` | 600 | `hammer_env_cfg.py:189` | ratcheted positive depth progress |
| `impact_progress` | 8.0 | `hammer_env_cfg.py:201` | double-gated ante-impact axial speed (`rewards.py:144-211`; first-contact gate `rewards.py:210` = audit F7's fragile spot) |
| `completion` | 100 | `hammer_env_cfg.py:216` | one-shot on depth ≥ 0.030 |
| `action_rate` | −0.01 | `hammer_env_cfg.py:224` | in `_NEG_TERMS` |
| `joint_pos_limits` | −10 | `hammer_env_cfg.py:228` | in `_NEG_TERMS` |
| `delivered_impulse` | 2.0 | `env_cfgs.py:350-358` | pays delivered-∫F·dt increments only on nail-advancing steps, normalized by `I_REF_DELIVERED = 0.6094` N·s (`env_cfgs.py:45`); non-progress impulse is DISCARDED every step (F1 fix, `rewards.py:270-275`) — the old escrow and the `depth_gate` toggle are gone |

Task IDs (registered in `src/tasks/hammer/config/z1/__init__.py`): the workhorse is
**`Unitree-Z1-Hammer-CaT-Impulse`** (cat_impulse arm, log-only). `-Track` adds the annealed
imitation prior; `-NoTerm` removes success termination (known tap-and-park pathology — not an
N-strike task).

---

## 3. ALL RESULTS IN DETAIL

### 3.1 Phase 0 — CPU diagnostic battery (2026-07-17, no GPU)
Source: `docs/results/2026-07-17_phase0_diagnostics.md`. Raw JSON + probes under
`docs/results/assets/2026-07-17_fixed_impedance_diag/`.

**A2 — solref sensitivity (THE press discriminator).** Reference strike, baseline vs contact
solref ×2 (`sensitivity_run.py --config baseline|solref2x`):

| quantity | baseline | solref×2 | Δ |
|---|---|---|---|
| worst-joint Λ (j2, full-event, baseline-subtracted) | 0.3029 | 0.0718 | **−76.3%** |
| object-side delivered ∫F·dt | 0.6094 (= i_ref exactly, harness sanity check) | 0.1429 | **−76.6%** |
| peak axial force | 21.6 N | 12.4 N | −42.6% |
| contact window | 44 ms | 38 ms | −13.6% |

Λ tracks delivered in near-perfect lockstep and both are maximally solref-fragile. A
momentum-pinned ballistic integral would be solref-ROBUST ⇒ **the enforced Λ is a PRESS
integral**. (Real contact window 44 ms sits between the 27.3 ms cap basis and the 50 ms window.)

**A1 — approach-velocity sweep 0.5×–8× (`probe_battery.py` T1).** Effort-clamped + press:

| cmd factor | v_touch (m/s) | window | peak/mean F | rebound | Λmax |
|---|---|---|---|---|---|
| 0.5× | 0.77 | 48 ms | 1.37 | no | 0.175 |
| 1.0× | 1.33 | 44 ms | 1.56 | no | 0.347 |
| 2.0× | 1.41 | 38 ms | 1.68 | no | 0.291 |
| 4.0× | 1.38 | 28 ms | 1.51 | no | 0.182 |
| 8.0× | 1.31 | 42 ms | 1.61 | no | 0.362 |

Commanding 8× harder does not raise contact speed past ~1.4 m/s; Λ is **non-monotonic in v and
never exceeds 0.36**; peak/mean 1.4–1.7 (a collision shows 5–20×); zero rebound anywhere.
(T3: a sustained press integrates Λ smoothly to ~0.29 over 38 ms.)

**A3 — window/cap pairing on a RIGID target (`impedance_one.py`, fixed gains, 1.38 m/s;
armature-locked nail, lock verified: displacement 0.017 mm). The decision-(e) table.**
Formula-consistent 50 ms caps `[3,6,3,3,3,3]` = 1.829× the shipped `[1.64,3.28,…]`:

| quantity | worst Λ/cap — shipped caps (27.3 ms basis) | — formula caps (50 ms basis) | binds? |
|---|---|---|---|
| IMPACT-only (9 substeps, genuine ballistic) | **0.609** | 0.333 | NO |
| ACCUM (shipped 50 ms sliding window = *the enforced quantity*) | **1.184** | 0.647 | shipped YES / formula NO |
| FULL (588 ms sustained press, uncapped diagnostic) | 10.87 | 5.94 | n/a |

"The constraint binds" is true ONLY at the intersection of rigid target + press-through + the
shipped cap/window mismatch. Change any one and it does not bind. On the real yielding nail,
ACCUM ≈ 0.19–0.36 (A1/A2).

**A5 — rigid + velocity-injection ceiling (`ceiling.py`, analytic).** m_eff (armature-coupled)
0.544 kg; kinematic head-speed ceiling 3.97 m/s; effort-reachable 1.35 m/s. Ballistic worst
Λ/cap at 1.35 m/s: **e=0 (real yielding nail) 0.194; e=1 (perfectly elastic) 0.388**. The cap is
crossed only under the doubly-unreachable combo e=1 AND 3.48 m/s (2.6× the effort limit; joint3).
Ballistic-loaded joints are j2/j3/j4 (moment arms 0.50/0.43/0.33) vs the press-regime's j1 — a
clean press-vs-ballistic tell in any future data.

**A6 — armature sensitivity (`armature_check.py`; Khadiv (f) model-uncertainty bound).**
m_eff: ×1 = 0.544, ×3 = 0.671, ×10 = 0.901, ×30 = 1.403 kg. Ballistic Λ ∝ m_eff at 1.35 m/s:

| armature | m_eff (kg) | Λ/cap e=0 (real) | Λ/cap e=1 (elastic) |
|---|---|---|---|
| ×1 (modeled) | 0.544 | 0.194 | 0.388 |
| ×10 | 0.901 | 0.321 | 0.643 |
| ×30 (gear²-plausible) | 1.403 | 0.500 | 1.001 |

Even at 30× armature the realistic (e=0) ballistic impulse is **0.50× cap** — vacuity survives
the model uncertainty; only ×30 AND e=1 reaches cap.

### 3.2 Phase 2 — GPU reward-lever sweeps (Vega; `.../2026-07-17_depth_gate_sweep/RESULT.md`)
Protocol: `Unitree-Z1-Hammer-CaT-Impulse`, 500 iters, 4096 envs, `imp_max_p=0`, 3 seeds/arm.

**Depth-gate removal = NO-OP** (train JIDs 39626161/39626162, eval 39627853):

| arm (n=3) | delivered/i_ref det. | sampled | worst Λ/cap sampled | worst Λ/cap det. | success | ep_len |
|---|---|---|---|---|---|---|
| dg1 gate-ON (shipped) | 0.778 (0.58–1.10) | 0.819 | 0.533 | 0.174 | 1.00 | 7.3 |
| dg0 gate-OFF | 0.713 (0.50–1.08) | 0.698 | 0.661 | 0.157 | 1.00 | 7.3 |

Statistically indistinguishable (gate-off if anything slightly lower); invariants clean both
ways. Why: the terminating task ends the instant the nail seats, so the post-seating phase the
ungated reward would pay for does not exist — and underneath, the ~1.4 m/s ceiling caps delivered
regardless of reward. (This no-op result is why the F1 fix later *deleted* the
`depth_gate=False` path entirely.)

**impact_progress weight sweep = FLAT, then DIVERGENT** (JIDs 39631659/39631660; data from
training TB `delivered_total` because the eval sbatch flaked exit-53 on the ip24 single-glob
campaign — a launcher bug, not a physics result):

| arm | impact_progress.w | delivered_total/i_ref (converged seeds) | imp_peak_joint2 |
|---|---|---|---|
| dg1 (control) | 8 | 0.87 (0.84/0.89/0.87) | ~0.17 |
| ip24 | 24 | **0.86** (0.86/0.86) — FLAT; seed0 diverged early | 0.21–0.27 |
| ip48 | 48 | **training DIVERGED 3/3** ~iter 50 (only model_50 saved, exit 120 on teardown) | — |

Tripling the ante-impact velocity reward did not raise delivered impulse; 6× broke training.
Launcher lesson from ip48/ip24: the failure-propagation fix (Codex P1) correctly surfaced the
diverged runs as FAILED — always check job state AND that `model_499.pt` exists before eval, and
expect the eval sbatch to fail loudly (exit 53) on a glob matching zero completed checkpoints.

**Unified Phase-2 conclusion:** reward reshaping cannot push delivered past ~0.87× i_ref. Keep
`depth_gate` semantics as shipped and `impact_progress.weight=8`.

### 3.3 B.2 — the ceiling ablations (2026-07-20, CPU; `2026-07-20_B2_effort_ceiling.md`)
**Effort ablation (`effort_sweep.py`): torque is an EXACT no-op.** Scale ONLY the arm
`effort_limit` (30/60 → ×1/1.5/2/3), everything else fixed, scripted strike at nominal (speed 1)
and aggressive (speed 6) tracking:

| speed | effort×1 | ×1.5 | ×2 | ×3 | reading |
|---|---|---|---|---|---|
| nominal (1) | v_touch 1.334, deliv 1.000× i_ref | 1.334 / 0.986× | 1.334 / 0.986× | 1.334 / 0.986× | flat |
| aggressive (6) | v_touch 1.408, deliv 0.678× | 1.408 / 0.678× | 1.408 / 0.678× | 1.408 / 0.678× | flat |

Contact speed and delivered are **identical to 4 decimals across effort ×1→×3** (30→90 N·m).
30 N·m already saturates what the controller demands. (Caveat: single-strike delivered at speed 6
is one-shot noisy, 0.34–0.75 across earlier runs; the robust signals are v_touch and the exact
effort-invariance.)

**delta_pos_scale sweep (`dps_sweep.py`): bandwidth lifts speed, hardware forbids it.**
Saturated straight-down descent; real Z1 joint-velocity limit 3.1415 rad/s (all joints,
z1_description URDF):

| delta_pos_scale | v_touch (m/s) | deliv ×i_ref | max\|q̇\| (rad/s) | reachable on real Z1? |
|---|---|---|---|---|
| **0.15 (shipped)** | 1.267 | 1.01 | **2.556** | **YES** (81% of 3.1415) |
| 0.30 | 2.745 | 1.17 | 4.945 | NO |
| 0.50 | 3.519 | 0.16* | 5.122 | NO |
| 1.00 | 4.963 | 0.47* | 5.223 | NO |

(*delivered at dps≥0.5 is single-strike chaos — v_touch + max|q̇| are the robust signals.)

**UNIFIED CEILING VERDICT:** not reward-limited (Phase 2), not torque-limited (effort ablation),
controller-bandwidth is the mechanism (dps) but capped by the real Z1 3.1415 rad/s per-joint
velocity limit, which 0.15 already sits near. **The fixed-impedance impact-impulse ceiling is a
HARDWARE JOINT-VELOCITY limit.** Consistent with the prior closed-loop finding that trained
policies' worst-case |q̇| already exceeds π. This is precisely the wall VIC is designed to beat
(stiff windup stores strain energy; compliant release decouples contact speed from steady-state
joint velocity) — the VIC motivation now rests on a measured ceiling, not assertion. NOT
VIC-in-disguise: dps is an action-bandwidth knob with a strike-placement accuracy tradeoff, and
it is already at the hardware cap.

### 3.4 The audit — 11 findings F1–F11 and where each stands
Source: `2026-07-19_reward_code_fault_audit.md` (two independent reviewers, Fable 5 + Codex;
CONSENSUS = both flagged). None corrupted banked results; all were latent/enforcement-day.

| # | finding (one line) | status |
|---|---|---|
| **F1** | `delivered_impulse` escrow: the depth-gate *delayed* press-farm payout to the next nudge instead of preventing it; `depth_gate=False` + `no_terminate` was an unbounded farm | **FIXED** (`0d95ead`): escrow→discard — credit baseline advances every step (`rewards.py:270-275`); the `depth_gate` param + ungated path DELETED. Discriminating test: `tests/test_delivered_impulse_reward.py:53` `test_no_press_backlog_farm`. Proven not-a-no-op: a locked-nail press probe accrues 7.87 N·s → OLD code would bank **12.9× i_ref** collectable on any later nudge; NEW code pays **0.0**, while a real strike is still paid in full. Independently corroborated by Gemini + Codex. |
| **F2** | IMP_J_LIMIT time-basis mismatch: caps derived at Δt≈27.3 ms, window integrates 50 ms — legitimate strikes would be graded against mis-calibrated caps the moment `imp_max_p>0` | **DEFERRED, enforcement-gated.** Guarded by comments (`env_cfgs.py:27-34`) + the placeholder guard only; the planned window-metadata assert is NOT yet in code. Must be fixed before any `imp_max_p>0`. |
| **F3** | δ multi-read: window(25) > decimation(10) ⇒ ~3 reads/violation, but a task-completing strike gets ~1 (reset truncation) — enforcement pressure becomes outcome-dependent, ~3× the C2-calibrated intent | **DEFERRED, enforcement-gated.** Calibration note at `impulse_bound.py:98-104`. Fix = event-level single δ pulse (or per-event survival recalibration) as part of enforcement bring-up. |
| F4 | `_NEG_TERMS` polarity checked once ⇒ a curriculum ramping a weight negative would get (1−δ)-discounted (penalty evasion) | **FIXED** (`1776740`): re-validated every call, `hook.py:165-177`. |
| F5 | incomplete CaT param domain validation (imp_max_p=2 ⇒ δ>1 etc.) | **FIXED** (`0d95ead`): full-domain guards, `hook.py:61-101`. |
| F6 | reward normalizers (`std`, `v_expected`, `i_ref`, `sigma`) accept 0/NaN via CLI sweep | **FIXED** (`8ae51c9`): construction guards, e.g. `rewards.py:186-187`, `:258-259`. |
| **F7** | `impact_progress` first-contact gate (`rewards.py:210`, `compute_first_contact`) misses within-interval rebounds and boundary-straddling contacts ⇒ phase-random underpayment of the largest per-strike term, which *selects for press-through over clean strikes* | **DEFERRED — behavior-changing.** The one open fix that could shift the fixed-impedance result. Needs the B1-style CPU phase-sweep then a Vega A/B (§7 CODEX-3). |
| F8 | Λ is blind to non-nail impacts (arm/handle vs block/floor) — under enforcement, δ-pressure could route energy through unmonitored contacts | **NO-CODE**: thesis scoping sentence ("per-joint impulse bounded *at the nail contact*"); possible second constraint term later. |
| F9 | `impossible_success` OR'd `reset_terminated` ⇒ a velocity-CaT termination pre-contact would false-fire the kill-the-run alarm | **FIXED** (`0d95ead`): keyed on depth predicate alone, `impulse_bound.py:346-355`. |
| F10 | hook + stock-PPO silently enforces nothing (pairing check is one-directional) | **DEFERRED.** The env-side "a δ-consumer flagged extras" assert is not landed. Mitigation: always launch `-CaT-Impulse` via the registered rl_cfg (CatPPO); never hand-pair. |
| F11 | i_ref was an unversioned literal | **FIXED** (`1776740`): single provenance-bound module constant `I_REF_DELIVERED` (`env_cfgs.py:37-45`); any geometry/reference/solver/action change ⇒ re-run `derive_impulse_thresholds.py` and update that ONE constant. |

**af1 GPU validation (2026-07-19, `RESULT.md` bottom):** the fully audit-fixed code
(F1/F5/F9/F6/F4/F11), 3 seeds, imp_max_p=0, eval git_hash `6b447bf`: delivered/i_ref sampled
1.21/0.87/0.64 (mean 0.90×), det 1.25/0.56/0.61 (mean 0.81×), success 1.00 all seeds, invariants
clean — **indistinguishable from the pre-fix dg1 baseline (0.82× sampled): behavior-neutral,
regression-free, triple-verified** (the F1 evidence is the unit test, not the flat GPU curve —
the healthy policy never presses, so the curve can't distinguish "works" from "no-op").
Provenance footnote: the first-pass CSV carried a stale hash from a pre-fix scp-on-dirty-tree
deploy; content was shasum-verified identical and re-stamped — this incident is why the
provenance discipline in §4.3 now exists.

---

## 4. HOW TO RUN EXPERIMENTS

### 4.1 CPU probes (this Mac, conda env `unitree_mjlab`, ~1–2 min each, no GPU/GL)
```
cd /Users/nikerane/repos/unitree_rl_mjlab
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/results/assets/2026-07-17_fixed_impedance_diag/probes/<probe>.py
```
All probes spec-wrap at runtime — **no repo/asset file is mutated**. The suite
(`docs/results/assets/2026-07-17_fixed_impedance_diag/probes/`):

| probe | what it does |
|---|---|
| `effort_sweep.py` | B.2 effort ablation (scale arm effort_limit only; scripted strike; v_touch + delivered) |
| `dps_sweep.py` | delta_pos_scale sweep (saturated descent; v_touch, delivered, max\|q̇\| vs 3.1415) — the sweep grid is the literal `for dps in (0.15, 0.30, 0.50, 1.00):` at `dps_sweep.py:74`; edit it for finer grids |
| `probe_battery.py` | A1: T1 speed sweep 0.5–8×, T2 stiffness, T3 sustained press, T4 lift-restrike; writes `A1_speed_battery.json` |
| `sensitivity_run.py` | A2: `--config baseline\|halfdt\|solref2x` solver-sensitivity measurement (JSON out) |
| `impedance_one.py` | A3: rigid-target (armature-locked nail, <1 mm assert) window/cap 2-point; also exposes `--kp_scale` — **kp_scale>1 is VIC-in-disguise, do not use it for results** |
| `ceiling.py` | A5: analytic m_eff / crossing-velocity / VIC-feasibility bound |
| `armature_check.py` | A6: dof_armature verification + ×1/×10/×30 m_eff sensitivity |
| `nail_sweep.py` | harder-target frontier: `mass_scale`, `friction`, `speed`, `hold` knobs (asset wrapped, not edited) |
| `frontier_one.py` / `frontier_sweep.py` | rigid-target velocity-injection Λ frontier (one env/process) |
| others (`whip*.py`, `live_lambda.py`, `plumbing.py`, `tun_dt.py`, …) | scratch diagnostics from the same campaign; read before trusting |

**Tracking caveat:** only `effort_sweep.py` and `dps_sweep.py` in that directory are
git-TRACKED; the rest are untracked working copies (originals of most live committed under
`docs/results/assets/2026-07-12_impulse_vacuity/probes/`). If a probe becomes load-bearing for a
result you write up, commit it (named file, see §5).

### 4.2 Vega GPU training
- Access: `ssh vega`. Repo at `~/repos/unitree_rl_mjlab`. Python: **`.venv/bin/python`** (a uv
  venv — it has **no pytest and no pip**; all test-gating happens locally on conda FIRST).
- Launch pattern (array required — the sbatch asserts `SLURM_ARRAY_TASK_ID`,
  `vega_train.sbatch:41`); single-arm campaigns use SINGLE_TASK mode (`vega_train.sbatch:44-48`):
```
ssh vega
cd ~/repos/unitree_rl_mjlab
CAMPAIGN=myexp SEEDS="0 1 2 3 4" \
  SINGLE_TASK=Unitree-Z1-Hammer-CaT-Impulse SINGLE_SHORT=myarm \
  sbatch --array=0-4 --export=ALL,CAMPAIGN,SEEDS,SINGLE_TASK,SINGLE_SHORT \
  scripts/slurm/vega_train.sbatch
```
  This runs, per array task (`vega_train.sbatch:57-60`): `scripts/train.py <task> --gpu-ids [0]
  --agent.run-name ${CAMPAIGN}_${short}_seed${seed} --agent.max-iterations 500
  --agent.save-interval 50 --env.scene.num-envs 4096 --env.metrics.cat-soft.params.imp-max-p 0
  --env.metrics.substep-impulse-rows.params.enabled False`.
- Reward-weight overrides via env vars (`vega_train.sbatch:34-38`): `NAIL_DRIVEN_W`,
  `DELIVERED_W`, `IMPACT_W` (add them to `--export=ALL,...`). Any other term (e.g.
  `completion`) has **no passthrough** — either add one line to the sbatch (commit it) or pass
  the tyro flag directly in a custom launcher: `--env.rewards.<term-name>.weight X`
  (kebab-case, e.g. `--env.rewards.impact-progress.weight 24`).
- Eval:
```
CAMPAIGN=myexp CKGLOBS="*myexp_*" \
  sbatch --export=ALL,CAMPAIGN,CKGLOBS scripts/slurm/vega_eval.sbatch
```
  This globs `logs/rsl_rl/z1_hammer/$pat/model_499.pt` (`vega_eval.sbatch:33-40`) and drives
  `scripts/eval_impulse.sh` (protocol pinned in its header: play cfg, imp_max_p forced 0,
  256 envs, ≥512 episodes/policy, mean-action row + `_sampled` repeat, seed 42), appending rows
  to `eval/${CAMPAIGN}/summary.csv`.
- **Vega ops that BITE:**
  - Bad nodes gn03/gn34 are baked into `--exclude` (`vega_train.sbatch:18`,
    `vega_eval.sbatch:15`). They kill jobs at SLURM setup: state FAILED, ExitCode **0:53**, NO
    log file. A submit-time `--exclude` OVERRIDES the baked one — if you add nodes, pass the
    FULL list (`gn03,gn34,...`).
  - ~8–10 min warp-kernel/cephfs startup before iteration 1; a 500-iter run is ~19 min total.
    Budget both into `--time`.
  - The train script self-guards A100 presence and requeues otherwise (`vega_train.sbatch:56`).
  - Diverged runs: check `sacct` state and the presence of `model_499.pt` before evaling (the
    ip48 lesson — only `model_50.pt` existed, and the eval glob correctly found nothing).

### 4.3 PROVENANCE DISCIPLINE (critical; the fix is one day old)
- Deploy = **git commit locally → push → `ssh vega "cd ~/repos/unitree_rl_mjlab && git pull"`**.
  NEVER `scp` tracked files onto Vega — that's how the af1 stale-hash incident happened.
- Both the train log (`vega_train.sbatch:52-55`) and every eval CSV row
  (`scripts/eval_impulse.py:244-258`) stamp the short git hash **with `-dirty` appended if any
  tracked file is uncommitted**. A `-dirty` row is QUARANTINED: commit, redeploy, re-run.
- Note the LOCAL tree currently has `CLAUDE.md` modified + several untracked probes/docs — fine
  for CPU work, but commit anything a Vega run depends on before launching.
- **Pre-training gate (run locally on conda BEFORE every Vega launch):**
```
cd /Users/nikerane/repos/unitree_rl_mjlab
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python -m pytest \
  tests/test_delivered_impulse_reward.py tests/test_impulse_bound.py \
  tests/test_cat_soft_hook.py tests/test_impact_progress_reward.py \
  tests/test_configs.py tests/test_signal_safety_net.py
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/validate_rewards.py     # ALL phases A–M must pass
PYTHONPATH=. /Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python \
  docs/research/reward-design/verify_contact_sensor.py
```

### 4.4 Eval comparison protocol (binding for every A/B; `2026-07-20_eval_protocol.md`)
1. **≥5 seeds per arm** (n=3 is not enough: af1 seed spread was 0.56–1.25× i_ref).
2. **Use the `*_sampled` CSV columns** (`eval_impulse.py:382-384`) for every distributional
   claim — the deterministic columns are n=1-effective (play cfg: zero reset noise + mean action,
   all 256 envs identical).
3. **Pre-register the primary metric + decision rule** before launching (template at the bottom
   of the protocol doc). One primary metric; the rest are secondary/diagnostic.
4. Report seed-level mean ± std + min/max and a **Mann–Whitney U across seeds** (two-sided).
   "Real" = p<0.05 AND the pre-registered practical threshold; otherwise say "not
   distinguishable at this n".
5. **One held-out robustness eval** with training-matched ±0.05 rad reset-pose noise.
6. **Instrument gate:** any row with `impossible_success_n>0` or `lambda_dead_n>0` fails the
   whole arm — do not analyze it.
7. **Provenance:** every row needs a clean `git_hash` (no `-dirty`).
Aggregate with `scripts/compare_runs.py` (per-arm mean±std bands, strips `_seedN`), MWU as a
small post-step over `summary.csv`.

---

## 5. HARD GUARDRAILS (violating any of these invalidates the work)

1. **`IMP_J_LIMIT` (`env_cfgs.py:35`) stays at the hardware-derived values. NEVER tighten the
   caps to manufacture a binding result** — the entire defense of the constraint rests on the
   caps being manufacturer/hardware provenance, and a tuned cap destroys it. (Explicit user
   decision; also `2026-07-17_fixed_impedance_execution_plan.md` guardrails.)
2. **`imp_max_p` stays 0 until Khadiv decision (e).** Enforcement is additionally blocked on the
   F2 cap re-derivation + F3 δ recalibration (§6c). Any experiment touching enforcement must be
   framed as *characterization* (offline reprocessing, counterfactual replay), never as a live
   `imp_max_p>0` run.
3. **NO VIC**: no policy-commanded stiffness, no `set_gains`, no kp/kd sweeps promoted as
   results (`impedance_one.py --kp_scale` exists for diagnostics only and is explicitly
   "VIC-in-disguise" for anything else). VIC is the next phase and requires explicit user +
   Khadiv sign-off.
4. **The superlinear excess-over-i_ref reward (`max(0, I/i_ref−1)²`) is FORBIDDEN while
   `imp_max_p=0`** — with no CaT counter-pressure it is an unbounded "hit as hard as possible"
   farm (execution plan §"Safety gate"; menu List B.6). It may only exist behind weight 0 until
   enforcement is live.
5. **Commits:** only named files (`git add <file> ...`), never `git add -A`; **no co-author
   trailer** (user preference — overrides any harness default); never commit
   `.env`/`.codex/`/`AGENTS.md`/`CLAUDE.md`.
6. **Never edit installed packages** (mjlab / rsl_rl / mujoco_warp) — runtime spec-wrapping and
   repo-side subclassing only.
7. **One GPU per run** (`--gres=gpu:1`, `--gpu-ids [0]`); don't gang-schedule.
8. Re-read files before editing (the tree is edited concurrently by the user and other agents).

---

## 6. OPEN QUESTIONS — where more research is needed (honest status)

- **(a) Khadiv decision (e) — what should Λ bound?** The options, with the A3 table as decision
  support: (i) the **ballistic** impulse — clean "impact-safe" semantics but *provably vacuous*
  (≤0.61× even on a rigid target at reachable speed; 0.19–0.39× on the real nail); (ii) the
  **windowed-press** reaction (the shipped quantity) — binds only under the mismatched 27.3 ms-cap
  / 50 ms-window pairing (1.184× vs 0.647× formula-consistent), and reduces to
  avg-torque × window, inviting "why not just bound torque?"; (iii) **split** constraints
  (ballistic + torque/press). This is a supervisor decision, not more seeds. Fold Phase-0 + any
  new characterization into the decision memo (execution plan Phase 3).
- **(b) Can the constraint be made to bind LEGITIMATELY** — i.e. a *healthy, successful* strike
  approaching the fixed hardware cap — by making the task harder (heavier/stiffer/spring-loaded
  nail) **without touching the caps**? Unknown. The frontier probe exists (`nail_sweep.py`:
  mass_scale/frictionloss/speed; runtime spring stiffness would need a small spec-wrapper
  addition). Qualifying bar (menu A.3): reference success + worst Λ/cap ∈ [0.7,1.2] +
  impact-only fraction ≥70% + solref sensitivity <20%. Expectation: most high-resistance points
  go press-dominated or infeasible; a moderate spring target is the most promising. Any modified
  target re-derives `I_REF_DELIVERED` and needs Khadiv sign-off to become the thesis benchmark.
- **(c) Enforcement-day calibration (F2 + F3)** — MUST land before any `imp_max_p>0`: re-derive
  all six caps at the shipped 25-substep window (or change the window to match the cap basis —
  that's part of (a)), bind `(window_substeps, physics_dt)` metadata to the caps with an assert,
  and replace the outcome-dependent ~3-reads-per-violation δ with one event-level pulse (or
  recalibrate per-read survival). Neither is coded; F2's planned hard-assert is also not yet in
  the tree — today the only guards are comments plus the placeholder-cap guard (`hook.py:95-101`).
- **(d) F7 — the impact_progress first-contact latch** (`rewards.py:210`): behavior-changing
  (the current gate statistically underpays clean elastic strikes and thereby selects for
  press-through — the wrong direction given the press finding). Could genuinely shift the
  canonical fixed-impedance result. Needs the CPU contact-phase sweep (≥99% payout recall,
  exactly one payout/event across all 10 substep offsets) then a 5-seed Vega A/B per §4.4.
- **(e) Does a TRAINED policy ever incidentally press long enough to bind in exploration?** The
  g-campaign saw sampled worst Λ/cap 0.50–0.80 with one seed at 1.004; but training-time
  visibility is missing — the C1 instrumentation (readers `worst_ratio`, `over_cap_frac`,
  `contact_window_len`, `press_dom_frac`; ~15 lines across `impulse_bound.py`,
  `mdp/__init__.py`, `env_cfgs.py` per execution-plan Phase 1) is **not yet wired**. Without it,
  no "does it bind in exploration" claim is possible.
- **(f) delta_pos_scale at the hardware cap:** 0.15 uses 81% of the 3.1415 rad/s budget
  (2.556 rad/s). Is there a small reachable gain at ~0.17–0.20 before max|q̇| crosses 3.1415 —
  and does the coarser strike placement cost success/accuracy in closed loop? The dps grid
  (0.15→0.30) is too coarse to answer; nobody has run the fine sweep or the closed-loop
  accuracy check. Note: changing the shipped action scale would invalidate i_ref and the
  reference calibration — characterize first, decide later.
- **(g) Armature model verification (Khadiv (f)):** the sim's dof_armature vs the real Z1 rotor
  inertia × gear² is unverified. A6 bounds the impact (vacuity survives ×30), so this is a
  defense-hardening item, not a blocker — but it needs a datasheet/measurement pass.
- **(h) VIC feasibility (the deferred chapter):** kp ∉ M(q), so *static* stiffness has no
  ballistic lever in MuJoCo (`ceiling.py` header; A6 note). Only DYNAMIC gain scheduling via
  `set_gains` (stiff windup → compliant release, exploiting stored strain energy / decoupling
  contact speed from steady-state joint velocity) might beat the hardware-velocity wall — and it
  is completely untested. This is the next thesis phase, gated on user + Khadiv.

---

## 7. RANKED EXPERIMENT SHORTLIST (updated for the B.2 hardware-ceiling verdict)

Supersedes the ranking in `2026-07-19_next_experiments_codex.md`: its #3 (effort-limit Vega
ablation, List B.2) is now **LOW-VALUE — do not run it**; the CPU physics already proves effort
is a trajectory-independent no-op on contact speed. The live experiments:

### CODEX-1 — delta_pos_scale fine sweep at the hardware-velocity cap  [CPU-first]
- **Hypothesis:** a small reachable v_touch gain (maybe 1.27 → ~1.4-1.5 m/s) exists between
  dps=0.15 (max|q̇|=2.556) and the 3.1415 rad/s wall; or 0.15 is already optimal.
- **Knobs:** edit the sweep grid at
  `docs/results/assets/2026-07-17_fixed_impedance_diag/probes/dps_sweep.py:74` to
  `(0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25)`. Baseline constant:
  `z1_constants.py:229` (do NOT edit it; the probe overrides at runtime).
- **Primary metric + rule:** max reachable dps* = largest scale with max|q̇| ≤ 3.1415 on the
  saturated descent; report v_touch(dps*) vs 1.267. If v_touch gain ≥ 10% → stage a closed-loop
  strike-accuracy check (scripted + rendered) before even proposing a Vega A/B; else close (f)
  as "0.15 is optimal, hardware-calibrated".
- **Expected:** small or no reachable gain (the q̇ curve was steep: 0.15→0.30 jumped 2.56→4.95).
- **Risk/farm surface:** none on CPU. Promoting a changed dps to the shipped config invalidates
  `I_REF_DELIVERED` + the reference calibration → **user sign-off required to ship**; no Khadiv
  gate for characterization.

### CODEX-2 — harder-target frontier: make a healthy strike approach the cap  [CPU-first → Vega if a point qualifies]
- **Hypothesis:** some nail physics (mass/friction/spring) creates a feasible band where a
  successful strike reads worst Λ/cap ∈ [0.7, 1.2] with impact-dominated (not press) character —
  the only legitimate path to a binding constraint, since the caps are untouchable.
- **Knobs:** `probes/nail_sweep.py` (`mass_scale`, `friction`, `speed`, `hold` — asset wrapped
  at runtime): grid mass_scale {1,3,10,30} × frictionloss {30,45,60,90} × speed {0.5,1,2,4}. If
  no point qualifies, add a runtime return-spring stiffness {250,500,1000} N/m via the same
  spec-wrapper pattern (the XML at `nail_block_scene.xml` is loaded by
  `src/tasks/hammer/nail_block.py`; do not edit the asset file).
- **Primary metric + rule (pre-register):** a config with (reference strike succeeds) ∧
  (worst Λ/cap ∈ [0.7,1.2] under the shipped caps) ∧ (impact-only fraction ≥ 70%, via the
  `impedance_one.py:104`-style impact-lobe split) ∧ (solref×2 sensitivity < 20%) ⇒ that ONE
  target earns a 5-seed Vega training run; else report "no legitimately binding target exists on
  this platform" (itself a strong, defensible vacuity result).
- **Expected:** most points press-dominated or infeasible; a moderate spring is the best bet.
- **Risk/farm surface:** friction selects for pressing; a spring rewards oscillation/tapping —
  check the per-event delivered accounting, not episode totals. **Khadiv sign-off required** to
  adopt any modified target as the thesis benchmark; caps and `imp_max_p` untouched throughout.

### CODEX-3 — F7 first-contact latch: CPU phase-sweep → Vega A/B  [CPU-first → Vega, 5 seeds/arm]
- **Hypothesis:** latching pre-contact v_axial at the substep rising edge and paying exactly
  once per event removes the phase-random underpayment and stops selecting for press-through ⇒
  cleaner strikes (shorter contact window) at equal success; delivered likely stays ~0.87×
  (physics ceiling).
- **Knobs:** `rewards.py:210` (`compute_first_contact`) → mirror the `ImitationPriorTerm`
  latch idiom (`rewards.py:360-373`: OR `current_contact_time`/`last_contact_time`) + a
  per-event once-latch; keep weight 8, `v_expected=1.0`, `eps=5e-4`
  (`hammer_env_cfg.py:199-209`). CPU gate first: scripted contact-phase sweep across all 10
  substep offsets — require ≥99% payout recall and exactly one payout/event (menu List B.1).
- **Primary metric + rule (pre-register per §4.4):** sampled delivered_mean/i_ref and sampled
  contact-window length, latched vs current, 5 seeds, MWU p<0.05 + ≥10% practical threshold on
  either ⇒ adopt the latch (and annotate the canonical result); else keep current and record
  "F7 is measurement noise, not a result-shifter".
- **Risk/farm surface:** removing the once-latch enables chatter farming — keep it. This is the
  ONE change that could shift the banked fixed-impedance numbers; run it before the result is
  frozen in the thesis text. No Khadiv gate; user sign-off to change the shipped reward.

### CODEX-4 — C1 training-time Λ instrumentation  [code + 1 small Vega run]
- **Hypothesis:** exploration occasionally approaches/crosses cap (g-campaign sampled 1.004 on
  one seed); live training metrics will show whether the constraint would EVER act if enabled.
- **Knobs:** per execution-plan Phase 1: add full-step readers `worst_ratio`, `over_cap_frac`,
  `contact_window_len`, `press_dom_frac` to `impulse_bound.py` (reading
  `_episode_peak_perjoint` / the sensor's `last_contact_time`), export in
  `src/tasks/hammer/mdp/__init__.py`, register 4 `MetricsTermCfg(reduce="last")` in the
  `if cat_impulse:` block of `env_cfgs.py` (~:315, AFTER `cat_soft` — the insertion-order guard
  at `env_cfgs.py:340-344` shows the pattern). Unit test + `validate_rewards.py` re-run.
  True p95/max stays in `eval_impulse.py` (env-mean reduction limits training-time metrics).
- **Primary metric + rule:** on a 3-seed baseline rerun, TB `Episode_Metrics/over_cap_frac` > 0
  at any training stage ⇒ open question (e) answered "yes, exploration binds" (feeds the Khadiv
  memo); flat 0 ⇒ "enforcement would be a no-op even on exploration tails at the shipped pairing".
- **Risk:** ~none (log-only). Prereq for any future enforcement claim. No gates.

### CODEX-5 — matched window/cap + ballistic-gate characterization  [CPU-only]
- **Hypothesis:** no window choice makes a *ballistic* quantity bind (shorter windows drop press
  contribution but stay far under matched caps); the 50 ms quantity is a reaction/press
  constraint. This pins what Λ *means* for the Khadiv (e) memo.
- **Knobs:** offline-reprocess the SAME substep traces (probe outputs under
  `.../2026-07-17_fixed_impedance_diag/`, plus fresh `impedance_one.py`/`sensitivity_run.py`
  runs) at matched pairs: 9 substeps/18 ms with caps `[1.08,2.16,1.08,…]`; 14/28 ms
  `[1.68,3.36,…]`; 22/44 ms `[2.64,5.28,…]`; 25/50 ms `[3.00,6.00,…]`; plus the impact-lobe
  gate (first contact → v_axial ≤ 0, as in `impedance_one.py`). Shipped values for reference:
  window `env_cfgs.py:258`, caps `env_cfgs.py:35`.
- **Primary metric + rule:** a candidate ballistic quantity qualifies only if <10% change under
  solref×2 ∧ agrees with m_eff·Δv within 15% ∧ excludes >90% of a sustained press ∧ survives
  hold-then-spike/flicker. Report each Λ only against its matched-duration cap.
- **Risk:** none (offline). **Khadiv sign-off required to SHIP any semantics change**; free to
  characterize. This + CODEX-4 + the A3 table = the complete decision-(e) package.

### CODEX-6 — empirical bindability certificate (black-box max over action sequences)  [CPU]
- **Hypothesis:** no reachable fixed-impedance action sequence produces a successful, clean
  ballistic strike at Λ/cap ≥ 1; any cap-crossing found will be press/dwell/geometry exploitation.
- **Knobs:** keep dps=0.15, 50 Hz control, effort 30/60, fixed gains, shipped caps. Warm-start
  ~10k 12–15-step action sequences around the scripted reference (perturbation amplitudes
  {0.05,0.15,0.30}), refine top ~200 with random shooting/CMA-ES; objective = diagnostic
  max_j Λ_j/cap_j (NOT reward). Base it on `frontier_one.py`/`run_one.py` harness patterns.
- **Primary metric + rule:** Pareto over (success depth ≥30 mm, Λ/cap, delivered, contact speed,
  contact duration, loaded-joint identity); render the top candidates
  (`scripts/render_reference.py` idiom). Any crossing that is impact-dominated (j2/j3/j4-loaded,
  short window) would FALSIFY the vacuity claim — that's the point of the certificate.
- **Risk:** optimizer may exploit head-edge/unmonitored contacts (the sensor sees only
  `hammer_head_0`→nail, `env_cfgs.py:108`); classify those separately (they are F8 evidence,
  not bindability). No gates.

### CODEX-7 — enforcement bring-up code (F2+F3), behind a hard OFF switch  [code-only; Khadiv-gated to ENABLE]
- **Hypothesis:** none — this is plumbing so that IF Khadiv picks a bindable pairing, enforcement
  is correct on day one instead of miscalibrated.
- **Knobs:** (F2) re-derive the six caps at the 25-substep window via
  `derive_impulse_thresholds.py`; bind `(window_substeps, physics_dt)` metadata to the cap
  vector and assert the match in `CatSoftHook._validate_params` (`hook.py:43`) whenever
  `imp_max_p>0` — this converts the current comment-guard into a code-guard that makes flipping
  `imp_max_p` alone IMPOSSIBLE until the calibration lands. (F3) event-level single δ pulse per
  violating window (or per-event survival recalibration) in `hook.py:192-237` +
  `impulse_bound.py:98-104`.
- **Primary metric + rule:** unit tests only (a bad pairing raises; one violation ⇒ one δ event
  regardless of completion timing). `imp_max_p` stays 0 in every registered config.
- **Risk:** touching live constraint code — full pytest + validate_rewards A–M gate, and an
  af1-style 3-seed behavior-neutrality rerun before it's trusted. **Enabling enforcement remains
  Khadiv-gated (decision (e)); this item only removes the technical debt blocking it.**

**Suggested order:** CODEX-4 (cheap, unblocks (e)) → CODEX-1 (hours, closes (f)) → CODEX-5 +
CODEX-2 in parallel (the Khadiv-(e) decision package + the bind-legitimately search) → CODEX-3
(the one result-shifting reward fix) → CODEX-6 (certificate) → CODEX-7 (only if enforcement is
on the horizon).

---

## Appendix — quick reference

- Conda (local gates/probes): `/Users/nikerane/miniconda3/envs/unitree_mjlab/bin/python`
  (mjlab 1.4.0, mujoco 3.8.1, mujoco_warp 3.8.1). Always `PYTHONPATH=.` from the repo root.
- Vega python: `~/repos/unitree_rl_mjlab/.venv/bin/python` (no pip/pytest).
- Key numbers to keep straight: i_ref delivered = **0.6094 N·s** (`env_cfgs.py:45`); caps
  `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64]` N·m·s (`env_cfgs.py:35`); window 25 substeps = 50 ms
  at physics dt 2 ms; control 50 Hz; real Z1 joint-velocity limit **3.1415 rad/s**; effort
  30/60 N·m; delta_pos_scale 0.15; nail: 7 g, frictionloss 30 N, gravcomp, success 0.030 m,
  stop 0.032 m; fixed-impedance ceiling ≈ 1.4 m/s contact speed ≈ 0.87× i_ref delivered.
- Rendering a strike for visual inspection (headless-safe on macOS):
  `python scripts/render_reference.py --distance 0.85 --elevation -25` → PNGs + mp4 under
  `/tmp/hammer_ref/`; policy rollouts via `scripts/render_policy.py`.
