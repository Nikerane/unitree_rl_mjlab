# Deep Dive v2 — Reward Design for RL Hammering & Repetitive Impact Tasks

**Pipeline:** ARS deep-research (`full` mode) — external literature sweep (not local-corpus).
**Date compiled:** 2026-05-29
**Author model:** Claude Opus 4.8, synthesising 6 parallel literature-search agents + 18 self-verified citations.
**Scope of grounding:** read against the *actual* implementation on branch `hammer-z1` (`src/tasks/hammer/**`), not just the prior notes.
**Supersedes nothing — extends:** `docs/research/reward-design/DEEP_RESEARCH_REPORT.md` and the 6 files it indexes. Treat the prior corpus as the baseline this report is built to exceed.

> **Evidence legend** (used throughout):
> `[E1]` peer-reviewed venue, citation **independently verified** this session against arXiv/DOI.
> `[E2]` preprint / arXiv-only, verified to exist, venue unconfirmed or none.
> `[E2*]` reported by a literature-search subagent, canonical/high-confidence, **not re-fetched** in the final integrity pass — treat author lists with mild caution.
> `[E3]` inherited from the prior project corpus (`REWARD_LITERATURE.md`) — carries that corpus's verification *and its flags* (one prior fabrication, three preprint-venue ambiguities).
> `[ANALOGY]` the source is **not** about hammering/impact; an idea is transferred by analogy (analogy stated explicitly).
> `[PROPOSAL]` my own design proposal with **no** direct citation — speculative, flagged as such.

---

## 1. Executive Summary

**The 13-point strategy (ranked, most load-bearing first).**

1. **Reward geometric outcomes, never simulated contact force.** Nail-depth progress transfers; instantaneous MuJoCo contact force at the impact instant is a solver artefact that oscillates with `solref` stiffness, not physics (van Steen et al. 2024 `[E3]`; van Steen et al. 2024 QP-RS `[E1]`; Ma et al. 2024 DrEureka `[E3]`). The implemented `nail_depth_delta` is the right primary signal; keep it.
2. **Make the impact lever the only thing you *can* control with this action space: end-effector momentum along the nail axis.** The arm is **position-only DifferentialIK** (no torque/impedance/orientation command). For a stiff position-controlled arm, impact behaves as a *composite rigid body* (Wang, Dehio & Kheddar 2022 `[E1]`); the controllable impact quantity is `m_eff(q)·v_n`, not commanded force. Reward **pre-impact axial velocity** and/or **directional manipulability** (Ti et al. 2024 `[E2]`, who do exactly this for *real nail hammering*).
3. **Decompose the task as an event-driven reward machine, not a time-clock.** Hammering rhythm is triggered by contact events, not a fixed period; a finite-state reward machine (Toro Icarte et al. 2018/2022 `[E1]`) over `{APPROACH→WINDUP→STRIKE→REBOUND→REPOSITION}` is the principled structure. A periodic phase-clock (Siekmann et al. 2021 `[E1]`) is a *fallback*, valid only if you adopt an adaptive-frequency oscillator (CPG-RL, Bellegarda & Ijspeert 2022 `[E1]`).
4. **Gate every event bonus on *useful* progress.** An ungated `impact_velocity_bonus × first_contact` is directly hackable by scraping/tapping. Gate it on `first_contact ∧ (depth_delta > ε)` so velocity is only paid when it actually drives the nail. (Synthesised from Skalse et al. 2022 `[E1]` + Pan et al. 2022 `[E1]`.)
5. **Re-balance the dense/sparse ratio — it is currently inverted.** Live `nail_depth_delta=+2000` yields ≈132 cumulative vs `completion=+100`; the spec *intended* completion ≈2.5× the dense total. A single 66 mm strike pays +132 in one step — a larger spike than completion. Drop to ≈500–800 or add value/return normalisation (§4.7, §11).
6. **Add domain randomisation — there is currently none.** Reset ranges are `(0,0)`; nail friction/damping fixed. This is the single biggest sim-to-real risk and the cheapest high-value change. Randomise nail `frictionloss`/`damping`/position, arm damping/mass, and action latency (Peng et al. 2018-style dynamics randomisation `[ANALOGY]`; OpenAI ADR-style `[ANALOGY]`).
7. **Use an asymmetric (privileged) critic.** The critic currently sees the actor's terms minus corruption — no true privileged information. Give the critic-only group: GT contact force/impulse, axial impact velocity, `m_eff`, reward-machine state, and time-to-contact. Actor keeps only deployable signals (DeXtreme `[E1]`, RMA-style `[ANALOGY]`).
8. **Treat safety as constraints, not just soft penalties, during the strike.** A large task-reward gradient at impact will overwhelm a soft `−10·joint_pos_limits` (Kim et al. 2023 `[E3]`). Use a Lagrangian/CPO-style cost on peak joint torque, joint velocity, and impulse (Achiam et al. 2017 `[E1]`), or at minimum an episode-terminating safety envelope.
9. **Emulate compliance you cannot command.** Position-only control cannot do impedance control; you get it via **admittance wrapping** (read wrist force → compliant position correction) (Beltran-Hernandez et al. 2020 `[E1]`; Abu-Dakka & Saveriano 2020 `[E2*]`) or, more cheaply, via **step-size/action-rate modulation near contact** (small steps = soft effective stiffness).
10. **Add smoothness terms designed for sim-to-real and anti-chatter.** `action_rate=−0.01` alone is weak. Add CAPS temporal+spatial smoothness (Mysore et al. 2021 `[E1]`) and/or a Lipschitz constraint (Kobayashi 2022 `[E2*]`); raise the weight specifically in the REBOUND/REPOSITION states to kill double-hit chatter.
11. **Model actuator stress and heat as a per-episode budget.** Repetitive impact is a worst case for `I²t` thermal load (Qian et al. 2026 `[E2]`) and gear shock (Hwangbo et al. 2019 actuator nets `[E2*]`). Add a thermal accumulator state + budget penalty; this also makes the policy *choose* rest beats between strikes.
12. **Reward consistency across strikes, not one lucky hit.** A sparse-completion design rewards a single fluke. Add a strike-to-strike variance penalty and require ≥N consistent strikes under randomisation (synthesised from D'Ambrosio et al. 2023 `[E3]` multi-component philosophy + reward-machine loops).
13. **Anneal dense→sparse to prevent shaping-farming.** Once the policy reliably drives the nail, decay shaping weights toward the sparse completion signal (Luo et al. 2022 Dense2Sparse `[E3]`; Reward Training Wheels `[E3]`).

**What to implement first (one paragraph).** Keep the implemented 6-term baseline, but (a) **fix the dense/sparse balance** (lower `nail_depth_delta` to ~600 *or* turn on PPO return normalisation), (b) **add domain randomisation** to the two reset events and the nail joint, (c) **wire the already-present `ContactSensor` into a single gated event reward** — `impact_progress = ‖v_axial‖ · first_contact · 𝟙[depth_delta>ε]` — and (d) **add a privileged critic group**. These four changes are low-risk, code-local, and address the three things the prior notes never confronted (balance, randomisation, privileged critic). Everything phase/rhythm/reward-machine related is Stage 2+, gated on observing that single strikes don't solve the task (`test_single_strike.py`, Q1).

---

## 2. Existing-Notes Audit

### 2.1 What the prior corpus already covers well (do not redo)

- **Progress-not-position framing** for nail depth (`nail_depth_delta`), with a correct stateful `ManagerTermBase` + monotone max-tracking + `clamp_min(0)` (anti-bounce). Validated by `validate_rewards.py` (8 phases). Grounded in Wu et al. 2021 DREM `[E3]`. **This is genuinely good and survives v2.**
- **PBRS theory awareness** (Ng et al. 1999 `[E3]`) and the hovering critique of always-on Gaussians.
- **The "trust the code, not the spec" discipline** and the augment-not-replace minimum-viable strategy. Sound engineering hygiene; v2 respects it.
- **A real citation-integrity process** — the Opus 4.7 audit caught a fabricated author list (Meta-World, H3) and flagged 3 preprint-venue claims (M1). This is exactly the right posture and I have continued it (every load-bearing v2 citation is tier-tagged).
- **The settle dead-zone insight** (`_SETTLE_OFFSET = 0.004`): absorbing MuJoCo's gravity-settling drift so it isn't mistaken for progress. This is a *correct, real reward-hacking defence* already in the code.

### 2.2 What is missing, shallow, or not actionable (the gap this report fills)

| # | Gap in prior notes | Severity | Where v2 addresses it |
|---|---|---|---|
| G1 | **No external literature search** — the prior "deep research report" is explicitly a *local-corpus* synthesis of ~17 papers. Whole requested subfields are absent. | High | This entire report (29 net-new sources). |
| G2 | **Repetitive hammering is undesigned.** Rhythm, recovery, repositioning, anti-chatter, long-horizon completion are flagged as open (Q2) but have *no reward architecture*. | High | §3, §5 (Proposal C), §7. |
| G3 | **Action-space mismatch never confronted.** Notes recommend torque penalties, an orientation gate, and impedance ideas — but the env is **position-only DiffIK** (`orientation_weight=0.0`). Several recommendations are not actionable as written. | High | §2.3, §4, §8. |
| G4 | **No domain randomisation.** Reset ranges `(0,0)`; nail physics fixed. Sim-to-real is "deferred" but transfer is impossible without it. | High | §1.6, §8.2, §9. |
| G5 | **Safety = soft penalties only.** No constrained-RL design, no impulse/jerk/torque budget, no thermal/shock accumulation, despite citing Kim et al. 2023. | High | §6 (terms), §8. |
| G6 | **No reward-hacking taxonomy or literature.** "Hovering" and "slow press" are mentioned; the formal reward-hacking field is absent. | Medium | §11, with Skalse/Pan/Amodei `[E1/E2]`. |
| G7 | **Privileged critic underused.** Critic = actor − corruption. No contact force / impact velocity / phase as critic-only. | Medium | §1.7, §4 (column), §6. |
| G8 | **Weight justification is hand-wavy** (the audit itself found §8 arithmetic doesn't balance) and there is **no normalisation strategy**. | Medium | §4.7, §11. |
| G9 | **No phase/event/FSM representation** at all (the obs has no phase variable). | Medium | §3, §5C, §7. |
| G10 | **Discount-horizon vs episode-length mismatch** never analysed (`γ=0.99` ⇒ ~100-step horizon vs 1000-step episodes). | Medium | §2.3, §7.6. |
| G11 | **No ablation matrix / metrics / logging framework** beyond "watch the curves." | Medium | §9, §10. |
| G12 | **"Impact = force" intuition** persists in the MATRIX (`contact_force_magnitude` term). For a stiff position-controlled arm the controllable quantity is **momentum**, not force. | Medium | §2.3, §4 (C-terms), §6. |

### 2.3 Assumptions in the prior notes that must be challenged

1. **"Add a `torque_sum_sq_penalty` / `orientation_alignment` quat-gate when needed."**
   *Challenge:* The policy commands **EE position deltas**, not torques or orientation. A torque penalty is still computable on the *resulting* `applied_torque` (and is worth it as a soft penalty + critic signal), but the policy can only reduce it *indirectly* (move slower/smoother) — so its gradient is weak and indirect. The Meta-World orientation gate is **largely moot**: with `orientation_weight=0.0` the hammer's orientation is whatever the nominal IK posture yields; the policy cannot rotate the head to fix alignment. If face-alignment actually matters, the fix is an **action-space change** (add an orientation DoF or a wrist joint to the IK frame), not a reward term. *Do not* add an orientation-gate reward and expect it to work on the current action space.

2. **"`nail_depth_delta=+2000` is the primary learning signal."**
   *Challenge:* True, but the weight inverts the intended dense/sparse hierarchy and creates a single-step +132 spike larger than the +100 completion bonus, raising value-target variance. Either re-balance to ~500–800 or enable value/return normalisation. (See §4.7.)

3. **"Single ContactSensor isn't consumed yet — it's deferred."**
   *Challenge:* It is **already fully wired** (`config/z1/env_cfgs.py:33`, fields `found/force`, `track_air_time=True`, `reduce="maxforce"`). The deferred event terms are *infrastructurally ready now*; the only reason to defer is learnability, not plumbing.

4. **"Q11: the completion bonus creates a `100/(1−γ)` value cliff."**
   *Challenge:* Overstated. The success **terminates the episode**, truncating the bootstrap — so the value target at the success step is just the immediate ≈+100 (plus same-step dense), *not* `100/(1−γ)≈10⁴`. The real effect is an `O(100)` discontinuity in the value *function* across the success boundary versus `O(1)` per-step dense — a ~100× *local* jump, smoothed by GAE (λ=0.95). Real, bounded, watch the value loss; don't panic.

5. **"Geometry transfers, force doesn't — so we're safe."**
   *Challenge:* Necessary but not sufficient. The geometric signal (`nail_depth`) depends on the nail's `frictionloss`/`damping`, which are **un-randomised and uncalibrated**. A policy tuned to one nail stiffness will mis-time strikes on a different real nail. Geometry-based reward + **domain randomisation of the contact parameters** is what transfers.

---

## 3. Task Decomposition

Hammering is a **cyclic, event-driven** skill. I model it as a reward machine (RM) — a finite-state automaton whose transitions are triggered by *observable events*, with a per-state reward and a per-state potential for shaping (Toro Icarte et al. 2018/2022 `[E1]`). Single-hit hammering is the special case where the loop executes once.

```
              ┌────────────────────────────────────────────────────────────┐
              │                      (episode reset)                          │
              ▼                                                               │
   ┌──────────────────┐  head within r_a of nail axis   ┌──────────────────┐ │
   │  S0 APPROACH      │ ───────────────────────────────▶│  S1 ALIGN/WINDUP │ │
   │  (gross reach)    │                                  │  (axis align +   │ │
   └──────────────────┘ ◀───────────── lost alignment ───│   retract/raise) │ │
              ▲                                            └──────────────────┘ │
              │                                                     │ air_time≥t_min, v_axial building
              │                                                     ▼            │
              │                                            ┌──────────────────┐ │
              │                                            │  S2 STRIKE       │ │
              │                                            │  (accelerate     │ │
              │                                            │   down the axis) │ │
              │                                            └──────────────────┘ │
              │                                                     │ first_contact (nail)
              │                                                     ▼            │
   ┌──────────────────┐    depth ≥ success_threshold      ┌──────────────────┐ │
   │  S5 SUCCESS/TERM │ ◀────────────────────────────────│  S3 IMPACT       │ │
   └──────────────────┘                                   │  (impulse xfer,  │ │
                                                          │   Δdepth)        │ │
                          depth < threshold               └──────────────────┘ │
                                ┌──────────────────┐               │ contact released
                                │  S4 REBOUND /     │◀──────────────┘            │
                                │     REPOSITION    │── recovered, realign ──────┘
                                └──────────────────┘
```

**Phase table** (signals are what the *env* can produce today or with the wired ContactSensor):

| Phase | Trigger in | Key signals available | Goal of the phase | Single-hit? |
|---|---|---|---|---|
| **S-tool. Tool grasp / pose stabilise** | episode start | (N/A — hammer is welded to the EE in this env; `--no-weld` only in viewer) | none for now | n/a |
| **S0. Approach** | reset | `head_pos`, `nail_top_pos`, dist | bring head into a strike cone above the nail | yes |
| **S1. Align / Wind-up (backswing)** | near nail | `head_pos`, `nail_top_pos`, `track_air_time` | raise head for clearance; align axis | yes |
| **S2. Impact preparation (strike accel)** | wound-up | finite-diff `v_axial`, `m_eff(q)` | maximise axial momentum toward nail | yes |
| **S3. Contact event** | `compute_first_contact()` | `found`, `last_air_time`, `v_axial` | register a valid strike | yes |
| **S3b. Impulse transfer** | during contact | `nail_depth_delta`, (GT impulse → critic) | convert momentum to nail travel | yes |
| **S4. Nail/object displacement** | post-contact | `nail_depth`, `nail_depth_delta` | accumulate depth toward goal | yes |
| **S4b. Rebound / recovery** | contact released | `head_vel`, `action_rate`, posture | absorb rebound, re-stabilise without chatter | **multi only** |
| **S5. Repeated-strike repositioning** | recovered | `head_pos`, air-time | return to wind-up; maintain alignment over strikes | **multi only** |
| **S6. Termination / success** | `depth ≥ 0.07` | termination term | end episode (already implemented) | yes |

The current code collapses S0–S4 into always-on dense terms and never represents S4b/S5 — which is precisely why **repetition, recovery, and anti-chatter are unaddressed**.

---

## 4. Reward Taxonomy

Notation: `B` = batch (`num_envs`); `n̂` = unit nail-strike axis (≈ world −Z for the slide joint); `J(q)`,`M(q)` = arm Jacobian and joint-space inertia at the head site (from `mjData`); `v_axial = n̂·ẋ_head`; `Φ` = a potential function; `dt = env.step_dt = 0.02 s`; `first_contact = sensor.compute_first_contact(dt)`.

**How to read the table:** *Dense* = every step; *Event* = fires at a transition; *Sparse* = once. *A/C* = Actor-observable / Critic-only. *Hack* = primary reward-hacking risk. *S2R* = sim-to-real concern. Weights are starting points for the **re-balanced** scale of §4.7 (assume `nail_depth_delta≈600`, not 2000).

### 4.1 Approach / alignment

| Term | Math form | Sim signals | Type | A/C | Hack risk | S2R | Weight |
|---|---|---|---|---|---|---|---|
| `approach` (current) | `exp(−‖p_h−p_n‖²/σ²)`, σ=0.08 | head/nail site pos | Dense | A | hovering at high-reward shell | robust (geometry) | +0.1 (keep low) |
| `approach_pbrs` `[E3]`(Ng99) | `γΦ(s′)−Φ(s)`, `Φ=−‖p_h−p_n‖` | same | Dense | A | **none** (policy-invariant) | robust | +1.0·(potential scale) |
| `axis_align` `[PROPOSAL]` | `exp(−d_perp²/σ_⊥²)`, `d_perp` = head distance from nail *axis line* | head pos, nail axis | Dense | A | approach off-axis then slide in | robust | +0.2 |
| `dir_manip` (Ti et al. 2024 `[E2]`,DIRECT) | `w(q,n̂)=n̂ᵀ J M⁻¹ Jᵀ n̂` | `mjData` J, M | Dense | A or C | could chase manipulability not nail | robust (kinematic) | +0.05–0.2 |

- **`approach_pbrs`** is the principled replacement for the always-on Gaussian: as a potential difference it provably cannot create a hovering optimum (Ng et al. 1999 `[E3]`). Implement `Φ(s) = −‖p_h − p_n‖`; reward `= γ·Φ(s′) − Φ(s)`. Net episode contribution telescopes to `γΦ(terminal) − Φ(start)`, so it cannot be farmed.
- **`dir_manip`** is the one genuinely new *approach-phase* idea backed by a **real hammering paper**: Ti et al. (2024 `[E2]`) plan the pre-grasp posture and trajectory to maximise directional velocity manipulability `w(q,n̂)` along the strike axis, validated on a 7-DoF arm driving a nail. It is computable purely from kinematics (transfers well), is non-zero throughout the swing (dense gradient), and *rewards configurations from which a fast strike is even possible*. **Caveat (my analysis):** high `w` ⇒ high reachable EE speed but **low effective inertia** `m_eff = 1/w`. Momentum is `m_eff·v_n`. So `dir_manip` shapes *speed capacity*, while the impact reward (§4.3) should capture *delivered momentum*. Treat the speed/inertia trade-off as an ablation knob, not a solved question.

### 4.2 Pre-impact / wind-up

| Term | Math form | Sim signals | Type | A/C | Hack risk | S2R | Weight |
|---|---|---|---|---|---|---|---|
| `air_time` (Rudin22 `[E3]`,ANALOGY) | `(t_air−t_min).clamp(0,t_max−t_min)·first_contact` | `last_air_time`, first_contact | Event | A | hover to farm air-time (capped) | robust | +1–3 |
| `windup_potential` `[PROPOSAL]` | `Φ=clearance_height`; PBRS | head z above nail | Dense | A | none (PBRS) | robust | scale |

- `air_time` is the locomotion `feet_air_time` analogy (Rudin et al. 2022 `[E3]`): reward the *retraction* that precedes each strike. The `t_max` cap is the anti-hover guard; set `t_min`/`t_max` from `verify_reward_setup.py` percentiles (Q5).

### 4.3 Contact / impact (the part the prior notes get conceptually wrong)

| Term | Math form | Sim signals | Type | A/C | Hack risk | S2R | Weight |
|---|---|---|---|---|---|---|---|
| `impact_speed` (ARMADA `[E3]`,Kim25) | `‖v_axial‖·first_contact` | finite-diff head pos, first_contact | Event | A | **scrape/tap to trigger contact** | velocity OK (3.1% gap, van Steen `[E3]`) | gate it! |
| **`impact_progress`** `[PROPOSAL]` | `v_axial·first_contact·𝟙[Δdepth>ε]` | + `nail_depth_delta` | Event | A | low (double-gated) | robust | +5–10 |
| `impact_momentum` `[PROPOSAL]`(Wang22 `[E1]`) | `m_eff(q)·v_axial·first_contact` | J,M,vel,first_contact | Event | A | flailing (clip) | robust (kinematic) | tune |
| `useful_impulse_clip` `[PROPOSAL]` | `min(m_eff·v_axial, J_safe)·first_contact·𝟙[Δdepth>0]` | + impulse cap | Event | A | low | robust | tune |
| `contact_force` ❌ | `‖F_contact‖·first_contact` | sensor force | Event | A | **exploits solver artefact** | **fragile — avoid** | — |

- **The conceptual fix:** the MATRIX's `contact_force_magnitude` term should be **deleted from consideration**. For a stiff position-controlled arm, the strike behaves as a composite rigid body (Wang, Dehio & Kheddar 2022 `[E1]`); the *controllable, transferable* quantity is **axial momentum** `m_eff·v_axial`, and the *reliable measured* outcome is **Δdepth over a post-impact window**, not the instantaneous force (which van Steen et al. 2024 `[E3]/[E1]` show is dominated by the compliant-contact stiffness, ~oscillatory). 
- **The hacking fix:** `impact_speed` alone (ARMADA's signal `[E3]`) is hackable — the policy can earn it by triggering *any* nail contact (a glancing scrape) at speed without driving depth. Double-gate it: `first_contact ∧ Δdepth>ε`. This is the single most important new reward-integrity change for the event layer.

### 4.4 Outcome / progress

| Term | Math form | Sim signals | Type | A/C | Hack risk | S2R | Weight |
|---|---|---|---|---|---|---|---|
| `nail_depth_delta` (current; Wu21 `[E3]`) | `max(0, depth − max_depth)` | nail joint pos | Dense | A | tiny taps (mitigated by max-track) | robust | **~600** (down from 2000) |
| `nail_driven` Gaussian (current) | `exp(−(g−depth)²/σ²)`, σ=0.03 | nail joint pos | Dense | A | hold-at-goal (low wt OK) | robust | +1–2 |
| `progress_per_impulse` `[PROPOSAL]` | `Δdepth / (m_eff·v_axial + δ)` at contact | depth + momentum | Event | A or C | reward tiny-but-efficient taps | robust | small + |
| `completion` (current) | `𝟙[depth ≥ 0.07]` | nail joint pos | Sparse | A | none (terminates) | robust | +100 |

- **`progress_per_impulse`** (efficiency) is a novel, *desirable-direction* term: depth gained per unit momentum spent. It pushes the policy toward *well-aimed* strikes over brute force, which is exactly the actuator-friendly, transferable behaviour. Risk: it can favour timid efficient taps; pair with a minimum-depth-rate term or only activate it after a coarse policy exists (curriculum).

### 4.5 Recovery / repetition (entirely new vs prior notes)

| Term | Math form | Sim signals | Type | A/C | Hack risk | S2R | Weight |
|---|---|---|---|---|---|---|---|
| `rebound_damp` `[PROPOSAL]` | `−‖ẋ_head‖²·𝟙[state=REBOUND]` | head vel, RM state | Dense (gated) | A | freeze after strike (bound) | robust | −0.05 |
| `interstrike_interval` `[PROPOSAL]` | reward `t_air∈[τ_lo,τ_hi]`; penalise `t_air<τ_lo` | air-time | Event | A | none | robust | + / − |
| `strike_consistency` `[PROPOSAL]` | `−Var_k(Δdepth_k)` per episode | per-strike depths | Sparse | **C** preferred | game variance by 1 strike | robust | − |
| `rm_phase_potential` (Toro Icarte `[E1]`,ANALOGY) | `γΦ(u′)−Φ(u)`, `u`=RM state | RM state index | Dense | A | none (PBRS) | robust | scale |

### 4.6 Safety / smoothness / actuator

| Term | Math form | Sim signals | Type | A/C | Hack risk | S2R | Weight |
|---|---|---|---|---|---|---|---|
| `action_rate` (current; Rudin22 `[E3]`) | `Σ(a_t−a_{t−1})²` | actions | Dense | A | — | **improves S2R** | −0.01→−0.02 |
| `caps_smooth` (Mysore21 `[E1]`) | `−λ_T‖π(s_t)−π(s_{t+1})‖² −λ_S 𝔼‖π(s)−π(s̃)‖²`, `s̃∼N(s,σ)` | policy net | Dense | A | — | improves S2R | λ_T,λ_S small |
| `joint_vel` `[E3]` | `Σ|q̇|` or `Σq̇²` (arm only) | `joint_vel` | Dense | A | — | improves S2R | −0.005 |
| `joint_pos_limits` (current) | soft barrier | joint pos | Dense | A | — | safety | −10 |
| `torque_sq` (Rudin22 `[E3]`,ANALOGY) | `Σ τ²` on arm | `applied_torque` | Dense | A+**C** | weak gradient (indirect) | safety/heat | −1e−3 |
| `jerk` `[PROPOSAL]` | `Σ(a_t−2a_{t−1}+a_{t−2})²` | actions | Dense | A | — | anti-vibration | small − |
| `impulse_spike` (cost) `[PROPOSAL]` | `max(0, ‖F‖−F_cap)` as **CMDP cost** | sensor force | Event | **C/cost** | — | safety | Lagrangian |
| `thermal_budget` (Qian26 `[E2]`,ANALOGY) | `−λΣ_j max(0, T_j−T_max)`, `T_j(t+1)=αT_j+βτ_j²` | torque accumulator | Dense | A+C | — | hardware life | tune |

### 4.7 Weighting & normalisation strategy (replaces the unbalanced §8 table)

The prior spec's per-episode arithmetic admittedly "doesn't balance" (audit M4). Use a principled scheme instead:

1. **Target per-term episode-sum bands, not per-step magnitudes.** Log `Episode_Reward/<term>` (mjlab does this automatically). Aim for: dense shaping terms each contribute `O(1–10)` per successful episode; the **completion** sparse term `O(50–100)`; safety penalties each `O(−1…−20)`. This makes the sparse goal the dominant attractor — the intended hierarchy (Berducci et al. 2024 HPRS `[E3]`: *safety > target > comfort*).
2. **Fix the current inversion.** With `nail_depth_delta=2000`, cumulative dense ≈ `2000·(0.07−0.004)=132 > 100` completion, and a single strike spikes +132 in one step. Set it so full-drive cumulative ≈ 30–40 (i.e. weight ≈ 450–600) **or** turn on PPO **return/value normalisation** and keep 2000 if empirically it learns faster — but then *watch the value loss* (Q11).
3. **Express shaping as PBRS wherever possible** (`approach_pbrs`, `windup_potential`, `rm_phase_potential`). PBRS terms telescope and cannot be farmed (Ng et al. 1999 `[E3]`), so their weights don't need balancing against task reward — a major simplification.
4. **Normalise event bonuses by their natural scale.** `impact_progress`: divide `v_axial` by an expected impact speed (~1 m/s) so the bonus is `O(1)` per strike before weighting.
5. **Keep safety as constraints when the task gradient is large** (impact). A soft penalty that is `O(10)` will lose to a task spike that is `O(100)`; use a Lagrangian multiplier or an episode-terminating envelope for the hard limits (Kim et al. 2023 `[E3]`; Achiam et al. 2017 `[E1]`).

---

## 5. Concrete Reward Proposals

All three are written against the real mjlab API (`RewardTermCfg`, `ManagerTermBase`, the wired `hammer_nail_contact` sensor). Pseudocode uses the patterns already in `src/tasks/hammer/mdp/rewards.py`.

### 5.A — Minimal baseline (ship this first; ~4 edits to current code)

Goal: fix the three real defects (balance, randomisation, gated event) without architectural change.

```python
rewards = {
  "approach":        RewardTermCfg(hammer_approach_reward, weight=0.1,  params={...std=0.08...}),  # keep
  "nail_driven":     RewardTermCfg(nail_driven_reward,     weight=2.0,  params={...std=0.03...}),  # keep
  "nail_depth_delta":RewardTermCfg(NailDepthDeltaTerm,     weight=600.0,params={...}),             # 2000 -> 600
  "impact_progress": RewardTermCfg(ImpactProgressTerm,     weight=8.0,  params={"sensor_name":"hammer_nail_contact", ...}),  # NEW, gated
  "completion":      RewardTermCfg(completion_bonus,       weight=100.0,params={...}),             # keep
  "action_rate":     RewardTermCfg(action_rate_penalty,    weight=-0.02),                          # -0.01 -> -0.02
  "joint_pos_limits":RewardTermCfg(joint_pos_limits,       weight=-10.0,params={...}),             # keep
}
```

`ImpactProgressTerm` (stateful; finite-diff velocity + depth-delta gate — robust to lazy `site_vel_w` per Q10):

```python
class ImpactProgressTerm(ManagerTermBase):
    """v_axial * first_contact * 1[depth advanced this step]. Pays speed only when it drives the nail."""
    def __init__(self, cfg, env):
        super().__init__(env)
        self._prev = torch.zeros(env.num_envs, 3, device=env.device)
        self._prev_depth = torch.zeros(env.num_envs, device=env.device)
        self._init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    def reset(self, env_ids):
        self._init[env_ids] = False; self._prev_depth[env_ids] = 0.0
    def __call__(self, env, sensor_name, robot_cfg, nail_cfg, axis=(0,0,-1), eps=5e-4):
        robot, nail = env.scene[robot_cfg.name], env.scene[nail_cfg.name]
        sensor = env.scene[sensor_name]; dt = env.step_dt
        head = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
        vel  = torch.where(self._init[:,None], (head-self._prev)/dt, torch.zeros_like(head))
        self._prev = head.clone(); self._init.fill_(True)
        n = torch.tensor(axis, device=head.device, dtype=head.dtype)
        v_axial = (vel * n).sum(-1).clamp_min(0.0)                      # downward speed only
        depth = nail.data.joint_pos[:, nail_cfg.joint_ids].squeeze(1)
        advanced = (depth - self._prev_depth > eps).float()
        self._prev_depth = torch.maximum(self._prev_depth, depth)
        fc = sensor.compute_first_contact(dt=dt).any(-1).float()
        return v_axial * fc * advanced                                  # (B,)
```

Plus **domain randomisation** in the existing events (currently `(0,0)`):

```python
events["reset_nail_friction"] = EventTermCfg(func=randomize_joint_friction, mode="reset",
    params={"asset_cfg": SceneEntityCfg("nail_block", joint_names=("nail_slide",)),
            "frictionloss_range": (0.15, 0.45), "damping_range": (5.0, 11.0)})
events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)   # small posture jitter
# (optional) action latency: 1-2 step random delay on the IK target
```

And an **asymmetric critic** group (critic-only privileged terms):

```python
critic_terms = {**actor_terms,
    "gt_contact_force": ObservationTermCfg(contact_force_w, params={"sensor_name":"hammer_nail_contact"}),
    "axial_impact_vel": ObservationTermCfg(head_axial_speed),
    "eff_inertia":      ObservationTermCfg(effective_inertia_axis),   # n^T J M^-1 J^T n
}
observations["critic"] = ObservationGroupCfg(critic_terms, enable_corruption=False)
```

**Literature grounding (added 2026-05-29).** This skeleton is not speculative — it mirrors the *published, working* reward of the **Adroit `hammer` RL env** (Rajeswaran et al. 2018; see [`hammering_literature_notes.md`](hammering_literature_notes.md) §2.6): palm/hammer→nail distance shaping + a **lift/wind-up bonus** (+2 for lifting the hammer >0.04 m) + **heavily-weighted nail-depth progress** (×10) + a velocity penalty + **staged progressive insertion bonuses** (+25→+75) + a **sparse success bonus with a per-step penalty** (+10 / −0.1). Two cheap borrowings: **(a)** add a **`time_penalty` of −0.1/step** — Adroit's `−0.1` is exactly the missing term the prior notes flagged, and it pressures earlier completion without a separate tuning question; **(b)** keep the staged near-goal Gaussian (`nail_driven`) — it is Adroit's `+25→+75` idea. The difference to preserve: Adroit *presses* a nail with a 24-DoF hand (quasi-static); the Z1 *strikes* with a welded hammer, so the momentum-gated `impact_progress` replaces Adroit's grasp-and-press reward, and the impact terms (§4.3) carry the strike physics Adroit abstracts away.

### 5.B — Robust staged reward (Stage 2; PBRS-clean, safety-aware)

Adds principled shaping (PBRS), the manipulability lever, smoothness for S2R, and constraints for safety. Single-strike still; no rhythm yet.

```
r = r_task + r_shape + r_impact + r_safety
r_task    =  600·Δdepth_clamped            +  100·𝟙[success]          + 2·gauss_neargoal
r_shape   =  PBRS(Φ_approach=−‖p_h−p_n‖)    +  PBRS(Φ_windup=clearance)+ 0.1·w(q,n̂)
r_impact  =  8·[v_axial·first_contact·𝟙[Δdepth>ε]]                     + small·progress_per_impulse
r_safety  = −0.02·action_rate −0.005·Σq̇²  −10·jpos_limits −λ_T·CAPS_T −λ_S·CAPS_S
constraints (CMDP / terminating envelope): peak|τ| ≤ τ_max,  ‖F_impact‖ ≤ F_cap,  |q̇| ≤ q̇_max
```

Implementation notes: PBRS terms return `γ·Φ(s′) − Φ(s)` (compute `Φ(s)` from the *previous* step — store it stateful, reset to the initial potential). `w(q,n̂)` uses `mjData` `J`,`M`; if too costly per step, compute on the critic only or every k steps. CAPS spatial term needs a second forward pass on perturbed obs (cheap at this network size, `(256,128,64)`).

### 5.C — Advanced event/phase reward machine (repetitive hammering)

A reward machine (Toro Icarte et al. 2018/2022 `[E1]`) makes rhythm, recovery, and repetition *first-class*. State `u ∈ {APPROACH, WINDUP, STRIKE, IMPACT, REBOUND, REPOSITION}`; transitions on observable events; reward and potential per state.

```python
# Maintain RM state per env (a stateful ManagerTermBase or an mjlab "command"/extra).
EVENTS:
  e_near      = ‖p_h − p_n‖ < r_a
  e_woundup   = clearance ≥ h_min  AND  v_axial ≈ 0
  e_striking  = v_axial > v_lo (accelerating down)
  e_contact   = compute_first_contact()
  e_advanced  = Δdepth > ε
  e_released  = contact ended  AND  clearance rising
  e_success   = depth ≥ 0.07

TRANSITIONS (u, event) -> u':
  (APPROACH, e_near)->WINDUP ; (WINDUP, e_striking)->STRIKE ;
  (STRIKE, e_contact)->IMPACT ; (IMPACT, e_advanced & depth<thr)->REBOUND ;
  (REBOUND, e_released)->REPOSITION ; (REPOSITION, e_near)->WINDUP ;  # the loop
  (IMPACT, e_success)->SUCCESS(terminate)

PER-STATE REWARD  r(u, ·):
  APPROACH:   PBRS(−dist)                      # get there, no hovering
  WINDUP:     +air_time_bonus  + 0.1·w(q,n̂)    # retract & build manipulability
  STRIKE:     + small·v_axial                   # accelerate (bounded)
  IMPACT:     + 8·[v_axial·𝟙[e_advanced]] + 600·Δdepth   # the only big payouts
  REBOUND:    − 0.05·‖ẋ_head‖²  (keep LIGHT)    # damp chatter, but do NOT over-damp the rebound — §7.1
  REPOSITION: PBRS(−dist) + interstrike_interval band reward
  ALL:        − smoothness/safety (CAPS, action_rate, limits, thermal)
RM POTENTIAL (automated shaping, Toro Icarte): Φ(u) increases APPROACH<...<SUCCESS;
  add γΦ(u')−Φ(u) — policy-invariant progress pressure through the cycle.
EPISODE-LEVEL (critic-friendly): − Var_k(Δdepth_k)   # consistency across strikes
```

Why an RM and not a phase-clock: hammer cadence is *contact-triggered*, not time-periodic. A fixed sine/cosine clock (Siekmann et al. 2021 `[E1]`) assumes a known period; if you want a clock, make its frequency a learned oscillator parameter (CPG-RL, Bellegarda & Ijspeert 2022 `[E1]`) so the policy sets its own rhythm. The RM avoids committing to a period at all and gives you free, policy-invariant shaping via the state potential.

**Proof-of-concept (added 2026-05-29).** This is not hypothetical: **Robot Drummer** (Shahid et al. 2025; [`notes`](hammering_literature_notes.md) §2.7C) shows PPO *can* learn precisely-timed *repetitive* impact by exactly this device — it reformulates the task as a **"Rhythmic Contact Chain"** of timed contact events to fulfil, which is the RM above with `e_contact` as the firing event. Two transfers from the drumming-RL line: (i) reward *chain completion* (a strike fires on `e_contact ∧ Δdepth>ε` inside a cadence window), not per-step pressing; (ii) heed **Karbasi et al. 2024** — a compliant arm's *rebound*, left un-fought, yields free multi-strikes, so the `REBOUND` damping above must stay **light**, and (with future end-effector compliance) a rebound that sets up the next `WINDUP` should be *credited*, not killed.

---

## 6. Novel / Non-Obvious Ideas (≥10)

Each tagged with evidence basis and a one-line risk. "Novel" = not in the prior corpus; several are my proposals built on cited mechanisms.

1. **Double-gated impact reward** `v_axial·first_contact·𝟙[Δdepth>ε]` — pay impact speed only when it advances the nail. *Basis:* reward-hacking theory (Skalse et al. 2022 `[E1]`, Pan et al. 2022 `[E1]`). *Risk:* ε too large → no early signal. **This is the highest-value single idea.**
2. **Directional-manipulability shaping** `w(q,n̂)=n̂ᵀJM⁻¹Jᵀn̂` throughout the swing. *Basis:* Ti et al. 2024 `[E2]` (real nail hammering). *Risk:* speed-vs-inertia trade-off (it shapes reachable speed, not delivered momentum).
3. **Impact-momentum event reward** `m_eff(q)·v_axial` at first contact, with `m_eff=1/w`. *Basis:* composite-rigid-body impact for kinematic-controlled arms (Wang, Dehio & Kheddar 2022 `[E1]`). *Risk:* needs clipping or it rewards flailing.
4. **Progress-per-impulse efficiency** `Δdepth/(m_eff·v_axial)`. *Basis:* `[PROPOSAL]` (actuator-friendly aim). *Risk:* favours timid taps — curriculum-gate it.
5. **Reward-machine phase potential** — PBRS over RM states for free, hack-proof cycle progress. *Basis:* Toro Icarte et al. 2018/2022 `[E1]` + Ng et al. 1999 `[E3]`. *Risk:* mis-specified transitions desync state from reality.
6. **Critic-only contact labels & impulse** — true contact force/impulse, `m_eff`, RM-state, time-to-contact in the critic group only. *Basis:* DeXtreme privileged training `[E1]`, RMA `[ANALOGY]`. *Risk:* critic/actor information gap too large → value over-optimistic.
7. **Learned strike-success classifier as reward** — a small net trained (offline, on logged rollouts) to predict "this contact will advance the nail," used as a shaped contact reward. *Basis:* DrS reusable dense rewards `[E3]`, goal-image rewards (Schoettler et al. 2020 `[E2*]`). *Risk:* classifier exploitation (classic reward-model hacking) — freeze it and anneal to sparse.
8. **Adversarial / randomised contact** — per-episode randomise nail `frictionloss`/`damping`/tilt and surface restitution; optionally an adversary that picks the worst nail in a band. *Basis:* dynamics randomisation `[ANALOGY]`, DrEureka auto-DR `[E3]`, Q9. *Risk:* too-wide band → policy learns nothing (use ADR-style curriculum).
8b. **Bounded-stiffness contact for reward integrity** — cap net contact stiffness (`solref`/`solimp`) so force spikes don't scale with contact count. *Basis:* Vuong & Pham 2023 `[E1]`. *Risk:* over-soft contact changes the very dynamics you transfer.
9. **Recovery-quality reward** — in REBOUND, reward low residual head speed + posture return with *low* action energy (absorb, don't fight). *Basis:* reference-spreading "interim mode" idea (van Steen et al. 2024 QP-RS `[E1]`, by analogy — they *disable velocity feedback* during the impact window). *Risk:* encourages freezing; bound it and pair with `interstrike_interval`.
10. **Actuator-stress budget per episode** — an `I²t` thermal accumulator state with a per-episode cap; exceeding it penalises and (optionally) terminates. *Basis:* Qian et al. 2026 `[E2]`, Hwangbo et al. 2019 actuator nets `[E2*]`. *Risk:* needs a torque estimate (use `applied_torque` or an actuator net). Makes the policy *insert rest beats* — desirable for repetition.
11. **Sim-to-real reward-observability filter** — a static lint that asserts no *actor*-fed reward/observation reads a sim-only signal (true contact force, penetration depth, GT nail pose without noise). Those go to the **critic only**. *Basis:* DrEureka outcome-vs-physics finding `[E3]`, van Steen `[E3]`. *Risk:* none — it's a guardrail. **Cheap, high value.**
12. **Constrained impact maximisation (CMDP)** — maximise delivered depth subject to `peak|τ|≤τ_max`, `‖F‖≤F_cap`, `|q̇|≤q̇_max` via a Lagrangian/CPO update rather than soft penalties. *Basis:* Achiam et al. 2017 CPO `[E1]`, Kim et al. 2023 `[E3]`. *Risk:* CPO can stall; PPO-Lagrangian with a PID multiplier is the pragmatic choice.
13. **Anti-chatter interval band** — reward inter-strike air-time in `[τ_lo,τ_hi]`; penalise `t_air<τ_lo` (double-hit bounce). *Basis:* `feet_air_time` `[E3]` + RM loop `[E1]`. *Risk:* band too tight → policy can't satisfy it; learn it via curriculum.
14. **Residual-over-scripted-swing action** — let the policy output a *residual* on a hand-coded swing primitive (safe, always-executing base). *Basis:* Residual RL (Johannink et al. 2019 `[E2*]`), TossingBot residual physics (Zeng et al. 2020 `[E1]`). *Risk:* base primitive biases exploration; needs a decent hand-coded swing. **Strong sim-to-real + safety win; needs an action-space tweak.**
15. **Dense→sparse annealing** — schedule shaping weights toward zero as success-rate rises, leaving the sparse completion. *Basis:* Luo et al. 2022 Dense2Sparse `[E3]`, Reward Training Wheels `[E3]`. *Risk:* anneal too fast → forgetting.
16. **Recoil/reaction-impulse as a first-class objective** *(now literature-backed)* — reward *low* arm-joint reaction impulse at first contact (critic/safety), achieved via effective-mass posture + smoothness. *Basis:* "Let Robots Swing a Hammer" (Xu et al., IROS 2025 `[E1]`) cut arm recoil **64 %** by controlling hammer slide; the welded Z1 lacks a slip DoF, so it must reduce recoil through posture/`m_eff` instead. *Risk:* trades off against impact speed — tune jointly with `impact_progress`.
17. **Fatigue-in-reward via Remaining-Useful-Life** *(now literature-backed)* — budget repeated impacts against a per-episode RUL from FEA stress + **Miner's Rule**, not just an `I²t` proxy; combine with the harmonic-drive **Repeated-Peak-Torque** limit. *Basis:* Prolonging Tool Life (Wu et al. 2025 `[E2]`, RUL-in-reward — *not* hammering, analogy) + Romanyuk et al. 2019 (`notes §2.5`, harmonic-drive RPT) `[E1]`. *Risk:* FEA is heavy; start from the RPT torque threshold.
18. **Rhythmic-Contact-Chain repetition** *(now literature-backed)* — express the multi-strike objective as a sequence of timed contact events to fulfil, rewarding chain completion over per-step pressing. *Basis:* Robot Drummer (Shahid et al. 2025 `[E2]`) — drumming RL is the proof-of-concept that PPO learns timed repetitive impact. *Risk:* needs a cadence window; learn it (CPG-RL) rather than fixing it.

---

## 7. Repetitive Hammering — Detailed Design

The prior corpus has *nothing* here beyond "does retract-and-restrike emerge? (Q2)". This section is the substantive new contribution.

### 7.1 Rhythm
Two valid mechanisms; pick by whether cadence should be fixed or emergent.
- **Emergent (recommended):** the **reward machine** loop `WINDUP→STRIKE→IMPACT→REBOUND→REPOSITION→WINDUP`. Cadence is whatever the policy finds; the RM potential rewards *completing cycles*, not matching a clock. Robust to nail stiffness changes.
- **Imposed cadence:** inject a **phase clock** `(sin 2πt/T, cos 2πt/T)` and reward periodic costs (Siekmann et al. 2021 `[E1]`). Only do this if you want a *target* strike frequency, and let the policy modulate `T` via a learned oscillator (CPG-RL, Bellegarda & Ijspeert 2022 `[E1]`) — a fixed `T` will fight contact-timing variability. DMPs (Ijspeert et al. 2013 `[E1]`) provide rhythmic primitives as an alternative parameterisation `[ANALOGY]`.
- **Grounding (added 2026-05-29):** *learned, timed, repetitive impact already works* — in drumming RL. **Robot Drummer** (Shahid et al. 2025) realises it via the **Rhythmic-Contact-Chain** (the emergent-RM option above), and **Karbasi et al. 2024** by **exploiting passive rebound** for emergent double/triple strokes on a flexible-spring arm ([`notes`](hammering_literature_notes.md) §2.7C). Lessons for the Z1: prefer the emergent RM/contact-chain over a fixed clock, and don't penalise a rebound you can recycle into the next wind-up (keep `REBOUND` damping light).

### 7.2 Strike consistency
Add an episode-level (critic-friendly) penalty `−Var_k(Δdepth_k)` over the per-strike depth increments. This rewards *repeatable* strikes over one fluke (directly counters failure mode "one lucky hit"). Compute by logging each first-contact's `Δdepth` into a per-env buffer; emit the variance at episode end.

### 7.3 Progressive nail depth
The monotone `nail_depth_delta` already gives uniform per-mm reward across 0→75 mm — good. But because total dense reward is *path-independent* (telescopes to `weight·final_max_depth`), it does **not** by itself prefer "several controlled strikes" over "one huge strike." The RM/air-time/impact terms are what produce *striking* behaviour; depth-delta only scores the *outcome*. Consider a **depth-band curriculum**: success threshold ramps `0.02→0.04→0.07` so early training gets frequent completion signal (reverse-curriculum spirit, Florensa et al. 2017 `[E1]`).

### 7.4 Avoiding double-hit chatter
- `interstrike_interval` band reward (§6.13): penalise contacts separated by `< τ_lo`.
- **CAPS temporal smoothness** with a *higher* weight in REBOUND/REPOSITION states (Mysore et al. 2021 `[E1]`) — chatter is high-frequency action oscillation, exactly what CAPS suppresses.
- Sensor hygiene: with `decimation=10`, set `ContactSensorCfg.history_length=10` and detect "a contact happened this control step" over the substep history, not just the last substep (the prior notes' Q12/M3). Prevents the RM from missing intra-step bounces.

### 7.5 Maintaining alignment over multiple impacts
`axis_align` (perpendicular distance to the nail axis line, §4.1) is dense and survives across strikes. Because orientation is *not* commandable, alignment here means *positioning the head over the nail axis*, which the policy can do. If the nail tilts under repeated off-axis strikes (a real failure), randomising initial nail tilt in DR forces robustness.

### 7.6 Deciding when to stop & episode design
- **Termination:** keep `nail_fully_driven` (success) + `time_out`. Add a **safety-envelope termination** (peak torque / impulse exceeded) so unsafe policies die early rather than farming reward.
- **Discount horizon:** `γ=0.99` ⇒ effective horizon ≈100 steps = 2 s, but episodes are 1000 steps = 20 s. For long repetitive sequences the later strikes are heavily discounted and the policy is myopic. **Raise `γ` to 0.995–0.997** (horizon 200–333 steps) for multi-strike, or shorten the episode to match the discount. This is a concrete, currently-wrong setting.
- **Episode length:** 20 s is generous; once a cadence emerges, shorten to ~`(N_target strikes)·(cycle time)·1.5` to reduce idle exploration.

### 7.7 Single vs multi (resolve Q1 first)
Run `test_single_strike.py`. Analytic predictor: max axial momentum `≈ m_eff·v_max`, with `v_max ≲ delta_pos_scale·f_ctrl = 0.05·50 = 2.5 m/s` (commanded; realisable less due to IK tracking + arm dynamics). If a single strike clears 70 mm, the RM loop is optional and `air_time`/repetition terms are pure insurance. If `<30 mm`, the RM/Proposal-C is **mandatory** and should be folded into Stage 1.

---

## 8. Safety & Actuator Protection

The action space (position-only DiffIK) shapes what's possible: you protect actuators **indirectly** (smoother/slower motion) and via **constraints + early termination**, not by commanding low torque.

### 8.1 Penalty / constraint catalogue

| Concern | Mechanism | Signal | How (given position-only control) |
|---|---|---|---|
| **Joint torque** | soft `−Στ²` + CMDP cost on `peak|τ|` | `applied_torque` | computable; gradient is indirect — pair soft penalty (shaping) with a hard constraint (Achiam et al. 2017 `[E1]`; Kim et al. 2023 `[E3]`) |
| **Joint velocity** | `−Σq̇²` + envelope term on `|q̇|>q̇_max` | `joint_vel` | direct, cheap |
| **Motor temperature proxy** | `I²t` accumulator + per-episode budget | `τ²` integral (or actuator net) | Qian et al. 2026 `[E2]`; Hwangbo et al. 2019 `[E2*]` |
| **Impulse spikes** | CMDP cost `max(0,‖F‖−F_cap)` (critic/cost, not actor reward) | sensor `force` | force is fragile as *reward* but fine as a *constraint threshold* (van Steen `[E3]`) |
| **End-effector wrench** | bound axial impulse via `impact_momentum` clip | `m_eff·v_axial` | clip the event reward; transferable (Wang 2022 `[E1]`) |
| **Self-collision** | penalty/termination on disallowed contacts | extra ContactSensor pair | add a sensor for arm-vs-arm/base geoms |
| **Joint limits** | current `−10·joint_pos_limits` (keep) + envelope | joint pos | already implemented |
| **Vibration / HF action change** | CAPS + jerk penalty | actions, policy net | Mysore et al. 2021 `[E1]`; Kobayashi 2022 L2C2 `[E2*]` |
| **Unsafe posture** | `dir_manip` *upper* bound + posture potential | `q`, J, M | discourage singular/over-extended strike postures |
| **Repeated shock accumulation** | per-episode RUL budget (FEA + Miner's Rule) **and** harmonic-drive **RPT** limit | Σ impulse / Στ² at contacts | Prolonging Tool Life (Wu et al. 2025 `[E2]`, RUL-in-reward); Romanyuk et al. 2019 `[E1]` (RPT fatigue limit, `notes §2.5`); induces rest beats |
| **Recoil / reaction at impact** | reward *low* joint reaction-impulse at first contact (critic/safety) | reaction torque/impulse at `first_contact` | "Let Robots Swing a Hammer" (Xu et al. 2025 `[E1]`) cut recoil **64 %** via controlled slip; welded Z1 → reduce via `m_eff` posture + smoothness |

### 8.2 Why constraints, not just penalties
At impact the task-reward gradient is largest, so a soft penalty that is `O(10)` is overwhelmed by a depth spike that is `O(100)` (Kim et al. 2023 `[E3]` make exactly this argument for legged impacts). Use **PPO-Lagrangian** (a PID-controlled multiplier on each constraint) or **CPO** (Achiam et al. 2017 `[E1]`) for the hard limits, keeping soft penalties only for *comfort* terms (smoothness). At minimum, a **terminating safety envelope** (episode ends with no terminal bonus if `peak|τ|`/`|q̇|`/`‖F‖` exceed limits) gives a strong, un-overwhelmable safety signal cheaply.

### 8.3 Sim-to-real protections that double as safety
- **Domain randomisation** of nail+arm dynamics (the missing piece) prevents over-fitting a strike profile that saturates real actuators.
- **CAPS/L2C2 smoothness** reduces the high-frequency command content that excites gear/belt resonances on hardware (Mysore et al. 2021 `[E1]`).
- **Admittance wrapping** (Beltran-Hernandez et al. 2020 `[E1]`; Abu-Dakka & Saveriano 2020 `[E2*]`) gives genuine contact compliance on a position-controlled arm: read wrist F/T, `Δx_compliant = C·(F_des − F_meas)`, add to the IK target. This both protects the arm and recovers a "stiffness knob" you otherwise lack.

---

## 9. Implementation Plan

**Phase 0 — instrumentation & guardrails (½ day, no training).**
1. Add `effective_inertia_axis`, `head_axial_speed`, `contact_force_w` observation funcs (read `mjData`/sensor). Put them in a **critic-only** group.
2. Add a **reward-observability lint** (unit test): assert no actor-group obs/reward func name is in a sim-only blocklist (`contact_force`, `penetration`, GT pose). Fails CI if violated. (Idea #11.)
3. Extend `validate_rewards.py` with phases for `ImpactProgressTerm` (fires only at first-contact *and* when depth advanced; zero on scrape) and PBRS terms (telescoping sum check).
4. Run `verify_reward_setup.py` (random-policy sweep) and `test_single_strike.py` (**resolve Q1**) — this decides whether Proposal C is mandatory.

**Phase 1 — minimal baseline (Proposal A).** Implement the 4 edits (balance, gated impact, DR, privileged critic). Train. Gate to Phase 2 on: success-rate > 0 by ~iter 300, no value-loss blowup.

**Phase 2 — robust staged (Proposal B).** Add PBRS approach/windup, `dir_manip`, CAPS, `joint_vel`, and the safety envelope/Lagrangian. Re-tune `γ→0.995`.

**Phase 3 — repetitive (Proposal C).** Add the reward machine, `air_time`, `interstrike_interval`, `strike_consistency`, thermal budget. Only if Q1 says multi-strike is needed (else keep as insurance terms).

**Signals to log (per term + diagnostics):**
- `Episode_Reward/<term>` (already automatic) — confirm each term's episode-sum band (§4.7).
- `n_contacts/episode`, `Δdepth/strike` distribution, inter-strike interval histogram, `v_axial` at contact, `peak|τ|`, `|q̇|max`, `I²t` per joint, success-rate, episode length.
- value-loss & KL around first successes (Q11), time-to-first-contact (Q6).

**Plots to produce:** success-rate vs iter; per-term episode-sum stacked area; `Δdepth/strike` violin per ablation; impact-speed vs Δdepth scatter (detects scraping); torque/`I²t` vs iter (safety); sim-vs-(eventual)-real strike-count distribution (Q9).

**Unit / sanity tests to add:** gated-impact zero-on-scrape; PBRS telescoping; RM transition table (each event drives the right state); DR ranges actually applied at reset; critic-only terms absent from actor group; contact `history_length=decimation` captures intra-step bounce.

**Success metrics to track:** (1) success-rate under DR; (2) median strikes-to-success; (3) `Δdepth/strike` consistency (low variance); (4) peak torque / `I²t` within budget; (5) action smoothness (spectral content); (6) robustness: success-rate across the DR band edges.

**Curriculum stages:** success-depth ramp `0.02→0.07`; DR width ramp (ADR-style); dense→sparse weight anneal once success-rate > ~70%.

---

## 10. Ablation Matrix

| # | Variant (vs Proposal A) | Hypothesis | Primary metric | Expected failure mode | What it teaches |
|---|---|---|---|---|---|
| A0 | Current live config (depth_delta=2000) | reproduces current behaviour | success-rate, value-loss | value-spike variance; slow-press | baseline + Q11 evidence |
| A1 | depth_delta 2000→600 | balance ↑ stability, ↓ variance | value-loss, success-rate | weaker early signal | dense/sparse ratio effect |
| A2 | + return/value normalisation, keep 2000 | normalisation removes need to re-weight | value-loss | none expected | whether re-weighting vs norm. is better |
| A3 | + gated `impact_progress` | teaches swing-that-advances | `v_axial`@contact, Δdepth/strike | scraping if gate removed | value of double-gating |
| A4 | ungated `impact_speed` (ablate the gate) | should reward-hack (scrape) | impact-speed↑ but Δdepth↓ | **scraping/tapping** | demonstrates the hack the gate prevents |
| A5 | + domain randomisation | ↑ robustness, maybe ↓ peak sim success | success-rate at DR edges | under-fit if band too wide | DR necessity for S2R |
| A6 | + privileged critic | ↓ value error, ↑ sample-eff | explained variance of value | critic over-optimism | asymmetric-critic payoff |
| A7 | PBRS approach vs Gaussian approach | PBRS removes hovering | time hovering near nail | slower initial reach | Ng99 in practice |
| A8 | + `dir_manip` shaping | ↑ impact speed / fewer strikes | strikes-to-success | chases manip not nail | speed-vs-inertia trade-off |
| A9 | γ 0.99 vs 0.995 (multi-strike) | higher γ ↑ long-horizon | strikes-to-success, late-strike quality | instability if γ too high | horizon/episode match |
| A10 | + reward machine (Proposal C) | enables rhythm/recovery | n_cycles, chatter rate | RM desync if events mis-tuned | event-driven vs always-on |
| A11 | + safety constraints (Lagrangian) | ↓ peak τ/impulse at small reward cost | peak|τ|, success-rate | over-conservative (timid taps) | constraint vs penalty |
| A12 | + thermal budget | induces rest beats | `I²t`, cadence | starves striking if λ high | shock-accumulation control |
| A13 | dense→sparse anneal | ↓ shaping-farming, keeps success | success-rate post-anneal | forgetting if too fast | Dense2Sparse in practice |

---

## 11. Failure Modes & Reward-Hacking Checklist

Concrete, env-specific. ✅ = current design already defends; ⚠️ = partial; ❌ = unaddressed (v2 fix noted).

| Failure | Why it happens here | Status | Defence (v2) |
|---|---|---|---|
| **Hit the table/block, not the nail** | approach reward is to `nail_top`; contact only on body `nail` | ✅ | ContactSensor secondary = body `nail` only; `nail_depth_delta` needs actual nail motion |
| **Exploit contact-solver artefacts** (force spikes from `impratio=10`, soft `solref`) | rewarding sensor force would chase these | ✅/❌ | never reward force (delete `contact_force_magnitude`); bound stiffness (Vuong & Pham `[E1]`); keep the 4 mm settle dead-zone |
| **Gravity-settling counted as progress** | solver settles nail ~3.5 mm at reset | ✅ | `_SETTLE_OFFSET=0.004` dead-zone (already in code) — but **derive it from a no-contact rollout under DR**, don't hardcode |
| **Scraping / glancing tap to trigger contact** | `impact_speed·first_contact` pays speed on any nail contact | ❌→✅ | **double-gate** `·𝟙[Δdepth>ε]` (Idea #1) |
| **Huge unsafe velocity / flailing** | ungated velocity/momentum reward unbounded | ⚠️→✅ | clip `impact_momentum`; safety envelope on `|q̇|`,`τ`; CAPS |
| **One lucky hit, no repeatability** | sparse completion rewards a fluke | ❌→✅ | `strike_consistency` variance penalty (critic); DR; require N strikes |
| **Policy freezes near target** | discount + soft hold-at-goal Gaussian | ⚠️ | PBRS approach (no hovering optimum); optional time penalty; raise γ |
| **Slow-press instead of swing** | depth-delta is path-independent; pressing also advances nail | ⚠️→✅ | `impact_progress`/`air_time` reward *striking*; (the original motivation, now properly instrumented) |
| **Action jitter / chatter** | `action_rate=−0.01` too weak | ⚠️→✅ | CAPS temporal+spatial, jerk penalty, L2C2; higher weight in REBOUND |
| **Gentle-tapping local optimum (farm small deltas)** | many tiny advances | ✅ | monotone max-depth + `clamp_min(0)` already prevents re-farming; `progress_per_impulse` only post-coarse |
| **Double-hit bounce (chatter at contact)** | substep bounce within `decimation=10` | ❌→✅ | `interstrike_interval` band; `history_length=decimation` contact detection |
| **Reward-model exploitation** (if using a learned success classifier, Idea #7) | policy games the frozen net | ⚠️ | freeze classifier; anneal to sparse; OOD detector (Pan et al. 2022 `[E1]`) |
| **DR-band over-fit / under-fit** | too-narrow → no transfer; too-wide → no learning | n/a | ADR-style curriculum on the band |

General principle (Skalse et al. 2022 `[E1]`): a proxy is "unhackable" only under restrictive conditions; *assume* every dense term is hackable and pair it with either a gate, a PBRS form (Ng et al. 1999 `[E3]`), or a constraint.

---

## 12. Final Recommendation

**Implement first (low-risk, code-local, fixes real defects):**
1. Re-balance `nail_depth_delta` (2000→~600) **or** enable PPO return/value normalisation. (A1/A2)
2. Wire the already-present `ContactSensor` into the **double-gated `impact_progress`** term. (A3, Idea #1)
3. Add **domain randomisation** to the two reset events + nail joint. (A5)
4. Add a **privileged critic** group + the **reward-observability lint**. (A6, Idea #11)
5. Resolve **Q1** with `test_single_strike.py` before any of the above consumes a training run.

**Implement second (principled, after baseline learns):**
6. PBRS approach/windup + `dir_manip` shaping. (A7/A8)
7. CAPS smoothness + `joint_vel` + safety **envelope**; raise `γ→0.995` for multi-strike. (A9/A11)
8. Dense→sparse anneal once success-rate is high. (A13)

**Risky but interesting (worth a dedicated experiment):**
9. Reward machine for repetition (Proposal C) — only if Q1 says multi-strike is required. (A10)
10. PPO-Lagrangian constrained impact maximisation (CPO/PID-Lagrangian). (A11)
11. Residual-over-scripted-swing action space (Idea #14) — strong S2R/safety upside, needs an action-interface change.
12. Learned strike-success classifier reward (Idea #7) — powerful but adds reward-model-hacking surface.

**Do not implement yet (not actionable on the current action space / premature):**
13. Orientation-alignment quat-gate — **moot** with `orientation_weight=0.0`; needs an action-space change first (§2.3).
14. Direct contact-force reward — fragile sim artefact (use force only as a *constraint*).
15. Full variable-impedance action space (VICES) — requires torque-level control the env doesn't have; use **admittance wrapping** instead if you need compliance.
16. Fixed phase-clock rhythm — only after CPG-RL-style learned frequency; otherwise it fights contact timing.

**Open research questions (carried + new):**
- **Q1** single-strike feasibility (script ready) — *gates everything repetitive.*
- **Q2** does retract-restrike emerge without an air-time/RM term? (now testable via A3-vs-A10).
- **Q9** single deep vs many moderate strikes — easier to learn / better transfer? (needs DR + hardware).
- **New-Q13** speed-vs-inertia: does `dir_manip` (speed) or `impact_momentum` (inertia·speed) yield deeper, safer strikes? (A8).
- **New-Q14** does an asymmetric privileged critic actually reduce value error here, or does the actor/critic gap hurt? (A6).
- **New-Q15** is an event-driven reward machine more sample-efficient than always-on dense terms for repetition? (A10).
- **New-Q16** what `γ`/episode-length pairing is optimal for N-strike sequences? (A9).
- **New-Q17** can the 4 mm settle dead-zone be replaced by a DR-robust, rollout-derived threshold?

---

## Devil's Advocate Pass (deep-research Checkpoint 3)

1. **"You're over-engineering — the implemented baseline might just work."** Possible. That's why **Implement-first** is four small edits and **resolving Q1** comes first; everything heavy (RM, constraints, residual) is explicitly gated on observing a failure. The report's structure is "minimum viable, escalate on evidence," matching the user's own augment-not-replace strategy.
2. **"Most of your strongest ideas are `[ANALOGY]` from locomotion/insertion, not hammering."** Partly. **Update (2026-05-29):** a deeper external search (logged in [`hammering_literature_notes.md`](hammering_literature_notes.md) §2.6–2.7) overturned this report's original "near-empty niche" claim. Learning-to-hammer is a **standard RL benchmark** (Adroit/DAPG `hammer` → D4RL `hammer-v0` → RRL/VRL3/H-InDex/MoDem/DexHandDiff/DORA/CQL/IQL) **and** has a real long tail of direct nail-driving RL/IL: Tool-as-Interface (CoRL 2025), HMAMP (Robotica 2025), Teramae et al. 2018 (PAM arm, 3–5 strikes), and "Let Robots Swing a Hammer" (IROS 2025). So the analogies remain load-bearing for the *impact-physics* terms, but the **reward skeleton** is now grounded in a real RL hammering env (the Adroit reward, `notes §2.6`), and the **repetitive-rhythm** machinery is grounded in drumming RL (Robot Drummer, Karbasi). The genuine gap is narrower than "nobody hammers with RL": it is **sim-to-real RL for repetitive, momentum/impact-explicit nail-driving on a manipulator** — the dexterous-hand benchmarks press a nail quasi-statically; none model the strike physics as the learning objective.
3. **"The manipulability term could be a distraction (speed≠momentum)."** Acknowledged explicitly (Idea #2/#3, New-Q13) — it's an *ablation*, not a recommendation to ship blind.
4. **"Privileged critic can make value over-optimistic and hurt the actor."** Real risk (New-Q14); that's why it's an ablation with explained-variance as the metric, reversible if it hurts.
5. **"Constraints can make the policy timid (no real strikes)."** The classic CPO stall (A11 failure mode). Mitigation: start with a terminating envelope (cheap, strong) before full Lagrangian.

## Limitations

1. **No training or hardware data** — every "expected failure mode" is a hypothesis; the ablation matrix is the means to test them.
2. **Direct-evidence sparsity** — hammering-specific RL literature barely exists; much rests on analogy (flagged per item).
3. **Citation integrity is tiered, not uniform.** 18 sources were re-fetched and verified this session (`[E1]`/`[E2]`); ~11 are subagent-reported canonical works not re-fetched (`[E2*]`) and should be spot-checked before formal publication; 17 are inherited (`[E3]`) and carry the prior corpus's flags (notably ARMADA/DrEureka/RTW venue claims, and the previously-corrected Meta-World author list).
4. **One agent claim was demoted:** AutoMate's specific "interpenetration-penalty reward formula" could not be confirmed from its abstract — the *concept* (penalise unphysical simulator penetration) is presented as a `[PROPOSAL]` inspired by IndustReal/AutoMate, not as their verified term.
5. **The XML-level nail parameters** (`frictionloss=0.3`, `damping=8`) are taken from the prior notes, not re-read from the sibling-repo XML this pass.
6. **`mjData` Jacobian/inertia access** for `dir_manip`/`m_eff` assumes mjlab/mujoco_warp exposes `J`,`M` cheaply per step; verify the API cost before using it on the actor path (else keep it critic-only or every-k-steps).

## AI-Assisted Research Disclosure

This report was produced by Claude (Opus 4.8) executing the ARS `deep-research` pipeline in `full` mode, with an **external** literature sweep (unlike the prior local-corpus report). Six parallel literature-search subagents (Sonnet) performed discovery; the orchestrating model independently re-verified 18 load-bearing citations against arXiv/DOI and tier-tagged all others. Evidence-backed claims are separated from `[PROPOSAL]` speculation throughout, and analogical transfers are flagged `[ANALOGY]`. Residual fabrication risk concentrates in the `[E2*]` tier (subagent-reported, not re-fetched) — verify those author lists before any external use.

---

## References

Grouped by tier. arXiv IDs / DOIs given for verification. `[E1]` verified this session at a peer-reviewed venue; `[E2]` preprint verified to exist; `[E2*]` subagent-reported canonical, not re-fetched; `[E3]` inherited from prior corpus.

**Direct hammering / striking RL & IL literature (added 2026-05-29 via a deep external sweep; full annotations + the wider field map in [`hammering_literature_notes.md`](hammering_literature_notes.md) §2.6–2.7):**
- Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman, J., Todorov, E., & Levine, S. (2018). Learning Complex Dexterous Manipulation with Deep RL and Demonstrations (DAPG; origin of the Adroit `hammer` env). *RSS 2018*. arXiv:1709.10087. `[E1]`
- Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). D4RL (standard offline-RL `hammer-v0` Adroit datasets). arXiv:2004.07219. `[E1]`
- Chen, H., Zhu, C., Liu, S., Li, Y., & Driggs-Campbell, K. (2025). Tool-as-Interface: Learning Robot Policies from Observing Human Tool Use (nail-hammering, 13/13). *CoRL 2025*. arXiv:2504.04612. `[E1]`
- Ma, Z., Tian, C., & Gao, Y. (2025). Manipulate as Human (HMAMP — adversarial motion priors; energy-storing back-swing; real-arm hammering). *Robotica, 43*(6), 2320–2332. arXiv:2510.24257. `[E1]`
- Xu, Y., … Sun, Z. (2025). High-dynamic Tactile Sensing for Tactile Servo Manipulation: Let Robots Swing a Hammer (controlled slide; −64 % arm recoil). *IROS 2025*. IEEE Xplore 11246617. `[E1]`
- Teramae, T., Ishihara, K., Babič, J., Morimoto, J., & Oztop, E. (2018). Human-In-The-Loop Control and Task Learning for Pneumatically Actuated Muscle Based Robots (learned nail-driving, 3–5 strikes). *Frontiers in Neurorobotics*. PMC6232299. `[E1]`
- Shahid, A. A., Braghin, F., & Roveda, L. (2025). Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming (Rhythmic Contact Chain). arXiv:2507.11498. `[E2]`
- Karbasi, S. M., Jensenius, A. R., Godøy, R. I., & Torresen, J. (2024). Embodied Intelligence for Drumming (DDPG; passive-rebound multi-strokes). *Frontiers in Robotics & AI, 11*. PMC11609846. `[E1]`
- Wu, P.-Y., Kuo, C.-Y., Kadokawa, Y., & Matsubara, T. (2025). Prolonging Tool Life: Lifespan-guided RL (RUL / Miner's-Rule fatigue in the reward; *not* hammering — used by analogy). arXiv:2507.17275. `[E2]`
- Orbik, J., Agostini, A., & Lee, D. (2021). Inverse Reinforcement Learning for Dexterous Hand Manipulation (Adroit hammer; from a TU Munich MSc thesis, mediaTUM 1553993). *IEEE ICDL 2021*. DOI:10.1109/ICDL49984.2021.9515637. `[E1]`
- Vu, J., Erens, R., Stefanelli, H., Cisneros-Limón, R., Benallegue, M., & Benallegue, A. (2026). QP-based impact-momentum maximization for a hammering task by a humanoid robot (effective-mass `m_e,n = (nᵀJνM⁻¹Jνᵀn)⁻¹`; model-based, not learning). *IEEE AMC 2026*. DOI:10.1109/AMC67705.2026.11435814. `[E1]`

**Self-verified this session — peer-reviewed `[E1]`:**
- Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). Constrained Policy Optimization. *ICML 2017*. arXiv:1705.10528.
- Beltran-Hernandez, C. C., Petit, D., Ramirez-Alpizar, I. G., Nishi, T., Kikuchi, S., Matsubara, T., & Harada, K. (2020). Learning Force Control for Contact-rich Manipulation Tasks with Rigid Position-controlled Robots. *IEEE RA-L (IROS) 2020*. arXiv:2003.00628.
- Bellegarda, G., & Ijspeert, A. (2022). CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion. *IEEE RA-L 2022*. arXiv:2211.00458.
- Florensa, C., Held, D., Wulfmeier, M., Zhang, M., & Abbeel, P. (2017). Reverse Curriculum Generation for Reinforcement Learning. *CoRL 2017*. arXiv:1707.05300.
- Handa, A., Allshire, A., Makoviychuk, V., Petrenko, A., et al. (2023). DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality. *ICRA 2023*. arXiv:2210.13702.
- Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., & Schaal, S. (2013). Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors. *Neural Computation, 25*(2), 328–373. DOI:10.1162/NECO_a_00393.
- Martín-Martín, R., Lee, M. A., Gardner, R., Savarese, S., Bohg, J., & Garg, A. (2019). Variable Impedance Control in End-Effector Space (VICES). *IROS 2019*. arXiv:1906.08880.
- Mysore, S., Mabsout, B., Mancuso, R., & Saenko, K. (2021). Regularizing Action Policies for Smooth Control (CAPS). *ICRA 2021*. arXiv:2012.06644.
- Pan, A., Bhatia, K., & Steinhardt, J. (2022). The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models. *ICLR 2022*. arXiv:2201.03544.
- Siekmann, J., Godse, Y., Fern, A., & Hurst, J. (2021). Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. *ICRA 2021*. arXiv:2011.01387.
- Skalse, J., Howe, N. H. R., Krasheninnikov, D., & Krueger, D. (2022). Defining and Characterizing Reward Hacking. *NeurIPS 2022*. arXiv:2209.13085.
- Toro Icarte, R., Klassen, T. Q., Valenzano, R., & McIlraith, S. A. (2018). Using Reward Machines for High-Level Task Specification and Decomposition in RL. *ICML 2018*; (2022) *JAIR 73*, 173–208. arXiv:2010.03950.
- Wang, Y., Dehio, N., & Kheddar, A. (2022). Predicting Impact-Induced Joint Velocity Jumps on Kinematic-Controlled Manipulator. *IEEE RA-L 2022*. DOI:10.1109/LRA.2022.3167614. arXiv:2202.12646.
- Zeng, A., Song, S., Lee, J., Rodriguez, A., & Funkhouser, T. (2020). TossingBot: Learning to Throw Arbitrary Objects with Residual Physics. *IEEE T-RO 2020*. arXiv:1903.11239. *(verified via search this session.)*

**Self-verified this session — preprint `[E2]`:**
- Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete Problems in AI Safety. arXiv:1606.06565.
- Tang, B., Akinola, I., Xu, J., Wen, B., Handa, A., Van Wyk, K., Fox, D., Sukhatme, G. S., Ramos, F., & Narang, Y. (2024). AutoMate: Specialist and Generalist Assembly Policies over Diverse Geometries. arXiv:2407.08028. *(reward-detail claim demoted — see Limitations §4.)*
- Ti, B., Gao, Y., Zhao, J., & Calinon, S. (2024). An Optimal Control Formulation of Tool Affordance Applied to Impact Tasks. arXiv:2402.05502. *(DIRECT: real nail-hammering on a 7-DoF arm.)*
- Qian, L., Wan, Y., Wang, S., & Luo, X. (2026). Learning Thermal-Aware Locomotion Policies for an Electrically-Actuated Quadruped Robot. arXiv:2603.01631. *(very recent preprint.)*

**Subagent-reported, canonical, not re-fetched `[E2*]` — verify author lists before external use:**
- Abu-Dakka, F. J., & Saveriano, M. (2020). Variable Impedance Control and Learning — A Review. *Frontiers in Robotics and AI, 7*. arXiv:2010.06246.
- Bogdanovic, M., Khadiv, M., & Righetti, L. (2020). Learning Variable Impedance Control for Contact Sensitive Tasks. *IEEE RA-L 2020*. arXiv:1907.07500.
- Buchli, J., Stulp, F., Theodorou, E., & Schaal, S. (2011). Learning variable impedance control. *IJRR, 30*(7), 820–833. DOI:10.1177/0278364911402527.
- Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., & Hutter, M. (2019). Learning agile and dynamic motor skills for legged robots (actuator nets). *Science Robotics, 4*(26). arXiv:1901.08652.
- Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Loskyll, M., Ojea, J. A., Solowjow, E., & Levine, S. (2019). Residual Reinforcement Learning for Robot Control. *ICRA 2019*. arXiv:1812.03201.
- Kobayashi, T. (2022). L2C2: Locally Lipschitz Continuous Constraint towards Stable and Smooth RL. *IROS 2022*. arXiv:2202.07152.
- Lee, M. A., Zhu, Y., Zachares, P., Tan, M., Srinivasan, K., Savarese, S., Fei-Fei, L., Garg, A., & Bohg, J. (2020). Making Sense of Vision and Touch. *IEEE T-RO 2020*. arXiv:1907.13098.
- Schoettler, G., Nair, A., Luo, J., Bahl, S., Ojea, J. A., Solowjow, E., & Levine, S. (2020). Deep RL for Industrial Insertion Tasks with Visual Inputs and Natural Rewards. *IROS 2020*. arXiv:1906.05841.
- Vuong, N., & Pham, Q.-C. (2023). Contact Reduction with Bounded Stiffness for Robust Sim-to-Real Transfer of Robot Assembly. *IROS 2023*. arXiv:2306.06675.
- Zhang, X., Sun, L., Kuang, Z., & Tomizuka, M. (2021). Learning Variable Impedance Control via Inverse RL for Force-Related Tasks. *IEEE RA-L 2021*. arXiv:2102.06838.
- van Steen, J. J. C., van den Brandt, G., van de Wouw, N., Kober, J., & Saccon, A. (2024). QP-based Reference Spreading Control for Dual-Arm Manipulation with Planned Simultaneous Impacts. *IEEE T-RO 2024*. arXiv:2305.08643. *(subagent-reported; not re-fetched this session.)*

**Inherited from prior corpus `[E3]` (see `REWARD_LITERATURE.md`; carries prior flags):**
- Ng, Harada & Russell (1999) PBRS · Wu et al. (2021) DREM, arXiv:2011.08458 · Tang et al. (2023) IndustReal, arXiv:2305.17110 · Mu et al. (2024) DrS, arXiv:2404.16779 · Luo et al. (2022) Dense2Sparse, arXiv:2003.02740 · Kumar, Todorov & Levine (2016) · D'Ambrosio et al. (2023) Robotic Table Tennis, arXiv:2309.03315 · Kim et al. (2025) ARMADA, arXiv:2502.16908 *(RSS'25 venue not independently verified)* · van Steen et al. (2024) sim-to-real velocity jumps, arXiv:2411.06319 · Rudin et al. (2022) RSL-RL, arXiv:2109.11978 · Ma et al. (2023) Eureka, arXiv:2310.12931 · Ma et al. (2024) DrEureka, arXiv:2406.01967 *(venue not verified)* · Yu et al. (2020) Meta-World, arXiv:1910.10897 · Narang et al. (2022) Factory, arXiv:2205.03532 · Berducci et al. (2024) HPRS, arXiv:2110.02792 · Wang et al. (2025) Reward Training Wheels, arXiv:2503.15724 *(venue not verified)* · Kim et al. (2023) Not Only Rewards but also Constraints, arXiv:2308.12517.
