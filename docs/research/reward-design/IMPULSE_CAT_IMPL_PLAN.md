# Impulse-CaT implementation plan — extending faithful soft γ(1−δ) CaT from joint-velocity to per-joint impact impulse

> **Step-1 result (2026-08-15):** The frozen survey and authorized uniform-`0.9` diagnostic pair are complete. The provisional no-`2x` vector `[0.82, 1.64, 0.82, 0.82, 0.82, 0.82]` was nonbinding in 64 deterministic mean-policy worlds but bound 593/98,304 sampled training-like controller reads, almost entirely at J3. At offline `imp_max_p=0.2`, impulse won 592/593 active max soft-OR reads. Complete-event dose is unresolved: 585/593 violating reads were terminal and 571/580 unambiguous one-read physical-event associations were right-censored. The pooled observed-prefix `P_event` median/p95/max was `0.0123/0.1092/0.2`, while the completed unambiguous subset had only N=9 (`0.0746/0.2/0.2`). No dose or provisional-boundary canary is selected; `0.2` remains an analytical starting point. Verdict 2 is calibration revision before scientific training. See `../../results/2026-08-15_z1_impulse_cat_step1.md`.
>
> **Current threshold/calibration correction (2026-08-14; supersedes all `kappa=2`, “manufacturer cap,” and `27.3 ms` interpretations below):** Unitree publishes Z1 actuator maximum/URDF effort values, not a per-joint external reaction-impulse damage limit. The historical `[1.64, 3.28, 1.64, 1.64, 1.64, 1.64] N·m·s` vector remains in existing registered tasks and banked evaluators solely to preserve identity. Step 1 opts into the provisional no-`2x` vector `[0.82, 1.64, 0.82, 0.82, 0.82, 0.82]`; its uniform `0.9` derivative `[0.738, 1.476, 0.738, 0.738, 0.738, 0.738]` is diagnostic, not hardware. The completed survey and diagnostic canary do not turn either vector into a hardware limit or scientific treatment. See `IMPULSE_CAP_PROVENANCE.md`.

> **Current status correction (2026-08-02; supersedes operational language below):** The 2026-07-10 C2 graded-δ result and its `imp_max_p=0.5` selection are dated historical evidence, not current authorization. The repository and fixed-reset campaign must remain fail-closed and log-only at `imp_max_p=0.0`; enforcement is deferred. Deterministic direct-reference repeats establish liveness and repeatability against the frozen caps only—they cannot calibrate the normalizer or select an enforcement setting. C2 must be re-established under an authorized calibration protocol, with the outstanding enforcement gates discharged, before any C3 enforcement run.

**Historical status (2026-07-14):** **C0–C2 DONE; the machinery is SHIPPED, exploit-hardened, and LOG-ONLY (`imp_max_p=0` repo default; a composed velocity arm keeps its own `max_p`). Next milestone: the first GPU training of the impulse arms (C3, prior-vs-none pair on Lightning, from commit ≥`117e070`).** Timeline: C0 built + CPU-validated 2026-06-18; hardened by two deep code-review rounds (2026-07-05). Khadiv gave **verbal go-ahead** for soft-CaT as the mechanism (2026-07-06; C3's full GPU run stands as the formal confirmation), and the fixture-era threshold re-derivation is DONE — the measured `IMP_J_LIMIT` and `i_ref` are live in `env_cfgs.py`. **C2 PASSED 2026-07-10** — graded-δ certificate, `imp_max_p=0.5` chosen for enforcement campaigns; durable record: `docs/results/2026-07-10_c2_enforcement_record.md` (read its STALENESS notes — the delivered share and the 0.5 calibration both predate later changes). **Λ semantics amended 2026-07-13** (`581578d`): the enforced accumulator is now a TIME-based **sliding window** (max 25-substep ≈ 50 ms contact-masked sum), replacing the per-event pulse — closes the gentle-prefix masking bypass found by adversarial review; note the multi-read consequence for any future `imp_max_p` calibration (window > decimation ⇒ ~3 reads per violation). **Enforcement (`imp_max_p>0`) is additionally gated on Khadiv decision (e)** — what quantity Λ should bound (ballistic impulse is VACUOUS for reachable strikes; the windowed press-through reaction is what binds, conditionally on the window/cap pairing) — plus four pre-enforcement gates; see `docs/results/2026-07-12_impulse_vacuity.md`, `2026-07-12_state_of_everything.md`, and `2026-07-12_khadiv_vic_addendum.md`. The **Pinocchio decision is RECORDED**: the object-side ∫F·dt stays the accepted ground truth and `impulseDynamics` is deferred indefinitely (mirrors the 2026-06 decision in visual plan `plan-c00a0e0144274fe4`) — see §4/§7 below. Design produced by a multi-agent deep-dive (4 readers → synthesis → 2 adversarial critics) over the thesis impulse-design docs, the implemented soft-CaT code, the mjlab substep machinery, and external literature.

> **C0 build (done):** `src/tasks/hammer/mdp/impulse_bound.py` (`SubstepImpulseAccumulator` robot-side Λ_j, contact-anchored + sensor-gated, optional pre-contact baseline subtraction; `SubstepDeliveredImpulse` object-side ∫F_axial dt via a `reduce="netforce"` sensor) · `cat/constraints.py::joint_impulse_excess` (log-only) · `cat/hook.py` (use_vel/use_impulse split + sparse-signal robust normalizer; `imp_max_p=0` ⇒ δ≡0) · `rewards.py::DeliveredImpulseTerm` (maximize objective) · arm `Unitree-Z1-Hammer-CaT-Impulse` (cat_impulse flag) · `derive_impulse_thresholds.py` (quantity gate) · `validate_rewards.py` Phase M · 38 new unit tests. Full local pre-train gate GREEN.
> **C0 findings (EE-dependent measurements — gripper-era C0, friction range revised 2026-07-04 fixture-era; superseded by any EE change, re-derive via `derive_impulse_thresholds.py`):** the dominant contaminant is dof-FRICTION (efc `mjCNSTR_FRICTION_DOF`, ~41–48% of the raw contact-window Λ_j, EE-dependent, spread across the load-bearing arm joints), not the weld (whose contribution is minor under gravity compensation); the object-side delivered ∫F·dt is the clean weld/friction-immune ground truth; per-joint J_limit and the reference-strike ratio are fixture-dependent — do not quote the old gripper-era numbers, re-run `derive_impulse_thresholds.py`. efc-row isolation of the *specific* contact is not clean in mujoco_warp (deferred). Pinocchio cross-check SKIPPED (not installed); **decision RECORDED 2026-07-06** — the object-side ∫F·dt is the accepted ground truth in its place, and `impulseDynamics` is deferred indefinitely (mirrors the 2026-06 decision in visual plan `plan-c00a0e0144274fe4`) — this is no longer a "pending" decision. `max_p>0` is NOT enabled by default (`imp_max_p=0.0` repo default); the dated C2 record documents a prior per-run raise, not current authorization (see the current correction above).

> **Track 2 (Tasks 8–10, SHIPPED 2026-07-10, LOG-ONLY):** efc-row contact isolation now exists as an independent, rigorous ground truth alongside the enforced Track-1 quantity — `contact_row_qfrc`/`ContactRowImpulseAccumulator` (`src/tasks/hammer/mdp/contact_row_impulse.py`, Task 8/9), wired as the `substep_impulse_rows` metric (stashed on env as `_hammer_substep_impulse_rows`), immune to dof-friction contamination BY CONSTRUCTION (sums only the hammer↔nail efc rows) rather than by baseline subtraction. **Never consumed for control or δ** — diagnostic/validation only. Task 10 adds the three-way quantity gate: `derive_impulse_thresholds.py` section [6] prints raw Λ | baseline-subtracted Λ | contact-row Λ | object-side ∫F·dt per joint + diagnostic raw-vs-rows residual/residual-after-subtraction, saves `figures/impulse_contamination.png`, and checks the sign-aware `rows ≤ raw + noncontact` (triangle inequality, 5% tolerance; **AMENDED 2026-07-10**, see the Track-2 finding note below; **DEMOTED to INFORMATIONAL 2026-07-14**, adversarial-review I5 — the identity holds by construction when all three sums share the same substeps, so it certifies nothing; it is kept as a shipped-vs-script drift tripwire only, and the ASSERTED Track-2 gates are the object-side ∫F·dt cross-checks) + a Track-2-vs-object-side consistency cross-check (ratio-spread ≤ 5×, "[TRACK-2 BUG]" on disagreement); `validate_rewards.py` Phase M gained an M5 sub-check (rows wired, zero pre-contact, positive after the strike, ≤ the sign-aware bound via a manually-driven raw-mode comparator instance plus a manually-accumulated noncontact term — M5 still HARD-fails its variant, a deliberate asymmetry vs the derive script's informational demotion: M5 is a cheap always-run drift tripwire, the derive gate is the quantity authority). Incidental fix: the pre-existing `ximp_err` shipped-vs-script cross-check in the derive script was comparing the shipped (subtract_baseline=True since the 2026-07-10 C2 enforcement commit) window value against the unsubtracted `raw` sum — a stale comparison that always spuriously failed; now compares against the sliding-window mirror `sub_capped` (2026-07-13).
>
> **Track-2 finding (joint3 sign-cancellation, corroborated not a bug):** the `rows ≤ raw` hard gate genuinely fails for `joint3` (reproduced independently in both the derive script's oblique reference strike, ~2.9× over, and `validate_rewards.py` Phase M5's straight-down driven strike, ~1.2× over) — dof-friction on joint3 partially CANCELS the contact reaction inside the raw (unsubtracted) `qfrc_constraint` sum at some substeps, so `raw` *undershoots* the true per-joint reaction there and is not a universal upper bound. Three independent lines of evidence say this is real physics, not a Track-2 defect: (1) `joint3` already showed an anomalous negative contamination% in section [2] (`sub_mean > raw_mean`) before this task touched anything; (2) `rows` agrees almost exactly with the independently-computed baseline-subtracted `sub` for the load-bearing joints (joint2/3/4: e.g. joint3 `sub=0.0420` vs `rows=0.0420`, ~0% residual) — two separately-coded quantities converging on the same number; (3) the Track-2-vs-object-side cross-check passes with an essentially perfect ~1.0× ratio spread across all 15 reference strikes, and Task 8's own `reconstruct_qfrc_from_efc` invariant test already certifies the underlying efc math bit-exact. **Net effect: this further validates the C2 `subtract_baseline=True` enforcement decision** (the shipped quantity tracks the rigorous ground truth even on the one joint where the naive `raw` signal is misleading). **Decision ADJUDICATED 2026-07-10:** the hard gate is amended to the sign-aware triangle-inequality bound `rows ≤ raw + noncontact` (where `noncontact_j = Σ|qfrc_constraint_j − contact_rows_j|·dt` over the same window — an exact per-substep identity given the Task-8 reconstruction invariant, `|contact| = |qfrc − (qfrc − contact)| ≤ |qfrc| + |qfrc − contact|`), replacing the literal `rows ≤ raw` bound that assumed same-sign contamination; both `derive_impulse_thresholds.py` and `validate_rewards.py` Phase M5 now PASS under the amended gate. The joint3 sign-cancellation finding above is kept as a documented, thesis-relevant result — it is evidence, not an error, and is not deleted; see `task-10-report.md` for the full per-joint numbers and the fix-wave report for the amended-gate verification transcript.

**Relationship to other docs:**
- `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` — the soft-CaT machinery this reuses (CaT δ-math, `CatPPO` scale-positives + dual-mask GAE, `CatSoftHook`). All of it is **unchanged** by this plan.
- `../../archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (stages T0–T5) — the *prior* impulse plan, written before soft-CaT was chosen. It mandates a **contact-anchored window** (Stage T4) and prescribes an **excess-penalty → PID-Lagrangian CMDP** enforcement ladder. This plan supersedes the *enforcement-mechanism* portion (soft-CaT instead of penalty/Lagrangian) and inherits its quantity/windowing/threshold decisions.
- `../tracking_impact_impulse_design_research.md` — the cited research (D1–D6); the quantity `Λ_j = Σ|qfrc_constraint_j|·h` over a contact-anchored window comes from SQ3 (line 99).
- `IMPULSE_CAP_PROVENANCE.md` — the current threshold interpretation; retires the unsupported historical `kappa=2` conversion without rewriting dated evidence.

---

## ⚠️ Decision to confirm with Khadiv before building

The thesis direction docs prescribe the **CMDP / PID-Lagrangian** target form for the impulse constraint (`E[Σ_episode Λ_j] ≤ d_j`, `tracking_impact_impulse_design_research.md:140`). They do **not** yet name soft-CaT as the impulse enforcement path. Extending soft-CaT (validated on the velocity bound) to impulse is the **user's stated direction** and is well-justified (per-step state-wise constraint → CaT is the correct *type*, see `CONSTRAINED_RL_LANDSCAPE.md`; Lagrangian λ is sensitive, Spoor et al. 2025 / arXiv:2510.17564), but it is a deviation from the written plan. **Surface it explicitly.** The two are not exclusive: soft-CaT (per-step incentive) and an episodic-sum CMDP target (budget over the episode) sit at different rungs of the escalation ladder and can coexist.

---

## 1. The quantity to bound (resolved)

**Per-joint generalized reaction impulse, contact-anchored, contact-only:**

```
Λ_j = Σ_{substeps s in the CONTACT window}  |qfrc_contact_{s,j}|  · physics_dt        [N·m·s], shape (B, 6)
```

where `qfrc_contact = J_contactᵀ · efc_force_contact` is the **contact-constraint** generalized force projected to arm joints — i.e. the *robot-side* reaction `JᵀΛ` that travels back into each joint at impact. This is the **constraint** the thesis bounds, the opposite side of the same collision from the *object-side* delivered impulse on the nail (the reward objective; never touched here). `thesis_handoff_brief_original.md:47-50`.

**Why this is impulse-not-energy:** it is a time-integral of *force* (momentum transfer, ∝ `m_eff·Δv`), not force·velocity (energy). It bounds the mechanical shock the harmonic drive absorbs, invariant to *how* the velocity jump happens. An energy bound would let the policy trade a brief huge force-spike for a long gentle push of equal energy — the opposite of gearbox protection. *(2026-07-10 qualifier: the `∝ m_eff·Δv` momentum-pinning reading holds only for completed ballistic/impulsive events with a release observed inside the window — the 2026-07-10 solver-sensitivity run ([`../../results/2026-07-10_solver_sensitivity.md`](../../results/2026-07-10_solver_sensitivity.md)) showed the current Z1 reference strike is a protocol-truncated press, where Λ ≈ F̄·T_window instead: timestep-robust (~5%) but solref-fragile (−37…−38%). The impulse-not-energy rationale for the quantity choice stands; report solref/solimp/timestep as modeling provenance. Design consequence (UPDATED for the 2026-07-13 sliding window): an unreleased sustained press no longer grows one unbounded event — Λ saturates at F̄·window·dt (one 50 ms window's worth), and the press pressure comes from the reading persisting across ~3 consecutive 50 Hz reads (the δ multi-read effect), not from unbounded growth. Whether Λ *should* bound this windowed press reaction at all is Khadiv decision (e).)*

### 1.1 The blocker the deep-dive caught (and the fix)

The synthesis first proposed accumulating `qfrc_constraint` over the **full control step**. **Both adversarial critics independently flagged this as a blocker**, and it realigns with the upstream plan:

> The Z1 hammer XML runs an **active weld equality** (`z1_mocap` ↔ `ee_center_body`, `z1_mocap_hammer.xml:199-201`) — DiffIK drives the arm by setting the mocap pose and letting the weld drag the EE. `qfrc_constraint = Jᵀ·efc_force` sums over **all** active constraints (weld + joint-limits + contacts). The weld reaction is present on the arm DoFs at **every** substep and dominates the brief nail-impact spike. A full-step rectified sum therefore measures *"how hard the weld is dragging the arm"* (≈ gross commanded motion), **not** the impact impulse — defeating the premise.

**Resolution (design + critics + upstream docs now agree):**
1. **Contact-only masking** — isolate the hammer↔nail contact rows of `efc_force` (exclude `mjCNSTR_EQUALITY` weld and `mjCNSTR_LIMIT_JOINT`) before projecting to joint space; *or*, cheaper and weld-immune, **gate accumulation on the contact sensor** (already verified in `verify_contact_sensor.py`) so non-contact substeps contribute zero. Build the mask into the accumulator from day one — **not** a post-hoc "C0 audit".
2. **Contact-anchored window** — accumulate from the first-contact substep to the last-contact substep of a strike (per `../../archive/TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` Stage T4), **not** over the fixed 20 ms control grid. This also closes the *spread-across-time* evasion (see §4).

The C0 gate is then a true test, not an audit: **on a contact-free control step `Λ_j` must be ≈ 0**, and the weld contribution to arm DoFs must be below tolerance (assert, fail-not-warn).

---

## 2. Substep accumulation

- **Rate:** 500 Hz substep (`physics_dt = 0.002 s`, `decimation = 10`), **never** the 50 Hz control rate (which aliases the short impact). `tracking_impact_impulse_design_research.md:54`.
- **Where:** a `per_substep=True` MetricsTerm runs inside the decimation loop at `manager_based_rl_env.py:421-427` (`compute_substep()` after `sim.step()`+`scene.update()`), so `qfrc_constraint`/`efc_force` reflect the just-integrated substep — current, not stale. The full-step `CatSoftHook` runs after the loop (line 441), reading a fully-accumulated `Λ_j`. Same handshake as the proven `SubstepPeakJointVel → CaTJointVelConstraint(detection='substep')`.
- **Rectified** per-substep (`Σ|·|·dt`), not `|net integral|`: the post-impact response is a damped oscillation; rectified is the conservative bound. (Valid **only** on the contact-masked signal — rectifying the weld baseline would integrate a monotone bias.)
- **Reset:** contact-window open/close from the sensor; episode reset zeros the `(B,6)` buffer.

---

## 3. Code sketches

### 3.1 `src/tasks/hammer/mdp/impulse_bound.py` (NEW — mirror `velocity_bound.py` `SubstepPeakJointVel`)

> **This sketch is the original design** — the shipped `impulse_bound.py` supersedes it (TIME-based
> sliding-window Λ since 2026-07-13, replacing the 2026-07-05 per-event pulse; episode-cumulative
> capped delivered with debounced re-arm since 2026-07-14). The code is the authoritative semantics.

```python
# Per-joint contact-reaction impulse limit (N·m·s). C0 PLACEHOLDER — the term ships LOG-ONLY
# (max_p=0) until a hardware-validated threshold protocol is available. Historical
# Repeated-Peak arithmetic is retired; see IMPULSE_CAP_PROVENANCE.md.
Z1_JOINT_IMPULSE_LIMIT: float = 0.1            # placeholder — NEVER a silent default: joint_impulse_excess
                                               # requires an explicit limit, and CatSoftHook rejects this
                                               # VALUE under enforcement (imp_max_p>0)
_ARM_CFG = SceneEntityCfg("robot", joint_names=("joint1",...,"joint6"))
_ENV_SUBSTEP_IMPULSE_ATTR = "_hammer_substep_impulse"

class SubstepImpulseAccumulator(ManagerTermBase):   # per_substep=True
    # Buffer (B, 6) per-joint (matches joint_velocity_excess's (B,J) per-column convention so each
    # joint self-normalizes by its own EMA c_max). ACCUMULATES (+=) instead of peak-holding.
    def __init__(self, cfg, env):
        ...
        self.impulse = torch.zeros(env.num_envs, len(self._joint_ids), device=env.device)
        self._contact = env.scene["hammer_nail_contact"]    # sensor gate (weld-immune masking)
        setattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, self)
    def reset(self, env_ids): self.impulse[env_ids or slice(None)] = 0.0
    def __call__(self, env):
        in_contact = (self._contact.data.found > 0).any(-1)         # (B,) — gate
        # open window on rising contact; while in contact accumulate the CONTACT-only joint reaction:
        qfrc = self._robot.data._joint_dof_field("qfrc_constraint")[:, self._joint_ids]  # (B,6)
        # (preferred: project contact-only efc rows; sensor-gate is the cheap weld-immune fallback)
        self.impulse += torch.where(in_contact[:, None], qfrc.abs() * self._dt, 0.0)
        # close + reset window on contact release is handled by the contact-anchored bookkeeping
        return self.impulse.amax(dim=1)                              # (B,) for logging
```

> `qfrc_constraint` **does** exist on the mujoco_warp 3.8.1 `Data` and is sliceable via the existing `entity._joint_dof_field('qfrc_constraint')` helper (`data.py:237-240`) — the `velocity_bound.py:22` comment claiming it was inaccessible is **outdated** and already corrected in C0. Prefer adding a clean `qfrc_constraint` `@property` to `data.py` (mirror `qfrc_actuator` at `data.py:414-423`). `torque_source ∈ {'constraint','actuator'}` param kept for the documented ablation.

### 3.2 `src/tasks/hammer/cat/constraints.py` (add alongside `joint_velocity_excess`)

```python
def joint_impulse_excess(env, limit) -> torch.Tensor:   # limit REQUIRED; no default, no robot_cfg (the accumulator's own robot_cfg fixes the joint set)
    """Raw per-joint impulse margin Λ_j − limit, shape (B, J). Positive ⇒ over the limit.
    Reads the contact-anchored substep accumulator stashed on env. Returns the RAW signed margin
    only — CaT does all clamp/EMA. NEVER a probability, NEVER a reward term (hard constraint #1)."""
    acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
    if acc is None: raise RuntimeError("needs SubstepImpulseAccumulator in cfg.metrics")
    return acc.impulse - limit          # (B,J) − (J,) broadcasts to per-joint thresholds
```

### 3.3 `CatSoftHook` diff — a dedicated `_add_impulse_constraint` (sparse-signal per-column self-seeding normalizer + `imp_max_p==0` log-only short-circuit), everything else unchanged

```python
# impulse takes a DEDICATED path (not cat.add's shared batch-max EMA — that saturates on the sparse Λ signal):
self._add_impulse_constraint(env)   # writes cat.probs['joint_impulse_excess'] via a per-column
                                    # self-seeding normalizer; get_probs() then MAXes it in (soft-OR)
```

`get_probs()` MAXes over all terms and all columns (6 vel + 6 impulse = 12 columns → one per-env δ), so the soft-OR is automatic. `constraint_manager.py`, `cat_ppo.py`, `cat_storage.py`, `keys.py` are **untouched**.

---

## 4. The evasion modes the critics surfaced (and the gates that catch them)

| Evasion / failure | Mechanism | Gate / fix |
|---|---|---|
| **Weld pollution** (blocker) | `qfrc_constraint` dominated by the weld baseline | contact-only mask / sensor gate; C0 assert `Λ≈0` off-contact, weld < tol (fail-not-warn) |
| **Spread-across-time** (blocker) | per-control-step reset → split one impact across two 20 ms windows, each under threshold | **contact-anchored** window from the sensor (not the control grid); test: inject a boundary-straddling strike, assert δ fires |
| **Spread-across-joints** (major) | soft-OR MAX + per-column normalize → only worst joint drives δ; shuffle load off it | C4 reports a **chain-aggregate** (Σ\|Λ_j\| or Cartesian EE reaction impulse) alongside per-joint; if aggregate creeps while columns stay flat, add one aggregate column (`.add()`); echoes the known chain-coupled residual ([[z1-velocity-bound-finding]]) |
| **Sparse-signal EMA collapse** (major) | Λ≈0 between strikes → a batch-max EMA c_max would decay to the floor → δ saturates at max_p on every contact | **do not inherit the velocity EMA**: the shipped fix is CaT-style FIRST-VIOLATION per-column seeding (`cmax = max(first over-limit batch-max, imp_seed floor)`) with a per-column violation-masked EMA afterwards; `imp_seed` is a decay FLOOR only (any small value safe), NOT a p95/excess statistic; C2 plots δ-vs-excess and asserts it is *graded*, not saturated |
| **Vacuous limit** (major) | δ≈0 is ambiguous: "safe" vs "J_limit too loose"; placeholder 0.1 would silently ship | **binding-ness gate** (C1/C2): report fraction of strikes in `[0.8·J_limit, J_limit]` and δ-attributable peak-impulse reduction vs the same-seed unconstrained baseline; ≈0 reduction + δ≈0 ⇒ vacuous, fail |
| **Wrong quantity** (major) | `qfrc_constraint` (impact reaction) vs `qfrc_actuator` (transmitted motor effort) under-justified on a position-servo + weld arm; the Repeated-Peak rating bounds *transmitted joint torque* | promote the **Pinocchio `impulseDynamics` (r_coeff=0) cross-check** from a C4 validation to a **C0 quantity-selection GATE** *(as designed — at C0 the object-side ∫F·dt was accepted as ground truth instead; **Pinocchio decision RECORDED 2026-07-06 as deferred indefinitely — see the Status line at the top of this doc, no longer a "revisit before `max_p>0`" open item**)* |

---

## 5. Staged plan (mirrors the velocity plan's C0–C5 — the two plans' stage numberings are separate namespaces)

- **C0 — accumulator + log-only + physics probe + quantity gate.** Implement `SubstepImpulseAccumulator` (contact-gated) + `joint_impulse_excess`; wire LOG-ONLY (`max_p=0`) behind a `cat_impulse` flag; correct `velocity_bound.py:22`. Write `derive_impulse_thresholds.py` / `log_strike_window.py`: probe the open-loop reference strike → per-joint contact-impulse histogram (real magnitudes; **weld contribution asserted < tol**; Λ≈0 off-contact). **The MuJoCo-native object-side ∫F·dt is the accepted ground truth; Pinocchio `impulseDynamics` cross-check is DEFERRED INDEFINITELY (not installed — decision RECORDED 2026-07-06, not pending; see the Status line at the top of this doc).** *Tests:* accumulator sums over the contact window, zeros at boundaries, shape (B,6); off-contact ≈ 0; margin never a probability.
- **C1 — pre-train gate (SHIPPED — as Phase M; the letter L went to the overshoot clamp).** The shipped `validate_rewards.py` Phase M certifies the M1–M4 invariants: Λ_j logged nonzero on a strike; `cat_delta` published and identically 0 under `max_p=0` (proves log-only is a true no-op); the delivered-impulse reward fires; the raw-margin identity (Λ−limit) holds at nonzero Λ. The **binding-ness metric** shipped in `derive_impulse_thresholds.py` (the quantity gate), not in Phase M. All existing phases still pass; `verify_contact_sensor.py` + `verify_reward_setup.py` green.
- **C2 — smoke + normalizer/max_p gate (CURRENTLY DEFERRED; the 2026-07-10 result is historical).** Keep `imp_max_p=0.0` until an authorized calibration protocol re-establishes this gate; deterministic direct-reference repeats are insufficient. The gate must show c_max(impulse) bounded, not pinned at 1e-6 or single-event-spiked, and δ-vs-excess **graded**, not saturated. Only then may a mini-sweep choose a robust statistic / τ while preserving the strike skill.
- **C3 — full GPU results run (BLOCKED on current C2 authorization).** After C2 is re-established, run `Unitree-Z1-Hammer-CaT-Impulse` with its authorized `max_p`/τ and real per-joint `J_limit`; compare vs `r_imit`-only and the existing CaT-velocity arm on the same seed budget.
- **C4 — peak-impulse eval** (impulse analogue of `diag_policy_trace` peak-qv). Per-joint **and chain-aggregate** peak impulse (max/p95/worst-joint); delivered nail impulse retained; Pinocchio `impulseDynamics` cross-check within tolerance.
- **C5 — combined soft-OR composition (SHIPPED at flag level: `cat_soft=True` + `cat_impulse=True` composes into ONE hook — velocity ∪ impulse soft-OR, 12 columns — unit-tested; no combined task id is registered yet).** Still to do at results time: report the **column-contribution (argmax) histogram** to show the impulse column actually drives terminations and is not masked by the correlated velocity column under soft-OR saturation; scaffold (not train) the escalation ladder beyond soft-CaT (excess-penalty / episodic-sum PID-Lagrangian CMDP) as future work.

**Run isolation:** impulse runs as a **separate arm** (`Unitree-Z1-Hammer-CaT-Impulse`), not soft-OR'd with velocity, for clean attribution — velocity and impulse are physically correlated (both ∝ `m_eff·v`), so a combined arm can't attribute the safety gain. Combined is the trivial last experiment.

---

## 6. Net-new literature (verified this session; not in `../hammering_reward_design_deep_dive_v2.md`)

> **Gap found:** the v2 deep-dive cites generic constrained-RL but **never cites CaT itself** — the mechanism the thesis is built on.

1. **CaT: Constraints as Terminations for Legged Locomotion RL** — Chane-Sane et al., IROS 2024, arXiv:2403.18765 — *the* foundational mechanism (must-cite).
2. **Constrained RL for Unstable Point-Feet Bipedal Locomotion / Bolt** — Roux, Chane-Sane et al., Humanoids 2025, arXiv:2508.02194 — same-group CaT follow-up to **real hardware** + DR (the sim-to-real story).
3. **A Contact-Safe RL Framework for Contact-Rich Manipulation** — Zhu, Kang, Chen, IROS 2022, arXiv:2207.13438 — controller-shield (VIC) alternative; the explicit baseline.
4. **Bresa: Bio-inspired Reflexive Safe RL for Contact-Rich Tasks** — Zhang, Solak, Ajoudani, RA-L 2025, arXiv:2503.21989 — higher-frequency safety critic ≈ substep-rate enforcement.
5. **Learning Impact-Rich Rotational Maneuvers (one-leg hopper flip)** — Kang et al., 2025, arXiv:2505.12222 — impact-rich momentum + transmission-load regularization within actuator limits.
6. **Whole-Body Constrained Learning via Hierarchical Optimization** — Wang et al., RA-L 2025, arXiv:2506.05115 — hard+soft constraints incl. excessive torque via RL+HQP (the QP-shield counterpoint).
7. **Constraints as Rewards** — Ishihara et al., 2025, arXiv:2501.04228 — CaT-adjacent; constraint-vs-reward-vs-termination framing.
8. **Towards a Practical Understanding of Lagrangian Methods in Safe RL** — Spoor et al., 2025, arXiv:2510.17564 — λ sensitivity; *empirical* case for preferring CaT's scale-free termination over PPO-Lagrangian (the impulse mechanism-choice defense).
9. **Higher-Order Action Regularization (jerk)** — Ahmed et al., NeurIPS-W 2025/2026, arXiv:2601.02061 — jerk as impulse-onset proxy; analogy only (different domain).

**Novelty note:** no found RL paper bounds *per-joint impulse* (vs force/torque/momentum) accumulated at substep rate — a genuine gap supporting the thesis claim. State it; don't over-cite a near-miss.

---

## 6b. Start-pose / grasp cleanliness (investigated 2026-06-18)

The Z1 reset pose (`NEAR_NAIL_JOINT_POS`, `z1_constants.py:184`) looks out-of-plane. Investigated with `solve_ik.py` (the DLS solver that produced it). It decomposes into three parts:

| solve | joint1 (base yaw) | strike-axis tilt from vertical | face at nail | within limits |
|---|---|---|---|---|
| current `NEAR_NAIL` (position-only IK) | −6.9° | **7.4° (oblique)** | ✓ | — |
| lock joint1=0, **position-only** IK | 0° | 13.0° (worse) | ✓ | — |
| joint1 **free**, orientation-aware IK | −6.2° | 0.05° (vertical) | ✓ | ✓ |
| **lock joint1=0, orientation-aware IK** | **0.0°** | **0.00° (vertical)** | ✓ (0.010 cm) | ✓ |

- **Oblique 7.4° strike — FIXABLE, no re-grasp.** `NEAR_NAIL` was solved *position-only*, so the strike-axis orientation was incidental. An **orientation-aware IK** (position + strike-axis → world −Z) yields a dead-vertical strike.
- **In-plane (`joint1=0`) IS achievable — *correction* to the earlier "grasp-forced" claim.** Locking `joint1=0` and re-solving with the *orientation-aware* IK converges to a fully **in-plane + perpendicular (0.00° tilt) + on-nail** pose, within limits: `{j1=0, j2=114.1, j3=−102.9, j4=78.8, j5=0, j6=94.5}°`. The **wrist** absorbs the lateral face offset (`joint6` → 94.5° vs 70.5°), so the base need not yaw. The earlier −6° was a *position-only-IK artifact* (it never used the wrist to compensate), NOT a grasp constraint. **No re-grasp needed.**
- **Wrist twist** (`joint4≈79°, joint6≈94°`) remains — the sideways-claw-grasp cost; benign.

**Action — DONE for the gripper-era EE (2026-06-18, commit `1f4a6f8`, sibling `hammer_z1_env/solve_ik_oriented.py`):** `NEAR_NAIL_JOINT_POS` re-solved via the orientation-aware IK with `joint1` locked to 0 → an **in-plane, perpendicular, on-nail** reset pose (drop-in replacement; no re-grasp, no action-space change, no hand-tuning), re-verified with `playback_reference.py` (note: that script's "PHASE M GATE" print is its own legacy reference-strike label — unrelated to `validate_rewards.py` Phase M) + `validate_rewards.py`. The solve table above is gripper-era. **The L6 fixture EE re-solve is DONE (2026-07-06):** `NEAR_NAIL_JOINT_POS` adopted the user's windup-sweep pose — reset poised at world `(0.5, 0, 0.25)`, in-plane (`joint1=0`), dead-vertical, ~15 cm above the floor-level nail — because a perfectly vertical strike *at* the floor nail is kinematically infeasible with the new grasp (in-plane face-down only becomes reachable above z≈0.20); the policy/reference discovers the necessarily oblique near-floor contact angle itself. **Gate re-greened 2026-07-10:** `playback_reference.py` (shaped reference) PASSES at all three approach heights (best 27.0 mm); the crude constant-vertical `test_single_strike.py` probe FAILS from this pose (max 13.3 mm) and is **retired** for this grasp (see `OPEN_QUESTIONS.md` Q1). That dated feasibility result is not current C2/C3 authorization; see the current status correction above. The diagnostic comparison runs (a_base / c_a3_cat / c_hardterm / soft-CaT) used the old pose but were only for *choosing the enforcement path* (→ soft-CaT), so no re-train is owed.

> **Future robustness arm (orientation control + Vicon sim-to-real):** once the nail is no longer fixed-vertical, the strike must align to the actual board/nail normal — a **6-DoF DiffIK action** + board-tilt domain randomization, with a Vicon-only state scheme (board-cluster normal + a nail-shaft tracking ball for depth, no perception). Full spec: `ORIENTATION_ROBUST_SIM2REAL_ARM.md`. Build **after** the fixed-impedance impulse result.

## 7. Open questions

- Is soft-CaT (not CMDP/Lagrangian) the agreed impulse enforcement? (§ top — confirm with Khadiv.)
- `J_limit` per joint (Harmonic-Drive Repeated-Peak × duration) — derive in C0, not invent.
- Contact window length / single-strike vs cyclic (depends on Q1 single-strike feasibility).
- Per-joint vs summed cost for any future Lagrangian rung.
- ~~Pinocchio `impulseDynamics` is named in the docs but **unbuilt** in the Z1 task~~ — **RESOLVED 2026-07-06:** was a C0 prerequisite as designed; at C0 the object-side ∫F·dt was accepted as ground truth instead, and this is now a **recorded decision** — `impulseDynamics` is deferred indefinitely (mirrors the 2026-06 decision in visual plan `plan-c00a0e0144274fe4`), not an open item. See the Status line at the top of this doc.
