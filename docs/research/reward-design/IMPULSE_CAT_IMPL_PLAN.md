# Impulse-CaT implementation plan — extending faithful soft γ(1−δ) CaT from joint-velocity to per-joint impact impulse

**Status:** design (2026-06-18), pre-implementation. Produced by a multi-agent deep-dive (4 readers → synthesis → 2 adversarial critics) over the thesis impulse-design docs, the implemented soft-CaT code, the mjlab substep machinery, and external literature.

**Relationship to other docs:**
- `FAITHFUL_SOFT_CAT_IMPL_PLAN.md` — the soft-CaT machinery this reuses (CaT δ-math, `CatPPO` scale-positives + dual-mask GAE, `CatSoftHook`). All of it is **unchanged** by this plan.
- `TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md` (stages T0–T5) — the *prior* impulse plan, written before soft-CaT was chosen. It mandates a **contact-anchored window** (T4:95) and prescribes an **excess-penalty → PID-Lagrangian CMDP** enforcement ladder. This plan supersedes the *enforcement-mechanism* portion (soft-CaT instead of penalty/Lagrangian) and inherits its quantity/windowing/threshold decisions.
- `tracking_impact_impulse_design_research.md` — the cited research (D1–D6); the quantity `Λ_j = Σ|qfrc_constraint_j|·h` over a contact-anchored window comes from SQ3 (line 99).

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

**Why this is impulse-not-energy:** it is a time-integral of *force* (momentum transfer, ∝ `m_eff·Δv`), not force·velocity (energy). It bounds the mechanical shock the harmonic drive absorbs, invariant to *how* the velocity jump happens. An energy bound would let the policy trade a brief huge force-spike for a long gentle push of equal energy — the opposite of gearbox protection.

### 1.1 The blocker the deep-dive caught (and the fix)

The synthesis first proposed accumulating `qfrc_constraint` over the **full control step**. **Both adversarial critics independently flagged this as a blocker**, and it realigns with the upstream plan:

> The Z1 hammer XML runs an **active weld equality** (`z1_mocap` ↔ `ee_center_body`, `z1_mocap_hammer.xml:199-201`) — DiffIK drives the arm by setting the mocap pose and letting the weld drag the EE. `qfrc_constraint = Jᵀ·efc_force` sums over **all** active constraints (weld + joint-limits + contacts). The weld reaction is present on the arm DoFs at **every** substep and dominates the brief nail-impact spike. A full-step rectified sum therefore measures *"how hard the weld is dragging the arm"* (≈ gross commanded motion), **not** the impact impulse — defeating the premise.

**Resolution (design + critics + upstream docs now agree):**
1. **Contact-only masking** — isolate the hammer↔nail contact rows of `efc_force` (exclude `mjCNSTR_EQUALITY` weld and `mjCNSTR_LIMIT_JOINT`) before projecting to joint space; *or*, cheaper and weld-immune, **gate accumulation on the contact sensor** (already verified in `verify_contact_sensor.py`) so non-contact substeps contribute zero. Build the mask into the accumulator from day one — **not** a post-hoc "C0 audit".
2. **Contact-anchored window** — accumulate from the first-contact substep to the last-contact substep of a strike (per `TRACKING_IMPACT_IMPULSE_IMPL_PLAN.md:95`), **not** over the fixed 20 ms control grid. This also closes the *spread-across-time* evasion (see §4).

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

```python
# Per-joint contact-reaction impulse limit (N·m·s). C0 PLACEHOLDER — the term ships LOG-ONLY
# (max_p=0) until derive_impulse_thresholds.py emits real Harmonic-Drive Repeated-Peak × duration
# values (≈2× rated torque, 1e4-event fatigue budget). NOT a magic constant to ship.
Z1_JOINT_IMPULSE_LIMIT: float = 0.1            # placeholder
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

> `qfrc_constraint` **does** exist on the mujoco_warp 3.8.1 `Data` and is sliceable via the existing `entity._joint_dof_field('qfrc_constraint')` helper (`data.py:237-240`) — the `velocity_bound.py:22` comment ("not exposed on the Entity") is **outdated**; correct it in C0. Prefer adding a clean `qfrc_constraint` `@property` to `data.py` (mirror `qfrc_actuator` at `data.py:414-423`). `torque_source ∈ {'constraint','actuator'}` param kept for the documented ablation.

### 3.2 `src/tasks/hammer/cat/constraints.py` (add alongside `joint_velocity_excess`)

```python
def joint_impulse_excess(env, limit=Z1_JOINT_IMPULSE_LIMIT, robot_cfg=_ARM_CFG) -> torch.Tensor:
    """Raw per-joint impulse margin Λ_j − limit, shape (B, J). Positive ⇒ over the limit.
    Reads the contact-anchored substep accumulator stashed on env. Returns the RAW signed margin
    only — CaT does all clamp/EMA. NEVER a probability, NEVER a reward term (hard constraint #1)."""
    acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR, None)
    if acc is None: raise RuntimeError("needs SubstepImpulseAccumulator in cfg.metrics")
    return acc.impulse - limit          # (B,J) − (J,) broadcasts to per-joint thresholds
```

### 3.3 `CatSoftHook` diff — **two lines**, everything else unchanged

```python
c_imp = joint_impulse_excess(env, limit=self._imp_limit, robot_cfg=self._robot_cfg)
self._cat.add("joint_impulse_excess", c_imp, max_p=self._imp_max_p)   # get_probs() already MAXes
```

`get_probs()` MAXes over all terms and all columns (6 vel + 6 impulse = 12 columns → one per-env δ), so the soft-OR is automatic. `constraint_manager.py`, `cat_ppo.py`, `cat_storage.py`, `keys.py` are **untouched**.

---

## 4. The evasion modes the critics surfaced (and the gates that catch them)

| Evasion / failure | Mechanism | Gate / fix |
|---|---|---|
| **Weld pollution** (blocker) | `qfrc_constraint` dominated by the weld baseline | contact-only mask / sensor gate; C0 assert `Λ≈0` off-contact, weld < tol (fail-not-warn) |
| **Spread-across-time** (blocker) | per-control-step reset → split one impact across two 20 ms windows, each under threshold | **contact-anchored** window from the sensor (not the control grid); test: inject a boundary-straddling strike, assert δ fires |
| **Spread-across-joints** (major) | soft-OR MAX + per-column normalize → only worst joint drives δ; shuffle load off it | C4 reports a **chain-aggregate** (Σ\|Λ_j\| or Cartesian EE reaction impulse) alongside per-joint; if aggregate creeps while columns stay flat, add one aggregate column (`.add()`); echoes the known chain-coupled residual ([[z1-velocity-bound-finding]]) |
| **Sparse-signal EMA collapse** (major) | Λ≈0 between strikes → batch-max EMA c_max decays to the 1e-6 floor → δ saturates at max_p on *every* contact (hard lottery, kills graded δ) | **do not inherit the velocity EMA**: mask the EMA update to contact steps / use a robust statistic (p95 of recent contact events) / **seed c_max from the C0 reference histogram**; C2 plots δ-vs-excess and asserts it is *graded*, not saturated |
| **Vacuous limit** (major) | δ≈0 is ambiguous: "safe" vs "J_limit too loose"; placeholder 0.1 would silently ship | **binding-ness gate** (C1/C2): report fraction of strikes in `[0.8·J_limit, J_limit]` and δ-attributable peak-impulse reduction vs the same-seed unconstrained baseline; ≈0 reduction + δ≈0 ⇒ vacuous, fail |
| **Wrong quantity** (major) | `qfrc_constraint` (impact reaction) vs `qfrc_actuator` (transmitted motor effort) under-justified on a position-servo + weld arm; the Repeated-Peak rating bounds *transmitted joint torque* | promote the **Pinocchio `impulseDynamics` (r_coeff=0) cross-check** from a C4 validation to a **C0 quantity-selection GATE** — it is the only ground truth for "is this the real impact impulse" |

---

## 5. Staged plan (mirrors the C0–C5 that worked for velocity)

- **C0 — accumulator + log-only + physics probe + quantity gate.** Implement `SubstepImpulseAccumulator` (contact-gated) + `joint_impulse_excess`; wire LOG-ONLY (`max_p=0`) behind a `cat_impulse` flag; correct `velocity_bound.py:22`. Write `derive_impulse_thresholds.py` / `log_strike_window.py`: probe the open-loop reference strike → per-joint contact-impulse histogram (real magnitudes; **weld contribution asserted < tol**; Λ≈0 off-contact). **Pinocchio cross-check decides the quantity here**, not later. *Tests:* accumulator sums over the contact window, zeros at boundaries, shape (B,6); off-contact ≈ 0; margin never a probability.
- **C1 — pre-train gate (Phase L).** Add a `validate_rewards.py` Phase L: Λ_j logged nonzero on a normal strike, excess ≈ 0 on a gentle/reference strike, positive on an injected violent strike; δ identically 0 under `max_p=0` (proves log-only is a true no-op); **binding-ness metric** present. All existing phases still pass; `verify_contact_sensor.py` + `verify_reward_setup.py` green.
- **C2 — smoke + normalizer/max_p gate (CPU/tiny GPU).** Turn on `max_p` with the real `J_limit`. **Hard gate** (not "watch"): c_max(impulse) bounded, not pinned at 1e-6, not single-event-spiked; δ-vs-excess **graded**, not saturated. Mini-sweep `max_p ∈ {0.25, 0.5}`; pick robust statistic / τ. Strike skill still forms.
- **C3 — full GPU results run (one GPU).** `Unitree-Z1-Hammer-CaT-Impulse`, chosen `max_p`/τ, real per-joint `J_limit`. Compare vs `r_imit`-only and the existing CaT-velocity arm on the same seed budget.
- **C4 — peak-impulse eval** (impulse analogue of `diag_policy_trace` peak-qv). Per-joint **and chain-aggregate** peak impulse (max/p95/worst-joint); delivered nail impulse retained; Pinocchio `impulseDynamics` cross-check within tolerance.
- **C5 — combined soft-OR arm (optional, last).** velocity ∪ impulse in one CaT instance (one `.add()`; 12 columns). Report the **column-contribution (argmax) histogram** to show the impulse column actually drives terminations and is not masked by the correlated velocity column under soft-OR saturation — otherwise the composition claim is unsupported. Scaffold (not train) the escalation ladder beyond soft-CaT (excess-penalty / episodic-sum PID-Lagrangian CMDP) as future work.

**Run isolation:** impulse runs as a **separate arm** (`Unitree-Z1-Hammer-CaT-Impulse`), not soft-OR'd with velocity, for clean attribution — velocity and impulse are physically correlated (both ∝ `m_eff·v`), so a combined arm can't attribute the safety gain. Combined is the trivial last experiment.

---

## 6. Net-new literature (verified this session; not in `hammering_reward_design_deep_dive_v2.md`)

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

**Action:** re-solve `NEAR_NAIL_JOINT_POS` via the orientation-aware IK with `joint1` locked to 0 → an **in-plane, perpendicular, on-nail** reset pose (drop-in replacement; no re-grasp, no action-space change, no hand-tuning). Re-verify with `playback_reference.py` (Phase M — strike still drives the nail) + `validate_rewards.py` before training on it. The diagnostic comparison runs (a_base / c_a3_cat / c_hardterm / soft-CaT) used the old pose but were only for *choosing the enforcement path* (→ soft-CaT), so no re-train is owed.

> **Future robustness arm (orientation control + Vicon sim-to-real):** once the nail is no longer fixed-vertical, the strike must align to the actual board/nail normal — a **6-DoF DiffIK action** + board-tilt domain randomization, with a Vicon-only state scheme (board-cluster normal + a nail-shaft tracking ball for depth, no perception). Full spec: `ORIENTATION_ROBUST_SIM2REAL_ARM.md`. Build **after** the fixed-impedance impulse result.

## 7. Open questions

- Is soft-CaT (not CMDP/Lagrangian) the agreed impulse enforcement? (§ top — confirm with Khadiv.)
- `J_limit` per joint (Harmonic-Drive Repeated-Peak × duration) — derive in C0, not invent.
- Contact window length / single-strike vs cyclic (depends on Q1 single-strike feasibility).
- Per-joint vs summed cost for any future Lagrangian rung.
- Pinocchio `impulseDynamics` is named in the docs but **unbuilt** in the Z1 task — C0 prerequisite.
