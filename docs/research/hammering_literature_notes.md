# Hammering × Robot Learning — Field Notes & Annotated Bibliography

**Purpose:** a niche-specific running bibliography for the (very small) intersection of **robot learning** and **hammering / percussive nail-driving**. Companion to `hammering_reward_design_deep_dive_v2.md`. Grows as sources from `~/Desktop/_Munich_hammering/` and the web are read.

**Tier legend** (same as v2): `[E1]` peer-reviewed, verified · `[E2]` preprint verified · `[E2*]` reported, not re-fetched · `[E3]/gray` gray literature / non-peer-reviewed · dates marked *(uncertain)* where the source gives none.

---

## 1. State of the field (why this niche is worth its own file)

**Finding: robot *learning* for hammering is nearly empty; hammering *control/optimization* is where the work is.** Across the v2 external sweep + this folder, the hammering-specific literature clusters as:

- **Impact / momentum control & optimization (the bulk):** QP-based impact-momentum maximization for hammering (Vu et al. 2026 `[E1]`, AMC — **read, §2.3; the strongest direct match to the v2 "momentum not force" thesis**); optimal-control tool-affordance for nail driving (Ti et al. 2024 `[E2]`); flexible-link hammer control (Hitaka & Izumi 1995; Izumi & Zhou 1999); "multiple working mode" modular-robot hammering. These treat the hammer as a dynamics/optimization problem, **not** a learned policy.
- **Striking RL that *mentions* hammering:** ARMADA (Kim et al. 2025 `[E3]`) lists hammering among striking/snatching tasks; dynamic-manipulation RL (TossingBot `[E1]`, table tennis `[E3]`) is adjacent but not nail-driving.
- **Actual RL-for-hammering:** MORE than first thought. Beyond the Tandon toy thesis, hammering is a **standard RL benchmark** — the **Adroit "hammer" task** (Rajeswaran et al. 2018) and its **D4RL `hammer-v0`** offline-RL datasets (Fu et al. 2020); see §2.6. But these frame hammering as **sparse-reward dexterous tool-use with a 24-DoF hand**, not as impact/momentum dynamics on an arm.

**Implication for a paper/thesis intro (revised after finding Adroit/D4RL).** Do **not** claim "RL can't hammer" — it can, and is benchmarked (Adroit/D4RL `hammer`, §2.6). The defensible gap is narrower and sharper: **RL hammering as an IMPACT / momentum / repetitive-strike problem on a position- or torque-controlled manipulator, with sim-to-real**, is under-explored. The dexterous-hand benchmark presses a nail quasi-statically with a 24-DoF hand; model-based hammering work (momentum maximization, impact-aware control) has the impact physics but no learning. The Z1 project sits in the **intersection**: *learning* + *impact dynamics* + *arm* + *sim-to-real* + *repetition*. Frame the gap as physics + embodiment, not "nobody has learned to hammer."

---

## 2. Annotated entries

### 2.1 Tandon senior thesis — `[E3]/gray, date uncertain (~c. 2010)`

**Tandon, P. (n.d., ~2009–2012).** *Hitting the Nail on the Head: An Investigation of Robot Learning.* Senior Thesis, advisor Prof. Michael Arbib (USC). File: `~/Desktop/_Munich_hammering/SeniorThesis.pdf` (20 pp.).
*Dating note:* no year on the PDF; internal references run to 2004 (Johnson-Frey) / 2002 (Stanley & Miikkulainen), and the hobby-grade Lynxmotion AL5D + WordPress CV place the hardware era ~2009–2012. **Treat the year as uncertain; do not cite a specific year without confirming.**

**What it is.** Teaches a robot to drive nails into a board over a long sequence (250 nails, 30 trials) without bending them. Compares a tabular **Q-Learning** controller vs a biologically-inspired **population-coded neural network**. Abstraction is high-level: the policy outputs **2 parameters per swing** — `swing_theta` (backswing angle) and angular `acceleration` — and an **analytical ballistic arm + 2-D nail model** executes the strike and returns visual feedback `(amount_into_board, amount_bent)`.

**Reward function (verbatim structure).**
```
R(s) = A_CONST · amountIntoBoard − B_CONST · amountBent      (per successful swing)
R(s) = FAILURE_COST                                          (if nail bent beyond repair)
```
State `s = (Nail_AmountIntoBoard, Nail_AmountBent)`; actions `a = (swing_theta, acceleration)`.

**Headline results.**
- Q-Learning: oscillates, **no net efficiency gain** (15.5 → 15.5 swings/nail) but **very reliable** (success:fail ≈ 276; rarely bends a nail).
- Population-coded: **+37% efficiency** (8.13 → 5.1 swings/nail), generalizes & transfers across nail-head sizes, but **less reliable** (success:fail ≈ 4; weights can overflow → failures).
- **Fitt's Law emerges:** with velocity-dependent positional noise, the controller *learns to lower peak velocity for larger nails* to avoid off-center hits and bending — keeping final bend ~constant across nail sizes despite higher bend risk.

**Tier / use:** gray literature, toy physics, dated methods (tabular Q / hand-coded pop-codes), hobby-arm prototype "work in progress." **Weak as a citation; valuable as an idea source and as a "the niche is sparse" data point.**

### 2.2 Foundational hammering-control works it surfaces — *cited by Tandon; VERIFY before use*

These appear in Tandon's reference list and are worth chasing for a proper lit review (older, hammering-specific control). **Not independently verified here** — confirm authors/venues before citing.
- **Hitaka, Y., & Izumi, T. (1995).** Minimum energy driving of a flexible-link hammer using neural networks. *IEEE/RSJ IROS 1995.*
- **Izumi, T., & Zhou, H. (1999).** Hitting robot with a flexible-link hammer. *Artificial Life and Robotics, 3*(2), 73–78.
- **Karniel, A., & Inbar, G. F. (1997).** A model for learning human reaching movements. *Biological Cybernetics, 77*(3), 173–183. *(ballistic-movement characterization Tandon builds on.)*

### 2.3 Vu et al. — QP-based impact-momentum maximization for hammering — `[E1], verified by direct read`

**Vu, J., Erens, R., Stefanelli, H., Cisneros-Limón, R., Benallegue, M., & Benallegue, A. (2026).** *QP-based impact momentum maximization for a hammering task by a humanoid robot.* IEEE 19th International Conference on Advanced Motion Control (AMC 2026). DOI:10.1109/AMC67705.2026.11435814. CNRS-AIST JRL (Tsukuba) + Univ. Versailles. Robot: **HRP-5P**, simulated in **mc_mujoco / mc_rtc** (torque control). File: `~/Desktop/_Munich_hammering/QP-based_impact_momentum_maximization_for_a_hammering_task_by_a_humanoid_robot.pdf`.

**This is the closest peer-reviewed match to the v2 report's central reward thesis — and it independently derives the same quantity.**

**Core machinery (verbatim).**
```
Mobility matrix:        Λ(q) = Jν(q) M(q)⁻¹ Jν(q)ᵀ            (operational-space inverse inertia, translational part)
Effective Mass (EMR):   m_e,n(q) = ( nᵀ Λ(q) n )⁻¹            (effective robot mass along impact normal n)
Projected momentum:     P(t) = m_e,n(q) · ẋνᵀ n
Impact impulse:         i = ∫_{tc}^{tc+δt} nᵀ Fʰ dt = ΔP      (maximize impulse ⇔ maximize EMR at fixed target velocity)
EMMT objective (QP):    arg min_q̈  −∇m_e,nᵀ(q) q̈   s.t. joint position, velocity, torque limits
Full QP cost:           arg min_q̈  ½ Wp‖q̈ + b‖² − Wm ∇m_e,nᵀ(q) q̈ + A(q̈),   b = Kp(q−qd)+Kd(q̇−q̇d)−q̈d,  Kd = 2√Kp
Trajectory:             5th-order Bézier curve for the hammer tip (jerk-bounded, smooth from any reachable pose)
Orientation:            "collinear impact" — align hammer-tip normal with nail axis (constrain roll+pitch, yaw free) via a Vector Orientation Task
```

**Why momentum, not force or energy (their Section III-C — the citable justification for v2's "never reward contact force").** Force-based model fails because the hammer-tip velocity is *discontinuous* at impact, so `dẋ/dt` at `tc` does not exist. Energy-based model loses energy to heat/sound/deformation, hard to estimate. **Momentum-based model handles discontinuous velocities and folds the microscopic interaction into the scalar coefficient of restitution `e*`.** → exactly the v2 argument, now with a rigorous mechanics basis (they follow Stronge, *Impact Mechanics*, 2018).

**Motivation = recoil/joint-stress damage mitigation.** Frames hammering as offloading "repetitive, load-bearing operations that expose workers to cumulative joint stress," and explicitly aims to "mitigate the recoil damage on the robot's joints." Direct support for v2 §8 (actuator protection, repeated-shock budget).

**What to steal for the position-only Z1 (and the caveat).**
- **Reward target:** the EMR `m_e,n(q)` is computable from `mjData` (`Jν`, `M`) — use it as the v2 `dir_manip`/`impact_momentum` term, and **reward `m_e,n(q)·v_axial` at first contact**. This paper is the `[DIRECT]` citation that replaces v2's analogy-based momentum argument.
- **Caveat (reinforces v2 §2.3):** their EMMT and orientation alignment are realised by a **torque-level QP with joint-torque constraints** and **commanded orientation**. The Z1's position-only DiffIK can *reward* effective mass and *reward* axial velocity, but cannot *optimize* them online via QP, and cannot command the collinear-impact orientation without an action-space change. So: steal the reward target, not the controller.
- **Their difference from Wang & Kheddar (2019, RSS):** Wang & Kheddar maximize impact momentum with a QP but treat effective mass as **constant**; Vu et al. make **effective-mass maximization part of the objective**. (Useful nuance: there are two impact-momentum-QP lineages.)

**Cross-relevance:** uses **mc_mujoco** (Singh, Gergondet & Kanehiro 2022, arXiv:2209.00274), so the sim substrate is MuJoCo — directly comparable to mjlab.

### 2.4 Field debate surfaced — does hammering obey Fitt's Law?

A genuine, citable disagreement worth knowing (and flagging in any thesis/paper intro):
- **Tandon (§2.1, gray):** claims his learned swing controller exhibits **Fitt's Law** — a speed–accuracy trade-off — learning to slow down for larger nails to avoid bending.
- **Petrič, T., et al. (2017). "Hammering Does Not Fit Fitt's Law." *Frontiers in Computational Neuroscience, 11*.** *(cited as ref [4] by Vu et al.; title only — NOT yet read/verified here.)* Argues the opposite for human hammering.
→ **Action:** read Petrič 2017 before leaning on any Fitt's-Law framing for impact reward design; the speed–accuracy idea (idea #2 below) may need to be motivated by velocity-dependent contact noise *empirically*, not by appeal to Fitt's Law.

**Other references Vu et al. surface (worth chasing for the lit review; verify before citing):**
- Wang, Y., & Kheddar, A. (2019). Impact-Friendly Robust Control Design with Task-Space Quadratic Optimization. *RSS 2019.* — impact-friendly control / robot damage mitigation.
- van Steen, J., van de Wouw, N., & Saccon, A. (2022). Robot Control for Simultaneous Impact Tasks via QP-based Reference Spreading. *ACC 2022.* arXiv:2111.05211. *(ACC version of the reference-spreading line cited in v2.)*
- Tassi, F., et al. (2025). IMA-catcher: An IMpact-aware Nonprehensile Catching Framework… *Int. J. Robotics Research.*
- Cisneros, R., et al. (2025). Impulsive Pedipulation of a Spherical Object… 3D Targeted Kicking Motion Generator. *Int. J. Humanoid Robotics.*
- Stronge, W. J. (2018). *Impact Mechanics* (2nd ed.). Cambridge Univ. Press. — foundational impact-mechanics text (restitution, momentum model).
- Singh, R. P., Gergondet, P., & Kanehiro, F. (2022). mc-mujoco: Simulating Articulated Robots with FSM Controllers in MuJoCo. arXiv:2209.00274.

### 2.5 Romanyuk, Soleymanpour & Liu — Multiple Working Mode hammering (MRR) — `[E1] control, not learning`

**Romanyuk, V., Soleymanpour, S., & Liu, G. (2019).** A Multiple Working Mode Approach to Hammering with a Modular Reconfigurable Robot. *Proc. 2019 IEEE Int. Conf. Mechatronics and Automation (ICMA)*, Tianjin, pp. 774–779. Ryerson Univ. (Aerospace Eng.). *(A 13-pp expanded version is also in the folder, unread.)*

**What it is.** **Control, not learning.** Lets a modular reconfigurable robot hammer nails in 3D *without* special hardware (no VSA/VIA, no flexible hammer) by switching selected joints into a **passive working mode** at impact so they free-rotate and don't accumulate damaging internal joint impulse. Analytic external- (restitution) + internal- (Newton–Euler) impulse models + an effective-mass model decide *which* joints to passivate and *when* (timed by link momentum `P=M(q)q̇` and EE-vs-nail pose projection). Validated on a 3-DOF arm hammering styrofoam + hardwood.

**Why it still matters for the Z1 (despite control + different robot).**
- **Gear-fatigue limits = a concrete basis for v2's "repeated shock accumulation" budget.** Repeated impacts are bounded by the harmonic drive's **Repeated Peak Torque (RPT, infinite fatigue life)** vs **Momentary Peak Torque (MPT, emergency only)**; a joint predicted to exceed RPT (but below MPT) is passivated. → adds a **gear-fatigue (RPT) constraint** to v2 §8 alongside the I²t thermal proxy.
- **Position control is the *worst case* for impact safety — confirms the v2 action-space critique.** Their data shows a position-controlled joint *fighting* the impact (motor reverses to recover its reference), twisting the harmonic-drive bearing — exactly the recoil damage to avoid. The Z1's position-only DiffIK **cannot** passivate a joint, so it is structurally exposed; mitigation = smoothness + impulse-clip + early-termination (or a future admittance/passive layer).
- **Repeated-strike + board-collision framing** (swing until nearly flush; avoid striking the board) corroborates v2 §7.

**More hammering references it surfaces (DIRECT, mostly control/optimization; verify before use):**
- **Tsujita, Konno, Komizunai, Nomura, Owa, Myojin, Ayaz & Uchiyama (2008).** Humanoid Robot Motion Generation for Nailing Task. *IEEE AIM 2008.* — **humanoid nailing.**
- Matsumoto, Konno, Lou & Uchiyama (2006). A Humanoid Robot that Breaks Wooden Boards Applying Impulsive Force. *IROS 2006.*
- Imran & Yi (2016). Impulse Modeling & Analysis of Dual-Arm Hammering Task: Human-like Closed-Chain Manipulator. *IROS 2016* (+ a RA-L impulse-measure version).
- Garabini, Passaglia, Belo, Salaris & Bicchi (2011). Optimality Principles in Variable Stiffness Control: The VSA Hammer. *IROS 2011.*
- Kobayashi, Suzuki, Hirogaki & Aoyama (2016). Impact Task by a Humanoid Using Input Shaping Control.

### 2.6 Adroit "hammer" task / DAPG / D4RL — `[E1] — the standard RL hammering benchmark (MOST USEFUL FIND)`

**Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman, J., Todorov, E., & Levine, S. (2018).** Learning Complex Dexterous Manipulation with Deep RL and Demonstrations. *RSS 2018.* arXiv:1709.10087. — introduces the **Adroit "hammer" task**: a 28-DoF Adroit platform (24-DoF ShadowHand + 4-DoF arm) **picks up a hammer and drives a nail into a board**; nail has dry friction absorbing up to 15 N; success = nail fully driven. Method: **DAPG** (policy gradient + demonstrations).
**Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020).** D4RL: Datasets for Deep Data-Driven RL. arXiv:2004.07219. — ships the Adroit hammer as **offline-RL datasets `hammer-human/expert/cloned-v0/v1`**, benchmarked by *hundreds* of offline-RL papers.
**Gymnasium-Robotics `AdroitHandHammer-v1`** — the maintained env; documents the exact reward below. *(All three verified by direct fetch this session.)*

**The published reward (a concrete template for the Z1 task):**
```
DENSE:
  -0.1 * dist(palm, hammer)                       # reach the tool
  -      dist(hammer_head, nail)                   # bring tool to nail
  -10  * dist(nail_head, board)                    # nail-depth progress, heavily weighted
  -0.01* ||velocity||                              # smoothness
  +2     if hammer lifted > 0.04 m                 # pick-up / wind-up bonus
  +25 then +75 progressive bonuses as nail nears full insertion   # staged progress
SPARSE:
  +10 on success, -0.1 per step otherwise          # success bonus + built-in time penalty
```

**Why it is the most valuable find.** It is a *published, working* reward for an RL hammering task, and its structure independently matches the v2 design: approach/alignment shaping, a **lift/wind-up bonus** (≈ v2 `air_time`), **heavily-weighted nail-depth progress** (≈ v2 `nail_depth_delta`), a **velocity/smoothness penalty**, **staged progressive bonuses**, and a **sparse success bonus with a per-step penalty** (≈ v2 `completion` + the "time penalty" v2 flagged as missing). **Differences to respect:** Adroit uses a **dexterous hand** that *grasps* the hammer (the Z1 hammer is welded); it uses **distance-to-board** as the progress proxy (the Z1 uses nail-slide qpos directly); and it is **quasi-static "pressing," not impact dynamics** (no momentum/restitution). So borrow the **reward skeleton**, but it does NOT solve the impact / ballistic / repetitive-strike problem the Z1 targets.

**Field reframing (corrects the earlier "nearly empty" claim).** Learning-to-hammer is a **standard, heavily-benchmarked RL task**. The genuinely under-explored region is **hammering as an impact/momentum/repetitive-strike problem on a manipulator with sim-to-real** — the Z1 niche. The gap is the *physics + embodiment framing*, not "can RL hammer."

### 2.7 The wider learning-for-hammering landscape (deep-search sweep)

Two multi-lens deep searches (13 search lenses + adversarial verification) mapped the field. **Verdict: learning-to-hammer is a standard RL benchmark + a small but real long tail of direct nail-driving and percussive-impact work.** Load-bearing items below were **re-verified by direct fetch this session** (✅); `◔` = agent-reported with a key attribute (usually the hammer subtask) unconfirmed from the abstract.

**A. Direct nail-driving, learning-based, real robot — the genuine long tail**

| Work | Citation | Tier | Why it matters |
|---|---|---|---|
| Tool-as-Interface ✅ | Chen, Zhu, Liu, Li, Driggs-Campbell (2025), **CoRL 2025**, arXiv:2504.04612 | E1 | Imitation/visuomotor policy from human video; **"Nail Hammering" is task 1 of 5** (locate nail, draw back, strike tip <15.5 mm), **13/13 vs 0/13** teleop baseline. Most on-target learning result. |
| "Let Robots Swing a Hammer" ✅ | Xu (徐映天) … Sun (孙正隆) (2025), **IROS 2025** (CUHK-Shenzhen), IEEE Xplore 11246617 | E1 | High-dynamic tactile servo (STFT + dual-stream PIML, 1 kHz / 1.04 ms) controls hammer **slide** in a 2-finger gripper; **cuts arm-joint recoil 64.3 % (223→80 N)** while raising effectiveness. Closest to the impact/recoil framing. |
| HMAMP "Manipulate as Human" ✅ | Ma, Tian, Gao (2025), **Robotica 43(6):2320–2332**, arXiv:2510.24257 | E1 | Adversarial motion priors learn a human-style **energy-storing back-swing**; hammering is the headline task; real arm. |
| Teramae et al. ✅ | Teramae, Ishihara, Babič, Morimoto, Oztop (2018), **Frontiers in Neurorobotics**, PMC6232299 | E1 | 1-DoF pneumatic-muscle arm; human-in-the-loop (EMG) learning → autonomous nail-driving in **3–5 strikes**. Rare genuine learning-based nail-driving; inherently *repetitive*. |

**B. Dexterous-hand "hammer" benchmark (Adroit lineage) — net-new users beyond §2.6**
RRL (Shah & Kumar, ICML 2021, 2107.03380); VRL3 (Wang et al., NeurIPS 2022, 2202.10324); **H-InDex ✅** (Ze et al., NeurIPS 2023, 2310.01404); MoDem (Hansen et al., ICLR 2023, 2212.05698); DexHandDiff (Liang et al., CVPR 2025, "hammer nail half-drive"); DORA (Zhang et al., 2025, 2505.14819, "hammer use"); SimToolReal ◔ (Kedia, Lum, Bohg, C. K. Liu, 2026, 2602.16863); **Orbik et al. ✅** (deep RL + adversarial **inverse RL** that *learns the reward from demos* on the Adroit hammer — IEEE **ICDL 2021**, doc 9515637; from a TUM MSc thesis, see §2.7F); **"Single-Demonstration / BBE" ◔** (EAAI 2025, ScienceDirect S0952197625016082 — nail-hammering a validated MuJoCo task; *author list unverified*); offline-RL users CQL & IQL (2110.06169). **All press a nail with a 24-DoF hand — quasi-static, sparse binary success, not impact dynamics.** `[E2*/◔ except H-InDex, Orbik]`

**C. Repetitive / ballistic impact via drumming (directly relevant to *repetitive* hammering)**

| Work | Citation | Tier | Why |
|---|---|---|---|
| Robot Drummer ✅ | Shahid, Braghin, Roveda (2025), arXiv:2507.11498 | E2 | RL reframes drumming as a **"Rhythmic Contact Chain"** of timed strikes; emergent cross-arm strategy. The cleanest learning model of *repetitive timed impact*. |
| Karbasi et al. ✅ | Karbasi, Jensenius, Godøy, Torresen (2024), **Frontiers Robotics & AI 11**, PMC11609846 | E1 | DDPG + intrinsic motivation on a flexible-spring arm (ZRob) that **exploits passive rebound** → emergent double/triple strokes. Ballistic compliant multi-strike. |
| DexDrummer ✅ | Fang, Xie, Grannen, Llontop, Sadigh (2026), arXiv:2603.22263 | E2 | Trajectory planning + residual RL, sim-to-real bimanual drumming; "repeated striking" as contact-rich impact. |

**D. Adjacent dynamic striking (no nail)** — Poke-and-Strike ✅ (Aoyama, Moura, Del Aguila Ferrandis, Vijayakumar, **CoRL 2025**, 2509.00178; RL striking on a KUKA iiwa, 90 %); HITTER table tennis (2508.21043 ◔); badminton / diffusion-striking ◔.

**E. Novel reward angles (analogy)**
- **Prolonging Tool Life** ✅ (Wu, Kuo, Kadokawa, Matsubara, 2025, NAIST, arXiv:2507.17275) — integrates **Remaining-Useful-Life (FEA stress → Miner's Rule fatigue)** into the RL reward. **NOT hammering** (object-moving + door-opening) → analogy for §8 actuator/tool-wear budget.
- **Impedance Adaptation by RL with contact DMPs** ◔ (Chang et al., IEEE AIM 2022, 2203.07191) — RL adapts impedance online over a contact-DMP of demonstrated position+force.
- **Robometer** ✅ (Liang et al., **RSS 2026**, arXiv:2603.02115) — general **learned reward model from trajectory comparisons** (progress loss + preferences, 1M+ trajectories). *Not* hammering; an alternative to hand-shaping the reward → relevant to v2 Idea #7.

**F. Theses (near-empty)** — **Orbik ✅ (MSc 2020, TU Munich, HCAR / Prof. D. Lee — mediaTUM 1553993):** deep RL + adversarial **inverse RL** learns the reward from demos on the **Adroit hammer task** to show reward transferability; peer-reviewed spin-off = Orbik, Agostini & Lee, *Inverse RL for Dexterous Hand Manipulation*, **IEEE ICDL 2021** (DOI:10.1109/ICDL49984.2021.9515637, doc 9515637). The one net-new confirmed thesis — and a **Munich** one. Van Rooyen ◔ (PhD 2018, U. Victoria — percussion via stochastic models; not nail-driving); Qin ◔ (PhD 2022, Yale — robot tool use; hammer content unconfirmed). **No dedicated *real-robot* RL-hammering dissertation exists; the only confirmed thesis uses the Adroit *simulation* hammer task.**

**Coverage honesty (CN/JP now searched directly).** The Chinese/Japanese long tail has been **searched, not just assumed**: **CiNii Research returned 0 results** for both *釘打ち + 強化学習* (nail-driving + RL) and *ハンマリング + ロボット + 学習* (hammering + robot + learning); **CNKI/知网, Wanfang/VIP, and J-STAGE** surfaced only **general RL/IL robotics** (navigation, grasping, GAIL humanoid) — **no learning-based hammering**. So the CN/JP niche appears **empty**, with one caveat: CNKI/Wanfang full-text search is login-walled, so it was queried via Google rather than its internal engine — "empty" is high-confidence, not 100% exhaustive. The **industrial** long tail is in practice *fastening/assembly* RL — **IndustReal** (§corpus), **FORGE** (arXiv:2408.04587, force-guided contact-rich, zero-shot sim-to-real), **DexScrew** (screwdriving) — contact-rich but **not percussive nail-driving**. Net: the field map is now **thorough across EN + CN + JP**, exhaustive only modulo login-walled CNKI/Wanfang full text.

**Integrity corrections to the agent sweep** (caught on re-verification): CIMER (2404.05582) does **not** contain a hammer/nail-force task — claim dropped; Prolonging Tool Life does **not** hammer — analogy only; "Let Robots Swing" headline is **recoil −64 %**, not the agent's "+180 % force"; several Adroit-ecosystem hammer-subtask uses are agent-reported (◔), not abstract-confirmed.

**State of the field (the sharpened gap).** Learning-to-hammer is a *standard, heavily-benchmarked RL task* (Adroit/D4RL) **and** has a *small real long tail* of direct nail-driving (Tool-as-Interface, HMAMP, Teramae, the IROS-25 tactile work). What no one has done: **sim-to-real RL for *repetitive, momentum/impact-explicit* nail-driving on a low-cost manipulator**, with strike physics (effective mass, restitution, force-on-nail, recoil, multi-strike sequencing) as the *learning objective*. The drumming papers own "learned timed repetitive impact"; the QP/Adroit papers own "impact physics" and "RL hammering" respectively — **the Z1 project is the first to combine all three on an arm.** That is the defensible novelty for a paper/thesis.

---

## 3. Reward-design ideas harvested (actionable distillation)

The five ideas worth carrying into the Z1 reward design. Each tied to the v2 section it strengthens.

1. **Add a damage / strike-quality dimension to the reward.** Tandon's `R = A·depth − B·bend − FAILURE_COST` is an independent precedent (~15 yr earlier) for "progress minus damage plus a sparse failure cost." The Z1 task models **only depth** — it has no analogue of *bending/ruining the nail* or *off-axis mis-hits*. **Add a strike-quality / damage penalty** (in MuJoCo: nail lateral deflection / tilt, off-axis contact, or excessive lateral contact force). → strengthens v2 §4 (outcome terms), §6 (#1 double-gated impact), §11 (scraping / off-axis / flailing failure modes).

2. **Velocity-dependent noise → Fitt's-Law speed–accuracy trade-off (the deepest idea).** Model *faster strike → larger positional uncertainty at contact → higher miss/bend probability.* This makes "maximize impact velocity" **self-limiting**: the optimum is a *well-aimed* strike, not the fastest one. Two uses: (a) as a **domain-randomization recipe** — inject **velocity-scaled** action/observation noise so the policy must trade speed for accuracy; (b) as the principled reason the impact reward must be **gated/clipped**, not raw-velocity. → directly enriches v2 New-Q13 (speed-vs-inertia), §6 (#3 impact-momentum with clip, #8 adversarial/randomized contact), §8 (safety).

3. **Per-nail repetition + "swings-to-drive" efficiency metric.** Tandon's whole task is *drive many nails, get more efficient per nail* — a repeated-impact learning curve, not a single strike. Its core metric (swings per nail) = v2's **strikes-to-success**. Confirms repetition is the right framing. → v2 §7 (repetitive hammering), §9 metrics, §10 ablations.

4. **Curriculum / transfer over target geometry.** Tandon varies nail-head radius {2…20} and runs train-on-one-size→switch experiments; finds **smaller targets easier** (less hammer-nail coverage → less bend). Concrete precedent for v2's **nail friction/depth/size curriculum + domain randomization** (Q9). → v2 §6 (#8), §9 curriculum stages.

5. **"Learning ≠ reliability" — optimize both.** Q-learning was reliable-but-flat; population-coded improved-but-flaky. *"Learning a task and reliably performing it can be two different optimization targets."* = v2's **one-lucky-hit-vs-repeatability** concern → track **success-rate AND efficiency** as separate metrics, and add a **strike-consistency** penalty. → v2 §4.5 (`strike_consistency`), §6 (#12 → maps to consistency), §10.

6. **Effective-mass-maximization as the reward target (Vu et al. 2026, §2.3).** Peer-reviewed confirmation of v2's `m_eff·v_axial` lever: reward the Effective Mass of the Robot `m_e,n(q) = (nᵀ Jν M⁻¹ Jνᵀ n)⁻¹` (computable from `mjData`) × axial velocity at first contact. For the position-only Z1 it becomes a **reward term**, not an online QP objective. → upgrades v2 §4.1/§4.3 and §6 (#2,#3) from `[ANALOGY]` to a `[DIRECT]` citation; their collinear-impact orientation constraint also confirms v2 §2.3 (orientation matters but needs a torque/orientation action space).

7. **Momentum-over-force is mechanically rigorous, not a heuristic (Vu et al. §III-C).** Force-based impact reward is ill-posed (velocity discontinuous at impact → derivative undefined); energy-based loses unmodelled heat/sound/deformation; momentum + restitution `e*` is the correct model. → hardens v2's "never reward simulated contact force" rule (§4.3, §11) with a textbook-mechanics basis (Stronge 2018). Also: their entire motivation is **recoil/joint-stress mitigation**, corroborating v2 §8 (actuator protection, repeated-shock budget).

8. **Bound repeated impacts by gear-fatigue, not just heat (Romanyuk et al. 2019, §2.5).** Harmonic drives have a Repeated Peak Torque (RPT, infinite-fatigue) and a higher Momentary Peak Torque (MPT, emergency-only); repeated hammering must keep per-strike joint torque below RPT. → add a **gear-fatigue (RPT) constraint** to v2 §8 alongside the I²t thermal budget. Their evidence that a *position-controlled* joint fights the impact (twisting the harmonic drive) reinforces v2 §2.3/§8: the position-only Z1 is the *worst case* for recoil, so smoothness + impulse-clip + early-termination matter **more**, not less.

9. **Borrow the Adroit hammer reward skeleton, then make it impact-aware (§2.6).** The published `AdroitHandHammer` reward already encodes approach shaping + lift/wind-up bonus + heavily-weighted nail-depth progress + velocity penalty + staged bonuses + sparse success with a per-step penalty — almost exactly v2's term list, from a working RL hammering env. Port the *skeleton*, but: (a) keep the Z1's nail-slide qpos progress instead of distance-to-board; (b) replace Adroit's quasi-static "press" with the impact `m_eff·v_axial` / gated-velocity reward (Adroit presses with a hand; the Z1 strikes with a welded hammer); (c) adopt their **+10 success / −0.1 per-step**, which supplies the *time penalty* v2 listed as missing. → directly informs v2 §4, §5A, and the deferred `time_penalty` term.

10. **Model repetition as a "Rhythmic Contact Chain" (Robot Drummer, §2.7C).** Drumming RL proves that *repetitive, precisely-timed impact* is learnable with PPO by treating the task as a sequence of timed contact events to fulfill — exactly v2's event-driven reward machine (§5C). Borrow the formulation: a strike "fires" when contact occurs within a timing/▢depth window; reward chain-fulfillment, not per-step pressing. → v2 §5C / §7.

11. **Exploit passive rebound; don't over-damp it (Karbasi, §2.7C).** A compliant arm's rebound, left un-fought, yields *free* multi-strikes (emergent double/triple strokes). Implication for v2 §4.5 `rebound_damp`: do NOT penalize all post-impact velocity — reward rebound that sets up the next wind-up (a recovery→windup credit). The welded rigid Z1 can't store spring energy, but the principle argues against a blanket velocity penalty in the REBOUND state and *for* (eventually) some end-effector compliance.

12. **Learn the back-swing instead of hand-coding it (HMAMP, §2.7A).** Adversarial motion priors / imitation can produce the energy-storing wind-up without a hand-tuned `air_time`/`windup` term — useful if you collect a few human or scripted hammer demos. → alternative to v2 §6 #14 (residual-over-scripted-swing).

13. **Make recoil a first-class objective (Let Robots Swing a Hammer, §2.7A).** Arm-joint recoil at impact is measurable and reducible (−64 % via controlled slip). The welded Z1 has no slip DoF, so instead **reward low joint reaction-impulse / recoil at impact** (critic/safety signal), achieved through effective-mass posture (the `m_eff` lever) + smoothness. Ties the impact reward to the safety objective. → v2 §8 + the momentum lever.

14. **Fatigue-in-reward via Remaining-Useful-Life (Prolonging Tool Life, §2.7E).** A concrete, citable fatigue model — FEA stress → **Miner's Rule** cumulative damage → RUL — to implement v2 §8's "actuator/gear shock budget" with real physics instead of a crude `I²t` proxy. Combine with the harmonic-drive RPT limit (idea #8). → v2 §8.

15. **Fallback: learn the reward from comparisons (Robometer, §2.7E).** If hand-shaping the impact reward stalls (reward-hacking, weight-tuning churn), a preference/progress-based learned reward model is a documented alternative. Lower priority — only if the hand-designed staged reward proves intractable. → v2 Idea #7.

**One-line takeaway:** the thesis's lasting contribution is the **damage-aware, accuracy-traded reward** (don't reward raw speed; reward aimed strikes that don't wreck the nail) — a dimension the current Z1 reward entirely omits.

---

## 4. To integrate next (from `~/Desktop/_Munich_hammering/`)

Priority order for the Z1 reward work:
- [x] **`QP-based_impact_momentum_maximization_for_a_hammering_task_by_a_humanoid_robot.pdf`** — ✅ read → §2.3. Confirms the `m_eff·v_axial` lever; strong **[DIRECT]** citation. **Recommend folding into v2 §1/§4.3/§6/§8.**
- [ ] `A_Multiple_Working_Mode_Approach_to_Hammering_with_a_Modular_Reconfigurable_Robot.pdf` + `a-multiple-working-mode-approach-to-robotic-hammering-analysis-and-experiments.pdf`
- [ ] `Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation.pdf` (humanoid transfer / force-capable RL)
- [ ] `Learning Variable Impedance.pdf` (ties to v2 §8 admittance/impedance)
