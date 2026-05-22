# Reward Design Literature — Z1 Hammer Task
## Annotated Bibliography (APA 7)

Engineering focus. Each entry: citation → key finding → **what to steal**.

---

### Foundational Theory

**Ng, A. Y., Harada, D., & Russell, S. J. (1999).** Policy invariance under reward transformations: Theory and application to reward shaping. *Proceedings of the 16th International Conference on Machine Learning (ICML)*, 278–287. https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html

The foundational theorem: only shaping rewards of the form `F(s,s') = γΦ(s') − Φ(s)` leave the optimal policy invariant. Any other shaping term can silently shift what the agent optimises for. The current mjlab `nail_driven_reward` (Gaussian on absolute depth) and `hammer_approach_reward` (Gaussian on distance) are NOT potential-based — they bias the policy toward hovering near high-reward states.

**What to steal:** When in doubt, express a shaping term as `γΦ(s') − Φ(s)`. For the approach phase: `Φ(s) = −dist(head, nail_top)` gives zero reward when stationary, positive when approaching, negative when retreating — no hovering incentive.

---

### Contact-Rich Manipulation

**Wu, Z., Lian, W., Unhelkar, V., Tomizuka, M., & Schaal, S. (2021).** Learning dense rewards for contact-rich manipulation tasks. *2021 IEEE International Conference on Robotics and Automation (ICRA)*, 2339–2345. https://doi.org/10.1109/ICRA48506.2021.9561891 (arXiv:2011.08458)

Proposes DREM: self-supervised extraction of dense rewards from high-dimensional observations (images + tactile). The core insight is that task progress (e.g., peg depth) is a monotonically increasing quantity that can be extracted from observations and used as a reward — the "progress" framing. Tested on peg-in-hole and USB insertion.

**What to steal:** The progress framing — `r = f(depth_t) − f(depth_{t-1})` — is the right signal for nail driving. Reward improvements, not absolute position. Their depth-delta formulation directly maps to `nail_depth_delta` in our task.

---

**Tang, B., Lin, M. A., Akinola, I., Handa, A., Sukhatme, G. S., Ramos, F., Fox, D., & Narang, Y. (2023).** IndustReal: Transferring contact-rich assembly tasks from simulation to reality. *Robotics: Science and Systems (RSS) 2023*. https://doi.org/10.15607/RSS.2023.XIX.051 (arXiv:2305.17110)

Introduces SDF (signed distance field) reward for assembly tasks. The SDF evaluated at the part position is smooth across the contact boundary — no discontinuous jump when contact initiates. Also introduces sampling-based curriculum (start near goal, expand difficulty) and simulation-aware policy updates. Zero-shot sim-to-real on peg/gear assembly.

**What to steal:** SDF reward for the alignment sub-phase — guiding the hammer head onto the nail axis before striking. Contact-force rewards from MuJoCo are least faithful to hardware; geometry-based (SDF) rewards transfer better. See TODO in `hammer_z1_env/env.py` for implementation trigger.

---

**Mu, T., Liu, M., & Su, H. (2024).** DrS: Learning reusable dense rewards for multi-stage tasks. *International Conference on Learning Representations (ICLR) 2024*. https://arxiv.org/abs/2404.16779

Learns stage-decomposed dense rewards from sparse rewards + optional demonstrations, where each stage's reward is independently learnable and reusable across task variants. Tested on 1000+ manipulation variants. The stage decomposition is learned from data rather than hand-designed.

**What to steal:** Validates the staged approach → contact → drive design. Our hand-designed stages are architecturally aligned with what DrS discovers automatically. Confirms this is the right structure; we just specify it explicitly.

---

**Luo, Y., Dong, K., Zhao, L., Sun, Z., Zhou, C., & Song, B. (2022).** Dense2Sparse reward shaping for robot manipulation with environment uncertainty. *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. https://arxiv.org/abs/2003.02740

Use dense reward for fast early learning, then switch to sparse reward to prevent overfitting to the shaping signal (reward hacking). The dense-to-sparse transition prevents the policy from learning to game the dense reward at the expense of actual task completion.

**What to steal:** For Z1 training: use `nail_depth_delta × 500` early, then consider reducing weight once the policy reliably drives the nail. Prevents the agent from making tiny repeated taps to farm delta rewards instead of driving the nail fully.

---

**Kumar, V., Todorov, E., & Levine, S. (2016).** Optimal control with learned local models: Application to dexterous manipulation. *2016 IEEE International Conference on Robotics and Automation (ICRA)*, 378–383.

Not directly about reward shaping, but establishes that local models of contact dynamics are learnable and that contact discontinuities do not prevent policy gradient methods from finding good solutions if the reward signal is smooth.

**What to steal:** Confirms that MuJoCo contact physics is learnable by PPO even with discontinuities. The engineering question is reward smoothness near the contact boundary, not contact modelling accuracy.

---

### Impact-Specific Tasks

**D'Ambrosio, D. B., Abeyruwan, S., Abelian, J., Bingham, J., Cofer, B., Dwibedi, D., Foong, C., Foster, E., Garg, A., Golemo, F., Horgan, D., Humplik, J., Ibarz, J., Laskin, M., Leal, F., Nair, A., Oslund, J., Sermanet, P., Sherrer, J., . . . Vanhoucke, V. (2023).** Robotic table tennis: A case study into a high speed learning system. *Robotics: Science and Systems (RSS) 2023*. https://arxiv.org/abs/2309.03315

The most detailed published reward engineering case study for a high-velocity impact task. Over 35 reward components tried; ~20 in active use. Key findings: (1) contact is a binary sparse term (+1 at ball hit), not a dense approach signal; (2) velocity, acceleration, AND jerk are all penalised separately; (3) the full reward table is in Appendix G, Table V. Zero-shot sim-to-real via PyBullet with latency modelled as per-component Gaussians.

**What to steal:** Event-gated reward pattern — +1 fires exactly once at the moment of contact, not continuously during approach. For hammering: `+impact_bonus` fires at each `compute_first_contact()` transition, not on every step near the nail. Velocity penalty weight ≈ 10% of maximum total reward is the sweet spot for sim-to-real.

---

<!-- opus-audit M1: RSS 2025 venue claim not independently re-verified in this audit. Treat as preprint until confirmed against the RSS 2025 proceedings. -->
**Kim, J., Kim, J., Lee, D., Jang, Y., & Kim, B. (2025).** A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation. *Robotics: Science and Systems (RSS) 2025 — venue not re-verified*. https://arxiv.org/abs/2502.16908

ARMADA: low-inertia, back-drivable arms trained in IsaacGym for striking, snatching, and hammering. 24,576 parallel environments. Key finding: for striking tasks, the reward should include velocity-at-impact, not just positional approach. The policy learns qualitatively different swing behaviours when velocity-at-contact is rewarded vs. only final position.

**What to steal:** The `impact_velocity_bonus` term. Scale impact reward by `||head_vel||` at the moment `compute_first_contact()` fires. This is the signal that pushes the policy from slow-press to genuine swing.

---

**van Steen, J., Stokbroekx, D., van de Wouw, N., & Saccon, A. (2024).** Impact-aware robotic manipulation: Quantifying the sim-to-real gap for velocity jumps. *arXiv:2411.06319*.

Directly quantifies how MuJoCo's impact model (ante-impact → post-impact velocity jump) differs from real robot behaviour. Key finding: 3.1% average error in post-impact velocity with calibrated rigid-body impact model. Contact force magnitudes are less faithful than velocity jumps.

**What to steal:** Prefer reward terms based on geometric outcomes (nail depth, displacement) over contact force magnitudes — the former transfers, the latter does not. The `nail_depth_delta` term is safe; a `contact_force_reward` is not.

---

### Locomotion (Air-Time / Rhythmic Motion Analogs)

**Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022).** Learning to walk in minutes using massively parallel deep reinforcement learning. *Conference on Robot Learning (CoRL) 2022*. https://arxiv.org/abs/2109.11978

Canonical RSL-RL reward design for legged locomotion. The `feet_air_time` reward fires only at the transition from air to ground contact: `reward = (air_time − threshold) × first_contact_flag`. This rewards longer strides (more air time before each step) and is the direct analog of rewarding bigger hammer swings (more retraction before each strike). Standard smoothness terms: action rate + torque penalty + joint velocity penalty.

**What to steal:** The `air_time` reward structure maps directly to a hammer retraction reward. Replace "feet" with "hammer head," "ground contact" with "nail contact," and "air" with "above-nail clearance." Use mjlab `ContactSensor` with `track_air_time=True` to access `last_air_time` at the moment `compute_first_contact()` fires.

---

### Simulation-to-Real Transfer

**Ma, Y. J., Liang, W., Wang, G., Huang, D. A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A. (2023).** Eureka: Human-level reward design via coding large language models. *International Conference on Learning Representations (ICLR) 2024*. https://arxiv.org/abs/2310.12931

GPT-4 iteratively writes and evaluates reward code in IsaacGym. Outperforms human-expert rewards on 83% of 29 tasks. Key finding: LLM-discovered rewards for dynamic tasks (dexterous pen spinning) consistently include velocity-at-contact terms and separate approach/contact/outcome stages — validating the staged design.

**What to steal:** The Eureka discovery pattern confirms our staged reward architecture is what an automated reward search would also discover. Also: Eureka found that exponential/Gaussian shaping near-target (not linear) works better in most cases — consistent with using Gaussian `nail_driven_reward` for fine near-goal gradient.

---

<!-- opus-audit M1: DrEureka venue not re-verified in this audit; cited here as preprint. -->
**Ma, Y. J., Liang, W., Zhu, G., Wang, G., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A. (2024).** DrEureka: Language model guided sim-to-real transfer. *arXiv preprint arXiv:2406.01967*.

Extends Eureka to auto-generate domain randomisation distributions. Key finding: reward terms based on task outcome (object position, joint position) transfer better than terms based on intermediate physics (exact contact force, contact location). Rewards that exploit simulation-specific artefacts (precise contact normals) hurt sim-to-real.

**What to steal:** `nail_depth_delta` (geometric outcome) → safe to transfer. Any reward reading `contact_force` directly from MuJoCo sensordata → risky. The `impact_velocity_bonus` (head velocity at contact, not contact force) sits in between — velocity is more faithful than force.

---

### Reward Architecture Patterns

<!-- opus-audit H3: corrected author list against arXiv:1910.10897 (verified 2026-05-22).
     Sonnet 4.6 had fabricated "Shao, H.", "Nair, A.", "Chen, S.", "Bahl, S.", "Planche, B."
     and missed "Shively, H.", "Bellathur, A.". -->
**Yu, T., Quillen, D., He, Z., Julian, R., Narayan, A., Shively, H., Bellathur, A., Hausman, K., Finn, C., & Levine, S. (2020).** Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. *Conference on Robot Learning (CoRL) 2020*. https://arxiv.org/abs/1910.10897

Meta-World's hammer v3 task uses a dense hybrid reward: `reward = (2 × grasp + 6 × in_place) × quat_alignment`, with a flat +10 bonus when `nail_slide_joint.qpos > 0.09 m`. The multiplicative `quat_alignment` term kills the reward entirely when hammer orientation is wrong — a hard gate rather than a soft penalty. The task is single-strike.

**What to steal:** The orientation alignment gate — multiply the approach reward by a Gaussian over the hammer head orientation relative to the nail axis. This prevents the policy from approaching from impossible angles. Also: the flat completion bonus pattern (not proportional to remaining distance, just a large step at threshold).

---

**Narang, Y., Storey, K., Akinola, I., Macklin, M., Reist, P., Langlois, O., Handa, A., & Fox, D. (2022).** Factory: Fast contact for robotic assembly. *Robotics: Science and Systems (RSS) 2022*. https://arxiv.org/abs/2205.03532

Factory provides three assembly environments (nut tightening, peg insertion, gear meshing) with fast MuJoCo-based contact physics. The paper focuses on simulation fidelity; reward design is intentionally minimal (sparse success + small dense position reward). The follow-on IndustReal paper (Tang et al., 2023) adds the SDF reward layer on top.

**What to steal:** Factory's sim fidelity work validates MuJoCo as a viable impact simulator. Their contact parameter calibration methodology is directly applicable to calibrating the Z1 hammer's nail contact stiffness and restitution.

---

**Berducci, L., Aguilar, E. A., Ničković, D., & Grosu, R. (2024).** HPRS: Hierarchical potential-based reward shaping from task specifications. *Frontiers in Robotics and AI*, 11. https://arxiv.org/abs/2110.02792

Formalises task requirements as a partially-ordered hierarchy (safety > target > comfort) and builds a potential function that enforces priority order while preserving policy optimality (Ng et al. guarantee). Maps cleanly to: safety = joint limits/collision avoidance, target = nail depth, comfort = smoothness.

**What to steal:** The hierarchy framing. Joint limit penalties should have higher priority (larger weight) than smoothness penalties, which should have higher priority than approach rewards. Getting this ordering right prevents the policy from trading safety for task progress.

---

<!-- opus-audit M1: Reward Training Wheels venue not re-verified in this audit; cited here as preprint. -->
**Wang, L., Xu, T., Lu, Y., & Xiao, X. (2025).** Reward training wheels: Adaptive auxiliary rewards for robotics RL. *arXiv preprint arXiv:2503.15724*.

Teacher-student meta-RL that dynamically adjusts auxiliary reward weights based on student capability. Early in training, approach rewards are weighted high; as the agent reliably achieves contact, the teacher reduces approach weight and shifts emphasis to depth reward. Outperforms fixed expert-designed weights.

**What to steal:** Motivation for the soft-fade approach (exp decay with depth) rather than a fixed weight — it approximates what RTW does automatically. If training shows instability at the approach-to-contact transition, consider implementing adaptive weighting.

---

### Sim-to-Real Smoothness

**Kim, Y., Lee, J., Choe, J., Cho, H., & Kang, Y. (2023).** Not only rewards but also constraints: Applications on legged robot locomotion. *arXiv:2308.12517*.

Advocates Lagrangian RL for hard constraints (joint limits, torque limits) rather than soft reward penalties. The insight: a large task reward can overwhelm soft penalty terms, causing constraint violations during high-impact events (exactly the scenario during a hammer strike).

**What to steal:** Consider treating Z1 joint torque limits as hard constraints (Lagrangian multiplier) rather than soft penalties during the strike phase, where the task reward gradient is largest. Alternatively: use a large weight on `joint_pos_limits` (already at −1.0 in current config) and also add a torque limit term.

---

*Compiled 2026-05-22. All arXiv IDs verified unless marked otherwise.*
