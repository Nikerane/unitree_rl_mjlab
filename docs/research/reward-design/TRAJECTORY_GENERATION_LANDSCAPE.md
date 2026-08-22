# Trajectory generation for impact-safe Z1 hammering

**Date:** 2026-08-22
**Status:** Primary-source literature review and design recommendation; no implementation or training is authorized by this note.
**Scope:** How to construct multiple hammer-head routes, preserve a precise nail impact, and vary the physical hammer starting point on a plane without changing the thesis into a two-level generate-then-track architecture.

## Executive verdict

The next useful step is **not another training run**. The current horizontal pilot used a temporary world-X detour that disappears halfway through the strike, while route identity is absent at the first policy action. Its negative result therefore does not decide whether a well-conditioned family of impact trajectories is learnable.

The literature supports the first three conclusions below; the fourth is a project-specific engineering recommendation based on those findings and the present Z1 scope:

1. **Use a nail-relative, endpoint-constrained curve.** The hammer must pass through the nail with a controlled incoming tangent and nominal impact speed. A spline is appropriate, but “B-spline” is not by itself a design: the boundary conditions, internal degrees of freedom, phase law, and impact switch matter.
2. **For a complete impact controller, split the reference at physical contact.** Plan overlapping ante- and post-impact branches and select between them using the contact event. Extend the executable ante branch slightly past the nominal nail state so a late contact does not exhaust the reference; do not demand physical velocity or acceleration continuity through an impulse. The first trajectory-conditioning pilot can remain ante-impact-only because the current imitation term already stops at contact.
3. **A start plane must move the physical reset, not merely the reference.** Sample reachable hammer-head start poses in a plane perpendicular to the nail axis, solve and qualify corresponding robot configurations, then anchor the curve at the realized start.
4. **For three designed routes and a small start plane, a deterministic constrained spline is the highest-ROI first method.** DMPs, ProMPs, KMPs, and task-parameterized models become attractive when trajectories must be learned from several demonstrations or adapted over a much larger family. They are not necessary for the first causal experiment.
5. **The broader autonomous-driving, UAV, and manipulator literature strengthens this choice rather than replacing it.** Its most transferable pattern is: express motion in a task-aligned frame, generate a small interpretable family of smooth paths, enforce whole-curve constraints where useful, and parameterize speed separately. Car-specific clothoids, large state lattices, and general obstacle optimizers are background methods, not the default hammer generator.

The recommended sequence is: bank the current negative pilot result; qualify a deterministic spline family on CPU; test start variation and route variation separately; expose route intent from the first action; only then train a small factorial pilot. Curriculum learning and domain randomization come later.

## Questions and search method

This review asked:

- How do impact and striking papers specify the contact point, direction, and nonzero impact velocity?
- Which trajectory representations can adapt to different initial points without losing the contact boundary?
- How should pre- and post-impact references be joined?
- What does a defensible “starting-point plane” experiment mean for the Z1?

The search prioritized peer-reviewed primary papers and official author/publisher copies. It covered robot hammering, table-tennis striking, impact-aware trajectory optimization and reference spreading, deterministic splines, DMPs, ProMPs, KMPs, task-parameterized movement models, autonomous-driving and mobile-robot trajectory generation, UAV polynomial/B-spline planners, whole-curve corridor methods, and manipulator path timing. The direct 2026 hammering paper supplied by the user was inspected locally in full. Secondary surveys were used only as gateways to primary papers.

No primary paper located in this search studies exactly the same combination as this thesis: a single online-RL variable-impedance hammer policy, active per-joint impulse CaT, a prescribed multi-route family, and a two-dimensional start-plane distribution. The recommendation below is therefore a synthesis of adjacent primary evidence plus project-specific engineering constraints, not a copied published recipe.

## 1. Current repository ground truth

The authoritative implementation is [`SingleStrikeReference`](../../../src/tasks/hammer/mdp/references.py):

- The live hammer-head position at reset is anchor `A`.
- The endpoint `B` is 0.15 m below the live nail top, so the straight reference passes through the nail and continues downward.
- Phase is a monotone spatial projection onto the original straight chord, not clock time and not curve arc length.
- The horizontal pilot adds

  \[
  r\,0.020\sin^2(2\pi\phi)\,e_x,
  \qquad r\in\{-1,0,+1\},\ 0<\phi<0.5,
  \]

  and adds zero afterward. The peak offset is 20 mm at `phi=0.25`; all routes merge at `phi=0.5`, well before contact.
- The policy observes phase and instantaneous 3-D reference error, but not route identity. At reset all routes have the same waypoint and zero reference error, so the first policy input is route-ambiguous.
- The imitation term is an ante-impact, position-only Gaussian with `sigma=50 mm`. Ignoring a 20 mm detour still earns about 85% of its maximum raw value.

This is useful as a deliberately small observability pilot, but it is not a general spline or trajectory-family test. The current evaluated horizontal result also remains temporary rather than durably banked under `docs/results/`; it must be banked before later work treats it as thesis evidence.

## 2. What the impact and striking literature actually does

### 2.1 Constrained Bézier/spline trajectories

Vu et al. (2026) construct one fifth-order Bézier curve from the initial hammer-tip position to the nail. They set the two endpoint positions and use four other controls to encode desired initial/final velocity and acceleration. They track this curve with a QP while optimizing effective mass. The construction is directly relevant because it makes the nail and incoming velocity explicit. It does **not** provide route diversity, start-plane sampling, or a post-impact trajectory; post-impact reference generation is listed as future work. The user-supplied six-page IEEE PDF was inspected in full. [DOI](https://doi.org/10.1109/AMC67705.2026.11435814)

A mathematical consequence matters for our design: a degree-5 Bézier has six control points. At a fixed duration/parameterization, fixing position, velocity, and acceleration at both endpoints consumes all six. With identical endpoint data, there is **no remaining route-shape degree of freedom**. Vu et al. fix `T=1 s`, so this applies to their treatment. If duration itself is optimized, that adds a scalar timing degree of freedom, but not an independent designed route waypoint. Calling the curve a B-spline does not change this count. To obtain left/straight/right routes, we need either more control points/multiple spans, a higher-degree curve with an internal constraint, or a separate boundary-preserving deformation.

The printed acceleration/control-point equations in the paper should also be independently rederived before use. For a degree-`N` Bézier with duration `T`, the standard endpoint identities are

\[
P_1=P_0+\frac{T}{N}v_0,\qquad
P_2=2P_1-P_0+\frac{T^2}{N(N-1)}a_0,
\]

\[
P_{N-1}=P_N-\frac{T}{N}v_T,\qquad
P_{N-2}=2P_{N-1}-P_N+\frac{T^2}{N(N-1)}a_T.
\]

The paper prints inconsistent denominators in its equations (28)–(30). This review treats the curve concept as evidence, not the printed algebra as implementation authority.

### 2.2 Smoothness, jerk, and impact-state polynomials

“Minimum jerk,” “quintic,” and “smooth at impact” are not equivalent claims. Flash and Hogan minimize integrated squared Cartesian jerk for point-to-point human reaching. Piazzi and Visioli construct joint-space cubic splines and globally optimize their timing for a constrained minimax-jerk objective. Gasparetto and Zanotto optimize fifth-order B-splines under a weighted execution-time/integrated-squared-jerk objective and derivative bounds. These papers show that **jerk is an objective or constraint**, while polynomial degree and basis are separate design choices. [Flash and Hogan 1985](https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985), [Piazzi and Visioli 2000](https://doi.org/10.1109/41.824136), [Gasparetto and Zanotto 2007](https://doi.org/10.1016/j.mechmachtheory.2006.04.002)

A conventional rest-to-rest minimum-jerk segment is wrong for the final downswing if it forces zero terminal velocity at the nail. A jerk-regularized quintic can still be appropriate when its terminal velocity is deliberately nonzero. Jia et al. provide a useful counterexample to the claim that striking inherently requires a quintic: their online batting planner uses a **quartic** joint polynomial because it needs five scalar conditions per joint—current position/velocity/acceleration at the replan splice and desired position/velocity at impact. Degree should follow the required boundary conditions and remaining optimization freedom. [Jia, Gardner, and Mu 2019](https://doi.org/10.1177/0278364918817116)

A spatial curve alone also does not specify impact speed. For a path `x(s)`, physical velocity is `x'(s) * s_dot`. Geometry and the phase-rate/speed law must therefore be designed and tested separately.

### 2.3 Hitting movement templates and DMPs

Kober et al. reformulate DMP-style movement templates so a strike can reach a specified hitting point with a specified nonzero velocity while preserving the learned global shape. They explicitly decompose a stroke into swing-back, hit, follow-through, and return phases, and demonstrate adaptation over different hitting locations. This is strong evidence that start/contact adaptation and follow-through should be represented as a composed striking skill, not as one zero-velocity point-to-point primitive. [Kober et al. 2010](https://doi.org/10.1109/ROBOT.2010.5509672), [author PDF](https://pub.ista.ac.at/~chl/papers/kober-icra2010.pdf)

The standard discrete DMP is a poor literal fit for the nail contact boundary because its terminal attractor normally approaches the goal at zero velocity. Nail contact would need to be an interior hitting/via state, or the hitting-template modification must be used. Classical DMP spatial scaling can also create unnatural trajectories under large changes; later work explicitly corrects this by adapting weights under initial/final/via-point constraints. [Ijspeert et al. 2013](https://doi.org/10.1162/NECO_a_00393), [Sidiropoulos and Doulgeri 2024](https://doi.org/10.1007/s10846-024-02051-0)

Task-specific DMP work adds an important route-diversity warning: nearby task queries can be interpolated only when they belong to one locally smooth motion family. Motions passing on different sides of an obstacle should be clustered as different modes rather than averaged into one primitive. That is the closest direct literature analogue to keeping start-plane adaptation separate from left/straight/right route identity. [Ude et al. 2010](https://doi.org/10.1109/TRO.2010.2065430)

### 2.4 Probabilistic and task-parameterized primitives

ProMPs represent a distribution over trajectories and can be conditioned on desired positions, velocities, and via points. A striking-specific extension adapts a learned joint-trajectory distribution to a desired end-effector position, velocity, orientation, and intercept timing. This is attractive when multiple demonstrations describe legitimate strike variability; it is unnecessary overhead for three hand-designed curves. [Paraschos et al. 2013](https://papers.nips.cc/paper_files/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html), [Gómez-González et al. 2016](https://doi.org/10.1109/HUMANOIDS.2016.7803322)

Task-parameterized models express demonstrations in object- or landmark-relative coordinate frames and reproduce them under changed frame poses. This strongly supports defining trajectories in a **nail frame**, rather than permanent world X/Y. KMPs add conditioning on task constraints and local coordinate systems. These methods are candidates if future work learns a broad family from demonstrations, different nail poses, or richer contexts. [Calinon 2015/2018](https://doi.org/10.1007/978-3-319-60916-4_7), [Huang et al. 2019](https://doi.org/10.1177/0278364919846363)

Dragan et al. show that adapting a demonstration to a new start and goal is an optimization over trajectory space: minimize deformation while satisfying new endpoint constraints. The key lesson is that “preserve the same shape” depends on the chosen norm; start adaptation is not neutral and must be judged by task-relevant geometry. [Dragan et al. 2015](https://publications.ri.cmu.edu/movement-primitives-via-optimization)

### 2.5 Free-time and impact-aware trajectory optimization

Koç et al. optimize both a striking trajectory and its impact time, avoiding a fixed virtual hitting plane that can make strokes restrictive or infeasible. For our project, this warns against confusing a **start-state sampling plane** with a mandatory impact plane. [Koç et al. 2018](https://doi.org/10.1016/j.robot.2018.03.012)

Stouraitis et al. jointly optimize continuous trajectories, contact timing, contact forces, and compliance across hybrid free/contact modes. Ti et al. optimize tool pose and directional velocity manipulability for real nail hammering. Both show that geometry changes can alter posture, effective mass, achievable velocity, and compliance; a start-plane comparison therefore cannot be interpreted as “trajectory only” unless those downstream changes are measured. [Stouraitis et al. 2020](https://doi.org/10.1109/IROS45743.2020.9341246), [Ti et al. 2024](https://arxiv.org/abs/2402.05502)

Konno et al. optimize a whole-body humanoid impact state using an impact model, while Teramae et al. segment and replay a collision-terminated command for a real 1-DoF pneumatic hammer. These are very different systems, but both reinforce that the state and command at impact—not merely the geometric nail point—are load-bearing. Neither establishes a preferred spline family. [Konno et al. 2011](https://doi.org/10.1177/0278364911405870), [Teramae et al. 2018](https://doi.org/10.3389/fnbot.2018.00071)

### 2.6 References across impact

Impact-time mismatch makes ordinary time-indexed tracking ill posed near contact. Reference-spreading work instead defines compatible, overlapping ante- and post-impact references, switches according to contact state, and uses position-dominant interim behavior when velocity feedback is unreliable. The time-invariant formulation uses state-based vector fields and has been experimentally validated over hundreds of impacts. This supports our spatial phase and an event-split reference with a late-contact continuation; it does not require us to copy the full model-based controller into RL. [Biemond et al. 2013](https://doi.org/10.1109/TAC.2012.2223351), [van Steen et al. 2023](https://doi.org/10.23919/ACC55779.2023.10156028), [van Steen et al. 2024](https://arxiv.org/abs/2411.09870)

### 2.7 What trajectory-generation work in other fields adds

This extension deliberately extracts only **trajectory-generation mechanisms**. Perception, behavior prediction, feedback control, and vehicle-specific dynamics are outside the comparison unless they impose a mathematical condition on the generated curve or its timing.

Terminology is kept strict: a **path** `c(s)` is geometric; a **time law** `s(t)` says how quickly it is traversed; and the resulting **trajectory** is `x(t)=c(s(t))`. State lattices search over candidate paths, while CHOMP and TrajOpt refine initialized paths or trajectories; they are included only as adjacent generation methods, not treated as equivalent representations.

#### Autonomous driving and mobile robots

Werling et al. generate road trajectories in a Frenet frame: progress along the road and lateral displacement are represented separately, candidate longitudinal/lateral polynomials are constructed for terminal states, and candidates are scored. Apollo's EM planner later operationalized a related industrial pattern: search over candidate paths, refine the selected path with spline optimization, and optimize speed separately. The transferable idea is not a road or lane model; it is a **task-attached coordinate system and a compact candidate family**. For the hammer, the analogous coordinates are progress along the nail axis plus one or two transverse route offsets in the nail frame. [Werling et al. 2010](https://doi.org/10.1109/ROBOT.2010.5509799), [Fan et al. 2018](https://arxiv.org/abs/1807.08048)

State-lattice work constructs a finite graph whose edges are already feasible motion primitives connecting complete states, rather than searching arbitrary point sequences and smoothing them afterward. This is useful precedent for qualifying a small bank of left/straight/right hammer curves before learning. A large lattice is unnecessary for three routes, and automotive polynomial spirals or clothoids encode nonholonomic steering/curvature constraints that the Z1 hammer head does not share. [Pivtoraiko, Knepper, and Kelly 2009](https://doi.org/10.1002/rob.20285), [McNaughton et al. 2011](https://doi.org/10.1109/ICRA.2011.5980223), [Fraichard and Scheuer 2004](https://doi.org/10.1109/TRO.2004.833789)

The precise transfer is therefore:

- define a nail-relative frame;
- keep axial progress separate from transverse route shape;
- generate only a small, interpretable set of boundary-conditioned candidates;
- qualify each candidate before exposing it to the policy.

#### UAV polynomial, Bézier, and B-spline trajectories

Mellinger and Kumar and, more generally, Richter et al. generate piecewise polynomial trajectories through waypoints while constraining endpoint derivatives. Richter et al. formulate many segments in a sparse, numerically stable quadratic program and separately allocate time to each segment. These are strong precedents for a multi-span hammer curve with exact start/contact conditions plus interior route freedom. Minimum snap was chosen in a quadrotor differential-flatness formulation; it is **not evidence that snap is the correct hammer objective**. The hammer objective could instead regularize jerk, acceleration, curvature, or joint realizability after comparison. [Mellinger and Kumar 2011](https://doi.org/10.1109/ICRA.2011.5980409), [Richter, Bry, and Roy 2016](https://doi.org/10.1007/978-3-319-28872-7_37)

Uniform B-splines provide local support: moving one control point changes only a local part of the curve, which is attractive for a larger route family. Bernstein/Bézier safe-corridor work contributes a different property: because a Bézier segment lies in the convex hull of its controls, constraining all controls to a convex corridor constrains the complete **position curve**, rather than only sampled points. Velocity, acceleration, or jerk bounds require separate constraints on the corresponding derivative control polygons through the hodograph property. For this fixture that could certify clearance tubes and, when explicitly added, derivative bounds between the start and nail. Dynamic-obstacle replanning and 3-D map construction do not transfer. [Usenko et al. 2017](https://doi.org/10.1109/IROS.2017.8202160), [Gao et al. 2018](https://doi.org/10.1109/ICRA.2018.8462878)

#### Path timing and state-to-state generation

Kunz and Stilman state the key distinction directly: first choose a differentiable geometric path, then convert it into a time-parameterized trajectory that respects velocity and acceleration bounds. TOPP-RA extends this path-parameterization problem with a reachability formulation. Ruckig solves a simpler but complementary problem: jerk-limited state-to-state timing with arbitrary target position, velocity, and acceleration. It could generate a scalar phase law or a short boundary splice with nonzero terminal velocity, but it does not create a curved route. These papers do not choose the hammer route and do not enforce impact impulse; they show how to keep the two design questions separate:

\[
\text{geometry } c_r(s) \quad\text{and}\quad \text{timing } s(t),
\qquad
v_{\mathrm{impact}}=c_r'(s_I)\,\dot{s}(t_I).
\]

A spline that passes exactly through the nail still does not specify a useful strike unless its phase law gives a predeclared nonzero incoming speed. Conversely, retiming cannot repair a bad geometric route. For joint-limit timing, the Cartesian hammer curve must first be mapped to a continuous, single-IK-branch differentiable path `q(s)`; TOPP does not perform that mapping. [Kunz and Stilman 2012](https://doi.org/10.15607/RSS.2012.VIII.027), [Pham and Pham 2018](https://doi.org/10.1109/TRO.2018.2819195), [Berscheid and Kroeger 2021](https://doi.org/10.15607/RSS.2021.XVII.015)

#### General trajectory optimizers

CHOMP, STOMP, and TrajOpt optimize an initialized trajectory for smoothness, obstacle clearance, or other costs. Direct collocation optimizes states and controls under dynamics. Modern sparse polynomial optimizers such as GCOPTER/MINCO jointly optimize interior waypoints and segment durations while retaining corridor constraints. These are important gateway families when a hand-parameterized curve cannot satisfy workspace, collision, or joint constraints. They are not the highest-ROI first generator here: the fixture has one known contact target, a small route family, and no evidence yet that a constrained spline is infeasible. Starting with a large nonlinear optimizer would make the scientific treatment harder to interpret. [CHOMP](https://doi.org/10.1177/0278364913488805), [STOMP](https://doi.org/10.1109/ICRA.2011.5980280), [TrajOpt](https://doi.org/10.1177/0278364914528132), [GCOPTER](https://doi.org/10.1109/TRO.2022.3160022), [Kelly 2017](https://doi.org/10.1137/16M1062569)

#### Cross-domain generator distilled for the Z1

The common transferable pipeline is:

```text
nail frame
    -> compact route parameters (start u,v; route r; contact tangent)
    -> endpoint-constrained multi-span polynomial/Bézier/B-spline c_r(s)
    -> whole-curve and joint-realizability qualification
    -> separate monotone phase/speed law s(t)
```

This does **not** introduce another controller or learned planner. It only makes the reference trajectory well defined. How the existing policy receives the route command is a separate experiment-design question covered later in this note. For the first generator, use deterministic multi-span polynomial or Bézier geometry. Use a clamped B-spline when the number of internal route controls grows; use TOPP-style timing only if a simple predeclared phase law cannot meet velocity/acceleration requirements; use CHOMP/TrajOpt/direct collocation only if obstacles or joint feasibility defeat the compact generator.

The most useful reading order for this trajectory-only question is: [Gasparetto et al.'s path/trajectory overview](https://doi.org/10.1007/978-3-319-14705-5_1) for the map; [Richter, Bry, and Roy](https://doi.org/10.1007/978-3-319-28872-7_37) for practical multi-segment polynomial construction; [Kunz and Stilman](https://doi.org/10.15607/RSS.2012.VIII.027) for geometry-versus-timing; [Werling et al.](https://doi.org/10.1109/ROBOT.2010.5509799) for task-aligned candidate generation; and [Gao et al.](https://doi.org/10.1109/ICRA.2018.8462878) for whole-curve Bézier constraints. [Paden et al.](https://doi.org/10.1109/TIV.2016.2578706) is a broader vehicle-literature gateway that also includes control; [Kelly 2017](https://doi.org/10.1137/16M1062569) is an optional advanced tutorial if later work requires full trajectory optimization. The mechanism claims above remain grounded in the cited primary papers.

## 3. Method comparison for this thesis

| Family | Exact nail and incoming velocity | Different starts | Internal route control | Post-impact handling | Data/engineering | Best use here |
|---|---|---|---|---|---|---|
| Single quintic Bézier with endpoint `p/v/a` | Yes | Recompute endpoints | None if all six controls are fixed at fixed duration | Not supplied by Vu et al. | Low | Baseline boundary construction, not a route family |
| Multi-span/clamped spline or higher-degree Bézier | Yes | Recompute/condition start | Yes, via internal controls/knots | Separate contact-switched span | Low–medium | **Recommended first deterministic family** |
| Hitting-template DMP | Yes, with striking modification | Strong start/target adaptation | Learned shape parameters | Natural phase composition | Medium | Later reusable learned primitive |
| Standard discrete DMP | Goal is normally zero-velocity | Yes | Learned forcing term | Awkward if nail is terminal goal | Medium | Do not use literally for impact endpoint |
| ProMP striking | Condition position/velocity/orientation | Yes | Distribution/via-point conditioning | Can compose primitives | Medium–high; needs demonstrations | Later multi-demo family and uncertainty |
| TP-GMM/KMP | Task-frame adaptation | Yes | Rich contextual constraints | Must be designed explicitly | High; needs demonstrations | Later nail-pose/context generalization |
| Free-time/hybrid trajectory optimization | Yes; can optimize impact time | Yes | Full optimization | Explicit hybrid modes | High/model dependent | Qualification or future generator, not first RL reference |
| Time-invariant reference spreading | Impact-consistent reference fields | Robust to state/timing variation | Not a route-learning method | Strongest impact transition treatment | High if copied fully | Design principle for event splitting/spatial reference |
| Frenet/candidate-polynomial generation | Exact terminal state when encoded in the candidate | Recompute in task frame | Yes, through sampled terminal/via parameters | Not an impact method | Low–medium | Nail-frame axial/lateral decomposition and compact route bank |
| Multi-segment minimum-derivative polynomial | Exact boundary derivatives and internal waypoints | Re-solve boundary conditions | Yes | Must be event-split for impact | Low–medium | Strong cross-domain basis for the first deterministic family |
| Bernstein/Bézier corridor optimization | Exact endpoints; whole segment lies in control-point hull | Re-solve controls/corridor | Yes | Must be event-split for impact | Medium | Whole-curve clearance/derivative bounds when needed |
| TOPP/TOPP-RA path parameterization | Does not choose the contact geometry | Re-time a qualified differentiable joint path | No | Not an impact model | Medium | Separate phase/speed law after route geometry and a continuous IK branch are fixed |
| Ruckig jerk-limited state-to-state generation | Exact target position/velocity/acceleration | Native arbitrary initial state | No internal route control | Not an impact model | Low | Scalar phase law or short boundary splice, not route geometry |
| CHOMP/STOMP/TrajOpt/direct collocation | Depends on explicit constraints | Strong | Strong but initialization/local-optimum dependent | Can model modes only if added | High | Escalation only when compact splines fail qualification |

## 4. A defensible starting-point plane

Let `N` be the live **hammer-face contact point** on the nail and `n` the unit impact direction from hammer toward nail. Choose a stable task direction `r0` that is not parallel to `n` (for example the projected robot-base-to-nail direction), and define

\[
e_1=\frac{(I-nn^T)r_0}{\|(I-nn^T)r_0\|},
\qquad e_2=n\times e_1.
\]

With nominal stand-off distance `h`, a physical start candidate is

\[
A(u,v)=N-hn+u e_1+v e_2.
\]

For the present vertical nail, this is a horizontal plane above the nail. It must be expressed from live nail sites, not hard-coded world coordinates.

```text
      reachable start plane Pi_h

       o------o------o
       |      |      |
       o------o------o       pre-impact curves
       |      |      |     \      |      /
       o------o------o       \     |     /
                              \    |    /
                               *  nail contact N
                               |
                               |  event-switched follow-through
                               v
```

The sampled set should initially be a small, reachability-filtered ellipse, not an arbitrary rectangle:

\[
\mathcal D=\{(u,v):u^2/a^2+v^2/b^2\leq 1\}\cap\mathcal F,
\]

where `F` is the feasible set. Each point needs a real robot reset configuration with:

- hammer face orientation aligned with the nail;
- zero or predeclared initial velocity;
- joint-limit, self-collision, scene-collision, and actuator feasibility;
- adequate manipulability and no singular configuration;
- a productive scripted strike with acceptable 500 Hz joint velocity and impulse exposure.

Changing only `A` inside the reference while leaving the robot at the old reset is **not** start-plane randomization. It creates a deliberate initial reference error. Genuine start variation moves the physical hammer head through a reviewed IK-generated pose library or an equivalent qualified reset solver.

Use the nominal joint pose as the IK seed and retain one connected IK branch. Otherwise two nearby plane points may silently produce very different elbow/wrist postures, turning a start-location study into a redundancy-branch study.

Sampling law is part of the treatment:

- for **coverage qualification**, use a deterministic grid plus uniform-in-ellipse-area samples (`radius` sampled with a square root, not uniformly);
- for **deployment-like training**, use a documented truncated Gaussian based on expected reset error;
- for **robustness training**, use a preregistered mixture of that Gaussian and a lower-weight uniform component so the boundary is not unseen;
- hold stand-off `h`, initial velocity, face orientation, desired impact speed, and route ID fixed while studying start-plane robustness.

Report ellipse axes `(a,b)`, stand-off `h`, rejection rules, acceptance rate, and realized density. There is no literature-derived universal plane size; it must come from the qualified Z1 workspace.

### Boundary-preserving start deformation

A compact deterministic construction is possible without learning a primitive. Let `c0(s)` be an endpoint-constrained nominal ante-impact curve, `s in [0,1]`, with `c0(0)=N-hn`. Define

\[
\beta(s)=1-10s^3+15s^4-6s^5.
\]

Then

\[
c_{u,v}(s)=c_0(s)+\beta(s)(u e_1+v e_2)
\]

starts at the requested plane offset while preserving the nominal contact position and first/second **path derivatives with respect to `s`** because `beta(1)=beta'(1)=beta''(1)=0`. At the shifted start, the deformation contributes zero first/second derivative, so `c'_{u,v}(0)=c'_0(0)` and `c''_{u,v}(0)=c''_0(0)`; the full derivatives are not generally zero. Physical velocity and acceleration still depend on the separately chosen phase-rate law. This formula is a project-specific synthesis of spline boundary logic, not a claim copied from one paper.

Route diversity can be added independently with a boundary-preserving bump such as

\[
\gamma(s)=64s^3(1-s)^3,
\qquad
c_{u,v,r}(s)=c_{u,v}(s)+r a\gamma(s)e_1,
\]

where `r in {-1,0,+1}` and `a` is the route amplitude. `gamma` and its first two path derivatives vanish at both endpoints, so route shape does not perturb the start or impact boundary in the curve parameter. The same family can be represented by a clamped spline with sufficient interior controls if local curvature control is preferred.

Implement `r` as a continuous normalized parameter in `[-1,1]` even if the first controlled study uses only `{-1,0,+1}`. This preserves the simple three-route screen while enabling a later preregistered interpolation test and a commanded-versus-executed lateral-offset regression. Success at the three anchors alone is not evidence of continuous-route generalization.

## 5. Recommended Z1 architecture

### Ante-impact

- Construct curves in the live nail frame.
- Anchor at the realized physical start `A(u,v)`.
- Pass exactly through the nail at `s=1`.
- Fix the incoming tangent to the nail axis and predeclare a nominal impact-speed profile.
- Use a clamped multi-span quintic spline, or the boundary-preserving polynomial family above. Do not use the six-control quintic from Vu et al. unchanged for multiple routes.
- Parameterize progress by curve arc length or a monotone forward-window projection. Do not retain projection onto the old straight chord for a substantially curved path.
- Specify an along-path velocity field or phase-rate law `s_dot = f(s,state)` with a predeclared impact-speed window. A path plus lateral position error alone is not a trajectory: a stationary hammer sitting on the curve can have zero geometric error.

### At and after impact

- Extend the executable ante-impact branch slightly beyond the nominal nail state, so the desired nonzero-velocity hit is an interior state and late contact remains defined.
- Define an overlapping post-impact branch around the same nominal contact state and switch on the actual first-contact event.
- Require positional compatibility at contact. Do not require the real robot velocity or acceleration to be continuous through the physical impulse.
- If a post-impact velocity reference is used, make it compatible with an impact model or use a position-dominant interim phase as reference-spreading work recommends.
- Treat this branch as a later upgrade unless follow-through is an explicit policy objective or an observed failure. For the immediate ante-impact conditioning study, contact-triggered logging plus the existing task dynamics is sufficient and avoids adding an unnecessary policy mode.

### Policy information

The desired route must be identifiable at the first action. At minimum expose normalized route command `r`; for a richer family expose a small look-ahead description such as the next waypoint/tangent or route parameters. Also expose or encode the desired impact direction/speed: a target position alone does not define a strike. The physical start is visible through proprioception, but normalized `(u,v)` may still be included as an explicit task parameter if it materially simplifies identification. Keep command information separate from tracking error: an error that is initially zero cannot communicate route intent.

The smallest sufficient task context is therefore: robot state; hammer pose/twist relative to the nail frame; spatial phase; desired impact velocity/orientation; a short look-ahead/tangent or explicit curve parameters; route ID when routes vary; and contact/event state after impact. Raw `(u,v)` can be omitted only after an executable uniqueness test proves that the retained observation identifies the future curve over every qualified start/route cell. Instantaneous reference position alone is insufficient if curves approach or cross.

This remains a single online-RL policy. The spline supplies a task reference/command; it is not a second learned policy or a generate-then-track architecture.

### Define the estimand before training

Two scientifically different questions must not share one pass criterion:

1. **Route-command compliance:** can one policy execute the requested left/straight/right reference? This requires a persistent, calibrated route objective or explicit route-following gate. Route separation and tracking error are primary endpoints.
2. **Exploration benefit:** does exposure to multiple reference shapes improve task learning, constraint behavior, or robustness? Here route separation is not itself success; the comparison is against a single-route baseline on task/safety endpoints.

The present horizontal pilot mostly asked the first question while using a deliberately weak route objective. A future experiment must declare which estimand it targets; otherwise ignoring an arbitrary route can be the return-maximizing behavior and cannot diagnose the spline representation.

### Z1 action-semantics gate

Physical start resets interact with the current absolute joint-position action around one default pose. Before accepting a start-pose library, verify for every cell:

- the normalized action required to hold the reset pose is within the qualified action range;
- action/target buffers initialize without a first-step snap back to the old default;
- clipped position targets can realize the required path;
- VIC stiffness/gain state resets consistently and without a transient;
- one continuous, single-IK-branch joint-target tape realizes each complete curve, without cell-specific branch switching;
- one preregistered scripted VIC gain schedule is shared across cells during qualification, rather than tuning stiffness separately until each cell passes;
- the final joint-position-plus-stiffness action stack—not only Cartesian DiffIK playback—can execute the reference.

DiffIK playback is a geometry check. It is not sufficient evidence that the learned VIC action space can track the same route.

## 6. Staged experiment sequence

### Stage 0 — bank and freeze existing evidence

- Bank the completed horizontal-pilot result under `docs/results/` with its exact artifact hash and claim limits.
- Freeze the old tasks/checkpoints as historical treatments.
- No new training.

### Stage 1 — CPU geometry qualification

Create one deterministic candidate family and qualify it without PPO:

- exact start and nail interpolation;
- contact tangent/angular alignment;
- no upward command from any start;
- bounded curvature, acceleration, and jerk;
- monotone spatial phase and finite projection;
- IK success, collision clearance, joint limits, 500 Hz joint-velocity limits;
- productive strike, delivered impulse, per-joint impulse exposure, and first-contact gains;
- a predeclared along-path speed/phase-rate law and an accepted physical impact-speed window;
- exact execution with the intended joint-position-plus-stiffness action, actuator, shared scripted VIC-gain schedule, and CaT stack—not only Cartesian DiffIK;
- a continuous single-branch IK/joint-target realization over each full curve, with no per-cell branch or gain-schedule rescue.

Use a fixed grid on the start plane (center, axial lateral extremes, and diagonals) and all three route commands. Reject unreachable cells; do not silently resample them.

### Stage 2 — physical-start robustness

The boundary-preserving deformation changes both the reset pose and the anchored path from that pose. It is therefore a **coupled physical-start plus anchored-path robustness test**, not a pure start-state intervention.

- Keep one categorical route only and use several qualified physical starts.
- Evaluate the center-trained baseline at every start before retraining.
- Compare it with one mixed-start policy; small per-start specialist policies are optional feasibility ceilings.
- Explicitly distinguish training-grid points from held-out/interleaved evaluation points.

If a pure reset-adaptation estimand is required, add a deliberately slow prefix from each reset to one common staging state `M`, then use the identical `M -> nail` strike segment for all trials. `M` must be a full state within preregistered tolerances—joint pose/velocity, hammer pose/twist, and VIC gain state—not only a common Cartesian head position. That design isolates recovery from the reset pose, but it asks a different question from adapting a strike to begin at each sampled start.

### Stage 3 — isolate route conditioning

- Center start only.
- Left/straight/right boundary-preserving spline routes.
- Explicit route command from the first action.
- Same task/reward/CaT/action/reset/training budget as the qualified baseline.
- Compare three route-specialist policies (learnability ceilings), one shared route-conditioned policy, and a route-unconditioned or constant-command control with the same observation dimensions.
- Use a calibrated persistent route objective for a route-compliance experiment; do not expect an arbitrary command to be followed after its only distinguishing reward has annealed away.

For both mixed-start and mixed-route comparisons, report two compute views: (1) equal total environment steps, which measures deployment/compute efficiency, and (2) equal per-cell exposure or cell-normalized learning curves, which tests representational learnability. Under equal total compute, a three-route shared policy sees roughly one third of the samples per route received by a specialist; failure in that comparison alone does not establish inadequate conditioning capacity.

### Stage 4 — combine only after both pass

- Small start x route factorial design.
- Train on a preregistered subset and evaluate both matched training cells and held-out start/route combinations.
- In any held-out cell design, every start level and every route level must appear somewhere in training. Otherwise the test confounds compositional generalization with an unseen categorical command or unseen start domain.
- Use independent training seeds as the inferential unit; thousands of rollout worlds do not replace training-seed replication.

### Stage 5 — curriculum and domain randomization

- Use curriculum only if every cell learns independently but the mixture fails from scratch.
- Add domain randomization only after nominal start/route conditioning works. DR tests robustness/sim-to-real, not whether the trajectory representation is identifiable.

## 7. Decision gates and claim limits

The generator passes geometry qualification only if every declared cell is reachable, finite, collision-free, exact at the nail, and productive under the scripted controller. The learned policy passes conditioning only if routes separate under forced commands while retaining success, delivered impulse, joint-velocity behavior, and per-joint impulse evidence.

The literature does not supply universal numeric gates for this Z1 fixture. Before implementation or data access, a separate executable spec must freeze thresholds for contact-face angular/position error, premature claw/shaft/table contact, clearance, projection uniqueness, impact-speed error, effort, gain transients, 500 Hz joint velocity, per-joint impulse frequency and tail severity, route tracking, and worst-cell task success. “Acceptable” or “bounded” is not a preregistered result criterion.

The minimum controlled matrix is:

| Question | Required comparisons | What a pass means |
|---|---|---|
| Physical-start robustness | center-trained baseline evaluated at all starts; mixed-start policy; optional per-start specialists | adaptation over the declared plane without unacceptable action-reset or safety changes |
| Route-command compliance | route specialists; one shared conditioned policy; equal-width unconditioned/constant-command control | the shared policy responds to the requested route rather than merely selecting one successful strike |
| Start x route composition | factor levels all observed separately, selected combinations held out | interpolation of known factors within this bounded family, not arbitrary generalization |

Claims must remain narrow:

- Training and evaluation on the same sampled plane support robustness over that tested distribution.
- Success on preregistered held-out plane points supports interpolation within that bounded plane, not arbitrary spatial generalization.
- A spline reference does not prove hard constraint enforcement.
- A successful route-conditioned policy does not establish that the spline is optimal.
- Changing start points also changes robot posture, effective mass, available path length, and possibly impact energy. Those are measured mediators, not nuisance details that can be ignored.

## 8. Recommended decision

**Yes, use spline geometry next—but after this deep dive, use it precisely:**

- deterministic, nail-frame, endpoint-constrained and contact-split;
- sufficient interior degrees of freedom for route shape;
- real physical start-plane resets;
- explicit route intent from the first policy action;
- spatial/curve-relative phase;
- CPU qualification before any GPU training.

Do **not** copy the paper's single quintic verbatim, do not call a virtual reference shift a new physical start, and do not introduce DMP/ProMP/KMP machinery until the small deterministic family has answered the simpler scientific question.

## Primary source matrix

| Source | Directly supports | Important limitation for this thesis |
|---|---|---|
| [Vu et al. 2026](https://doi.org/10.1109/AMC67705.2026.11435814) | Exact nail endpoint and specified endpoint velocity/acceleration with a fifth-order Bézier; QP hammering | One fixed-duration pre-impact curve; no free route control after endpoint jets; no post-impact reference; simulation only |
| [Flash and Hogan 1985](https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985) | Integrated-squared-jerk objective for smooth point-to-point motion | Human reaching; no impact and no evidence for a rest-to-rest strike |
| [Piazzi and Visioli 2000](https://doi.org/10.1109/41.824136) | Joint-space cubic splines and constrained global minimax-jerk timing | Generic manipulator simulation, not an impact method |
| [Gasparetto and Zanotto 2007](https://doi.org/10.1016/j.mechmachtheory.2006.04.002) | Fifth-order B-spline via-point trajectories with time/jerk optimization and derivative bounds | No impact or comparison against Bézier on the same task |
| [Jia, Gardner, and Mu 2019](https://doi.org/10.1177/0278364918817116) | Online quartic replanning to a desired impact position and nonzero velocity while matching current `p/v/a` | Batting; planner ends at impact and supplies no follow-through branch |
| [Kober et al. 2010](https://doi.org/10.1109/ROBOT.2010.5509672) | Hitting point plus nonzero hitting velocity; strike/follow-through phase composition; adaptation to different hit locations | Table tennis, not contact-rich hammering or impulse constraints |
| [Ijspeert et al. 2013](https://doi.org/10.1162/NECO_a_00393) | Stable DMP framework and start/goal/time adaptation | Standard terminal attractor is not a nonzero-velocity nail impact boundary |
| [Ude et al. 2010](https://doi.org/10.1109/TRO.2010.2065430) | Task-conditioned DMP generalization and explicit need to cluster distinct route modes | Does not address rigid impact or per-joint constraints |
| [Paraschos et al. 2013](https://papers.nips.cc/paper_files/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) | Conditioning trajectory distributions on position/velocity/via points | Needs multiple demonstrations; soft probabilistic conditioning is not an exact safety boundary |
| [Gómez-González et al. 2016](https://doi.org/10.1109/HUMANOIDS.2016.7803322) | ProMP adaptation for striking position, velocity, orientation, and timing | Moving-ball interception, not post-impact contact or CaT |
| [Dragan et al. 2015](https://publications.ri.cmu.edu/movement-primitives-via-optimization) | New-start/new-goal adaptation as constrained shape-preserving optimization | Shape preservation depends on the chosen norm |
| [Calinon, Bruno, and Caldwell 2014](https://doi.org/10.1109/ICRA.2014.6907339) | Multiple task-attached coordinate frames for movement and impedance adaptation | Demonstration-based; impact event still has to be designed |
| [Calinon 2015/2018](https://doi.org/10.1007/978-3-319-60916-4_7) | Task-frame generalization under changed object/landmark poses | Demonstration/model overhead; impact transition must still be designed |
| [Huang et al. 2019](https://doi.org/10.1177/0278364919846363) | KMP adaptation with additional constraints, position/velocity via points, and local coordinate systems | More machinery/data than the first three-route experiment needs |
| [Koç et al. 2018](https://doi.org/10.1016/j.robot.2018.03.012) | Free-time striking trajectory optimization; fixed hitting plane can be restrictive | Moving-object sport; does not directly define our start-state plane |
| [Stouraitis et al. 2020](https://doi.org/10.1109/IROS45743.2020.9341246) | Hybrid impact-aware optimization of trajectory, timing, force, and compliance | Model/optimization heavy; different task objective |
| [Ti et al. 2024](https://arxiv.org/abs/2402.05502) | Real nail hammering; initial posture and directional manipulability affect impact capability | Plans tool/grasp/posture rather than learned multi-route following |
| [Konno et al. 2011](https://doi.org/10.1177/0278364911405870) | Humanoid whole-body impact-state optimization with an impact model | Does not establish a spline basis or learned route family |
| [Teramae et al. 2018](https://doi.org/10.3389/fnbot.2018.00071) | Real collision-terminated hammer command replay; terminal angle/velocity relevance | One DoF, pneumatic actuation, no multi-joint spline or hybrid post-impact reference |
| [Biemond et al. 2013](https://doi.org/10.1109/TAC.2012.2223351) | Why conventional time tracking is ill posed under impact-time mismatch | Control theory, not trajectory learning |
| [Rijnen, Saccon, and Nijmeijer 2020](https://doi.org/10.1109/TCST.2019.2898953) | Physical impact-triggered switching between extended ante/post reference branches | One-DoF tracking/control, not route generation |
| [van Steen et al. 2023](https://doi.org/10.23919/ACC55779.2023.10156028) and [2024](https://arxiv.org/abs/2411.09870) | Ante/post reference fields, impact switching, time-invariant state-based execution | Full method needs an impact model/controller; here it supplies design principles |
| [Sidiropoulos and Doulgeri 2024](https://doi.org/10.1007/s10846-024-02051-0) | DMP adaptation under initial/final and dynamic via-point constraints | General manipulation evidence, not striking-specific validation |
| [Werling et al. 2010](https://doi.org/10.1109/ROBOT.2010.5509799) | Task-aligned Frenet coordinates; separate longitudinal/lateral polynomial candidate generation | Road/lane and vehicle assumptions do not transfer |
| [Fan et al. 2018](https://arxiv.org/abs/1807.08048) | Industrial path/speed separation; candidate search followed by spline refinement | Full autonomous-driving stack, not a manipulator recipe |
| [Pivtoraiko, Knepper, and Kelly 2009](https://doi.org/10.1002/rob.20285) | Finite library of feasible state-connecting motion primitives | Large lattice and nonholonomic vehicle constraints are unnecessary here |
| [McNaughton et al. 2011](https://doi.org/10.1109/ICRA.2011.5980223) | Task-conformal spatiotemporal lattice and smooth motion primitives | Automotive timing/road topology; far larger search than three routes need |
| [Mellinger and Kumar 2011](https://doi.org/10.1109/ICRA.2011.5980409) | Piecewise polynomial waypoint trajectory with derivative constraints | Minimum snap is quadrotor-motivated, not universally optimal |
| [Richter, Bry, and Roy 2016](https://doi.org/10.1007/978-3-319-28872-7_37) | Sparse multi-segment polynomial optimization and segment-time allocation | Quadrotor flatness/actuation assumptions do not transfer |
| [Usenko et al. 2017](https://doi.org/10.1109/IROS.2017.8202160) | Uniform B-spline local support and efficient local reshaping | Replanning/obstacle-map system is outside this task; exact contact still needs explicit constraints |
| [Gao et al. 2018](https://doi.org/10.1109/ICRA.2018.8462878) | Bernstein convex-hull property for whole-curve corridor and derivative constraints | UAV safe-corridor planner; corridor construction is likely unnecessary initially |
| [Kunz and Stilman 2012](https://doi.org/10.15607/RSS.2012.VIII.027) | Explicit separation of geometric path from feasible timing | Joint velocity/acceleration timing only; no impact objective |
| [Pham and Pham 2018](https://doi.org/10.1109/TRO.2018.2819195) | Reachability-based time-optimal path parameterization under constraints | Retimes a fixed path; cannot choose route geometry or guarantee impulse safety |
| [Berscheid and Kroeger 2021](https://doi.org/10.15607/RSS.2021.XVII.015) | Jerk-limited generation from arbitrary initial to arbitrary target `p/v/a` | State-to-state timing/splicing only; no curved-route or impact model |
| [Wang et al. 2022, GCOPTER](https://doi.org/10.1109/TRO.2022.3160022) | Sparse joint optimization of polynomial waypoints, durations, and corridor constraints | More general optimization machinery than the first three-route experiment needs |
| [Zucker et al. 2013, CHOMP](https://doi.org/10.1177/0278364913488805) | Smoothness/obstacle functional optimization over trajectories | General local optimization is unnecessary for the first fixed-scene family |
| [Schulman et al. 2014, TrajOpt](https://doi.org/10.1177/0278364914528132) | Sequential convex trajectory optimization and continuous-time collision checking | Higher implementation/interpretation cost than a compact spline |

## Search protocol and limitations

This was a targeted primary-source scoping review, not a systematic review or meta-analysis. Sources were retained when an official publisher, proceedings, institutional, author, or arXiv copy exposed at least one relevant mechanism: curve/order, boundary derivatives, via-point or task-frame adaptation, nonzero impact state, spatial/time/event phase, or pre/post-impact switching. Surveys, vendor pages, blogs, accidental-collision work, and papers without an inspectable trajectory mechanism were excluded as evidence.

Searches were run on 2026-08-22 using combinations of:

- `robot hammering trajectory generation B-spline Bézier impact velocity`;
- `robot striking movement primitive hitting position velocity start generalization`;
- `minimum jerk spline impact trajectory endpoint velocity acceleration`;
- `ProMP KMP conditioning position velocity via point striking`;
- `task-parameterized movement nail frame start goal coordinate system`;
- `impact-aware trajectory optimization contact timing compliance`;
- `reference spreading impact ante post time invariant spatial phase`;
- `autonomous driving Frenet trajectory generation polynomial candidate lattice`;
- `UAV minimum snap polynomial B-spline Bernstein safe corridor trajectory generation`;
- `manipulator path parameterization TOPP velocity acceleration trajectory`;
- `CHOMP STOMP TrajOpt direct collocation trajectory generation`;
- exact-title and DOI queries for every retained paper.

The review intentionally covers seminal work from 1985 onward because the foundational spline and movement-primitive mechanisms are older than the current RL literature. Results across papers are not directly comparable, and no retained paper compares Bézier, B-spline, DMP, ProMP, and hybrid optimization under one impact task. “Exact” in this note refers to the mathematical reference constraints; controller bandwidth, discretization, compliance, and contact uncertainty prevent an exact curve from guaranteeing exact physical contact.

## AI disclosure

This note was prepared with repository code inspection, direct inspection of the user-supplied six-page IEEE paper, primary-source web searches, and parallel research agents separated into impact-trajectory, movement-primitive/start-plane, autonomous-driving/mobile, UAV/manipulator, foundational-method verification, and repository-ground-truth streams. Source claims were checked against official publisher, author, institutional, proceedings, or arXiv pages. Technical recommendations and the `beta/gamma` deformation are explicitly marked as synthesis rather than attributed to a paper. Sanitized aggregate design summaries were also sent to Kimi K3 and DeepSeek V4 Pro for advisory critiques after the first primary-source synthesis; a GLM-5.2 call returned no model content. They received no raw traces, checkpoints, repository paths/hashes, credentials, or personal information, and their opinions were not treated as source evidence. No model training, simulation run, or trajectory code implementation was performed for this review.
