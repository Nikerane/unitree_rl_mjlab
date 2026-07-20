# Fixed-Impedance Z1 Hammering: Exact Mathematical Formulation

**Progress-report formulation, 2026-07-20.** This transcribes branch `soft-cat` at commit
`9191709`. Code is authoritative. Equations labelled **thesis objective** state the intended
constrained problem; equations labelled **shipped** reproduce the implementation. Sources use
`file:start-end` line citations.

Scope: registered task `Unitree-Z1-Hammer-CaT-Impulse` (fixed gains, position-only DiffIK,
delivered-impulse reward, and per-joint impulse instrumentation). The optional `-Track` sibling adds
only the imitation prior and curriculum (`src/tasks/hammer/config/z1/__init__.py:91-120`).

## 1. Constrained problem (CMDP)

### 1.1 Thesis objective

Let
$$
\mathcal M_c=(\mathcal S,\mathcal A,P,\rho_0,r,\gamma,T,\{g_j,L_j\}_{j=1}^{6}),
\qquad \gamma=0.99,\quad T\le1000.
$$
Here $s_t\in\mathcal S$ is the Markov simulator/controller state, $a_t\in\mathcal A$ is the policy
action, $P$ is the MuJoCo transition kernel, and $\rho_0$ is the reset distribution. The horizon is
$20/0.02=1000$ control steps unless success occurs first. Sources:
`src/tasks/hammer/config/z1/rl_cfg.py:32-47`, `src/tasks/hammer/hammer_env_cfg.py:298-310`.

The intended policy solves
$$
\boxed{\begin{aligned}
\pi^*&\in\arg\max_\pi J(\pi)
=\arg\max_\pi\mathbb E_{\rho_0,P,\pi}\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],\\
\text{s.t.}\quad g_j(s_t,a_t)&:=\Lambda_{t,j}-L_j\le0,
\quad\forall t<T,\ \forall j\in\{1,\ldots,6\},\quad\text{a.s.}
\end{aligned}}
$$
Equivalently,
$$\Pr_\pi\!\left[\max_{0\le t<T}\max_{1\le j\le6}\Lambda_{t,j}/L_j\le1\right]=1.$$
This is a state-wise, per-control-step worst-case constraint, not an expected cumulative-cost
budget. The return includes the positive object-side delivered-impulse term; the constraint bounds
the opposite, robot-side per-joint reaction quantity.

### 1.2 Shipped status

The raw margin is exactly
$$c_{t,j}=\Lambda_{t,j}-L_j.$$
and is not a reward (`src/tasks/hammer/cat/constraints.py:35-52`). However, the registered impulse
arm has `imp_max_p=0.0`, hence $\delta_t^{\rm imp}\equiv0$: impulse-CaT is currently log-only and
does not enforce the constraint (`src/tasks/hammer/config/z1/env_cfgs.py:291-313`,
`src/tasks/hammer/cat/hook.py:206-220`). The boxed CMDP is therefore the thesis target, not a hard
guarantee claimed for the current fixed-impedance run.

## 2. MDP used by the policy

### 2.1 37-dimensional observation

Strictly, the 37-vector is the policy observation $o_t$, not complete Markov state $s_t$; the latter
also contains MuJoCo state and stateful reward/reference/impulse buffers. Actor and critic use the
same components; actor observations are noisy and critic observations clean
(`src/tasks/hammer/hammer_env_cfg.py:44-108`). With seven robot joints (arm 1--6 plus gripper),
$$
o_t=\operatorname{concat}(q_t-q_0,\dot q_t-\dot q_0,p^b_{ee,t},v^w_{ee,t},p^b_{h,t},v^w_{h,t},
p^w_{n,t},\max(q^n_t,0),\phi_t,p^*(\phi_t)-p^w_{h,t},a_{t-1})\in\mathbb R^{37}.
$$

| Component | Dim. | Exact implementation |
|---|---:|---|
| $q_t-q_0$ | 7 | Config: `hammer_env_cfg.py:47-50`; subtraction: installed `mjlab/envs/mdp/observations.py:51-61`. |
| $\dot q_t-\dot q_0$ | 7 | Config: `hammer_env_cfg.py:51-54`; subtraction: installed `mjlab/envs/mdp/observations.py:64-72`. |
| $p^b_{ee,t}=p^w_{ee,t}-p^w_{base,t}$ | 3 | `src/tasks/hammer/mdp/observations.py:45-53`. |
| $v^w_{ee,t}$ | 3 | Despite suffix `_b`, world-frame velocity: `observations.py:56-63`. |
| $p^b_{h,t}=p^w_{h,t}-p^w_{base,t}$ | 3 | `observations.py:24-32`. |
| $v^w_{h,t}$ | 3 | `observations.py:35-42`. |
| $p^w_{n,t}$ | 3 | `observations.py:66-72`. |
| $\max(q^n_t,0)$ | 1 | Lower-clamped only: `observations.py:112-129`. |
| $\phi_t$ | 1 | Monotone reference update: `observations.py:75-91`. |
| $p^*(\phi_t)-p^w_{h,t}$ | 3 | `observations.py:94-109`. |
| $a_{t-1}$ | 3 | Config: `hammer_env_cfg.py:101`; reader: installed `mjlab/envs/mdp/observations.py:80-83`. |
| **Total** | **37** | Pinned by `tests/test_env.py:64-73`. |

Actor noise is componentwise uniform: joint position $\pm0.01$, joint velocity $\pm1.5$, EE/head
position $\pm0.005$, EE/head velocity $\pm0.01$, nail-top position $\pm0.002$, and nail depth
$\pm0.001$; phase, reference error, and action have no noise (`hammer_env_cfg.py:47-101`).

### 2.2 Action, DiffIK, and fixed PD

PPO clips $a_t\in[-1,1]^3$ (`src/tasks/hammer/config/z1/rl_cfg.py:51-57`). The relative,
position-only hammer-head target is
$$\boxed{p^{\rm des}_{h,t}=p^w_{h,t}+\alpha_pa_t=p^w_{h,t}+0.15a_t.}$$
using `orientation_weight=0` (`src/tasks/hammer/hammer_env_cfg.py:110-123`) and
$\alpha_p=0.15$ m (`src/assets/robots/unitree_z1/z1_constants.py:222-229`).

At each physics substep, mjlab solves
$$
(J_p^\top J_p+\lambda_{IK}^2I)\Delta q=J_p^\top(p^{\rm des}_{h,t}-p^w_h),
\quad\lambda_{IK}=0.05,
$$
then $\Delta q\leftarrow\operatorname{clip}(\Delta q,-0.5,0.5)$ and $q^*=q+\Delta q$. This is
installed mjlab 1.4.0 `mjlab/envs/mdp/actions/differential_ik.py:174-255`; repo wiring is
`hammer_env_cfg.py:114-122`, `config/z1/env_cfgs.py:116-128`.

The native fixed position actuator applies
$$
\tau^{raw}_{t,j}=k_{p,j}(q^*_{t,j}-q_{t,j})-k_{d,j}\dot q_{t,j},\qquad
\tau_{t,j}=\operatorname{clip}(\tau^{raw}_{t,j},-\tau_j^{max},\tau_j^{max}),
$$
with
$$
(k_{p,j},k_{d,j},\tau_j^{max})=
\begin{cases}(1500,150,60\ \mathrm{N\,m}),&j=2,\\
(1000,100,30\ \mathrm{N\,m}),&j\in\{1,3,4,5,6\}.
\end{cases}
$$
Gains/limits: `src/assets/robots/unitree_z1/z1_constants.py:77-96`; native actuator construction:
installed `mjlab/actuator/builtin_actuator.py:33-92`. Robot bodies also use gravity compensation
(`z1_constants.py:55-69`). Stiffness is not an action.

### 2.3 Timing and termination

$$h=\Delta t_{physics}=0.002\ \mathrm s,\quad D=10,\quad\Delta t_c=Dh=0.02\ \mathrm s\ (50\ \mathrm{Hz}).$$
Source: `src/tasks/hammer/hammer_env_cfg.py:298-310`. Define the reward-safe depth
$$d_t=\operatorname{clip}(q^n_t,0,d_{goal}),\qquad d_{goal}=0.032\ \mathrm m.$$
This is exact `clamped_nail_depth` (`src/tasks/hammer/mdp/rewards.py:23-34`); $d_{goal}$ is
`src/tasks/hammer/nail_block.py:46-50`. Success terminates at $d_t\ge d_{succ}=0.030$ m; otherwise
the episode truncates at 20 s (`nail_block.py:51-66`, `mdp/terminations.py:19-29`,
`hammer_env_cfg.py:233-243`).

## 3. Exact shipped reward

Code-name mapping: `approach`=`hammer_approach_reward`, `nail_driven`=`nail_driven_reward`,
`nail_depth_delta`=`NailDepthDeltaTerm`, `impact_progress`=`ImpactProgressTerm`,
`completion`=`completion_bonus`, `action_rate`=`action_rate_penalty`,
`delivered_impulse`=`DeliveredImpulseTerm`, and `r_imit`=`ImitationPriorTerm`; all task formulas use
the shared `clamped_nail_depth` defined in Section 2.3.

### 3.1 Aggregation

The configured reward rate and actual PPO reward are
$$
\bar r_t=\sum_k w_k r_{k,t},\qquad
\boxed{r_t^{total}=\Delta t_c\bar r_t=0.02\sum_k w_k r_{k,t}.}
$$
The $0.02$ factor is exact because mjlab defaults to `scale_by_dt=True`
(`mjlab/managers/reward_manager.py:30-63,116-133`). The tables list configured weights; e.g. the
terminal completion contribution to `reward_buf` is $0.02\times100=2$.

### 3.2 Task/progress terms

| Term | Weight | Exact raw formula | Source |
|---|---:|---|---|
| `approach` | $0.1$ | $\exp(-\|p^w_{h,t}-p^w_{n,t}\|^2/0.08^2)$ | `mdp/rewards.py:53-71`; `hammer_env_cfg.py:151-163`. |
| `nail_driven` | $0.5$ | $\exp(-(0.032-d_t)^2/0.013^2)$ | `rewards.py:37-50`; current config `hammer_env_cfg.py:164-183`. |
| `nail_depth_delta` | $600$ | $[d_t-m_{t-1}]_+$; $m_{-1}=0.004$, $m_t=\max(m_{t-1},d_t)$ | `rewards.py:98-141`; `hammer_env_cfg.py:184-193`. |
| `completion` | $100$ | $\mathbf1[d_t\ge0.030]$ | `rewards.py:82-95`; `hammer_env_cfg.py:211-220`. |

Here $[x]_+=\max(x,0)$. The 4 mm initial maximum is a real dead zone.

### 3.3 Impact-maximization terms

Let $n=(0,0,-1)$ and
$$
\hat v^w_{h,t}=\begin{cases}0,&\text{first step after reset},\\
(p^w_{h,t}-p^w_{h,t-1})/\Delta t_c,&\text{otherwise},\end{cases}
\qquad v_t^{axial}=[\hat v^w_{h,t}\cdot n]_+.
$$
With $m^I_{-1}=0$, $m^I_t=\max(m^I_{t-1},d_t)$,
$$
r_t^{impact}=\frac{v_t^{axial}}{1.0\ \mathrm{m\,s^{-1}}}
\mathbf1[\text{first contact in step }t]\mathbf1[d_t-m^I_{t-1}>5\times10^{-4}\ \mathrm m].
$$
Its weight is $8$. Exact finite difference, clamp, max-depth update, and gates:
`src/tasks/hammer/mdp/rewards.py:175-211`; config: `hammer_env_cfg.py:194-210`.

Let $I_t$ be Section 4.2's delivered impulse, credited impulse $C^I_{-1}=0$, and reward max depth
$m^D_{-1}=0$. Then
$$
\Delta I_t=[I_t-C^I_{t-1}]_+,\quad A^D_t=\mathbf1[d_t-m^D_{t-1}>5\times10^{-4}],\quad
r_t^{delivered}=A^D_t\frac{\Delta I_t}{I_{ref}},\quad I_{ref}=0.6094\ \mathrm{N\,s},
$$
followed every step by
$$C^I_t=\max(C^I_{t-1},I_t),\qquad m^D_t=\max(m^D_{t-1},d_t).$$
Thus non-progress impulse is discarded, not banked. Weight $2.0$. Source:
`src/tasks/hammer/mdp/rewards.py:214-276`; normalizer/weight:
`src/tasks/hammer/config/z1/env_cfgs.py:37-45,345-358`.

### 3.4 Regularizers

| Term | Weight | Exact raw formula | Source |
|---|---:|---|---|
| `action_rate` | $-0.01$ | $\|a_t-a_{t-1}\|_2^2$ | `mdp/rewards.py:74-79`; `hammer_env_cfg.py:222-225`. |
| `joint_pos_limits` | $-10$ | $\sum_{j=1}^7([q_j^{low}-q_{t,j}]_+ + [q_{t,j}-q_j^{high}]_+)$ | Installed mjlab `mjlab/envs/mdp/rewards.py:81-96`; `hammer_env_cfg.py:226-230`. |

The joint-limit term includes all seven robot joints (`joint_names=(".*",)`).

### 3.5 Optional imitation prior (`-Track` only)

$$
r_t^{imit}=\exp(-\|p^w_{h,t}-p^*(\phi_t)\|^2/0.05^2)
\mathbf1[\text{no hammer--nail contact observed yet in the episode}].
$$
The latch ORs `found`, `current_contact_time>0`, and `last_contact_time>0`, including a contact wholly
inside one control interval (`src/tasks/hammer/mdp/rewards.py:279-375`). The waypoint is
$$
p^*(\phi)=\begin{cases}p_0+2\phi(p_{apex}-p_0),&0\le\phi<0.5,\\
p_{apex}+2(\phi-0.5)(p_{target}-p_{apex}),&0.5\le\phi\le1,
\end{cases}
$$
with interpolation arguments clipped to $[0,1]$ (`mdp/references.py:254-262`). Its stagewise weight is
$$
w_{imit}(n)=\begin{cases}
0.10,&0\le n<1200,\\0.08,&1200\le n<2400,\\0.06,&2400\le n<3600,\\
0.04,&3600\le n<4800,\\0.02,&4800\le n<6000,\\0,&n\ge6000.
\end{cases}
$$
Config: `hammer_env_cfg.py:245-275`; installed stage semantics:
`mjlab/envs/mdp/curriculums.py:110-148`.

## 4. Three distinct impulse quantities

All are evaluated at physics-substep period $h=0.002$ s.

### 4.1 Robot-side quantity read by the constraint

Index substeps by $k$, control steps by $t$, and $\mathcal K_t=\{10t,\ldots,10t+9\}$. Let
$G_k\in\{0,1\}$ indicate hammer-face--nail contact, $Q_{k,j}$ be arm-joint `qfrc_constraint`, and
$B_{k,j}$ its most recent off-contact value, updated only when $G_k=0$ and frozen during contact.
Because `subtract_baseline=True`,
$$u_{k,j}=hG_k|Q_{k,j}-B_{k,j}|.$$
For $W=25$ (50 ms), with zero padding before episode start,
$$
R_{k,j}=\sum_{\ell=k-W+1}^{k}u_{\ell,j},\qquad
P_{t,j}=\max_{k\in\mathcal K_t}R_{k,j},\qquad
\boxed{\Lambda_{t,j}=\max(P_{t,j},R_{10t+9,j}).}
$$
Off-contact substeps contribute zero but still slide the time-based window. Exact buffer/latch:
`src/tasks/hammer/mdp/impulse_bound.py:115-194`; config:
`src/tasks/hammer/config/z1/env_cfgs.py:245-260`.

[^lambda]: **Important implementation/thesis mismatch.** $\Lambda_{t,j}$ is not clean ballistic
    contact impulse. `qfrc_constraint` sums friction, limits, welds, and contacts; code contact-masks
    it, subtracts a frozen pre-contact baseline, takes absolute value, and integrates a 50 ms window.
    It deliberately counts one window of sustained press reaction
    (`mdp/impulse_bound.py:14-36,72-112`).

### 4.2 Object-side delivered impulse

Let $F^w_{k,r}$ be the world-frame net contact force for sensor primary $r$ and $n=(0,0,-1)$. The
code sums projections before rectifying:
$$f_k^{axial}=\left[\sum_r F^w_{k,r}\cdot n\right]_+.$$
Let $A_k$ be event age, $O_k$ consecutive off-contact count, and $G_{k-1}$ previous contact. A rising
edge re-arms only after 25 off-contact substeps:
$$E_k=G_k(1-G_{k-1})\mathbf1[O_k\ge25].$$
On $E_k=1$, set $A_k\leftarrow0$. Then
$$
Y_k=G_k\mathbf1[A_k<25],\qquad I_k=I_{k-1}+h f_k^{axial}Y_k,\qquad
\boxed{I_t=\sum_{k\le10t+9}h f_k^{axial}Y_k.}
$$
Afterward $A$ increments on contact; $O$ resets on contact and otherwise increments. Reset uses
$O=25$, so first contact is armed. $I_t$ is episode-cumulative and monotone, with at most 25 payable
contact substeps per debounced event. Implementation: `mdp/impulse_bound.py:197-279`; config:
`config/z1/env_cfgs.py:232-243,274-281`.

### 4.3 Log-only ContactRow diagnostic

For hammer--nail constraint rows $\mathcal R_{h\leftrightarrow n}(k)$,
$$
Q^{row}_k=\sum_{r\in\mathcal R_{h\leftrightarrow n}(k)}J_{k,r,:}^{\top}f^{efc}_{k,r},\qquad
u^{row}_{k,j}=hG_k|Q^{row}_{k,j}|.
$$
This diagnostic uses an **uncapped per-contact-event** running sum, closes it on a falling edge,
max-latches closed events within a control step, and logs the episode peak of the worst joint.
Nothing consumes it for reward or $\delta$. Current location:
`src/tasks/hammer/mdp/contact_row_impulse.py:143-226,229-327`; wiring:
`config/z1/env_cfgs.py:261-273`.

## 5. Per-joint caps and hardware derivation

Arm order is $(j_1,\ldots,j_6)$ (`src/assets/robots/unitree_z1/z1_constants.py:207-217`). With
$$\tau^{rated}=(30,60,30,30,30,30)\ \mathrm{N\,m},\qquad \kappa=2.$$
and measured $\Delta t_{cap}\simeq27.3$ ms,
$$
L_j=\tau_j^{rated}\kappa\Delta t_{cap},\qquad
\boxed{L=(1.640,3.280,1.640,1.640,1.640,1.640)\ \mathrm{N\,m\,s}.}
$$
Source/provenance: `src/tasks/hammer/config/z1/env_cfgs.py:27-35`. These caps use about 27.3 ms,
whereas shipped $\Lambda$ uses 50 ms and the later reference contact is reported near 44 ms. Code
freezes the caps pending the open window/cap-pairing decision; the durations are not identical.

## 6. Soft-CaT relaxation

### 6.1 Normalizer, map, and soft OR

For batch environment $b$,
$$c_{b,t,j}=\Lambda_{b,t,j}-L_j,\qquad \hat c_{t,j}=\max_b[c_{b,t,j}]_+.$$
The impulse normalizer starts at $c_{seed}=10^{-3}$ N m s. A column's first violation seeds it to
$\max(\hat c_{t,j},c_{seed})$; later it updates only on positive-violation steps:
$$c^{max}_{t,j}\leftarrow\tau c^{max}_{t-1,j}+(1-\tau)\hat c_{t,j},\qquad\tau=0.95.$$
and otherwise stays unchanged; it is floored at $c_{seed}$ before use. Exact sparse per-column EMA:
`src/tasks/hammer/cat/hook.py:192-237`; config: `config/z1/env_cfgs.py:297-313`.

For constraint term $m$ and column $j$,
$$
\delta^{(m,j)}_{b,t}=\begin{cases}
p_{min}+\operatorname{clip}(c^{(m)}_{b,t,j}/c^{max,(m)}_{t,j},0,1)(p^{(m)}_{max}-p_{min}),
&c^{(m)}_{b,t,j}>0,\\0,&c^{(m)}_{b,t,j}\le0,
\end{cases}
$$
(`src/tasks/hammer/cat/constraint_manager.py:48-57`), and
$$\boxed{\delta_{b,t}=\max_m\max_j\delta^{(m,j)}_{b,t}.}$$
(`constraint_manager.py:76-80`, `cat/hook.py:239-250`).

### 6.2 Reward and return modification

Split the actual dt-scaled reward as $r_t^{total}=r_t^++r_t^-$, where $r_t^-$ contains
negative-weight `action_rate` and `joint_pos_limits`. CatPPO stores
$$\boxed{r_t^{CaT}=r_t^{total}-\delta_t r_t^+=(1-\delta_t)r_t^++r_t^-.}$$
Penalties remain unscaled. Exact positive reconstruction: `src/tasks/hammer/cat/hook.py:33-36,149-190`;
learner: `src/tasks/hammer/rl/cat_ppo.py:34-70`.

For timeout indicator $z_t$, CatPPO adds
$$r_t^{CaT}\leftarrow r_t^{CaT}+z_t(1-\delta_t)\gamma V(s_t).$$
and suppresses stock PPO's full bootstrap (`rl/cat_ppo.py:73-84`). For true hard-done indicator
$d_t^{hard}$, define $\chi_t=(1-\delta_t)(1-d_t^{hard})$. Exact GAE is
$$
\epsilon_t=r_t^{CaT}+\gamma\chi_tV(s_{t+1})-V(s_t),\qquad
A_t=\epsilon_t+\gamma\lambda_{GAE}\chi_tA_{t+1},\qquad R_t=A_t+V(s_t),
$$
with $\lambda_{GAE}=0.95$ (`rl/cat_ppo.py:86-106`, `config/z1/rl_cfg.py:32-47`).

### 6.3 Current log-only state

The registered impulse arm sets $p^{imp}_{max}=0$. The hook explicitly writes zero probabilities
before normalization, even if $p_{min}>0$:
$$\boxed{\delta^{imp}_{b,t}\equiv0.}$$
Source: `src/tasks/hammer/cat/hook.py:206-220`. Because impulse-only registration also sets
`use_vel=False`, aggregate $\delta_t=0$, $r_t^{CaT}=r_t^{total}$, and
$\chi_t=1-d_t^{hard}$. The code logs margins and impulse quantities but supplies no current
impulse-avoidance incentive.

## 7. Notation

| Symbol | Meaning | Units/dimension |
|---|---|---|
| $s_t,o_t$ | Full Markov state; policy observation | --; $\mathbb R^{37}$ |
| $a_t$ | Cartesian delta action | $[-1,1]^3$ |
| $p_h,p_{ee},p_n,p^*$ | Hammer, EE, nail-top, reference positions | m |
| $q,\dot q,q^*$ | Joint position, velocity, IK position target | rad; rad s$^{-1}$; rad |
| $q^n,d_t$ | Nail coordinate; clamped reward depth | m |
| $h,D,\Delta t_c$ | Physics period; decimation; control period | 0.002 s; 10; 0.02 s |
| $G_k$ | Hammer--nail contact indicator | $\{0,1\}$ |
| $Q_{k,j},B_{k,j}$ | `qfrc_constraint`; frozen off-contact baseline | N m |
| $u_{k,j},R_{k,j},P_{t,j},\Lambda_{t,j}$ | Substep contribution, rolling sum, latch, constraint read | N m s |
| $L_j$ | Per-joint cap | N m s |
| $F^w_{k,r},f_k^{axial}$ | Contact force; rectified axial force | N |
| $A_k,O_k,Y_k$ | Event age, off streak, payable indicator | substeps; substeps; $\{0,1\}$ |
| $I_t,I_{ref},C_t^I$ | Delivered total, normalizer, credited baseline | N s |
| $c_{t,j},c^{max}_{t,j}$ | Signed margin and EMA scale | N m s |
| $p_{min},p_{max},\delta_t$ | CaT floor, ceiling, aggregate probability | dimensionless |
| $r_t^+,r_t^-$ | dt-scaled positive/negative contributions | reward |
| $z_t,d_t^{hard},\chi_t$ | Timeout, hard done, continuation mask | $\{0,1\}$; $\{0,1\}$; $[0,1]$ |
| $\gamma,\lambda_{GAE}$ | Discount and GAE parameters | 0.99; 0.95 |
