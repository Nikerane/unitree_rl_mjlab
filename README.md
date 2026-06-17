# Unitree RL Mjlab


## ✳️ Overview
Unitree RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its 
physics simulation backend, currently supporting Unitree Go2, A2, As2, G1, R1, H1_2 and H2.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 🔨 Z1 Hammer-Nail (`hammer-z1` branch)

This branch adds **`Unitree-Z1-Hammer`**: Z1 arm + differential IK hammer control.
Assets live in the sibling repo [`safe_impact_manipulation`](https://github.com/Nikerane/safe_impact_manipulation)
(`hammer_z1_env/assets/`). Docs: [reward design](docs/research/reward-design/)
(historical snapshots in [docs/archive/](docs/archive/)).

```bash
export MUJOCO_GL=egl
python scripts/train.py Unitree-Z1-Hammer --agent.logger tensorboard --env.scene.num-envs 64
python scripts/play.py Unitree-Z1-Hammer --checkpoint-file logs/rsl_rl/z1_hammer/<run>/model_xx.pt
pytest tests/ -m "not integration"
```

### Reward design

At each control step $t$ the total reward is a weighted sum of seven terms:

$$
R_t \;=\; \sum_{k} w_k\, r_k
\;=\; w_{\text{app}}\,r_{\text{app}}
   + w_{\text{drv}}\,r_{\text{drv}}
   + w_{\Delta}\,r_{\Delta}
   + w_{\text{imp}}\,r_{\text{imp}}
   + w_{\text{cmp}}\,r_{\text{cmp}}
   + w_{\text{act}}\,r_{\text{act}}
   + w_{\text{jl}}\,r_{\text{jl}}
$$

**Notation.** $d_t$ = nail depth (slide qpos), goal $d_{\text{goal}}=0.075$ m, success $d_{\text{succ}}=0.07$ m; $\bar d_t=\max_{\tau\le t} d_\tau$ = max depth so far (reset to the $4\,\text{mm}$ settling dead-zone); $p_h,p_n$ = hammer-head / nail-top world positions; $\dot{x}_h$ = hammer-head world velocity (finite difference of position); $\hat{n}=(0,0,-1)$ = nail strike axis; $v_{\text{axial}}=\max\!\big(0,\,\hat{n}\cdot\dot{x}_h\big)$ = downward (axial) impact speed; $c_t$ = first-contact flag (true only on the step a contact begins); $a_t$ = action; $q_i$ = joint $i$ with soft limits $q_i^{\min},q_i^{\max}$; control period $\Delta t=0.02$ s.

**Momentum-gated impact reward** — pays axial impact speed *only* when a fresh contact actually advances the nail (double-gated, so a glancing scrape or any contact that makes no progress earns nothing):

$$
r_{\text{imp}}
\;=\;
\frac{v_{\text{axial}}}{v_{\exp}}
\;\cdot\;
\mathbb{1}\!\left[\,c_t\,\right]
\;\cdot\;
\mathbb{1}\!\left[\,d_t - \bar d_{t-1} > \varepsilon\,\right],
\qquad
v_{\exp}=1\ \text{m/s},\quad \varepsilon = 5\times 10^{-4}\ \text{m}.
$$

| Term | $r_k$ | $w_k$ | Purpose |
|---|---|---|---|
| `approach`         | $\exp\!\big(-\lVert p_h - p_n\rVert^2 / \sigma_a^2\big),\ \ \sigma_a = 0.08\ \text{m}$ | $+0.1$ | Guide arm near nail |
| `nail_driven`      | $\exp\!\big(-(d_{\text{goal}} - d_t)^2 / \sigma_d^2\big),\ \ \sigma_d = 0.03\ \text{m}$ | $+2.0$ | Pull toward full depth (near-zero until ~30 mm driven) |
| `nail_depth_delta` | $\max\!\big(0,\ d_t - \bar d_{t-1}\big)$ | $+600$ | Reward every new mm of nail travel |
| `impact_progress`  | $\dfrac{v_{\text{axial}}}{v_{\exp}}\,\mathbb{1}[c_t]\,\mathbb{1}\!\left[d_t-\bar d_{t-1}>\varepsilon\right]$ | $+8$ | **(new)** Reward fast, *productive* strikes |
| `completion`       | $\mathbb{1}\!\left[d_t \ge d_{\text{succ}}\right]$ | $+100$ | Sparse success bonus |
| `action_rate`      | $\lVert a_t - a_{t-1}\rVert^2$ | $-0.01$ | Penalise jerky motion |
| `joint_pos_limits` | $\sum_i\big[\max(0,\,q_i - q_i^{\max}) + \max(0,\,q_i^{\min} - q_i)\big]$ | $-10$ | Penalise exceeding soft joint limits |

> **Status / changelog.** This block reflects the agreed reward design (changes **#1 + #2**); see [`docs/research/hammering_reward_design_deep_dive_v2.md`](docs/research/hammering_reward_design_deep_dive_v2.md) §4.3, §5.A.
> - **#1 rebalance:** `nail_depth_delta` weight $2000 \to 600$, so the full-drive dense total (~30–40) no longer dwarfs the $+100$ completion bonus.
> - **#2 new term:** `impact_progress` is the double-gated reward above. On this **position-only** differential-IK arm the controllable impact lever is end-effector axial *momentum*, not commanded contact force — so the term rewards pre-impact axial speed, gated on actual nail progress.
> - The previous README listed a `strike_vel` term ($w=5.0$); that term is **not** present in the code (`src/tasks/hammer/`) and has been removed here.


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


## 🛠️ Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Mjlab-Velocity-Flat-Unitree-G1) specifies the training task.
Available velocity tracking tasks:
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. Motion Imitation Training

Train a Unitree G1 to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 2.1 Prepare Motion Files

Prepare csv motion files in mjlab/motions/g1/ and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz files will be stored at:**：`src/motions/g1/...`

#### 2.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

Available tasks:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Motion imitation:
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. Real Deployment

Before deployment, install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 4.1 Power On the Robot
Start the robot in suspended state and wait until it enters `zero-torque` mode.

#### 4.2 Enable Debug Mode
While in `zero-torque` mode, press `L2 + R2` on the controller. The robot will enter `debug mode` with joint damping enabled.

#### 4.3 Connect to the Robot
Connect your PC to the robot via Ethernet. Configure the network as:
- Address：`192.168.123.222`
- Netmask：`255.255.255.0`

Use `ifconfig` to determine the Ethernet device name for deployment.

#### 4.4 Compilation

Example: Unitree G1 velocity control.
Place `policy.onnx` and `policy.onnx.data` into: `deploy/robots/g1/config/policy/velocity/v0/exported`.
Then compile:

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

#### 4.5 Deployment

## 4.5.1 Simulation Deployment

Before deploying on the real robot, it is recommended to perform simulation deployment using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
to prevent abnormal behaviors on the physical robot. This framework has already integrated it.

Build unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

Launch the simulator (note that a gamepad must be connected):

```bash
./simulate/build/unitree_mujoco
```

You can select the corresponding robot in `simulate/config`

Launch the simulation control program:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

## 4.5.2 Real-Robot Deployment

Launch the control program on the real robot:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

**Arguments**：
- `network`: The network interface used to connect to the robot. Use `lo` for simulation deployment, and `enp5s0` for the real robot(You can check it using the `ifconfig` command) 

</div>

**Deployment Results**：

| Go2                                                    | G1                                                    | H1_2           | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|----------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
