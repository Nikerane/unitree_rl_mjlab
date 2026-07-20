"""Trace hammer-head trajectory for ALL current-scene (July) policies, from an identical fixed reset.
Build the CaT-Impulse env ONCE, reload each actor, roll out one deterministic episode, log the head
path at 2 ms substep resolution. Output JSON -> scratchpad for the plot script.

Only July / L6-fixture-scene policies are valid here (June gripper-era policies use a different scene).
"""
from __future__ import annotations
import glob, json
from dataclasses import asdict
import torch
import mjlab.tasks  # noqa
import src.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME

TASK = "Unitree-Z1-Hammer-CaT-Impulse"
ROOT = "logs/rsl_rl/z1_hammer"
OUT = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad/traj_all.json"
SEED = 12345

# (label, dir-prefix) — one representative seed per distinct July arm.
ARMS = [
    ("nf1_none",    "2026-07-15_19-36-35_nf1_none_s0"),
    ("nf1_track",   "2026-07-15_19-36-22_nf1_track_s0"),
    ("a05_none",    "2026-07-16_21-14-22_a05_none_s0"),
    ("a05_track",   "2026-07-16_21-18-58_a05_track_s0"),
    ("b1_noterm",   "2026-07-16_21-31-06_b1_noterm_s0"),
    ("g1_none",     "2026-07-17_00-21-56_g1_none_s0"),
    ("g1_track",    "2026-07-17_00-21-58_g1_track_s0"),
    ("g2_delivoff", "2026-07-17_00-21-56_g2_delivoff_s0"),
    ("dg0_nogate",  "2026-07-17_20-28-55_dg0_nogate_seed0"),
    ("dg1_gate",    "2026-07-17_20-28-55_dg1_gate_seed0"),
    ("ip24_ip",     "2026-07-17_21-43-22_ip24_ip_seed1"),
    ("af1_fixed",   "2026-07-19_19-25-38_af1_fixed_seed0"),
]

env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
agent_cfg = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device="cpu")

robot = base.scene["robot"]; nail = base.scene["nail_block"]; sensor = base.scene["hammer_nail_contact"]
rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(base.scene)
nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(base.scene)
head = lambda: robot.data.site_pos_w[0, rc.site_ids].squeeze(0).tolist()
depth_mm = lambda: float(nail.data.joint_pos[0, 0]) * 1000.0

results = {}
n0 = None
for label, prefix in ARMS:
    dirs = glob.glob(f"{ROOT}/{prefix}")
    ckpts = sorted(glob.glob(f"{dirs[0]}/model_*.pt")) if dirs else []
    if not ckpts:
        print(f"[skip] {label}: no ckpt ({prefix})"); continue
    ckpt = max(ckpts, key=lambda p: int(p.split("model_")[1].split(".pt")[0]))
    try:
        runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
    except Exception as e:
        print(f"[FAIL-load] {label}: {type(e).__name__}: {e}"); continue
    policy = runner.get_inference_policy(device="cpu")

    torch.manual_seed(SEED)  # identical reset for every policy
    obs, _ = env.reset()
    nt = nail.data.site_pos_w[0, nc.site_ids].squeeze(0).tolist()
    if n0 is None: n0 = nt
    sub = []  # substep head path
    orig = base.metrics_manager.compute_substep
    def rec():
        orig(); h = head()
        sub.append([h[0], h[1], h[2], bool((sensor.data.found > 0).any())])
    base.metrics_manager.compute_substep = rec
    ctrl = []  # per control step: (x,y,z,nail_mm,contact)
    for k in range(1, 16):
        with torch.inference_mode():
            a = policy(obs)
        obs = env.step(a)[0]
        h = head()
        ctrl.append([h[0], h[1], h[2], depth_mm(), bool((sensor.data.found > 0).any())])
        if int(base.episode_length_buf[0]) == 0:
            break
    base.metrics_manager.compute_substep = orig
    maxd = max(r[3] for r in ctrl)
    results[label] = dict(ckpt=ckpt.split("z1_hammer/")[1], substep=sub, ctrl=ctrl,
                          nail_top=nt, max_nail_mm=maxd, success=maxd >= 30.0, n_ctrl=len(ctrl))
    print(f"[ok] {label:12s} steps={len(ctrl):2d} max_nail={maxd:5.1f}mm "
          f"success={maxd>=30.0} substeps={len(sub)}")

json.dump(dict(nail_top=n0, arms=results), open(OUT, "w"))
print(f"\n-> {OUT}  ({len(results)} policies traced)")
