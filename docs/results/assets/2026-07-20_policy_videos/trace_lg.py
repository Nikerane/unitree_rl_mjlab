"""Trace the decomposition: maxoff(0/0) / imponly(24/0) / delonly(0/4) / maxmax(24/4), 3 seeds each,
same fixed reset (12345). Which reward — impact_progress (speed) or delivered_impulse — drives the swing?"""
from __future__ import annotations
import glob, json
from dataclasses import asdict
import torch
import mjlab.tasks, src.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR

TASK = "Unitree-Z1-Hammer-CaT-Impulse"; ROOT = "logs/rsl_rl/z1_hammer"
OUT = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad/traj_lg.json"
SEED = 12345; I_REF = 0.6094
# label -> dir glob (any timestamp); ordered maxoff / imponly / delonly / maxmax
ARMS = [(f"{lab}_s{s}", f"*_{pat}_seed{s}") for lab, pat in
        [("maxoff500","mx_maxoff"),("maxoff1500","lg_maxoff1500"),("maxmax500","mx_maxmax"),("maxmax1500","lg_maxmax1500")]
        for s in (0, 1, 2)]

cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
agent_cfg = load_rl_cfg(TASK); env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
robot = base.scene["robot"]; nail = base.scene["nail_block"]; sensor = base.scene["hammer_nail_contact"]
rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(base.scene)
nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(base.scene)
head = lambda: robot.data.site_pos_w[0, rc.site_ids].squeeze(0).tolist()
dt = float(base.physics_dt)

results = {}; n0 = None
for label, pat in ARMS:
    dirs = sorted(glob.glob(f"{ROOT}/{pat}"))
    ck = sorted(glob.glob(f"{dirs[0]}/model_*.pt"), key=lambda p: int(p.split("model_")[1].split(".pt")[0])) if dirs else []
    if not ck: print(f"[skip] {label}: no ckpt ({pat})"); continue
    runner.load(ck[-1], load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu"); dacc = getattr(base, _ENV_SUBSTEP_DELIVERED_ATTR)
    torch.manual_seed(SEED); obs, _ = env.reset(); nt = nail.data.site_pos_w[0, nc.site_ids].squeeze(0).tolist()
    if n0 is None: n0 = nt
    sub = []; dpk = [0.0]; orig = base.metrics_manager.compute_substep
    def rec():
        orig(); h = head(); cd = min(max(float(nail.data.joint_pos[0, 0]), 0.), .032) * 1000
        dpk[0] = max(dpk[0], float(dacc.delivered[0])); sub.append([h[0], h[1], h[2], bool((sensor.data.found > 0).any()), cd])
    base.metrics_manager.compute_substep = rec
    succ = False
    for k in range(1, 16):
        with torch.inference_mode(): a = policy(obs)
        out = env.step(a); obs = out[0]; d0 = out[2][0]; succ = succ or bool(d0.item() if hasattr(d0, "item") else d0)
        if int(base.episode_length_buf[0]) == 0: break
    base.metrics_manager.compute_substep = orig
    fc = next((i for i, p in enumerate(sub) if p[3]), None)
    swing = (max(p[0] for p in sub[:(fc + 1 if fc is not None else len(sub))]) - nt[0]) * 100
    vt = (sub[fc - 1][2] - sub[fc][2]) / dt if fc and fc >= 1 else float("nan")
    results[label] = dict(substep=sub, nail_top=nt, max_nail_mm=max(s[4] for s in sub), success=succ,
                          swing_pre_cm=swing, v_touch=vt, delivered_x_iref=dpk[0] / I_REF, physics_dt=dt)
    print(f"[ok] {label:12s} swing={swing:5.1f}cm v_touch={vt:5.2f} deliv={dpk[0]/I_REF:5.2f}x success={succ}")

json.dump(dict(nail_top=n0, arms=results), open(OUT, "w"))
print(f"\n-> {OUT} ({len(results)} policies)")
