"""Trace the 12 mx-ablation policies (maxoff/maxon/maxmax/maxofftrk × 3 seeds) from the SAME fixed reset
(seed 12345) as the July analysis. Records pre-contact forward-swing, contact speed, success — to answer:
does the forward-swing scale with impact-maximization reward dose, or is it kinematic?

Reuses the corrected tracer (substep-depth peak + env.step done flag). All 12 load in the CaT-Impulse env
(maxofftrk trained on -Track, but obs is identical, so the rollout is valid)."""
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
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR
I_REF = 0.6094

TASK = "Unitree-Z1-Hammer-CaT-Impulse"
ROOT = "logs/rsl_rl/z1_hammer"
OUT = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad/traj_mx.json"
SEED = 12345
STAMP = "2026-07-20_22-07-49_mx_"
ARMS = [(f"{arm}_s{s}", f"{STAMP}{arm}_seed{s}")
        for arm in ("maxoff", "maxon", "maxmax", "maxofftrk") for s in (0, 1, 2)]

env_cfg = load_env_cfg(TASK, play=True); env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
agent_cfg = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")

robot = base.scene["robot"]; nail = base.scene["nail_block"]; sensor = base.scene["hammer_nail_contact"]
rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(base.scene)
nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(base.scene)
head = lambda: robot.data.site_pos_w[0, rc.site_ids].squeeze(0).tolist()
dt = float(base.physics_dt)

results = {}; n0 = None
for label, prefix in ARMS:
    ck = sorted(glob.glob(f"{ROOT}/{prefix}/model_*.pt"), key=lambda p: int(p.split("model_")[1].split(".pt")[0]))
    if not ck:
        print(f"[skip] {label}: no ckpt"); continue
    try:
        runner.load(ck[-1], load_cfg={"actor": True}, strict=True, map_location="cpu")
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}"); continue
    policy = runner.get_inference_policy(device="cpu")
    dacc = getattr(base, _ENV_SUBSTEP_DELIVERED_ATTR)  # object-side delivered ∫F·dt accumulator
    torch.manual_seed(SEED); obs, _ = env.reset()
    nt = nail.data.site_pos_w[0, nc.site_ids].squeeze(0).tolist()
    if n0 is None: n0 = nt
    sub = []; deliv_peak = [0.0]; orig = base.metrics_manager.compute_substep
    def rec():
        orig(); h = head()
        cd = min(max(float(nail.data.joint_pos[0, 0]), 0.0), 0.032) * 1000.0
        deliv_peak[0] = max(deliv_peak[0], float(dacc.delivered[0]))  # pre-reset peak
        sub.append([h[0], h[1], h[2], bool((sensor.data.found > 0).any()), cd])
    base.metrics_manager.compute_substep = rec
    succeeded = False; nctrl = 0
    for k in range(1, 16):
        with torch.inference_mode():
            a = policy(obs)
        out = env.step(a); obs = out[0]; nctrl += 1
        d0 = out[2][0]; succeeded = succeeded or bool(d0.item() if hasattr(d0, "item") else d0)
        if int(base.episode_length_buf[0]) == 0:
            break
    base.metrics_manager.compute_substep = orig
    fc = next((i for i, p in enumerate(sub) if p[3]), None)
    swing = (max(p[0] for p in sub[:(fc + 1 if fc is not None else len(sub))]) - nt[0]) * 100
    vtouch = (sub[fc - 1][2] - sub[fc][2]) / dt if fc and fc >= 1 else float("nan")
    peak = max(s[4] for s in sub)
    dvr = deliv_peak[0] / I_REF  # delivered impulse × i_ref
    results[label] = dict(substep=sub, nail_top=nt, max_nail_mm=peak, success=succeeded,
                          swing_pre_cm=swing, v_touch=vtouch, delivered_x_iref=dvr, physics_dt=dt)
    print(f"[ok] {label:14s} swing={swing:5.1f}cm v_touch={vtouch:5.2f} deliv={dvr:5.2f}xiref "
          f"peak={peak:4.1f}mm success={succeeded}")

json.dump(dict(nail_top=n0, arms=results), open(OUT, "w"))
print(f"\n-> {OUT} ({len(results)} policies)")
