"""ENFORCEMENT-PLUMBING test: with imp_max_p=0.5 (enforcement ON), show the full chain fire ---
per-joint Λ (shipped accumulator) crosses the cap  ->  margin c = (Λ - cap)+ > 0  ->  the CaT
normalizer activates  ->  δ = env.extras['cat_delta'] > 0 (the per-step termination probability
CatPPO applies). A COMPLIANT strike gives δ = 0; an OVER-CAP event gives δ > 0. Proves the
constraint actually ENFORCES when triggered --- no training needed. One env/process.

We trigger the over-cap Λ with a heavy nail (a PRESS reaches Λ > cap; a clean fixed-impedance impact
cannot -- see NAIL_SWEEP_FINDINGS). That's fine here: the point is the PLUMBING (Λ -> δ), not the
physical realism of the trigger."""
from __future__ import annotations
import argparse, json, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.unitree_z1.z1_constants import HAMMER_HEAD_SITE_NAME, Z1_HAMMER_DELTA_POS_SCALE
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg, IMP_J_LIMIT
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_IMPULSE_ATTR

ARM = ("joint1","joint2","joint3","joint4","joint5","joint6")
NAIL_GEOMS = ("nail_shaft", "nail_head")
ap = argparse.ArgumentParser()
ap.add_argument("--mass_scale", type=float, default=1.0)
ap.add_argument("--imp_max_p", type=float, default=0.5)
ap.add_argument("--speed", type=float, default=4.0)
ap.add_argument("--hold", type=int, default=25)
args = ap.parse_args()

cfg = z1_hammer_env_cfg(play=True, cat_impulse=True); cfg.scene.num_envs = 1
cfg.metrics["cat_soft"].params["imp_max_p"] = args.imp_max_p   # <-- ENFORCEMENT ON
cap = torch.tensor(IMP_J_LIMIT)

nb = cfg.scene.entities["nail_block"]; orig_fn = nb.spec_fn
def patched_spec():
    spec = orig_fn()
    for g in spec.geoms:
        if g.name in NAIL_GEOMS:
            try: g.mass = float(g.mass) * args.mass_scale
            except Exception: pass
    return spec
nb.spec_fn = patched_spec

env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]; nail_e = env.scene["nail_block"]
rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rcfg.resolve(env.scene)
ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",)); ncfg.resolve(env.scene)
def head(): return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)
def ntop(): return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

env.reset()
ref = SingleStrikeReference(1, env.device, approach_height=0.10)
ref.update(head(), ntop(), torch.zeros(1, dtype=torch.long, device=env.device))
n = ref.playback_length()
acc = getattr(env, _ENV_SUBSTEP_IMPULSE_ATTR)

from src.tasks.hammer.cat.constraints import joint_impulse_excess
imp_limit_t = torch.as_tensor(IMP_J_LIMIT, dtype=torch.float32)
peak_lambda = torch.zeros(6)   # peak per-joint Λ the constraint read (accumulator.impulse)
peak_excess = torch.zeros(6)   # peak per-joint margin the HOOK actually computes
peak_delta = 0.0               # peak δ (env.extras['cat_delta'])
delta_trace = []
step_trace = []
for k in range(1, n + args.hold + 1):
    target = ref.playback_target(min(k, n))
    env.step(((target - head()) / Z1_HAMMER_DELTA_POS_SCALE * args.speed).clamp(-1., 1.))
    lam = acc.impulse[0].detach().clone()                 # (6,) current per-joint Λ
    exc = joint_impulse_excess(env, limit=imp_limit_t)[0].detach().clone()  # (6,) the hook's margin
    peak_lambda = torch.maximum(peak_lambda, lam)
    peak_excess = torch.maximum(peak_excess, exc)
    d = env.extras.get("cat_delta")
    dv = float(d.reshape(-1)[0]) if d is not None else float("nan")
    peak_delta = max(peak_delta, dv)
    if dv > 1e-6: delta_trace.append(round(dv, 3))
    if dv > 1e-6 or float(exc.max()) > 1e-6:
        step_trace.append({"k": k, "lam_j2": round(float(lam[1]), 3), "excess_max": round(float(exc.max()), 3), "d": round(dv, 3)})
    if int(env.episode_length_buf[0]) == 0: break

margin = (peak_lambda - cap).clamp_min(0.0)
over = [float(peak_lambda[j] / IMP_J_LIMIT[j]) for j in range(6)]
worst = max(range(6), key=lambda j: over[j])
print(json.dumps({
    "mass_scale": args.mass_scale, "imp_max_p": args.imp_max_p,
    "peak_Lambda_perjoint": [round(float(x), 3) for x in peak_lambda],
    "worst_joint": worst + 1, "worst_Lambda_over_cap": round(over[worst], 3),
    "margin_perjoint_Nms": [round(float(x), 3) for x in margin],
    "any_over_cap_POSTstep": bool((peak_lambda > cap).any()),
    "peak_excess_HOOK_perjoint": [round(float(x), 3) for x in peak_excess],
    "hook_saw_over_cap": bool((peak_excess > 1e-6).any()),
    "peak_delta": round(peak_delta, 3),
    "delta_fired": bool(peak_delta > 1e-6),
    "step_trace_when_active": step_trace[:20],
}))
