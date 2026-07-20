"""Follow-ups #1+#2: evaluate the 12 mx checkpoints over N RANDOMIZED resets (play=False -> reset noise
active, the IC distribution the policy was trained on; obs noise also on = real deployment conditions).
Per rollout records swing / v_touch / delivered / dwell / PEAK FORCE / #contact-EVENTS / success — closing
the single-fixed-reset gap AND testing 'flat force' + 'longer dwell vs extra taps' at the physics level.

Paired design: reset i uses seed BASE+i for ALL 12 policies (same IC set per policy). CPU, no GPU."""
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
OUT = "/private/tmp/claude-501/-Users-nikerane-repos-unitree-rl-mjlab/2cd822e7-752b-4070-969b-f93a7bd62f87/scratchpad/mx_multireset.json"
STAMP = "2026-07-20_22-07-49_mx_"; I_REF = 0.6094; N = 50; BASE = 1000
ARMS = [(f"{arm}_s{s}", f"{STAMP}{arm}_seed{s}")
        for arm in ("maxoff", "maxon", "maxmax", "maxofftrk") for s in (0, 1, 2)]

cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = 1   # play=False -> reset noise ACTIVE
base = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
agent_cfg = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
robot = base.scene["robot"]; nail = base.scene["nail_block"]; sensor = base.scene["hammer_nail_contact"]
netf = base.scene["hammer_nail_impulse"]
rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(base.scene)
nc = SceneEntityCfg("nail_block", site_names=("nail_top",)); nc.resolve(base.scene)
axis = torch.tensor([0., 0., -1.]); dt = float(base.physics_dt)
hx = lambda: float(robot.data.site_pos_w[0, rc.site_ids].squeeze(0)[0])
hz = lambda: float(robot.data.site_pos_w[0, rc.site_ids].squeeze(0)[2])
nx = lambda: float(nail.data.site_pos_w[0, nc.site_ids].squeeze(0)[0])

results = {}
for label, prefix in ARMS:
    ck = sorted(glob.glob(f"{ROOT}/{prefix}/model_*.pt"), key=lambda p: int(p.split("model_")[1].split(".pt")[0]))
    if not ck: print(f"[skip] {label}"); continue
    runner.load(ck[-1], load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    dacc = getattr(base, _ENV_SUBSTEP_DELIVERED_ATTR)
    rollouts = []
    for i in range(N):
        torch.manual_seed(BASE + i); obs, _ = env.reset(); n0x = nx()
        rec = []; dpk = [0.0]; orig = base.metrics_manager.compute_substep
        def cb():
            orig()
            f = float((netf.data.force * axis).sum(-1).sum(-1).clamp_min(0.)[0])
            dpk[0] = max(dpk[0], float(dacc.delivered[0]))
            rec.append((bool((sensor.data.found > 0).any()), hx(), hz(), f))
        base.metrics_manager.compute_substep = cb
        succ = False
        for k in range(1, 16):
            with torch.inference_mode(): a = policy(obs)
            out = env.step(a); obs = out[0]
            d0 = out[2][0]; succ = succ or bool(d0.item() if hasattr(d0, "item") else d0)
            if int(base.episode_length_buf[0]) == 0: break
        base.metrics_manager.compute_substep = orig
        ct = [r[0] for r in rec]; fc = next((j for j, c in enumerate(ct) if c), None)
        swing = (max(r[1] for r in rec[:(fc + 1 if fc is not None else len(rec))]) - n0x) * 100
        vt = (rec[fc - 1][2] - rec[fc][2]) / dt if fc and fc >= 1 else float("nan")
        dwell = sum(ct)
        events = sum(1 for j in range(len(ct)) if ct[j] and not (j and ct[j - 1]))  # rising edges
        pf = max((r[3] for r in rec if r[0]), default=0.0)
        rollouts.append(dict(swing=swing, v_touch=vt, delivered=dpk[0] / I_REF,
                             dwell_ms=dwell * dt * 1000, peak_force=pf, events=events, success=succ))
    results[label] = rollouts
    sw = [r["swing"] for r in rollouts]; dv = [r["delivered"] for r in rollouts]
    pf = [r["peak_force"] for r in rollouts]; sc = sum(r["success"] for r in rollouts)
    print(f"[ok] {label:14s} N={len(rollouts)} swing={sum(sw)/len(sw):5.2f}cm deliv={sum(dv)/len(dv):5.2f}x "
          f"peakF={sum(pf)/len(pf):5.1f}N success={sc}/{N}")

json.dump(dict(N=N, arms=results), open(OUT, "w"))
print(f"\n-> {OUT}")
