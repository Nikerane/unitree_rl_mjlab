"""Two Fable-suggested causal checks (CPU, play=False so reset noise is live).

CHECK 2 (per-joint load): record per-joint peak Lambda for maxoff/imponly/delonly/maxmax. Fable: press
loads j1, ballistic loads j2-4. If pure-press delonly loads a DIFFERENT joint than the constraint targets
-> calibration gap for the impulse-CaT going into VIC.

CHECK 1 (is the swing causal?): for a swinging maxmax policy, roll out closed-loop (record actions + head
path + delivered), then REPLAY the same action sequence open-loop with the lateral (x,y) deltas ZEROED
(straight descent, same z-timing -> same dwell). If delivered is unchanged, the swing adds nothing to the
objective (kinematic residue, not a chosen maneuver)."""
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
from src.tasks.hammer.mdp.impulse_bound import _ENV_SUBSTEP_DELIVERED_ATTR, _ENV_SUBSTEP_IMPULSE_ATTR
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT

TASK = "Unitree-Z1-Hammer-CaT-Impulse"; ROOT = "logs/rsl_rl/z1_hammer"; I_REF = 0.6094; BASE = 2000
cfg = load_env_cfg(TASK, play=False); cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
agent_cfg = load_rl_cfg(TASK); env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
robot = base.scene["robot"]; nail = base.scene["nail_block"]; sensor = base.scene["hammer_nail_contact"]
rc = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,)); rc.resolve(base.scene)
dacc = getattr(base, _ENV_SUBSTEP_DELIVERED_ATTR); iacc = getattr(base, _ENV_SUBSTEP_IMPULSE_ATTR)
cap = torch.tensor(IMP_J_LIMIT); dt = float(base.physics_dt)
hx = lambda: float(robot.data.site_pos_w[0, rc.site_ids].squeeze(0)[0])
def load(pat):
    d = sorted(glob.glob(f"{ROOT}/{pat}")); ck = sorted(glob.glob(f"{d[0]}/model_*.pt"), key=lambda p: int(p.split("model_")[1].split(".pt")[0]))
    runner.load(ck[-1], load_cfg={"actor": True}, strict=True, map_location="cpu"); return runner.get_inference_policy(device="cpu")

# ---------------- CHECK 2: per-joint peak Lambda ----------------
print("=== CHECK 2: per-joint peak Λ (mean over resets), which joint each arm loads ===")
ARMS = [("maxoff", "mx_maxoff"), ("imponly", "dc_imponly"), ("delonly", "dc_delonly"), ("maxmax", "mx_maxmax")]
NJ = 15
for lab, pat in ARMS:
    perj = torch.zeros(6); nrun = 0
    for s in (0, 1, 2):
        policy = load(f"*_{pat}_seed{s}")
        for i in range(NJ):
            torch.manual_seed(BASE + i); obs, _ = env.reset(); pk = [torch.zeros(6)]
            orig = base.metrics_manager.compute_substep
            def cb():
                orig(); pk[0] = torch.maximum(pk[0], iacc._episode_peak_perjoint[0].detach().clone())
            base.metrics_manager.compute_substep = cb
            for k in range(1, 16):
                with torch.inference_mode(): a = policy(obs)
                obs = env.step(a)[0]
                if int(base.episode_length_buf[0]) == 0: break
            base.metrics_manager.compute_substep = orig
            perj += pk[0]; nrun += 1
    perj /= nrun; ratio = perj / cap
    jmax = int(ratio.argmax()) + 1
    print(f"  {lab:9s} Λ/cap per joint = [{', '.join(f'{r:.3f}' for r in ratio)}]  -> MAX at j{jmax}")

# ---------------- CHECK 1: is the swing causal? (action replay, x/y zeroed) ----------------
print("\n=== CHECK 1: swing causality — maxmax closed-loop vs same actions with lateral zeroed ===")
policy = load("*_mx_maxmax_seed1")  # a robust swinger
def rollout_record(seed):
    torch.manual_seed(seed); obs, _ = env.reset(); acts = []; dpk = [0.0]; ct = [0]
    orig = base.metrics_manager.compute_substep
    def cb():
        orig(); dpk[0] = max(dpk[0], float(dacc.delivered[0])); ct[0] += int((sensor.data.found > 0).any())
    base.metrics_manager.compute_substep = cb
    xs = [hx()]
    for k in range(1, 16):
        with torch.inference_mode(): a = policy(obs)
        acts.append(a.detach().clone()); obs = env.step(a)[0]; xs.append(hx())
        if int(base.episode_length_buf[0]) == 0: break
    base.metrics_manager.compute_substep = orig
    return acts, dpk[0] / I_REF, ct[0] * dt * 1000, (max(xs) - 0.5) * 100  # deliv, dwell_ms, swing_cm
def rollout_replay(seed, acts, zero_xy):
    torch.manual_seed(seed); env.reset(); dpk = [0.0]; ct = [0]
    orig = base.metrics_manager.compute_substep
    def cb():
        orig(); dpk[0] = max(dpk[0], float(dacc.delivered[0])); ct[0] += int((sensor.data.found > 0).any())
    base.metrics_manager.compute_substep = cb
    xs = [hx()]
    for a in acts:
        aa = a.clone()
        if zero_xy: aa[0, 0] = 0.0; aa[0, 1] = 0.0  # kill lateral deltas -> straight descent
        env.step(aa); xs.append(hx())
        if int(base.episode_length_buf[0]) == 0: break
    base.metrics_manager.compute_substep = orig
    return dpk[0] / I_REF, ct[0] * dt * 1000, (max(xs) - 0.5) * 100
print(f"  {'seed':>4} | {'deliv_asis':>10} {'swing_asis':>10} | {'deliv_straight':>14} {'swing_straight':>14} {'dwell_str':>9} {'contact?':>8}")
das, dst = [], []
for i in range(12):
    sd = BASE + 100 + i
    acts, da, dw, sw = rollout_record(sd)
    ds, dws, sws = rollout_replay(sd, acts, zero_xy=True)
    das.append(da); dst.append(ds)
    print(f"  {i:>4} | {da:>10.2f} {sw:>10.2f} | {ds:>14.2f} {sws:>14.2f} {dws:>9.1f} {'yes' if dws>0 else 'MISS':>8}")
import statistics as st
print(f"\n  MEAN delivered: as-is(swing) {st.mean(das):.2f} vs straightened {st.mean(dst):.2f}  "
      f"(swing adds {st.mean(das)-st.mean(dst):+.2f}× i_ref)")
