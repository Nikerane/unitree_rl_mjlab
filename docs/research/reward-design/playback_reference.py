"""Run the one nominal direct strike against the production fixed-reset C0 config.

This is a control-rate-sampled Phase-M feasibility check.  Authoritative 500 Hz
qvel/contact/finite certification belongs to
``evaluation/guideline/qualify_reference.py``.  The slow-press run is a
diagnostic only.  The printed max-force integral is a crude control-rate proxy,
not either reward impulse normalizer.

Run:
    python docs/research/reward-design/playback_reference.py
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from evaluation.guideline.qualify_reference import QVEL_LIMIT_RAD_S
from src.assets.robots.unitree_z1.z1_constants import (
    ARM_JOINT_NAMES,
    HAMMER_HEAD_SITE_NAME,
    Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD

from reward_design_util import (
    assert_log_only_reference_contract,
    load_direct_reference_c0_cfg,
)


HOLD_STEPS = 10
CONTACT_SLACK = 2
CONTACT_SPEED_FLOOR = 0.5
PRESS_STEPS = 200
PRESS_ACTION = -0.1
CONTROL_RATE_FEASIBILITY_SCOPE = (
    "control-rate sampled feasibility only; authoritative 500 Hz "
    "qvel/contact/finite certification is delegated to "
    "evaluation/guideline/qualify_reference.py"
)


def main() -> int:
    cfg = load_direct_reference_c0_cfg()
    env = ManagerBasedRlEnv(cfg, device="cpu")
    hook = env.metrics_manager.cfg["cat_soft"].func
    assert_log_only_reference_contract(
        cfg,
        live_imp_limit=hook._imp_limit,
        live_imp_max_p=hook._imp_max_p,
    )

    rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    rcfg.resolve(env.scene)
    arm_cfg.resolve(env.scene)
    ncfg.resolve(env.scene)
    robot = env.scene["robot"]
    nail_e = env.scene["nail_block"]
    sensor = env.scene["hammer_nail_contact"]

    def head() -> torch.Tensor:
        return robot.data.site_pos_w[:, rcfg.site_ids].squeeze(1)

    def nail_top() -> torch.Tensor:
        return nail_e.data.site_pos_w[:, ncfg.site_ids].squeeze(1)

    def depth() -> float:
        return float(nail_e.data.joint_pos[0, 0])

    # Exactly one nominal default direct reference.
    env.reset()
    ref = SingleStrikeReference(1, env.device)
    ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
    n = ref.playback_length()
    max_d = 0.0
    contact_step = None
    contact_speed = None
    terminated_step = None
    force_impulse_proxy = 0.0
    sampled_qvel_peak = 0.0
    sampled_finite = True
    for k in range(1, n + HOLD_STEPS + 1):
        target = ref.playback_target(min(k, n))
        head_z_pre = float(head()[0, 2])
        action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
        env.step(action)
        qvel = robot.data.joint_vel[0, arm_cfg.joint_ids]
        sampled_qvel_peak = max(sampled_qvel_peak, float(qvel.abs().max()))
        sampled_finite = sampled_finite and bool(
            torch.isfinite(action).all()
            and torch.isfinite(head()).all()
            and torch.isfinite(qvel).all()
            and torch.isfinite(nail_e.data.joint_pos).all()
        )
        found = bool((sensor.data.found > 0).any())
        if found:
            if contact_step is None:
                contact_step = k
                contact_speed = (head_z_pre - float(head()[0, 2])) / env.step_dt
            force = sensor.data.force
            force_magnitude = (
                float(force.norm(dim=-1).max())
                if force.shape[-1] == 3
                else float(force.abs().max())
            )
            force_impulse_proxy += force_magnitude * env.step_dt
        max_d = max(max_d, depth())
        if bool(env.reset_terminated.any()):
            terminated_step = k
            break

    drove_nail = terminated_step is not None or max_d >= NAIL_SUCCESS_THRESHOLD
    if terminated_step is not None:
        max_d = max(max_d, NAIL_SUCCESS_THRESHOLD)
    in_script = contact_step is not None and contact_step <= n + CONTACT_SLACK
    fast = contact_speed is not None and contact_speed >= CONTACT_SPEED_FLOOR
    sampled_qvel_within_rail = sampled_qvel_peak <= QVEL_LIMIT_RAD_S
    sampled_feasible = (
        drove_nail
        and in_script
        and fast
        and sampled_qvel_within_rail
        and sampled_finite
    )
    speed_text = f"{contact_speed:.2f} m/s" if contact_speed is not None else "n/a"
    print(
        f"nominal direct strike (script n={n}): max depth >= {max_d * 1000:.1f} mm  "
        f"contact@step {contact_step} (in-script<={n + CONTACT_SLACK}: {in_script})  "
        f"v_contact={speed_text}  sampled_qvel_peak={sampled_qvel_peak:.4f} rad/s  "
        f"sampled_finite={sampled_finite}  "
        f"control-rate max-force impulse proxy={force_impulse_proxy:.2f} N·s"
    )
    print(f"scope: {CONTROL_RATE_FEASIBILITY_SCOPE}")

    # Diagnostic only: this does not affect the direct-strike gate verdict.
    env.reset()
    press = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    press[:, 2] = PRESS_ACTION
    stall = 0.0
    press_steps_to_success = None
    for k in range(1, PRESS_STEPS + 1):
        env.step(press)
        stall = max(stall, depth())
        if bool(env.reset_terminated.any()):
            press_steps_to_success = k
            break
    if press_steps_to_success is None:
        print(
            f"slow-press diagnostic (non-gating): stalled at {stall * 1000:.1f} mm"
        )
    else:
        print(
            "slow-press diagnostic (non-gating): reached threshold in "
            f"{press_steps_to_success} steps; press-exploit pressure exists"
        )

    if sampled_feasible:
        print(
            "PHASE M CONTROL-RATE FEASIBILITY: PASS — the nominal direct reference "
            "drives the nail with in-script strike-speed contact at the sampled "
            "control-rate states"
        )
        return 0
    if drove_nail:
        print(
            "PHASE M CONTROL-RATE FEASIBILITY: NOT MET — nail depth was reached "
            "without the required sampled in-script strike-speed, finite-state, "
            "and qvel conditions; stop the experiment"
        )
        return 1
    print(
        "PHASE M CONTROL-RATE FEASIBILITY: NOT MET — the nominal direct reference "
        "did not drive the nail; stop the experiment"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
