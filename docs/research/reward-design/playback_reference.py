"""Run the frozen R-/R0/R+ reference family on the fixed-reset C0 plant.

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
ROUTE_SIGNS = (-1, 0, 1)
HORIZONTAL_DETOUR_M = 0.020


def route_target_tape(
    reference: SingleStrikeReference, route_sign: int
) -> torch.Tensor:
    """Return the complete scripted target tape for one forced route."""
    if route_sign not in ROUTE_SIGNS or reference.num_envs != 1:
        raise ValueError("route tape requires one environment and sign in {-1, 0, +1}")
    reference.set_route_signs(
        torch.tensor([route_sign], device=reference.device, dtype=torch.int8)
    )
    return torch.stack(
        [
            reference.playback_target(step)
            for step in range(reference.playback_length() + 1)
        ]
    )


def horizontal_route_geometry_matches(
    target_tapes: dict[int, torch.Tensor],
) -> bool:
    """Require R-/R0/R+ to share tape shape, Z profile, start, and endpoint."""
    if set(target_tapes) != set(ROUTE_SIGNS):
        return False
    center = target_tapes[0]
    if center.ndim != 3 or center.shape[1:] != (1, 3):
        return False
    return all(
        tape.shape == center.shape
        and torch.equal(tape[:, :, 2], center[:, :, 2])
        and torch.equal(tape[0], center[0])
        and torch.equal(tape[-1], center[-1])
        for tape in target_tapes.values()
    )


def horizontal_route_qualification_passes(
    route_results: dict[int, dict[str, object]],
    target_tapes: dict[int, torch.Tensor],
) -> bool:
    """Pass only when every frozen route is productive and sampled-feasible."""
    if set(route_results) != set(ROUTE_SIGNS):
        return False
    return horizontal_route_geometry_matches(target_tapes) and all(
        bool(route_results[sign].get("drove_nail"))
        and bool(route_results[sign].get("sampled_feasible"))
        for sign in ROUTE_SIGNS
    )


def qualify_horizontal_routes(run_route):
    """Run R-/R0/R+ once each and apply the shared fail-closed verdict."""
    results: dict[int, dict[str, object]] = {}
    tapes: dict[int, torch.Tensor] = {}
    for sign in ROUTE_SIGNS:
        results[sign], tapes[sign] = run_route(sign)
    return horizontal_route_qualification_passes(results, tapes), results, tapes


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

    def run_route(route_sign: int):
        env.reset()
        ref = SingleStrikeReference(
            1, env.device, horizontal_detour_m=HORIZONTAL_DETOUR_M
        )
        ref.update(
            head(),
            nail_top(),
            torch.zeros(1, dtype=torch.long, device=env.device),
        )
        tape = route_target_tape(ref, route_sign)
        n = ref.playback_length()
        max_d = 0.0
        contact_step = None
        contact_speed = None
        terminated_step = None
        force_impulse_proxy = 0.0
        sampled_qvel_peak = 0.0
        sampled_finite = True
        for k in range(1, n + HOLD_STEPS + 1):
            target = tape[min(k, n)]
            head_z_pre = float(head()[0, 2])
            action = ((target - head()) / Z1_HAMMER_DELTA_POS_SCALE).clamp(
                -1.0, 1.0
            )
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
        sampled_feasible = (
            drove_nail
            and in_script
            and fast
            and sampled_qvel_peak <= QVEL_LIMIT_RAD_S
            and sampled_finite
        )
        speed_text = (
            f"{contact_speed:.2f} m/s" if contact_speed is not None else "n/a"
        )
        route_label = {-1: "R-", 0: "R0", 1: "R+"}[route_sign]
        print(
            f"route {route_label} (script n={n}): max depth >= "
            f"{max_d * 1000:.1f} mm  contact@step {contact_step} "
            f"(in-script<={n + CONTACT_SLACK}: {in_script})  "
            f"v_contact={speed_text}  sampled_qvel_peak={sampled_qvel_peak:.4f} "
            f"rad/s  sampled_finite={sampled_finite}  control-rate max-force "
            f"impulse proxy={force_impulse_proxy:.2f} N·s"
        )
        return {
            "drove_nail": drove_nail,
            "sampled_feasible": sampled_feasible,
        }, tape

    routes_passed, route_results, target_tapes = qualify_horizontal_routes(run_route)
    geometry_matches = horizontal_route_geometry_matches(target_tapes)
    print(
        "horizontal route geometry: "
        f"identical_z_start_endpoint={geometry_matches}"
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

    if routes_passed:
        print(
            "PHASE M CONTROL-RATE FEASIBILITY: PASS — R-/R0/R+ all drive the nail "
            "with in-script strike-speed contact at sampled control-rate states "
            "and identical vertical profiles/endpoints"
        )
        return 0
    productive = all(
        bool(route_results.get(sign, {}).get("drove_nail")) for sign in ROUTE_SIGNS
    )
    print(
        "PHASE M CONTROL-RATE FEASIBILITY: NOT MET — all routes must be productive, "
        "sampled-feasible, and geometry-matched; "
        f"productive_all={productive} geometry_matches={geometry_matches}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
