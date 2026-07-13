"""Phase M gate (plan stage T1): execute the scripted single-strike reference
open-loop against the live env and certify reference quality.

Reports per approach height:
  - contact step, max nail depth, success vs NAIL_SUCCESS_THRESHOLD,
  - head axial (downward) speed over the first-contact step (control-rate),
  - crude I_ref = sum(max contact |F|) * step_dt over contact steps
    (control-rate approximation; substep-accurate windowed impulse is stage T3).

Also runs a slow-press probe (no swing) to locate the quasi-static press-stall
depth, then prints the Q1 threshold-invariant verdict:

    press_stall  <  NAIL_SUCCESS_THRESHOLD  <=  best strike depth

GATE HARDENING (2026-07-13, adversarial review F3): depth-only certification
let the gate "pass" entirely through the post-script endpoint-HOLD — a servo
drive with ZERO contact during the scripted trajectory (measured: without the
hold, no contact at all; with it, contact at hold-step 5 at 0.46 m/s). The
gate now ALSO requires, for at least one approach height:
  - first contact within the scripted steps + CONTACT_SLACK hold steps
    (the arm lags the open-loop waypoints by a few steps), and
  - head axial speed over the first-contact step >= CONTACT_SPEED_FLOOR
    (a strike, not the quasi-static press regime: the press probe commands
    ~0.25 m/s; V1's trained strike measured ~0.45 m/s at contact).
The reference apex is additionally floored at head0+min_windup_clearance
(references.py), so the wind-up is genuine at the L6 near-nail reset.

Run:
    python docs/research/reward-design/playback_reference.py
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_z1.z1_constants import (
    HAMMER_HEAD_SITE_NAME,
    Z1_HAMMER_DELTA_POS_SCALE,
)
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg
from src.tasks.hammer.mdp.references import SingleStrikeReference
from src.tasks.hammer.nail_block import NAIL_SUCCESS_THRESHOLD

APPROACH_HEIGHTS = [0.06, 0.10, 0.15]  # 0.15 = SingleStrikeReference default
# NOTE: at the L6 near-nail reset (head z≈0.250) the apex clearance floor
# (references.py min_windup_clearance) dominates all three heights — the swept
# apexes coincide at head0+clearance; the sweep is kept for lower resets.
HOLD_STEPS = 10      # keep commanding the strike target after the descent ends
CONTACT_SLACK = 2    # first contact must land within n_script + this many hold steps
CONTACT_SPEED_FLOOR = 0.5  # m/s axial at first contact: strike, not press (~0.25) — see docstring
PRESS_STEPS = 200    # slow-press probe duration
PRESS_ACTION = -0.1  # 5 mm/step commanded descent — quasi-static by design


def main() -> None:
    cfg = z1_hammer_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")

    rcfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
    ncfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
    rcfg.resolve(env.scene)
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

    # --- scripted strike playback per approach height ---
    results = []
    for h in APPROACH_HEIGHTS:
        env.reset()
        ref = SingleStrikeReference(1, env.device, approach_height=h)
        ref.update(head(), nail_top(), torch.zeros(1, dtype=torch.long, device=env.device))
        n = ref.playback_length()
        apex_z = float(ref._apex[0, 2])
        max_d, contact_step, i_ref = 0.0, None, 0.0
        contact_speed = None
        terminated_step = None
        for k in range(1, n + HOLD_STEPS + 1):
            target = ref.playback_target(min(k, n))
            head_z_pre = float(head()[0, 2])
            delta = target - head()
            action = (delta / Z1_HAMMER_DELTA_POS_SCALE).clamp(-1.0, 1.0)
            env.step(action)
            # Read the sensor BEFORE the termination check: the success step
            # usually carries the peak contact force, and the pre-reset substep
            # data is still in the sensor buffer (review finding).
            found = bool((sensor.data.found > 0).any())
            if found:
                if contact_step is None:
                    contact_step = k
                    # Axial (downward) speed over the step that first made
                    # contact — control-rate mean, same convention as
                    # diag_policy_trace's contact-speed metric.
                    contact_speed = (head_z_pre - float(head()[0, 2])) / env.step_dt
                f = sensor.data.force
                fmag = float(f.norm(dim=-1).max()) if f.shape[-1] == 3 else float(f.abs().max())
                i_ref += fmag * env.step_dt
            max_d = max(max_d, depth())
            if int(env.episode_length_buf[0]) == 0:
                # Success termination auto-resets the env (and zeroes the nail)
                # before the depth read above — the termination itself is the
                # success signal; depth readings this step are stale.
                terminated_step = k
                break
        ok = terminated_step is not None or max_d >= NAIL_SUCCESS_THRESHOLD
        if ok and terminated_step is not None:
            max_d = max(max_d, NAIL_SUCCESS_THRESHOLD)  # crossed at least the threshold
        in_script = contact_step is not None and contact_step <= n + CONTACT_SLACK
        fast = contact_speed is not None and contact_speed >= CONTACT_SPEED_FLOOR
        strike_ok = ok and in_script and fast
        results.append((h, max_d, contact_step, i_ref, ok, n, contact_speed, strike_ok))
        done_txt = f"SUCCESS (terminated @step {terminated_step})" if terminated_step else f"success={ok}"
        spd_txt = f"{contact_speed:5.2f} m/s" if contact_speed is not None else "  n/a"
        print(
            f"approach {h:.2f} m (apex z={apex_z:.3f}, script n={n}): max depth ≥{max_d * 1000:6.1f} mm  "
            f"contact@step {contact_step} (in-script≤{n + CONTACT_SLACK}: {in_script})  "
            f"v_contact={spd_txt}  I_ref≈{i_ref:6.2f} N·s  {done_txt}"
        )

    # --- slow-press probe (no swing): press-pressure measurement, NOT a gate.
    # Measured 2026-06-10: from the reset pose the arm pushes through 30 N
    # friction and a sustained press DOES reach the threshold (~11 steps with
    # a -1.0 command). Quasi-static solutions persist at calibrated friction on
    # this stiff-PD position action space (Path-A outcome #2); press exclusion
    # is the reward design's job (one-payout impact window, time penalty,
    # reference prior) and this number is the watchdog baseline for it.
    env.reset()
    press = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    press[:, 2] = PRESS_ACTION
    stall = 0.0
    press_steps_to_success = None
    for k in range(1, PRESS_STEPS + 1):
        env.step(press)
        stall = max(stall, depth())
        if int(env.episode_length_buf[0]) == 0:
            press_steps_to_success = k
            break
    if press_steps_to_success is None:
        print(f"\nslow press (no swing): stalled at {stall * 1000:.1f} mm — press cannot solve the task")
    else:
        print(f"\nslow press (no swing): reached threshold in {press_steps_to_success} steps "
              f"— press exploit pressure EXISTS; reward design must out-score it")

    # --- reference-quality verdict (the actual Phase M gate) ---
    # Hardened 2026-07-13 (F3): depth alone is necessary but NOT sufficient —
    # at least one height must also make first contact within the script
    # (+slack) at >= CONTACT_SPEED_FLOOR, i.e. the reference itself must
    # demonstrate a strike, not an endpoint-hold press-through.
    best = max(r[1] for r in results)
    thr = NAIL_SUCCESS_THRESHOLD
    best_contact = min((r[2] for r in results if r[2] is not None), default=None)
    any_strike = any(r[7] for r in results)
    best_speed = max((r[6] for r in results if r[6] is not None), default=None)
    print(f"best strike depth: {best * 1000:.1f} mm | threshold {thr * 1000:.0f} mm | "
          f"best v_contact: {f'{best_speed:.2f} m/s' if best_speed is not None else 'n/a'} "
          f"(floor {CONTACT_SPEED_FLOOR})")
    if best >= thr and any_strike:
        print("PHASE M GATE: PASS — scripted reference strike reaches success "
              "with in-script contact at strike speed; single-strike task is well-posed")
        if press_steps_to_success is not None and best_contact is not None:
            print(f"  (strike contacts at step ~{best_contact}; press needs "
                  f"{press_steps_to_success} steps — speed gap is the anti-press margin "
                  "the reward design must exploit via time penalty + one-payout impact window)")
    elif best >= thr:
        print(
            "PHASE M GATE: NOT MET — depth reached ONLY via the endpoint-hold "
            "(no in-script contact at strike speed): the reference encodes a "
            "press-through, not a strike. Retune the reference (apex clearance, "
            "windup/descent pacing) until contact lands within the script at "
            f">= {CONTACT_SPEED_FLOOR} m/s."
        )
    else:
        print(
            "PHASE M GATE: NOT MET — improve the reference (approach height, "
            f"alignment) or lower NAIL_SUCCESS_THRESHOLD toward ~{0.95 * best * 1000:.0f} mm."
        )


if __name__ == "__main__":
    main()
