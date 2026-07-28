"""Pre-training verification for the hammer reward setup.

Run before any training run:
    python docs/research/reward-design/verify_reward_setup.py

Catches the two silent-zero failure modes:
1. ContactSensor geom pattern not matching any geom (Q4)
2. site_vel_w not populated / lazy-eval'd (Q10)

Plus: every reward term must produce at least one nonzero value within
N random-policy steps, otherwise the term is dead.

NOTE (2026-06-02): refreshed to the current mjlab manager API
(`reward_manager.active_terms` + `_step_reward[:, idx]`,
`action_manager.total_action_dim`; the old `_terms` / `_term_rewards` /
`action_dim` / `make_z1_hammer_env_cfg` names were stale). The `impact_progress`
term is EXEMPT from the liveness check — it is an event reward gated on a fresh
productive strike, which a random policy essentially never produces. Its
behaviour is covered by validate_rewards.py Phase I and the unit tests.
"""

from __future__ import annotations

import sys
import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.hammer.config.z1.env_cfgs import z1_hammer_env_cfg


N_STEPS = 200
N_ENVS = 16

# Rewards that may legitimately stay silent under a random policy.
# completion: sparse SUCCESS bonus. A random policy reaches the 0.030 m success
# threshold only occasionally (16 envs x 200 steps), so its liveness here is
# non-deterministic (flaky FAIL/OK across runs). Its firing is verified
# deterministically by validate_rewards Phase H, so exempt it from this
# random-policy liveness sweep rather than rely on a lucky random success.
# nail_depth_delta likewise requires a random strike to move the nail beyond
# its 4 mm settling dead zone. Its stateful payout/reset/clamp behavior is
# verified deterministically by validate_rewards Phases C/E/F/L.
LIVENESS_EXEMPT = {"impact_progress", "completion", "nail_depth_delta"}


def main() -> None:
    cfg = z1_hammer_env_cfg()
    cfg.scene.num_envs = N_ENVS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg, device=device)

    # --- Check 1: ContactSensor resolves to expected primaries ---
    sensor = env.scene["hammer_nail_contact"]
    primaries = sensor.primary_names
    assert len(primaries) > 0, (
        f"FAIL: ContactSensor 'hammer_nail_contact' resolved to ZERO primary geoms. "
        f"Check the pattern in ContactSensorCfg.primary. "
        f"Available geoms: {[env.scene['robot'].model.geom(i).name for i in range(env.scene['robot'].model.ngeom)]}"
    )
    print(f"[OK] ContactSensor primaries: {primaries}")

    # --- Check 2: site_vel_w is non-zero after the arm moves ---
    obs, _ = env.reset()
    # Drive a large action to force the arm to move
    big_action = torch.ones(N_ENVS, env.action_manager.total_action_dim, device=env.device)
    for _ in range(5):
        env.step(big_action)

    robot = env.scene["robot"]
    head_site_id = robot.find_sites("hammer_head_site")[0]
    head_vel = robot.data.site_vel_w[:, head_site_id, :3]
    head_speed = torch.norm(head_vel, dim=-1)
    assert (head_speed > 1e-4).any(), (
        f"FAIL: site_vel_w is zero across all envs after 5 steps of max action. "
        f"site_vel_w may be lazy-evaluated or disabled in entity DataCfg. "
        f"Mean head speed: {head_speed.mean().item()}. "
        f"(impact_progress uses finite-differenced velocity and is unaffected, "
        f"but the head_vel observation reads site_vel_w.)"
    )
    print(f"[OK] site_vel_w populated. Mean head speed at step 5: {head_speed.mean().item():.4f} m/s")

    # --- Check 3: every (non-exempt) reward term has fired at least once ---
    env.reset()
    reward_term_names = list(env.reward_manager.active_terms)
    seen_nonzero = {name: False for name in reward_term_names}

    for step in range(N_STEPS):
        # Random policy
        action = (torch.rand(N_ENVS, env.action_manager.total_action_dim, device=env.device) - 0.5) * 2.0
        env.step(action)

        # Inspect per-term reward buffer ([B] per term; columns match active_terms).
        for idx, name in enumerate(reward_term_names):
            term_value = env.reward_manager._step_reward[:, idx]
            if (term_value.abs() > 1e-8).any():
                seen_nonzero[name] = True

    dead_terms = [
        n for n, seen in seen_nonzero.items() if not seen and n not in LIVENESS_EXEMPT
    ]
    if dead_terms:
        print(f"[FAIL] These reward terms NEVER fired in {N_STEPS} random-policy steps:")
        for name in dead_terms:
            print(f"        - {name}")
        sys.exit(1)

    n_checked = len(reward_term_names) - len(LIVENESS_EXEMPT & set(reward_term_names))
    exempt_status = {n: seen_nonzero[n] for n in LIVENESS_EXEMPT if n in seen_nonzero}
    print(f"[OK] All {n_checked} non-exempt reward terms fired at least once.")
    print(f"     (exempt event rewards, silence expected under random policy: {exempt_status})")

    # --- Check 4: collect air_time / impact_speed distributions (resolves Q5) ---
    # Run another batch of random-policy steps and snapshot last_air_time at each contact
    # transition. Use percentiles to set min_air_time threshold so the bonus distinguishes
    # genuine retract-and-strike from random jitter.
    print("\n--- Collecting random-policy contact statistics (Q5) ---")
    env.reset()
    air_times = []
    impact_speeds = []
    prev_head_pos = robot.data.site_pos_w[:, head_site_id].clone()
    for _ in range(N_STEPS):
        action = (torch.rand(N_ENVS, env.action_manager.total_action_dim, device=env.device) - 0.5) * 2.0
        env.step(action)
        first_contact = sensor.compute_first_contact(dt=env.step_dt).any(dim=-1)
        if first_contact.any():
            last_air = sensor.data.last_air_time.max(dim=-1).values
            head_pos = robot.data.site_pos_w[:, head_site_id]
            head_vel = (head_pos - prev_head_pos) / env.step_dt
            speed = torch.norm(head_vel, dim=-1)
            air_times.extend(last_air[first_contact].cpu().tolist())
            impact_speeds.extend(speed[first_contact].cpu().tolist())
        prev_head_pos = robot.data.site_pos_w[:, head_site_id].clone()

    if len(air_times) == 0:
        print("[WARN] Random policy never made contact in 200 steps. "
              "Threshold tuning skipped — try a partially trained policy or longer rollout.")
    else:
        air_t = torch.tensor(air_times)
        spd_t = torch.tensor(impact_speeds)
        q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"  Contact events collected: {len(air_times)}")
        print(f"  last_air_time percentiles [10/25/50/75/90]: "
              f"{torch.quantile(air_t, q).tolist()}")
        print(f"  impact_speed percentiles [10/25/50/75/90]: "
              f"{torch.quantile(spd_t, q).tolist()}")
        print(f"\n  Recommended air_time_bonus thresholds:")
        print(f"    min_air_time ≈ {torch.quantile(air_t, torch.tensor(0.50)).item():.3f} s (median = above-average retraction)")
        print(f"    max_air_time ≈ {torch.quantile(air_t, torch.tensor(0.95)).item():.3f} s (cap at 95th pct)")

    print("\n=== Reward setup verified. Safe to start training. ===")


if __name__ == "__main__":
    main()
