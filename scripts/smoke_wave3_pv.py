"""Wave-3 P+V launch gate: prove velocity-CaT is REALLY on, on the device that will train.

Run on Vega (CUDA) before submitting the six seeds, and on CPU as a pre-deploy rehearsal:

    python scripts/smoke_wave3_pv.py --device cuda --num-envs 64

Every assertion is made against a LIVE environment -- the resolved managers and the actual
CatSoftHook instance -- not against a config dataclass. A config test cannot prove that the
hook reads the substep peak at run time on the backend that will train, which is exactly the
claim Wave 3 rests on: Wave 2 measured 0/18 velocity-legal policies with enforcement OFF, so
an unenforced Wave 3 would be indistinguishable from a failed one.

Exit code 0 = every check passed. Non-zero = do not submit.
"""

from __future__ import annotations

import argparse
import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import src.tasks.hammer.config.z1  # noqa: F401  (registers tasks)
from src.tasks.hammer.cat.hook import CatSoftHook
from src.tasks.hammer.cat.keys import CAT_DELTA_KEY
from src.tasks.hammer.config.z1.env_cfgs import IMP_J_LIMIT
from src.tasks.hammer.mdp.velocity_bound import _ENV_SUBSTEP_ATTR, Z1_JOINT_VEL_LIMIT

PV_TASK = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress-Vel"
P_TASK = "Unitree-Z1-Hammer-CaT-Impulse-Event-Linear-Guideline-CProgress"
BASE_REWARDS = {
  "approach": 0.1, "nail_driven": 0.5, "nail_depth_delta": 600.0,
  "impact_progress": 8.0, "completion": 100.0, "action_rate": -0.01,
  "joint_pos_limits": -10.0, "delivered_impulse": 2.0,
}
FROZEN_CAPS = [1.640, 3.280, 1.640, 1.640, 1.640, 1.640]


def _hook_of(env: ManagerBasedRlEnv) -> CatSoftHook:
  """MetricsManager replaces term_cfg.func with the constructed ManagerTermBase instance."""
  manager = env.metrics_manager
  instance = manager._term_cfgs[manager._term_names.index("cat_soft")].func
  if not isinstance(instance, CatSoftHook):
    raise RuntimeError(f"cat_soft resolved to {type(instance)}, not CatSoftHook")
  return instance


def run_checks(device: str = "cpu", num_envs: int = 8, steps: int = 3) -> list[tuple[str, bool, str]]:
  """Return [(name, passed, detail)]. Importable so a CPU test can rehearse the GPU gate."""
  results: list[tuple[str, bool, str]] = []

  def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))

  cfg = load_env_cfg(PV_TASK, play=True)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  hook = _hook_of(env)
  tracker = getattr(env.unwrapped, _ENV_SUBSTEP_ATTR, None)
  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  for _ in range(steps):
    env.step(zero)

  # (a) substep velocity detection is ACTIVE -- tracker present, per-joint, on the train device.
  check("substep tracker installed and per-joint on the training device",
        tracker is not None
        and tuple(tracker.peak_qv_joint.shape) == (env.num_envs, 6)
        and tracker.peak_qv_joint.device.type == torch.device(device).type,
        f"shape={tuple(tracker.peak_qv_joint.shape) if tracker else None} "
        f"device={tracker.peak_qv_joint.device if tracker else None}")
  check("hook is wired to the substep peak, not the control-rate sample",
        hook._vel_detection == "substep", f"vel_detection={hook._vel_detection!r}")
  check("tracker peak-holds >= the control-rate sample after real stepping",
        bool((tracker.peak_qv_joint
              >= env.scene["robot"].data.joint_vel[:, :6].abs() - 1e-6).all()))

  # (b) the frozen dose: use_vel=True, max_p=0.5, vel_detection=substep (+ min_p, tau, limit).
  check("use_vel is True", hook._use_vel is True, str(hook._use_vel))
  check("max_p == 0.5", hook._max_p == 0.5, str(hook._max_p))
  check("min_p == 0.0 and tau == 0.95",
        hook._cat.min_p == 0.0 and hook._cat.tau == 0.95,
        f"min_p={hook._cat.min_p} tau={hook._cat.tau}")
  check("limit == 3.1415", hook._limit == Z1_JOINT_VEL_LIMIT, str(hook._limit))

  # (c) a velocity violation above 3.1415 produces a POSITIVE delta -- with the control-rate
  #     sample left legal, so this cannot pass by accidentally reading joint_vel.
  tracker.peak_qv_joint.fill_(Z1_JOINT_VEL_LIMIT)
  delta_at = hook(env).clone()
  check("peak exactly at 3.1415 -> zero velocity violation",
        bool(torch.count_nonzero(delta_at) == 0), f"delta={delta_at.max().item():.6f}")

  control_rate_peak = float(env.scene["robot"].data.joint_vel[:, :6].abs().max())
  tracker.peak_qv_joint.fill_(1.0)
  tracker.peak_qv_joint[:, 3] = Z1_JOINT_VEL_LIMIT + 0.75
  delta_over = hook(env).clone()
  check("one joint 0.75 rad/s above 3.1415 -> positive delta",
        bool((delta_over > 0).all()) and control_rate_peak <= Z1_JOINT_VEL_LIMIT,
        f"delta={delta_over.max().item():.6f}; control-rate peak "
        f"{control_rate_peak:.6f} still legal")
  check("delta never exceeds the frozen max_p ceiling 0.5",
        bool((delta_over <= 0.5 + 1e-6).all()), f"max={delta_over.max().item():.6f}")
  check("delta is published to env.extras for CatPPO",
        CAT_DELTA_KEY in env.extras
        and env.extras[CAT_DELTA_KEY].shape == (env.num_envs,))
  env.step(zero)
  check("injected peak does not survive the control-window boundary",
        float(tracker.peak_qv_joint.max()) < Z1_JOINT_VEL_LIMIT,
        f"peak now {float(tracker.peak_qv_joint.max()):.6e}")

  # (d) the progress treatment stays isolated.
  live = {n: env.reward_manager.get_term_cfg(n).weight
          for n in env.reward_manager.active_terms}
  check("eight base reward terms and weights unchanged",
        {k: v for k, v in live.items() if k in BASE_REWARDS} == BASE_REWARDS)
  check("r_waypoint_progress present at exactly 8.0",
        live.get("r_waypoint_progress") == 8.0, str(live.get("r_waypoint_progress")))
  check("r_gate absent", "r_gate" not in live, str(sorted(live)))
  check("reward set is exactly the base eight plus r_waypoint_progress",
        set(live) == set(BASE_REWARDS) | {"r_waypoint_progress"})
  check("no deterministic velocity termination",
        not ({"vel_hard", "cat_vel"} & set(env.termination_manager.active_terms)),
        str(sorted(env.termination_manager.active_terms)))
  check("no action clipping / velocity brake term",
        set(env.action_manager.active_terms) == {"ik_hammer_head"},
        str(env.action_manager.active_terms))
  check("CatPPO is selected", load_rl_cfg(PV_TASK).algorithm.class_name
        == "src.tasks.hammer.rl.cat_ppo:CatPPO")
  check("observation contract matches the frozen P control",
        env.observation_manager.group_obs_dim
        == {"actor": (44,), "critic": (44,)},
        str(env.observation_manager.group_obs_dim))

  # (e) impulse stays log-only against the unchanged manufacturer caps.
  check("imp_max_p is exactly 0.0",
        hook._imp_max_p == 0.0 and isinstance(hook._imp_max_p, float), repr(hook._imp_max_p))
  check("manufacturer impulse caps unchanged",
        torch.equal(hook._imp_limit.detach().cpu(),
                    torch.tensor(IMP_J_LIMIT, dtype=torch.float32))
        and list(map(float, IMP_J_LIMIT)) == FROZEN_CAPS,
        str(hook._imp_limit.detach().cpu().tolist()))
  check("impulse constraint still evaluated (log-only, not disabled)",
        hook._use_impulse and "joint_impulse_excess" in hook._cat.raw_constraints)

  return results


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--num-envs", type=int, default=8)
  ap.add_argument("--steps", type=int, default=3)
  args = ap.parse_args()

  print(f"[smoke] task   : {PV_TASK}")
  print(f"[smoke] control: {P_TASK}")
  print(f"[smoke] device : {args.device}  envs={args.num_envs}\n")
  results = run_checks(device=args.device, num_envs=args.num_envs, steps=args.steps)
  for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
  failed = [n for n, ok, _ in results if not ok]
  print(f"\n=== {len(results) - len(failed)}/{len(results)} checks passed ===")
  if failed:
    print("FAILED:", *failed, sep="\n  ")
    print("\nDO NOT SUBMIT.")
    return 1
  print("P+V velocity-CaT verified active on this device.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
