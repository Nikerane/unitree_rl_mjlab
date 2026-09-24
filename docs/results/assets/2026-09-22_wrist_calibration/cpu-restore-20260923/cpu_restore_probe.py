"""CPU-only construction and exact saved-state restore; no physics rollout."""
import argparse
import json
from pathlib import Path
import traceback

import torch

from contact_refinement import validate_reports
from contact_replay_helpers import duration_steps, require_pair
from contact_runtime import preflight, restore_initial, runtime_identity, write_json


DT = .00025
CONTROL_DT = .02
WINDOW_DT = .05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, type=int, choices=(1, 3))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    env, error, close_error = None, None, None
    stage, capture, meta, saved, sensor_status = "accepted_capture", None, None, None, None
    try:
        capture, _, _ = validate_reports(args.validation_root, args.arm)
        stage = "capture_identity"
        meta = json.loads((capture / "capture.json").read_text())
        if runtime_identity() != meta["runtime_identity"]:
            raise RuntimeError("source/dependency identity differs from native capture")
        if meta["num_envs"] != 128:
            raise RuntimeError("CPU probe requires original 128-world cohort")
        saved = torch.load(capture / "initial.pt", map_location="cpu", weights_only=False)
        require_pair(meta, saved)

        stage = "configure"
        import mjlab.tasks
        import src.tasks
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg
        from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_experiment

        decimation, window = duration_steps(CONTROL_DT, DT), duration_steps(WINDOW_DT, DT)
        cfg = load_env_cfg(meta["task"], play=False)
        configure_impact_speed_experiment(cfg, bounds=tuple(meta["speed_bounds"]),
                                          cost=meta["speed_cost"], mode="external")
        cfg.metrics["cat_soft"].params["imp_max_p"] = meta["dose"]
        cfg.scene.num_envs, cfg.seed = meta["num_envs"], meta["seed"]
        cfg.sim.mujoco.timestep, cfg.decimation = DT, decimation
        cfg.metrics["first_strike"].params["window_substeps"] = window
        cfg.metrics["substep_impulse"].params["event_window_substeps"] = window
        cfg.metrics["substep_delivered"].params.update(event_window_substeps=window,
                                                      rearm_gap_substeps=window)

        stage = "construct_cpu"
        env = ManagerBasedRlEnv(cfg, device="cpu")
        stage = "seed"
        env.seed(meta["seed"])
        stage = "reset"
        env.reset()
        stage = "preflight"
        sensor_status = preflight(env)
        if sensor_status != meta["sensor_status"]:
            raise RuntimeError("native source/CPU sensor semantics differ")
        stage = "restore_initial"
        restore_initial(env, saved)
        if env._sim_step_counter != 0:
            raise RuntimeError("CPU probe unexpectedly advanced physics rollout counter")
        stage = "complete"
    except Exception as exc:
        error = exc
        failure_trace = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                close_error = str(exc)
                if error is None:
                    error, stage = exc, "close"
                    failure_trace = traceback.format_exc()

    report = {"status": "passed" if error is None else "failed", "stage": stage,
              "arm": args.arm, "device": "cpu", "physics_dt_s": DT,
              "control_dt_s": CONTROL_DT, "window_s": WINDOW_DT, "num_worlds": 128,
              "physics_rollout_steps": 0, "reset_may_forward": True,
              "capture": None if capture is None else str(capture.resolve()),
              "source_runtime_identity": None if meta is None else meta["runtime_identity"],
              "saved_model_hash": None if saved is None else saved["model_hash"],
              "saved_state_hash": None if saved is None else saved["state_hash"],
              "saved_expanded_hash": None if saved is None else saved["expanded_hash"],
              "sensor_status": sensor_status,
              "error": None if error is None else str(error),
              "traceback": None if error is None else failure_trace,
              "close_error": close_error,
              "interpretation": "CPU construction/restore compatibility only; no rollout or numerical equivalence claim"}
    write_json(args.output / "probe.json", report)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
