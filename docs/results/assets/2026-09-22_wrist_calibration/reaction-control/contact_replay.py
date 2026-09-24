"""Checkpoint-free frozen-command replay; no learning, resets or policy feedback."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contact_replay_helpers import check_fidelity, command_at, duration_steps, fidelity_detail, native_audit_verdict, passive_window, prefix_mask, replay_schedule, require_pair
from contact_runtime import check_native_target, position_command, preflight, restore_initial, runtime_identity, sample, stacked, tensor_hash, write_json


def replay(capture, output, dt, device, horizon_steps=None, native_audit=False, engineering_equivalence=False):
    import mjlab.tasks
    import src.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_experiment

    meta = json.loads((capture / "capture.json").read_text())
    if runtime_identity() != meta["runtime_identity"]:
        raise RuntimeError("source/dependency identity differs from native capture")
    state = torch.load(capture / "initial.pt", map_location="cpu", weights_only=False)
    require_pair(meta, state)
    tape = dict(np.load(capture / "tape.npz"))
    source = dict(np.load(capture / "source.npz"))
    terminal = tape.pop("terminal_steps")
    if horizon_steps is not None and horizon_steps < meta["horizon_steps"]:
        raise ValueError("common horizon cannot truncate the source-derived contact margin")
    decimation, window = duration_steps(.02, dt), duration_steps(.05, dt)
    cfg = load_env_cfg(meta["task"], play=False)
    configure_impact_speed_experiment(cfg, bounds=tuple(meta["speed_bounds"]),
                                     cost=meta["speed_cost"], mode="external")
    cfg.metrics["cat_soft"].params["imp_max_p"] = meta["dose"]
    cfg.scene.num_envs, cfg.seed = meta["num_envs"], meta["seed"]
    cfg.sim.mujoco.timestep, cfg.decimation = dt, decimation
    cfg.metrics["first_strike"].params["window_substeps"] = window
    cfg.metrics["substep_impulse"].params["event_window_substeps"] = window
    cfg.metrics["substep_delivered"].params.update(event_window_substeps=window, rearm_gap_substeps=window)
    env = ManagerBasedRlEnv(cfg, device=device)
    rows, commands, fidelity_max = [], [], {}
    audit_rows, first_failure = [], None
    native = abs(dt - .002) < 1e-12
    if native_audit and not native:
        raise ValueError("audit is native 2ms only; finer replay prohibited")
    if engineering_equivalence and not native_audit:
        raise ValueError("engineering equivalence requires explicit native-only audit")
    tolerances = {"qpos": 1e-6, "qvel": 1e-4, "velocity": 1e-4,
                  "reaction": 1e-6, "delivered_native": 1e-6,
                  "native_target": 0., "ctrl": 0., "encoder_bias": 0.}
    try:
        env.seed(meta["seed"])
        env.reset()
        sensor_status = preflight(env)
        if sensor_status != meta["sensor_status"]:
            raise RuntimeError("native source/replay sensor semantics differ")
        restore_initial(env, state)
        position = env.action_manager.get_term("joint_position")
        gains = env.action_manager.get_term("joint_stiffness")
        if engineering_equivalence:
            from engineering_fidelity import engineering_check, model_layout
            engineering_layout = model_layout(env.sim.mj_model,
                [f"robot/{name}" for name in gains.joint_names], meta["velocity_cap"])
        expected_bias = state["encoder_bias"][:, position._target_ids.cpu()].numpy()
        step_count = meta["horizon_steps"] if horizon_steps is None else horizon_steps
        for step in range(step_count):
            requested = {key: command_at(value, terminal, step) for key, value in tape.items()}
            position._processed_actions.copy_(torch.as_tensor(requested["target"], device=device))
            gains._kp.copy_(torch.as_tensor(requested["kp"], device=device))
            gains._kd.copy_(torch.as_tensor(requested["kd"], device=device))
            commands.append(requested)
            for substep in range(decimation):
                env._sim_step_counter += 1
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                # Prove effective native gains, not only the staging buffers.
                ids = gains.control_ids
                actual = {"kp": env.sim.model.actuator_gainprm[:, ids, 0],
                          "kd": -env.sim.model.actuator_biasprm[:, ids, 2]}
                for key, value in actual.items():
                    if not np.array_equal(value.detach().cpu().numpy(), requested[key]):
                        raise RuntimeError(f"applied {key} differs from frozen tape")
                command = position_command(env)
                check_native_target(command["native_target"], command["ctrl"], requested["target"],
                                    command["encoder_bias"], expected_bias)
                env.sim.step()
                env.scene.update(dt=dt)
                env.metrics_manager.compute_substep()
                row = sample(env)
                rows.append(row)
                if native:
                    index, active = step * decimation + substep, prefix_mask(terminal, step)
                    if active.any():
                        if index >= len(source["qpos"]):
                            raise RuntimeError("source prefix missing")
                        expected = {k: v[index] for k, v in source.items()}
                        if native_audit:
                            detail = fidelity_detail(row, expected, active, tolerances)
                            audit_rows.append(detail["arrays"])
                            errors = {key: float(detail["arrays"]["abs_" + key].max()) for key in tolerances}
                            if first_failure is None and detail["first_failure"] is not None:
                                first_failure = {**detail["first_failure"], "control_step": step,
                                                 "substep": substep, "physics_sample": index,
                                                 "sample_time_s": (index + 1) * dt,
                                                 "source_any_world_already_reset": bool(np.any(terminal < step))}
                                world = first_failure["world"]
                                first_failure["source_onset_s"] = float(expected["onset_s"][world])
                                first_failure["replay_onset_s"] = float(row["onset_s"][world])
                                first_failure["source_terminal_s"] = float((terminal[world] + 1) * .02)
                        else:
                            errors = check_fidelity(row, expected, active, tolerances)
                        for key, error in errors.items():
                            fidelity_max[key] = max(fidelity_max.get(key, 0), error)
                    elif native_audit:
                        # Preserve the fixed-horizon time axis after every world terminates.
                        audit_rows.append(fidelity_detail(row, row, active, tolerances)["arrays"])
            # Match native physical control-boundary placement. No terminations,
            # reward/observation feedback or autoresets are executed in replay.
            env.episode_length_buf += 1
            env.common_step_counter += 1
            env.sim.forward()
            env.command_manager.compute(dt=.02)
            env.sim.sense()
        traces = stacked(rows)
        commands = stacked(commands)
        passive = passive_window(traces["contact"], traces["force_axial"], dt)
        last = {key: value[-1] for key, value in traces.items()}
        arming_match = bool(np.allclose(passive["onset_s"], last["onset_s"], atol=dt * 1e-3, rtol=0))
        if not arming_match and not native_audit:
            raise RuntimeError("raw and accepted contact onset differ: initial arming confounds passive window")
        audit_status, audit_failure_reasons = native_audit_verdict(first_failure, arming_match)
        force_range = np.asarray(meta["force_range"])
        force = traces["actuator_force"]
        saturated = (force <= force_range[None, None, :, 0] + 1e-4) | (force >= force_range[None, None, :, 1] - 1e-4)
        reaction = traces["reaction"].max(axis=0)
        velocity = np.abs(traces["velocity"]).max(axis=0)
        finalized = traces["finalized"]
        finalize_s = np.where(finalized.any(axis=0), (finalized.argmax(axis=0) + 1) * dt, 0)
        # Signed chatter rate is descriptive; zeros are not counted as reversals.
        signed = np.sign(force)
        reversals = (signed[1:] * signed[:-1] < 0).sum(axis=0)
        output.mkdir()
        np.savez_compressed(output / "substeps.npz", **traces)
        np.savez_compressed(output / "commands.npz", **commands)
        engineering_result = None
        if engineering_equivalence:
            engineering_result = engineering_check(traces, source, terminal, engineering_layout,
                                                   meta["impulse_caps"], meta["velocity_cap"])
            if not arming_match:
                engineering_result["status"] = "failed"
                engineering_result["failures"].append("raw_accepted_contact_onset")
            engineering_result.update(
                strict_verdict_preserved=audit_status, strict_first_failure=first_failure,
                source_capture=str(capture.resolve()), horizon_s=step_count * .02,
                native_dt=dt, source_model_state_identity_gates="passed",
                effective_target_ctrl_gain_bias_gates="passed every substep",
                physical_finiteness_and_monitored_capacity_gates="passed",
                automatic_refinement_authorized=False,
            )
            write_json(output / "engineering-equivalence.json", engineering_result)
        if native_audit:
            audit = stacked(audit_rows)
            np.savez_compressed(output / "fidelity-prefix.npz", **audit)
            write_json(output / "fidelity-audit.json", {
                "status": audit_status,
                "failure_reasons": audit_failure_reasons,
                "purpose": "diagnose native replay discrepancies only; does not authorize refinement",
                "first_failure": first_failure, "dt": dt, "horizon_s": step_count * .02,
                "source_first_reset_s": float((terminal.min() + 1) * .02),
                "source_capture": str(capture.resolve()),
                "per_world_max_absolute_error": {k: audit["abs_" + k].max(axis=0).tolist() for k in tolerances},
                "per_world_violating_samples": {k[4:]: value.sum(axis=0).tolist()
                                                for k, value in audit.items() if k.startswith("bad_")},
                "active_samples_per_world": audit["active"].sum(axis=0).tolist(),
                "tolerances_unchanged": True, "rtol": 1e-5, "atol": tolerances,
            })
        summary = {
            "dt": dt, "horizon_s": step_count * .02, "window_s": window * dt,
            "control_dt": .02, "world_ids": meta["world_ids"], "state_hash": meta["state_hash"],
            "model_hash": meta["model_hash"], "expanded_hash": state["expanded_hash"],
            "runtime_identity": meta["runtime_identity"],
            "command_hash": tensor_hash({k: torch.from_numpy(v) for k, v in commands.items()}),
            "native_prefix_fidelity": audit_status if native else "not applicable at modified timestep; acceptance basis recorded by calling driver",
            "native_audit_only": native_audit,
            "native_audit_failure_reasons": audit_failure_reasons if native_audit else [],
            "engineering_equivalence_status": None if engineering_result is None else engineering_result["status"],
            "raw_contact_onset_matches_accepted": arming_match,
            "fidelity_max_absolute_error": fidelity_max,
            "fidelity_rtol": 1e-5, "fidelity_atol": tolerances,
            "fidelity_scope": "every native physics sample through first native terminal interval, limited to common horizon",
            "first_contact_s": last["onset_s"].tolist(), "finalize_s": finalize_s.tolist(),
            "native_terminal_s": ((terminal + 1) * .02).tolist(),
            "native_terminal_beyond_horizon": ((terminal + 1) > step_count).tolist(),
            "native_success_censored_delivered_ns": last["delivered_native"].tolist(),
            "native_finalize_reason": last["reason"].tolist(),
            "passive_50ms_delivered_ns": passive["impulse"].tolist(),
            "passive_started": passive["started"].tolist(),
            "passive_complete": passive["complete"].tolist(),
            "passive_onset_s": passive["onset_s"].tolist(),
            "contact_duration_s": (traces["contact"].sum(axis=0) * dt).tolist(),
            "contact_rising_edge_count": (traces["contact"][0].astype(int) +
                 ((~traces["contact"][:-1]) & traces["contact"][1:]).sum(axis=0)).tolist(),
            "reaction_peak_nms": reaction.tolist(),
            "reaction_cap_ratio": (reaction / np.asarray(meta["impulse_caps"])).tolist(),
            "velocity_peak_rad_s": velocity.tolist(),
            "velocity_cap_ratio": (velocity / meta["velocity_cap"]).tolist(),
            "saturation_s": (saturated.sum(axis=0) * dt).tolist(),
            "saturation_fraction": saturated.mean(axis=0).tolist(),
            "force_sign_changes_per_s": (reversals / (step_count * .02)).tolist(),
            "peak_nail_depth_m": traces["nail_depth"].max(axis=0).tolist(),
            "sensor_status": sensor_status,
            "global_capacity_overflow_count": 0,
            "monitored_contact_match_buffer_overflow_count": 0,
            "point_slot_overflow_count": None if sensor_status["point_slot_quality"] == "unavailable" else 0,
            "nonfinite_count": 0,
            "tail": meta["tail"], "learning_updates": 0,
            "limits": ["reaction proxy, not contact-row isolation", "fixed command replay, not policy qualification",
                       "postterminal tail is a held-command counterfactual", "no automated convergence verdict"],
        }
        write_json(output / "summary.json", summary)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-audit", action="store_true",
                        help="collect complete 2ms prefix discrepancies; never run finer timesteps or relax the gate")
    parser.add_argument("--engineering-equivalence", action="store_true",
                        help="explicit prospective engineering check alongside strict native audit; never authorizes finer runs")
    parser.add_argument("--horizon-steps", type=int,
                        help="common 20ms horizon across captures; must be >= each source-derived minimum")
    args = parser.parse_args()
    if args.engineering_equivalence and not args.native_audit:
        parser.error("--engineering-equivalence requires --native-audit")
    args.output.mkdir(parents=True, exist_ok=False)
    previous = None
    for dt, name in replay_schedule(args.native_audit):
        try:
            result = replay(args.capture, args.output / name, dt, args.device, args.horizon_steps,
                            args.native_audit, args.engineering_equivalence)
            if previous is not None:
                require_pair(previous, result)
                if previous["command_hash"] != result["command_hash"]:
                    raise RuntimeError("command tape differs across timesteps")
            previous = result
        except Exception as exc:
            write_json(args.output / "failure.json", {"dt": dt, "error": str(exc),
                       "status": "numerical attribution blocked; no later refinement executed"})
            raise
    print(json.dumps({"output": str(args.output), "native_fidelity": previous["native_prefix_fidelity"],
                      "engineering_equivalence": previous["engineering_equivalence_status"],
                      "status": "native-only audit complete; no refinement authorized" if args.native_audit else
                      "bounded replay complete; numerical convergence not adjudicated"}))
    if args.native_audit and previous["native_prefix_fidelity"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
