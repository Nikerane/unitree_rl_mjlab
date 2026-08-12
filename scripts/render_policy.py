"""Headless offscreen render of a TRAINED policy rollout (no viewer) -> PNG frames + mp4.

Loads a checkpoint (like `scripts/play.py --agent trained`) but renders with MuJoCo's
offscreen renderer (like `scripts/render_reference.py`) so the trained policy can be
inspected from images alone -- no browser, no display. Works headless on macOS (CGL,
leave MUJOCO_GL unset) and on Linux (export MUJOCO_GL=egl first). CPU is fine:
the checkpoint is map_location'd to the chosen device.

Checkpoints train locally under logs/rsl_rl/ (Lightning writes them in place via
scripts/lightning_pair.sh); for a remote box, rsync the .pt back first. Then:
    python scripts/render_policy.py --checkpoint-file /tmp/model_499.pt
    python scripts/render_policy.py --task Unitree-Z1-Hammer-Track \
        --checkpoint-file /tmp/a_track_seed0_model_499.pt --steps 80 --distance 0.85 --elevation -25

Reads out: montage.png (6 evenly-spaced frames), frame_*.png, policy.mp4,
trajectory.png, trace.npz, and metadata.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import platform as _platform

import imageio.v3 as iio
import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401  (populate the task registry)
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from evaluation.analysis.fixed_reset_video_library import (
  ARTIFACT_FILENAMES,
  EXECUTION_DEVICE_CAMPAIGN_EXEMPT,
  FIXED_RESET_ENVELOPE,
  RENDERER_CONTRACT,
  SUBSTEP_RENDERER_CONTRACT,
  TIMING_CONTRACT,
  WAVE1_ARTIFACT_FILENAMES,
  compose_trajectory_title,
  expected_task,
  load_fixed_reset,
  trajectory_outcome,
  treatment_for_task,
  validate_substep_trace,
  write_metadata,
  write_substep_trajectory_png,
  write_trajectory_png,
)
from scripts.eval_impulse import restore_reset_state
from src.assets.robots.unitree_z1.z1_constants import (
  ARM_JOINT_NAMES,
  HAMMER_HEAD_SITE_NAME,
)
from src.tasks.hammer.mdp.guideline import (
  _ENV_GUIDELINE_ATTR,
  GUIDELINE_NUM_GATES,
  project_to_reference,
)
from src.tasks.hammer.mdp.references import get_strike_reference


@dataclass(frozen=True)
class Cfg:
  checkpoint_file: str
  """Path to a model_*.pt checkpoint (pulled from Vega)."""
  campaign: str
  """Registered campaign identity, e.g. fq4x8 or fq3x8."""
  arm: str
  """Registered treatment arm identity."""
  training_seed: int
  """Training seed for this checkpoint."""
  checkpoint_sha256: str
  """Expected SHA-256 of checkpoint_file, recorded in metadata."""
  code_revision: str
  """Immutable source revision used to render this policy."""
  asset_revision: str
  """Immutable hammer asset revision used to render this policy."""
  task: str = "Unitree-Z1-Hammer"
  """Gym task id: Unitree-Z1-Hammer (A-BASE) or Unitree-Z1-Hammer-Track (A-TRACK)."""
  out_dir: str = "/tmp/hammer_policy"
  steps: int = 80
  fixed_reset_envelope: str = str(FIXED_RESET_ENVELOPE)
  """Stage-0 envelope containing the canonical shared fixed reset."""
  metadata_provenance: str = ""
  """Free-form immutable provenance note for this rendering invocation."""
  device: str = "cpu"
  substep_trace: bool = False
  """Wave-1 mode: record 500 Hz substep evidence and encode slow-motion video."""
  trace_only: bool = False
  """Wave-1 screening: identical rollout and trace, but render no frames or video."""
  training_revision: str = ""
  """Revision the checkpoint was TRAINED at (Wave-1 mode; distinct from analysis)."""
  analysis_revision: str = ""
  """Revision of the renderer/analysis code (Wave-1 mode; must differ from training)."""


# Campaigns whose rendered artifacts are already frozen with recorded SHA-256 hashes. Their
# trajectory.png must re-render byte-identically, so they keep the legacy plot call; every later
# campaign gets treatment-faithful geometry and a full title.
FROZEN_RENDER_CAMPAIGNS = frozenset({"wave1", "wave2", "fq4x8", "fq3x8"})

DIRECT_REFERENCE_LEGEND = (
  "SingleStrikeReference (reward prior 0.10→0 by iteration 250)"
)
DIRECT_REFERENCE_FOOTER = (
  f"{DIRECT_REFERENCE_LEGEND} · black dashed · ante-impact reward only"
)


def trajectory_plot_kwargs(cfg: "Cfg") -> dict[str, str]:
  """Return truthful control-rate trajectory labels for the direct-reference campaign."""
  if cfg.campaign != "fic-direct-reference":
    return {}
  return {
    "reference_legend": DIRECT_REFERENCE_LEGEND,
    "reference_footer": DIRECT_REFERENCE_FOOTER,
  }


def substep_plot_kwargs(cfg: "Cfg", trace: dict, *, terminal_reason: str) -> dict:
  """Plot arguments for one substep leaf, derived from the VALIDATED task, not the directory.

  Success is read off the rollout's terminal reason (``terminated`` == the nail_driven success
  termination fired), matching the frozen analysis definition — never assumed from contact.
  """
  if cfg.campaign in FROZEN_RENDER_CAMPAIGNS:
    return {"title": f"wave1 / {cfg.arm} / seed {cfg.training_seed}"}
  treatment = treatment_for_task(expected_task(cfg.campaign, cfg.arm))
  return {
    "treatment": treatment,
    "title": compose_trajectory_title(
      campaign=cfg.campaign,
      arm=cfg.arm,
      training_seed=cfg.training_seed,
      treatment=treatment,
      outcome=trajectory_outcome(trace, success=terminal_reason == "terminated"),
    ),
  }


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_revision(value: object) -> bool:
  text = str(value)
  return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def validate_wave1_revisions(
  *, training_revision: str, asset_revision: str, analysis_revision: str
) -> dict[str, str]:
  """Require three well-formed revisions with analysis recorded separately.

  The renderer runs at a later commit than the policies it inspects; collapsing the
  two would make a re-render indistinguishable from the training state.
  """
  if not _is_revision(training_revision):
    raise ValueError("training_revision must be a full 40-hex revision")
  if not _is_revision(asset_revision):
    raise ValueError("asset_revision must be a full 40-hex revision")
  if not _is_revision(analysis_revision):
    raise ValueError("analysis_revision must be a full 40-hex revision")
  if analysis_revision == training_revision:
    raise ValueError(
      "analysis_revision must differ from training_revision: analysis provenance is "
      "recorded separately from the frozen training revision"
    )
  return {
    "training": training_revision,
    "asset": asset_revision,
    "analysis": analysis_revision,
  }


def wave1_execution_device(*, requested: str, env_device, tensor) -> dict:
  """Record the device this render actually ran on, not the one it asked for.

  Wave 1 recorded nothing here and had to argue CPU provenance from a bridge.
  The two `actual_*` fields are read back off the live environment and a real
  tensor, so a silent accelerator promotion is recorded verbatim rather than
  normalised away — the validator, not this function, is what rejects it.
  """
  return {
    "requested": str(requested),
    "actual_env_device": str(env_device),
    "actual_tensor_device": str(getattr(tensor, "device", tensor)),
    "platform": _platform.platform(),
  }


def wave1_frame_substep_indices(
  substep_count: int, stride: int, *, capture: bool = True
) -> np.ndarray:
  """Substep indices that emit an RGB frame: the last substep of each stride group.

  `capture=False` is the trace-only screening pass: the rollout, the reset and the
  500 Hz trace are identical, but no image is rendered and no video is written.
  """
  if substep_count <= 0 or stride <= 0:
    raise ValueError("substep_count and stride must be positive")
  if not capture:
    return np.empty(0, dtype=np.int64)
  return np.arange(stride - 1, substep_count, stride, dtype=np.int64)


def build_wave1_substep_trace(
  *,
  head_positions,
  contacts,
  nail_depths,
  arm_qvels,
  arm_qvels_pre,
  arm_joint_names,
  gate_indices,
  perpendicular_errors,
  d_starts,
  f_bests,
  episode_indices,
  control_step_payouts,
  control_step_totals,
  payout_names,
  control_steps_recorded,
  entry,
  nail,
  physics_dt_s: float,
  control_decimation: int,
  executed_control_steps: int,
) -> dict[str, np.ndarray]:
  """Assemble a validated 500 Hz trace from per-substep and per-control recordings.

  Reward payouts stay at their real control-rate timing; broadcasting them to substep
  length would invent measurements the manager never produced.
  """
  if (
    len(control_step_payouts) != executed_control_steps
    or len(control_step_totals) != executed_control_steps
  ):
    raise ValueError(
      "reward payouts must be recorded at the control rate: expected "
      f"{executed_control_steps} rows, got {len(control_step_payouts)}"
    )
  substeps = len(head_positions)
  entry = np.asarray(entry, dtype=np.float64)
  nail = np.asarray(nail, dtype=np.float64)
  gate_centers = entry + (
    np.arange(1, GUIDELINE_NUM_GATES + 1, dtype=np.float64)[:, None]
    / (GUIDELINE_NUM_GATES + 1)
  ) * (nail - entry)
  boundary = np.zeros(substeps, dtype=bool)
  boundary[control_decimation - 1 :: control_decimation] = True
  trace = {
    "substep_head_position_m": np.asarray(head_positions, dtype=np.float64),
    "substep_contact": np.asarray(contacts, dtype=bool),
    "substep_nail_depth_m": np.asarray(nail_depths, dtype=np.float64),
    "substep_arm_qvel_rad_s": np.asarray(arm_qvels, dtype=np.float64),
    "substep_arm_qvel_pre_rad_s": np.asarray(arm_qvels_pre, dtype=np.float64),
    "arm_joint_names": np.asarray(list(arm_joint_names), dtype="<U32"),
    "substep_gate_index": np.asarray(gate_indices, dtype=np.int64),
    "substep_perpendicular_error_m": np.asarray(perpendicular_errors, dtype=np.float64),
    "substep_d_start_m": np.asarray(d_starts, dtype=np.float64),
    "substep_f_best": np.asarray(f_bests, dtype=np.float64),
    "substep_control_step": np.asarray(control_steps_recorded, dtype=np.int64),
    "substep_is_control_boundary": boundary,
    "substep_episode_index": np.asarray(episode_indices, dtype=np.int64),
    "control_step_reward_terms": np.asarray(control_step_payouts, dtype=np.float64),
    "control_step_reward_term_names": np.asarray(payout_names, dtype="<U64"),
    "control_step_reward_total": np.asarray(control_step_totals, dtype=np.float64),
    "guideline_entry_m": entry,
    "guideline_nail_m": nail,
    "guideline_gate_centers_m": gate_centers,
    "guideline_reference_length_m": np.float64(float(np.linalg.norm(nail - entry))),
    "physics_dt_s": np.float64(physics_dt_s),
    "control_decimation": np.int64(control_decimation),
    "executed_control_steps": np.int64(executed_control_steps),
  }
  validate_substep_trace(trace)
  return trace


def _reference_polyline_m(reference, device: str) -> np.ndarray:
  """Return the direct-reference analytical endpoints of the anchored segment."""
  return np.stack(
    [
      reference.waypoint(torch.tensor([phi], device=device))
      .squeeze(0)
      .detach()
      .cpu()
      .numpy()
      .astype(np.float64)
      for phi in (0.0, 1.0)
    ]
  )


def _run_wave1_substep_rollout(
  *,
  cfg: Cfg,
  out: Path,
  contract: dict,
  revisions: dict[str, str],
  env,
  base_env,
  policy,
  obs,
  robot,
  nail,
  sensor,
  head_cfg,
  checkpoint_sha256: str,
  fixed_reset: dict,
  physics_dt_s: float,
  control_decimation: int,
) -> None:
  """Record 500 Hz evidence by wrapping the production per-substep metrics call.

  This reuses the same `metrics_manager.compute_substep` seam that
  `evaluation/guideline/qualify_reference.py` already relies on, so the recorded
  state is the one production physics actually produced.
  """
  stride = int(contract["frame_substep_stride"])
  tracker = getattr(base_env, _ENV_GUIDELINE_ATTR, None)
  if tracker is None:
    raise RuntimeError(
      "task has no WaypointProgressTracker; Wave-1 substep tracing requires a "
      "registered guideline task"
    )
  arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
  arm_cfg.resolve(base_env.scene)

  recorded: dict[str, list] = {
    key: []
    for key in (
      "head_positions", "contacts", "nail_depths", "arm_qvels", "arm_qvels_pre",
      "gate_indices", "perpendicular_errors", "d_starts", "f_bests",
      "episode_indices", "control_steps_recorded",
    )
  }
  frames: list[np.ndarray] = []
  state = {"active": False, "episode": 0, "control_step": 0}
  original_substep = base_env.metrics_manager.compute_substep
  original_sim_step = base_env.sim.step

  def arm_qvel() -> np.ndarray:
    return (
      robot.data.joint_vel[0, arm_cfg.joint_ids]
      .detach().cpu().numpy().astype(np.float64).copy()
    )

  def preintegration_then_step() -> None:
    # MuJoCo's semi-implicit integrator leaves qvel one substep ahead of site_xpos.
    # qualify_reference.py solves this with the same pre-hook; without it the speed
    # and the position in one trace row are 2 ms apart.
    if state["active"]:
      recorded["arm_qvels_pre"].append(arm_qvel())
    original_sim_step()

  def record_substep() -> None:
    original_substep()  # post-integration, exactly as production sees it
    if not state["active"]:
      return
    head_t = robot.data.site_pos_w[:, head_cfg.site_ids].squeeze(1)
    if bool(tracker.initialized[0]):
      _, error = project_to_reference(head_t, tracker.entry, tracker.nail)
      error_m = float(error[0])
    else:
      error_m = 0.0
    recorded["head_positions"].append(
      head_t[0].detach().cpu().numpy().astype(np.float64).copy()
    )
    recorded["contacts"].append(bool((sensor.data.found[0] > 0).any()))
    recorded["nail_depths"].append(float(nail.data.joint_pos[0, 0]))
    recorded["arm_qvels"].append(arm_qvel())
    recorded["gate_indices"].append(int(tracker.next_gate[0]))
    recorded["perpendicular_errors"].append(error_m)
    recorded["d_starts"].append(float(tracker.target_start_distance[0]))
    recorded["f_bests"].append(float(tracker.best_target_fraction[0]))
    recorded["episode_indices"].append(int(state["episode"]))
    recorded["control_steps_recorded"].append(int(state["control_step"]))
    if (len(recorded["head_positions"]) - 1) in frame_indices:
      frame = base_env.render()
      if frame is None:
        raise RuntimeError("offscreen renderer returned no frame during a substep")
      frames.append(np.asarray(frame))

  frame_indices = set(
    wave1_frame_substep_indices(
      cfg.steps * control_decimation, stride, capture=not cfg.trace_only
    ).tolist()
  )
  payout_names = list(base_env.reward_manager.active_terms)
  payouts: list[np.ndarray] = []
  totals: list[float] = []
  reward_dt = physics_dt_s * control_decimation
  terminal_boundary = {"detected": False, "step": None, "reason": "step_limit"}
  rows = [f"{'step':>4} {'nail_mm':>8} {'contact':>8} {'reward':>9}"]
  base_env.metrics_manager.compute_substep = record_substep
  base_env.sim.step = preintegration_then_step
  try:
    state["active"] = True
    for k in range(1, cfg.steps + 1):
      state["control_step"] = k - 1
      with torch.inference_mode():
        actions = policy(obs)
      step_out = env.step(actions)
      obs, rew, dones = step_out[0], step_out[1], step_out[2]
      # Control-rate manager payouts, kept at their real timing.
      payouts.append(
        base_env.reward_manager._step_reward[0]
        .detach().cpu().numpy().astype(np.float64).copy()
        * reward_dt  # _step_reward stores value/dt; restore the payout actually added
      )
      totals.append(float(rew[0]))
      rows.append(
        f"{k:>4} {float(nail.data.joint_pos[0, 0]) * 1000.0:>8.1f} "
        f"{str(bool((sensor.data.found[0] > 0).any())):>8} {float(rew[0]):>9.3f}"
      )
      if bool(dones[0]):
        # auto_reset is disabled and we break immediately, so in practice no
        # post-boundary substep occurs. The counter advances anyway as
        # defence-in-depth: if the break is ever removed, a post-boundary substep
        # would be stamped episode 1 and rejected by validate_substep_trace
        # rather than silently folded into episode 0.
        state["episode"] += 1
        terminal_boundary = {
          "detected": True,
          "step": k,
          "reason": (
            "terminated" if bool(base_env.reset_terminated[0])
            else "timeout" if bool(base_env.reset_time_outs[0])
            else "done"
          ),
        }
        rows.append(f"  -- episode boundary after step {k}; no post-reset samples --")
        break
  finally:
    state["active"] = False
    base_env.metrics_manager.compute_substep = original_substep
    base_env.sim.step = original_sim_step

  executed = len(payouts)
  trace = build_wave1_substep_trace(
    **recorded,
    arm_joint_names=ARM_JOINT_NAMES,
    control_step_payouts=payouts,
    control_step_totals=totals,
    payout_names=payout_names,
    entry=tracker.entry[0].detach().cpu().numpy().astype(np.float64).copy(),
    nail=tracker.nail[0].detach().cpu().numpy().astype(np.float64).copy(),
    physics_dt_s=physics_dt_s,
    control_decimation=control_decimation,
    executed_control_steps=executed,
  )
  # Wave 1 is frozen with no device block and a documented inferred-CPU bridge;
  # re-rendering it must stay byte-identical, so only later campaigns record one.
  device_block = (
    {}
    if cfg.campaign in EXECUTION_DEVICE_CAMPAIGN_EXEMPT
    else {
      "execution_device": wave1_execution_device(
        requested=cfg.device,
        env_device=base_env.device,
        tensor=nail.data.joint_pos,
      )
    }
  )
  if cfg.trace_only:
    np.savez(out / "trace.npz", **trace)
    write_metadata(
      out / "metadata.json",
      {
        **device_block,
        "campaign": cfg.campaign,
        "arm": cfg.arm,
        "training_seed": cfg.training_seed,
        "task": cfg.task,
        "checkpoint_sha256": checkpoint_sha256,
        "training_revision": revisions["training"],
        "asset_revision": revisions["asset"],
        "analysis_revision": revisions["analysis"],
        "reset_state_digest": fixed_reset["reset_state_digest"],
        "trace_only": True,
        "rollout": {
          "requested_control_steps": cfg.steps,
          "executed_control_steps": executed,
          "substep_count": int(len(trace["substep_head_position_m"])),
          "frame_count": 0,
          "auto_reset_enabled": False,
          "terminal_boundary": terminal_boundary,
        },
        "artifacts": {"trace.npz": _sha256(out / "trace.npz")},
      },
    )
    contacted = bool(trace["substep_contact"].any())
    print(
      f"[screen] wave1 {cfg.arm}/seed{cfg.training_seed}: "
      f"contact={contacted} gates={int(trace['substep_gate_index'].max())} "
      f"nail_depth_max={float(trace['substep_nail_depth_m'].max()):.5f} "
      f"substeps={len(trace['substep_head_position_m'])} steps={executed}"
    )
    return

  expected_frames = len(trace["substep_head_position_m"]) // stride
  if len(frames) != expected_frames:
    raise RuntimeError(
      f"frame/substep mismatch: {len(frames)} frames for "
      f"{len(trace['substep_head_position_m'])} substeps at stride {stride}"
    )

  np.savez(out / "trace.npz", **trace)
  iio.imwrite(out / "policy.mp4", np.stack(frames), fps=contract["fps"])
  idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
  iio.imwrite(out / "montage.png", np.concatenate([frames[i] for i in idx], axis=1))
  for j, i in enumerate(idx):
    iio.imwrite(out / f"frame_{j}_substep{int(i) * stride + stride - 1:04d}.png", frames[i])
  # Frame k shows the pose AFTER the substep its index names: the offscreen renderer
  # runs its own mj_forward, so the image leads the trace row by one integration.
  write_substep_trajectory_png(
    trace,
    out / "trajectory.png",
    **substep_plot_kwargs(cfg, trace, terminal_reason=str(terminal_boundary["reason"])),
  )
  write_metadata(
    out / "metadata.json",
    {
      **device_block,
      "campaign": cfg.campaign,
      "arm": cfg.arm,
      "training_seed": cfg.training_seed,
      "task": cfg.task,
      "checkpoint_sha256": checkpoint_sha256,
      "checkpoint_file": str(Path(cfg.checkpoint_file)),
      "training_revision": revisions["training"],
      "asset_revision": revisions["asset"],
      "analysis_revision": revisions["analysis"],
      "reset_state_digest": fixed_reset["reset_state_digest"],
      "reset_envelope": str(cfg.fixed_reset_envelope),
      "renderer_contract": contract,
      "timing": TIMING_CONTRACT,
      "rollout": {
        "requested_control_steps": cfg.steps,
        "executed_control_steps": executed,
        "substep_count": int(len(trace["substep_head_position_m"])),
        "frame_count": len(frames),
        "auto_reset_enabled": False,
        "terminal_boundary": terminal_boundary,
      },
      "video_seconds": len(frames) / float(contract["fps"]),
      "simulated_seconds": executed * physics_dt_s * control_decimation,
      "reward_term_names": payout_names,
      "guideline_reference": "WaypointProgressTracker entry->nail (frozen)",
      "frame_substep_alignment": (
        "frame j shows the pose after substep j*stride+stride-1; the offscreen "
        "renderer runs mj_forward, so an image leads its trace row by one integration"
      ),
      "reward_payout_semantics": (
        "control_step_reward_terms are dt-scaled manager payouts (RewardManager "
        "_step_reward * control_dt); control_step_reward_total is the env step reward"
      ),
      "metadata_provenance": cfg.metadata_provenance,
      "artifacts": {name: _sha256(out / name) for name in WAVE1_ARTIFACT_FILENAMES},
    },
  )
  print("\n".join(rows))
  print(
    f"\n[render] wave1 {cfg.arm}/seed{cfg.training_seed}: "
    f"{len(trace['substep_head_position_m'])} substeps, {executed} control steps, "
    f"{len(frames)} frames @ {contract['fps']} fps -> {out}/"
  )


def main(cfg: Cfg) -> None:
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  ckpt = Path(cfg.checkpoint_file)
  if not ckpt.exists():
    raise FileNotFoundError(f"checkpoint not found: {ckpt}")
  if not cfg.code_revision or not cfg.asset_revision:
    raise ValueError("code_revision and asset_revision are required")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive")
  contract = SUBSTEP_RENDERER_CONTRACT if cfg.substep_trace else RENDERER_CONTRACT
  revisions = (
    validate_wave1_revisions(
      training_revision=cfg.training_revision,
      asset_revision=cfg.asset_revision,
      analysis_revision=cfg.analysis_revision,
    )
    if cfg.substep_trace
    else None
  )
  registered_task = expected_task(cfg.campaign, cfg.arm)
  if cfg.task != registered_task:
    raise ValueError(
      f"task identity mismatch: {cfg.campaign}/{cfg.arm} requires {registered_task}, "
      f"got {cfg.task}"
    )
  checkpoint_sha256 = _sha256(ckpt)
  if checkpoint_sha256 != cfg.checkpoint_sha256:
    raise ValueError(
      "checkpoint SHA-256 mismatch: "
      f"expected {cfg.checkpoint_sha256}, got {checkpoint_sha256}"
    )
  fixed_reset = load_fixed_reset(cfg.fixed_reset_envelope)

  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = 1
  # Keep the terminal strike state readable; auto-reset would replace it with
  # the next episode before its trajectory/contact/frame could be recorded.
  env_cfg.auto_reset = False
  env_cfg.viewer.width = contract["frame_width_px"]
  env_cfg.viewer.height = contract["frame_height_px"]
  env_cfg.viewer.distance = contract["camera_distance_m"]
  env_cfg.viewer.elevation = contract["camera_elevation_deg"]
  env_cfg.viewer.azimuth = contract["camera_azimuth_deg"]

  agent_cfg = load_rl_cfg(cfg.task)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  physics_dt_s = float(base_env.physics_dt)
  control_decimation = int(env_cfg.decimation)
  control_dt_s = physics_dt_s * control_decimation
  if (
    abs(physics_dt_s - TIMING_CONTRACT["physics_dt_s"]) > 1e-12
    or control_decimation != TIMING_CONTRACT["control_decimation"]
    or abs(control_dt_s - TIMING_CONTRACT["control_dt_s"]) > 1e-12
  ):
    raise RuntimeError(
      "renderer timing contract mismatch: "
      f"dt={physics_dt_s}, decimation={control_decimation}, control_dt={control_dt_s}"
    )

  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device)

  nail = base_env.scene["nail_block"]
  sensor = base_env.scene["hammer_nail_contact"]
  head_cfg = SceneEntityCfg("robot", site_names=(HAMMER_HEAD_SITE_NAME,))
  nail_cfg = SceneEntityCfg("nail_block", site_names=("nail_top",))
  head_cfg.resolve(base_env.scene)
  nail_cfg.resolve(base_env.scene)
  robot = base_env.scene["robot"]
  head = lambda: (
    robot.data.site_pos_w[0, head_cfg.site_ids]
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
    .copy()
  )
  nail_top = lambda: (
    nail.data.site_pos_w[0, nail_cfg.site_ids]
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
    .copy()
  )
  depth_mm = lambda: float(nail.data.joint_pos[0, 0]) * 1000.0

  env.reset()
  # restore_reset_state() is the established D2 replay helper and accepts the
  # complete reset record, rather than its nested reset_state mapping.
  restore_reset_state(base_env, fixed_reset)
  base_env.sim.forward()
  base_env.sim.sense()
  base_env.obs_buf = base_env.observation_manager.compute(update_history=True)
  obs = env.get_observations()
  # Freeze target geometry before policy inference can drive the nail.
  nail_top_m = nail_top()
  # Observation construction owns the task's shared reference and anchors it
  # against this restored state.  Preserve its analytical phi vertices, not a
  # realized playback trace or a start-to-contact chord.
  reference = get_strike_reference(base_env)
  if not bool(reference._anchored[0]):
    raise RuntimeError("SingleStrikeReference was not anchored by reset observations")
  reference_polyline = _reference_polyline_m(reference, base_env.device)

  if cfg.substep_trace:
    assert revisions is not None
    _run_wave1_substep_rollout(
      cfg=cfg,
      out=out,
      contract=contract,
      revisions=revisions,
      env=env,
      base_env=base_env,
      policy=policy,
      obs=obs,
      robot=robot,
      nail=nail,
      sensor=sensor,
      head_cfg=head_cfg,
      checkpoint_sha256=checkpoint_sha256,
      fixed_reset=fixed_reset,
      physics_dt_s=physics_dt_s,
      control_decimation=control_decimation,
    )
    return

  frames: list[np.ndarray] = []
  head_positions = [head()]
  contacts = [bool((sensor.data.found[0] > 0).any())]
  actions_recorded: list[np.ndarray] = []
  terminal_boundary = {"detected": False, "step": None, "reason": "step_limit"}
  rows = [f"{'step':>4} {'nail_mm':>8} {'contact':>8} {'reward':>9}"]
  f0 = base_env.render()
  if f0 is not None:
    frames.append(np.asarray(f0))
    iio.imwrite(out / "step_000_reset.png", frames[-1])

  for k in range(1, cfg.steps + 1):
    with torch.inference_mode():
      actions = policy(obs)
    actions_recorded.append(actions[0].detach().cpu().numpy().astype(np.float32).copy())
    step_out = env.step(actions)
    obs, rew, dones = step_out[0], step_out[1], step_out[2]
    contact = bool((sensor.data.found > 0).any())
    head_positions.append(head())
    contacts.append(contact)
    rows.append(f"{k:>4} {depth_mm():>8.1f} {str(contact):>8} {float(rew[0]):>9.3f}")
    fr = base_env.render()
    if fr is not None:
      frames.append(np.asarray(fr))
    if bool(dones[0]):
      terminal_boundary = {
        "detected": True,
        "step": k,
        "reason": (
          "terminated" if bool(base_env.reset_terminated[0])
          else "timeout" if bool(base_env.reset_time_outs[0])
          else "done"
        ),
      }
      rows.append(f"  -- episode boundary after step {k}; terminal frame retained --")
      break
  if not frames:
    raise RuntimeError("renderer produced no RGB frames")
  idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
  iio.imwrite(out / "montage.png", np.concatenate([frames[i] for i in idx], axis=1))
  for j, i in enumerate(idx):
    iio.imwrite(out / f"frame_{j}_idx{int(i):03d}.png", frames[i])
  iio.imwrite(out / "policy.mp4", np.stack(frames), fps=RENDERER_CONTRACT["fps"])
  trace = {
    "head_position_m": np.asarray(head_positions, dtype=np.float64),
    "contact": np.asarray(contacts, dtype=bool),
    "contact_live_at_control_boundary": np.asarray(contacts, dtype=bool),
    "control_step": np.arange(len(head_positions), dtype=np.int64),
    "action": np.asarray(actions_recorded[: len(head_positions) - 1], dtype=np.float32),
    "reference_polyline_m": reference_polyline,
    "nail_top_m": nail_top_m,
  }
  np.savez(out / "trace.npz", **trace)
  write_trajectory_png(
    trace,
    out / "trajectory.png",
    **trajectory_plot_kwargs(cfg),
  )
  # Same gating as the substep branch: the validator requires a device block for
  # every non-exempt campaign, so BOTH metadata sites must emit one or a default
  # render of a wave2 policy would fail validation it should pass.
  write_metadata(
    out / "metadata.json",
    {
      **(
        {}
        if cfg.campaign in EXECUTION_DEVICE_CAMPAIGN_EXEMPT
        else {
          "execution_device": wave1_execution_device(
            requested=cfg.device,
            env_device=base_env.device,
            tensor=nail.data.joint_pos,
          )
        }
      ),
      "campaign": cfg.campaign,
      "arm": cfg.arm,
      "training_seed": cfg.training_seed,
      "task": cfg.task,
      "checkpoint_sha256": checkpoint_sha256,
      "checkpoint_file": str(ckpt),
      "code_revision": cfg.code_revision,
      "asset_revision": cfg.asset_revision,
      "presentation_generator_revision": cfg.code_revision,
      "reset_state_digest": fixed_reset["reset_state_digest"],
      "reset_envelope": str(cfg.fixed_reset_envelope),
      "renderer_contract": RENDERER_CONTRACT,
      "timing": TIMING_CONTRACT,
      "rollout": {
        "requested_control_steps": cfg.steps,
        "executed_control_steps": len(actions_recorded),
        "frame_count": len(frames),
        "auto_reset_enabled": False,
        "terminal_boundary": terminal_boundary,
      },
      "output_dimensions_px": {
        "frame": [RENDERER_CONTRACT["frame_width_px"], RENDERER_CONTRACT["frame_height_px"]],
        "montage": [
          RENDERER_CONTRACT["frame_width_px"] * min(6, len(frames)),
          RENDERER_CONTRACT["frame_height_px"],
        ],
      },
      "metadata_provenance": cfg.metadata_provenance,
      "contact_semantics": "live hammer_head_0/nail contact at control boundary",
      "artifacts": {name: _sha256(out / name) for name in ARTIFACT_FILENAMES},
    },
  )

  print("\n".join(rows))
  print(
    f"\n[render] {len(frames)} frames + artifact contract -> {out}/  "
    f"(ckpt={ckpt.name}, task={cfg.task})"
  )
  print("[render] read montage.png / trajectory.png to inspect the rollout.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
