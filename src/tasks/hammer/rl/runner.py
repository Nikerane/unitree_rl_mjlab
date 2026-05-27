"""On-policy runner for the hammer-nail task."""

import wandb

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import DifferentialIKAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner


def _get_hammer_metadata(env, run_path: str) -> dict:
    robot: Entity = env.scene["robot"]
    ik_action = env.action_manager.get_term("ik_hammer_head")
    assert isinstance(ik_action, DifferentialIKAction)
    return {
        "run_path": run_path,
        "action_type": "ik_delta_pos",
        "frame_name": ik_action.cfg.frame_name,
        "delta_pos_scale": ik_action.cfg.delta_pos_scale,
        "joint_names": list(robot.joint_names),
        "observation_names": env.observation_manager.active_terms["actor"],
    }


class HammerOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = _get_hammer_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg.get("upload_model"):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
