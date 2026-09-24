"""Illustrative first episodes, same fixed controller, red commanded curve."""
from dataclasses import asdict
import json
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_runner_cls
from mjlab.viewer import ViewerConfig
from pilot import config, TASK
from scripts.evaluate_impact_speed import override_latched_goal
from src.tasks.hammer.mdp.sampled_pd500 import install_sampled_pd500
from src.tasks.hammer.rl.impact_speed_wrapper import wrap_impact_speed_env


def record(checkpoint, output, device="cuda:0"):
    output = Path(output)
    output.mkdir(exist_ok=False,parents=True)
    for goal in (1.5,1.85,2.2):
        cfg = config(1)
        cfg.env.seed = 2026092401
        # Preserve terminal pose for rendering; stop at the first native done.
        cfg.env.auto_reset = False
        cfg.env.viewer.width, cfg.env.viewer.height = 960,640
        cfg.env.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.env.viewer.lookat = (.3,0.,.18)
        cfg.env.viewer.distance = .95
        cfg.env.viewer.elevation, cfg.env.viewer.azimuth = -25.,135.
        env = ManagerBasedRlEnv(cfg.env,device=device,render_mode="rgb_array")
        try:
            install_sampled_pd500(env)
            wrapper = wrap_impact_speed_env(env,1.)
            runner = load_runner_cls(TASK)(wrapper,asdict(cfg.agent),device=device)
            runner.load(str(checkpoint),load_cfg={"actor":True},strict=True,map_location=device)
            env.seed(2026092401)
            obs,_ = wrapper.reset()
            obs = TensorDict(override_latched_goal(env,goal),batch_size=[1])
            original = env.update_visualizers
            def reference_overlay(visualizer):
                original(visualizer)
                reference = getattr(env,"_strike_reference",None)
                if reference is not None:
                    points = reference.reference_polyline(0,65).detach().cpu().numpy()
                    for start,end in zip(points[:-1],points[1:]):
                        visualizer.add_cylinder(start,end,radius=.0006,color=(.9,.04,.02,1.))
            env.update_visualizers = reference_overlay
            frames = [np.asarray(env.render())]
            policy = runner.get_inference_policy(device=device)
            with torch.inference_mode():
                for _ in range(env.max_episode_length+1):
                    obs,_,done,_ = wrapper.step(policy(obs))
                    frames.append(np.asarray(env.render()))
                    if bool(done[0]):
                        break
            label = f"goal-{goal}"
            iio.imwrite(output/f"{label}.mp4",np.stack(frames),fps=50)
            iio.imwrite(output/f"{label}-slow.mp4",np.stack(frames),fps=12.5)
            indices = np.linspace(0,len(frames)-1,6).astype(int)
            iio.imwrite(output/f"{label}-montage.png",np.concatenate([frames[i] for i in indices],axis=1))
            (output/f"{label}.json").write_text(json.dumps(dict(checkpoint=str(checkpoint),seed=2026092401,
                goal_mps=goal,frames=len(frames),physics_dt=.00025,pd_dt=.002,policy_dt=.02,
                note="Illustrative single-world first episode, no cherry-picking; video is not the128-world evaluation. Red is commanded reference. Auto-reset off to retain terminal frame."),indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    args = p.parse_args()
    record(args.checkpoint,args.output,args.device)
