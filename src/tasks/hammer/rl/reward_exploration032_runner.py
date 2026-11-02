"""Existing runner with matched fresh initialization for three exploration modes."""
from copy import deepcopy
import json
from pathlib import Path
import random
import numpy as np
import torch
from tensordict import TensorDict
from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
from src.tasks.hammer.rl.impact_speed_policy import ImpactSpeedAgeModel
from src.tasks.hammer.rl.reward_exploration032 import RewardExplorationActor
from scripts.z1_exploration_pilot_training import tensor_state_digest


def initialize_shared_models(runner):
    """Same canonical mean/critic/normalizers, without copying exploration extras."""
    seed = runner.cfg['seed']
    device = torch.device(runner.device)
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        x = torch.zeros(1,43,device=device)
        obs = TensorDict({'actor':x, 'critic':x.clone()}, batch_size=[1])
        groups = {'actor':['actor'], 'critic':['critic']}
        # The existing Mjlab runner has already stripped unsupported None options.
        ac = deepcopy(runner.cfg['actor']);ac.pop('class_name',None)
        ac['distribution_cfg'] = {'class_name':'src.tasks.hammer.rl.reward_exploration032:RewardExplorationGaussian',
                                  'mode':'independent'}
        cc = deepcopy(runner.cfg['critic']);cc.pop('class_name',None)
        cc.pop('distribution_cfg',None)
        a = RewardExplorationActor(obs,groups,'actor',13,**ac).to(device)
        c = ImpactSpeedAgeModel(obs,groups,'critic',1,distribution_cfg=None,**cc).to(device)
    actor,critic = runner.alg.actor,runner.alg.critic
    actor.mlp.load_state_dict(a.mlp.state_dict(),strict=True)
    actor.obs_normalizer.load_state_dict(a.obs_normalizer.state_dict(),strict=True)
    critic.load_state_dict(c.state_dict(),strict=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    return dict(seed=seed, actor_mean=tensor_state_digest(actor.mlp.state_dict()),
                critic=tensor_state_digest(critic.state_dict()),
                actor_normalizer=tensor_state_digest(actor.obs_normalizer.state_dict()))


class RewardExplorationRunner(HammerOnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.cfg.get('resume') or self.current_learning_iteration != 0 or self.alg.optimizer.state:
            raise ValueError('032 study entry supports fresh training only')
        initial = initialize_shared_models(self)
        output = Path(self.logger.log_dir)
        with (output / 'initial-state032.json').open('x') as f:
            json.dump(initial,f,sort_keys=True,indent=2)
        return super().learn(num_learning_iterations,init_at_random_ep_len)
