import os
import numpy as np

import torch
import torch.nn as nn
from leaderboard.autoagents.rl_agent.models.ppo.ppo import ActorCritic

class PPOAgent(object):
    def __init__(self, action_std_init=0.4):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = 100
        self.action_dim = 2
        self.action_std = action_std_init
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std).to(self.device)


    def get_action(self, obs, train):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, _ = self.policy.get_action_and_log_prob(obs.to(self.device))
        return action.detach().cpu().numpy().flatten()

   
    def load(self):
        checkpoint_file = "leaderboard/autoagents/rl_agent/checkpoints/ppo_policy.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))