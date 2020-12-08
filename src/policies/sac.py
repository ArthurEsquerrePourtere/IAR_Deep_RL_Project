from machin.frame.algorithms import SAC
from machin.utils.logging import default_logger as logger
from torch.nn.functional import softplus
from torch.distributions import Normal
import torch as t
import torch.nn as nn
import gym
import numpy as np
import random
from typing import List


def atanh(x):
    return 0.5 * t.log((1 + x) / (1 - x))


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = action_range


    @t.jit.ignore
    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = self.mu_head(a)
        sigma = softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        act = (atanh(action / self.action_range)
               if action is not None
               else dist.rsample())
        #print(type(dist.rsample()))
        act_entropy = dist.entropy()

        act_log_prob = dist.log_prob(act)
        act_tanh = t.tanh(act)
        act = act_tanh * self.action_range

        act_log_prob -= t.log(self.action_range *
                              (1 - act_tanh.pow(2)) +
                              1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)

        # If your distribution is different from "Normal" then you may either:
        # 1. deduce the remapping function for your distribution and clamping
        #    function such as tanh
        # 2. clamp you action, but please take care:
        #    1. do not clamp actions before calculating their log probability,
        #       because the log probability of clamped actions might will be
        #       extremely small, and will cause nan
        #    2. do not clamp actions after sampling and before storing them in
        #       the replay buffer, because during update, log probability will
        #       be re-evaluated they might also be extremely small, and network
        #       will "nan". (might happen in PPO, not in SAC because there is
        #       no re-evaluation)
        # Only clamp actions sent to the environment, this is equivalent to
        # change the action reward distribution, will not cause "nan", but
        # this makes your training environment further differ from you real
        # environment.
        #print(type(act))
        return act, act_log_prob, act_entropy

    def save_model(self, filename) -> None:
        """
        Save a neural network model into a file
        :param filename: the filename, including the path
        :return: nothing
        """
        #t.save(self, filename)
        traced=t.jit.script(self)
        t.jit.save(traced,filename)


    @t.jit.export
    def select_action(self,state: List[float],deterministic: bool=False) -> List[float]:
        state = t.tensor(state, dtype=t.float32)
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = self.mu_head(a)
        sigma = softplus(self.sigma_head(a))
        #print(type(sigma))
        #act = t.tensor(random.gauss(mu, sigma))
        if deterministic:
            act=mu
        else:
            act=t.empty(1).normal_(mean=mu.item(),std=sigma.item())
        #print(type(act))
        act_tanh = t.tanh(act)
        act = act_tanh * self.action_range
        action: List[float]=act.data.tolist()
        #print(type(act))
        return action

    # def select_action(self,state,deterministic=False):
    #     state = t.tensor(state, dtype=t.float32).view(1, 3)
    #     return self.forward(state)[0].detach().numpy()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q
