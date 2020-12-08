from machin.frame.algorithms import DDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
#from policies.generic_net import GenericNet
#from policies.policy_wrapper import PolicyWrapper
from os import chdir
import numpy as np
from typing import List


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a

    def save_model(self, filename) -> None:
        traced=t.jit.script(self)
        t.jit.save(traced,filename)


    @t.jit.export
    def select_action(self,state: List[float],deterministic: bool=False) -> List[float]:
        state = t.tensor(state, dtype=t.float32)
        act=self.forward(state)
        action: List[float] = act.data.tolist()
        return action

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


