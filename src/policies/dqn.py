from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
import numpy as np
import random
from typing import List


# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)

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
        act=self.forward(state)
        action: List[float] = act.data.tolist()
        return action
