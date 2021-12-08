from agent.agent import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):

    def __init__(self, alpha, num_action=5, move_penalty=-0.01):
        self.alpha = alpha
        self.num_action = num_action
        self.dist = np.random.dirichlet(np.full((num_action, ), alpha))

    def act(self, obs):
        action = np.random.choice(np.arange(self.num_action), p=self.dist)
        return action




