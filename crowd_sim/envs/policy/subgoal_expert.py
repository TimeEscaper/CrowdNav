import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionPoint


class SubgoalExpert(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'subgoal'
        self.multiagent_training = None

    def configure(self, config):
        assert True

    def predict(self, state):
        return ActionPoint.create_empty()
