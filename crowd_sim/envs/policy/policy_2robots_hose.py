import abc
import numpy as np


class Policy(object):
    def __init__(self):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        # 添加一个额外的裕度，当机器人非常接近目标点时，视为已到达
        goal_distance = np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx))
        
        # 标准条件：距离小于半径
        standard_condition = goal_distance < self_state.radius
        
        # 宽松条件：距离小于一个小常数（比半径更小，但足够接近目标）
        relaxed_condition = goal_distance < 0.1
        
        # 当任一条件满足时，认为到达目标
        if standard_condition or relaxed_condition:
            return True
        else:
            return False
