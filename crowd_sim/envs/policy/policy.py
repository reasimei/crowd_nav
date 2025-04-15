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
    def predict(self, state,robot_index):
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        # 计算当前位置到目标点的距离
        goal_distance = np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx))
        
        # 获取移动方向
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        
        # 根据主要移动方向调整阈值
        if abs(dx) > abs(dy):
            # 主要水平移动（左右）
            if dx > 0:  # 左到右
                # 左到右移动可能过头，使用更严格的条件
                threshold = self_state.radius * 0.8
            else:  # 右到左
                # 右到左移动可能不足，使用宽松一点的条件
                threshold = self_state.radius * 1.5
        else:
            # 主要垂直移动（上下）
            if dy > 0:  # 下到上
                # 下到上移动可能不足，使用宽松一点的条件
                threshold = self_state.radius * 1.5
            else:  # 上到下
                # 上到下移动可能过头，使用更严格的条件
                threshold = self_state.radius * 0.8
        
        # 使用调整后的阈值判断是否到达目标
        return goal_distance < threshold
