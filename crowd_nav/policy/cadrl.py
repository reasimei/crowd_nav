import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value


class CADRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('cadrl', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        logging.info('Policy: CADRL without occupancy map')

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            # next_px1 = state.px1 + action1.vx * self.time_step
            # next_px2 = state.px2 + action2.vx * self.time_step
            # next_py1 = state.py1 + action1.vy * self.time_step
            # next_py2 = state.py2 + action2.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                # next_px1 = state.px + action1.vx * self.time_step
                # next_px2 = state.px + action2.vx * self.time_step
                # next_py1 = state.py + action1.vy * self.time_step
                # next_py2 = state.py + action2.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                # next_theta1 = state.theta + action1.r
                # next_theta2 = state.theta + action2.r
                # next_vx1 = action1.v * np.cos(next_theta1)
                # next_vy1 = action1.v * np.sin(next_theta1)
                # next_vx2 = action2.v * np.cos(next_theta2)
                # next_vy2 = action2.v * np.sin(next_theta2)
                # next_px1 = state.px1 + next_vx1 * self.time_step
                # next_px2 = state.px2 + next_vx2 * self.time_step
                # next_py1 = state.py1 + next_vy1 * self.time_step
                # next_py2 = state.py2 + next_vy2 * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta)
        else:
            raise ValueError('Type error')

        return next_state

    def predict(self, state, robot_index):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                next_observed_state = self.propagate(state.other_robot_state, action)
                ob, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_observed_state + next_human_state]).to(self.device)
                                              for next_human_state in ob], dim=0)
                # VALUE UPDATE
                outputs = self.model(self.rotate(batch_next_states))
                min_output, min_index = torch.min(outputs, 0)
                min_value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * min_output.data.item()
                self.action_values.append(min_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def transform(self, state):
        """
        Transform the state passed from the agent to the input of the value network.
        :param state:
        :return: tensor of shape (# of humans + 1, state_length)
        """
        state_list = []

        # Self state and other robot state
        self_state = torch.Tensor(state.self_state).to(self.device)
        other_robot_state = torch.Tensor(state.other_robot_state).to(self.device)

        # Include other robot's state
        joint_state = torch.cat((self_state, other_robot_state))

        # Now, for each human
        for human_state in state.human_states:
            human_state_tensor = torch.Tensor(human_state).to(self.device)
            combined_state = torch.cat((joint_state, human_state_tensor))
            state_list.append(combined_state)

        # Stack the state tensors
        state_tensor = torch.stack(state_list)
        state_tensor = self.rotate(state_tensor)
        return state_tensor

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 确保状态有效性
        if state is None:
            logging.warning("Rotate received None state")
            return torch.zeros((1, 13), device=self.device)
            
        try:
            # 检查输入维度并适当处理
            if len(state.shape) == 1:
                # 1D tensor, 添加batch维度
                state = state.unsqueeze(0)
            elif len(state.shape) == 3:
                # 3D tensor, 压缩为2D
                state = state.reshape(state.shape[0] * state.shape[1], state.shape[2])
                
            batch = state.shape[0]
            
            # 检查特征维度
            feature_dim = state.shape[1]
            if feature_dim < 6:
                logging.warning(f"State has too few features: {feature_dim}, padding to minimum required")
                padding = torch.zeros((batch, 6 - feature_dim), device=state.device)
                state = torch.cat([state, padding], dim=1)
                feature_dim = 6
                
            # 获取相关特征 - 安全访问
            if feature_dim >= 7:
                # px, py, vx, vy, radius, gx, gy, v_pref, theta, px1, py1, vx1, vy1, radius1
                # 0   1   2   3   4      5   6   7      8      9    10   11   12   13
                px = state[:, 0].unsqueeze(1)
                py = state[:, 1].unsqueeze(1)
                vx = state[:, 2].unsqueeze(1)
                vy = state[:, 3].unsqueeze(1)
                radius = state[:, 4].unsqueeze(1)
                gx = state[:, 5].unsqueeze(1)
                gy = state[:, 6].unsqueeze(1)
            else:
                # 最小化版本
                px = state[:, 0].unsqueeze(1)
                py = state[:, 1].unsqueeze(1)
                vx = state[:, 2].unsqueeze(1)
                vy = state[:, 3].unsqueeze(1)
                radius = state[:, 4].unsqueeze(1)
                gx = torch.zeros_like(px)
                gy = torch.zeros_like(py)
            
            # 如果有额外特征（如v_pref和theta）
            if feature_dim >= 9:
                v_pref = state[:, 7].unsqueeze(1)
                theta = state[:, 8].unsqueeze(1)
            else:
                v_pref = torch.ones_like(px)
                theta = torch.zeros_like(px)
            
            # 计算差值
            dx = gx - px
            dy = gy - py
            rot = torch.atan2(dy, dx)
            
            # 旋转速度矢量
            dg = torch.sqrt(dx ** 2 + dy ** 2)
            new_vx = torch.cos(rot) * vx + torch.sin(rot) * vy
            new_vy = -torch.sin(rot) * vx + torch.cos(rot) * vy
            
            # 设置新状态
            new_state = torch.zeros((batch, 7 if feature_dim < 9 else 9), device=state.device)
            new_state[:, 0] = dg.squeeze(1)
            new_state[:, 1] = new_vx.squeeze(1)
            new_state[:, 2] = new_vy.squeeze(1)
            new_state[:, 3] = radius.squeeze(1)
            
            # 处理人类/其他机器人特征（如果存在）
            if feature_dim >= 14:
                # 确保索引有效
                px1 = state[:, 9].unsqueeze(1)
                py1 = state[:, 10].unsqueeze(1)
                vx1 = state[:, 11].unsqueeze(1)
                vy1 = state[:, 12].unsqueeze(1)
                radius1 = state[:, 13].unsqueeze(1)
                
                # 将坐标转换为以agent为中心并旋转
                dx1 = px1 - px
                dy1 = py1 - py
                dx1_rot = torch.cos(rot) * dx1 + torch.sin(rot) * dy1
                dy1_rot = -torch.sin(rot) * dx1 + torch.cos(rot) * dy1
                
                # 旋转速度矢量
                vx1_rot = torch.cos(rot) * vx1 + torch.sin(rot) * vy1
                vy1_rot = -torch.sin(rot) * vx1 + torch.cos(rot) * vy1
                
                # 添加到状态
                new_state = torch.cat([new_state, dx1_rot, dy1_rot, vx1_rot, vy1_rot, radius1], dim=1)
            
            # 添加v_pref和theta（如果存在）
            if feature_dim >= 9:
                new_state[:, 4] = v_pref.squeeze(1)
                new_state[:, 5] = theta.squeeze(1)
                
            # 解释剩余的特征（如果存在）
            remaining_features = feature_dim - (14 if feature_dim >= 14 else feature_dim)
            if remaining_features > 0 and feature_dim > 14:
                # 附加剩余特征（不做任何转换）
                new_state = torch.cat([new_state, state[:, 14:]], dim=1)
            
            return new_state
        
        except Exception as e:
            logging.error(f"Error in rotate: {e}")
            # 返回安全的张量以避免异常中断
            return torch.zeros((1 if len(state.shape) == 1 else state.shape[0], 13), device=self.device)
