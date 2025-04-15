import torch
import torch.nn as nn
import os
from torch.nn.functional import softmax
import logging
import numpy as np
import itertools
import json
from scipy.stats import norm
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.state import HierarchicalState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from llm_decision.llm_decision import LLMDecisionMaker


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans + # of robots, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value

class HierarchicalValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        
        # Human avoidance network
        self.human_net = ValueNetwork(input_dim, self_state_dim, mlp1_dims, 
                                    mlp2_dims, mlp3_dims, attention_dims, 
                                    with_global_state, cell_size, cell_num)
        
        # Robot coordination network
        self.robot_net = ValueNetwork(input_dim, self_state_dim, mlp1_dims,
                                    mlp2_dims, mlp3_dims, attention_dims,
                                    with_global_state, cell_size, cell_num)
        
        # Store attention weights from both networks
        self.attention_weights = None
        self.current_phase = None

    def forward(self, state, training_phase=None):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            
        if training_phase is None:
            # 训练阶段，根据当前状态动态调整权重
            human_value = self.human_net(state)
            robot_value = self.robot_net(state)
            
            # 计算与人类和机器人的距离
            human_dist = torch.min(state[:, 1:, 0:2].norm(dim=2))  # 假设前2维是位置信息
            robot_dist = torch.min(state[:, 1:, 2:4].norm(dim=2))  # 假设3-4维是机器人位置
            
            # 根据距离动态调整权重
            human_weight = torch.sigmoid(1.0 - human_dist/2.0)  # 距离人类越近，权重越大
            robot_weight = torch.sigmoid(1.0 - robot_dist/2.0)  # 距离机器人越近，权重越大
            
            # 归一化权重
            total_weight = human_weight + robot_weight
            human_weight = human_weight / total_weight
            robot_weight = robot_weight / total_weight
            
            self.attention_weights = (self.human_net.attention_weights * human_weight.item() + 
                                    self.robot_net.attention_weights * robot_weight.item())
            
            return human_value * human_weight + robot_value * robot_weight
        else:
            # 测试阶段保持不变
            if training_phase == 'human_avoidance':
                value = self.human_net(state)
                self.attention_weights = self.human_net.attention_weights
                return value
            elif training_phase == 'robot_avoidance':
                value = self.robot_net(state)
                self.attention_weights = self.robot_net.attention_weights
                return value
            else:
                raise ValueError(f"Unknown training phase: {training_phase}")

class LLMEnhancedValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                 attention_dims, with_global_state, cell_size, cell_num):
        super().__init__()
        
        # Human avoidance network
        self.human_net = ValueNetwork(input_dim, self_state_dim, mlp1_dims, 
                                    mlp2_dims, mlp3_dims, attention_dims, 
                                    with_global_state, cell_size, cell_num)
        
        # Robot coordination network
        self.robot_net = ValueNetwork(input_dim, self_state_dim, mlp1_dims,
                                    mlp2_dims, mlp3_dims, attention_dims,
                                    with_global_state, cell_size, cell_num)
        
        # LLM特征处理层
        self.llm_net = nn.Sequential(
            nn.Linear(3, mlp1_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp1_dims[-1], mlp1_dims[-1])
        )
        
        # 存储注意力权重
        self.attention_weights = None
        self.current_phase = None
        
    def forward(self, state, llm_input=None, training_phase=None):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            
        if training_phase is None:
            # 训练阶段，根据当前状态动态调整权重
            human_value = self.human_net(state)
            robot_value = self.robot_net(state)
            
            # 计算与人类和机器人的距离
            human_dist = torch.min(state[:, 1:, 0:2].norm(dim=2))
            robot_dist = torch.min(state[:, 1:, 2:4].norm(dim=2))
            
            # 根据距离动态调整权重
            human_weight = torch.sigmoid(1.0 - human_dist/2.0)
            robot_weight = torch.sigmoid(1.0 - robot_dist/2.0)
            
            # 归一化权重
            total_weight = human_weight + robot_weight
            human_weight = human_weight / total_weight
            robot_weight = robot_weight / total_weight
            
            # 如果有LLM输入，处理LLM特征
            if llm_input is not None:
                try:
                    risk_level = float(llm_input.get('risk_assessment', 5.0)) / 10.0
                    recommended_action = llm_input.get('recommended_action', {'vx': 0, 'vy': 0})
                    
                    llm_features = torch.tensor([
                        risk_level,
                        float(recommended_action.get('vx', 0)),
                        float(recommended_action.get('vy', 0))
                    ], device=state.device)
                    
                    llm_value = self.llm_net(llm_features.unsqueeze(0)).squeeze(0)
                    
                    # 结合LLM特征
                    final_value = (human_value * human_weight + 
                                 robot_value * robot_weight + 
                                 llm_value * risk_level)
                except Exception as e:
                    logging.warning(f"Error in LLM feature processing: {e}")
                    final_value = human_value * human_weight + robot_value * robot_weight
            else:
                final_value = human_value * human_weight + robot_value * robot_weight
            
            self.attention_weights = (self.human_net.attention_weights * human_weight.item() + 
                                    self.robot_net.attention_weights * robot_weight.item())
            
            return final_value
        else:
            # 测试阶段
            if training_phase == 'human_avoidance':
                value = self.human_net(state)
                self.attention_weights = self.human_net.attention_weights
                return value
            elif training_phase == 'robot_avoidance':
                value = self.robot_net(state)
                self.attention_weights = self.robot_net.attention_weights
                return value
            else:
                raise ValueError(f"Unknown training phase: {training_phase}")

    def get_attention_weights(self):
        """返回最近计算的注意力权重"""
        return self.attention_weights if self.attention_weights is not None else None

class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights

class HierarchicalSARL(SARL):
    def __init__(self):
        super().__init__()
        self.name = 'H-SARL'
        self.training_phase = 'human_avoidance'  # Default phase
        self.action_space = None  # Will be initialized in configure

    def set_training_phase(self, phase):
        self.training_phase = phase

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        with_global_state = config.getboolean('sarl', 'with_global_state')
        
        # Initialize action space
        self.initialize_action_space()
        
        self.model = HierarchicalValueNetwork(
            self.input_dim(), self.self_state_dim,
            mlp1_dims, mlp2_dims, mlp3_dims,
            attention_dims, with_global_state, self.cell_size, self.cell_num
        )
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')

    def initialize_action_space(self):
        """Initialize the action space of the agent"""
        holonomic = True  # Set based on your requirements
        kinematic_constrained = False  # Set based on your requirements
        
        speeds = [(i + 1) * 0.25 for i in range(5)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, 5)

        action_space = []
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(rotation, speed))
        self.action_space = action_space


    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        
        if self.phase == 'train' or self.phase == 'val':
            self.last_state = self.transform(state)
            max_value = float('-inf')
            max_action = None
            
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                
                try:
                    if self.query_env:
                        # Create a list of the same action for each robot
                        actions = [action] * (len(self.env.robots) if hasattr(self.env, 'robots') else 1)
                        next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
                    else:
                        next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                        for human_state in state.human_states]
                        # Also propagate other robot states for better prediction
                        next_robot_states = [self.propagate(robot_state, ActionXY(robot_state.vx, robot_state.vy))
                                        for robot_state in state.other_robot_states]
                        reward = self.compute_reward(next_self_state, next_human_states, next_robot_states)
                    
                    # Create a combined state tensor with both humans and robots
                    all_agent_states = next_human_states + next_robot_states if hasattr(self, 'query_env') and not self.query_env else next_human_states
                    
                    batch_next_states = torch.cat([torch.Tensor([next_self_state + agent_state]).to(self.device)
                                            for agent_state in all_agent_states], dim=0)
                    
                    rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                    
                    if self.with_om:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                        rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                    
                    # Use the appropriate training phase
                    value = self.model(rotated_batch_input, self.training_phase).data.item()
                    value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                    
                    if value > max_value:
                        max_value = value
                        max_action = action
                        
                except Exception as e:
                    logging.warning(f"Error in predict: {e}")
                    continue
            
            # Default action if no valid action found
            if max_action is None:
                direction = np.array([state.self_state.gx - state.self_state.px, 
                                    state.self_state.gy - state.self_state.py])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                max_action = ActionXY(direction[0] * self.v_pref * 0.5, direction[1] * self.v_pref * 0.5)
            
            self.last_value = max_value
            return max_action

    def set_phase(self, phase):
        """
        Override set_phase to handle training phase updates
        """
        super().set_phase(phase)
        if hasattr(self, 'model'):
            self.model.training_phase = self.training_phase
    
    def compute_reward(self, next_state, next_human_states):
        # 基础奖励（成功/碰撞/超时）
        reward = 0
        
        # 检查是否到达目标
        if self.env.success:
            reward += 10.0  # 成功奖励
        elif self.env.collision:
            reward -= 2.5   # 碰撞惩罚
        
        # 与人类的距离惩罚
        for human in next_human_states:
            dist = np.linalg.norm([human.px - next_state.px, human.py - next_state.py])
            if dist < 1.0:  # 小于安全距离
                reward -= 0.5 * (1.0 - dist)  # 距离越近，惩罚越大
        
        # 机器人间的距离惩罚
        for robot in self.env.robots:
            if robot != next_state:
                dist = np.linalg.norm([robot.px - next_state.px, robot.py - next_state.py])
                if dist < 2.0:  # 机器人间安全距离
                    reward -= 1.0 * (2.0 - dist)
        
        # 朝向目标的奖励
        goal_dir = np.array([next_state.gx - next_state.px, next_state.gy - next_state.py])
        if np.linalg.norm(goal_dir) > 0:
            goal_dir = goal_dir / np.linalg.norm(goal_dir)
            current_dir = np.array([next_state.vx, next_state.vy])
            if np.linalg.norm(current_dir) > 0:
                current_dir = current_dir / np.linalg.norm(current_dir)
                reward += 0.2 * np.dot(goal_dir, current_dir)  # 朝向目标方向的奖励
                
        return reward

class HierarchicalLLMSARL(HierarchicalSARL):
    def __init__(self):
        super().__init__()
        self.name = 'H-LLM-SARL'
        self.llm_decision_maker = None
        self.llm_enabled = False
        self.use_llm_in_training = False
        self.v_pref = 1.0  # 添加默认值
        
    def configure(self, config):
        # 首先调用父类的configure来设置基本参数
        super().configure(config)
        
        # 获取网络维度参数
        self.mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        self.mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        self.mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        self.attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_global_state = config.getboolean('sarl', 'with_global_state')
        
        # 初始化LLM配置
        self.llm_enabled = config.getboolean('llm', 'enabled', fallback=True)
        print(f"LLM enabled: {self.llm_enabled}")
        api_key = "sk-23eda93c109b4b18acf20cd56b97ac02"
        self.use_llm_in_training = config.getboolean('llm', 'use_in_training', fallback=True)
        print(f"Use LLM in training: {self.use_llm_in_training}")
        
        # 从配置中获取速度参数
        self.v_pref = config.getfloat('robot', 'v_pref', fallback=1.0)

        if self.llm_enabled and api_key:
            # 传入环境引用
            self.llm_decision_maker = LLMDecisionMaker(api_key, env=self.env)
            logging.info('LLM decision maker initialized')
            
        # 使用增强的网络替换基础网络
        self.model = LLMEnhancedValueNetwork(
            self.input_dim(), self.self_state_dim,
            self.mlp1_dims, self.mlp2_dims, self.mlp3_dims,
            self.attention_dims, self.with_global_state,
            self.cell_size, self.cell_num
        )
        
        logging.info('Policy: {} {} global state'.format(
            self.name, 
            'w/' if self.with_global_state else 'w/o'
        ))
        
    def predict(self, state):
        if not self.llm_enabled or (self.phase == 'train' and not self.use_llm_in_training):
            return super().predict(state)
            
        try:
            # 获取LLM建议
            state_desc = self.llm_decision_maker.format_state_for_llm(
                state.self_state,
                state.human_states,
                state.other_robot_state
            )
            llm_advice = self.llm_decision_maker.get_llm_decision(
                state_desc, 
                is_training=(self.phase == 'train')
            )
            
            # 使用增强网络进行预测
            self.last_state = self.transform(state)
            max_value = float('-inf')
            max_action = None
            
            # 记录LLM建议的动作
            llm_suggested_action = ActionXY(
                llm_advice['recommended_action']['vx'],
                llm_advice['recommended_action']['vy']
            )
            
            # 首先评估LLM建议的动作
            value = self._evaluate_action(state, llm_suggested_action, llm_advice)
            if value > max_value:
                max_value = value
                max_action = llm_suggested_action
            
            # 然后评估其他可能的动作
            for action in self.action_space:
                value = self._evaluate_action(state, action, llm_advice)
                if value > max_value:
                    max_value = value
                    max_action = action
                    
        except Exception as e:
            logging.warning(f"Error in LLM prediction: {e}")
            return super().predict(state)
            
        if max_action is None:
            return super().predict(state)
            
        self.last_value = max_value
        return max_action

    def _evaluate_action(self, state, action, llm_advice):
        """评估单个动作的值"""
        next_self_state = self.propagate(state.self_state, action)
        actions = [action] * len(self.env.robots)
        
        if self.query_env:
            next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
        else:
            next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                            for human_state in state.human_states]
            reward = self.compute_reward(next_self_state, next_human_states)
        
        batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                    for next_human_state in next_human_states], dim=0)
        rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
        
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
            rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
        
        value = self.model(rotated_batch_input, llm_advice, self.training_phase).data.item()
        return reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value