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
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from llm_decision.llm_decision import LLMDecisionMaker


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                 attention_dims, with_global_state, cell_size, cell_num, robot_num):
        super().__init__()
        self.robot_num = robot_num
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        self.attention = mlp(mlp1_dims[-1] * 2, attention_dims + [1], last_relu=False)
        
        # 计算mlp3的输入维度
        if with_global_state:
            mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        else:
            mlp3_input_dim = mlp2_dims[-1]
            
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        
        # 打印网络维度信息
        # print(f"Network dimensions:")
        # print(f"Input dim: {input_dim}")
        # print(f"Self state dim: {self_state_dim}")
        # print(f"MLP1 dims: {mlp1_dims}")
        # print(f"MLP2 dims: {mlp2_dims}")
        # print(f"MLP3 dims: {mlp3_dims}")
        # print(f"MLP3 input dim: {mlp3_input_dim}")
        
        self.attention_weights = None

    def forward(self, state, robot_id=None):
        """
        state: (batch_size, # of humans, state_dim)
        """
        if robot_id is None:
            robot_id = 0
            
        size = state.shape
        #print(f"Input state shape: {size}")
        
        self_state = state[:, 0, :self.self_state_dim]
        #print(f"Self state shape: {self_state.shape}")
        
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        #print(f"MLP1 output shape: {mlp1_output.shape}")
        
        mlp2_output = self.mlp2(mlp1_output)
        #print(f"MLP2 output shape: {mlp2_output.shape}")

        if self.with_global_state:
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            #print(f"Global state shape: {global_state.shape}")
            
            global_state_repeated = global_state.expand((size[0], size[1], self.global_state_dim))
            attention_input = torch.cat([mlp1_output.view(size[0], size[1], -1), global_state_repeated], dim=2)
            #print(f"Attention input shape: {attention_input.shape}")
            
            attention_weights = self.attention(attention_input.view((-1, self.global_state_dim * 2)))
            attention_weights = attention_weights.view(size[0], size[1], 1)
            #print(f"Attention weights shape: {attention_weights.shape}")
            
            self.attention_weights = attention_weights[0, 1:, 0].data.cpu().numpy()

            features = mlp2_output.view(size[0], size[1], -1)
            #print(f"Features shape: {features.shape}")
            
            weighted_feature = torch.sum(features * attention_weights, 1)
            #print(f"Weighted feature shape: {weighted_feature.shape}")
            
            joint_state = torch.cat([self_state, weighted_feature], dim=1)
            #print(f"Joint state shape: {joint_state.shape}")
            
            value = self.mlp3(joint_state)
            #print(f"Final value shape: {value.shape}")
        else:
            features = mlp2_output.view(size[0], size[1], -1)
            attention_weights = torch.ones((size[0], size[1], 1), device=state.device) / size[1]
            self.attention_weights = attention_weights[0, 1:, 0].data.cpu().numpy()
            weighted_feature = torch.sum(features * attention_weights, 1)
            value = self.mlp3(weighted_feature)

        return value

class HierarchicalValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                 attention_dims, with_global_state, cell_size, cell_num, robot_num):
        super().__init__()
        self.robot_num = robot_num
        
        # 为每个机器人创建独立的Human avoidance network
        self.human_nets = nn.ModuleList([
            ValueNetwork(input_dim, self_state_dim, mlp1_dims, 
                        mlp2_dims, mlp3_dims, attention_dims, 
                        with_global_state, cell_size, cell_num, robot_num)
            for _ in range(robot_num)
        ])
        
        # 为每个机器人创建独立的Robot coordination network
        self.robot_nets = nn.ModuleList([
            ValueNetwork(input_dim, self_state_dim, mlp1_dims,
                        mlp2_dims, mlp3_dims, attention_dims,
                        with_global_state, cell_size, cell_num, robot_num)
            for _ in range(robot_num)
        ])
        
        # 存储每个机器人的注意力权重
        self.attention_weights = [None] * robot_num
        self.current_phase = None

    def forward(self, state, training_phase=None, robot_id=None):
        """
        state: (batch_size, # of humans, state_dim)
        """
        if robot_id is None:
            robot_id = 0
            
        if training_phase is None or training_phase == 'human_avoidance':
            value = self.human_nets[robot_id](state)
            self.attention_weights = self.human_nets[robot_id].attention_weights
        else:  # robot_coordination phase
            value = self.robot_nets[robot_id](state)
            self.attention_weights = self.robot_nets[robot_id].attention_weights
            
        return value

class LLMEnhancedValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                 attention_dims, with_global_state, cell_size, cell_num, robot_num):
        super().__init__()
        
        # 保存机器人数量
        self.robot_num = robot_num
        
        # 为每个机器人创建独立的Human avoidance network
        self.human_nets = nn.ModuleList([
            ValueNetwork(input_dim, self_state_dim, mlp1_dims, 
                        mlp2_dims, mlp3_dims, attention_dims, 
                        with_global_state, cell_size, cell_num, robot_num)
            for _ in range(robot_num)
        ])
        
        # 为每个机器人创建独立的Robot coordination network
        self.robot_nets = nn.ModuleList([
            ValueNetwork(input_dim, self_state_dim, mlp1_dims,
                        mlp2_dims, mlp3_dims, attention_dims,
                        with_global_state, cell_size, cell_num, robot_num)
            for _ in range(robot_num)
        ])
        
        # 为每个机器人创建独立的LLM特征处理层
        self.llm_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, mlp1_dims[-1]),
                nn.ReLU(),
                nn.Linear(mlp1_dims[-1], mlp1_dims[-1])
            )
            for _ in range(robot_num)
        ])
        
        # 存储每个机器人的注意力权重
        self.attention_weights = [None] * robot_num
        self.current_phase = None

    def forward(self, state, llm_input=None, training_phase=None, robot_id=None):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            
        if robot_id is None:
            robot_id = 0
            
        if training_phase is None:
            # 获取对应机器人的网络
            human_net = self.human_nets[robot_id]
            robot_net = self.robot_nets[robot_id]
            llm_net = self.llm_nets[robot_id]
            
            # 计算值
            human_value = human_net(state, robot_id=robot_id)
            robot_value = robot_net(state, robot_id=robot_id)
            
            # 安全地计算距离和权重
            if state.shape[1] > 1:  # 如果有其他智能体
                human_dist = torch.min(state[:, 1:, 0:2].norm(dim=2))
                robot_dist = torch.min(state[:, 1:, 2:4].norm(dim=2))
            else:  # 如果没有其他智能体
                human_dist = torch.tensor(float('inf'), device=state.device)
                robot_dist = torch.tensor(float('inf'), device=state.device)
            
            # 计算权重
            human_weight = torch.sigmoid(1.0 - human_dist/2.0)
            robot_weight = torch.sigmoid(1.0 - robot_dist/2.0)
            
            # 避免除零
            total_weight = human_weight + robot_weight
            if total_weight == 0:
                human_weight = robot_weight = 0.5
            else:
                human_weight = human_weight / total_weight
                robot_weight = robot_weight / total_weight
            
            # 处理LLM输入
            if llm_input is not None and isinstance(llm_input, dict):
                try:
                    # 获取风险评估
                    risk_level = 5.0  # 默认值
                    if 'risk_assessment' in llm_input:
                        risk_level = float(llm_input['risk_assessment'])
                    risk_level = risk_level / 10.0  # 归一化到 [0,1]

                    # 获取推荐动作
                    vx = vy = 0.0  # 默认值
                    if 'robots_decisions' in llm_input and isinstance(llm_input['robots_decisions'], list):
                        for decision in llm_input['robots_decisions']:
                            if isinstance(decision, dict) and 'robot_id' in decision:
                                if int(decision['robot_id']) == robot_id:
                                    if 'recommended_action' in decision:
                                        action = decision['recommended_action']
                                        if isinstance(action, dict):
                                            vx = float(action.get('vx', 0.0))
                                            vy = float(action.get('vy', 0.0))
                                    break
                    
                    # 创建LLM特征张量
                    llm_features = torch.tensor([
                        risk_level, vx, vy
                    ], device=state.device).unsqueeze(0)
                    
                    # 计算LLM值
                    llm_value = llm_net(llm_features)
                    
                    # 组合所有值
                    final_value = (
                        human_weight * human_value + 
                        robot_weight * robot_value + 
                        0.2 * llm_value  # LLM影响权重
                    )
                except Exception as e:
                    logging.warning(f"Error processing LLM input: {e}")
                    final_value = human_weight * human_value + robot_weight * robot_value
            else:
                final_value = human_weight * human_value + robot_weight * robot_value
                
            return final_value
        else:
            if training_phase == 'human_avoidance':
                value = self.human_nets[robot_id](state, robot_id=robot_id)
                self.attention_weights[robot_id] = self.human_nets[robot_id].attention_weights
            else:  # robot_coordination phase
                value = self.robot_nets[robot_id](state, robot_id=robot_id)
                self.attention_weights[robot_id] = self.robot_nets[robot_id].attention_weights
            return value

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
        
        # 获取机器人数量
        robot_num = config.getint('sim', 'robot_num')
        
        self.model = HierarchicalValueNetwork(
            self.input_dim(), self.self_state_dim, 
            mlp1_dims, mlp2_dims, mlp3_dims,
            attention_dims, with_global_state, 
            self.cell_size, self.cell_num, robot_num
        )
        
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
        self.with_global_state = config.getboolean('sarl', 'with_global_state')
        
        # 获取机器人数量
        robot_num = config.getint('sim', 'robot_num')
        
        self.model = HierarchicalValueNetwork(
            self.input_dim(), self.self_state_dim,
            mlp1_dims, mlp2_dims, mlp3_dims,
            attention_dims, self.with_global_state,
            self.cell_size, self.cell_num, robot_num
        )

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
        return action_space


    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        
        if self.phase == 'train' or self.phase == 'val':
            self.last_state = self.transform(state)
            max_value = float('-inf')
            max_action = None
            
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                # 创建与机器人数量相同的动作列表
                actions = [action] * len(self.env.robots)
                
                try:
                    if self.query_env:
                        next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
                    else:
                        next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                        for human_state in state.human_states]
                        reward = self.compute_reward(next_self_state, next_human_states)
                    
                    # 使用父类的状态转换方法
                    batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                                for next_human_state in next_human_states], dim=0)
                    rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                    
                    if self.with_om:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                        rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                    
                    # 根据训练阶段选择不同的网络
                    value = self.model(rotated_batch_input, self.training_phase).data.item()
                    value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                    
                    if value > max_value:
                        max_value = value
                        max_action = action
                        
                except Exception as e:
                    logging.warning(f"Error in predict: {e}")
                    continue
            
            # 如果没有找到有效动作，返回安全的默认动作
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
        
        # 获取机器人数量
        self.robot_num = config.getint('sim', 'robot_num')
        self.human_num = config.getint('sim', 'human_num')

        # 初始化动作空间
        self.initialize_action_space()
        
        # 初始化LLM配置
        self.llm_enabled = config.getboolean('llm', 'enabled', fallback=True)
        print(f"LLM enabled: {self.llm_enabled}")

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
            self.cell_size, self.cell_num, self.robot_num
        )
        
        logging.info('Policy: {} {} global state'.format(
            self.name, 
            'w/' if self.with_global_state else 'w/o'
        ))
        
    def predict(self, state):
        if not self.llm_enabled or (self.phase == 'train' and not self.use_llm_in_training):
            return super().predict(state)
                
        try:
            # 确保 action_space 已初始化
            if self.action_space is None:
                self.initialize_action_space()
                
            # 获取当前机器人ID
            current_robot_id = state.self_state.robot_id if hasattr(state.self_state, 'robot_id') else 0
                
            # 创建状态描述
            state_desc = {
                "robot_num": self.robot_num,
                "human_num": self.human_num,
                "current_robot": {
                    "id": str(current_robot_id),
                    "position": {
                        "x": str(float(state.self_state.px)),
                        "y": str(float(state.self_state.py))
                    },
                    "velocity": {
                        "x": str(float(state.self_state.vx)),
                        "y": str(float(state.self_state.vy))
                    },
                    "goal": {
                        "x": str(float(state.self_state.gx)),
                        "y": str(float(state.self_state.gy))
                    },
                    "radius": str(float(state.self_state.radius))
                }
            }

            # 添加所有人类状态
            state_desc["humans"] = []
            for i in range(self.human_num):
                if i < len(state.human_states):
                    human = state.human_states[i]
                    state_desc["humans"].append({
                        "id": str(i),
                        "position": {
                            "x": str(float(human.px)),
                            "y": str(float(human.py))
                        },
                        "velocity": {
                            "x": str(float(human.vx)),
                            "y": str(float(human.vy))
                        },
                        "radius": str(float(human.radius))
                    })
                else:
                    # 如果实际人类数量不足，添加默认值
                    state_desc["humans"].append(self._get_default_human_state(i))
            
            # 添加所有机器人状态
            state_desc["other_robots"] = []
            for i in range(self.robot_num):
                # if i == current_robot_id:
                    # 当前机器人的信息已经在 current_robot 中
                    #state_desc["all_robots"].append(state_desc["current_robot"])
                if hasattr(state, 'other_robots_states') and i < len(state.other_robots_states):
                    robot = state.other_robots_states[i]
                    state_desc["other_robots"].append({
                        "id": str(i),
                        "position": {
                            "x": str(float(robot.px)),
                            "y": str(float(robot.py))
                        },
                        "velocity": {
                            "x": str(float(robot.vx)),
                            "y": str(float(robot.vy))
                        },
                        "goal": {
                            "x": str(float(robot.gx)),
                            "y": str(float(robot.gy))
                        },
                        "radius": str(float(robot.radius))
                    })
                else:
                    # 如果实际机器人数量不足，添加默认值
                    state_desc["other_robots"].append(self._get_default_robot_state(i))
                    logging.debug(f"Created default robot state: {state_desc['other_robots'][-1]}")

            # 获取LLM决策
            llm_decision = self.llm_decision_maker.get_llm_decision(state_desc, self.phase == 'train')
            if llm_decision is None:
                return super().predict(state)
                
            # 评估所有可能的动作
            max_value = float('-inf')
            best_action = None
            
            for action in self.action_space:
                value = self._evaluate_action(state, action, llm_decision, current_robot_id)
                if value > max_value:
                    max_value = value
                    best_action = action
                    
            # 如果没有找到有效动作，使用默认动作
            if best_action is None:
                direction = np.array([state.self_state.gx - state.self_state.px, 
                                    state.self_state.gy - state.self_state.py])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                best_action = ActionXY(direction[0] * self.v_pref * 0.5, direction[1] * self.v_pref * 0.5)
                
            return best_action
            
        except Exception as e:
            logging.warning(f"Error in predict: {e}")
            # 发生错误时使用父类的预测方法
            return super().predict(state)

    def _get_default_human_state(self, human_id):
        """返回默认的人类状态"""
        return {
            "id": str(human_id),
            "position": {"x": "0.0", "y": "0.0"},
            "velocity": {"x": "0.0", "y": "0.0"},
            "radius": "0.3"
        }

    def _get_default_robot_state(self, robot_id):
        """返回默认的机器人状态"""
        return {
            "id": str(robot_id),
            "position": {"x": "0.0", "y": "0.0"},
            "velocity": {"x": "0.0", "y": "0.0"},
            "goal": {"x": "0.0", "y": "0.0"},
            "radius": "0.3"
        }

    def _evaluate_action(self, state, action, llm_advice, robot_id):
        """评估单个动作的价值"""
        try:
            # 生成下一个状态
            next_self_state = self.propagate(state.self_state, action)
            
            # 获取人类状态
            if hasattr(state, 'human_states') and state.human_states:
                next_human_states = []
                for human_state in state.human_states:
                    try:
                        human_action = ActionXY(human_state.vx, human_state.vy)
                        next_human_state = self.propagate(human_state, human_action)
                        next_human_states.append(next_human_state)
                    except Exception as e:
                        logging.debug(f"Error propagating human state: {e}")
                        continue
            else:
                # 如果没有人类状态，创建一个虚拟状态
                next_human_states = [self.get_dummy_human_state()]
                logging.debug(f"Created dummy human state: {next_human_states[0]}")
                
            # 计算奖励
            try:
                reward = self.compute_reward(next_self_state, next_human_states)
            except Exception as e:
                logging.debug(f"Error computing reward: {e}")
                reward = -1.0  # 默认奖励值
                
            # 构建状态张量
            try:
                batch_next_states = []
                for next_human_state in next_human_states:
                    combined_state = self.combine_states(next_self_state, next_human_state)
                    if combined_state is not None:
                        batch_next_states.append(combined_state)
                        
                if not batch_next_states:
                    return float('-inf')
                    
                batch_tensor = torch.stack(batch_next_states).to(self.device)
                rotated_batch_input = self.rotate(batch_tensor).unsqueeze(0)
                
                # 确保llm_advice是有效的字典
                if not isinstance(llm_advice, dict):
                    llm_advice = self._get_default_llm_advice()
                    
                # 计算值
                value = self.model(rotated_batch_input, llm_advice, self.training_phase, robot_id)
                if isinstance(value, torch.Tensor):
                    value = value.item()
                    
                return reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                
            except Exception as e:
                logging.debug(f"Error in state evaluation: {e}")
                return float('-inf')
                
        except Exception as e:
            logging.warning(f"Error in action evaluation: {e}")
            return float('-inf')

    def get_dummy_human_state(self):
        """创建一个虚拟的人类状态用于评估"""
        from crowd_sim.envs.utils.state import FullState
        return FullState(0, 0, 0, 0, 0, 0, 0)

    def combine_states(self, self_state, human_state):
        """安全地组合机器人和人类状态"""
        try:
            combined = []
            # 添加机器人状态
            combined.extend([
                self_state.px, self_state.py,
                self_state.vx, self_state.vy,
                self_state.radius,
                self_state.gx, self_state.gy,
                self_state.v_pref,
                self_state.theta
            ])
            # 添加人类状态
            combined.extend([
                human_state.px, human_state.py,
                human_state.vx, human_state.vy,
                human_state.radius,
                0, 0,  # 人类目标位置（如果没有）
                0,     # 人类首选速度（如果没有）
                0      # 人类朝向（如果没有）
            ])
            return torch.tensor(combined, device=self.device)
        except Exception as e:
            logging.debug(f"Error combining states: {e}")
            return None
            
    def _get_default_llm_advice(self):
        """返回默认的LLM建议"""
        return {
            "risk_assessment": "5.0",
            "robots_decisions": [
                {
                    "robot_id": "0",
                    "recommended_action": {
                        "vx": "0.0",
                        "vy": "0.0"
                    }
                }
            ]
        }