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
from crowd_sim.envs.utils.action import ActionXY, ActionRot, limit_speed
from llm_decision.llm_decision import LLMDecisionMaker
import inspect


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
        try:
            # Ensure state has expected dimensions
            if len(state.shape) == 2:
                state = state.unsqueeze(0)  # Add batch dimension
            
            size = state.shape
            
            # Check for empty input
            if size[0] == 0 or size[1] == 0:
                logging.warning("Empty state tensor in forward pass")
                return torch.zeros(1, 1, device=state.device)
            
            # Check if state dimensions match network expectations
            if size[2] != self.mlp1[0].weight.size(1):
                logging.warning(f"Input dimension mismatch: mlp1 expects {self.mlp1[0].weight.size(1)}, but got {size[2]}")
                
                # Adjust input dimensions to match weights
                if size[2] < self.mlp1[0].weight.size(1):
                    # Pad with zeros 
                    padding = torch.zeros(size[0], size[1], self.mlp1[0].weight.size(1) - size[2], device=state.device)
                    state = torch.cat([state, padding], dim=2)
                else:
                    # Truncate
                    state = state[:, :, :self.mlp1[0].weight.size(1)]
                
                # Update size after adjustment
                size = state.shape
                logging.info(f"Adjusted state tensor shape to: {size}")
            
            # Extract self state
            if size[1] > 0 and size[2] >= self.self_state_dim:
                self_state = state[:, 0, :self.self_state_dim]
            else:
                # Create a dummy self state if dimensions are incorrect
                logging.warning(f"Invalid state dimensions: {size}, creating dummy self_state")
                self_state = torch.zeros((size[0], self.self_state_dim), device=state.device)
            
            # Reshape and pass through mlp1
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

            # masked softmax with safe handling
            scores_exp = torch.exp(scores) * (scores != 0).float()
            sum_scores = torch.sum(scores_exp, dim=1, keepdim=True).clamp(min=1e-10)
            weights = (scores_exp / sum_scores).unsqueeze(2)
            
            # Save attention weights for visualization
            if size[0] > 0:
                self.attention_weights = weights[0, :, 0].detach().cpu().numpy()
            
            # output feature is a linear combination of input features
            features = mlp2_output.view(size[0], size[1], -1)
            
            # Debug dimensions
            logging.debug(f"Weights shape: {weights.shape}, Features shape: {features.shape}")
            
            # Ensure dimensions match for multiplication
            if weights.shape[1] != features.shape[1]:
                logging.warning(f"Dimension mismatch: weights {weights.shape}, features {features.shape}")
                
                if weights.shape[1] < features.shape[1]:
                    # Pad weights
                    padding = torch.zeros((weights.shape[0], features.shape[1] - weights.shape[1], weights.shape[2]), 
                                         device=weights.device)
                    weights = torch.cat([weights, padding], dim=1)
                else:
                    # Truncate weights or expand features
                    if features.shape[1] == 0:
                        # Special case: no features, create dummy
                        features = torch.zeros((features.shape[0], 1, features.shape[2]), device=features.device)
                        weights = weights[:, :1, :]
                    else:
                        weights = weights[:, :features.shape[1], :]
            
            # Multiply and sum with dimension check
            try:
                weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
            except RuntimeError as e:
                logging.error(f"Error in weighted sum: {e}")
                # Fallback: use mean of features
                weighted_feature = torch.mean(features, dim=1)
            
            # concatenate agent's state with global weighted humans' state
            joint_state = torch.cat([self_state, weighted_feature], dim=1)
            value = self.mlp3(joint_state)
            return value
            
        except Exception as e:
            logging.error(f"Error in ValueNetwork forward pass: {e}")
            # Return a dummy value as fallback
            return torch.tensor([[0.0]], device=state.device)

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
        try:
            if len(state.shape) == 2:
                state = state.unsqueeze(0)
                
            # Handle empty input
            if state.shape[0] == 0 or state.shape[1] == 0:
                logging.warning("Empty state tensor in hierarchical forward pass")
                return torch.zeros(1, 1, device=state.device)
                
            if training_phase is None:
                # 训练阶段，根据当前状态动态调整权重
                try:
                    human_value = self.human_net(state)
                except Exception as e:
                    logging.warning(f"Error in human net: {e}")
                    human_value = torch.zeros(1, 1, device=state.device)
                    
                try:
                    robot_value = self.robot_net(state)
                except Exception as e:
                    logging.warning(f"Error in robot net: {e}")
                    robot_value = human_value  # Fallback to human value
                
                # 计算与人类和机器人的距离
                try:
                    # Safely access position info - handle different tensor shapes
                    if state.shape[1] > 1:
                        if state.shape[2] >= 4:  # Ensure we have at least 4 dimensions
                            human_positions = state[:, 1:, 0:2]
                            robot_positions = state[:, 1:, 2:4]
                            
                            # Compute minimum distances safely
                            human_dist = torch.min(human_positions.norm(dim=2) + 1e-5)  # Add small constant to avoid div by zero
                            robot_dist = torch.min(robot_positions.norm(dim=2) + 1e-5)
                        else:
                            # Not enough features, use default values
                            human_dist = torch.tensor(1.0, device=state.device)
                            robot_dist = torch.tensor(1.0, device=state.device)
                    else:
                        # Not enough agents, use default values
                        human_dist = torch.tensor(1.0, device=state.device)
                        robot_dist = torch.tensor(1.0, device=state.device)
                except Exception as e:
                    logging.warning(f"Error computing distances: {e}")
                    human_dist = torch.tensor(1.0, device=state.device)
                    robot_dist = torch.tensor(1.0, device=state.device)
                
                # 根据距离动态调整权重
                human_weight = torch.sigmoid(1.0 - human_dist/2.0)  # 距离人类越近，权重越大
                robot_weight = torch.sigmoid(1.0 - robot_dist/2.0)  # 距离机器人越近，权重越大
                
                # 归一化权重 - 防止除零
                total_weight = human_weight + robot_weight
                if total_weight > 0:
                    human_weight = human_weight / total_weight
                    robot_weight = robot_weight / total_weight
                else:
                    human_weight = torch.tensor(0.5, device=state.device)
                    robot_weight = torch.tensor(0.5, device=state.device)
                
                # Save attention weights
                try:
                    self.attention_weights = (self.human_net.attention_weights * human_weight.item() + 
                                           self.robot_net.attention_weights * robot_weight.item())
                except Exception as e:
                    logging.warning(f"Error computing attention weights: {e}")
                    if hasattr(self.human_net, 'attention_weights') and self.human_net.attention_weights is not None:
                        self.attention_weights = self.human_net.attention_weights
                    elif hasattr(self.robot_net, 'attention_weights') and self.robot_net.attention_weights is not None:
                        self.attention_weights = self.robot_net.attention_weights
                    else:
                        self.attention_weights = None
                
                # Final weighted value
                return human_value * human_weight + robot_value * robot_weight
            else:
                # 测试阶段保持不变
                if training_phase == 'human_avoidance':
                    try:
                        value = self.human_net(state)
                        self.attention_weights = self.human_net.attention_weights
                        return value
                    except Exception as e:
                        logging.error(f"Error in human_avoidance phase: {e}")
                        return torch.zeros(1, 1, device=state.device)
                elif training_phase == 'robot_avoidance':
                    try:
                        value = self.robot_net(state)
                        self.attention_weights = self.robot_net.attention_weights
                        return value
                    except Exception as e:
                        logging.error(f"Error in robot_avoidance phase: {e}")
                        return torch.zeros(1, 1, device=state.device)
                else:
                    logging.error(f"Unknown training phase: {training_phase}")
                    # Default to human avoidance as fallback
                    try:
                        return self.human_net(state)
                    except Exception:
                        return torch.zeros(1, 1, device=state.device)
        except Exception as e:
            logging.error(f"Error in hierarchical forward: {e}")
            return torch.zeros(1, 1, device=state.device)

class LLMValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                 attention_dims, with_global_state, cell_size, cell_num):
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
        # LLM特征处理层
        self.llm_net = nn.Sequential(
            nn.Linear(3, mlp1_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp1_dims[-1], 1) # 输出维度为1
        )
        
        # 存储注意力权重
        self.attention_weights = None
        self.current_phase = None

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    def init_attention_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, llm_input=None):
        try:
            # Ensure state has expected dimensions
            if len(state.shape) == 2:
                state = state.unsqueeze(0)  # Add batch dimension
            
            size = state.shape
            
            # Check for empty input
            if size[0] == 0 or size[1] == 0:
                logging.warning("Empty state tensor in forward pass")
                return torch.zeros(1, 1, device=state.device)
            
            # Check if state dimensions match network expectations
            if size[2] != self.mlp1[0].weight.size(1):
                logging.warning(f"Input dimension mismatch: mlp1 expects {self.mlp1[0].weight.size(1)}, but got {size[2]}")
                
                # Adjust input dimensions to match weights
                if size[2] < self.mlp1[0].weight.size(1):
                    # Pad with zeros 
                    padding = torch.zeros(size[0], size[1], self.mlp1[0].weight.size(1) - size[2], device=state.device)
                    state = torch.cat([state, padding], dim=2)
                else:
                    # Truncate
                    state = state[:, :, :self.mlp1[0].weight.size(1)]
                
                # Update size after adjustment
                size = state.shape
                logging.info(f"Adjusted state tensor shape to: {size}")
            
            # Extract self state
            if size[1] > 0 and size[2] >= self.self_state_dim:
                self_state = state[:, 0, :self.self_state_dim]
            else:
                # Create a dummy self state if dimensions are incorrect
                logging.warning(f"Invalid state dimensions: {size}, creating dummy self_state")
                self_state = torch.zeros((size[0], self.self_state_dim), device=state.device)
            # 重新初始化权重（如果尚未初始化）
            if not hasattr(self, '_weights_initialized'):
                init_weights = LLMValueNetwork.init_weights
                self.mlp1.apply(init_weights)
                self.mlp2.apply(init_weights)
                self.mlp3.apply(init_weights)
                self._weights_initialized = True
            # Reshape and pass through mlp1
            mlp1_input = state.view((-1, size[2]))
            mlp1_output = self.mlp1(mlp1_input)
            mlp2_output = self.mlp2(mlp1_output)
            # print("mlp1_input:",mlp1_input)
            # print("mlp2_output:",mlp2_output)
            if self.with_global_state:
                # compute attention scores
                global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
                global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                    contiguous().view(-1, self.global_state_dim)
                attention_input = torch.cat([mlp1_output, global_state], dim=1)
            else:
                attention_input = mlp1_output
            # for name, param in self.attention.named_parameters():
            #     print(f"{name}: mean={param.data.mean()}, std={param.data.std()}, NaN={torch.isnan(param.data).any()}")    
            # 重新初始化注意力层权重
            if not hasattr(self, '_attention_weights_initialized'):
                init_attention_weights = LLMValueNetwork.init_attention_weights
            self.attention.apply(init_attention_weights)  # 初始化所有 Linear 层
            scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
 
            # masked softmax with safe handling
            scores_exp = torch.exp(scores) * (scores != 0).float()
            sum_scores = torch.sum(scores_exp, dim=1, keepdim=True).clamp(min=1e-10)
            weights = (scores_exp / sum_scores).unsqueeze(2)

            # Save attention weights for visualization
            if size[0] > 0:
                self.attention_weights = weights[0, :, 0].detach().cpu().numpy()
            
            # output feature is a linear combination of input features
            features = mlp2_output.view(size[0], size[1], -1)
            
            # Debug dimensions
            logging.debug(f"Weights shape: {weights.shape}, Features shape: {features.shape}")
            
            # Ensure dimensions match for multiplication
            if weights.shape[1] != features.shape[1]:
                logging.warning(f"Dimension mismatch: weights {weights.shape}, features {features.shape}")
                
                if weights.shape[1] < features.shape[1]:
                    # Pad weights
                    padding = torch.zeros((weights.shape[0], features.shape[1] - weights.shape[1], weights.shape[2]), 
                                         device=weights.device)
                    weights = torch.cat([weights, padding], dim=1)
                else:
                    # Truncate weights or expand features
                    if features.shape[1] == 0:
                        # Special case: no features, create dummy
                        features = torch.zeros((features.shape[0], 1, features.shape[2]), device=features.device)
                        weights = weights[:, :1, :]
                    else:
                        weights = weights[:, :features.shape[1], :]

            # print("weights2:",weights)
            # print("features:",features)
            # Multiply and sum with dimension check
            try:
                weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
            except RuntimeError as e:
                logging.error(f"Error in weighted sum: {e}")
                # Fallback: use mean of features
                weighted_feature = torch.mean(features, dim=1)
            # print("self_state:",self_state)
            # print("weighted_feature:",weighted_feature)
            # concatenate agent's state with global weighted humans' state
            joint_state = torch.cat([self_state, weighted_feature], dim=1)
            # print("Input joint_state:", joint_state)
            # print("joint_state shape:", joint_state.shape)
            # print("Has NaN:", torch.isnan(joint_state).any())
            # print("Has Inf:", torch.isinf(joint_state).any())
            value = self.mlp3(joint_state)
            # print("value1:",value.item())
            # 如果有LLM输入，处理LLM特征
            if llm_input is not None:
                try:
                    # print("risk_level:",llm_input[0,2].item())
                    # risk_level = float(llm_input.get('risk_assessment', 5.0)) / 10.0
                    # recommended_action = llm_input.get('recommended_action', {'vx': 0, 'vy': 0})
                    
                    # llm_features = torch.tensor([
                    #     risk_level,
                    #     float(recommended_action.get('vx', 0)),
                    #     float(recommended_action.get('vy', 0))
                    # ], device=state.device)
                    
                    llm_value = self.llm_net(llm_input.unsqueeze(0)).squeeze(0)
                    # print("llm_value:",llm_value.item())
                    # 结合LLM特征
                    value = (value + llm_value * llm_input[0,2].item())

                except Exception as e:
                    logging.warning(f"Error in LLM feature processing: {e}")
            return value
            
        except Exception as e:
            logging.error(f"Error in ValueNetwork forward pass: {e}")
            # Return a dummy value as fallback
            return torch.tensor([[0.0]], device=state.device)
        # size = state.shape
        # self_state = state[:, 0, :self.self_state_dim]
        # mlp1_output = self.mlp1(state.view((-1, size[2])))
        # mlp2_output = self.mlp2(mlp1_output)

        # if self.with_global_state:
        #     # compute attention scores
        #     global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        #     global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
        #         contiguous().view(-1, self.global_state_dim)
        #     attention_input = torch.cat([mlp1_output, global_state], dim=1)
        # else:
        #     attention_input = mlp1_output
        # scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # # masked softmax
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        # self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # # output feature is a linear combination of input features
        # features = mlp2_output.view(size[0], size[1], -1)
        # weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # # concatenate agent's state with global weighted humans' state
        # joint_state = torch.cat([self_state, weighted_feature], dim=1)
        # value = self.mlp3(joint_state)
        # print("value1:",value)

        
        
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
            nn.Linear(mlp1_dims[-1], 1)
        )
        
        # 存储注意力权重
        self.attention_weights = None
        self.current_phase = None
        
    def forward(self, state, llm_input=None, training_phase=None):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        llm_value = 0
        if llm_input is not None:# 如果有LLM输入，处理LLM特征
            try:           
                llm_value = self.llm_net(llm_input.unsqueeze(0)).squeeze(0)  
            except Exception as e:
                logging.warning(f"Error in LLM feature processing: {e}")  
        else:
            logging.warning("no llm input")

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
            
            # 结合LLM特征
            final_value = (human_value * human_weight + 
                            robot_value * robot_weight + 
                            llm_value * llm_input[0,2].item())
    
            self.attention_weights = (self.human_net.attention_weights * human_weight.item() + 
                                    self.robot_net.attention_weights * robot_weight.item())
            
            return final_value
        else:
            # 测试阶段
            print("training phase:",training_phase)
            if training_phase == 'human_avoidance':
                value = self.human_net(state)+llm_value * llm_input[0,2].item()
                self.attention_weights = self.human_net.attention_weights
                return value
            elif training_phase == 'robot_avoidance':
                value = self.robot_net(state)+llm_value * llm_input[0,2].item()
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
        """
        预测方法关键修复部分 - 确保训练阶段正确获取奖励和处理不同维度的状态
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        
        # 检查状态有效性
        if state is None:
            logging.warning("State is None in predict method")
            if self.kinematics == 'holonomic':
                return ActionXY(0.3, 0)
            else:
                return ActionRot(0, 0.3)
        
        # 到达目标检查
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        
        # 确保动作空间已初始化
        if self.action_space is None:
            try:
                self.build_action_space(state.self_state.v_pref)
            except Exception as e:
                logging.error(f"Error building action space: {e}")
                return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        
        # 保存当前状态用于后续处理 - 使用安全的transform方法
        try:
            self.last_state = self.transform(state)
        except Exception as e:
            logging.error(f"Error transforming state: {e}")
            # 继续执行，因为我们可能仍然能够找到有效的动作
        
        max_value = float('-inf')
        max_action = None
        
        # 随机探索 - 确保探索空间
        if self.phase == 'train' and np.random.random() < 0.15:  # 15% 随机探索
            logging.debug("Using random exploration")
            return self.action_space[np.random.randint(len(self.action_space))]
        
        # 评估所有动作，选择最优的
        for action in self.action_space:
            try:
                # 1. 传播状态
                next_self_state = self.propagate(state.self_state, action)
                next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                   for human_state in state.human_states]
                
                # 2. 处理其他机器人状态
                next_robot_states = []
                if hasattr(state, 'other_robot_states') and state.other_robot_states:
                    robot_states = state.other_robot_states
                    if not isinstance(robot_states, list):
                        robot_states = [robot_states]
                        
                    next_robot_states = [self.propagate(robot_state, ActionXY(robot_state.vx, robot_state.vy))
                                      for robot_state in robot_states if robot_state is not None]
                
                # 3. 获取奖励 - 修复奖励计算
                reward = self.compute_reward(next_self_state, next_human_states, next_robot_states)
                
                # 4. 构建下一个状态对象用于评估
                from crowd_sim.envs.utils.state import JointState
                # 确保正确传递human_states参数
                if not next_human_states:
                    next_human_states = []  # 确保为空列表而不是None
                
                # 修复JointState初始化，正确顺序传递参数：self_state, other_robot_states, human_states
                next_state = JointState(next_self_state, next_robot_states if next_robot_states else [], next_human_states)
                
                # 不再需要单独设置other_robot_states，因为已经在构造函数中传递了
                # if next_robot_states:
                #     next_state.other_robot_states = next_robot_states
                
                # 6. 使用安全的转换方法处理状态
                state_tensor = self.transform(next_state)
                
                # 7. 安全地评估状态值
                try:
                    # 检查张量的维度
                    if len(state_tensor.shape) == 2:
                        # 添加批次维度
                        state_tensor = state_tensor.unsqueeze(0)
                    
                    # 根据当前训练阶段评估值
                    if hasattr(self.model, 'forward') and hasattr(self, 'training_phase'):
                        value = self.model(state_tensor, self.training_phase).data.item()
                    else:
                        value = self.model(state_tensor).data.item()
                    
                    # 计算最终值
                    value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                    
                except Exception as e:
                    logging.warning(f"Error evaluating state value: {e}")
                    # 如果评估失败，只使用奖励作为值
                    value = reward
                
                # 8. 更新最佳动作
                if value > max_value:
                    max_value = value
                    max_action = action
                    
            except Exception as e:
                logging.debug(f"Error evaluating action: {e}")
                continue
        
        # 确保找到有效动作
        if max_action is None:
            logging.warning("No valid action found, using goal-directed action")
            direction = np.array([state.self_state.gx - state.self_state.px, 
                                 state.self_state.gy - state.self_state.py])
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            max_action = ActionXY(direction[0] * state.self_state.v_pref * 0.5, 
                                  direction[1] * state.self_state.v_pref * 0.5)
        
        # 记录值信息用于调试
        self.action_values = max_value
        
        return max_action

    def set_phase(self, phase):
        """
        Override set_phase to handle training phase updates
        """
        super().set_phase(phase)
        if hasattr(self, 'model'):
            self.model.training_phase = self.training_phase
    
    def compute_reward(self, next_state, next_human_states, next_robot_states=None):
        """
        重新设计的奖励函数，确保明确的奖励信号，增加软管安全性考虑与目标引导
        """
        # 1. 首先检查环境状态
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'success') and self.env.success:
                logging.debug("计算奖励：成功到达目标，奖励 +10.0")
                return 10.0
            if hasattr(self.env, 'collision') and self.env.collision:
                logging.debug("计算奖励：发生碰撞，惩罚 -2.5") 
                return -2.5
        
        # 2. 基于状态计算奖励 (环境状态不可用时)
        reward = 0
        
        # 添加少量随机性以避免相同的奖励值
        noise = np.random.normal(0, 0.001)  # 降低噪声幅度，避免抖动
        reward += noise
        
        # 检查人类是否已经全部到达目标
        all_humans_at_goal = False
        humans_stable_at_goal = False
        
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'all_humans_at_goal'):
                all_humans_at_goal = self.env.all_humans_at_goal
                # 检查人类是否稳定在目标位置
                if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                    humans_stable_at_goal = all_humans_at_goal and (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold
        
        # 目标检查
        dist_to_goal = np.linalg.norm([next_state.gx - next_state.px, next_state.gy - next_state.py])
        if dist_to_goal < next_state.radius:
            reward += 10.0  # 到达目标
            logging.debug(f"奖励：到达目标 +10.0")
        elif dist_to_goal < 0.5:  # 接近目标时给予额外奖励
            # 距离越近，奖励越大 (平滑增长)
            # 人类稳定在目标时增加接近目标的奖励
            close_reward_factor = 2.0 if not humans_stable_at_goal else 3.0
            close_reward = close_reward_factor * (1.0 - dist_to_goal / 0.5)
            reward += close_reward
            logging.debug(f"奖励：接近目标 +{close_reward:.2f}")
        elif humans_stable_at_goal and dist_to_goal < 2.0:
            # 人类稳定且不太远时，额外增加朝向目标的奖励
            goal_reward = 0.5 * (1.0 - dist_to_goal / 2.0)
            reward += goal_reward
            logging.debug(f"奖励：人类稳定状态下接近目标 +{goal_reward:.2f}")
        
        # 获取使用软管的状态
        use_hose = False
        has_hose_partner = False
        hose_partner = None
        
        try:
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            # 检查是否有软管伙伴
            if use_hose and hasattr(self, 'env') and self.env is not None:
                if hasattr(next_state, 'robot_index'):
                    robot_index = next_state.robot_index
                    partner_idx = robot_index + 1 if robot_index % 2 == 0 else robot_index - 1
                    
                    if hasattr(self.env, 'robots') and 0 <= partner_idx < len(self.env.robots):
                        has_hose_partner = True
                        hose_partner = self.env.robots[partner_idx]
        except Exception as e:
            logging.debug(f"Error checking hose status: {e}")
        
        # 与人类的碰撞检查
        human_collision = False
        min_human_dist = float('inf')
        for human in next_human_states:
            dist = np.linalg.norm([human.px - next_state.px, human.py - next_state.py]) - human.radius - next_state.radius
            min_human_dist = min(min_human_dist, dist)
            if dist < 0:
                human_collision = True
                break
        
        # 与其他机器人的碰撞检查
        robot_collision = False
        min_robot_dist = float('inf')
        if next_robot_states:
            for robot in next_robot_states:
                if robot.px != next_state.px or robot.py != next_state.py:  # 不是同一机器人
                    dist = np.linalg.norm([robot.px - next_state.px, robot.py - next_state.py]) - robot.radius - next_state.radius
                    min_robot_dist = min(min_robot_dist, dist)
                    if dist < 0:
                        robot_collision = True
                        break
        
        # 软管碰撞检查
        hose_collision = False
        min_hose_dist = float('inf')
        
        if use_hose and has_hose_partner and hose_partner is not None:
            try:
                # 获取软管长度
                hose_length = getattr(self.env, 'hose_length', 0)
                hose_thickness = getattr(self.env, 'hose_thickness', 0.1)
                
                if hose_length > 0:
                    # 导入工具函数
                    from crowd_sim.envs.utils.utils import point_to_hose_curve
                    
                    # 检查每个人与软管的碰撞
                    for human in next_human_states:
                        human_pos = (human.px, human.py)
                        robot1_pos = (next_state.px, next_state.py)
                        robot2_pos = (hose_partner.px, hose_partner.py)
                        
                        # 计算人到软管的距离
                        distance = point_to_hose_curve(
                            human_pos, robot1_pos, robot2_pos, hose_length
                        ) - human.radius - hose_thickness
                        
                        min_hose_dist = min(min_hose_dist, distance)
                        
                        # 检测碰撞
                        if distance < 0:
                            hose_collision = True
                            break
            except Exception as e:
                logging.debug(f"Error checking hose collision: {e}")
        
        # 根据情况分配奖励
        if human_collision:
            reward -= 2.5  # 与人碰撞惩罚
            logging.debug(f"惩罚：与人碰撞 -2.5")
        elif robot_collision:
            reward -= 2.0  # 与机器人碰撞惩罚
            logging.debug(f"惩罚：与机器人碰撞 -2.0")
        elif hose_collision:
            reward -= 3.0  # 软管碰撞惩罚更严重
            logging.debug(f"惩罚：软管碰撞 -3.0")
        else:
            # 安全距离奖励/惩罚
            if min_human_dist < 0.5:
                # 人类稳定时减轻惩罚
                penalty_factor = 0.5 if humans_stable_at_goal else 1.0
                penalty = (0.5 - min_human_dist) * 1.0 * penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：接近人类 -{penalty:.2f}")
            
            if next_robot_states and min_robot_dist < 1.0:
                # 人类稳定时减轻惩罚
                robot_penalty_factor = 0.7 if humans_stable_at_goal else 1.5
                penalty = (1.0 - min_robot_dist) * robot_penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：接近机器人 -{penalty:.2f}")
            
            # 软管安全距离惩罚
            if use_hose and min_hose_dist < 0.5:
                # 人类稳定时减轻惩罚
                hose_penalty_factor = 0.8 if humans_stable_at_goal else 2.0
                penalty = (0.5 - min_hose_dist) * hose_penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：软管接近人类 -{penalty:.2f}")
            
            # 软管张力惩罚
            if use_hose and has_hose_partner and hose_partner is not None:
                hose_length = getattr(self.env, 'hose_length', 0)
                
                if hose_length > 0:
                    # 计算与伙伴的距离
                    dist = np.linalg.norm([
                        hose_partner.px - next_state.px,
                        hose_partner.py - next_state.py
                    ])
                    
                    # 张力阈值，小于此值不惩罚（使用环境设置的阈值，如果存在）
                    tension_threshold = getattr(self.env, 'hose_tension_threshold', 0.2)
                    
                    # 过度张紧惩罚 - 人类稳定时减轻
                    if dist > hose_length:
                        tension_diff = dist - hose_length
                        # 只有当超过阈值才惩罚
                        if tension_diff > tension_threshold:
                            tension_penalty_factor = 0.5 if humans_stable_at_goal else 1.5
                            tension_penalty = tension_diff * tension_penalty_factor
                            reward -= tension_penalty
                            logging.debug(f"惩罚：软管过度张紧 -{tension_penalty:.2f}")
                    # 软管松弛奖励 - 人类稳定时增加
                    elif abs(dist - hose_length) < tension_threshold * 0.5:
                        slack_reward = 0.1 if not humans_stable_at_goal else 0.2
                        reward += slack_reward
                        logging.debug(f"奖励：软管松弛适当 +{slack_reward:.2f}")
            
            # 朝向目标奖励 - 人类稳定时增加
            goal_dir = np.array([next_state.gx - next_state.px, next_state.gy - next_state.py])
            if np.linalg.norm(goal_dir) > 0:
                goal_dir = goal_dir / np.linalg.norm(goal_dir)
                current_dir = np.array([next_state.vx, next_state.vy])
                current_speed = np.linalg.norm(current_dir)
                
                if current_speed > 0:
                    current_dir = current_dir / current_speed
                    # 朝向目标的奖励系数 - 人类稳定时加倍
                    direction_reward_factor = 0.8 if humans_stable_at_goal else 0.4
                    dir_reward = direction_reward_factor * np.dot(goal_dir, current_dir)
                    reward += dir_reward
                    logging.debug(f"奖励：朝向目标 +{dir_reward:.2f}")
                    
                    # 人类稳定时，额外奖励朝向目标的速度值
                    if humans_stable_at_goal and np.dot(goal_dir, current_dir) > 0.8:
                        speed_reward = 0.2 * current_speed * (np.dot(goal_dir, current_dir))
                        reward += speed_reward
                        logging.debug(f"奖励：目标方向上的速度 +{speed_reward:.2f}")
            
            # 进度奖励
            if hasattr(self, 'last_dist_to_goal'):
                progress = self.last_dist_to_goal - dist_to_goal
                if progress > 0:
                    # 人类稳定时增加进度奖励
                    prog_reward_factor = 1.0 if not humans_stable_at_goal else 2.0
                    prog_reward = progress * prog_reward_factor
                    reward += prog_reward
                    logging.debug(f"奖励：朝目标前进 +{prog_reward:.2f}")
                # 轻微惩罚远离目标，但不要太强，避免抖动
                elif progress < -0.1:  # 只有明显远离才惩罚
                    # 人类稳定时减轻惩罚
                    regression_penalty_factor = 0.05 if humans_stable_at_goal else 0.1
                    penalty = -progress * regression_penalty_factor
                    reward -= penalty
                    logging.debug(f"惩罚：远离目标 -{penalty:.2f}")
            self.last_dist_to_goal = dist_to_goal
            
            # 速度阻尼 - 惩罚速度变化，减少抖动
            if hasattr(self, 'last_velocity'):
                vel_change = np.linalg.norm([
                    next_state.vx - self.last_velocity[0],
                    next_state.vy - self.last_velocity[1]
                ])
                
                # 当接近目标或有软管危险时增加阻尼效果
                damping_factor = 0.1
                if dist_to_goal < 1.0 or (use_hose and min_hose_dist < 1.0):
                    damping_factor = 0.3
                
                # 人类稳定时，减少速度阻尼惩罚
                if humans_stable_at_goal:
                    damping_factor *= 0.5
                
                if vel_change > 0.2:  # 只惩罚较大的速度变化
                    vel_penalty = damping_factor * (vel_change - 0.2)
                    reward -= vel_penalty
                    logging.debug(f"阻尼：速度变化 -{vel_penalty:.2f}")
            
            # 保存当前速度用于下次比较
            self.last_velocity = [next_state.vx, next_state.vy]
        
        logging.debug(f"总奖励: {reward:.4f}")
        return reward
    
class LLMSARL(SARL):
    def __init__(self):
        super().__init__()
        self.name = 'LLM-SARL'
        self.llm_decision_maker = None
        self.api_key = None
        self.debug_llm = False
        self.llm_failure_count = 0
        self.max_llm_failures = 100  # Max failures before disabling LLM temporarily

    def configure(self, config):
        # 首先调用父类的configure来设置基本参数
        super().configure(config)
        
        # 设置LLM相关参数
        self.api_key = "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"
        self.debug_llm = True
        
        # 如果API密钥可用，初始化LLM决策器
        if self.api_key:
            try:
                api_key_file = "/home/zjw/catkin_ws/src/crowd_nav/llm_decision/api_key.txt"
                # 使用动态导入避免不必要的依赖
                self.llm_decision_maker = LLMDecisionMaker(api_key_file, None)  # env会在set_env中设置 self.api_key
                logging.info("LLM decision maker initialized")
            except ImportError as e:
                logging.error(f"Failed to import LLMDecisionMaker: {e}")
                self.llm_decision_maker = None
            except Exception as e:
                logging.error(f"Failed to initialize LLM decision maker: {e}")
                self.llm_decision_maker = None
        else:
            logging.warning("No API key provided for LLM, running without LLM advice")
            self.llm_decision_maker = None

        # 获取网络维度参数
        self.mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        self.mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        self.mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        self.attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_global_state = config.getboolean('sarl', 'with_global_state')
        
        # # 初始化LLM配置
        # self.llm_enabled = config.getboolean('llm', 'enabled', fallback=True)
        # print(f"LLM enabled: {self.llm_enabled}")
        # api_key = "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"
        # self.use_llm_in_training = config.getboolean('llm', 'use_in_training', fallback=True)
        # print(f"Use LLM in training: {self.use_llm_in_training}")
        
        # 从配置中获取速度参数
        self.v_pref = config.getfloat('robot', 'v_pref', fallback=1.0)

        # if self.llm_enabled and api_key:
        #     # 传入环境引用
        #     self.llm_decision_maker = LLMDecisionMaker(api_key, env=self.env)
        #     logging.info('LLM decision maker initialized')
            
        # 使用增强的网络替换基础网络
        self.model = LLMValueNetwork(
            self.input_dim(), self.self_state_dim,
            self.mlp1_dims, self.mlp2_dims, self.mlp3_dims,
            self.attention_dims, self.with_global_state,
            self.cell_size, self.cell_num
        )
        
        logging.info('Policy: {} {} global state'.format(
            self.name, 
            'w/' if self.with_global_state else 'w/o'
        ))

    def set_env(self, env):
        super().set_env(env)
        # 更新LLM决策器中的环境引用
        if self.llm_decision_maker:
            self.llm_decision_maker.env = env

    def predict(self, state, robot_index):
        """
        级联式处理：
        1. LLM提供初始决策
        2. RL基于LLM的决策进行优化
        """
        # 如果LLM失败次数过多或不可用，直接使用基础RL
        if not self.llm_decision_maker or self.llm_failure_count > self.max_llm_failures:
            if self.llm_failure_count > self.max_llm_failures:
                logging.warning("Too many LLM failures, temporarily disabling LLM advice")
            return super().predict(state,robot_index)
        
        try:
            # 确保other_robots是一个列表
            other_robots = state.other_robot_states
            if not isinstance(other_robots, list):
                other_robots = [other_robots]   
            # 格式化状态用于LLM
            state_desc = self.llm_decision_maker.format_state_for_llm(
                robot_index,
                state.self_state, 
                state.human_states,
                other_robots
            )
            
            # 获取LLM决策建议
            if robot_index == 0:
                llm_advice = None
            else:
                llm_advice = self.llm_decision_maker.get_llm_decision(
                    state_desc, 
                    llm_advice,
                    is_training=(self.phase == 'train')
                )
            if self.debug_llm:
                logging.info(f"LLM advice: {llm_advice}")
            
            # # 如果LLM建议无效，使用父类的预测
            # if not isinstance(llm_advice, dict) or 'recommended_action' not in llm_advice:
            #     self.llm_failure_count += 1
            #     logging.warning(f"Invalid LLM advice format, using standard prediction ({self.llm_failure_count}/{self.max_llm_failures})")
            #     return super().predict(state)
                
            # 提取LLM建议的速度
            robot_advices = []
            try:
                # 提取所有robot_id的vx和vy
                for decision in llm_advice['robots_decisions']:
                    robot_advices.append({
                        'robot_id': decision['robot_id'],
                        'vx': decision['recommended_action']['vx'],
                        'vy': decision['recommended_action']['vy']
                    })

                # 将建议速度限制在合理范围内
                max_speed = state.self_state.v_pref  # 最大允许速度
                for advice in robot_advices:
                    vx, vy = advice['vx'], advice['vy']
                    vx_limited, vy_limited = limit_speed(vx, vy, max_speed)
                    # 更新建议速度
                    advice['vx'] = vx_limited
                    advice['vy'] = vy_limited

                 # 提取第一个robot_id为0（current robot）的vx和vy
                llm_vx, llm_vy = 0,0
                for advice in robot_advices:
                    if advice["robot_id"] == robot_index:
                        llm_vx = advice["vx"]
                        llm_vy = advice["vy"]   
                    
                other_robots_advices = [advice for advice in robot_advices if advice["robot_id"] != robot_index]
                # 创建LLM建议的动作
                llm_action = ActionXY(llm_vx, llm_vy)
                
                if self.debug_llm:
                    logging.info(f"LLM suggested velocity for Robot{robot_index} : vx={llm_vx:.2f}, vy={llm_vy:.2f}")
                
                logging.info("optimize with RL (LLM)")
                # 确保动作空间已初始化
                
                try:
                    self.build_llm_action_space(state.self_state.v_pref, llm_vx, llm_vy)
                except Exception as e:
                    logging.error(f"Error building action space: {e}")
                    return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)    
                # print("动作空间个数:",len(self.action_space))
                # 将LLM建议传递给RL优化
                max_value = -100
    
                for action in self.action_space:
                    value = self._optimize_with_rl(state, llm_advice, action, other_robots_advices)
                    # print(f"action:{action},value: {value}")
                    if value > max_value:
                        max_value = value
                        llm_action = action
                logging.info(f"RL finally suggested velocity for Robot{robot_index} : vx={llm_action.vx:.2f}, vy={llm_vy:.2f} with value of {max_value:.2f}")
                return llm_action
                
            except (KeyError, ValueError, TypeError) as e:
                self.llm_failure_count += 1
                logging.warning(f"Error processing LLM action values: {e}")
                return super().predict(state,robot_index)
            
        except Exception as e:
            self.llm_failure_count += 1
            logging.error(f"Error in LLM prediction: {e}")
            return super().predict(state,robot_index)
        
    def build_llm_action_space(self, v_pref, llm_vx, llm_vy):
        holonomic = True if self.kinematics == 'holonomic' else False
        
        if holonomic:
            # Calculate vx range (±20% of llm_vx, but ensure it's not zero-range)
            action_space = [ActionXY(llm_vx,llm_vy)]
            if llm_vx != 0:
                delta_vx = 0.2 * abs(llm_vx)
                vx_min = llm_vx - delta_vx
                vx_max = llm_vx + delta_vx
            else:
                # If llm_vx=0, use ±20% of v_pref
                delta_vx = 0.05 * v_pref
                vx_min = -delta_vx
                vx_max = delta_vx
            
            # Calculate vy range (±20% of llm_vy, but ensure it's not zero-range)
            if llm_vy != 0:
                delta_vy = 0.2 * abs(llm_vy)
                vy_min = llm_vy - delta_vy
                vy_max = llm_vy + delta_vy
            else:
                # If llm_vy=0, use ±20% of v_pref
                delta_vy = 0.05 * v_pref
                vy_min = -delta_vy
                vy_max = delta_vy
            
            # Clamp vx and vy to be within [-v_pref, v_pref] to avoid exceeding max speed
            vx_min = max(vx_min, -v_pref)
            vx_max = min(vx_max, v_pref)
            vy_min = max(vy_min, -v_pref)
            vy_max = min(vy_max, v_pref)
            
            # Generate samples for vx and vy
            vx_samples = np.linspace(vx_min, vx_max, self.speed_samples)
            vy_samples = np.linspace(vy_min, vy_max, self.speed_samples)
            
            # action_space = [ActionXY(0, 0)]  # Stop action
            
            # Create all combinations of vx and vy
            for vx, vy in itertools.product(vx_samples, vy_samples):
                action_space.append(ActionXY(vx, vy))
            
        else:
            # For non-holonomic, adjust speed around the magnitude of (llm_vx, llm_vy)    
            llm_speed = np.sqrt(llm_vx**2 + llm_vy**2)
            action_space = [ActionRot(llm_speed,np.arctan(llm_vy / llm_vx))]
            if llm_speed > 0:
                delta_speed = 0.2 * llm_speed
                speed_min = max(0, llm_speed - delta_speed)  # speed cannot be negative
                speed_max = min(llm_speed + delta_speed, v_pref)  # speed cannot exceed v_pref
            else:
                # If llm_speed=0, use ±20% of v_pref
                delta_speed = 0.05 * v_pref
                speed_min = 0
                speed_max = delta_speed
            
            speeds = np.linspace(speed_min, speed_max, self.speed_samples)
            
            # Assume rotation is derived from the original direction (if applicable)
            rotations = np.linspace(np.arctan(llm_vy / llm_vx)-np.pi/20, np.arctan(llm_vy / llm_vx)+np.pi/20, self.rotation_samples)
            
            # action_space = [ActionRot(0, 0)]  # Stop action
            for rotation, speed in itertools.product(rotations, speeds):
                action_space.append(ActionRot(speed, rotation))
            
            self.speeds = speeds
            self.rotations = rotations
        
        self.action_space = action_space

    def _optimize_with_rl(self, state, llm_advice, llm_action, other_robot_advices):
        """
        使用RL优化LLM提供的初始动作
        这是级联过程的第二阶段
        """
        self.last_state = self.transform(state)
        
        try:
            # 计算下一个状态
            next_self_state = self.propagate(state.self_state, llm_action)
            
            if self.query_env:
                actions = [llm_action] * (len(self.env.robots) if hasattr(self.env, 'robots') else 1)
                next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
            else:
                next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                    for human_state in state.human_states]
                reward = self.compute_reward(next_self_state, next_human_states)
            
            # print("reward:",reward)
            # 将状态和LLM建议一起传递给价值网络
            # 处理next_robot_states（确保处理其他机器人状态）
            # next_robot_states = []
            # if hasattr(state, 'other_robot_states') and state.other_robot_states:
            #     robot_states = state.other_robot_states
            #     if not isinstance(robot_states, list):
            #         robot_states = [robot_states]
            #     next_robot_states = [
            #         self.propagate(robot_state, ActionXY(advice["vx"], advice["vy"]))
            #         for robot_state, advice in zip(robot_states, other_robot_advices)
            #         if robot_state is not None
            #     ] #robot_states[i] 对应 robot_advices[i]（即顺序一致）
            next_robot_states = [self.propagate(robot_state, ActionXY(robot_state.vx, robot_state.vy))
                                  for robot_state in state.other_robot_states if robot_state is not None]
            
            # 创建下一个状态对象用于评估
            from crowd_sim.envs.utils.state import JointState
            next_state = JointState(next_self_state, next_robot_states, next_human_states)
            
            # 安全地转换状态为张量
            try:
                state_tensor = self.transform(next_state)
                
                # 检查张量的维度
                if len(state_tensor.shape) == 2:
                    # 添加批次维度
                    state_tensor = state_tensor.unsqueeze(0)
                
                # 准备LLM信息张量（确保维度正确）
                llm_info = torch.Tensor([
                    [
                        float(llm_action.vx) / state.self_state.v_pref,  # 归一化速度x
                        float(llm_action.vy) / state.self_state.v_pref,  # 归一化速度y
                        float(llm_advice.get('risk_assessment', 5)) / 10.0  # 风险评估
                    ]
                ]).to(self.device)
                
                # # 根据模型的预期输入修改调用方式
                # if hasattr(self.model, 'forward') and callable(self.model.forward):
                #     if 'training_phase' in inspect.signature(self.model.forward).parameters:
                #         value = self.model(state_tensor, llm_info, self.training_phase).data.item()
                #     else:
                #         value = self.model(state_tensor, llm_info).data.item()
                # else:
                #     value = self.model(state_tensor).data.item()

                value = self.model(state_tensor, llm_info).data.item()
                # print("value:",value)
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                
                # 重置失败计数
                self.llm_failure_count = 0
                self.last_value = value
                
                # 返回RL优化后的动作 (与LLM相同，因为级联架构中RL已内部优化)
                return value
                
            except Exception as e:
                logging.error(f"Error in state evaluation: {e}")
                # 如果张量处理失败，直接返回LLM动作
                return value
            
        except Exception as e:
            logging.error(f"Error in RL optimization: {e}")
            # 如果RL优化失败，仍然使用LLM的动作
            return value

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
        
        value = self.model(rotated_batch_input, llm_advice).data.item()
        return reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value

    def compute_reward(self, robot_state, human_states, other_robot_states=None):
        """
        计算奖励函数 - 提供明确的成功/失败信号和过程中的中间奖励
        
        Args:
            robot_state: 机器人状态
            human_states: 人类状态列表
            other_robot_states: 其他机器人状态列表
            
        Returns:
            float: 计算的奖励值
        """
        # 奖励初始化
        reward = 0.0
        
        # 1. 检查是否到达目标 (较高的正向奖励)
        if self.reach_destination(robot_state):
            logging.debug("Reached goal - high positive reward")
            return 10.0  # 到达目标获得高奖励
        
        # 2. 检查与人的碰撞 (高惩罚)
        for human_state in human_states:
            # 计算距离
            dx = robot_state.px - human_state.px
            dy = robot_state.py - human_state.py
            dist = np.sqrt(dx**2 + dy**2)
            
            # 如果碰撞
            if dist < self.robot_radius + human_state.radius:
                logging.debug("Collision with human - high penalty")
                return -10.0  # 碰撞人类获得高惩罚
            
            # 根据距离添加中间奖励/惩罚
            if dist < 1.0:  # 当距离较近时增加小惩罚
                reward -= 0.1 * (1.0 / max(0.1, dist))
        
        # 3. 检查与其他机器人的碰撞 (中等惩罚)
        if other_robot_states:
            for robot_state2 in other_robot_states:
                if robot_state2 is None:
                    continue
                    
                # 计算距离
                dx = robot_state.px - robot_state2.px
                dy = robot_state.py - robot_state2.py
                dist = np.sqrt(dx**2 + dy**2)
                
                # 如果碰撞
                if dist < self.robot_radius + robot_state2.radius:
                    logging.debug("Collision with robot - medium penalty")
                    return -8.0  # 碰撞其他机器人获得中等惩罚
                
                # 根据距离添加中间奖励/惩罚
                if dist < 1.5:  # 当距离较近时增加小惩罚
                    reward -= 0.2 * (1.5 / max(0.1, dist))
        
        # 4. 朝向目标的进度奖励
        goal_dist = np.sqrt((robot_state.gx - robot_state.px)**2 + (robot_state.gy - robot_state.py)**2)
        
        # 朝目标前进的奖励 (基于速度和方向)
        vector_to_goal = np.array([robot_state.gx - robot_state.px, robot_state.gy - robot_state.py])
        if np.linalg.norm(vector_to_goal) > 0:
            vector_to_goal = vector_to_goal / np.linalg.norm(vector_to_goal)
        
        velocity_vector = np.array([robot_state.vx, robot_state.vy])
        speed = np.linalg.norm(velocity_vector)
        
        # 速度奖励 - 鼓励机器人移动但不要太快
        speed_reward = min(speed, robot_state.v_pref) / robot_state.v_pref
        reward += 0.3 * speed_reward
        
        # 方向奖励 - 鼓励朝着目标方向移动
        if speed > 0:
            velocity_direction = velocity_vector / speed
            direction_reward = np.dot(vector_to_goal, velocity_direction)
            reward += 0.4 * direction_reward
        
        # 5. 到达目标的距离奖励
        # 只有当距离小于初始距离时才给予奖励，避免重复奖励
        if hasattr(self, 'last_goal_dist'):
            progress = self.last_goal_dist - goal_dist
            reward += 0.2 * progress
        
        # 更新上次距离
        self.last_goal_dist = goal_dist
        
        # 6. 时间惩罚 - 鼓励更快完成任务
        reward -= 0.01  # 小的时间惩罚
        
        # 确保奖励有变化
        logging.debug(f"Computed reward: {reward}")
        
        return reward

class HierarchicalLLMSARL(HierarchicalSARL):
    def __init__(self):
        super().__init__()
        self.name = 'H-LLM-SARL'
        self.llm_decision_maker = None
        self.api_key = None
        self.debug_llm = False
        self.llm_failure_count = 0
        self.max_llm_failures = 100  # Max failures before disabling LLM temporarily

    def configure(self, config):
        # 首先调用父类的configure来设置基本参数
        super().configure(config)
        
        # 设置LLM相关参数
        self.api_key = "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"
        self.debug_llm = True
        
        # 如果API密钥可用，初始化LLM决策器
        if self.api_key:
            try:
                api_key_file = "/home/zjw/catkin_ws/src/crowd_nav/llm_decision/api_key.txt"
                # 使用动态导入避免不必要的依赖
                self.llm_decision_maker = LLMDecisionMaker(api_key_file, None)  # env会在set_env中设置
                logging.info("LLM decision maker initialized")
            except ImportError as e:
                logging.error(f"Failed to import LLMDecisionMaker: {e}")
                self.llm_decision_maker = None
            except Exception as e:
                logging.error(f"Failed to initialize LLM decision maker: {e}")
                self.llm_decision_maker = None
        else:
            logging.warning("No API key provided for LLM, running without LLM advice")
            self.llm_decision_maker = None

        # 获取网络维度参数
        self.mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        self.mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        self.mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        self.attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_global_state = config.getboolean('sarl', 'with_global_state')
        
        # # 初始化LLM配置
        # self.llm_enabled = config.getboolean('llm', 'enabled', fallback=True)
        # print(f"LLM enabled: {self.llm_enabled}")
        # api_key = "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"
        # self.use_llm_in_training = config.getboolean('llm', 'use_in_training', fallback=True)
        # print(f"Use LLM in training: {self.use_llm_in_training}")
        
        # 从配置中获取速度参数
        self.v_pref = config.getfloat('robot', 'v_pref', fallback=1.0)

        # if self.llm_enabled and api_key:
        #     # 传入环境引用
        #     self.llm_decision_maker = LLMDecisionMaker(api_key, env=self.env)
        #     logging.info('LLM decision maker initialized')
            
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

    def set_env(self, env):
        super().set_env(env)
        # 更新LLM决策器中的环境引用
        if self.llm_decision_maker:
            self.llm_decision_maker.env = env
    def predict(self, state, robot_index):
        """
        级联式处理：
        1. LLM提供初始决策
        2. RL基于LLM的决策进行优化
        """
        # 如果LLM失败次数过多或不可用，直接使用基础RL
        if not self.llm_decision_maker or self.llm_failure_count > self.max_llm_failures:
            if self.llm_failure_count > self.max_llm_failures:
                logging.warning("Too many LLM failures, temporarily disabling LLM advice")
            return super().predict(state,robot_index)
        
        try:
            # 确保other_robots是一个列表
            other_robots = state.other_robot_states
            if not isinstance(other_robots, list):
                other_robots = [other_robots]   
            # 格式化状态用于LLM
            state_desc = self.llm_decision_maker.format_state_for_llm(
                robot_index,
                state.self_state, 
                state.human_states,
                other_robots
            )
            
            # 获取LLM决策建议
            if robot_index == 0:
                LLMDecisionMaker.llm_advice = None
            LLMDecisionMaker.llm_advice = self.llm_decision_maker.get_llm_decision(
                state_desc,
                LLMDecisionMaker.llm_advice, 
                is_training=(self.phase == 'train')
            )
            llm_advice = LLMDecisionMaker.llm_advice
            if self.debug_llm:
                logging.info(f"LLM advice: {llm_advice}")
            
            # # 如果LLM建议无效，使用父类的预测
            # if not isinstance(llm_advice, dict) or 'recommended_action' not in llm_advice:
            #     self.llm_failure_count += 1
            #     logging.warning(f"Invalid LLM advice format, using standard prediction ({self.llm_failure_count}/{self.max_llm_failures})")
            #     return super().predict(state)
                
            # 提取LLM建议的速度
            robot_advices = []
            try:
                # 提取所有robot_id的vx和vy
                for decision in llm_advice['robots_decisions']:
                    robot_advices.append({
                        'robot_id': decision['robot_id'],
                        'vx': decision['recommended_action']['vx'],
                        'vy': decision['recommended_action']['vy']
                    })

                # 将建议速度限制在合理范围内
                max_speed = state.self_state.v_pref  # 最大允许速度
                for advice in robot_advices:
                    vx, vy = advice['vx'], advice['vy']
                    vx_limited, vy_limited = limit_speed(vx, vy, max_speed)
                    # 更新建议速度
                    advice['vx'] = vx_limited
                    advice['vy'] = vy_limited

                 # 提取第一个robot_id为current robot的vx和vy
                llm_vx, llm_vy = 0,0
                for advice in robot_advices:
                    if advice["robot_id"] == robot_index:
                        llm_vx = advice["vx"]
                        llm_vy = advice["vy"]   
                    
                other_robots_advices = [advice for advice in robot_advices if advice["robot_id"] != robot_index]
                # 创建LLM建议的动作
                llm_action = ActionXY(llm_vx, llm_vy)
                
                if self.debug_llm:
                    logging.info(f"LLM suggested velocity for Robot{robot_index+1} : vx={llm_vx:.2f}, vy={llm_vy:.2f}")
                
                logging.info("optimize with RL (LLM)")
                # 确保动作空间已初始化
                
                try:
                    self.build_llm_action_space(state.self_state.v_pref, llm_vx, llm_vy)
                except Exception as e:
                    logging.error(f"Error building action space: {e}")
                    return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)    
                # print("动作空间个数:",len(self.action_space))
                # 将LLM建议传递给RL优化
                max_value = -100
    
                for action in self.action_space:
                    value = self._optimize_with_rl(state, llm_advice, action, other_robots_advices)
                    # print(f"action:{action},value: {value}")
                    if value > max_value:
                        max_value = value
                        llm_action = action
                logging.info(f"RL finally suggested velocity for Robot{robot_index+1} : vx={llm_action.vx:.2f}, vy={llm_vy:.2f} with value of {max_value:.2f}")
                return llm_action
                
            except (KeyError, ValueError, TypeError) as e:
                self.llm_failure_count += 1
                logging.warning(f"Error processing LLM action values: {e}")
                return super().predict(state,robot_index)
            
        except Exception as e:
            self.llm_failure_count += 1
            logging.error(f"Error in LLM prediction: {e}")
            return super().predict(state,robot_index)
        
    def build_llm_action_space(self, v_pref, llm_vx, llm_vy):
        holonomic = True if self.kinematics == 'holonomic' else False
        
        if holonomic:
            # Calculate vx range (±20% of llm_vx, but ensure it's not zero-range)
            action_space = [ActionXY(llm_vx,llm_vy)]
            if llm_vx != 0:
                delta_vx = 0.2 * abs(llm_vx)
                vx_min = llm_vx - delta_vx
                vx_max = llm_vx + delta_vx
            else:
                # If llm_vx=0, use ±20% of v_pref
                delta_vx = 0.05 * v_pref
                vx_min = -delta_vx
                vx_max = delta_vx
            
            # Calculate vy range (±20% of llm_vy, but ensure it's not zero-range)
            if llm_vy != 0:
                delta_vy = 0.2 * abs(llm_vy)
                vy_min = llm_vy - delta_vy
                vy_max = llm_vy + delta_vy
            else:
                # If llm_vy=0, use ±20% of v_pref
                delta_vy = 0.05 * v_pref
                vy_min = -delta_vy
                vy_max = delta_vy
            
            # Clamp vx and vy to be within [-v_pref, v_pref] to avoid exceeding max speed
            vx_min = max(vx_min, -v_pref)
            vx_max = min(vx_max, v_pref)
            vy_min = max(vy_min, -v_pref)
            vy_max = min(vy_max, v_pref)
            
            # Generate samples for vx and vy
            vx_samples = np.linspace(vx_min, vx_max, self.speed_samples)
            vy_samples = np.linspace(vy_min, vy_max, self.speed_samples)
            
            # action_space = [ActionXY(0, 0)]  # Stop action
            
            # Create all combinations of vx and vy
            for vx, vy in itertools.product(vx_samples, vy_samples):
                action_space.append(ActionXY(vx, vy))
            
        else:
            # For non-holonomic, adjust speed around the magnitude of (llm_vx, llm_vy)    
            llm_speed = np.sqrt(llm_vx**2 + llm_vy**2)
            action_space = [ActionRot(llm_speed,np.arctan(llm_vy / llm_vx))]
            if llm_speed > 0:
                delta_speed = 0.2 * llm_speed
                speed_min = max(0, llm_speed - delta_speed)  # speed cannot be negative
                speed_max = min(llm_speed + delta_speed, v_pref)  # speed cannot exceed v_pref
            else:
                # If llm_speed=0, use ±20% of v_pref
                delta_speed = 0.05 * v_pref
                speed_min = 0
                speed_max = delta_speed
            
            speeds = np.linspace(speed_min, speed_max, self.speed_samples)
            
            # Assume rotation is derived from the original direction (if applicable)
            rotations = np.linspace(np.arctan(llm_vy / llm_vx)-np.pi/20, np.arctan(llm_vy / llm_vx)+np.pi/20, self.rotation_samples)
            
            # action_space = [ActionRot(0, 0)]  # Stop action
            for rotation, speed in itertools.product(rotations, speeds):
                action_space.append(ActionRot(speed, rotation))
            
            self.speeds = speeds
            self.rotations = rotations
        
        self.action_space = action_space

    def _optimize_with_rl(self, state, llm_advice, llm_action, other_robot_advices):
        """
        使用RL优化LLM提供的初始动作
        这是级联过程的第二阶段
        """
        self.last_state = self.transform(state)
        
        try:
            # 计算下一个状态
            next_self_state = self.propagate(state.self_state, llm_action)
            
            if self.query_env:
                actions = [llm_action] * (len(self.env.robots) if hasattr(self.env, 'robots') else 1)
                next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
            else:
                next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                    for human_state in state.human_states]
                reward = self.compute_reward(next_self_state, next_human_states)
            
            # print("reward:",reward)
            # 将状态和LLM建议一起传递给价值网络
            # 处理next_robot_states（确保处理其他机器人状态）
            # next_robot_states = []
            # if hasattr(state, 'other_robot_states') and state.other_robot_states:
            #     robot_states = state.other_robot_states
            #     if not isinstance(robot_states, list):
            #         robot_states = [robot_states]
            #     next_robot_states = [
            #         self.propagate(robot_state, ActionXY(advice["vx"], advice["vy"]))
            #         for robot_state, advice in zip(robot_states, other_robot_advices)
            #         if robot_state is not None
            #     ] #robot_states[i] 对应 robot_advices[i]（即顺序一致）
            next_robot_states = [self.propagate(robot_state, ActionXY(robot_state.vx, robot_state.vy))
                                  for robot_state in state.other_robot_states if robot_state is not None]
            
            # 创建下一个状态对象用于评估
            from crowd_sim.envs.utils.state import JointState
            next_state = JointState(next_self_state, next_robot_states, next_human_states)
            
            # 安全地转换状态为张量
            try:
                state_tensor = self.transform(next_state)
                
                # 检查张量的维度
                if len(state_tensor.shape) == 2:
                    # 添加批次维度
                    state_tensor = state_tensor.unsqueeze(0)
                
                # 准备LLM信息张量（确保维度正确）
                llm_info = torch.Tensor([
                    [
                        float(llm_action.vx) / state.self_state.v_pref,  # 归一化速度x
                        float(llm_action.vy) / state.self_state.v_pref,  # 归一化速度y
                        float(llm_advice.get('risk_assessment', 5)) / 10.0  # 风险评估
                    ]
                ]).to(self.device)
                
                # # 根据模型的预期输入修改调用方式
                # if hasattr(self.model, 'forward') and callable(self.model.forward):
                #     if 'training_phase' in inspect.signature(self.model.forward).parameters:
                #         value = self.model(state_tensor, llm_info, self.training_phase).data.item()
                #     else:
                #         value = self.model(state_tensor, llm_info).data.item()
                # else:
                #     value = self.model(state_tensor).data.item()

                value = self.model(state_tensor, llm_info,self.training_phase).data.item()
                # print("value:",value)
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                
                # 重置失败计数
                self.llm_failure_count = 0
                self.last_value = value
                
                # 返回RL优化后的动作 (与LLM相同，因为级联架构中RL已内部优化)
                return value
                
            except Exception as e:
                logging.error(f"Error in state evaluation: {e}")
                # 如果张量处理失败，直接返回LLM动作
                return value
            
        except Exception as e:
            logging.error(f"Error in RL optimization: {e}")
            # 如果RL优化失败，仍然使用LLM的动作
            return value
    # def predict(self, state,robot_index):
    #     """
    #     级联式处理：
    #     1. LLM提供初始决策
    #     2. RL基于LLM的决策进行优化
    #     """
    #     # 如果LLM失败次数过多或不可用，直接使用基础RL
    #     if not self.llm_decision_maker or self.llm_failure_count > self.max_llm_failures:
    #         if self.llm_failure_count > self.max_llm_failures:
    #             logging.warning("Too many LLM failures, temporarily disabling LLM advice")
    #         return super().predict(state,robot_index)
        
    #     try:
    #         # 确保other_robots是一个列表
    #         other_robots = state.other_robot_states
    #         if not isinstance(other_robots, list):
    #             other_robots = [other_robots]   
    #         # 格式化状态用于LLM
    #         state_desc = self.llm_decision_maker.format_state_for_llm(
    #             state.self_state, 
    #             state.human_states,
    #             other_robots
    #         )
            
    #         # 获取LLM决策建议
    #         llm_advice = self.llm_decision_maker.get_llm_decision(
    #             state_desc, 
    #             is_training=(self.phase == 'train')
    #         )
    #         if self.debug_llm:
    #             logging.info(f"LLM advice: {llm_advice}")
            
    #         # # 如果LLM建议无效，使用父类的预测
    #         # if not isinstance(llm_advice, dict) or 'recommended_action' not in llm_advice:
    #         #     self.llm_failure_count += 1
    #         #     logging.warning(f"Invalid LLM advice format, using standard prediction ({self.llm_failure_count}/{self.max_llm_failures})")
    #         #     return super().predict(state)
                
    #         # 提取LLM建议的速度
    #         robot_advices = []
    #         try:
    #             # 提取所有robot_id的vx和vy
    #             for decision in llm_advice['robots_decisions']:
    #                 robot_advices.append({
    #                     'robot_id': decision['robot_id'],
    #                     'vx': decision['recommended_action']['vx'],
    #                     'vy': decision['recommended_action']['vy']
    #                 })

    #             # 将建议速度限制在合理范围内
    #             max_speed = state.self_state.v_pref  # 最大允许速度
    #             for advice in robot_advices:
    #                 vx, vy = advice['vx'], advice['vy']
    #                 vx_limited, vy_limited = limit_speed(vx, vy, max_speed)
    #                 # 更新建议速度
    #                 advice['vx'] = vx_limited
    #                 advice['vy'] = vy_limited

    #              # 提取第一个robot_id为0（current robot）的vx和vy
    #             for advice in robot_advices:
    #                 if advice["robot_id"] == 0:
    #                     llm_vx = advice["vx"]
    #                     llm_vy = advice["vy"]   
                    
    #             other_robots_advices = [advice for advice in robot_advices if advice["robot_id"] != 0]
    #             # 创建LLM建议的动作
    #             llm_action = ActionXY(llm_vx, llm_vy)
                
    #             if self.debug_llm:
    #                 logging.info(f"LLM suggested velocity: vx={llm_vx:.2f}, vy={llm_vy:.2f}")
                    
    #             # 将LLM建议传递给RL优化
    #             return self._optimize_with_rl(state, llm_advice, llm_action, other_robots_advices)
                
    #         except (KeyError, ValueError, TypeError) as e:
    #             self.llm_failure_count += 1
    #             logging.warning(f"Error processing LLM action values: {e}")
    #             return super().predict(state,robot_index)
            
    #     except Exception as e:
    #         self.llm_failure_count += 1
    #         logging.error(f"Error in LLM prediction: {e}")
    #         return super().predict(state,robot_index)


    # def _optimize_with_rl(self, state, llm_advice, llm_action, other_robot_advices):
    #     """
    #     使用RL优化LLM提供的初始动作
    #     这是级联过程的第二阶段
    #     """
    #     self.last_state = self.transform(state)
        
    #     try:
    #         # 计算下一个状态
    #         next_self_state = self.propagate(state.self_state, llm_action)
            
    #         if self.query_env:
    #             actions = [llm_action] * (len(self.env.robots) if hasattr(self.env, 'robots') else 1)
    #             next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
    #         else:
    #             next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
    #                                 for human_state in state.human_states]
    #             reward = self.compute_reward(next_self_state, next_human_states)
            
    #         # 将状态和LLM建议一起传递给价值网络
    #         # 处理next_robot_states（确保处理其他机器人状态）
    #         next_robot_states = []
    #         if hasattr(state, 'other_robot_states') and state.other_robot_states:
    #             robot_states = state.other_robot_states
    #             if not isinstance(robot_states, list):
    #                 robot_states = [robot_states]
    #             next_robot_states = [
    #                 self.propagate(robot_state, ActionXY(advice["vx"], advice["vy"]))
    #                 for robot_state, advice in zip(robot_states, other_robot_advices)
    #                 if robot_state is not None
    #             ] #robot_states[i] 对应 robot_advices[i]（即顺序一致）
    #             # next_robot_states = [self.propagate(robot_state, ActionXY(robot_state.vx, robot_state.vy))
    #             #                   for robot_state in robot_states if robot_state is not None]
            
    #         # 创建下一个状态对象用于评估
    #         from crowd_sim.envs.utils.state import JointState
    #         next_state = JointState(next_self_state, next_robot_states, next_human_states)
            
    #         # 安全地转换状态为张量
    #         try:
    #             state_tensor = self.transform(next_state)
                
    #             # 检查张量的维度
    #             if len(state_tensor.shape) == 2:
    #                 # 添加批次维度
    #                 state_tensor = state_tensor.unsqueeze(0)
                
    #             # 准备LLM信息张量（确保维度正确）
    #             llm_info = torch.Tensor([
    #                 [
    #                     float(llm_action.vx) / state.self_state.v_pref,  # 归一化速度x
    #                     float(llm_action.vy) / state.self_state.v_pref,  # 归一化速度y
    #                     float(llm_advice.get('risk_assessment', 5)) / 10.0  # 风险评估
    #                 ]
    #             ]).to(self.device)
                
    #             # 根据模型的预期输入修改调用方式
    #             if hasattr(self.model, 'forward') and callable(self.model.forward):
    #                 if 'training_phase' in inspect.signature(self.model.forward).parameters:
    #                     value = self.model(state_tensor, llm_info, self.training_phase).data.item()
    #                 else:
    #                     value = self.model(state_tensor, llm_info).data.item()
    #             else:
    #                 value = self.model(state_tensor).data.item()
                
    #             value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * value
                
    #             # 重置失败计数
    #             self.llm_failure_count = 0
    #             self.last_value = value
                
    #             # 返回RL优化后的动作 (与LLM相同，因为级联架构中RL已内部优化)
    #             return llm_action
                
    #         except Exception as e:
    #             logging.error(f"Error in state evaluation: {e}")
    #             # 如果张量处理失败，直接返回LLM动作
    #             return llm_action
            
    #     except Exception as e:
    #         logging.error(f"Error in RL optimization: {e}")
    #         # 如果RL优化失败，仍然使用LLM的动作
    #         return llm_action


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

    def compute_reward(self, next_state, next_human_states, next_robot_states=None):
        """
        计算奖励函数 - 提供明确的成功/失败信号和过程中的中间奖励
        """
        # 1. 首先检查环境状态
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'success') and self.env.success:
                logging.debug("计算奖励：成功到达目标，奖励 +10.0")
                return 10.0
            if hasattr(self.env, 'collision') and self.env.collision:
                logging.debug("计算奖励：发生碰撞，惩罚 -2.5") 
                return -2.5
        
        # 2. 基于状态计算奖励 (环境状态不可用时)
        reward = 0
        
        # 添加少量随机性以避免相同的奖励值
        noise = np.random.normal(0, 0.001)  # 降低噪声幅度，避免抖动
        reward += noise
        
        # 检查人类是否已经全部到达目标
        all_humans_at_goal = False
        humans_stable_at_goal = False
        
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'all_humans_at_goal'):
                all_humans_at_goal = self.env.all_humans_at_goal
                # 检查人类是否稳定在目标位置
                if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                    humans_stable_at_goal = all_humans_at_goal and (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold
        
        # 目标检查
        dist_to_goal = np.linalg.norm([next_state.gx - next_state.px, next_state.gy - next_state.py])
        if dist_to_goal < next_state.radius:
            reward += 10.0  # 到达目标
            logging.debug(f"奖励：到达目标 +10.0")
        elif dist_to_goal < 0.5:  # 接近目标时给予额外奖励
            # 距离越近，奖励越大 (平滑增长)
            # 人类稳定在目标时增加接近目标的奖励
            close_reward_factor = 2.0 if not humans_stable_at_goal else 3.0
            close_reward = close_reward_factor * (1.0 - dist_to_goal / 0.5)
            reward += close_reward
            logging.debug(f"奖励：接近目标 +{close_reward:.2f}")
        elif humans_stable_at_goal and dist_to_goal < 2.0:
            # 人类稳定且不太远时，额外增加朝向目标的奖励
            goal_reward = 0.5 * (1.0 - dist_to_goal / 2.0)
            reward += goal_reward
            logging.debug(f"奖励：人类稳定状态下接近目标 +{goal_reward:.2f}")
        
        # 获取使用软管的状态
        use_hose = False
        has_hose_partner = False
        hose_partner = None
        
        try:
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            # 检查是否有软管伙伴
            if use_hose and hasattr(self, 'env') and self.env is not None:
                if hasattr(next_state, 'robot_index'):
                    robot_index = next_state.robot_index
                    partner_idx = robot_index + 1 if robot_index % 2 == 0 else robot_index - 1
                    
                    if hasattr(self.env, 'robots') and 0 <= partner_idx < len(self.env.robots):
                        has_hose_partner = True
                        hose_partner = self.env.robots[partner_idx]
        except Exception as e:
            logging.debug(f"Error checking hose status: {e}")
        
        # 与人类的碰撞检查
        human_collision = False
        min_human_dist = float('inf')
        for human in next_human_states:
            dist = np.linalg.norm([human.px - next_state.px, human.py - next_state.py]) - human.radius - next_state.radius
            min_human_dist = min(min_human_dist, dist)
            if dist < 0:
                human_collision = True
                break
        
        # 与其他机器人的碰撞检查
        robot_collision = False
        min_robot_dist = float('inf')
        if next_robot_states:
            for robot in next_robot_states:
                if robot.px != next_state.px or robot.py != next_state.py:  # 不是同一机器人
                    dist = np.linalg.norm([robot.px - next_state.px, robot.py - next_state.py]) - robot.radius - next_state.radius
                    min_robot_dist = min(min_robot_dist, dist)
                    if dist < 0:
                        robot_collision = True
                        break
        
        # 软管碰撞检查
        hose_collision = False
        other_hose_collision = False  # 与其他软管碰撞标志
        min_hose_dist = float('inf')
        min_hose_to_hose_dist = float('inf')  # 最小软管间距离
        
        if use_hose and hose_partner is not None:
            try:
                # 获取软管长度
                hose_length = getattr(self.env, 'hose_length', 0)
                hose_thickness = getattr(self.env, 'hose_thickness', 0.1)
                
                if hose_length > 0:
                    # 导入工具函数
                    from crowd_sim.envs.utils.utils import point_to_hose_curve
                    
                    # 检查每个人与软管的碰撞
                    for human in next_human_states:
                        human_pos = (human.px, human.py)
                        robot1_pos = (next_state.px, next_state.py)
                        robot2_pos = (hose_partner.px, hose_partner.py)
                        
                        # 计算人到软管的距离
                        distance = point_to_hose_curve(
                            human_pos, robot1_pos, robot2_pos, hose_length
                        ) - human.radius - hose_thickness
                        
                        min_hose_dist = min(min_hose_dist, distance)
                        
                        # 检测碰撞
                        if distance < 0:
                            hose_collision = True
                            break
                    
                    # 软管之间的碰撞检查
                    if hasattr(next_state, 'robot_index'):
                        # 获取当前机器人软管对的索引
                        curr_pair_idx = next_state.robot_index // 2
                        
                        # 检查所有其他软管对
                        for i in range(0, len(self.env.robots), 2):
                            pair_idx = i // 2
                            # 跳过自己的软管对
                            if pair_idx == curr_pair_idx or i+1 >= len(self.env.robots):
                                continue
                            
                            # 获取其他软管对的两个机器人
                            other_robot1 = self.env.robots[i]
                            other_robot2 = self.env.robots[i+1]
                            
                            # 检查当前机器人到其他软管的距离
                            robot_pos = (next_state.px, next_state.py)
                            other_robot1_pos = (other_robot1.px, other_robot1.py)
                            other_robot2_pos = (other_robot2.px, other_robot2.py)
                            
                            # 计算距离
                            distance = point_to_hose_curve(
                                robot_pos, other_robot1_pos, other_robot2_pos, hose_length
                            ) - next_state.radius - hose_thickness
                            
                            min_hose_to_hose_dist = min(min_hose_to_hose_dist, distance)
                            
                            if distance < 0:
                                other_hose_collision = True
                                break
            except Exception as e:
                logging.debug(f"Error checking hose collision: {e}")
        
        # 根据情况分配奖励
        if human_collision:
            reward -= 2.5  # 与人碰撞惩罚
            logging.debug(f"惩罚：与人碰撞 -2.5")
        elif robot_collision:
            reward -= 2.0  # 与机器人碰撞惩罚
            logging.debug(f"惩罚：与机器人碰撞 -2.0")
        elif hose_collision:
            reward -= 3.0  # 软管碰撞惩罚更严重
            logging.debug(f"惩罚：软管碰撞 -3.0")
        elif other_hose_collision:
            # 与其他软管碰撞惩罚更严重，即使人类已经静止
            reward -= 3.5
            logging.debug(f"惩罚：与其他软管碰撞 -3.5")
        else:
            # 安全距离奖励/惩罚
            if min_human_dist < 0.5:
                # 人类稳定时减轻惩罚
                penalty_factor = 0.5 if humans_stable_at_goal else 1.0
                penalty = (0.5 - min_human_dist) * 1.0 * penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：接近人类 -{penalty:.2f}")
            
            if next_robot_states and min_robot_dist < 1.0:
                # 人类稳定时减轻惩罚
                robot_penalty_factor = 0.7 if humans_stable_at_goal else 1.5
                penalty = (1.0 - min_robot_dist) * robot_penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：接近机器人 -{penalty:.2f}")
            
            # 软管安全距离惩罚
            if use_hose and min_hose_dist < 0.5:
                # 人类稳定时减轻惩罚
                hose_penalty_factor = 0.8 if humans_stable_at_goal else 2.0
                penalty = (0.5 - min_hose_dist) * hose_penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：软管接近人类 -{penalty:.2f}")
            
            # 软管与其他软管安全距离惩罚
            if use_hose and min_hose_to_hose_dist < 0.8:
                # 人类稳定时也保持较高惩罚
                hose_to_hose_penalty_factor = 1.2 if humans_stable_at_goal else 1.5
                penalty = (0.8 - min_hose_to_hose_dist) * hose_to_hose_penalty_factor
                reward -= penalty
                logging.debug(f"惩罚：接近其他软管 -{penalty:.2f}")
            
            # 软管张力惩罚
            if use_hose and hose_partner is not None:
                hose_length = getattr(self.env, 'hose_length', 0)
                
                if hose_length > 0:
                    # 计算与伙伴的距离
                    dist = np.linalg.norm([
                        hose_partner.px - next_state.px,
                        hose_partner.py - next_state.py
                    ])
                    
                    # 张力阈值，小于此值不惩罚（使用环境设置的阈值，如果存在）
                    tension_threshold = getattr(self.env, 'hose_tension_threshold', 0.2)
                    
                    # 过度张紧惩罚 - 人类稳定时减轻
                    if dist > hose_length:
                        tension_diff = dist - hose_length
                        # 只有当超过阈值才惩罚
                        if tension_diff > tension_threshold:
                            tension_penalty_factor = 0.5 if humans_stable_at_goal else 1.5
                            tension_penalty = tension_diff * tension_penalty_factor
                            reward -= tension_penalty
                            logging.debug(f"惩罚：软管过度张紧 -{tension_penalty:.2f}")
                    # 软管松弛奖励 - 人类稳定时增加
                    elif abs(dist - hose_length) < tension_threshold * 0.5:
                        slack_reward = 0.1 if not humans_stable_at_goal else 0.2
                        reward += slack_reward
                        logging.debug(f"奖励：软管松弛适当 +{slack_reward:.2f}")
            
            # 朝向目标奖励 - 人类稳定时增加
            goal_dir = np.array([next_state.gx - next_state.px, next_state.gy - next_state.py])
            if np.linalg.norm(goal_dir) > 0:
                goal_dir = goal_dir / np.linalg.norm(goal_dir)
                current_dir = np.array([next_state.vx, next_state.vy])
                current_speed = np.linalg.norm(current_dir)
                
                if current_speed > 0:
                    current_dir = current_dir / current_speed
                    # 朝向目标的奖励系数 - 人类稳定时加倍
                    direction_reward_factor = 0.8 if humans_stable_at_goal else 0.4
                    dir_reward = direction_reward_factor * np.dot(goal_dir, current_dir)
                    reward += dir_reward
                    logging.debug(f"奖励：朝向目标 +{dir_reward:.2f}")
                    
                    # 人类稳定时，额外奖励朝向目标的速度值
                    if humans_stable_at_goal and np.dot(goal_dir, current_dir) > 0.8:
                        speed_reward = 0.2 * current_speed * (np.dot(goal_dir, current_dir))
                        reward += speed_reward
                        logging.debug(f"奖励：目标方向上的速度 +{speed_reward:.2f}")
            
            # 进度奖励
            if hasattr(self, 'last_dist_to_goal'):
                progress = self.last_dist_to_goal - dist_to_goal
                if progress > 0:
                    # 人类稳定时增加进度奖励
                    prog_reward_factor = 1.0 if not humans_stable_at_goal else 2.0
                    prog_reward = progress * prog_reward_factor
                    reward += prog_reward
                    logging.debug(f"奖励：朝目标前进 +{prog_reward:.2f}")
                # 轻微惩罚远离目标，但不要太强，避免抖动
                elif progress < -0.1:  # 只有明显远离才惩罚
                    # 人类稳定时减轻惩罚
                    regression_penalty_factor = 0.05 if humans_stable_at_goal else 0.1
                    penalty = -progress * regression_penalty_factor
                    reward -= penalty
                    logging.debug(f"惩罚：远离目标 -{penalty:.2f}")
            self.last_dist_to_goal = dist_to_goal
            
            # 速度阻尼 - 惩罚速度变化，减少抖动
            if hasattr(self, 'last_velocity'):
                vel_change = np.linalg.norm([
                    next_state.vx - self.last_velocity[0],
                    next_state.vy - self.last_velocity[1]
                ])
                
                # 当接近目标或有软管危险时增加阻尼效果
                damping_factor = 0.1
                if dist_to_goal < 1.0 or (use_hose and min_hose_dist < 1.0):
                    damping_factor = 0.3
                
                # 人类稳定时，减少速度阻尼惩罚
                if humans_stable_at_goal:
                    damping_factor *= 0.5
                
                if vel_change > 0.2:  # 只惩罚较大的速度变化
                    vel_penalty = damping_factor * (vel_change - 0.2)
                    reward -= vel_penalty
                    logging.debug(f"阻尼：速度变化 -{vel_penalty:.2f}")
            
            # 保存当前速度用于下次比较
            self.last_velocity = [next_state.vx, next_state.vy]
            
            # 人类稳定时协调奖励
            if humans_stable_at_goal and use_hose and hose_partner is not None:
                # 奖励软管伙伴协调移动
                try:
                    # 计算当前机器人和伙伴的移动方向
                    if hasattr(self, 'last_velocity') and hasattr(hose_partner.policy, 'last_velocity'):
                        # 获取两个机器人的速度向量
                        robot_vel = np.array([next_state.vx, next_state.vy])
                        partner_vel = np.array(hose_partner.policy.last_velocity)
                        
                        # 计算两个速度向量的相似度
                        robot_speed = np.linalg.norm(robot_vel)
                        partner_speed = np.linalg.norm(partner_vel)
                        
                        if robot_speed > 0.1 and partner_speed > 0.1:
                            robot_dir = robot_vel / robot_speed
                            partner_dir = partner_vel / partner_speed
                            
                            # 方向一致性
                            direction_alignment = np.dot(robot_dir, partner_dir)
                            
                            # 如果方向一致，给予奖励
                            if direction_alignment > 0.7:  # 方向相似
                                coord_reward = 0.2 * direction_alignment
                                reward += coord_reward
                                logging.debug(f"奖励：软管伙伴协调移动 +{coord_reward:.2f}")
                except Exception as e:
                    logging.debug(f"Error calculating coordination reward: {e}")
        
        logging.debug(f"总奖励: {reward:.4f}")
        return reward
