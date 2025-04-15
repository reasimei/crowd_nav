# 导入所需的库
import os
import json
import logging
import numpy as np
import requests
from openai import OpenAI
from typing import List, Dict, Any, Optional
from crowd_sim.envs.utils.state import FullState, ObservableState

class LLMDecisionMaker:
    """使用大语言模型进行导航决策的类"""
    def __init__(self, api_key: str, env):
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.env = env  # 存储环境引用
        self.cache = {}  # 缓存之前的决策结果
        self.total_tokens_used = 0  # 记录使用的总token数
        self.max_tokens_allowed = 100000  # 允许使用的最大token数
        self.training_calls = 0  # 训练时的调用次数
        self.max_training_calls = 500  # 训练时允许的最大调用次数
        self.last_valid_decision = None  # 存储最后一次有效的决策
        logging.info("LLM client initialized with DeepSeek API")

    def format_state_for_llm(self, self_state: FullState, 
                           human_states: List[ObservableState],
                           other_robot_states: List[FullState]) -> str:
        """将机器人状态转换为LLM可理解的描述"""
        # 构建状态描述字典
        state_desc = {
            # 当前机器人的状态信息
            "current_robot": {
                "id": self_state.robot_id if hasattr(self_state, 'robot_id') else 0,
                "current_position": [round(self_state.px, 2), round(self_state.py, 2)],
                "current_velocity": [round(self_state.vx, 2), round(self_state.vy, 2)],
                "goal_position": [round(self_state.gx, 2), round(self_state.gy, 2)],
                "radius": round(self_state.radius, 2),
                "preferred_speed": round(self_state.v_pref, 2)
            },
            # 人类的状态信息列表
            "humans": [
                {
                    "id": i,
                    "position": [round(h.px, 2), round(h.py, 2)],
                    "velocity": [round(h.vx, 2), round(h.vy, 2)],
                    "radius": round(h.radius, 2),
                    "goal_position": [
                        round(self.env.humans[i].gx, 2),
                        round(self.env.humans[i].gy, 2)
                    ] if self.env and i < len(self.env.humans) else None
                } for i, h in enumerate(human_states)
            ],
            # 其他机器人的状态信息列表
            "other_robots": [
                {
                    "id": r.robot_id if hasattr(r, 'robot_id') else i,
                    "position": [round(r.px, 2), round(r.py, 2)],
                    "velocity": [round(r.vx, 2), round(r.vy, 2)],
                    "goal_position": [round(r.gx, 2), round(r.gy, 2)],
                    "radius": round(r.radius, 2),
                    "preferred_speed": round(r.v_pref, 2)
                } for i, r in enumerate(other_robot_states) if r is not None
            ],
            # 环境信息
            "environment": {
                "simulation_type": self.env.config.get('sim', 'train_val_sim') if self.env else None,
                "square_width": self.env.square_width if self.env else None,
                "circle_radius": self.env.circle_radius if self.env else None,
                "human_num": len(human_states),
                "robot_num": len([r for r in other_robot_states if r is not None]) + 1
            }
        }
        
        # 添加关键距离信息
        state_desc["analysis"] = self._analyze_distances(self_state, human_states, other_robot_states)
        
        # 添加路径规划建议
        state_desc["path_planning"] = self._analyze_path_planning(state_desc)
        
        return json.dumps(state_desc, indent=2)

    def _analyze_path_planning(self, state_desc: Dict) -> Dict:
        """分析当前机器人的路径规划建议"""
        current_robot = state_desc["current_robot"]
        current_pos = current_robot["current_position"]
        goal_pos = current_robot["goal_position"]
        
        # 计算到目标的直接距离和方向
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        direct_distance = round(np.sqrt(dx*dx + dy*dy), 2)
        
        # 分析潜在障碍物
        obstacles = []
        # 检查人类障碍物
        for human in state_desc["humans"]:
            if self._is_in_path(current_pos, goal_pos, human["position"], human["radius"]):
                obstacles.append({
                    "type": "human",
                    "id": human["id"],
                    "position": human["position"]
                })
        
        # 检查机器人障碍物
        for robot in state_desc["other_robots"]:
            if self._is_in_path(current_pos, goal_pos, robot["position"], robot["radius"]):
                obstacles.append({
                    "type": "robot",
                    "id": robot["id"],
                    "position": robot["position"]
                })
        
        return {
            "direct_distance_to_goal": direct_distance,
            "obstacles_in_path": obstacles,
            "suggested_path_type": "direct" if not obstacles else "detour"
        }

    def _is_in_path(self, start: List[float], goal: List[float], 
                    point: List[float], radius: float) -> bool:
        """检查点是否在路径上"""
        # 计算路径向量
        path_vector = [goal[0]-start[0], goal[1]-start[1]]
        point_vector = [point[0]-start[0], point[1]-start[1]]
        path_length = np.sqrt(path_vector[0]**2 + path_vector[1]**2)
        
        if path_length == 0:
            return False
            
        # 计算点到路径的投影距离
        dot_product = (path_vector[0]*point_vector[0] + path_vector[1]*point_vector[1]) / path_length
        projection = [
            start[0] + (dot_product/path_length) * path_vector[0],
            start[1] + (dot_product/path_length) * path_vector[1]
        ]
        
        # 检查投影点是否在路径段上
        if 0 <= dot_product <= path_length:
            # 计算点到路径的垂直距离
            dx = point[0] - projection[0]
            dy = point[1] - projection[1]
            distance = np.sqrt(dx*dx + dy*dy)
            return distance <= radius + 0.5  # 添加一些余量
            
        return False
        
    def _analyze_distances(self, self_state, human_states, other_robot_states):
        """分析与其他实体的距离和潜在风险"""
        analysis = {
            "distance_to_goal": round(np.linalg.norm([self_state.gx - self_state.px, 
                                                    self_state.gy - self_state.py]), 2),
            "nearest_human_distance": float('inf'),
            "nearest_robot_distance": float('inf'),
            "potential_collisions": []
        }
        
        # 分析与人类的距离
        for human in human_states:
            dist = np.linalg.norm([human.px - self_state.px, human.py - self_state.py])
            if dist < analysis["nearest_human_distance"]:
                analysis["nearest_human_distance"] = round(dist, 2)
            if dist < 1.0:  # 潜在碰撞阈值
                analysis["potential_collisions"].append({
                    "type": "human",
                    "distance": round(dist, 2)
                })
                
        # 分析与其他机器人的距离
        for robot in other_robot_states:
            if robot is not None:
                dist = np.linalg.norm([robot.px - self_state.px, robot.py - self_state.py])
                if dist < analysis["nearest_robot_distance"]:
                    analysis["nearest_robot_distance"] = round(dist, 2)
                if dist < 2.0:  # 机器人间安全距离
                    analysis["potential_collisions"].append({
                        "type": "robot",
                        "distance": round(dist, 2)
                    })
                    
        return analysis

    def get_llm_decision(self, state_desc: str, is_training: bool = False) -> Dict[str, Any]:
        """查询LLM获取导航决策建议"""
        # 检查缓存
        if state_desc in self.cache:
            self.last_valid_decision = self.cache[state_desc]
            return self.last_valid_decision
            
        # 如果token超限，使用最后一次有效决策
        if self.total_tokens_used >= self.max_tokens_allowed:
            if self.last_valid_decision is None:
                self.last_valid_decision = self._get_default_decision()
            return self.last_valid_decision
            
        # 检查训练调用次数
        if is_training and self.training_calls >= self.max_training_calls:
            if self.last_valid_decision is None:
                self.last_valid_decision = self._get_default_decision()
            return self.last_valid_decision
            
        logging.info("Attempting LLM API call...")
        
        try:
            # 构建并发送API请求
            prompt = self._format_prompt(state_desc)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an intelligent robot navigation system. Provide responses in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            logging.info("LLM response received: %s", content)
            
            # 清理响应内容
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            try:
                # 解析JSON响应
                result = json.loads(content)
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used
                logging.info("Tokens used: %d, Total: %d", tokens_used, self.total_tokens_used)
                
                # 更新最后一次有效决策
                self.last_valid_decision = result
                
                # 缓存结果
                if not is_training or self.training_calls < 50:
                    self.cache[state_desc] = result
                    if len(self.cache) > 500:
                        self.cache.pop(next(iter(self.cache)))
                        
                if is_training:
                    self.training_calls += 1
                    
                return result
                
            except json.JSONDecodeError as e:
                logging.warning("Failed to parse LLM response: %s", str(e))
                return self.last_valid_decision if self.last_valid_decision else self._get_default_decision()
                
        except Exception as e:
            logging.warning("LLM API call failed: %s", str(e))
            return self.last_valid_decision if self.last_valid_decision else self._get_default_decision()
            
    def _format_prompt(self, state_desc: str) -> str:
        """格式化发送给LLM的提示"""
        return f"""Analyze the following robot navigation state and provide advice:

{state_desc}

Consider:
1. Immediate collision risks with humans and other robots
2. Distance and direction to goal
3. Current velocities of all agents
4. Safe navigation paths

Provide a structured response with:
1. risk_assessment: A number from 0-10 indicating overall risk level
2. recommended_action: Suggested velocity adjustments as {{"vx": float, "vy": float}}
3. reasoning: Brief explanation of the decision

Response must be in valid JSON format."""

    def _get_default_decision(self) -> Dict[str, Any]:
        """当LLM调用失败时返回默认决策"""
        return {
            "risk_assessment": 5,
            "recommended_action": {
                "vx": 0.0,
                "vy": 0.0
            },
            "reasoning": "Default safe decision due to LLM error"
        }