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
            base_url="https://openrouter.ai/api/v1"
        )
        self.env = env  # 存储环境引用
        self.cache = {}  # 缓存之前的决策结果
        self.total_tokens_used = 0  # 记录使用的总token数
        self.max_tokens_allowed = 100000  # 允许使用的最大token数
        self.training_calls = 0  # 训练时的调用次数
        self.max_training_calls = 50  # 训练时允许的最大调用次数
        self.last_valid_decision = None  # 存储最后一次有效的决策
        self.is_training = False  # 是否处于训练模式
        logging.info("LLM client initialized with DeepSeek API")

    def format_state_for_llm(self, state_desc):
        """
        格式化状态信息为LLM可理解的格式
        state_desc: 包含机器人和人类状态信息的字典
        """
        try:
            formatted_state = state_desc.copy()
            
            # 确保所有值都是基本类型而不是字典
            if "current_robot" in formatted_state:
                robot = formatted_state["current_robot"]
                formatted_state["current_robot"] = {
                    "id": int(robot["id"]),
                    "position": [
                        float(robot["position"]["x"]),
                        float(robot["position"]["y"])
                    ],
                    "velocity": [
                        float(robot["velocity"]["x"]),
                        float(robot["velocity"]["y"])
                    ],
                    "goal": [
                        float(robot["goal"]["x"]),
                        float(robot["goal"]["y"])
                    ]
                }
                
            if "humans" in formatted_state:
                formatted_state["humans"] = [
                    {
                        "position": [
                            float(human["position"]["x"]),
                            float(human["position"]["y"])
                        ],
                        "velocity": [
                            float(human["velocity"]["x"]),
                            float(human["velocity"]["y"])
                        ]
                    } for human in formatted_state["humans"]
                ]
                
            if "other_robots" in formatted_state:
                formatted_state["other_robots"] = [
                    {
                        "id": int(robot["id"]),
                        "position": [
                            float(robot["position"]["x"]),
                            float(robot["position"]["y"])
                        ],
                        "velocity": [
                            float(robot["velocity"]["x"]),
                            float(robot["velocity"]["y"])
                        ],
                        "goal": [
                            float(robot["goal"]["x"]),
                            float(robot["goal"]["y"])
                        ]
                    } for robot in formatted_state["other_robots"]
                ]
                
            return formatted_state
            
        except Exception as e:
            logging.warning(f"Error formatting state for LLM: {e}")
            return self._get_default_state()
            
    def _get_default_state(self):
        """返回默认状态格式"""
        return {
            "current_robot": {
                "id": 0,
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [0.0, 0.0]
            },
            "humans": [],
            "other_robots": []
        }

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
        
    def _analyze_distances(self, self_state, human_states, other_robots_states):
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
        for robot in other_robots_states:
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

    def _state_to_hashable(self, state_desc):
        """将状态描述转换为可哈希的格式"""
        try:
            # 将字典转换为排序后的元组
            def dict_to_tuple(d):
                if isinstance(d, dict):
                    return tuple(sorted((k, dict_to_tuple(v)) for k, v in d.items()))
                elif isinstance(d, list):
                    return tuple(dict_to_tuple(x) for x in d)
                else:
                    return d
                    
            return dict_to_tuple(state_desc)
        except Exception as e:
            logging.warning(f"Error converting state to hashable: {e}")
            return None

    def get_llm_decision(self, state_desc: Dict, is_training: bool = False) -> Dict[str, Any]:
        """查询LLM获取导航决策建议"""
        self.is_training = is_training
        try:
            # 记录输入状态
            logging.info("Input state description:\n%s", json.dumps(state_desc, indent=2))
            
            # 将状态转换为可哈希格式
            hashable_state = self._state_to_hashable(state_desc)
            
            # 检查缓存
            if hashable_state is not None and hashable_state in self.cache:
                self.last_valid_decision = self.cache[hashable_state]
                logging.info("Using cached decision:\n%s", 
                            json.dumps(self.last_valid_decision, indent=2))
                return self.last_valid_decision
                
            # 检查token限制
            if self.total_tokens_used >= self.max_tokens_allowed:
                logging.warning("Token limit reached (%d/%d)", 
                            self.total_tokens_used, self.max_tokens_allowed)
                return self.last_valid_decision if self.last_valid_decision else self._get_default_decision(state_desc)
                
            # 检查训练调用限制
            if is_training and self.training_calls >= self.max_training_calls:
                logging.warning("Training calls limit reached (%d/%d)", 
                            self.training_calls, self.max_training_calls)
                return self.last_valid_decision if self.last_valid_decision else self._get_default_decision(state_desc)
                
            # 获取LLM决策
            formatted_state = self.format_state_for_llm(state_desc)
            prompt = self._format_prompt(formatted_state)
            
            # API调用和结果处理
            result = self._call_llm_api(prompt)
            
            # 缓存结果
            if result is not None and hashable_state is not None:
                if not is_training or self.training_calls < 50:
                    self.cache[hashable_state] = result
                    logging.info("Decision cached. Cache size: %d", len(self.cache))
                    if len(self.cache) > 500:
                        self.cache.pop(next(iter(self.cache)))
                        logging.info("Cache pruned. New size: %d", len(self.cache))
                        
            return result
            
        except Exception as e:
            logging.warning("Error in LLM decision making: %s", str(e))
            return self.last_valid_decision if self.last_valid_decision else self._get_default_decision(state_desc)

    def _call_llm_api(self, prompt):
        """处理LLM API调用"""
        try:
            # 记录发送的prompt
            logging.info("Sending prompt to LLM:\n%s", prompt)
            
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-chat:free",
                messages=[
                    {"role": "system", "content": "You are an intelligent robot navigation system. Provide responses in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # 记录原始响应
            raw_content = response.choices[0].message.content
            logging.info("Raw LLM response:\n%s", raw_content)
            
            # 清理和解析响应
            content = self._clean_response(raw_content)
            result = json.loads(content)
            
            # 记录token使用情况
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            logging.info("Tokens used in this call: %d", tokens_used)
            logging.info("Total tokens used so far: %d", self.total_tokens_used)
            
            # 记录解析后的决策结果
            logging.info("Parsed LLM decision:\n%s", json.dumps(result, indent=2))
            
            # 更新最后一次有效决策
            self.last_valid_decision = result
            
            if self.is_training:
                self.training_calls += 1
                logging.info("Training call count: %d", self.training_calls)
                
            return result
            
        except Exception as e:
            logging.warning("LLM API call failed: %s", str(e))
            return None
    
    def _clean_response(self, response: str) -> str:
        """清理和验证LLM响应"""
        try:
            # 移除可能的前缀和后缀
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
                
            # 尝试解析JSON以验证格式
            cleaned_response = response.strip()
            json.loads(cleaned_response)  # 验证JSON格式
            return cleaned_response
            
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回默认响应
            default_response = json.dumps(self._get_default_decision({}))
            logging.warning(f"Invalid JSON response from LLM, using default: {default_response}")
            return default_response

    def _format_prompt(self, state_desc: str) -> str:
        return f"""
    Given the following multi-robot navigation state:
    {state_desc}

    Please analyze the situation and provide navigation decisions for ALL robots in the system. Each robot needs its own decision.
    Your response must be in the following JSON format:
    {{
        "risk_assessment": <overall_risk_level_1_to_10>,
        "robots_decisions": [
            {{
                "robot_id": <robot_id>,
                "recommended_action": {{
                    "vx": <velocity_x>,
                    "vy": <velocity_y>
                }},
                "reasoning": "<explanation_for_this_robot's_decision>"
            }},
            // Repeat for ALL robots in the system
        ],
        "overall_reasoning": "<overall_situation_analysis>"
    }}

    Important: You must provide decisions for ALL robots in the system, not just robot_id 0.
    Consider the interactions between all robots and humans, and provide coordinated movement suggestions.
    """

    def _get_default_decision(self, state_desc):
        """
        生成默认决策
        """
        try:
            current_robot_id = state_desc["current_robot"]["id"]
            return {
                "risk_assessment": "5.0",
                "robots_decisions": [
                    {
                        "robot_id": str(current_robot_id),
                        "recommended_action": {
                            "vx": "0.0",
                            "vy": "0.0"
                        }
                    }
                ]
            }
        except:
            # 如果出现任何错误，返回最基本的默认值
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