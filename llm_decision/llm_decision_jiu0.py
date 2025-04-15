import os
import json
import logging
import numpy as np
import requests
from openai import OpenAI
from typing import List, Dict, Any, Optional
from crowd_sim.envs.utils.state import FullState, ObservableState

class LLMDecisionMaker:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.cache = {}
        self.total_tokens_used = 0
        self.max_tokens_allowed = 40000
        self.training_calls = 0
        self.max_training_calls = 100
        self.last_valid_decision = None  # 存储最后一次有效的LLM决策
        logging.info("LLM client initialized with DeepSeek API")
        
    def can_call_llm(self, is_training: bool) -> bool:
        """检查是否允许调用LLM"""
        if self.total_tokens_used >= self.max_tokens_allowed:
            return False
            
        if is_training:
            if self.training_calls >= self.max_training_calls:
                return False
            self.training_calls += 1
            
        return True
        
    def format_state_for_llm(self, self_state: FullState, 
                           human_states: List[ObservableState],
                           other_robot_states: List[FullState]) -> str:
        """将机器人状态转换为LLM可理解的描述"""
        state_desc = {
            "robot": {
                "current_position": [round(self_state.px, 2), round(self_state.py, 2)],
                "current_velocity": [round(self_state.vx, 2), round(self_state.vy, 2)],
                "goal_position": [round(self_state.gx, 2), round(self_state.gy, 2)],
                "radius": round(self_state.radius, 2),
                "preferred_speed": round(self_state.v_pref, 2)
            },
            "humans": [
                {
                    "position": [round(h.px, 2), round(h.py, 2)],
                    "velocity": [round(h.vx, 2), round(h.vy, 2)],
                    "radius": round(h.radius, 2)
                } for h in human_states
            ],
            "other_robots": [
                {
                    "position": [round(r.px, 2), round(r.py, 2)],
                    "velocity": [round(r.vx, 2), round(r.vy, 2)],
                    "goal": [round(r.gx, 2), round(r.gy, 2)],
                    "radius": round(r.radius, 2)
                } for r in other_robot_states if r is not None
            ]
        }
        
        # 添加关键距离信息
        state_desc["analysis"] = self._analyze_distances(self_state, human_states, other_robot_states)
        return json.dumps(state_desc, indent=2)
        
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