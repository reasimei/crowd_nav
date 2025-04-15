# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
import gym
from gym import spaces
from llama_cpp import Llama
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List, Tuple

# ==================== 配置参数 ====================
MODEL_PATH = "./deepseek-7b-robotics-finetuned.Q4_K_M.gguf"
NUM_ROBOTS = 3
BASE_OBS_DIM = 8  # 基础观测维度：x, y, vx, vy, 最近障碍物距离，角度，相对目标方向，速度限制
LLM_ACTION_DIM = 2  # LLM建议的维度（v, theta）

# ==================== 模型包装类（改进版） ====================
class LLMAdvisor:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=8,
            n_gpu_layers=20
        )
        
        self.system_prompt = """..."""  # 保持原有提示模板

    def get_llm_actions(self, state: Dict) -> np.ndarray:
        prompt = self.system_prompt + json.dumps(state, indent=2)
        output = self.llm(prompt=prompt, max_tokens=512, temperature=0.3)
        
        try:
            response = json.loads(output["choices"][0]["text"])
            actions = np.zeros((NUM_ROBOTS, LLM_ACTION_DIM))
            for rec in response["recommendations"]:
                actions[rec["id"]] = [rec["v"], rec["theta"]]
            return actions
        except:
            return np.zeros((NUM_ROBOTS, LLM_ACTION_DIM))

# ==================== 强化学习环境（级联架构） ====================
class CascadeRLEnv(gym.Env):
    def __init__(self):
        super(CascadeRLEnv, self).__init__()
        
        # 观测空间 = 基础状态 + LLM建议动作
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(NUM_ROBOTS * (BASE_OBS_DIM + LLM_ACTION_DIM),)
        )
        
        # 动作空间：对LLM建议的修正量（Δv, Δtheta）
        self.action_space = spaces.Box(
            low=np.array([-0.2, -15] * NUM_ROBOTS),  # 允许最大±0.2 m/s和±15度修正
            high=np.array([0.2, 15] * NUM_ROBOTS),
            dtype=np.float32
        )
        
        # 初始化组件
        self.llm_advisor = LLMAdvisor(MODEL_PATH)
        self.current_state = None
        self.current_llm_actions = None

    def _get_obs(self) -> np.ndarray:
        """构建包含LLM建议的观测向量"""
        # 假设基础观测已从仿真环境获取，形状为(NUM_ROBOTS, BASE_OBS_DIM)
        base_obs = np.random.randn(NUM_ROBOTS, BASE_OBS_DIM)  # 示例数据
        
        # 将LLM建议拼接到观测中
        llm_actions = self.current_llm_actions.reshape(NUM_ROBOTS, LLM_ACTION_DIM)
        full_obs = np.concatenate([base_obs, llm_actions], axis=1)
        return full_obs.flatten()

    def step(self, rl_actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行级联决策流程"""
        # 1. 获取LLM基础建议
        if self.current_llm_actions is None:
            self.current_llm_actions = self.llm_advisor.get_llm_actions(self._get_state_dict())
        
        # 2. 将RL动作转换为修正量
        rl_delta = rl_actions.reshape(NUM_ROBOTS, 2)
        
        # 3. 合成最终动作：[v_llm + Δv, theta_llm + Δtheta]
        final_actions = self.current_llm_actions + rl_delta
        
        # 4. 执行动作并获取新状态（需实现具体仿真逻辑）
        next_state, reward, done, info = self._execute_actions(final_actions)
        
        # 5. 准备下一步的LLM建议
        self.current_llm_actions = self.llm_advisor.get_llm_actions(self._get_state_dict())
        
        return next_state, reward, done, info

    def reset(self):
        """重置环境并获取初始观测"""
        self.current_state = self._init_simulation()
        self.current_llm_actions = self.llm_advisor.get_llm_actions(self._get_state_dict())
        return self._get_obs()

    def _execute_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行物理仿真（示例占位符）"""
        # 此处应集成真实机器人动力学模型
        # 返回：观测，奖励，终止标志，信息
        reward = self._calculate_reward(actions)
        return self._get_obs(), reward, False, {}

    def _calculate_reward(self, actions: np.ndarray) -> float:
        """奖励函数设计"""
        # 示例奖励项：
        # 1. 目标接近奖励
        # 2. 碰撞惩罚
        # 3. 动作平滑惩罚（加速度限制）
        return 1.0

    def _get_state_dict(self) -> Dict:
        """将内部状态转换为LLM所需的字典格式（示例）"""
        return {
            "robots": [{"id": i, "x": 0.0, "y": 0.0, "v": 0.0, "theta": 0.0} 
                      for i in range(NUM_ROBOTS)],
            "obstacles": []
        }

# ==================== 训练流程 ====================
def train_cascade_rl():
    env = DummyVecEnv([lambda: CascadeRLEnv()])
    
    # 策略网络需适应扩展后的观测空间
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 增大网络容量
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    model.learn(total_timesteps=2e5)
    model.save("cascade_llm_rl")

# ==================== 执行优化 ====================
if __name__ == "__main__":
    # 初始化环境测试
    env = CascadeRLEnv()
    obs = env.reset()
    print(f"初始观测维度: {obs.shape}")  # 应为 NUM_ROBOTS*(BASE_OBS_DIM+LLM_ACTION_DIM)
    
    # 示例RL动作
    dummy_actions = np.array([0.1, 5] * NUM_ROBOTS)  # 对每个机器人建议+0.1m/s和+5度修正
    next_obs, reward, _, _ = env.step(dummy_actions)
    print(f"优化后状态示例:\n{next_obs[:10]}...")
    
    # 启动训练
    train_cascade_rl()