import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
import numpy as np
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_nav.parser import args

class Explorer(object):
    def __init__(self, env, robots, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robots = robots
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        
        # Training phase parameters
        self.training_phase = 'human_avoidance'
        self.phase_success_threshold = 0.9
        self.phase_success_window = 100  # Number of episodes to average success rate
        self.success_history = []

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # def update_training_phase(self, success):
    #     """
    #     Update training phase based on recent performance
    #     """
    #     self.success_history.append(1 if success else 0)
    #     if len(self.success_history) > self.phase_success_window:
    #         self.success_history.pop(0)
        
    #     # Calculate success rate over recent episodes
    #     if len(self.success_history) == self.phase_success_window:
    #         success_rate = sum(self.success_history) / len(self.success_history)
            
    #         if self.training_phase == 'human_avoidance' and success_rate >= self.phase_success_threshold:
    #             self.training_phase = 'robot_avoidance'
    #             logging.info('Switching to robot avoidance training phase')
    #             self.success_history.clear()  # Reset history for new phase


    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, print_failure=False):
        phase_changed = False
        
        # 确保所有机器人初始阶段一致
        if args.policy == 'h_sarl' or args.policy == 'h_llm_sarl': 
            # 初始化训练阶段
            self.training_phase = 'human_avoidance'
            # 设置所有机器人的训练阶段
            for robot in self.robots:
                if hasattr(robot, 'set_training_phase'):
                    robot.set_training_phase('human_avoidance')
                if hasattr(robot.policy, 'set_training_phase'):
                    robot.policy.set_training_phase('human_avoidance')

        # 设置所有机器人的阶段
        for robot in self.robots:
            robot.policy.set_phase(phase)
        
        # 统计变量
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        
        # 运行k个回合
        for i in range(k):
            # 重置环境
            ob = self.env.reset(phase)
            done = False
            states = []
            action = []
            actions = []
            rewards = []
            ep_reward = 0  # 累计奖励记录变量
            
            # 单个回合循环
            while not done:
                # 人类避障/机器人协调训练阶段切换
                if (args.policy == 'h_sarl' or args.policy == 'h_llm_sarl'):
                    # 如果所有人类都到达目标，切换到机器人协调阶段
                    if not phase_changed and all(h.reached_destination() for h in self.env.humans):
                        self.training_phase = 'robot_avoidance'
                        phase_changed = True
                        # 传递新阶段给所有机器人
                        for robot in self.robots:
                            if hasattr(robot, 'set_training_phase'):
                                robot.set_training_phase('robot_avoidance')
                            if hasattr(robot.policy, 'set_training_phase'):
                                robot.policy.set_training_phase('robot_avoidance')
                
                # 清空动作
                action = []
                
                # 每个机器人计算动作
                for robot in self.robots:
                    try:
                        # 根据当前训练阶段选择适当的行为函数
                        if (args.policy == 'h_sarl' or args.policy == 'h_llm_sarl'):
                            if self.training_phase == 'human_avoidance':
                                robot_action = robot.act_avoid_humans(ob)
                            elif self.training_phase == 'robot_avoidance':
                                robot_action = robot.act_avoid_robots(ob)
                            else:
                                robot_action = robot.act(ob)
                        else:
                            robot_action = robot.act(ob)
                            
                        # 确保动作有效
                        if robot_action is None:
                            logging.warning(f"Robot {robot.robot_index} returned None action, using default")
                            if robot.kinematics == 'holonomic':
                                robot_action = ActionXY(0.3, 0)
                            else:
                                robot_action = ActionRot(0, 0.3)
                                
                        action.append(robot_action)
                        
                        # 调试日志
                        if args.debug and i % 5 == 0:
                            logging.debug(f"Robot {robot.robot_index} action: {robot_action}")
                            
                    except Exception as e:
                        logging.error(f"Error in robot action selection: {e}")
                        # 提供默认动作
                        if robot.kinematics == 'holonomic':
                            action.append(ActionXY(0.3, 0))
                        else:
                            action.append(ActionRot(0, 0.3))
                
                # 执行动作
                ob, reward, done, info = self.env.step1(action)
                ep_reward += reward  # 记录回合中的奖励
                
                # 保存状态和动作
                for robot in self.robots:
                    if hasattr(robot.policy, 'last_state'):
                        states.append(robot.policy.last_state)
                
                actions.append(action)
                for _ in self.robots:
                    rewards.append(reward)
                
                # 记录危险接近
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
            
            # 回合结束处理
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')
            
            # 更新经验池
            if update_memory and len(states) > 0:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    self.update_memory(states, actions, rewards, imitation_learning)
            
            # 计算折扣奖励
            cum_reward = sum([pow(self.gamma, t * sum(robot.time_step * robot.v_pref for robot in self.robots))
                             * reward for t, reward in enumerate(rewards)])
            cumulative_rewards.append(cum_reward)
            
            # 每5个回合打印进度
            if (i+1) % 5 == 0:
                logging.info(f"Episode {i+1}/{k}, reward so far: {ep_reward:.4f}")
        
        # 计算统计信息
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        
        # 打印回合统计
        extra_info = '' if episode is None else f'in episode {episode} '
        logging.info(f'{phase.upper()} {extra_info}has success rate: {success_rate:.2f}, '
                     f'collision rate: {collision_rate:.2f}, nav time: {avg_nav_time:.2f}, '
                     f'total reward: {average(cumulative_rewards):.4f}')
        
        # 打印测试阶段更多信息
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robots[0].time_step
            logging.info(f'Frequency of being in danger: {too_close/num_step:.2f} and '
                         f'average min separate distance in danger: {average(min_dist):.2f}')
        
        # 打印失败案例
        if print_failure:
            logging.info(f'Collision cases: {" ".join([str(x) for x in collision_cases])}')
            logging.info(f'Timeout cases: {" ".join([str(x) for x in timeout_cases])}')
        
        # 训练阶段切换逻辑（在回合结束时）
        if (args.policy == 'h_sarl' or args.policy == 'h_llm_sarl') and phase == 'train':
            # 如果所有人类已经到达目标，切换到机器人协调阶段
            all_humans_done = all(h.reached_destination() for h in self.env.humans)
            new_phase = 'robot_avoidance' if all_humans_done else 'human_avoidance'
            
            # 更新所有机器人的训练阶段
            self.training_phase = new_phase
            for robot in self.robots:
                if hasattr(robot, 'set_training_phase'):
                    robot.set_training_phase(new_phase)
                if hasattr(robot.policy, 'set_training_phase'):
                    robot.policy.set_training_phase(new_phase)
                
            logging.info(f'Training phase switched to: {new_phase}')

    
    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            all_time_step = 0
            all_v_pref = 0
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                for robot in self.robots:
                    all_time_step += robot.time_step
                    all_v_pref += robot.v_pref
                value = sum([pow(self.gamma, max(t - i, 0) * all_time_step * all_v_pref) * reward * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                # value = sum([pow(self.gamma, max(t - i, 0) * (self.robot1.time_step + self.robot2.time_step) * (self.robot1.v_pref + self.robot2.v_pref)) * reward
                #              * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, sum(robot.time_step * robot.v_pref for robot in self.robots))
                    # gamma_bar = pow(self.gamma, self.robot1.time_step * self.robot1.v_pref + self.robot2.time_step * self.robot2.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
