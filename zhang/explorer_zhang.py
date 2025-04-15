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
        
        if args.policy == 'h_sarl' or args.policy == 'h_llm_sarl': 
            # 初始化 training_phase
            if not hasattr(self, 'training_phase'):
                self.training_phase = 'human_avoidance'
            
            # 设置所有机器人的 training_phase
            for robot in self.robots:
                if hasattr(robot.policy, 'training_phase'):
                    robot.policy.training_phase = self.training_phase
            self.training_phase = 'human_avoidance'
            phase_changed = False

        for robot in self.robots:
            robot.policy.set_phase(phase)
        # self.robot1.policy.set_phase(phase)
        # self.robot2.policy.set_phase(phase)
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
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            action = []
            actions = []
            rewards = []
            while not done:
                if robot.policy.name == 'H-SARL' or robot.policy.name == 'H-LLM-SARL': 
                    if not phase_changed and all(h.reached_destination() for h in self.env.humans):
                        self.training_phase = 'robot_avoidance'
                        phase_changed = True
                        # for robot in self.robots:
                        #     robot.policy.current_phase = current_phase
                action = []
                for robot in self.robots:
                    try:
                        # Get robot position
                        robot_pos = np.array([robot.px, robot.py])
                        if robot.policy.name == 'H-SARL' or robot.policy.name == 'H-LLM-SARL': 
                            # Modify action selection based on training phase and zone
                            if self.training_phase == 'human_avoidance':
                                # Focus on avoiding humans
                                robot_action = robot.act_avoid_humans(ob)
                            elif self.training_phase == 'robot_avoidance':
                                # Focus on avoiding other robots
                                robot_action = robot.act_avoid_robots(ob)
                            else:
                                # Normal action selection
                                robot_action = robot.act(ob)
                        else:
                            # Normal action selection
                            robot_action = robot.act(ob)    
                        action.append(robot_action)
                    except Exception as e:
                        logging.error(f"Error in robot action selection: {e}")
                        # Default safe action
                        if robot.kinematics == 'holonomic':
                            action.append(ActionXY(0, 0))
                        else:
                            action.append(ActionRot(0, 0))

                ob, reward, done, info = self.env.step1(action)
                # ob, reward, done, info = self.env.step(action1, action2)
                #ob2, reward2, done2, info2 = self.env.step(action2)
                for robot in self.robots:
                    states.append(robot.policy.last_state)
                # states.append(self.robot1.policy.last_state)
                # states.append(self.robot2.policy.last_state)
                actions.append(action)
                #actions.append(action2)
                for robot in self.robots: #reward for each robot
                         rewards.append(reward)
                # rewards.append(reward)
           

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
                # if isinstance(info2, Danger):
                #     too_close += 1
                #     min_dist.append(info2.min_dist)

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

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * sum(robot.time_step * robot.v_pref for robot in self.robots))
                                           * reward for t, reward in enumerate(rewards)]))
            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot1.time_step * self.robot1.v_pref + t * self.robot2.time_step * self.robot2.v_pref)
            #                                * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robots[0].time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        if robot.policy.name == 'H-SARL' or robot.policy.name == 'H-LLM-SARL': # 区分分层强化学习和原来
            # 新增：若所有 human 均抵达目标，则切换到 robot_avoidance，否则维持 human_avoidance
            if phase == 'train':
                all_humans_done = all(h.reached_destination() for h in self.env.humans)
                new_phase = 'robot_avoidance' if all_humans_done else 'human_avoidance'
                self.training_phase = new_phase
                # 将训练阶段传递给所有机器人
                for robot in self.robots:
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
