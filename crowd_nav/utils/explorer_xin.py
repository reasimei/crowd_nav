import logging
import copy
import torch
from crowd_sim.envs.utils.info import *


class Explorer(object):
    def __init__(self, env, robot1, robot2, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot1 = robot1  # First robot
        self.robot2 = robot2  # Second robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

        # Set robots in the environment during initialization
        self.env.set_robot1(self.robot1)
        self.env.set_robot2(self.robot2)

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        # Set phase for each robot's policy
        self.robot1.policy.set_phase(phase)
        self.robot2.policy.set_phase(phase)

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
            # Set robots in the environment before resetting
            # (If already set in __init__, this may not be necessary)
            # self.env.set_robot1(self.robot1)
            # self.env.set_robot2(self.robot2)

            ob = self.env.reset(phase)  # ob is a list of observations, one for each robot
            done = False
            robot1_states = []
            robot1_actions = []
            robot1_rewards = []
            robot2_states = []
            robot2_actions = []
            robot2_rewards = []

            while not done:
                # Each robot acts based on its own observation
                action1 = self.robot1.act(ob[0])
                action2 = self.robot2.act(ob[1])
                actions = [action1, action2]

                ob, reward, done, info = self.env.step(actions)  # Pass list of actions to env.step()
                # Assuming shared reward for both robots
                robot1_states.append(self.robot1.policy.last_state)
                robot1_actions.append(action1)
                robot1_rewards.append(reward)  # Shared reward

                robot2_states.append(self.robot2.policy.last_state)
                robot2_actions.append(action2)
                robot2_rewards.append(reward)  # Shared reward

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            # Handle episode results
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
                    # Update memory for each robot
                    self.update_memory(robot1_states, robot1_actions, robot1_rewards, self.robot1, imitation_learning)
                    self.update_memory(robot2_states, robot2_actions, robot2_rewards, self.robot2, imitation_learning)

            # Calculate cumulative rewards (assuming shared reward)
            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot1.time_step * self.robot1.v_pref)
                                           * robot1_rewards[t] for t in range(len(robot1_rewards))]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))

        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot1.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, robot, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # Define the value of states in IL as cumulative discounted rewards
                state = self.target_policy.transform(state)
                value = sum([pow(self.gamma, max(t - i, 0) * robot.time_step * robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # Terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, robot.time_step * robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()

            value = torch.Tensor([value]).to(self.device)
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
