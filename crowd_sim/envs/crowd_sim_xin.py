import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import copy
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for multiple robots and humans.
        Robots are connected by a flexible hose.
        Humans are controlled by an unknown and fixed policy.
        Robots are controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot1 = None  # First robot
        self.robot2 = None  # Second robot
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.hose_penalty = None  # Penalty for violating hose constraints
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.hose_penalty = config.getfloat('reward', 'hose_penalty')  # New penalty
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('Human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot1(self, robot):
        self.robot1 = robot

    def set_robot2(self, robot):
        self.robot2 = robot

    def generate_robot(self, robot):
        """
        Randomly assign positions to a robot within certain bounds.
        """
        # Randomly assign positions within certain bounds
        px = np.random.uniform(-self.square_width / 2, self.square_width / 2)
        py = np.random.uniform(-self.square_width / 2, self.square_width / 2)
        gx = np.random.uniform(-self.square_width / 2, self.square_width / 2)
        gy = np.random.uniform(-self.square_width / 2, self.square_width / 2)
        robot.set(px, py, gx, gy, 0, 0, 0)
        return robot

    def generate_random_robot_position(self):
        """
        Generate initial and goal positions for robots.
        Robots are connected by a flexible hose of length 0.6 meters.
        """
        # Generate positions for robot1
        self.robot1 = self.generate_robot(self.robot1)
        # Generate positions for robot2, ensuring it's within hose length
        while True:
            self.robot2 = self.generate_robot(self.robot2)
            distance = norm((self.robot1.px - self.robot2.px, self.robot1.py - self.robot2.py))
            if distance <= 0.6:
                break

    def onestep_lookahead(self, action, robot_index=0):
        # Create a list of actions where actions for other robots are default or previous actions
        actions = []
        for i in range(2):  # Assuming there are two robots
            if i == robot_index:
                actions.append(action)
            else:
                # Use a default action or the previous action for the other robot
                default_action = ActionXY(0, 0)
                actions.append(default_action)
        return self.step(actions, update=False)

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for humans to reach their goals.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return: List of times when each human reaches their goal.
        """
        # Check if both robots have reached their destinations
        if not all(robot.reached_destination() for robot in [self.robot1, self.robot2]):
            raise ValueError('Episode is not done yet')

        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)

        # Add robots to the simulator
        sim.addAgent(self.robot1.get_position(), *params, self.robot1.radius, self.robot1.v_pref,
                    self.robot1.get_velocity())
        sim.addAgent(self.robot2.get_position(), *params, self.robot2.radius, self.robot2.v_pref,
                    self.robot2.get_velocity())

        # Add humans to the simulator
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        total_agents = 2 + len(self.humans)  # Number of agents in the simulator

        while not all(self.human_times):
            # Update preferred velocities for all agents
            for i, agent in enumerate([self.robot1, self.robot2] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))

            sim.doStep()
            self.global_time += self.time_step

            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
                break

            # Update positions of robots and humans
            self.robot1.set_position(sim.getAgentPosition(0))
            self.robot2.set_position(sim.getAgentPosition(1))

            for i, human in enumerate(self.humans):
                idx = i + 2  # Index in the simulator (after robots)
                human.set_position(sim.getAgentPosition(idx))

                # Check if human has reached the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # Save the state for visualization
            self.states.append([self.robot1.get_full_state(), self.robot2.get_full_state()] +
                            [[human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    
    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robots and humans
        """
        if self.robot1 is None or self.robot2 is None:
            raise AttributeError('Robots have to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * self.human_num
        self.generate_random_robot_position()
        if self.case_counter[phase] >= 0:
            if phase in ['train', 'val']:
                human_num = self.human_num
                self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
            else:
                self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        # Set time steps for all agents
        for agent in [self.robot1, self.robot2] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot1.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot1.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # Get current observation
        if self.robot1.sensor == 'coordinates':
            ob1 = [human.get_observable_state() for human in self.humans]
            ob2 = [human.get_observable_state() for human in self.humans]
            ob = [ob1, ob2]
        elif self.robot1.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def step(self, actions, update=True):
        """
        Compute actions for all agents, detect collision, update environment, and return (ob, reward, done, info)

        :param actions: List of actions for each robot
        """
        human_actions = []
        for human in self.humans:
            # Observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            ob += [robot.get_observable_state() for robot in [self.robot1, self.robot2] if robot.visible]
            human_actions.append(human.act(ob))

        # Collision detection
        collision = False
        hose_violation = False
        dmin = float('inf')
        robots = [self.robot1, self.robot2]
        for i, robot in enumerate(robots):
            action = actions[i]
            # Check collision with humans
            for j, human in enumerate(self.humans):
                px = human.px - robot.px
                py = human.py - robot.py
                if robot.kinematics == 'holonomic':
                    vx = human.vx - action.vx
                    vy = human.vy - action.vy
                else:
                    vx = human.vx - action.v * np.cos(action.r + robot.theta)
                    vy = human.vy - action.v * np.sin(action.r + robot.theta)
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # Closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot.radius
                if closest_dist < 0:
                    collision = True
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist
            if collision:
                break

        # Hose constraint enforcement
        distance = norm((self.robot1.px - self.robot2.px, self.robot1.py - self.robot2.py))
        if distance > 0.6:
            hose_violation = True

        # Check if robots reach their goals
        reaching_goals = [norm(np.array(robot.compute_position(actions[i], self.time_step)) - np.array(robot.get_goal_position())) < robot.radius for i, robot in enumerate(robots)]

        done = False
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif all(reaching_goals):
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif hose_violation:
            reward = -self.hose_penalty  # Negative reward for hose violation
            done = False
            info = ConstraintViolation()
        elif dmin < self.discomfort_dist:
            # Only penalize agent for getting too close
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        # Update environment
        self.states.append([self.robot1.get_full_state(), self.robot2.get_full_state()] +
                           [[human.get_full_state() for human in self.humans]])
        if hasattr(self.robot1.policy, 'action_values'):
            self.action_values.append([self.robot1.policy.action_values, self.robot2.policy.action_values])
        if hasattr(self.robot1.policy, 'get_attention_weights'):
            self.attention_weights.append([self.robot1.policy.get_attention_weights(), self.robot2.policy.get_attention_weights()])

        # Update all agents
        self.robot1.step(actions[0])
        self.robot2.step(actions[1])
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step
        for i, human in enumerate(self.humans):
            # Only record the first time the human reaches the goal
            if self.human_times[i] == 0 and human.reached_destination():
                self.human_times[i] = self.global_time

        # Compute the observation
        if self.robot1.sensor == 'coordinates':
            ob1 = [human.get_observable_state() for human in self.humans]
            ob2 = [human.get_observable_state() for human in self.humans]
            ob = [ob1, ob2]
        elif self.robot1.sensor == 'RGB':
            raise NotImplementedError

        return ob, reward, done, info

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human positions according to certain rule.

        :param human_num:
        :param rule:
        :return:
        """
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot1, self.robot2] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot1, self.robot2] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot1, self.robot2] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human


    def set_robots(self, robots):
        if not isinstance(robots, list):
            raise TypeError('Robots should be provided as a list of Robot instances.')
        self.robots = robots

    # Add any additional methods needed to support the new functionality


    def render(self, mode='traj', output_file='/home/zzy/catkin_ws/src/crowd_nav/crowd_nav/result'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/share/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
            #plt.savefig("./result/test_000.png")

        elif mode == 'traj':
            import matplotlib.pyplot as plt
            from matplotlib.colors import to_rgba
            import matplotlib.colors as mc
            import numpy as np
            import math

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.tick_params(labelsize=16)
            ax.set_xlabel('x (m)', fontsize=16)
            ax.set_ylabel('y (m)', fontsize=16)

            # Collect robot and human positions
            total_time_steps = len(self.states)
            robot_positions = [state[0].position for state in self.states]
            human_positions = [
                [state[1][j].position for j in range(len(self.humans))]
                for state in self.states
            ]

            # Function to lighten or darken colors
            def adjust_color(color, amount=0.5, lighten=True):
                try:
                    c = mc.cnames[color]
                except:
                    c = color
                c = mc.to_rgb(c)
                c = np.array(c)
                if lighten:
                    c = c + (1 - c) * amount  # Blend with white
                else:
                    c = c * (1 - amount)      # Blend with black
                return tuple(np.clip(c, 0, 1))

            # Function to interpolate colors over time from light to dark
            def interpolate_color(color_start, color_end, t):
                return tuple(np.array(color_start) * (1 - t) + np.array(color_end) * t)

            # Define base colors
            robot_base_color_name = 'blue'  # Base color name for robot

            # Assign specific colors to human1, human3, and human4
            human_colors_dict = {
                0: 'orange',  # human1
                2: 'pink',    # human3
                3: 'red'      # human4
            }

            # For other humans, get distinct colors from 'tab20' colormap
            default_colors = plt.cm.get_cmap('tab20').colors
            human_base_colors = []
            for i in range(len(self.humans)):
                if i in human_colors_dict:
                    base_color = human_colors_dict[i]
                else:
                    base_color = default_colors[i % len(default_colors)]
                human_base_colors.append(base_color)

            x_offset = 0.1
            y_offset = 0.1

            # Plot trajectory points every 0.1 seconds
            # Determine the interval based on time step
            time_step = self.time_step
            if time_step > 0:
                interval = max(1, int(math.ceil(0.5 / time_step)))
            else:
                interval = 1

            # Create lists to hold light and dark versions of colors
            robot_light_color = adjust_color(robot_base_color_name, amount=0.5, lighten=True)
            robot_dark_color = adjust_color(robot_base_color_name, amount=0.5, lighten=False)

            human_light_colors = []
            human_dark_colors = []
            for i in range(len(self.humans)):
                base_color = human_base_colors[i]
                # Convert base_color to RGB tuple if it's a color name
                if isinstance(base_color, str):
                    base_color_rgb = mc.to_rgb(base_color)
                else:
                    base_color_rgb = base_color
                light_color = adjust_color(base_color_rgb, amount=0.5, lighten=True)
                dark_color = adjust_color(base_color_rgb, amount=0.5, lighten=False)
                human_light_colors.append(light_color)
                human_dark_colors.append(dark_color)

            # Collect positions over time for each human
            human_positions_over_time = [[] for _ in range(len(self.humans))]
            for t in range(total_time_steps):
                for i in range(len(self.humans)):
                    human_positions_over_time[i].append(human_positions[t][i])

            # Start plotting
            for k in range(total_time_steps):
                t_norm = k / (total_time_steps - 1) if total_time_steps > 1 else 1.0

                # Interpolate robot color over time from light to dark
                robot_color = interpolate_color(robot_light_color, robot_dark_color, t_norm)

                # Interpolate human colors over time from light to dark
                human_colors = []
                for i in range(len(self.humans)):
                    light_color = human_light_colors[i]
                    dark_color = human_dark_colors[i]
                    adjusted_color = interpolate_color(light_color, dark_color, t_norm)
                    human_colors.append(adjusted_color)

                # Plot positions at every 0.1 seconds
                if k % interval == 0 or k == total_time_steps - 1:
                    # Plot robot
                    robot_circle = plt.Circle(robot_positions[k], self.robot.radius, facecolor=robot_color,
                                            edgecolor='black', linewidth=1.0, zorder=3)
                    ax.add_patch(robot_circle)

                    # Plot humans
                    humans = []
                    for i in range(len(self.humans)):
                        human_circle = plt.Circle(human_positions[k][i], self.humans[i].radius, facecolor=human_colors[i],
                                                edgecolor='black', linewidth=1.0, zorder=3)
                        ax.add_patch(human_circle)
                        humans.append(human_circle)

                    # Add time annotations
                    global_time = k * self.time_step
                    agents = [robot_circle] + humans
                    times = [ax.text(agent.center[0] + x_offset, agent.center[1] + y_offset,
                                    '{:.1f}'.format(global_time),
                                    color='black', fontsize=10, zorder=4)
                            for agent in agents]

            # Plot running path lines for robot
            robot_path_x = [pos[0] for pos in robot_positions]
            robot_path_y = [pos[1] for pos in robot_positions]
            ax.plot(robot_path_x, robot_path_y, color=robot_dark_color, linewidth=2, zorder=4)

            # Plot running path lines for each human
            for i in range(len(self.humans)):
                human_path_x = [pos[0] for pos in human_positions_over_time[i]]
                human_path_y = [pos[1] for pos in human_positions_over_time[i]]
                ax.plot(human_path_x, human_path_y, color=human_dark_colors[i], linewidth=2, zorder=4)

            # Set plot limits dynamically based on positions
            all_positions = np.concatenate([robot_positions] + [np.array(pos) for pos in human_positions_over_time])
            min_coords = np.min(all_positions, axis=0) - 1
            max_coords = np.max(all_positions, axis=0) + 1
            ax.set_xlim(min_coords[0], max_coords[0])
            ax.set_ylim(min_coords[1], max_coords[1])

            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Robot',
                                        markerfacecolor=robot_dark_color, markersize=10,
                                        markeredgecolor='black', markeredgewidth=1.0)] + \
                            [plt.Line2D([0], [0], marker='o', color='w', label=f'Human {i + 1}',
                                        markerfacecolor=human_dark_colors[i], markersize=10,
                                        markeredgecolor='black', markeredgewidth=1.0)
                            for i in range(len(self.humans))]
            ax.legend(handles=legend_elements, fontsize=12, loc='upper right')

            # Improve aesthetics
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_aspect('equal', 'box')
            plt.title('Robot and Pedestrian Trajectories Over Time', fontsize=18)
            plt.tight_layout()
            plt.savefig("./result/test_020.png", dpi=300)
            plt.show()



            
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0
            if not update:
                # Make copies of the robots and humans to simulate the next state
                robots = [copy.deepcopy(self.robot1), copy.deepcopy(self.robot2)]
                humans = [copy.deepcopy(human) for human in self.humans]
            else:
                robots = [self.robot1, self.robot2]
                humans = self.humans

            # The rest of the code remains mostly the same, but use 'robots' and 'humans' instead of self.robot1, self.robot2, and self.humans
            # Collision detection, hose constraint enforcement, and reward calculation will use the copied agents if update=False

            # Update all agents if update=True
            if update:
                self.robot1.step(actions[0])
                self.robot2.step(actions[1])
                for i, human_action in enumerate(human_actions):
                    self.humans[i].step(human_action)
                self.global_time += self.time_step
                for i, human in enumerate(self.humans):
                    # Only record the first time the human reaches the goal
                    if self.human_times[i] == 0 and human.reached_destination():
                        self.human_times[i] = self.global_time

                # Update environment states, action values, attention weights, etc.

            # Compute the observation
            if self.robot1.sensor == 'coordinates':
                ob1 = [human.get_observable_state() for human in humans]
                ob2 = [human.get_observable_state() for human in humans]
                ob = [ob1, ob2]
            elif self.robot1.sensor == 'RGB':
                raise NotImplementedError
        
            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()
                #plt.savefig("./result/test_002.png")

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save("aaa.gif", output_file, writer=writer)
            else:
                plt.show()
                #plt.savefig("./result/test_003.png")
                
        else:
            raise NotImplementedError
