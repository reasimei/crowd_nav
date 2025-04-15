import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist, point_to_segment_dist2
import os


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot1 = None
        self.robot2 = None
        self.robot = [self.robot1, self.robot2]
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
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
        self.hose_length = None
        self.hose_stiffness = None
        self.hose_damping = None
        self.hose_segments = None
        self.hose_points = None
        self.hose_mass = None  # Add this line
        self.hose_points = []  # List of positions of hose segments
        self.hose_velocities = []  # List of velocities of hose segments
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
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        # Add hose parameters after config is set
        self.hose_length = self.config.getfloat('reward', 'hose_length')
        self.hose_penalty = self.config.getfloat('reward', 'hose_penalty')
        self.hose_collision_penalty = self.config.getfloat('reward', 'hose_collision_penalty')
        self.hose_stiffness = config.getfloat('hose', 'stiffness')
        self.hose_damping = config.getfloat('hose', 'damping')
        self.hose_segments = config.getint('hose', 'segments')
        self.hose_mass = config.getfloat('hose', 'mass')  # Add this line

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

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot1, robot2):
        self.robot1 = robot1
        self.robot2 = robot2
        self.robot1.set_env(self)
        self.robot2.set_env(self)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
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
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot1.policy.multiagent_training and self.robot2.policy.multiagent_training else 1)
        if not self.robot1.policy.multiagent_training or self.robot2.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot1.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            self.robot2.set(1, -self.circle_radius, 1, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot1.policy.multiagent_training and self.robot2.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot1, self.robot2] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot1.policy, 'action_values') and hasattr(self.robot2.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot1.policy, 'get_attention_weights') and hasattr(self.robot2.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot1.sensor == 'coordinates' and self.robot2.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot1.sensor == 'RGB' and self.robot2.sensor == 'RGB':
            raise NotImplementedError
        
        robot1_pos = np.array(self.robot1.get_position())
        robot2_pos = np.array(self.robot2.get_position())

        self.hose_points = []
        self.hose_velocities = []

        for i in range(self.hose_segments + 1):
            t = i / self.hose_segments
            point = (1 - t) * robot1_pos + t * robot2_pos
            self.hose_points.append(point)
            self.hose_velocities.append(np.array([0.0, 0.0]))  # Initial velocities are zero

        return ob
    
    def update_hose_dynamics(self):
        rest_length = self.hose_length / self.hose_segments
        new_positions = [np.array(self.robot1.get_position())]
        new_velocities = [self.hose_velocities[0]]  # Placeholder for consistency

        # Update internal segments
        for i in range(1, self.hose_segments):
            # Positions and velocities
            p_current = self.hose_points[i]
            v_current = self.hose_velocities[i]

            # Forces from adjacent segments
            # Spring force from previous segment
            delta_prev = p_current - self.hose_points[i - 1]
            dist_prev = np.linalg.norm(delta_prev)
            direction_prev = delta_prev / (dist_prev + 1e-6)
            stretch_prev = dist_prev - rest_length
            force_prev = -self.hose_stiffness * stretch_prev * direction_prev

            # Spring force from next segment
            delta_next = self.hose_points[i + 1] - p_current
            dist_next = np.linalg.norm(delta_next)
            direction_next = delta_next / (dist_next + 1e-6)
            stretch_next = dist_next - rest_length
            force_next = -self.hose_stiffness * stretch_next * direction_next

            # Damping force
            damping_force = -self.hose_damping * v_current

            # Total force
            total_force = force_prev + force_next + damping_force

            # Acceleration
            acceleration = total_force / self.hose_mass

            # Update velocity and position
            v_new = v_current + acceleration * self.time_step
            p_new = p_current + v_new * self.time_step

            new_velocities.append(v_new)
            new_positions.append(p_new)

        # Last point is robot2's position
        new_positions.append(np.array(self.robot2.get_position()))
        new_velocities.append(self.hose_velocities[-1])  # Placeholder

        self.hose_points = new_positions
        self.hose_velocities = new_velocities


    def onestep_lookahead(self, action1, action2):
        return self.step(action1, action2, update=False)
    

    
    def calculate_hose_forces(self):
        rest_length = self.hose_length / self.hose_segments

        # Force on robot1 from the first hose segment
        delta1 = self.hose_points[1] - self.hose_points[0]
        dist1 = np.linalg.norm(delta1)
        direction1 = delta1 / (dist1 + 1e-6)
        stretch1 = dist1 - rest_length
        v_rel1 = self.hose_velocities[1] - np.array([self.robot1.vx, self.robot1.vy])
        force_on_robot1 = self.hose_stiffness * stretch1 * direction1 - self.hose_damping * v_rel1

        # Force on robot2 from the last hose segment
        delta2 = self.hose_points[-1] - self.hose_points[-2]
        dist2 = np.linalg.norm(delta2)
        direction2 = delta2 / (dist2 + 1e-6)
        stretch2 = dist2 - rest_length
        v_rel2 = np.array([self.robot2.vx, self.robot2.vy]) - self.hose_velocities[-2]
        force_on_robot2 = -self.hose_stiffness * stretch2 * direction2 - self.hose_damping * v_rel2

        return force_on_robot1, force_on_robot2

    def step(self, action1, action2, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot1.visible and self.robot2.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px1 = human.px - self.robot1.px
            py1 = human.py - self.robot1.py
            px2 = human.px - self.robot2.px
            py2 = human.py - self.robot2.py 
            px3 = self.robot1.px - self.robot2.px
            py3 = self.robot1.py - self.robot2.py
            if self.robot1.kinematics == 'holonomic' and self.robot2.kinematics == 'holonomic':
                vx1 = human.vx - action1.vx
                vy1 = human.vy - action1.vy
                vx2 = human.vx - action2.vx
                vy2 = human.vy - action2.vy 
                vx3 = action1.vx - action2.vx
                vy3 = action1.vy - action2.vy
            else:
                vx1 = human.vx - action1.v * np.cos(action1.r + self.robot1.theta)
                vy1 = human.vy - action1.v * np.sin(action1.r + self.robot1.theta)
                vx2 = human.vx - action2.v * np.cos(action2.r + self.robot2.theta)
                vy2 = human.vy - action2.v * np.sin(action2.r + self.robot2.theta)
                vx3 = action1.v * np.cos(action1.r + self.robot1.theta) - action2.v * np.cos(action2.r + self.robot2.theta)
                vy3 = action1.v * np.sin(action1.r + self.robot1.theta) - action2.v * np.sin(action2.r + self.robot2.theta)
                #vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex1 = px1 + vx1 * self.time_step
            ex2 = px2 + vx2 * self.time_step
            ex3 = px3 + vx3 * self.time_step
            ey1 = py1 + vy1 * self.time_step
            ey2 = py2 + vy2 * self.time_step
            ey3 = py3 + vy3 * self.time_step
            # closest distance between boundaries of two agents
            closest_dist1 = point_to_segment_dist(px1, py1, ex1, ey1, 0, 0) - human.radius - self.robot1.radius
            closest_dist2 = point_to_segment_dist(px2, py2, ex2, ey2, 0, 0) - human.radius - self.robot2.radius
            closest_dist3 = point_to_segment_dist(px3, py3, ex3, ey3, 0, 0) - self.robot1.radius - self.robot2.radius
            closest_dist = min(closest_dist1, closest_dist2, closest_dist3)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # Compute the hose constraints and update reward
        # Get robot positions
        robot1_pos = np.array(self.robot1.get_position())
        robot2_pos = np.array(self.robot2.get_position())
        distance_between_robots = np.linalg.norm(robot1_pos - robot2_pos)

        # Initialize reward and info
        reward = 0
        info = ''

        # Check if the distance exceeds hose length
        if distance_between_robots > self.hose_length:
            reward += self.hose_penalty
            info += 'Exceed hose length; '

        # Check for collision between hose and humans
        hose_start = robot1_pos
        hose_end = robot2_pos
        for human in self.humans:
            human_pos = np.array(human.get_position())
            human_radius = human.radius

            # Compute the distance from human to hose
            dist = point_to_segment_dist2(hose_start, hose_end, human_pos)
            if dist < human_radius:
                reward += self.hose_collision_penalty
                info += 'Hose-human collision; '
                break  # Only penalize once per step

        
        # Update robots' velocities based on actions
        self.robot1.vx, self.robot1.vy = action1.vx, action1.vy
        self.robot2.vx, self.robot2.vy = action2.vx, action2.vy

        # Update hose dynamics and apply forces
        self.update_hose_dynamics()

        # Calculate hose forces on robots
        force_on_robot1, force_on_robot2 = self.calculate_hose_forces()
        self.robot1.apply_force(force_on_robot1)
        self.robot2.apply_force(force_on_robot2)

        # Update robot positions
        self.robot1.update_position()
        self.robot2.update_position()


        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position1 = np.array(self.robot1.compute_position(action1, self.time_step))
        end_position2 = np.array(self.robot2.compute_position(action2, self.time_step))
        reaching_goal1 = norm(end_position1 - np.array(self.robot1.get_goal_position())) < self.robot1.radius
        reaching_goal2 = norm(end_position2 - np.array(self.robot2.get_goal_position())) < self.robot2.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal1 and reaching_goal2:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot1.get_full_state(), self.robot2.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot1.policy, 'action_values') and hasattr(self.robot2.policy, 'action_values'):
                self.action_values.append(self.robot1.policy.action_values)
                self.action_values.append(self.robot2.policy.action_values)
            if hasattr(self.robot1.policy, 'get_attention_weights') and hasattr(self.robot2.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot1.policy.get_attention_weights())
                self.attention_weights.append(self.robot2.policy.get_attention_weights())

            # update all agents
            self.robot1.step(action1)
            self.robot2.step(action2)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            # Record states
            self.states.append([
                self.robot1.get_full_state(),
                self.robot2.get_full_state(),
                [human.get_full_state() for human in self.humans]
            ])
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot1.sensor == 'coordinates' and self.robot2.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB' and self.robot2.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot1.sensor == 'coordinates' and self.robot2.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB' and self.robot2.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info


    def render(self, mode='traj', output_file='/home/zzy/catkin_ws/src/crowd_nav/crowd_nav/result'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
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
            robot_positions = [
                [state[0].position for state in self.states],  # Positions of robot1
                [state[1].position for state in self.states]   # Positions of robot2
            ]

            human_positions = [
                [state[2][j].position for j in range(len(self.humans))]
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

            # Define base colors for robots
            robot_base_color_names = ['blue', 'cyan']  # Base colors for robot1 and robot2

            # Assign specific colors to human1, human3, and human4
            human_colors_dict = {
                0: 'orange',  # human1
                2: 'pink',    # human3
                3: 'red'      # human4
                # We can also specify colors for other humans if needed
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
                interval = max(1, int(math.ceil(1 / time_step)))
            else:
                interval = 1

            # Create lists to hold light and dark versions of colors for robots
            robot_light_colors = []
            robot_dark_colors = []
            for color_name in robot_base_color_names:
                light_color = adjust_color(color_name, amount=0.5, lighten=True)
                dark_color = adjust_color(color_name, amount=0.5, lighten=False)
                robot_light_colors.append(light_color)
                robot_dark_colors.append(dark_color)

            # Create lists to hold light and dark versions of colors for humans
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

            # Collect robots list
            robots = [self.robot1, self.robot2]

            # Start plotting
            for k in range(total_time_steps):
                t_norm = k / (total_time_steps - 1) if total_time_steps > 1 else 1.0

                # Interpolate robot colors over time from light to dark
                robot_colors = [
                    interpolate_color(robot_light_colors[i], robot_dark_colors[i], t_norm)
                    for i in range(2)
                ]

                # Interpolate human colors over time from light to dark
                human_colors = []
                for i in range(len(self.humans)):
                    light_color = human_light_colors[i]
                    dark_color = human_dark_colors[i]
                    adjusted_color = interpolate_color(light_color, dark_color, t_norm)
                    human_colors.append(adjusted_color)

                # Plot positions at every 0.1 seconds
                if k % interval == 0 or k == total_time_steps - 1:
                    # Plot robots
                    robot_circles = []
                    for i in range(2):
                        robot_circle = plt.Circle(robot_positions[i][k], robots[i].radius, facecolor=robot_colors[i],
                                                edgecolor='black', linewidth=1.0, zorder=3)
                        ax.add_patch(robot_circle)
                        robot_circles.append(robot_circle)

                    # Plot humans
                    humans = []
                    for i in range(len(self.humans)):
                        human_circle = plt.Circle(human_positions[k][i], self.humans[i].radius, facecolor=human_colors[i],
                                                edgecolor='black', linewidth=1.0, zorder=3)
                        ax.add_patch(human_circle)
                        humans.append(human_circle)

                    # Add time annotations
                    global_time = k * self.time_step * 0.5
                    agents = robot_circles + humans
                    times = [ax.text(agent.center[0] + x_offset, agent.center[1] + y_offset,
                                    '{:.1f}'.format(global_time),
                                    color='black', fontsize=10, zorder=4)
                            for agent in agents]

            # Plot running path lines for each robot
            for i in range(2):
                robot_path_x = [pos[0] for pos in robot_positions[i]]
                robot_path_y = [pos[1] for pos in robot_positions[i]]
                ax.plot(robot_path_x, robot_path_y, color=robot_dark_colors[i], linewidth=2, zorder=4)

            # Plot running path lines for each human
            for i in range(len(self.humans)):
                human_path_x = [pos[0] for pos in human_positions_over_time[i]]
                human_path_y = [pos[1] for pos in human_positions_over_time[i]]
                ax.plot(human_path_x, human_path_y, color=human_dark_colors[i], linewidth=2, zorder=4)

            # Set plot limits dynamically based on positions
            all_robot_positions = np.array([pos for positions in robot_positions for pos in positions])
            all_positions = np.concatenate([all_robot_positions] + [np.array(pos) for pos in human_positions_over_time])
            min_coords = np.min(all_positions, axis=0) - 1
            max_coords = np.max(all_positions, axis=0) + 1
            ax.set_xlim(min_coords[0], max_coords[0])
            ax.set_ylim(min_coords[1], max_coords[1])

            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Robot {i+1}',
                                        markerfacecolor=robot_dark_colors[i], markersize=10,
                                        markeredgecolor='black', markeredgewidth=1.0)
                            for i in range(2)] + \
                            [plt.Line2D([0], [0], marker='o', color='w', label=f'Human {i + 1}',
                                        markerfacecolor=human_dark_colors[i], markersize=10,
                                        markeredgecolor='black', markeredgewidth=1.0)
                            for i in range(len(self.humans))]
            ax.legend(handles=legend_elements, fontsize=12, loc='upper right')

            # Improve aesthetics
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_aspect('equal', 'box')
            plt.title('Robots and Pedestrian Trajectories Over Time', fontsize=18)
            plt.tight_layout()
            if not os.path.exists('./result'):
                os.makedirs('./result')
            plt.savefig("./result/test_009.png", dpi=300)
            #plt.show()



            
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

    def get_observation(self, robot_positions):
        # Construct observation based on the real-time positions from ROS
        # Update robot positions in the environment
        for i, r in enumerate(self.robot):
            r.px, r.py = robot_positions[i]
        # Generate observation for the policy
        ob = [human.get_observable_state() for human in self.humans]
        return ob
