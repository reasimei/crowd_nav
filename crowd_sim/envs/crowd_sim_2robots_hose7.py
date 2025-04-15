import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist, point_to_segment_dist2, point_to_hose_curve, hose_model
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
        # self.robot1 = None
        # self.robot2 = None
        # self.robot = [self.robot1, self.robot2]
        self.robots = [None]
        self.humans = None
        self.hose = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.hose_penalty = None
        self.hose_collision_penalty = None
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
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        # Add hose parameters after config is set
        self.hose_length = self.config.getfloat('hose', 'hose_length')
        self.hose_thickness = self.config.getfloat('hose', 'hose_thickness')
        self.hose_mass = self.config.getfloat('hose', 'mass')
        self.hose_stiffness = self.config.getfloat('hose', 'stiffness')
        self.hose_damping = self.config.getfloat('hose', 'damping')
        self.hose_penalty = self.config.getfloat('reward', 'hose_penalty')
        self.hose_collision_penalty = self.config.getfloat('reward', 'hose_collision_penalty')

        self.robot_num = self.config.getint('sim', 'robot_num')
        self.robots = [None]*self.robot_num

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
        logging.info('robot number: {}'.format(self.robot_num))
        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        
    def set_robots(self, robots):
        for i in range(len(robots)):
            self.robots[i] = robots[i]
            self.robots[i].set_env(self)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate random human position using existing functions and modify self.humans directly
        """
        if rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        else:
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())

        # After generating positions, check and regenerate if there's conflict with robots or hoses
        for i in range(len(self.humans)):
            while True:
                position = np.array(self.humans[i].get_position())
                conflict = False

                # Check distance from all robots and their hoses
                for j, robot in enumerate(self.robots):
                    if j % 2 == 0 and j + 1 < len(self.robots):
                        # Check distance from hose segments between robot pairs
                        robot1_pos = np.array(robot.get_position())
                        robot2_pos = np.array(self.robots[j + 1].get_position())
                        
                        min_dist_to_hose = self.point_to_hose_min_distance(position, robot1_pos, robot2_pos)
                        if min_dist_to_hose < self.humans[i].radius + self.hose_thickness + 0.3:
                            conflict = True
                            break

                    # Check distance from robot
                    robot_pos = np.array(robot.get_position())
                    min_dist = np.linalg.norm(position - robot_pos)
                    if min_dist < self.humans[i].radius + robot.radius + 0.3:
                        conflict = True
                        break

                if not conflict:
                    break
                
                # If conflict exists, regenerate this human's position
                if rule == 'circle_crossing':
                    angle = np.random.random() * np.pi * 2
                    radius = self.circle_radius + np.random.random() * 0.5
                    self.humans[i].set(radius * np.cos(angle), radius * np.sin(angle), 
                                    -radius * np.cos(angle), -radius * np.sin(angle), 0, 0, 0)
                else:
                    self.humans[i].set(np.random.random() * self.square_width - self.square_width / 2,
                                    np.random.random() * self.square_width - self.square_width / 2,
                                    np.random.random() * self.square_width - self.square_width / 2,
                                    np.random.random() * self.square_width - self.square_width / 2,
                                    0, 0, 0)

    def point_to_hose_min_distance(self, point, robot1_pos, robot2_pos):
        """
        Calculate minimum distance from a point to the hose curve between two robots
        """
        # Get hose curve points
        x, y = hose_model(robot1_pos, robot2_pos, self.hose_length)
        hose_points = np.column_stack((x, y))
        
        # Calculate minimum distance to all segments of the hose
        min_dist = float('inf')
        for i in range(len(hose_points) - 1):
            segment_start = hose_points[i]
            segment_end = hose_points[i + 1]
            dist = point_to_segment_dist(segment_start[0], segment_start[1],
                                         segment_end[0], segment_end[1],
                                         point[0], point[1])
            min_dist = min(min_dist, dist)
        
        return min_dist

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
            # for agent in [self.robot1, self.robot2] + self.humans:
            for agent in self.robots + self.humans:
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
            for agent in self.robots + self.humans:
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
            self.states.append([[robot.get_full_state() for robot in self.robots], [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robots is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if all(robot.policy.multiagent_training for robot in self.robots) else 1)
        
        if any(not robot.policy.multiagent_training for robot in self.robots) or any(robot.policy.multiagent_training for robot in self.robots):
        # if not self.robot1.policy.multiagent_training or self.robot2.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            for i, robot in enumerate(self.robots):
                group_num = i // 8  # Determine which group of 8 this robot belongs to
                position_in_group = i % 8  # Position within the group (0-7)
                radius = self.circle_radius - group_num * 2  # Decrease radius for each group
                
                # Determine position based on position in group
                if position_in_group < 2:  # Bottom robots (0,1)
                    px = 2 * (position_in_group - 0.5)  # -1, 1
                    py = -radius
                    gx = px
                    gy = radius
                    theta = np.pi / 2  # Moving upward
                
                elif position_in_group < 4:  # Right robots (2,3)
                    px = radius
                    py = 2 * (position_in_group - 2.5)  # -1, 1
                    gx = -radius
                    gy = py
                    theta = np.pi  # Moving leftward
                
                elif position_in_group < 6:  # Top robots (4,5)
                    px = 2 * (position_in_group - 4.5)  # -1, 1
                    py = radius
                    gx = px
                    gy = -radius
                    theta = -np.pi / 2  # Moving downward
                
                else:  # Left robots (6,7)
                    px = -radius
                    py = 2 * (position_in_group - 6.5)  # -1, 1
                    gx = radius
                    gy = py
                    theta = 0  # Moving rightward
                
                robot.set(px, py, gx, gy, 0, 0, theta)
            # self.robot1.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            # self.robot2.set(1, -self.circle_radius, 1, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if all(robot.policy.multiagent_training for robot in self.robots) else 1
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

        for agent in self.robots + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        # if hasattr(self.robot1.policy, 'action_values') and hasattr(self.robot2.policy, 'action_values'):
        #     self.action_values = list()
        # if hasattr(self.robot1.policy, 'get_attention_weights') and hasattr(self.robot2.policy, 'get_attention_weights'):
        #     self.attention_weights = list()
            
        if all(hasattr(robot.policy, 'action_values') for robot in self.robots):
            self.action_values = list()
        if all(hasattr(robot.policy, 'get_attention_weights') for robot in self.robots):
            self.attention_weights = list()

        # get current observation
        # if self.robot1.sensor == 'coordinates' and self.robot2.sensor == 'coordinates':
        #     ob = [human.get_observable_state() for human in self.humans]
        # elif self.robot1.sensor == 'RGB' and self.robot2.sensor == 'RGB':
        #     raise NotImplementedError
            
        if all(robot.sensor == 'coordinates' for robot in self.robots):
            ob = [human.get_observable_state() for human in self.humans]
        elif all(robot.sensor == 'RGB' for robot in self.robots):                
            raise NotImplementedError
        else:
            raise ValueError("Inconsistent sensor types among robots.")       

        return ob

    # def onestep_lookahead(self, action1, action2):
    #     return self.step(action1, action2, update=False)
    
    def onestep_lookahead1(self, actions):
        return self.step1(actions, update=False)

    def is_in_human_zone(self, px, py):
        """
        Define zones where robots should prioritize avoiding humans
        Returns True if position is in human-avoidance priority zone
        """
        # Define outer zone (where robots start) as human-avoidance priority zone
        outer_radius = self.circle_radius * 0.7  # 70% of the environment radius
        distance_from_center = np.sqrt(px**2 + py**2)
        
        # Check if any humans are still moving (not reached their goals)
        humans_moving = any(not human.reached_destination() for human in self.humans)
        
        # If all humans have reached their goals, return False (everywhere becomes robot zone)
        if not humans_moving:
            return False
            
        # Outer area is human zone
        return distance_from_center > outer_radius

    def is_in_robot_zone(self, px, py):
        """
        Define zones where robots should prioritize avoiding other robots
        Returns True if position is in robot-avoidance priority zone
        """
        # If all humans have reached their goals, everywhere is robot zone
        humans_moving = any(not human.reached_destination() for human in self.humans)
        if not humans_moving:
            return True
            
        # Otherwise, inner area is robot zone
        return not self.is_in_human_zone(px, py)

    def step1(self, actions, update=True):
        """
        Modified step1 function with hierarchical reward structure
        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            # if self.robot1.visible and self.robot2.visible:
            #     ob += [self.robot.get_observable_state()]
            for robot in self.robots:
                if robot.visible:
                    ob.append(robot.get_observable_state())  
            human_actions.append(human.act(ob))


        # collision detection
        dmin = float('inf')
        collision = False

        # Calculate distances between robots and humans, and between robots
        for i, human in enumerate(self.humans):
            # Robot-human distances
            for j in range(self.robot_num):
                px = human.px - self.robots[j].px
                py = human.py - self.robots[j].py
                
                if self.robots[j].kinematics == 'holonomic':
                    vx = human.vx - actions[j].vx
                    vy = human.vy - actions[j].vy
                else:
                    vx = human.vx - actions[j].v * np.cos(actions[j].r + self.robots[j].theta)
                    vy = human.vy - actions[j].v * np.sin(actions[j].r + self.robots[j].theta)
                
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # Calculate closest distance between human and robot
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robots[j].radius
                
                if closest_dist < 0:
                    collision = True
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist

            if collision:
                break

            # Robot-robot distances
            for j in range(self.robot_num):
                for k in range(j + 1, self.robot_num):
                    # Calculate relative position
                    px = self.robots[j].px - self.robots[k].px
                    py = self.robots[j].py - self.robots[k].py

                    # Calculate relative velocity
                    if self.robots[j].kinematics == 'holonomic' and self.robots[k].kinematics == 'holonomic':
                        vx = actions[j].vx - actions[k].vx
                        vy = actions[j].vy - actions[k].vy
                    else:
                        vx = (actions[j].v * np.cos(actions[j].r + self.robots[j].theta)) - (actions[k].v * np.cos(actions[k].r + self.robots[k].theta))
                        vy = (actions[j].v * np.sin(actions[j].r + self.robots[j].theta)) - (actions[k].v * np.sin(actions[k].r + self.robots[k].theta))

                    # Calculate expected position
                    ex = px + vx * self.time_step
                    ey = py + vy * self.time_step

                    # Calculate closest distance between robots
                    closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.robots[j].radius - self.robots[k].radius

                    if closest_dist < 0:
                        collision = True
                        break
                    elif closest_dist < dmin:
                        dmin = closest_dist

                if collision:
                    break

    
        # Compute the hose constraints and update reward
        # Get robots_pair positions
        distance_between_robots_pair = [None]*(int)(self.robot_num/2)
        for i in range(0,self.robot_num,2):
            distance_between_robots_pair[(int)(i/2)] = np.linalg.norm(np.array(self.robots[i].get_position()) - np.array(self.robots[i+1].get_position()))
        # robot1_pos = np.array(self.robot1.get_position())
        # robot2_pos = np.array(self.robot2.get_position())
        # distance_between_robots = np.linalg.norm(robot1_pos - robot2_pos)

        # Initialize reward components
        reward = 0
        collision_reward = 0
        hose_reward = 0
        discomfort_reward = 0

        # Check if the distance exceeds hose length
        for i in range(len(distance_between_robots_pair)):
            if distance_between_robots_pair[i] > self.hose_length * 1.5:
                hose_reward += self.hose_penalty * (distance_between_robots_pair[i] - self.hose_length * 1.5)
                #collision = True
        # if distance_between_robots > self.hose_length:
        #     reward += self.hose_penalty * (distance_between_robots - self.hose_length)

        # Check for collision between hose and humans
        for human in self.humans:
            human_pos = np.array(human.get_position())
            # Compute the minimum distance from human to hose curve
            distance_to_hose = [None]*(len(distance_between_robots_pair))
            for i in range(len(distance_between_robots_pair)):
                distance_to_hose[i] = point_to_hose_curve(human_pos, self.robots[i*2].get_position(), self.robots[i*2+1].get_position(), self.hose_length)
            # distance_to_hose = point_to_hose_curve(human_pos, robot1_pos, robot2_pos, self.hose_length)
                # Define hose_thickness as per your model (e.g., 0.05 meters)
                if distance_to_hose[i] < (human.radius + self.hose_thickness):
                    hose_reward += self.hose_collision_penalty
                    collision = True
                    break  # Terminate the loop upon collision


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
        end_position = [None]*(self.robot_num)
        reaching_goal = [None]*(self.robot_num)
        for i in range(self.robot_num):
            end_position[i] = np.array(self.robots[i].compute_position(actions[i], self.time_step))
            reaching_goal[i] = norm(end_position[i] - np.array(self.robots[i].get_goal_position())) < self.robots[i].radius
        # end_position1 = np.array(self.robot1.compute_position(action1, self.time_step))
        # end_position2 = np.array(self.robot2.compute_position(action2, self.time_step))
        # reaching_goal1 = norm(end_position1 - np.array(self.robot1.get_goal_position())) < self.robot1.radius
        # reaching_goal2 = norm(end_position2 - np.array(self.robot2.get_goal_position())) < self.robot2.radius


        # Determine reward based on zones and collision types
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            # Higher penalty for any collision in human zone
            for i in range(self.robot_num):
                robot_pos = self.robots[i].get_position()
                if self.is_in_human_zone(robot_pos[0], robot_pos[1]):
                    collision_reward = self.collision_penalty * 3.0  # Even higher penalty in human zone
                else:
                    collision_reward = self.collision_penalty
            reward = collision_reward
            done = True
            info = Collision()
        elif all(reaching_goal):
            # Additional reward for reaching goal without collisions
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # Higher discomfort penalty in human zone
            for i in range(self.robot_num):
                robot_pos = self.robots[i].get_position()
                if self.is_in_human_zone(robot_pos[0], robot_pos[1]):
                    discomfort_reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * 3.0 * self.time_step
                else:
                    discomfort_reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            reward += discomfort_reward
            done = False
            info = Danger(dmin)
        else:
            # Small positive reward for making progress in robot zone
            #reward = 0
            for i in range(self.robot_num):
                robot_pos = self.robots[i].get_position()
                if not self.is_in_human_zone(robot_pos[0], robot_pos[1]):
                    goal_direction = np.array([self.robots[i].gx - robot_pos[0], 
                                            self.robots[i].gy - robot_pos[1]])
                    if np.linalg.norm(goal_direction) > 0:
                        current_velocity = np.array([self.robots[i].vx, self.robots[i].vy])
                        progress_reward = np.dot(current_velocity, goal_direction) * 0.1
                        reward += max(0, progress_reward)  # Only reward positive progress
            done = False
            info = Nothing()

        # Add hose constraints to final reward
        reward += hose_reward

        if update:
            # store state, action value and attention weights
            for i in range(self.robot_num):
                self.states.append(self.robots[i].get_full_state())
            
            self.states.append([human.get_full_state() for human in self.humans])
            # self.states.append([self.robot1.get_full_state(), self.robot2.get_full_state(), [human.get_full_state() for human in self.humans]])

            if all(hasattr(robot.policy, 'action_values') for robot in self.robots):
                for robot in self.robots:
                    self.action_values.append(robot.policy.action_values)
            if all(hasattr(robot.policy, 'get_attention_weights') for robot in self.robots):
                for robot in self.robots:
                    self.attention_weights.append(robot.policy.get_attention_weights()) 

            # if hasattr(self.robot1.policy, 'action_values') and hasattr(self.robot2.policy, 'action_values'):
            #     self.action_values.append(self.robot1.policy.action_values)
            #     self.action_values.append(self.robot2.policy.action_values)
            # if hasattr(self.robot1.policy, 'get_attention_weights') and hasattr(self.robot2.policy, 'get_attention_weights'):
            #     self.attention_weights.append(self.robot1.policy.get_attention_weights())
            #     self.attention_weights.append(self.robot2.policy.get_attention_weights())

            # update all agents
            for i, robot in enumerate(self.robots):
                robot.step(actions[i])
            # self.robot1.step(action1)
            # self.robot2.step(action2)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            # Record states
            for i in range(self.robot_num):
                self.states.append(self.robots[i].get_full_state())
            self.states.append([human.get_full_state() for human in self.humans])
            # self.states.append([
            #     self.robot1.get_full_state(),
            #     self.robot2.get_full_state(),
            #     [human.get_full_state() for human in self.humans]
            # ])
            # print(self.states)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if all(robot.sensor == 'coordinates' for robot in self.robots):
                ob = [human.get_observable_state() for human in self.humans]
            elif all(robot.sensor == 'RGB' for robot in self.robots):
                raise NotImplementedError
            else:
                raise ValueError("Inconsistent sensor types among robots.")

            # if self.robot1.sensor == 'coordinates' and self.robot2.sensor == 'coordinates':
            #     ob = [human.get_observable_state() for human in self.humans]
            # elif self.robot1.sensor == 'RGB' and self.robot2.sensor == 'RGB':
            #     raise NotImplementedError
        else:
            if all(robot.sensor == 'coordinates' for robot in self.robots):
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif all(robot.sensor == 'RGB' for robot in self.robots):
                raise NotImplementedError
            else:
                raise ValueError("Inconsistent sensor types among robots.")
            # if self.robot1.sensor == 'coordinates' and self.robot2.sensor == 'coordinates':
            #     ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            # elif self.robot1.sensor == 'RGB' and self.robot2.sensor == 'RGB':
            #     raise NotImplementedError

        return ob, reward, done, info


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
            total_time_steps = (int)(len(self.states)/(2*(1+self.robot_num)))  # Divide by 2 to account for pre/post action states
            
            # Initialize robot positions list for each robot
            robot_positions = [[] for _ in range(self.robot_num)]
            for step in range(total_time_steps):
                base_idx = step * 2 * (self.robot_num + 1)  # Multiply by 2 to skip the post-action state
                for i in range(self.robot_num):
                    robot_positions[i].append(self.states[base_idx + i].position)

            # Collect human positions
            human_positions = []
            for step in range(total_time_steps):
                base_idx = step * 2 * (self.robot_num + 1)  # Multiply by 2 to skip the post-action state
                human_state_positions = []
                for i in range(len(self.humans)):
                    human_state_positions.append(self.states[base_idx + self.robot_num][i].position)
                human_positions.append(human_state_positions)
            
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
            robot_base_color_names = ['blue', 'cyan', 'green', 'yellow', 'gray', 'purple', 'brown', 'lightgreen']

            # Assign specific colors to humans
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

            # Determine plotting interval (every 0.5 seconds)
            interval = max(1, int(math.ceil(0.5 / self.time_step)))

            # Create light and dark colors for robots
            robot_light_colors = []
            robot_dark_colors = []
            for i in range(self.robot_num):
                color_name = robot_base_color_names[i % len(robot_base_color_names)]
                light_color = adjust_color(color_name, amount=0.5, lighten=True)
                dark_color = adjust_color(color_name, amount=0.5, lighten=False)
                robot_light_colors.append(light_color)
                robot_dark_colors.append(dark_color)

            # Create light and dark colors for humans
            human_light_colors = []
            human_dark_colors = []
            for i in range(len(self.humans)):
                base_color = human_base_colors[i]
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

                # Interpolate robot colors over time from light to dark
                robot_colors = [
                    interpolate_color(robot_light_colors[i], robot_dark_colors[i], t_norm)
                    for i in range(self.robot_num)
                ]

                # Interpolate human colors over time from light to dark
                human_colors = [
                    interpolate_color(human_light_colors[i], human_dark_colors[i], t_norm)
                    for i in range(len(self.humans))
                ]

                # Plot positions at every 0.5 seconds
                if k % interval == 0 or k == total_time_steps - 1:
                    # Plot robots
                    robot_circles = []
                    for i in range(self.robot_num):
                        robot_circle = plt.Circle(robot_positions[i][k], self.robots[i].radius, 
                                            facecolor=robot_colors[i],
                                            edgecolor='black', linewidth=1.0, zorder=3)
                        ax.add_patch(robot_circle)
                        robot_circles.append(robot_circle)

                    # Plot hose every 2 seconds
                    if k % int(2 / self.time_step) == 0:
                        for i in range(0, self.robot_num, 2):
                            if i + 1 < self.robot_num:  # Ensure we have a pair of robots
                                robot1_pos = np.array(robot_positions[i][k])
                                robot2_pos = np.array(robot_positions[i + 1][k])
                                x, y = hose_model(robot1_pos, robot2_pos, self.hose_length)
                                ax.plot(x, y, color='black', linewidth=2, zorder=4)

                    # Plot humans
                    humans = []
                    for i in range(len(self.humans)):
                        human_circle = plt.Circle(human_positions[k][i], self.humans[i].radius,
                                            facecolor=human_colors[i],
                                            edgecolor='black', linewidth=1.0, zorder=3)
                        ax.add_patch(human_circle)
                        humans.append(human_circle)

                    # Add time annotations
                    global_time = k * self.time_step
                    agents = robot_circles + humans
                    times = [ax.text(agent.center[0] + x_offset, agent.center[1] + y_offset,
                                '{:.1f}'.format(global_time),
                                color='black', fontsize=10, zorder=4)
                            for agent in agents]

            # Plot running path lines for each robot
            for i in range(self.robot_num):
                robot_path_x = [pos[0] for pos in robot_positions[i]]
                robot_path_y = [pos[1] for pos in robot_positions[i]]
                ax.plot(robot_path_x, robot_path_y, color=robot_dark_colors[i], linewidth=2, zorder=4)

            # Plot running path lines for each human
            for i in range(len(self.humans)):
                human_path_x = [pos[0] for pos in human_positions_over_time[i]]
                human_path_y = [pos[1] for pos in human_positions_over_time[i]]
                ax.plot(human_path_x, human_path_y, color=human_dark_colors[i], linewidth=2, zorder=4)

            # Set plot limits dynamically
            all_positions = []
            for robot_pos in robot_positions:
                all_positions.extend(robot_pos)
            for human_pos in human_positions_over_time:
                all_positions.extend(human_pos)
            all_positions = np.array(all_positions)
            
            min_coords = np.min(all_positions, axis=0) - 1
            max_coords = np.max(all_positions, axis=0) + 1
            ax.set_xlim(min_coords[0], max_coords[0])
            ax.set_ylim(min_coords[1], max_coords[1])

            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Robot {i+1}',
                                        markerfacecolor=robot_dark_colors[i], markersize=10,
                                        markeredgecolor='black', markeredgewidth=1.0)
                            for i in range(self.robot_num)] + \
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
            plt.savefig("./result/test_003.png", dpi=300)

            
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
