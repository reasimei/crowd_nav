from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state import FullState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import numpy as np

# robot.py

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index
        self.env = None  # Will be set later

    def set_env(self, env):
        self.env = env

    # def get_other_robot_state(self):
    #     if self.robot_index == 0:
    #         return self.env.robot2.get_full_state()
    #     else:
    #         return self.env.robot1.get_full_state()

    def get_other_robot_state(self):
        # return self.env.robots[robot_index].get_full_state()
        # other_robots_num = 0
        other_robots_states = []
        for i, robot in enumerate(self.env.robots):
            if i != self.robot_index:
                other_robots_states.append(robot.get_full_state())
        return other_robots_states

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        other_robot_states = []
        other_robot_states = self.get_other_robot_state()
        # print(other_robot_states[0])
        state = JointState(self.get_full_state(), other_robot_states, ob)
        action = self.policy.predict(state)
        return action

    def act_avoid_humans(self, ob):
        """
        Conservative action selection in human zone - slow speed or wait
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        # Check if there are humans nearby
        humans_nearby = False
        for human_state in ob:
            if not isinstance(human_state, Robot):
                dist = np.linalg.norm(np.array([self.px - human_state.px, self.py - human_state.py]))
                if dist < 3.0:  # Conservative safety distance
                    humans_nearby = True
                    break
        
        if humans_nearby:
            # If humans are nearby, either wait or move very slowly
            if self.kinematics == 'holonomic':
                # Almost stop, with minimal movement towards goal
                direction = np.array([self.gx - self.px, self.gy - self.py])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                return ActionXY(direction[0] * self.v_pref * 0.1, direction[1] * self.v_pref * 0.1)
            else:
                return ActionRot(0, 0)  # Stop for non-holonomic robot
        else:
            # If no humans nearby, proceed with normal speed
            state = JointState(self.get_full_state(), [], ob)
            return self.policy.predict(state)

    def act_avoid_robots(self, ob):
        """
        More aggressive action selection in robot zone - normal speed with robot avoidance
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        # Get states of other robots
        other_robots = []
        for i, robot in enumerate(self.env.robots):
            if i != self.robot_index:
                other_robots.append(robot.get_full_state())
        
        # Create joint state with only robot states
        state = JointState(self.get_full_state(), other_robots, [])
        
        # Use normal speed for robot avoidance
        action = self.policy.predict(state)
        return action
