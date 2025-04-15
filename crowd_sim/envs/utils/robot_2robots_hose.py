from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import numpy as np

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index
        self.env = None  # Will be set later
        self.mass = 1.0  # Set mass of the robot

    def set_env(self, env):
        self.env = env

    def get_other_robot_state(self):
        if self.robot_index == 0:
            return self.env.robot2.get_full_state()
        else:
            return self.env.robot1.get_full_state()

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        other_robot_state = self.get_other_robot_state()
        state = JointState(self.get_full_state(), other_robot_state, ob)
        action = self.policy.predict(state)
        return action

    def apply_force(self, force):
        # Update velocity based on force and mass
        acceleration = force / self.mass
        self.vx += acceleration[0] * self.env.time_step
        self.vy += acceleration[1] * self.env.time_step
        # Limit speed to preferred speed
        speed = np.hypot(self.vx, self.vy)
        if speed > self.v_pref:
            self.vx = (self.vx / speed) * self.v_pref
            self.vy = (self.vy / speed) * self.v_pref

    def update_position(self):
        self.px += self.vx * self.env.time_step
        self.py += self.vy * self.env.time_step

    def get_position(self):
        return self.px, self.py
