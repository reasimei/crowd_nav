from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState

# robot.py

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index
        self.env = None

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
