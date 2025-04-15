from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState

import inspect

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        # Check if 'robot_index' is a parameter of the 'predict' method
        if 'robot_index' in inspect.signature(self.policy.predict).parameters:
            action = self.policy.predict(state, robot_index=self.robot_index)
        else:
            action = self.policy.predict(state)
        return action