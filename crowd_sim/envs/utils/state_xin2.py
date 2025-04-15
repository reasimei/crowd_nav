class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

# state.py

class JointState(object):
    def __init__(self, self_state, human_states, other_robot_state):
        assert isinstance(self_state, FullState)
        if other_robot_state is not None:
            assert isinstance(other_robot_state, FullState)
        if not isinstance(human_states, (list, tuple)):
            human_states = [human_states]
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)
        self.self_state = self_state
        self.human_states = human_states
        self.other_robot_state = other_robot_state

    def __str__(self):
        return f"JointState(self_state={self.self_state}, human_states={self.human_states}, other_robot_state={self.other_robot_state})"
# class JointState(object):
#     def __init__(self, self_state, human_states):
#         assert isinstance(self_state, FullState)
#         assert isinstance(other_robot_state, FullState)
#         if isinstance(human_states, ObservableState):
#             human_states = [human_states]
#         for human_state in human_states:
#             assert isinstance(human_state, ObservableState)
#         self.self_state = self_state
#         self.other_robot_state = other_robot_state
#         self.human_states = human_states


