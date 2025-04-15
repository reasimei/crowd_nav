class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        # self.px1 = px1
        # self.py1 = py1
        # self.px2 = px2
        # self.py2 = py2
        # self.vx1 = vx1
        # self.vy1 = vy1
        # self.vx2 = vx2
        # self.vy2 = vy2
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta
        # self.theta1 = theta1
        # self.theta2 = theta2

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)
        # self.position1 = (self.px1, self.py1)
        # self.position2 = (self.px2, self.py2)
        # self.goal_position = (self.gx, self.gy)
        # self.velocity1 = (self.vx1, self.vy1)
        # self.velocity2 = (self.vx2, self.vy2)

    def __add__(self, other): #重载加法运算符
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        # self.px1 = px1
        # self.py1 = py1
        # self.px2 = px2
        # self.py2 = py2
        # self.vx1 = vx1
        # self.vy1 = vy1
        # self.vx2 = vx2
        # self.vy2 = vy2
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)
        # self.position1 = (self.px1, self.py1)
        # self.position2 = (self.px2, self.py2)
        # self.velocity1 = (self.vx1, self.vy1)
        # self.velocity2 = (self.vx2, self.vy2)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])


# state.py

class JointState(object):
    def __init__(self, self_state, other_robot_states, human_states):
        assert isinstance(self_state, FullState)
        # assert isinstance(other_robot_states, FullState)
        # other_robot_states的type是list和Fullstate,FullState不能被索引
        # for i in range(other_robots_num):
        #     assert isinstance(other_robot_states[i], FullState)
        # assert all(isinstance(state, FullState) for state in other_robot_states)
        # print(type(other_robot_states))
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)
        self.self_state = self_state
        robot_num = 8
        self.other_robot_state = [FullState] * (robot_num-1)
        self.other_robot_state = other_robot_states
        self.human_states = human_states

class HierarchicalState(object):
    def __init__(self, self_state, robot_states, human_states, phase):
        self.self_state = self_state
        self.robot_states = robot_states
        self.human_states = human_states
        self.phase = phase  # 'human_avoidance' or 'robot_coordination'
        
    def get_robot_pair_state(self, robot_index):
        """Get paired robot state for hose constraints"""
        if robot_index % 2 == 0:
            pair_index = robot_index + 1
        else:
            pair_index = robot_index - 1
        if pair_index < len(self.robot_states):
            return self.robot_states[pair_index]
        return None