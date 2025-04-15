class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision(object):
    def __init__(self, human_collision=False, robot_collision=False):
        self.human_collision = human_collision
        self.robot_collision = robot_collision

    def get_human_collision(self):
        return self.human_collision

    def get_robot_collision(self):
        return self.robot_collision
    
    def __str__(self):
        return 'Collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''

class ConstraintViolation(object):
    def __init__(self):
        pass

    def __repr__(self):
        return 'ConstraintViolation'

    def __str__(self):
        return self.__repr__()
