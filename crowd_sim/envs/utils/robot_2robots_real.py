#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


# robot.py

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index
        self.env = None  # Will be set later

        # Initialize ROS node
        self.position_pub = rospy.Publisher(f'/robot_{robot_index}/position', PoseStamped, queue_size=10)
        self.velocity_sub = rospy.Subscriber(f'/robot_{robot_index}/cmd_vel', Twist, self.velocity_callback)

        self.current_velocity = None

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

    def velocity_callback(self, msg):
        # Update robot's velocity based on received message
        self.vx = msg.linear.x
        self.vy = msg.linear.y

    def publish_position(self):
        # Publish the robot's current position
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = self.px
        pose_msg.pose.position.y = self.py
        self.position_pub.publish(pose_msg)

    def step(self, action):
        # Update the robot's state
        self.px += action.vx * self.time_step
        self.py += action.vy * self.time_step
        self.publish_position()
