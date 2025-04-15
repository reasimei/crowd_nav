#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Twist
#from pibot_msgs.msg import Observation
from crowd_nav.policy.policy_factory import policy_factory

class UpperComputerNode:
    def __init__(self, robot_indices):
        rospy.init_node('upper_computer_node')
        self.robot_indices = robot_indices

        # Subscribers for robots' positions and observations
        for idx in robot_indices:
            rospy.Subscriber(f'/robot_{idx}/position', PoseStamped, self.position_callback, idx)
           # rospy.Subscriber(f'/robot_{idx}/observation', Observation, self.observation_callback, idx)

        # Publishers for velocity commands to robots
        self.cmd_vel_pubs = {
            idx: rospy.Publisher(f'/robot_{idx}/cmd_vel', Twist, queue_size=10)
            for idx in robot_indices
        }

        # Initialize policy
        self.policy = policy_factory['orca']()  # Replace 'orca' with your policy name
        # Load policy configuration and weights if necessary

        self.robot_positions = {}
       # self.robot_observations = {}

    def position_callback(self, msg, robot_index):
        self.robot_positions[robot_index] = msg

    #def observation_callback(self, msg, robot_index):
    #    self.robot_observations[robot_index] = msg

    def compute_and_send_commands(self):
        # Example implementation using ORCA policy
        for idx in self.robot_indices:
            position = self.robot_positions.get(idx)
            #observation = self.robot_observations.get(idx)
            if position:
                # Create state input for the policy
                state = self.create_state(position)

                # Compute action using the policy
                action = self.policy.predict(state)

                # Publish the velocity command
                cmd_vel = Twist()
                cmd_vel.linear.x = action['vx']
                cmd_vel.linear.y = action['vy']
                cmd_vel.angular.z = action['vz']
                self.cmd_vel_pubs[idx].publish(cmd_vel)

    def create_state(self, position):
        # Implement state creation based on position and observation
        state = {}
        state['position'] = position.pose
        #state['observation'] = observation
        return state

    def spin(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.compute_and_send_commands()
            rate.sleep()

if __name__ == '__main__':
    robot_indices = [0, 1]  # Add more robot indices if needed
    upper_computer = UpperComputerNode(robot_indices)
    upper_computer.spin()