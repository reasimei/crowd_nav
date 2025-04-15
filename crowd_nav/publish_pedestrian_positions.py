#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from crowd_nav.msg import PedestrianPositions  # Replace with your package name
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN

class PedestrianDetector:
    def __init__(self, robot_index):
        self.robot_index = robot_index
        rospy.init_node(f'pedestrian_detector_robot_{robot_index}', anonymous=True)

        # Parameters
        self.lidar_topic = f'/robot_{robot_index}/scan'
        self.pedestrian_pub_topic = f'/robot_{robot_index}/pedestrian_positions'
        self.min_cluster_size = 3  # Minimum number of points to consider a cluster
        self.epsilon = 0.3  # DBSCAN epsilon parameter

        # Publisher
        self.pedestrian_pub = rospy.Publisher(self.pedestrian_pub_topic, PedestrianPositions, queue_size=10)

        # Subscriber
        rospy.Subscriber(self.lidar_topic, LaserScan, self.lidar_callback)

    def lidar_callback(self, data):
        # Convert LaserScan to point cloud (in robot frame)
        angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
        ranges = np.array(data.ranges)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        # Filter out invalid points
        valid_indices = ~np.isnan(points).any(axis=1)
        points = points[valid_indices]

        # Cluster points to detect pedestrians
        clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_cluster_size).fit(points)
        labels = clustering.labels_

        # Extract cluster centers (pedestrian positions)
        pedestrian_positions = []
        for label in set(labels):
            if label == -1:
                continue  # Ignore noise
            cluster_points = points[labels == label]
            centroid = cluster_points.mean(axis=0)
            pedestrian_positions.append(Point(x=centroid[0], y=centroid[1], z=0.0))

        # Publish pedestrian positions
        msg = PedestrianPositions()
        msg.positions = pedestrian_positions
        self.pedestrian_pub.publish(msg)
        rospy.loginfo(f"Robot {self.robot_index} published {len(pedestrian_positions)} pedestrian positions.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: publish_pedestrian_positions.py <robot_index>")
        sys.exit(1)
    robot_index = sys.argv[1]
    detector = PedestrianDetector(robot_index)
    detector.run()