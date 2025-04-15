#! /usr/bin/env python
# -*- coding=utf-8 -*-
import socket
import struct
import rospy
import time
import tf
import numpy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from geometry_msgs.msg import Twist, Point, Quaternion,TransformStamped
from nav_msgs.msg import Odometry,OccupancyGrid

#message proto
# id |  length | data
def send_msg(sock, msg ,id):
    # Prefix each message with a 4-byte id and length (network byte order)
    msg = struct.pack('>I',int(id)) + struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def robot_cmd_Callback(msg):
    global robot_cmd_socket

    data = ""
    id = 1
    linear_x = msg.linear.x
    linear_y = msg.linear.y
    linear_z = msg.linear.z
    angular_x = msg.angular.x
    angular_y = msg.angular.y
    angular_z = msg.angular.z

    data += str(linear_x) + "," + str(linear_y)+ "," + str(linear_z)+ "," + str(angular_x)+"," + str(angular_y)+","+ str(angular_z)

    send_msg(robot_cmd_socket,data,id)

robot_vel_com = rospy.get_param('robot_vel_com')

robot_cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot_cmd_socket.connect(('127.0.0.1',robot_vel_com))

rospy.init_node('robot1_cmd_vel_node')
 
r = rospy.Rate(0.2)
while not rospy.is_shutdown():
    rospy.Subscriber('/robot_1/cmd_vel',Twist,robot_cmd_Callback)
  #  rospy.Subscriber('/cmd_vel',Twist,robot_cmd_Callback)
    r.sleep()
    r.sleep()

rospy.spin()