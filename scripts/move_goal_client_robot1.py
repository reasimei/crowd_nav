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
from move_base_msgs.msg import MoveBaseActionGoal


#message proto
# id |  length | data    
def send_msg(sock, msg ,id):
    # Prefix each message with a 4-byte id and length (network byte order)
    msg = struct.pack('>I',int(id)) + struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def robot_move_Callback(msg):
    global robot0_move_goal_socket
    global robot1_move_goal_socket
    global robot2_move_goal_socket
    data = ""
    id = msg.header.seq
    position_x = msg.pose.position.x
    position_y = msg.pose.position.y
    position_z = msg.pose.position.z
    orientation_x = msg.pose.orientation.x
    orientation_y = msg.pose.orientation.y
    orientation_z = msg.pose.orientation.z
    orientation_w = msg.pose.orientation.w

    data += str(position_x) + "," + str(position_y)+ "," + str(position_z)+ "," + str(orientation_x)+"," + str(orientation_y)+","+ str(orientation_z)+","+str(orientation_w)

#    send_msg(robot0_cmd_socket,data,id)
    send_msg(robot1_move_goal_socket,data,id)
    send_msg(robot2_move_goal_socket,data,id)



robot1_move_goal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot1_move_goal_socket.connect(('127.0.0.1',9001))
robot2_move_goal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot2_move_goal_socket.connect(('127.0.0.1',9002))

rospy.init_node('move_goal_client_vel_node')
 
r = rospy.Rate(0.2)
while not rospy.is_shutdown():
    rospy.Subscriber('move_base_simple/goal',PoseStamped,robot_move_Callback)
    r.sleep()
    r.sleep()

rospy.spin()