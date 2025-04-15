#! /usr/bin/env python
# -*- coding=utf-8 -*-
import socket
import time,os,fcntl
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


def recv_msg(sock):
    try:
        # Read message length and unpack it into an integer
        raw_id = recvall(sock, 4)
        if not raw_id:
            return None
        id = struct.unpack('>I', raw_id)[0]
        print("move goal receive id: ",id)
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        return recvall(sock, msglen)
    except Exception as e:
        return None

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
##把接受的数据重新打包成ros topic发出去
def msg_construct(data):
    global delta_robot_init_pose_x
    global delta_robot_init_pose_y
    list = data.split(',')
    
    pose = PoseStamped()

    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "map"
    pose.pose.position.x = float(list[0])-float(delta_robot_init_pose_x)
    pose.pose.position.y = float(list[1]) -float(delta_robot_init_pose_y)
    pose.pose.position.z = 0#loat(list[2]) 
    pose.pose.orientation.z = 0#float(list[3])
    pose.pose.orientation.y = -0#-float(list[4])
    pose.pose.orientation.x = 0#float(list[5])
    pose.pose.orientation.w = 1#float(list[6])
    
    return pose

#初始化socket，监听8000端口
move_goal_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
robot_goal_com = rospy.get_param('robot_goal_com') 
delta_robot_init_pose_x = rospy.get_param('delta_robot_init_pose_x')
delta_robot_init_pose_y = rospy.get_param('delta_robot_init_pose_y')
move_goal_socket.bind(('',robot_goal_com))
move_goal_socket.listen(10)
(client,address) = move_goal_socket.accept()

rospy.init_node("hunting_goal_client_node")
move_goal_pub = rospy.Publisher("move_base_simple/goal",PoseStamped,queue_size=1)
r = rospy.Rate(1000)


fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)


while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
       #msg_construct(data)
        print("data: ",data)
        move_goal_pub.publish(msg_construct(data))
    r.sleep()