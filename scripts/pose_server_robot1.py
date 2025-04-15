#! /usr/bin/env python
# -*- coding=utf-8 -*-
import socket
import time,os,fcntl
import struct
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry 
from tuw_multi_robot_msgs.msg import RobotInfo

#message proto
# id | length | data
def recv_msg(sock):
    try:
        # Read message length and unpack it into an integer
        raw_id = recvall(sock, 4)
        if not raw_id:
            return None
        id = struct.unpack('>I', raw_id)[0]
        print("pose receive id: ",id)
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
 
    list = data.split(',')
    robot_info = RobotInfo()

    robot_info.header.stamp = rospy.Time.now()
    robot_info.header.frame_id = "map"
    robot_info.robot_name = "robot_1"
    robot_info.pose.pose.position.x = float(list[0])
    robot_info.pose.pose.position.y = float(list[1])
    robot_info.pose.pose.position.z = 0
    robot_info.pose.pose.orientation.x = 0
    robot_info.pose.pose.orientation.y = 0
    robot_info.pose.pose.orientation.z = float(list[2])
    robot_info.pose.pose.orientation.w = float(list[3])
    robot_info.pose.covariance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return robot_info

#初始化socket，监听8000端口
pose_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
pose_com = rospy.get_param('pose_com') 
#odomtopic_name = rospy.get_param('odomtopic_name')
#baselink_name = rospy.get_param('baselink_name')
pose_socket.bind(('',pose_com))
pose_socket.listen(10)
 
(client,address) = pose_socket.accept()
 
rospy.init_node("pose_client_robot1")
pose_pub = rospy.Publisher("robot_info",RobotInfo,queue_size=100)
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
       #msg_construct(data)
        pose_pub.publish(msg_construct(data))
    r.sleep()