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
#message proto
# id | length | data
def recv_msg(sock):
    try:
        # Read message length and unpack it into an integer
        raw_id = recvall(sock, 4)
        if not raw_id:
            return None
        id = struct.unpack('>I', raw_id)[0]
        print("goal receive id: ",id)
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
    
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "/map"
    goal.pose.position.x = float(list[0])
    goal.pose.position.y = float(list[1])
    goal.pose.position.z = 0
    goal.pose.orientation.x = 0
    goal.pose.orientation.y = 0
    goal.pose.orientation.z = float(list[2])
    goal.pose.orientation.w = float(list[3])
    return goal
#初始化socket，监听8000端口
goal_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
goal_socket.bind(('',8005))
goal_socket.listen(10)
 
(client,address) = goal_socket.accept()
 
rospy.init_node("goal_client_node")
goal_pub = rospy.Publisher("/move_base_simple/goal",PoseStamped,queue_size=100)
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
       #msg_construct(data)
        goal_pub.publish(msg_construct(data))
    r.sleep()
