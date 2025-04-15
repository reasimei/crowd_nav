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

#message proto
# id | length | data
def recv_msg(sock):
    try:
        # Read message length and unpack it into an integer
        raw_id = recvall(sock, 4)
        if not raw_id:
            return None
        id = struct.unpack('>I', raw_id)[0]
        print("cmd_vel receive id: ",id)
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
    
    cmd_vel = Twist()
    cmd_vel.linear.x = float(list[0])
    cmd_vel.linear.y = float(list[1])
    cmd_vel.linear.z = float(list[2])
    cmd_vel.angular.x = float(list[3])
    cmd_vel.angular.y = float(list[4])
    cmd_vel.angular.z = float(list[5])
    
    return cmd_vel

#初始化socket，监听8000端口
cmd_vel_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
robot_vel_com = rospy.get_param('robot_vel_com') 
cmd_vel_socket.bind(('',robot_vel_com))
cmd_vel_socket.listen(10)
(client,address) = cmd_vel_socket.accept()

rospy.init_node("cmd_vel_client_node")
cmd_vel_pub = rospy.Publisher("cmd_vel",Twist,queue_size=200)
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
       #msg_construct(data)
        cmd_vel_pub.publish(msg_construct(data))
    r.sleep()