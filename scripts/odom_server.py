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
        print("odom receive id: ",id)
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
    
    odom = Odometry()
    odom.header.stamp = rospy.Time.now()
    odom.header.frame_id = "map"
    odom.child_frame_id = baselink_name
    odom.pose.pose.position.x = float(list[0])
    odom.pose.pose.position.y = float(list[1])
    odom.pose.pose.position.z = 0
    odom.pose.pose.orientation.x = 0
    odom.pose.pose.orientation.y = 0
    odom.pose.pose.orientation.z = float(list[2])
    odom.pose.pose.orientation.w = float(list[3])
    odom.pose.covariance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    odom.twist.twist.linear.x = float(list[4])
    odom.twist.twist.linear.y = float(list[5]) 
    odom.twist.twist.linear.z = 0 
    odom.twist.twist.angular.x = 0
    odom.twist.twist.angular.x = 0
    odom.twist.twist.angular.x = float(list[6])
    odom.twist.covariance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return odom
#初始化socket，监听8000端口
odom_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
odom_com = rospy.get_param('odom_com') 
odomtopic_name = rospy.get_param('odomtopic_name')
baselink_name = rospy.get_param('baselink_name')
odom_socket.bind(('',odom_com))
odom_socket.listen(10)
 
(client,address) = odom_socket.accept()
 
rospy.init_node("odom_client_node")
odom_pub = rospy.Publisher(odomtopic_name,Odometry,queue_size=100)
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
       #msg_construct(data)
        odom_pub.publish(msg_construct(data))
    r.sleep()
