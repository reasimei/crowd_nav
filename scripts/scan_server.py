#! /usr/bin/env python
# -*- coding=utf-8 -*-
import socket
import time,os,fcntl
import struct
import rospy
import tf
import numpy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import LaserScan
#message proto
# id | length | data
def recv_msg(sock):
    try:
        # Read message length and unpack it into an integer
        raw_id = recvall(sock, 4)
        if not raw_id:
            return None
        id = struct.unpack('>I', raw_id)[0]
        print("scan receive id: ",id)
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
    scan = LaserScan()
    scan.header.stamp = rospy.Time.now()
    scan.header.frame_id = laserlink_name
    scan.angle_min = float(list[0])
    scan.angle_max = float(list[1])
    scan.angle_increment = float(list[2])
    scan.time_increment = float(list[3])
    scan.scan_time = float(list[4])
    scan.range_min = float(list[5])
    scan.range_max = float(list[6])
    for i in range((len(list)-7)/2):
        scan.ranges.append(float(list[i+7]))
    for j in range((len(list)-7)/2):
        scan.intensities.append(float(list[j+367]))
    return scan

#初始化socket，监听8000端口
scan_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
scan_com = rospy.get_param('scan_com') 
scantopic_name = rospy.get_param('scantopic_name')
laserlink_name = rospy.get_param('laserlink_name')
scan_socket.bind(('',scan_com))
scan_socket.listen(10)
 
(client,address) = scan_socket.accept()
 
rospy.init_node("scan_client_node")
scan_pub = rospy.Publisher(scantopic_name,LaserScan,queue_size=100)
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
       #msg_construct(data)
        scan_pub.publish(msg_construct(data))
    r.sleep()
