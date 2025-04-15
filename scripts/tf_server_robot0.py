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
from nav_msgs.msg import Odometry,OccupancyGrid
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
        print("tf receive id: ",id)
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        print("maglen: ",msglen)
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
    print(len(list))
    br = tf.TransformBroadcaster()
    br.sendTransform((float(list[0]),float(list[1])-2.4, 0),
                     (0,0,float(list[2]),float(list[3])),
                     rospy.Time.now(),
                     odomtopic_name,
                     maptopic_name)
    br.sendTransform((float(list[4]),float(list[5]), 0),
                     (0,0,float(list[6]),float(list[7])),
                     rospy.Time.now(),
                     baselink_name,
                     odomtopic_name)

#初始化socket，监听8000端口
tf_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tf_com = rospy.get_param('tf_com') 
maptopic_name= rospy.get_param('maptopic_name')
odomtopic_name= rospy.get_param('odomtopic_name')
baselink_name = rospy.get_param('baselink_name')
tf_socket.bind(('',tf_com))
tf_socket.listen(10)
 
(client,address) = tf_socket.accept()
 
rospy.init_node("tf_client_node")
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
        msg_construct(data)
    r.sleep()