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
from map_msgs.msg import OccupancyGridUpdate
#message proto
# id | length | data
def recv_msg(sock):
    try:
        # Read message length and unpack it into an integer
        raw_id = recvall(sock, 4)
        if not raw_id:
            return None
        id = struct.unpack('>I', raw_id)[0]
        print("robot0_map receive id: ",id)
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
    map_pub = OccupancyGrid()
    map_pub.header.stamp = rospy.Time.now()
    map_pub.header.frame_id = "map"
    map_pub.info.map_load_time = rospy.Time.now()
    #map_pub.info.map_load_time.secs = uint(list[0])
    #map_pub.info.map_load_time.nsecs = uint(list[1])
    map_pub.info.resolution = float(list[0])
    map_pub.info.width = int(list[1])
    map_pub.info.height = int(list[2])
    map_pub.info.origin.position.x = float(list[3])
    map_pub.info.origin.position.y = float(list[4])
    map_pub.info.origin.position.z = float(list[5])
    map_pub.info.origin.orientation.x = float(list[6])
    map_pub.info.origin.orientation.y = float(list[7])
    map_pub.info.origin.orientation.z = float(list[8])
    map_pub.info.origin.orientation.w = float(list[9])
    for i in range(len(list)-10):
        map_pub.data.append(int(list[i+10]))
    return map_pub

#初始化socket，监听8000端口
map_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
map_socket.bind(('',9000))
map_socket.listen(10)
 
(client,address) = map_socket.accept()
 
rospy.init_node("map_client_robot0")
map_pub = rospy.Publisher("/tb3_0/map",OccupancyGrid,queue_size=1)
map_up_pub = rospy.Publisher("/tb3_0/map_updates",OccupancyGridUpdate,queue_size=1000)
r = rospy.Rate(1000)
 
#设置noblock，否则会阻塞在接听，下面while不会一直循环，只有在有数据才进行下一次循环
fcntl.fcntl(client, fcntl.F_SETFL, os.O_NONBLOCK)
 
 
while not rospy.is_shutdown():
    data = recv_msg(client)
    if data:
        #msg_construct(data)
        map_pub.publish(msg_construct(data))
        map_up_pub.publish()
    r.sleep()
