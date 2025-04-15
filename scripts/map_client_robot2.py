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
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


#message proto
# id |  length | data
def send_msg(sock, msg ,id):
    # Prefix each message with a 4-byte id and length (network byte order)
    msg = struct.pack('>I',int(id)) + struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def mapCallback(msg):
    global map_socket

    data = ""
    
    id = msg.header.seq
    print("map2 send id: ",id)
    info_resolution = msg.info.resolution
    data += str(info_resolution) + ","
    info_width = msg.info.width
    data += str(info_width) + ","
    info_height = msg.info.height
    data += str(info_height) + ","
    info_origin_position_x = msg.info.origin.position.x
    data += str(info_origin_position_x) + ","
    info_origin_position_y = msg.info.origin.position.y
    data += str(info_origin_position_y) + ","
    info_origin_position_z = msg.info.origin.position.z
    data += str(info_origin_position_z) + ","
    info_origin_orientation_x = msg.info.origin.orientation.x
    data += str(info_origin_orientation_x) + ","
    info_origin_orientation_y = msg.info.origin.orientation.y
    data += str(info_origin_orientation_y) + ","
    info_origin_orientation_z = msg.info.origin.orientation.z
    data += str(info_origin_orientation_z) + ","
    info_origin_orientation_w = msg.info.origin.orientation.w
    data += str(info_origin_orientation_w) 
    for i in range(len(msg.data)):
        data +="," + str(msg.data[i])
    print(len(msg.data))
    send_msg(map_socket,data,id)

def my_callback(event):
    sub=rospy.Subscriber("/map",OccupancyGrid,mapCallback)
    rospy.sleep(rospy.Duration(0.5))
    sub.unregister()
 
 

map_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
map_socket.connect(('127.0.0.1',9002))


 
rospy.init_node('map_server_robot2')
 
#r = rospy.Rate(0.1)
#while not rospy.is_shutdown():
#    rospy.Subscriber("/map",OccupancyGrid,mapCallback)
#    r.sleep()
rospy.Timer(rospy.Duration(1),my_callback)
rospy.spin()