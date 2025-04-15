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

if __name__ == '__main__':
    global tf_socket
    tf_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tf_com = rospy.get_param('tf_com')
    tf_socket.connect(('127.0.0.1',tf_com))
 
    rospy.init_node('tf_server_node')
    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            (trans1,rot1) = listener.lookupTransform('/map', '/odom', rospy.Time(0))
            data = ""
            print(trans1[0])
            print(trans1[1])
            print(trans1[2])
            print(rot1[0])
            print(rot1[1])
            print(rot1[2])
            print(rot1[3]) 
	    (trans2,rot2) = listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
            print(trans2[0])
            print(trans2[1])
            print(trans2[2])
            print(rot2[0])
            print(rot2[1])
            print(rot2[2])
            print(rot2[3])
    	    data += str(trans1[0]) + "," + str(trans1[1])+ "," + str(rot1[2])+ "," + str(rot1[3])+ "," +str(trans2[0]) + "," + str(trans2[1])+ "," + str(rot2[2])+ "," + str(rot2[3])
            send_msg(tf_socket,data,1)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()    
