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

Num=[0.0000000]*10
Data=[0.0000000]*10
#message proto
# id |  length | data
def send_msg(sock, msg ,id):
    # Prefix each message with a 4-byte id and length (network byte order)
    msg = struct.pack('>I',int(id)) + struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)
 
def odomCallback(msg):
    
    id = msg.header.seq
    #print "odom send id: ",id
    Num[0] = id
    Num[5] = msg.twist.twist.linear.x
    Num[6] = msg.twist.twist.linear.y
    Num[7] = msg.twist.twist.angular.z

def amclCallback(msg):
    
    id = msg.header.seq
    #print "odom send id: ",id
    Num[1] = msg.pose.pose.position.x
    Num[2] = msg.pose.pose.position.y
    Num[3] = msg.pose.pose.orientation.z
    Num[4] = msg.pose.pose.orientation.w


def scanCallback(msg):
    global scan_socket

    data = ""
    
    id = msg.header.seq
    print("scan send id: ",id)
    angle_min = msg.angle_min
    data += str(angle_min) + ","
    angle_max = msg.angle_max
    data += str(angle_max) + ","
    angle_increment = msg.angle_increment
    data += str(angle_increment) + ","
    time_increment = msg.time_increment
    data += str(time_increment) + ","
    scan_time = msg.scan_time
    data += str(scan_time) + ","
    range_min = msg.range_min
    data += str(range_min) + ","
    range_max = msg.range_max
    data += str(range_max) 
    for i in range(len(msg.ranges)):
        data +="," + str(msg.ranges[i])
    for j in range(len(msg.intensities)):
        data +="," + str(msg.intensities[j])
    send_msg(scan_socket,data,id)


def poseCallback(msg):
    global pose_socket

    data = ""
    
    id = msg.header.seq
    print("pose send id: ",id)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orien_z = msg.pose.pose.orientation.z
    orien_w = msg.pose.pose.orientation.w

    data += str(x) + "," + str(y)+ "," + str(orien_z)+ "," + str(orien_w)

    send_msg(pose_socket,data,id)

def my_callback(event):
    rospy.Subscriber("/odom",Odometry,odomCallback)
    rospy.sleep(rospy.Duration(0.05))
    sub.unregister()

if __name__ == '__main__':
    odom_com = rospy.get_param('odom_com')
    scan_com = rospy.get_param('scan_com')
    odom_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    odom_socket.connect(('127.0.0.1',odom_com))
    rospy.Subscriber("/odom",Odometry,odomCallback)
    rospy.Subscriber("/amcl_pose",PoseWithCovarianceStamped,amclCallback)
    rospy.Subscriber("/scan",LaserScan,scanCallback)
    scan_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scan_socket.connect(('127.0.0.1',scan_com))
    rospy.init_node('data_server_node')
    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            data=""
            Data[0]=Num[1]
            Data[1]=Num[2]-3.6
            Data[2]=Num[3]
            Data[3]=Num[4]
            Data[4]=Num[5]
            Data[5]=Num[6]
            Data[6]=Num[7]
            data += str(Data[0]) + "," + str(Data[1])+ "," + str(Data[2])+ "," + str(Data[3])+"," + str(Data[4])+","+ str(Data[5])+"," + str(Data[6])
            send_msg(odom_socket,data,Num[0])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        rate.sleep() 