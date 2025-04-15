#! /usr/bin/env python
# -*- coding=utf-8 -*-
import socket
import struct
import rospy


from  geometry_msgs.msg import PoseStamped

#message proto
# id |  length | data
def send_msg(sock, msg ,id):
    # Prefix each message with a 4-byte id and length (network byte order)
    msg = struct.pack('>I',int(id)) + struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def goalCallback(msg):
    global goal_socket
 
    data = ""
 
    id = msg.header.seq
    print "goal send id: ",id
    x = msg.pose.position.x
    y = msg.pose.position.y
    #orientation
    orien_z = msg.pose.orientation.z
    orien_w = msg.pose.orientation.w
 
    data += str(x) + "," + str(y)+ "," + str(orien_z)+ "," + str(orien_w)
 
    send_msg(goal_socket,data,id)
 
 
goal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
goal_socket.connect(('127.0.0.1',8005))
 
rospy.init_node('goal_server_node')
 
rospy.Subscriber('/move_base_simple/goal',PoseStamped,goalCallback)
rospy.spin()
