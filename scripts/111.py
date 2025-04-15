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
from move_base_msgs.msg import MoveBaseActionGoal

class pose:
    def __init__(self):
        self.position_x = 0
        self.position_y = 0
        self.position_z = 0
        self.orientation_x = 0
        self.orientation_y = 0
        self.orientation_z = 0
        self.orientation_w = 0
global pose1
pose1 = pose()
global pose2
pose2 = pose()
global p
p = [0,0,0]
global p1
p1 = [0,0,0,1]
global p2
p2 = [0,0,0,1]
Num = [0.00000]*10
#message proto
# id |  length | data    
def send_msg(sock, msg ,id):
    # Prefix each message with a 4-byte id and length (network byte order)
    msg = struct.pack('>I',int(id)) + struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def trans_to_rpy_from_quar(q): #q=[x,y,z,w]
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    r = math.atan2(2*(w*x +y*z),1-2*(x*x+y*y))#结果为弧度
    p = math.asin(2*(w*y-z*z))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    angleR = r*180/math.pi#弧度转换为角度
    angleP = p*180/math.pi
    angleY = y*180/math.pi
    p = [r,p,y]
    #p = [angleR,angleP,angleY]
    return p

def robot_Callback1(msg):
    global pose1
    global id1
    data = ""
    id = msg.header.seq
    Num[0] =id
    pose1 = PoseStamped()
    pose1 = pose()
    pose1.position_x = msg.pose.position.x
    pose1.position_y = msg.pose.position.y
    pose1.position_z = msg.pose.position.z
    pose1.orientation_x = msg.pose.orientation.x
    pose1.orientation_y= msg.pose.orientation.y
    pose1.orientation_z = msg.pose.orientation.z
    pose1.orientation_w = msg.pose.orientation.w
    orientation = [pose1.orientation_x,pose1.orientation_y,pose1.orientation_z,pose1.orientation_w]
    p2 = trans_to_rpy_from_quar(orientation)
    #print "p1: ",Num[0]                                     #暂时不考虑距离相机较远且两个坐标系具有夹角的情况
    return p2,pose1.position_y,pose1.position_z

def robot_Callback2(msg):
    global pose2
    global p
    data = ""
    id = msg.header.seq
    pose2 = PoseStamped()
    pose2 = pose()
    pose2.position_x = msg.pose.pose.position.x
    pose2.position_y = msg.pose.pose.position.y
    pose2.position_z = msg.pose.pose.position.z
   # print "pose111: ",pose2.position_x
    pose2.orientation_x = msg.pose.pose.orientation.x
    pose2.orientation_y= msg.pose.pose.orientation.y
    pose2.orientation_z = msg.pose.pose.orientation.z
    pose2.orientation_w = msg.pose.pose.orientation.w
    orientation = [pose2.orientation_x,pose2.orientation_y,pose2.orientation_z,pose2.orientation_w]
    p = trans_to_rpy_from_quar(orientation)
    #print "p1: ",id
    print("p[2] ",p[2])
    return p,pose2,id

def robot_move_Callback(pose1,pose2):
    data = ""
    #id = msg.header.seq
    global robot1_move_goal_socket
    global p
    position_x = pose2.position_x + pose1.position_z*math.cos(p[2])
   # print "pose2: ",pose2.position_x
    position_y = pose2.position_y + pose1.position_z*math.sin(p[2])
    
    #3print "pose222: ",pose1.position_z*math.cos(p1[2])
   # print "position_x: ",position_x
    #s k1print "position_y: ",position_y
    position_z = -pose1.position_x + pose2.position_z
    orientation_x = pose1.orientation_x + pose2.orientation_x
    orientation_y = pose1.orientation_y + pose2.orientation_y
    orientation_z = pose1.orientation_z - pose2.orientation_x
    orientation_w = pose1.orientation_w + pose2.orientation_w
    data += str(position_x) + "," + str(position_y)+ "," + str(position_z)+ "," + str(orientation_x)+"," + str(orientation_y)+","+ str(orientation_z)+","+str(orientation_w)
#    send_msg(robot0_cmd_socket,data,id)
    send_msg(robot1_move_goal_socket,data,Num[0])
    #send_msg(robot2_move_goal_socket,data,id)
def robot_move_Callback111(msg):
    global robot0_move_goal_socket
    global robot1_move_goal_socket
    global robot2_move_goal_socket
    data = ""
    id = 1
    #print "msg: ",msg
    position_x = msg.pose.pose.position.x
    position_y = msg.pose.pose.position.y
    position_z = msg.pose.pose.position.z
    orientation_x = msg.pose.pose.orientation.x
    orientation_y = msg.pose.pose.orientation.y
    orientation_z = msg.pose.pose.orientation.z
    orientation_w = msg.pose.pose.orientation.w

    data += str(position_x) + "," + str(position_y)+ "," + str(position_z)+ "," + str(orientation_x)+"," + str(orientation_y)+","+ str(orientation_z)+","+str(orientation_w)
    send_msg(robot1_move_goal_socket,data,id)
    
robot1_move_goal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot1_move_goal_socket.connect(('127.0.0.1',9002))
#robot2_move_goal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#robot2_move_goal_socket.connect(('127.0.0.1',9002))

rospy.init_node('move_goal_client_vel_node')
index = 0
r = rospy.Rate(0.2)
#while not rospy.is_shutdown():
 #  rospy.Subscriber('aruco_single/pose',PoseStamped,robot_Callback1)
  # print "pose1y: ",pose1.position_y
   #print "pose1z: ",pose1.position_z
  # if pose1.position_x != 0:
      # print "pose1: ",pose1.position_y
     #  if index == 0:
      #     rospy.Subscriber('odom',Odometry,robot_Callback2,queue_size = 1,buff_size = 52428800)
     #      print "pose2: ",pose2.position_y
    #       robot_move_Callback(pose1,pose2)
   #        index = 1
      #rospy.Subscriber('robot_pose_ekf/odom_combined',PoseWithCovarianceStamped,robot_move_Callback111)
  # r.sleep()
 #  r.sleep()

#rospy.spin()

while not rospy.is_shutdown():
   rospy.Subscriber('odom',Odometry,robot_Callback2,queue_size = 1,buff_size = 52428800)
   print("pose1 ",pose1.position_z)
   rospy.Subscriber('aruco_single/pose',PoseStamped,robot_Callback1,queue_size =1,buff_size = 52428800)
   print("pose2: ",pose2.position_x)
   #print "pose1z: ",pose1.position_z
   if pose1.position_z != 0:
       robot_move_Callback(pose1,pose2)
       #print "pose1: ",pose1.position_z
       #nprint "pose2: ",pose2.position_y
      #rospy.Subscriber('robot_pose_ekf/odom_combined',PoseWithCovarianceStamped,robot_move_Callback111)
   r.sleep()
   r.sleep()

rospy.spin()

# while not rospy.is_shutdown():
#    rospy.Subscriber('odom',Odometry,robot_Callback2,queue_size = 1,buff_size = 52428800)
#    rospy.Subscriber('aruco_single/pose',PoseStamped,robot_Callback1,queue_size =1,buff_size = 52428800)
   #print "pose2: ",id#pose2.position_x
   #print "pose1z: ",pose1.position_z
#    if pose1.position_z != 0:
#     #if pose2.position_x:
#     print "pose2 ",pose2.position_x
#     data = ""
#     #id = msg.header.seq
#     global robot1_move_goal_socket
    
#     position_x = pose2.position_x + pose1.position_z#*math.cos(p1[2])
#     position_y = pose2.position_y + pose1.position_z#*math.sin(p1[2])
    
#     #3print "pose222: ",pose1.position_z*math.cos(p1[2])
#    # print "position_x: ",position_x
#     #s k1print "position_y: ",position_y
#     position_z = -pose1.position_x + pose2.position_z
#     orientation_x = pose1.orientation_x + pose2.orientation_x
#     orientation_y = pose1.orientation_y + pose2.orientation_y
#     orientation_z = pose1.orientation_z + pose2.orientation_z
#     orientation_w = pose1.orientation_w + pose2.orientation_w
#     data += str(position_x) + "," + str(position_y)+ "," + str(position_z)+ "," + str(orientation_x)+"," + str(orientation_y)+","+ str(orientation_z)+","+str(orientation_w)
#     send_msg(robot1_move_goal_socket,data,Num[0])
#     r.sleep()
#     r.sleep()

# rospy.spin()