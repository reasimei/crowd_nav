cmake_minimum_required(VERSION 3.0.2)
project(crowd_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
  tf2_msgs
  map_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  PedestrianPositions.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs
)


install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  ./crowd_nav/test_2.py
  ./crowd_nav/publish_pedestrian_positions.py
  ./scripts/upper_computer_node.py
  ./crowd_sim/envs/utils/robot.py
  ./scripts/cmd_vel_client_robot0.py
  ./scripts/cmd_vel_client_robot1.py
  ./scripts/map_client_robot0.py
  ./scripts/map_client_robot1.py
  ./scripts/map_server_robot0.py
  ./scripts/map_server_robot1.py
  ./scripts/tf_client.py
  ./scripts/tf_server_robot0.py
  ./scripts/tf_server_robot1.py
  ./scripts/odom_server.py
  ./scripts/odom_server.py
  ./scripts/pose_server_robot0.py
  ./scripts/pose_server_robot1.py
  ./scripts/scan_server.py
  ./scripts/pose_server_robot0.py
  ./scripts/pose_server_robot1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Mark executables for installation
install(PROGRAMS
  ./crowd_nav/test_2.py
  ./scripts/upper_computer_node.py
  ./crowd_sim/envs/utils/robot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)