cmake_minimum_required(VERSION 3.0.2)
project(docking_project)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/docking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

