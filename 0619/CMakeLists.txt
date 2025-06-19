cmake_minimum_required(VERSION 3.0.2)
project(docking_project)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  rpicamera_node # RPi 카메라를 사용할 경우 필수
  aruco_ros    # ArUco 마커 기능 사용에 필
)

# catkin_package() 내부에 LIBRARIES나 DEPENDS를 명시할 필요가 있습니다.
# docking_project는 주로 Python 스크립트 기반이므로, CATKIN_DEPENDS를 명시하는 것이 일반적입니다.
catkin_package(
# INCLUDE_DIRS include
# LIBRARIES docking_project
CATKIN_DEPENDS rospy cv_bridge geometry_msgs sensor_msgs rpicamera_node aruco_ros # 추가된 부분
# DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Python 스크립트 실행 권한 부여 및 설치 (이 부분은 이미 올바르게 작성되어 있습니다.)
catkin_install_python(PROGRAMS
  scripts/docking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
