# CMakeLists.txt
cmake_minimum_required(VERSION 3.0.2)
project(camera_pose_pkg)

## 1. 사용하는 패키지 명시
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

## 2. OpenCV 필요 시
find_package(OpenCV REQUIRED)

## 3. catkin 패키지 선언
catkin_package()

## 4. 실행 파일 및 설치 디렉토리 설정
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## 5. 파이썬 스크립트 실행 가능하게 지정 (선택)
catkin_install_python(PROGRAMS
  scripts/solve_pnp.py
  scripts/make_list.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
