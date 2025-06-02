#  (1) find_package(catkin REQUIRED COMPONENTS ...) 안에 필요한 패키지 추가
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

# (2) OpenCV 사용하는 경우 추가
find_package(OpenCV REQUIRED)

# (3) 파이썬 실행 파일 등록
catkin_install_python(PROGRAMS
  scripts/solve_pnp.py
  scripts/make_list.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
-> scripts/ 폴더 안에 넣은 .py 파일만 등록하면 된다 
