#!/usr/bin/env python3

import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry

class ArucoDocking:
    def __init__(self):
        rospy.init_node("aruco_docking_node")
        
        # 카메라 파라미터 (기존과 동일)
        self.marker_length = rospy.get_param("~marker_length", 0.4)
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix")).reshape((3,3))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))
        
        # 상태 변수 추가
        self.state = "SEARCH"  # SEARCH, ALIGN, APPROACH, COMPLETED
        self.target_distance = 0.1  # 10cm
        
        # 제어 파라미터
        self.linear_speed = 0.15
        self.angular_speed = 0.5
        self.robot_pose = [0.0, 0.0, 0.0]  # [x, y, theta] 형태로 초기화
        
        # ROS 인터페이스
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/camera/image", Image, self.image_callback)

        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        
        # 로봇 상태
        self.robot_pose = [0, 0, 0]  # x, y, yaw
        
        rospy.spin()

    def odom_callback(self, msg):
        # 오도메트리 데이터 활용
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_pose[2] = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and 1 in ids:
            idx = list(ids.flatten()).index(1)
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners[idx], self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            
            # 마커 위치 (카메라 좌표계)
            x, y, z = tvec[0][0]  # x: 좌우, y: 상하, z: 전방
            distance = np.sqrt(x**2 + z**2)  # 2D 거리
            angle = np.arctan2(x, z)         # 수평 각도
            
            # 시각화 
            aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
            cv2.imshow("ArUco Detection", frame)
            cv2.waitKey(1)
            
            # 상태 머신 기반 제어
            self.state_machine(distance, angle)

    def state_machine(self, distance, angle):
        twist = Twist()
        
        if self.state == "SEARCH":
            # 마커 발견 시 정렬 단계로
            self.state = "ALIGN"
            
        elif self.state == "ALIGN":
            # 각도 정렬
            if abs(angle) > 0.1:  # ~5.7도
                twist.angular.z = self.angular_speed * (-1 if angle < 0 else 1)
            else:
                self.state = "APPROACH"
                
        elif self.state == "APPROACH":
            # 전진 접근
            if distance > self.target_distance:
                twist.linear.x = self.linear_speed
            else:
                self.state = "COMPLETED"
                
        elif self.state == "COMPLETED":
            # 정지
            #pass
            return
            
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        ArucoDocking()
    except rospy.ROSInterruptException:
        pass
