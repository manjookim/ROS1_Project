#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

class ArucoDockingNode:
    def __init__(self):
        rospy.init_node('aruco_docking_node', anonymous=True)
        
        # 카메라 파라미터
        self.marker_length = rospy.get_param("~marker_length", 0.1)
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix")).reshape((3,3))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))
        
        self.target_id = 1
        self.target_distance = 0.1  # 10cm
        
        # 오도메트리 기반 제어 파라미터
        self.docking_state = "SEARCH"  # SEARCH, APPROACH_MARKER, ALIGN_ANGLE, FINAL_APPROACH, COMPLETED
        self.initial_odom = None
        self.odom_distance = 0.0
        self.aligned_angle = False
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        rospy.loginfo("ArUco Docking with Odometry Tracking Started")

    def odom_callback(self, msg):
        # 쿼터니언 → 오일러 변환 (yaw만 사용)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True
        
        # 초기 위치 저장
        if self.initial_odom is None:
            self.initial_odom = msg.pose.pose.position
            rospy.loginfo("Initial Odom position set")
        
        # 이동 거리 계산 (피타고라스)
        if self.initial_odom:
            dx = msg.pose.pose.position.x - self.initial_odom.x
            dy = msg.pose.pose.position.y - self.initial_odom.y
            self.odom_distance = math.sqrt(dx**2 + dy**2)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            
            # ArUco 마커 검출
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            target_detected = False
            marker_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        marker_detected = True
                        
                        # 포즈 추정
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # 마커 정보 추출
                        dx = tvec[0][0][0]  # x축(좌우) 거리
                        dz = tvec[0][0][2]  # z축(전방) 거리
                        distance = math.sqrt(dx**2 + dz**2)
                        yaw = math.atan2(dx, dz)
                        
                        # 상태 머신 실행
                        self.execute_docking_state(distance, yaw, dx, dz)
                        
                        # 시각화
                        self.visualize(undistorted, corners[i], ids[i], rvec, tvec, distance, yaw)
                        break
            
            # 마커 미감지 시 처리
            if not marker_detected:
                if self.docking_state != "SEARCH":
                    self.docking_state = "SEARCH"
                    rospy.logwarn("Marker lost! Returning to search mode")
                
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 2.0:
                    self.search_for_marker()
                    
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def execute_docking_state(self, distance, yaw, dx, dz):
        twist = Twist()
        
        if self.docking_state == "SEARCH":
            # 마커 발견 시 접근 단계로 전환
            self.docking_state = "APPROACH_MARKER"
            self.initial_odom = None  # 오도메트리 리셋
            rospy.loginfo(f"Target found! Distance: {distance:.2f}m")
        
        elif self.docking_state == "APPROACH_MARKER":
            # 오도메트리 기반 접근 (1m 이내까지)
            if distance > 0.3:  # 30cm 이상 거리
                twist.linear.x = 0.2
                # 마커가 시야 중심에서 벗어나지 않도록 미세 조정
                twist.angular.z = 0.3 * yaw
                rospy.loginfo(f"APPROACHING: {distance:.2f}m, Odom: {self.odom_distance:.2f}m")
            else:
                # 30cm 도달, 각도 조정 단계로 전환
                self.docking_state = "ALIGN_ANGLE"
                self.aligned_angle = False
                rospy.loginfo("Reached 30cm! Starting angle alignment")
        
        elif self.docking_state == "ALIGN_ANGLE":
            # 마커 정면으로 각도 조정
            if not self.aligned_angle:
                target_angle = math.atan2(dx, dz)
                if abs(target_angle) > 0.1:  # 5.7° 이상 오차
                    twist.angular.z = 0.6 * target_angle
                    rospy.loginfo(f"ALIGNING: {math.degrees(target_angle):.1f}°")
                else:
                    self.aligned_angle = True
                    rospy.loginfo("Angle aligned! Starting final approach")
            else:
                # 각도 정렬 완료 후 전진
                self.docking_state = "FINAL_APPROACH"
        
        elif self.docking_state == "FINAL_APPROACH":
            # 마커 중심으로 직진 (10cm까지)
            if distance > self.target_distance:
                twist.linear.x = 0.15
                # 직선 유지를 위한 미세 조정
                angle_error = math.atan2(dx, dz)
                twist.angular.z = 0.4 * angle_error
                rospy.loginfo(f"FINAL APPROACH: {distance*100:.1f}cm")
            else:
                # 도킹 완료
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.docking_state = "COMPLETED"
                rospy.loginfo("DOCKING COMPLETED!")
        
        self.cmd_pub.publish(twist)

    def search_for_marker(self):
        """마커 탐색 (좌우 회전)"""
        twist = Twist()
        twist.angular.z = 0.5
        self.cmd_pub.publish(twist)

    def visualize(self, image, corners, marker_id, rvec, tvec, distance, yaw):
        """시각화 함수"""
        display_image = image.copy()
        
        # 상태 정보 표시
        state_colors = {
            "SEARCH": (0, 0, 255),
            "APPROACH_MARKER": (0, 255, 255),
            "ALIGN_ANGLE": (255, 255, 0),
            "FINAL_APPROACH": (255, 165, 0),
            "COMPLETED": (0, 255, 0)
        }
        color = state_colors.get(self.docking_state, (255, 255, 255))
        cv2.putText(display_image, f"STATE: {self.docking_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 마커 정보 표시
        if corners is not None:
            aruco.drawDetectedMarkers(display_image, [corners], np.array([[marker_id]]))
            cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.05)
            
            # 중심 좌표 계산
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            
            # 거리 및 각도 정보
            cv2.putText(display_image, f"Dist: {distance*100:.1f}cm", (center_x, center_y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, f"Angle: {math.degrees(yaw):.1f}°", (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 오도메트리 정보 표시
        cv2.putText(display_image, f"Odom Dist: {self.odom_distance:.2f}m", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_image, f"Odom Yaw: {math.degrees(self.odom_yaw):.1f}°", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("ArUco Docking with Odometry", display_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ArucoDockingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
