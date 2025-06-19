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
        
        # 도킹 상태 정의
        self.docking_state = "SEARCH"  # SEARCH, MOVE_TO_Y, TURN_TO_CENTER, APPROACH_CENTER, COMPLETED
        
        # 마커 좌표 및 로봇 상태
        self.marker_x = 0.0  # 마커의 x 좌표
        self.marker_y = 0.0  # 마커의 y 좌표 (카메라 좌표계에서 z축)
        self.current_theta = 0.0  # 현재 로봇 각도
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.search_start_time = rospy.Time.now()
        self.odom_received = False
        
        rospy.loginfo("ArUco Docking Node: (x,y,θ) Based Algorithm Started")

    def odom_callback(self, msg):
        # 쿼터니언 → 오일러 변환으로 현재 각도 θ 계산
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_theta = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True

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
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        
                        # 포즈 추정으로 마커 중심 좌표 (x, y) 계산
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # 마커 중심 좌표 추출
                        self.marker_x = tvec[0][0][0]  # x 좌표 (좌우)
                        self.marker_y = tvec[0][0][2]  # y 좌표 (전방, 카메라 z축)
                        
                        # (x, y, θ) 기반 도킹 알고리즘 실행
                        self.execute_xy_docking()
                        
                        # 시각화
                        self.visualize_xy_docking(undistorted, corners[i], ids[i], rvec, tvec)
                        break
            
            # 마커 미감지 시 탐색
            if not target_detected:
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 2.0:
                    self.docking_state = "SEARCH"
                    self.search_for_marker()
                # 마커 없어도 화면 갱신
                self.visualize_xy_docking(undistorted, None, None, None, None)
                    
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def execute_xy_docking(self):
        """(x, y, θ) 기반 도킹 로직"""
        twist = Twist()
        
        if self.docking_state == "SEARCH":
            # 마커 발견 시 첫 번째 단계로 전환
            self.docking_state = "MOVE_TO_Y"
            rospy.loginfo(f"Marker found at (x={self.marker_x:.2f}, y={self.marker_y:.2f})")
        
        elif self.docking_state == "MOVE_TO_Y":
            # 1단계: Y를 만족시킬 만큼 현재 방향으로 직진
            if self.marker_y > 0.3:  # 30cm 이상 거리
                twist.linear.x = 0.2  # 현재 각도 θ 방향으로 직진
                twist.angular.z = 0.1 * self.marker_x  # x 오차 미세 보정
                rospy.loginfo(f"MOVE_TO_Y: Forward to y={self.marker_y:.2f}m")
            else:
                # Y 목표 도달, 각도 조정 단계로 전환
                self.docking_state = "TURN_TO_CENTER"
                rospy.loginfo("Y reached! Now turning to center")
        
        elif self.docking_state == "TURN_TO_CENTER":
            # 2단계: 마커 중심을 향하도록 각도 θ 조정
            angle_to_center = math.atan2(self.marker_x, self.marker_y)  # θ_a 계산
            
            if abs(angle_to_center) > 0.08:  # 약 4.6° 이상 오차
                twist.angular.z = 0.8 * angle_to_center
                twist.linear.x = 0.0  # 회전 중 정지
                rospy.loginfo(f"TURN_TO_CENTER: Angle={math.degrees(angle_to_center):.1f}°")
            else:
                # 각도 조정 완료, 중심점 접근 단계로 전환
                self.docking_state = "APPROACH_CENTER"
                rospy.loginfo("Angle aligned! Approaching center")
        
        elif self.docking_state == "APPROACH_CENTER":
            # 3단계: 마커 중심점을 향해 직진 (10cm까지)
            center_distance = math.sqrt(self.marker_x**2 + self.marker_y**2)
            
            if center_distance > self.target_distance:
                # 거리에 비례한 속도
                speed = 0.4 * (center_distance - self.target_distance)
                twist.linear.x = max(0.08, min(speed, 0.25))
                
                # 직진 중 미세 각도 조정
                angle_error = math.atan2(self.marker_x, self.marker_y)
                twist.angular.z = 0.3 * angle_error
                
                rospy.loginfo(f"APPROACH_CENTER: {center_distance*100:.1f}cm")
            else:
                # 도킹 완료 (10cm 이내)
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.docking_state = "COMPLETED"
                rospy.loginfo("DOCKING COMPLETED!")
        
        elif self.docking_state == "COMPLETED":
            # 도킹 완료 상태 유지
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        self.cmd_pub.publish(twist)

    def search_for_marker(self):
        """마커 탐색"""
        twist = Twist()
        elapsed = (rospy.Time.now() - self.search_start_time).to_sec()
        
        # 3초 주기로 좌우 회전
        if elapsed < 3.0:
            twist.angular.z = -0.6  # 우회전
        elif elapsed < 6.0:
            twist.angular.z = 0.6   # 좌회전
        else:
            self.search_start_time = rospy.Time.now()  # 리셋
        
        twist.linear.x = 0.05  # 느린 전진
        self.cmd_pub.publish(twist)

    def visualize_xy_docking(self, image, corners, marker_id, rvec, tvec):
        """(x, y, θ) 시각화"""
        display_image = image.copy()
        
        # 상태별 색상
        state_colors = {
            "SEARCH": (0, 0, 255),
            "MOVE_TO_Y": (0, 255, 255), 
            "TURN_TO_CENTER": (255, 255, 0),
            "APPROACH_CENTER": (255, 165, 0),
            "COMPLETED": (0, 255, 0)
        }
        color = state_colors.get(self.docking_state, (255, 255, 255))
        
        # 상태 정보
        cv2.putText(display_image, f"STATE: {self.docking_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 마커 좌표 (x, y) 및 각도 θ 표시
        cv2.putText(display_image, f"Marker (x,y): ({self.marker_x:.2f}, {self.marker_y:.2f})", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_image, f"Robot θ: {math.degrees(self.current_theta):.1f}°", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 마커 감지된 경우 시각화
        if corners is not None and marker_id is not None:
            # 마커 경계 및 축
            aruco.drawDetectedMarkers(display_image, [corners], np.array([[marker_id]]))
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, 
                                 rvec, tvec, 0.05)
            
            # 마커 중심점에 목표 표시
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.circle(display_image, (center_x, center_y), 8, (0, 0, 255), 2)
            cv2.putText(display_image, f"CENTER", (center_x-25, center_y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 각도 θ_a 시각화 (마커 중심을 향한 각도)
            angle_to_center = math.atan2(self.marker_x, self.marker_y)
            cv2.putText(display_image, f"θ_a: {math.degrees(angle_to_center):.1f}°", 
                        (center_x-30, center_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        cv2.imshow("(x,y,θ) ArUco Docking", display_image)
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
