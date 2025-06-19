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
        
        # 2단계 도킹 파라미터
        self.docking_state = "SEARCH"  # SEARCH, APPROACH_Y, TURN_TO_CENTER, FINAL_APPROACH, COMPLETED
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.approach_tolerance = 0.05  # 5cm 허용 오차
        
        # 센서 퓨전 파라미터
        self.alpha = 0.8
        self.filtered_yaw = 0.0
        self.last_odom_yaw = 0.0
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        self.initial_y_position = None
        
        rospy.loginfo("2-Stage ArUco Docking Node Started")

    def odom_callback(self, msg):
        # 쿼터니언 → 오일러 변환 (yaw만 사용)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True
        
        # 현재 위치 저장 (Y 좌표 추적용)
        self.current_position = msg.pose.pose.position

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            
            # ArUco 마커 검출
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.01
            parameters.polygonalApproxAccuracyRate = 0.05
            
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            target_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        
                        # 포즈 추정
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # 마커 위치 정보 추출 (카메라 좌표계)
                        self.target_x = tvec[0][0][0]  # 좌우 (로봇 기준)
                        self.target_y = tvec[0][0][1]  # 상하 (사용 안 함)
                        self.target_z = tvec[0][0][2]  # 전방 거리
                        
                        # 2단계 도킹 제어
                        self.execute_2stage_docking()
                        
                        # 시각화
                        self.visualize_2stage(undistorted, [corners[i]], [ids[i]], rvec, tvec)
                        break
            
            # 마커 미감지 시 처리
            if not target_detected:
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 3.0:
                    if self.docking_state != "SEARCH":
                        rospy.logwarn("Marker lost! Returning to search mode")
                        self.docking_state = "SEARCH"
                    self.search_for_marker()
                    
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def execute_2stage_docking(self):
        """2단계 도킹 로직 실행"""
        twist = Twist()
        
        if self.docking_state == "SEARCH":
            # 마커 발견 시 Y축 접근 단계로 전환
            if self.initial_y_position is None and self.odom_received:
                self.initial_y_position = self.current_position.y
            self.docking_state = "APPROACH_Y"
            rospy.loginfo(f"Target found! X:{self.target_x:.2f}, Z:{self.target_z:.2f}")
        
        elif self.docking_state == "APPROACH_Y":
            # 1단계: 마커의 Y 좌표까지 현재 방향으로 직진
            if abs(self.target_z) > 0.3:  # 30cm 이상 거리
                twist.linear.x = 0.2
                twist.angular.z = 0.1 * self.target_x  # 미세 조정
                rospy.loginfo(f"APPROACH_Y: Moving forward, Z:{self.target_z:.2f}m")
            else:
                # Y축 도달, 중심점 회전 단계로 전환
                self.docking_state = "TURN_TO_CENTER"
                rospy.loginfo("Y-axis reached! Starting turn to center")
        
        elif self.docking_state == "TURN_TO_CENTER":
            # 2단계: 마커 중심점을 향해 회전
            angle_to_center = math.atan2(self.target_x, self.target_z)
            
            if abs(angle_to_center) > 0.1:  # 5.7도 이상 오차
                twist.angular.z = 0.8 * angle_to_center
                twist.linear.x = 0.0  # 회전 중 정지
                rospy.loginfo(f"TURN_TO_CENTER: Angle:{math.degrees(angle_to_center):.1f}°")
            else:
                # 회전 완료, 최종 접근 단계로 전환
                self.docking_state = "FINAL_APPROACH"
                rospy.loginfo("Turn completed! Starting final approach")
        
        elif self.docking_state == "FINAL_APPROACH":
            # 3단계: 마커 중심점으로 직진하여 10cm 앞에서 정지
            distance_to_marker = math.sqrt(self.target_x**2 + self.target_z**2)
            
            if distance_to_marker > self.target_distance:
                # 거리에 비례한 속도
                speed = 0.3 * (distance_to_marker - self.target_distance)
                twist.linear.x = max(0.05, min(speed, 0.2))
                
                # 직진 중 미세 조정
                angle_error = math.atan2(self.target_x, self.target_z)
                twist.angular.z = 0.3 * angle_error
                
                rospy.loginfo(f"FINAL_APPROACH: Dist:{distance_to_marker*100:.1f}cm")
            else:
                # 도킹 완료
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.docking_state = "COMPLETED"
                rospy.loginfo("DOCKING COMPLETED! (10cm)")
        
        elif self.docking_state == "COMPLETED":
            # 도킹 완료 상태 유지
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        self.cmd_pub.publish(twist)

    def search_for_marker(self):
        """마커 탐색 모드"""
        twist = Twist()
        twist.angular.z = 0.5  # 천천히 회전
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)
        rospy.loginfo("Searching for marker...")

    def visualize_2stage(self, image, corners, ids, rvec, tvec):
        """2단계 도킹 시각화"""
        # 마커 경계 및 축 표시
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.05)
        
        # 마커 중심 정보
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        # 도킹 상태 및 좌표 정보 표시
        state_colors = {
            "SEARCH": (0, 0, 255),
            "APPROACH_Y": (0, 255, 255),
            "TURN_TO_CENTER": (255, 255, 0),
            "FINAL_APPROACH": (255, 165, 0),
            "COMPLETED": (0, 255, 0)
        }
        
        color = state_colors.get(self.docking_state, (255, 255, 255))
        
        # 상태 표시
        cv2.putText(image, f"STATE: {self.docking_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 좌표 정보
        cv2.putText(image, f"X: {self.target_x*100:.1f}cm", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Z: {self.target_z*100:.1f}cm", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 마커 중심에 목표 지점 표시
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 2)
        cv2.putText(image, "TARGET", (center_x-30, center_y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("2-Stage ArUco Docking", image)
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
