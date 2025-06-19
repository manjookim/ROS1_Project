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

        # 도킹 파라미터
        self.target_id = 1
        self.target_distance = 0.015  # 1.5cm에서 정지
        self.angle_threshold = 0.087  # 5도 (0.087 라디안)
        self.approach_distance = 0.5   # 50cm까지 접근
        
        # 상태 관리
        self.state = "SEARCHING"  # SEARCHING, ALIGNING, APPROACHING, DOCKED
        self.last_marker_position = None  # (dx, dz) 마지막 마커 위치
        self.search_direction = 1  # 1: 오른쪽, -1: 왼쪽
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        
        rospy.loginfo("ArUco Docking Node - Simple State Machine Started")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
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

            marker_found = False
            current_distance = 0
            current_yaw = 0

            # 마커 검출 처리
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        marker_found = True
                        self.last_marker_time = rospy.Time.now()
                        
                        # 포즈 추정
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        dx = tvec[0][0][0]  # 좌우 거리
                        dz = tvec[0][0][2]  # 전방 거리
                        current_distance = math.sqrt(dx**2 + dz**2)
                        current_yaw = math.atan2(dx, dz)
                        
                        # 마지막 위치 저장 (탐색용)
                        self.last_marker_position = (dx, dz)
                        
                        # 시각화
                        self.visualize_marker(undistorted, [corners[i]], [ids[i]], 
                                            rvec, tvec, current_distance, current_yaw)
                        break

            # 상태 기반 제어
            if marker_found:
                self.process_marker_detected(current_distance, current_yaw)
            else:
                self.process_marker_lost()
            
            # 상태 정보 표시 (실시간)
            self.show_status(undistorted)
            
            # 항상 화면 업데이트 (실시간 표시)
            cv2.imshow("ArUco Docking", undistorted)
            cv2.waitKey(1)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def process_marker_detected(self, distance, yaw):
        """마커가 검출된 경우 상태 기반 처리"""
        
        # 도킹 완료 체크
        if distance <= self.target_distance and abs(yaw) < self.angle_threshold:
            self.state = "DOCKED"
            self.stop_robot()
            rospy.loginfo(f"🎯 DOCKED! Distance: {distance*100:.1f}cm")
            return
        
        # 방향 정렬 필요한 경우
        if abs(yaw) > self.angle_threshold:
            self.state = "ALIGNING"
            self.align_to_marker(yaw)
            rospy.loginfo(f"🔄 ALIGNING: {math.degrees(yaw):.1f}°")
        
        # 방향이 맞으면 직진
        else:
            if distance > self.target_distance:
                self.state = "APPROACHING"
                self.approach_marker(distance)
                rospy.loginfo(f"➡️ APPROACHING: {distance*100:.1f}cm")
            else:
                self.state = "DOCKED"
                self.stop_robot()
                rospy.loginfo(f"🎯 DOCKED! Distance: {distance*100:.1f}cm")

    def process_marker_lost(self):
        """마커를 잃어버린 경우 처리"""
        lost_time = (rospy.Time.now() - self.last_marker_time).to_sec()
        
        if lost_time > 1.0:  # 1초 이상 마커 미검출
            self.state = "SEARCHING"
            self.search_marker()
            rospy.loginfo("🔍 SEARCHING for marker...")
        else:
            # 잠시 정지
            self.stop_robot()

    def align_to_marker(self, yaw):
        """마커 방향으로 정렬"""
        twist = Twist()
        
        # 각도에 비례한 회전 속도 (부드러운 제어)
        angular_speed = max(0.2, min(0.6, abs(yaw) * 2.0))
        twist.angular.z = angular_speed * (-1 if yaw > 0 else 1)  # 반대 방향으로 회전
        
        # 정렬 중에는 매우 느린 전진 (마커 추적 유지)
        twist.linear.x = 0.03
        
        self.cmd_pub.publish(twist)

    def approach_marker(self, distance):
        """마커로 직진"""
        twist = Twist()
        twist.angular.z = 0.0  # 회전 없음, 직진만
        
        # 거리에 따른 속도 조절
        if distance > 0.3:  # 30cm 이상
            twist.linear.x = 0.2
        elif distance > 0.15:  # 15~30cm
            twist.linear.x = 0.1
        elif distance > 0.05:  # 5~15cm
            twist.linear.x = 0.05
        else:  # 5cm 이하
            twist.linear.x = 0.02
        
        self.cmd_pub.publish(twist)

    def search_marker(self):
        """마커 탐색 (회전)"""
        twist = Twist()
        twist.linear.x = 0.0  # 회전 중에는 전진하지 않음
        
        # 마지막 위치 기반 탐색 방향 결정
        if self.last_marker_position:
            dx, dz = self.last_marker_position
            # 마커가 왼쪽에 있었으면 왼쪽으로, 오른쪽에 있었으면 오른쪽으로
            if dx > 0:  # 마커가 오른쪽에 있었음
                self.search_direction = 1  # 오른쪽으로 회전
                twist.angular.z = -0.4  # 음수 = 오른쪽
                rospy.loginfo("🔍 Searching RIGHT (marker was on right)")
            else:  # 마커가 왼쪽에 있었음
                self.search_direction = -1  # 왼쪽으로 회전
                twist.angular.z = 0.4   # 양수 = 왼쪽
                rospy.loginfo("🔍 Searching LEFT (marker was on left)")
        else:
            # 기본 오른쪽 탐색
            twist.angular.z = -0.4
            rospy.loginfo("🔍 Default search (turning right)")
        
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """로봇 정지"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def visualize_marker(self, image, corners, ids, rvec, tvec, distance, yaw):
        """마커 시각화"""
        # 마커 경계 표시
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        
        # 좌표축 표시
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, 
                            rvec[i], tvec[i], 0.05)
        
        # 마커 중심점 계산
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        # 거리별 색상 변경
        if distance <= self.target_distance:
            color = (0, 0, 255)  # 빨간색 (도킹 완료)
        elif distance <= 0.1:
            color = (0, 165, 255)  # 주황색 (근접)
        else:
            color = (0, 255, 0)  # 초록색 (일반)
        
        # 정보 표시
        cv2.putText(image, f"ID: {ids[0]}", (center_x, center_y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Yaw: {math.degrees(yaw):.1f}°", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def show_status(self, image):
        """상태 정보 실시간 표시"""
        # 현재 상태 표시
        state_colors = {
            "SEARCHING": (0, 0, 255),    # 빨간색
            "ALIGNING": (0, 255, 255),   # 노란색
            "APPROACHING": (0, 255, 0),  # 초록색
            "DOCKED": (255, 0, 255)      # 마젠타색
        }
        
        color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(image, f"State: {self.state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 목표 정보
        cv2.putText(image, f"Target: ID={self.target_id}, Dist={self.target_distance*100:.1f}cm", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 마지막 마커 위치 정보
        if self.last_marker_position:
            dx, dz = self.last_marker_position
            side = "RIGHT" if dx > 0 else "LEFT"
            cv2.putText(image, f"Last marker: {side}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 실시간 프레임 표시
        cv2.putText(image, "LIVE", (image.shape[1] - 80, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ArucoDockingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
