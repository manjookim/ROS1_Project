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

        # 도킹 파라미터 (개선됨)
        self.target_id = 1
        self.target_distance = 0.015  # 1.5cm에서 정지
        self.angle_threshold = 0.087  # 5도 (0.087 라디안)
        self.approach_distance = 1.0   # 1m까지 접근 (30-40cm에서도 인식하도록)
        self.max_detection_distance = 1.5  # 최대 검출 거리 1.5m
        
        # 상태 관리
        self.state = "SEARCHING"  # SEARCHING, ALIGNING, APPROACHING, DOCKED
        self.last_marker_position = None  # (dx, dz) 마지막 마커 위치
        self.search_direction = 1  # 1: 오른쪽, -1: 왼쪽
        self.search_start_time = rospy.Time.now()
        self.total_search_rotation = 0.0  # 총 회전량 추적
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        self.initial_yaw = None
        self.markers_detected_count = 0  # 디버깅용
        
        rospy.loginfo("ArUco Docking Node - Enhanced Search Started")
        rospy.loginfo(f"Target: ID={self.target_id}, Distance={self.target_distance*100:.1f}cm")
        rospy.loginfo(f"Max detection range: {self.max_detection_distance*100:.0f}cm")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True
        
        # 초기 방향 저장
        if self.initial_yaw is None:
            self.initial_yaw = self.odom_yaw

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

            # ArUco 마커 검출 (더 관대한 파라미터)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            
            # 검출 파라미터 조정 (원거리 검출 향상)
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 10
            parameters.minMarkerPerimeterRate = 0.03  # 더 작은 마커도 검출
            parameters.maxMarkerPerimeterRate = 4.0
            
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            marker_found = False
            current_distance = 0
            current_yaw = 0
            all_markers_info = []

            # 모든 마커 정보 수집 (디버깅용)
            if ids is not None:
                self.markers_detected_count = len(ids)
                for i in range(len(ids)):
                    try:
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        dx = tvec[0][0][0]
                        dz = tvec[0][0][2]
                        distance = math.sqrt(dx**2 + dz**2)
                        all_markers_info.append((ids[i][0], distance, dx, dz))
                        
                        # 목표 마커 처리
                        if ids[i] == self.target_id:
                            marker_found = True
                            self.last_marker_time = rospy.Time.now()
                            
                            current_distance = distance
                            current_yaw = math.atan2(dx, dz)
                            
                            # 마지막 위치 저장
                            self.last_marker_position = (dx, dz)
                            
                            # 시각화
                            self.visualize_marker(undistorted, [corners[i]], [ids[i]], 
                                                rvec, tvec, current_distance, current_yaw)
                            
                            rospy.loginfo_throttle(2, f"Target marker found! Distance: {current_distance*100:.1f}cm, Yaw: {math.degrees(current_yaw):.1f}°")
                            
                    except Exception as e:
                        rospy.logwarn(f"Error processing marker {ids[i]}: {e}")
            else:
                self.markers_detected_count = 0

            # 상태 기반 제어
            if marker_found and current_distance <= self.max_detection_distance:
                self.process_marker_detected(current_distance, current_yaw)
            else:
                self.process_marker_lost()
            
            # 디버깅 정보 표시
            self.show_status(undistorted, all_markers_info)
            
            # 화면 업데이트
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
        
        # 너무 멀리 있으면 접근부터
        if distance > self.approach_distance:
            self.state = "APPROACHING"
            self.approach_marker_far(distance, yaw)
            rospy.loginfo_throttle(1, f"🚶 FAR APPROACH: {distance*100:.1f}cm, {math.degrees(yaw):.1f}°")
        
        # 방향 정렬 필요한 경우
        elif abs(yaw) > self.angle_threshold:
            self.state = "ALIGNING"
            self.align_to_marker(yaw)
            rospy.loginfo_throttle(1, f"🔄 ALIGNING: {math.degrees(yaw):.1f}°")
        
        # 방향이 맞으면 직진
        else:
            if distance > self.target_distance:
                self.state = "APPROACHING"
                self.approach_marker(distance)
                rospy.loginfo_throttle(1, f"➡️ APPROACHING: {distance*100:.1f}cm")
            else:
                self.state = "DOCKED"
                self.stop_robot()
                rospy.loginfo(f"🎯 DOCKED! Distance: {distance*100:.1f}cm")

    def process_marker_lost(self):
        """마커를 잃어버린 경우 처리"""
        lost_time = (rospy.Time.now() - self.last_marker_time).to_sec()
        
        if lost_time > 0.5:  # 0.5초 이상 마커 미검출 (더 빠른 반응)
            self.state = "SEARCHING"
            self.search_marker()
            rospy.loginfo_throttle(2, f"🔍 SEARCHING... (lost for {lost_time:.1f}s)")
        else:
            # 잠시 정지
            self.stop_robot()

    def align_to_marker(self, yaw):
        """마커 방향으로 정렬"""
        twist = Twist()
        
        # 각도에 비례한 회전 속도 (부드러운 제어)
        angular_speed = max(0.2, min(0.8, abs(yaw) * 3.0))  # 더 빠른 정렬
        twist.angular.z = angular_speed * (-1 if yaw > 0 else 1)
        
        # 정렬 중에는 매우 느린 전진 (마커 추적 유지)
        twist.linear.x = 0.05
        
        self.cmd_pub.publish(twist)

    def approach_marker_far(self, distance, yaw):
        """원거리에서 마커로 접근 (방향과 거리 동시 제어)"""
        twist = Twist()
        
        # 각도 오차가 크면 회전 우선
        if abs(yaw) > 0.3:  # 17도 이상
            twist.angular.z = 0.6 * (-1 if yaw > 0 else 1)
            twist.linear.x = 0.1  # 천천히 전진하면서 회전
        else:
            # 거리에 따른 속도 조절
            if distance > 0.8:  # 80cm 이상
                twist.linear.x = 0.3
            elif distance > 0.5:  # 50~80cm
                twist.linear.x = 0.2
            else:  # 50cm 이하
                twist.linear.x = 0.15
            
            # 미세 조정
            twist.angular.z = 0.3 * (-1 if yaw > 0 else 1)
        
        self.cmd_pub.publish(twist)

    def approach_marker(self, distance):
        """마커로 직진 (근거리)"""
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
        """마커 탐색 (360도 회전)"""
        twist = Twist()
        twist.linear.x = 0.0
        
        current_time = rospy.Time.now()
        search_duration = (current_time - self.search_start_time).to_sec()
        
        # 마지막 위치 기반 탐색 또는 체계적 탐색
        if self.last_marker_position and search_duration < 5.0:
            # 마지막 위치 기반 빠른 탐색
            dx, dz = self.last_marker_position
            if dx > 0:  # 오른쪽에 있었음
                twist.angular.z = -0.6  # 오른쪽으로
                rospy.loginfo_throttle(2, "🔍 Quick search RIGHT")
            else:  # 왼쪽에 있었음
                twist.angular.z = 0.6   # 왼쪽으로
                rospy.loginfo_throttle(2, "🔍 Quick search LEFT")
        else:
            # 체계적 360도 탐색
            if search_duration > 10.0:  # 10초마다 탐색 방향 변경
                self.search_start_time = current_time
                self.search_direction *= -1
            
            twist.angular.z = 0.5 * self.search_direction
            direction = "RIGHT" if self.search_direction < 0 else "LEFT"
            rospy.loginfo_throttle(3, f"🔍 Full search {direction} ({search_duration:.1f}s)")
        
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
        elif distance <= 0.5:
            color = (0, 255, 255)  # 노란색 (중거리)
        else:
            color = (0, 255, 0)  # 초록색 (원거리)
        
        # 정보 표시
        cv2.putText(image, f"ID: {ids[0]}", (center_x, center_y - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Yaw: {math.degrees(yaw):.1f}°", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 타겟 표시
        cv2.putText(image, "TARGET", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def show_status(self, image, all_markers_info):
        """상태 정보 실시간 표시"""
        h, w = image.shape[:2]
        
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
        
        # 검출된 마커 수
        cv2.putText(image, f"Markers detected: {self.markers_detected_count}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 모든 마커 정보 표시
        if all_markers_info:
            y_pos = 120
            for marker_id, distance, dx, dz in all_markers_info:
                side = "RIGHT" if dx > 0 else "LEFT"
                text = f"ID{marker_id}: {distance*100:.0f}cm {side}"
                color = (0, 255, 0) if marker_id == self.target_id else (150, 150, 150)
                cv2.putText(image, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20
        
        # 마지막 마커 위치 정보
        if self.last_marker_position:
            dx, dz = self.last_marker_position
            side = "RIGHT" if dx > 0 else "LEFT"
            cv2.putText(image, f"Last target: {side} ({dz*100:.0f}cm)", 
                        (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # 탐색 시간
        if self.state == "SEARCHING":
            search_time = (rospy.Time.now() - self.search_start_time).to_sec()
            cv2.putText(image, f"Search time: {search_time:.1f}s", 
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # 실시간 표시
        cv2.putText(image, "LIVE", (w - 80, 30), 
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
