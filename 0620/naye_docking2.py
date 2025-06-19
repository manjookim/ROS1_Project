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

        # 카메라 파라미터 (17cm 전체 크기, 실제 마커 부분은 약 14.5cm)
        self.marker_length = rospy.get_param("~marker_length", 0.145)  # 14.5cm 마커 (17cm 전체에서 테두리 제외)
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix")).reshape((3,3))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))

        # 도킹 파라미터 (단순화된 조건)
        self.target_id = 1
        self.target_distance = 0.05   # 5cm에서 정지 (3~5cm 범위)
        self.stop_distance = 0.03     # 3cm - 최종 정지 거리
        self.angle_threshold = 0.26   # 15도 (0.26 rad ≈ 15도)
        self.wide_angle_threshold = 0.35  # 20도 (0.35 rad ≈ 20도) - 원거리용
        self.stop_angle_threshold = 0.52  # 30도 (0.52 rad ≈ 30도) - 정지 조건
        self.approach_distance = 1.2  # 1.2m까지 접근
        self.max_detection_distance = 2.0  # 최대 검출 거리 2m
        
        # 상태 관리
        self.state = "SEARCHING"  # SEARCHING, ALIGNING, APPROACHING, DOCKED
        self.last_marker_position = None  # (dx, dz) 마지막 마커 위치
        self.search_direction = 1  # 1: 오른쪽, -1: 왼쪽
        self.search_start_time = rospy.Time.now()
        self.total_search_rotation = 0.0  # 총 회전량 추적
        self.quick_search_done = False  # 빠른 탐색 완료 여부
        self.stable_count = 0  # 안정적 접근 카운트 (빙글빙글 방지)
        
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
        
        # Odom 상태 확인용
        self.odom_check_timer = rospy.Timer(rospy.Duration(5.0), self.check_odom_status)
        
        rospy.loginfo("ArUco Docking Node - 14.5cm Marker (17cm total)")
        rospy.loginfo(f"Target: ID={self.target_id}, Distance={self.target_distance*100:.1f}cm")
        rospy.loginfo(f"Marker size: {self.marker_length*100:.1f}cm (total: 17cm)")
        rospy.loginfo(f"Max detection range: {self.max_detection_distance*100:.0f}cm")
        rospy.loginfo(f"Angle thresholds: Normal={math.degrees(self.angle_threshold):.0f}°, Wide={math.degrees(self.wide_angle_threshold):.0f}°, Stop={math.degrees(self.stop_angle_threshold):.0f}°")
        rospy.loginfo(f"Stop conditions: {self.stop_distance*100:.0f}cm~{self.target_distance*100:.0f}cm + within {math.degrees(self.stop_angle_threshold):.0f}°")

    def check_odom_status(self, event):
        """Odom 수신 상태 주기적 체크"""
        if not self.odom_received:
            rospy.logwarn("⚠️  Odometry not received! Check /odom topic")
        else:
            rospy.loginfo_throttle(30, f"✅ Odometry OK - Current yaw: {math.degrees(self.odom_yaw):.1f}°")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        if not self.odom_received:
            rospy.loginfo("✅ First odometry received!")
            self.odom_received = True
        
        # 초기 방향 저장
        if self.initial_yaw is None:
            self.initial_yaw = self.odom_yaw
            rospy.loginfo(f"Initial yaw set: {math.degrees(self.initial_yaw):.1f}°")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

            # ArUco 마커 검출 (더 관대한 파라미터)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            
            # 검출 파라미터 조정 (17cm 큰 마커용)
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 10
            parameters.minMarkerPerimeterRate = 0.02  # 큰 마커이므로 더 관대하게
            parameters.maxMarkerPerimeterRate = 5.0   # 큰 마커 검출 범위 확대
            
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
        """마커가 검출된 경우 상태 기반 처리 (단순화된 정지 조건)"""
        
        # 🎯 단순화된 도킹 완료 조건: 30도 이내 + 3~5cm 이내
        if distance <= self.target_distance and abs(yaw) <= self.stop_angle_threshold:
            self.state = "DOCKED"
            self.stop_robot()
            self.stable_count = 0
            rospy.loginfo(f"🎯 DOCKED! Distance: {distance*100:.1f}cm, Angle: {math.degrees(yaw):.1f}°")
            return
        
        # 3cm 이내면 무조건 정지 (안전장치)
        if distance <= self.stop_distance:
            self.state = "DOCKED"
            self.stop_robot()
            rospy.loginfo(f"🛑 EMERGENCY STOP! Too close: {distance*100:.1f}cm")
            return
        
        # 원거리 접근 (1.2m 이상)
        if distance > self.approach_distance:
            if abs(yaw) > self.wide_angle_threshold:
                self.state = "ALIGNING"
                self.align_to_marker(yaw)
                rospy.loginfo_throttle(1, f"🔄 FAR ALIGNING: {math.degrees(yaw):.1f}°")
            else:
                self.state = "APPROACHING"
                self.approach_marker_far(distance, yaw)
                rospy.loginfo_throttle(1, f"🚶 FAR APPROACH: {distance*100:.1f}cm")
            self.stable_count = 0
        
        # 중거리 방향 정렬 (15도 기준)
        elif abs(yaw) > self.angle_threshold:
            self.state = "ALIGNING"
            self.align_to_marker(yaw)
            self.stable_count = 0
            rospy.loginfo_throttle(1, f"🔄 ALIGNING: {math.degrees(yaw):.1f}°")
        
        # 직진 접근 (빙빙 도는 것 방지)
        else:
            self.state = "APPROACHING"
            self.approach_marker_simple(distance)
            rospy.loginfo_throttle(1, f"➡️ STRAIGHT APPROACH: {distance*100:.1f}cm")

    def process_marker_lost(self):
        """마커를 잃어버린 경우 처리"""
        lost_time = (rospy.Time.now() - self.last_marker_time).to_sec()
        
        if lost_time > 1.0:  # 1초 이상 마커 미검출
            self.state = "SEARCHING"
            self.search_marker()
            self.stable_count = 0
            rospy.loginfo_throttle(2, f"🔍 SEARCHING... (lost for {lost_time:.1f}s)")
        else:
            # 잠시 정지
            self.stop_robot()

    def align_to_marker(self, yaw):
        """마커 방향으로 정렬 (각도별 속도 조절)"""
        twist = Twist()
        
        # 각도 오차에 따른 속도 조절 (더 세밀하게)
        abs_yaw = abs(yaw)
        if abs_yaw > 0.52:  # 30도 이상 - 빠른 회전
            angular_speed = 0.5
        elif abs_yaw > 0.35:  # 20도 이상 - 중상 속도
            angular_speed = 0.4
        elif abs_yaw > 0.26:  # 15도 이상 - 중간 속도
            angular_speed = 0.25
        elif abs_yaw > 0.15:  # 8.5도 이상 - 중하 속도
            angular_speed = 0.2
        else:  # 작은 각도 - 느린 회전
            angular_speed = 0.15
            
        twist.angular.z = angular_speed * (-1 if yaw > 0 else 1)
        
        # 정렬 중에는 전진하지 않음 (안정성 향상)
        twist.linear.x = 0.0
        
        self.cmd_pub.publish(twist)

    def approach_marker_far(self, distance, yaw):
        """원거리에서 마커로 접근 (방향과 거리 동시 제어)"""
        twist = Twist()
        
        # 각도 오차가 크면 회전 우선 (20도 기준)
        if abs(yaw) > self.wide_angle_threshold:
            twist.angular.z = 0.6 * (-1 if yaw > 0 else 1)
            twist.linear.x = 0.1  # 천천히 전진하면서 회전
        else:
            # 거리에 따른 속도 조절 (17cm 마커 기준)
            if distance > 1.5:  # 150cm 이상
                twist.linear.x = 0.4  # 큰 마커이므로 더 빠르게 접근 가능
            elif distance > 1.0:  # 100~150cm
                twist.linear.x = 0.35
            elif distance > 0.6:  # 60~100cm
                twist.linear.x = 0.25
            else:  # 60cm 이하
                twist.linear.x = 0.15
            
            # 미세 조정 (15도 이내에서)
            if abs(yaw) > 0.1:  # 6도 이상이면 미세 조정
                twist.angular.z = 0.3 * (-1 if yaw > 0 else 1)
            else:
                twist.angular.z = 0.0  # 거의 정렬됨
        
        self.cmd_pub.publish(twist)

    def approach_marker_simple(self, distance):
        """단순한 직진 접근 (빙빙 도는 것 방지)"""
        twist = Twist()
        twist.angular.z = 0.0  # 회전 완전 금지
        
        # 거리에 따른 속도 조절 (더 보수적으로)
        if distance > 0.3:  # 30cm 이상
            twist.linear.x = 0.15  # 천천히
        elif distance > 0.15:  # 15~30cm
            twist.linear.x = 0.08  # 매우 천천히
        elif distance > 0.08:  # 8~15cm
            twist.linear.x = 0.04  # 극도로 천천히
        else:  # 8cm 이하
            twist.linear.x = 0.02  # 거의 정지 수준
        
        self.cmd_pub.publish(twist)

    def approach_marker(self, distance):
        """마커로 직진 (중거리)"""
        twist = Twist()
        twist.angular.z = 0.0  # 회전 없음, 직진만
        
        # 거리에 따른 속도 조절
        if distance > 0.8:  # 80cm 이상
            twist.linear.x = 0.25
        elif distance > 0.5:  # 50~80cm
            twist.linear.x = 0.2
        elif distance > 0.3:  # 30~50cm
            twist.linear.x = 0.15
        elif distance > 0.15:  # 15~30cm
            twist.linear.x = 0.08
        else:  # 15cm 이하는 final_approach로
            self.final_approach(distance, 0)
            return
        
        self.cmd_pub.publish(twist)

    def search_marker(self):
        """마커 탐색 (개선됨 - 과도한 회전 방지)"""
        twist = Twist()
        twist.linear.x = 0.0
        
        current_time = rospy.Time.now()
        search_duration = (current_time - self.search_start_time).to_sec()
        
        # 첫 번째: 빠른 탐색 (45도씩 좌우) - 각도 범위 확대
        if not self.quick_search_done and search_duration < 4.0:
            if search_duration < 2.0:  # 오른쪽 45도
                twist.angular.z = -0.4
                rospy.loginfo_throttle(1, "🔍 Quick search RIGHT (45°)")
            else:  # 왼쪽 90도 (중앙 기준 45도)
                twist.angular.z = 0.4
                rospy.loginfo_throttle(1, "🔍 Quick search LEFT (90°)")
        
        # 빠른 탐색 완료 후 중앙 복귀
        elif not self.quick_search_done and search_duration < 6.0:
            twist.angular.z = -0.4  # 중앙으로 복귀
            rospy.loginfo_throttle(1, "🔍 Return to center")
        
        # 빠른 탐색 완료 표시
        elif not self.quick_search_done:
            self.quick_search_done = True
            self.search_start_time = current_time  # 시간 리셋
            twist.angular.z = 0.0
            rospy.loginfo("✅ Quick search completed")
        
        # 두 번째: 전체 360도 천천히 탐색
        else:
            if search_duration > 10.0:  # 10초마다 방향 변경
                self.search_start_time = current_time
                self.search_direction *= -1
            
            twist.angular.z = 0.25 * self.search_direction  # 더 천천히
            direction = "RIGHT" if self.search_direction < 0 else "LEFT"
            rospy.loginfo_throttle(3, f"🔍 Full search {direction}")
        
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
        """상태 정보 실시간 표시 (Odom 상태 포함)"""
        h, w = image.shape[:2]
        
        # 현재 상태 표시
        state_colors = {
            "SEARCHING": (0, 0, 255),    # 빨간색
            "ALIGNING": (0, 255, 255),   # 노란색
            "APPROACHING": (0, 255, 0),  # 초록색
            "FINAL_APPROACH": (255, 0, 0),  # 파란색
            "DOCKED": (255, 0, 255)      # 마젠타색
        }
        
        color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(image, f"State: {self.state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 목표 정보 (업데이트된 정지 조건)
        cv2.putText(image, f"Target: ID={self.target_id}, Stop: {self.stop_distance*100:.0f}~{self.target_distance*100:.0f}cm", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 각도 임계값 정보
        cv2.putText(image, f"Angles: {math.degrees(self.angle_threshold):.0f}°/{math.degrees(self.stop_angle_threshold):.0f}° (stop)", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Odom 상태 표시
        odom_color = (0, 255, 0) if self.odom_received else (0, 0, 255)
        odom_status = f"Odom: {'OK' if self.odom_received else 'NO'}"
        if self.odom_received:
            odom_status += f" ({math.degrees(self.odom_yaw):.1f}°)"
        cv2.putText(image, odom_status, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, odom_color, 2)
        
        # 검출된 마커 수
        cv2.putText(image, f"Markers detected: {self.markers_detected_count}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 모든 마커 정보 표시
        if all_markers_info:
            y_pos = 180
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
                        (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # 탐색 시간
        if self.state == "SEARCHING":
            search_time = (rospy.Time.now() - self.search_start_time).to_sec()
            cv2.putText(image, f"Search time: {search_time:.1f}s", 
                        (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
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
