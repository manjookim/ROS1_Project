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
        self.target_distance = 0.02  # 2cm로 변경 (1~3cm 목표)
        self.angle_threshold = 0.1  # 라디안 (≈5.7°) - 더 정밀하게
        self.min_forward_speed = 0.05   # 최소 전진 속도 감소
        
        # 센서 퓨전 파라미터
        self.alpha = 0.8  # 카메라 신뢰 가중치
        self.filtered_yaw = 0.0
        self.last_odom_yaw = 0.0
        
        # 개선된 탐색 모드 파라미터
        self.search_mode = False
        self.search_start_time = None
        self.search_phase = 0  # 탐색 단계
        self.total_search_angle = 0.0  # 누적 회전 각도
        self.search_direction = 1  # 1: 우회전, -1: 좌회전
        self.last_marker_position = None  # 마지막 마커 위치 기억
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        self.consecutive_detections = 0  # 연속 감지 횟수
        rospy.loginfo("ArUco Docking Node (Enhanced Search + Precise Docking)")

    def odom_callback(self, msg):
        # 쿼터니언 → 오일러 변환 (yaw만 사용)
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
            
            # ArUco 마커 검출 파라미터 최적화
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.01  # 더 작은 마커도 검출
            parameters.polygonalApproxAccuracyRate = 0.02
            parameters.adaptiveThreshWinSizeMin = 5
            parameters.adaptiveThreshWinSizeMax = 35
            parameters.adaptiveThreshWinSizeStep = 10
            
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            target_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        self.consecutive_detections += 1
                        
                        # 탐색 모드 해제 (안정적인 감지 후)
                        if self.consecutive_detections >= 3:
                            self.search_mode = False
                            self.search_phase = 0
                            self.total_search_angle = 0.0
                        
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # 마커 위치 정보 추출
                        dx = tvec[0][0][0]  # x축(좌우) 거리
                        dz = tvec[0][0][2]  # z축(전방) 거리
                        horizontal_distance = math.sqrt(dx**2 + dz**2)
                        yaw_camera = math.atan2(dx, dz)
                        
                        # 마지막 마커 위치 저장
                        self.last_marker_position = (dx, dz)
                        
                        # 센서 퓨전: 상보필터 적용
                        if self.odom_received:
                            delta_odom = self.odom_yaw - self.last_odom_yaw
                            predicted_yaw = self.filtered_yaw - delta_odom
                            self.filtered_yaw = self.alpha * yaw_camera + (1 - self.alpha) * predicted_yaw
                            self.last_odom_yaw = self.odom_yaw
                        else:
                            self.filtered_yaw = yaw_camera
                        
                        # 제어 명령 생성
                        self.control_robot(horizontal_distance, self.filtered_yaw)
                        
                        # 디버깅 시각화
                        self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, 
                                      horizontal_distance, yaw_camera, self.filtered_yaw)
                        break
            
            # 마커 미감지 시 처리
            if not target_detected:
                self.consecutive_detections = 0
                
                # 1초간 마커 미감지 시 탐색 모드 시작 (더 빠른 반응)
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 1.0:
                    if not self.search_mode:
                        self.search_mode = True
                        self.search_start_time = rospy.Time.now()
                        self.search_phase = 0
                        self.total_search_angle = 0.0
                        # 마지막 위치가 왼쪽이면 왼쪽부터, 오른쪽이면 오른쪽부터 탐색
                        if self.last_marker_position:
                            self.search_direction = 1 if self.last_marker_position[0] > 0 else -1
                        rospy.loginfo("Starting enhanced search mode...")
                    
                    # 개선된 탐색 모드 실행
                    self.execute_enhanced_search()
                else:
                    # 잠시 정지
                    self.stop_robot()
                    
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def execute_enhanced_search(self):
        """개선된 탐색 패턴: 체계적인 좌우 스캔"""
        twist = Twist()
        elapsed = (rospy.Time.now() - self.search_start_time).to_sec()
        
        # 단계별 탐색 패턴
        if self.search_phase == 0:
            # 1단계: 마지막 위치 방향으로 빠른 회전 (2초)
            if elapsed < 2.0:
                twist.angular.z = self.search_direction * 0.8
                rospy.loginfo(f"Search Phase 1: Quick turn {'right' if self.search_direction > 0 else 'left'}")
            else:
                self.search_phase = 1
                self.search_start_time = rospy.Time.now()
                
        elif self.search_phase == 1:
            # 2단계: 반대 방향으로 천천히 스캔 (6초)
            if elapsed < 6.0:
                twist.angular.z = -self.search_direction * 0.4
                rospy.loginfo(f"Search Phase 2: Slow scan {'left' if self.search_direction > 0 else 'right'}")
            else:
                self.search_phase = 2
                self.search_start_time = rospy.Time.now()
                
        elif self.search_phase == 2:
            # 3단계: 원래 방향으로 복귀 (3초)
            if elapsed < 3.0:
                twist.angular.z = self.search_direction * 0.6
                rospy.loginfo(f"Search Phase 3: Return to center")
            else:
                self.search_phase = 3
                self.search_start_time = rospy.Time.now()
                
        elif self.search_phase == 3:
            # 4단계: 360도 회전 탐색 (8초)
            if elapsed < 8.0:
                twist.angular.z = 0.5  # 천천히 한 바퀴
                rospy.loginfo("Search Phase 4: 360 degree scan")
            else:
                # 탐색 실패
                self.search_mode = False
                self.search_phase = 0
                rospy.logwarn("Search completed - marker not found!")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        
        # 탐색 중 매우 느린 전진 (벽에 부딪히지 않도록)
        if self.search_phase < 3:
            twist.linear.x = 0.02
        
        self.cmd_pub.publish(twist)

    def control_robot(self, distance, filtered_yaw):
        twist = Twist()
        max_angular_speed = 0.6  # 회전 속도 제한 (더 부드럽게)
        
        # 도킹 완료 체크 (1~3cm 이내)
        if distance <= 0.03:  # 3cm 이내
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo(f"DOCKING COMPLETED! Final distance: {distance*100:.1f}cm")
            self.cmd_pub.publish(twist)
            return
        
        # 1. 정밀 방향 정렬 단계 (5.7° 이상 오차)
        if abs(filtered_yaw) > self.angle_threshold:
            # 각도에 따른 적응적 회전 속도
            angular_speed = min(0.4, abs(filtered_yaw) * 2.0)
            twist.angular.z = np.clip(angular_speed * np.sign(filtered_yaw), 
                                    -max_angular_speed, max_angular_speed)
            
            # 정렬 중에는 매우 느린 전진 (마커 추적 유지)
            twist.linear.x = 0.03
            rospy.loginfo(f"ALIGNING: {math.degrees(filtered_yaw):.1f}°")
        
        # 2. 정밀 접근 단계
        else:
            # 미세 각도 조정
            twist.angular.z = 0.2 * filtered_yaw
            
            # 거리별 적응적 속도 제어
            if distance > 0.20:  # 20cm 이상
                twist.linear.x = 0.15
            elif distance > 0.10:  # 10~20cm
                twist.linear.x = 0.08
            elif distance > 0.05:  # 5~10cm
                twist.linear.x = 0.04
            else:  # 5cm 이하 - 매우 느리게
                twist.linear.x = 0.02
            
            rospy.loginfo(f"APPROACHING: {distance*100:.1f}cm, Speed: {twist.linear.x:.3f}")
        
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def visualize(self, image, corners, ids, rvec, tvec, distance, raw_yaw, filtered_yaw):
        # 마커 경계 및 축 표시
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[i], tvec[i], 0.05)
        
        # 마커 중심점
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        # 거리에 따른 색상 변경 (가까울수록 빨간색)
        if distance <= 0.03:
            color = (0, 0, 255)  # 빨간색 (도킹 완료)
        elif distance <= 0.10:
            color = (0, 165, 255)  # 주황색 (근접)
        else:
            color = (0, 255, 0)  # 초록색 (일반)
        
        # 상세한 디버깅 정보
        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"RawYaw: {math.degrees(raw_yaw):.1f}deg", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"FusedYaw: {math.degrees(filtered_yaw):.1f}deg", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Detections: {self.consecutive_detections}", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 상태 표시
        if self.search_mode:
            status_text = f"SEARCH MODE - Phase {self.search_phase + 1}"
            cv2.putText(image, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif distance <= 0.03:
            cv2.putText(image, "DOCKED!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(image, "TRACKING", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 목표 거리 가이드라인
        cv2.putText(image, f"Target: 1-3cm", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("ArUco Docking", image)
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
