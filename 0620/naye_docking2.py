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
        self.target_distance = 0.12  # 12cm에서 멈춤 (부딪히기 직전)
        self.safety_distance = 0.08  # 8cm 안전 거리
        self.angle_threshold = 0.15  # 8.6도 (더 정밀한 정렬)
        
        # 센서 퓨전 파라미터
        self.alpha = 0.7  # 카메라 신뢰 가중치
        self.filtered_yaw = 0.0
        self.last_odom_yaw = 0.0
        
        # 개선된 탐색 모드 파라미터
        self.search_mode = False
        self.search_start_time = None
        self.search_total_time = 0.0
        self.search_direction = 1  # 1: 우회전, -1: 좌회전
        self.last_marker_position = None  # 마지막 마커 위치 기억
        self.search_phase = 0  # 0: 제자리 회전, 1: 이동하며 회전, 2: 반대 방향
        
        # 도킹 상태 관리
        self.docking_state = "SEARCHING"  # SEARCHING, ALIGNING, APPROACHING, DOCKED
        self.consecutive_detections = 0
        self.required_detections = 3  # 안정적 검출을 위한 연속 검출 횟수
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        rospy.loginfo("Improved ArUco Docking Node with Smart Search")

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
            parameters.minMarkerPerimeterRate = 0.005  # 더 작은 마커도 검출
            parameters.maxMarkerPerimeterRate = 4.0
            parameters.polygonalApproxAccuracyRate = 0.03
            parameters.minCornerDistanceRate = 0.05
            parameters.minDistanceToBorder = 3
            
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            target_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        target_detected = True
                        self.consecutive_detections += 1
                        self.last_marker_time = rospy.Time.now()
                        
                        # 안정적 검출 확인
                        if self.consecutive_detections >= self.required_detections:
                            self.search_mode = False
                            
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
                            
                            # 센서 퓨전
                            self.update_sensor_fusion(yaw_camera)
                            
                            # 도킹 제어
                            self.control_docking(horizontal_distance, self.filtered_yaw, dz)
                            
                            # 시각화
                            self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, 
                                          horizontal_distance, yaw_camera, self.filtered_yaw, dz)
                        break
            
            # 마커 미검출 처리
            if not target_detected:
                self.consecutive_detections = 0
                self.handle_marker_loss()
                
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def update_sensor_fusion(self, yaw_camera):
        """센서 퓨전으로 안정적인 각도 추정"""
        if self.odom_received:
            delta_odom = self.odom_yaw - self.last_odom_yaw
            predicted_yaw = self.filtered_yaw - delta_odom
            self.filtered_yaw = self.alpha * yaw_camera + (1 - self.alpha) * predicted_yaw
            self.last_odom_yaw = self.odom_yaw
        else:
            self.filtered_yaw = yaw_camera

    def handle_marker_loss(self):
        """마커 손실 시 지능적 탐색"""
        current_time = rospy.Time.now()
        
        # 1초간 마커 미검출 시 탐색 모드 시작
        if (current_time - self.last_marker_time).to_sec() > 1.0:
            if not self.search_mode:
                self.search_mode = True
                self.search_start_time = current_time
                self.search_total_time = 0.0
                self.search_phase = 0
                self.docking_state = "SEARCHING"
                rospy.loginfo("Starting intelligent search mode...")
            
            self.execute_smart_search()
        else:
            # 잠시 정지하여 마커 재검출 기회 제공
            self.stop_robot()

    def execute_smart_search(self):
        """지능적 탐색 알고리즘"""
        twist = Twist()
        elapsed = (rospy.Time.now() - self.search_start_time).to_sec()
        
        if elapsed > 15.0:  # 15초 후 탐색 중단
            rospy.logwarn("Search timeout! Stopping.")
            self.stop_robot()
            return
        
        # Phase 0: 제자리에서 360도 회전 탐색 (0~8초)
        if self.search_phase == 0:
            if elapsed < 8.0:
                twist.angular.z = 0.5 * self.search_direction
                rospy.loginfo(f"Phase 0: Rotating in place ({elapsed:.1f}s)")
            else:
                self.search_phase = 1
                self.search_direction *= -1  # 방향 반전
        
        # Phase 1: 이동하며 회전 탐색 (8~12초)
        elif self.search_phase == 1:
            if elapsed < 12.0:
                twist.linear.x = 0.1  # 천천히 이동
                twist.angular.z = 0.4 * self.search_direction
                rospy.loginfo(f"Phase 1: Moving and rotating ({elapsed:.1f}s)")
            else:
                self.search_phase = 2
                self.search_direction *= -1  # 다시 방향 반전
        
        # Phase 2: 반대 방향으로 이동하며 탐색 (12~15초)
        else:
            twist.linear.x = 0.08
            twist.angular.z = 0.6 * self.search_direction
            rospy.loginfo(f"Phase 2: Final search ({elapsed:.1f}s)")
        
        self.cmd_pub.publish(twist)

    def control_docking(self, distance, filtered_yaw, forward_distance):
        """개선된 도킹 제어"""
        twist = Twist()
        
        # 안전 거리 체크 (너무 가까우면 후진)
        if forward_distance < self.safety_distance:
            twist.linear.x = -0.1  # 후진
            twist.angular.z = 0.0
            self.docking_state = "BACKING"
            rospy.logwarn(f"Too close! Backing up. Distance: {forward_distance*100:.1f}cm")
        
        # 도킹 완료 체크
        elif forward_distance <= self.target_distance and abs(filtered_yaw) < self.angle_threshold:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.docking_state = "DOCKED"
            rospy.loginfo("🎯 DOCKING COMPLETED! 🎯")
        
        # 방향 정렬 단계
        elif abs(filtered_yaw) > self.angle_threshold:
            # 거리에 따라 회전 속도 조절
            if distance > 0.5:  # 멀리 있을 때는 빠르게
                angular_speed = 0.8
            elif distance > 0.3:  # 중간 거리
                angular_speed = 0.5
            else:  # 가까이 있을 때는 천천히
                angular_speed = 0.3
            
            twist.angular.z = np.clip(angular_speed * np.sign(filtered_yaw), -0.8, 0.8)
            twist.linear.x = 0.05  # 마커 추적을 위한 최소 전진
            self.docking_state = "ALIGNING"
            rospy.loginfo(f"ALIGNING: {math.degrees(filtered_yaw):.1f}°, Dist: {distance*100:.1f}cm")
        
        # 전진 단계
        elif forward_distance > self.target_distance:
            # 미세 각도 조정
            twist.angular.z = 0.2 * filtered_yaw
            
            # 거리 기반 속도 제어
            remaining_distance = forward_distance - self.target_distance
            if remaining_distance > 0.3:
                speed = 0.15  # 빠른 접근
            elif remaining_distance > 0.15:
                speed = 0.1   # 중간 속도
            else:
                speed = 0.05  # 느린 정밀 접근
            
            twist.linear.x = speed
            self.docking_state = "APPROACHING"
            rospy.loginfo(f"APPROACHING: {forward_distance*100:.1f}cm, Target: {self.target_distance*100:.1f}cm")
        
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """로봇 정지"""
        twist = Twist()
        self.cmd_pub.publish(twist)

    def visualize(self, image, corners, ids, rvec, tvec, distance, raw_yaw, filtered_yaw, forward_dist):
        """시각화 및 디버깅 정보 표시"""
        # 마커 경계 및 축 표시
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[i], tvec[i], 0.05)
        
        # 마커 중심점
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        # 정보 표시
        info_y = 30
        cv2.putText(image, f"State: {self.docking_state}", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"Forward Dist: {forward_dist*100:.1f}cm", (10, info_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Total Dist: {distance*100:.1f}cm", (10, info_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Angle: {math.degrees(filtered_yaw):.1f}°", (10, info_y + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Detections: {self.consecutive_detections}", (10, info_y + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 목표선 그리기
        cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 0), 2)
        cv2.line(image, (0, center_y), (image.shape[1], center_y), (0, 255, 0), 2)
        
        # 도킹 완료 체크
        if self.docking_state == "DOCKED":
            cv2.putText(image, "DOCKING SUCCESS!", (center_x - 100, center_y - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
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
