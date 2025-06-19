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
        self.angle_threshold = 0.2  # 라디안 (≈11.5°)
        self.min_forward_speed = 0.1   # 최소 전진 속도 증가
        
        # 센서 퓨전 파라미터
        self.alpha = 0.8  # 카메라 신뢰 가중치 (0.7~0.9)
        self.filtered_yaw = 0.0
        self.last_odom_yaw = 0.0
        
        # 탐색 모드 파라미터
        self.search_mode = False
        self.search_start_time = None
        self.search_duration = 10.0  # 탐색 총 지속 시간 (초)
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        rospy.loginfo("ArUco Docking Node (Sensor Fusion + Auto Search)")

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
            
            # ArUco 마커 검출 파라미터 최적화 (ID=1 검출 강화)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.01  # 더 작은 마커 검출 허용
            parameters.polygonalApproxAccuracyRate = 0.05  # 윤곽 검출 정확도 향상
            
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            target_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        self.search_mode = False  # 마커 감지 시 탐색 모드 해제
                        
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # 마커 위치 정보 추출
                        dx = tvec[0][0][0]  # x축(좌우) 거리
                        dz = tvec[0][0][2]  # z축(전방) 거리
                        horizontal_distance = math.sqrt(dx**2 + dz**2)
                        yaw_camera = math.atan2(dx, dz)
                        
                        # 센서 퓨전: 상보필터 적용
                        if self.odom_received:
                            # 1. 오도메트리 변화량 계산
                            delta_odom = self.odom_yaw - self.last_odom_yaw
                            
                            # 2. 예측값 계산 (이전 필터값 + 오도메트리 변화)
                            predicted_yaw = self.filtered_yaw - delta_odom
                            
                            # 3. 상보필터 적용
                            self.filtered_yaw = self.alpha * yaw_camera + (1 - self.alpha) * predicted_yaw
                            
                            # 4. 오도메트리 값 업데이트
                            self.last_odom_yaw = self.odom_yaw
                        else:
                            self.filtered_yaw = yaw_camera
                        
                        # 제어 명령 생성
                        self.control_robot(horizontal_distance, self.filtered_yaw)
                        
                        # 디버깅 시각화
                        self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, 
                                      horizontal_distance, yaw_camera, self.filtered_yaw)
                        break
            
            # 마커 미감지 시 탐색 모드 활성화
            if not target_detected:
                # 2초간 마커 미감지 시 탐색 모드 시작
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 2.0:
                    if not self.search_mode:
                        self.search_mode = True
                        self.search_start_time = rospy.Time.now()
                        rospy.loginfo("Starting search mode...")
                    
                    # 탐색 모드 동작
                    self.execute_search_mode()
                else:
                    # 아직 탐색 모드 시작 전
                    self.stop_robot()
                    
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def execute_search_mode(self):
        """마커 미감지 시 탐색 동작 수행"""
        twist = Twist()
        elapsed = (rospy.Time.now() - self.search_start_time).to_sec()
        
        if elapsed < self.search_duration:
            # 왕복 탐색 패턴: 2초 우회전 → 2초 좌회전 반복
            search_phase = int(elapsed) % 4
            if search_phase < 2:  # 첫 2초: 우회전
                twist.angular.z = -0.6
                rospy.loginfo("Searching: Turning right")
            else:  # 다음 2초: 좌회전
                twist.angular.z = 0.6
                rospy.loginfo("Searching: Turning left")
                
            # 탐색 중 느린 전진
            twist.linear.x = 0.05
        else:
            # 탐색 시간 초과
            self.search_mode = False
            rospy.logwarn("Search timeout! Stopping.")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        self.cmd_pub.publish(twist)

    def control_robot(self, distance, filtered_yaw):
        twist = Twist()
        max_angular_speed = 0.8  # 회전 속도 제한
        
        # 1. 방향 정렬 단계 (11.5° 이상 오차)
        if abs(filtered_yaw) > self.angle_threshold:
            # 방향 정렬에 집중 (느린 회전)
            angular_gain = 0.6 * min(1.0, 1.0 / (abs(filtered_yaw) + 0.1))  # 각도에 반비례한 이득
            twist.angular.z = np.clip(angular_gain * filtered_yaw, -max_angular_speed, max_angular_speed)
            
            # 마커 추적을 위한 최소 전진
            twist.linear.x = self.min_forward_speed * 0.3
            rospy.loginfo(f"ALIGNING: {math.degrees(filtered_yaw):.1f}°")
        
        # 2. 전진 단계 (방향 정렬 후)
        elif distance > self.target_distance:
            # 미세 조정 + 전진
            twist.angular.z = 0.3 * filtered_yaw  # 약한 회전
            
            # 거리에 비례한 속도 (0.1m~0.5m 범위)
            base_speed = 0.3 * (distance - self.target_distance)
            twist.linear.x = max(self.min_forward_speed, min(base_speed, 0.3))  # 속도 제한
            
            # 근접 감속 (50cm 이내)
            if distance < 0.5:
                speed_factor = max(0.3, distance / 0.5)
                twist.linear.x *= speed_factor
            rospy.loginfo(f"APPROACHING: {distance*100:.1f}cm")
        
        # 3. 도킹 완료 (10cm 이내)
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("DOCKING COMPLETED!")
        
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def visualize(self, image, corners, ids, rvec, tvec, distance, raw_yaw, filtered_yaw):
        # 마커 경계 및 축 표시
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[i], tvec[i], 0.05)
        
        # 마커 중심 정보
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        # 디버깅 정보 표시
        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"RawYaw: {math.degrees(raw_yaw):.1f}deg", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"FusedYaw: {math.degrees(filtered_yaw):.1f}deg", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 탐색 모드 상태 표시
        if self.search_mode:
            cv2.putText(image, "SEARCH MODE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(image, "TRACKING MODE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
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
