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
        self.angle_threshold = 0.15
        self.min_forward_speed = 0.08
        
        # 센서 퓨전 파라미터
        self.alpha = 0.8  # 카메라 신뢰 가중치 (0.7~0.9)
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
        rospy.loginfo("ArUco Docking Node (Sensor Fusion Enabled)")

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
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            target_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        dx = tvec[0][0][0]
                        dz = tvec[0][0][2]
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
                        
                        # 필터링된 yaw로 제어
                        self.control_robot(horizontal_distance, self.filtered_yaw)
                        self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, 
                                      horizontal_distance, yaw_camera, self.filtered_yaw)
                        break
            if not target_detected:
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 5.0:
                    self.stop_robot()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def control_robot(self, distance, filtered_yaw):
        twist = Twist()
        max_angular_speed = 0.8  # 회전 속도 제한 (너무 빠른 회전 방지)
        min_forward_speed = 0.1   # 최소 전진 속도 증가
        
        # 1. 방향 정렬 단계 (중요!)
        if abs(filtered_yaw) > 0.2:  # 11.5° 이상 오차
            # 우선 방향 정렬에 집중
            twist.angular.z = np.clip(0.6 * filtered_yaw, -max_angular_speed, max_angular_speed)
            
            # 전진은 최소한으로 유지 (마커 추적 유지)
            twist.linear.x = min_forward_speed * 0.3
            rospy.loginfo(f"ALIGNING: {math.degrees(filtered_yaw):.1f}°")
        
        # 2. 전진 단계 (방향이 어느정도 정렬된 후)
        elif distance > self.target_distance:
            # 방향 미세 조정 + 전진
            twist.angular.z = 0.4 * filtered_yaw  # 더 약한 회전
            
            # 전진 속도 계산 (거리 비례)
            base_speed = 0.3 * (distance - self.target_distance)
            twist.linear.x = max(min_forward_speed, base_speed)
            
            # 근접 감속
            if distance < 0.5:
                twist.linear.x *= max(0.4, distance/0.5)
            rospy.loginfo(f"APPROACHING: {distance*100:.1f}cm")
        
        # 3. 도킹 완료
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("DOCKING COMPLETED!")
        
        self.cmd_pub.publish(twist)


    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        rospy.loginfo("Stopping: Target marker not detected")

    def visualize(self, image, corners, ids, rvec, tvec, distance, raw_yaw, filtered_yaw):
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.05)
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        # 시각화 정보
        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"RawYaw: {math.degrees(raw_yaw):.1f}deg", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"FusedYaw: {math.degrees(filtered_yaw):.1f}deg", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"OdomYaw: {math.degrees(self.odom_yaw):.1f}deg", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 센서 퓨전 상태 표시
        cv2.putText(image, f"Fusion: alpha={self.alpha}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        
        cv2.imshow("ArUco Docking (Sensor Fusion)", image)
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
