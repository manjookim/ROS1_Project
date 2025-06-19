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
        max_angular_speed = 1.0
        min_forward_speed = 0.05
        
        # 1. 회전 제어 (필터링된 yaw 사용)
        twist.angular.z = np.clip(0.7 * filtered_yaw, -max_angular_speed, max_angular_speed)
        
        # 2. 전진 제어 (거리 + 각도 종속)
        if distance > self.target_distance:
            forward_factor = max(0.2, math.cos(abs(filtered_yaw)))
            base_speed = 0.4 * (distance - self.target_distance)
            twist.linear.x = max(min_forward_speed, base_speed * forward_factor)
            
            # 근접 구간 감속
            if distance < 0.5:  # 50cm 이내
                speed_reduction = max(0.3, distance / 0.5)
                twist.linear.x *= speed_reduction
        else:
            twist.linear.x = 0.0
        
        # 3. 로깅
        rospy.loginfo(f"Linear: {twist.linear.x:.2f}m/s, FusedYaw: {math.degrees(filtered_yaw):.1f}°")
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
