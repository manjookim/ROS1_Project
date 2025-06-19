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
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.last_yaw = 0.0
        self.odom_yaw = 0.0
        self.odom_received = False
        rospy.loginfo("ArUco Docking Node Started (Odometry Feedback Enabled)")

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
                        self.last_yaw = yaw_camera
                        self.control_robot(horizontal_distance, yaw_camera)
                        self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, horizontal_distance, yaw_camera)
                        break
            if not target_detected:
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 5.0:
                    self.stop_robot()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def control_robot(self, distance, angle_camera):
        twist = Twist()
        # 1. 각도 보정(odometry 사용)
        if self.odom_received and abs(angle_camera) > self.angle_threshold:
            # 목표 각도를 odom 기준으로 추적
            # (실제 각도 오차 = 마커 기준 yaw + 현재 odom_yaw)
            angle_error = angle_camera  # 필요시 odom_yaw와 결합해 보정 가능
            twist.angular.z = 0.7 * angle_error
            # 큰 각도면 전진 차단
            if abs(angle_error) > 0.3:
                twist.linear.x = 0.0
            else:
                twist.linear.x = self.min_forward_speed * 0.5
            rospy.loginfo(f"[Odom] Angle correction: {math.degrees(angle_error):.1f}° (odom_yaw: {math.degrees(self.odom_yaw):.1f})")
        # 2. 전진 제어 (10cm 목표)
        elif distance > self.target_distance:
            speed = 0.4 * (distance - self.target_distance)
            twist.linear.x = max(speed, self.min_forward_speed)
            twist.angular.z = 0.3 * angle_camera
            rospy.loginfo(f"Approaching: {distance*100:.1f}cm")
        # 3. 도킹 완료
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("Docking completed (10cm)!")
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        rospy.loginfo("Stopping: Target marker not detected")

    def visualize(self, image, corners, ids, rvec, tvec, distance, angle):
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.05)
        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Angle: {math.degrees(angle):.1f}deg", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"OdomYaw: {math.degrees(self.odom_yaw):.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("ArUco Docking (10cm Target)", image)
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
