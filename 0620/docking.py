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
        
        # 1. 기본 파라미터
        max_angular_speed = 1.0  # 최대 회전 속도 (rad/s)
        min_forward_speed = 0.05  # 최소 전진 속도 (m/s)
        
        # 2. 각도 오차 보정 (항상 수행)
        twist.angular.z = np.clip(0.7 * angle_camera, -max_angular_speed, max_angular_speed)
        
        # 3. 전진 속도 계산 (각도 오차에 반비례)
        if distance > self.target_distance:
            # 각도 오차가 클수록 전진 속도 감소 (cosine 함수 사용)
            forward_factor = max(0.2, math.cos(abs(angle_camera)))
            base_speed = 0.4 * (distance - self.target_distance)
            twist.linear.x = max(min_forward_speed, base_speed * forward_factor)
        else:
            twist.linear.x = 0.0
        
        # 4. 로깅
        rospy.loginfo(f"Linear: {twist.linear.x:.2f}m/s, Angular: {math.degrees(twist.angular.z):.1f}°")
        
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
