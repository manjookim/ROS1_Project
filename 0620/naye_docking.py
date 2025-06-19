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

        self.marker_length = rospy.get_param("~marker_length", 0.1)
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix")).reshape((3,3))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))

        self.target_id = 1
        self.target_distance = 0.1  # 10cm
        self.angle_threshold = 0.2
        self.min_forward_speed = 0.1

        self.alpha = 0.8
        self.filtered_yaw = 0.0
        self.last_odom_yaw = 0.0

        self.search_mode = False

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        rospy.loginfo("ArUco Docking Node (Improved Search + Precision Stop)")

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

            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.01
            parameters.polygonalApproxAccuracyRate = 0.05

            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            target_detected = False
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        self.search_mode = False

                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )

                        dx = tvec[0][0][0]
                        dz = tvec[0][0][2]
                        horizontal_distance = math.sqrt(dx**2 + dz**2)
                        yaw_camera = math.atan2(dx, dz)

                        if self.odom_received:
                            delta_odom = self.odom_yaw - self.last_odom_yaw
                            predicted_yaw = self.filtered_yaw - delta_odom
                            self.filtered_yaw = self.alpha * yaw_camera + (1 - self.alpha) * predicted_yaw
                            self.last_odom_yaw = self.odom_yaw
                        else:
                            self.filtered_yaw = yaw_camera

                        self.control_robot(horizontal_distance, self.filtered_yaw)
                        self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, 
                                       horizontal_distance, yaw_camera, self.filtered_yaw)
                        break

            if not target_detected:
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 2.0:
                    if not self.search_mode:
                        self.search_mode = True
                        rospy.loginfo("SEARCH MODE STARTED")
                    self.execute_search_mode()
                else:
                    self.stop_robot()

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def execute_search_mode(self):
        twist = Twist()
        twist.angular.z = 0.6  # 한쪽 방향으로 회전
        twist.linear.x = 0.0
        rospy.loginfo("SEARCHING FOR MARKER...")
        self.cmd_pub.publish(twist)

    def control_robot(self, distance, filtered_yaw):
        twist = Twist()
        max_angular_speed = 0.8

        if abs(filtered_yaw) > self.angle_threshold:
            angular_gain = 0.6 * min(1.0, 1.0 / (abs(filtered_yaw) + 0.1))
            twist.angular.z = np.clip(angular_gain * filtered_yaw, -max_angular_speed, max_angular_speed)
            twist.linear.x = self.min_forward_speed * 0.3
            rospy.loginfo(f"ALIGNING: {math.degrees(filtered_yaw):.1f}°")

        elif distance > self.target_distance:
            twist.angular.z = 0.3 * filtered_yaw
            base_speed = 0.3 * (distance - self.target_distance)
            twist.linear.x = max(self.min_forward_speed, min(base_speed, 0.3))

            if distance < 0.5:
                speed_factor = max(0.3, distance / 0.5)
                twist.linear.x *= speed_factor
            rospy.loginfo(f"APPROACHING: {distance*100:.1f}cm")

        elif distance <= self.target_distance and abs(filtered_yaw) < 0.1:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("DOCKING COMPLETED!")

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def visualize(self, image, corners, ids, rvec, tvec, distance, raw_yaw, filtered_yaw):
        aruco.drawDetectedMarkers(image, corners, np.array(ids))
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[i], tvec[i], 0.05)

        corner = corners[0][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))

        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"RawYaw: {math.degrees(raw_yaw):.1f}deg", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"FusedYaw: {math.degrees(filtered_yaw):.1f}deg", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
