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
        self.target_distance = 0.02
        self.angle_threshold = 0.1
        self.alpha = 0.8
        self.filtered_yaw = 0.0
        self.last_odom_yaw = 0.0

        self.search_mode = False
        self.search_phase = 0
        self.search_direction = 1
        self.last_marker_position = None

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        self.consecutive_detections = 0

        rospy.loginfo("Aruco Docking Node Started")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.01
            parameters.polygonalApproxAccuracyRate = 0.02
            parameters.adaptiveThreshWinSizeMin = 5
            parameters.adaptiveThreshWinSizeMax = 35
            parameters.adaptiveThreshWinSizeStep = 10

            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        self.consecutive_detections += 1

                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )

                        dx = tvec[0][0][0]
                        dz = tvec[0][0][2]
                        distance = math.sqrt(dx**2 + dz**2)
                        yaw_camera = math.atan2(dx, dz)

                        self.last_marker_position = (dx, dz)

                        if self.odom_received:
                            delta_odom = self.odom_yaw - self.last_odom_yaw
                            predicted_yaw = self.filtered_yaw - delta_odom
                            self.filtered_yaw = self.alpha * yaw_camera + (1 - self.alpha) * predicted_yaw
                            self.last_odom_yaw = self.odom_yaw
                        else:
                            self.filtered_yaw = yaw_camera

                        self.control_robot(distance, self.filtered_yaw)
                        self.visualize(cv_image, [corners[i]], [ids[i]], rvec, tvec, distance, yaw_camera, self.filtered_yaw)
                        return

            self.consecutive_detections = 0
            if (rospy.Time.now() - self.last_marker_time).to_sec() > 1.0:
                self.execute_search()
            else:
                self.stop_robot()

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Image Callback Error: {e}")

    def control_robot(self, distance, filtered_yaw):
        twist = Twist()
        max_angular_speed = 0.6

        if distance <= 0.02 and abs(filtered_yaw) < self.angle_threshold:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo(f"[DOCKING COMPLETED] Distance: {distance*100:.1f}cm, Yaw: {math.degrees(filtered_yaw):.1f}°")
            self.cmd_pub.publish(twist)
            return

        if abs(filtered_yaw) > self.angle_threshold:
            twist.angular.z = np.clip(0.4 * np.sign(filtered_yaw), -max_angular_speed, max_angular_speed)
            twist.linear.x = 0.03
            rospy.loginfo(f"[ALIGNING] Yaw Error: {math.degrees(filtered_yaw):.1f}°")
        else:
            twist.angular.z = 0.0
            if distance > 0.20:
                twist.linear.x = 0.20
            elif distance > 0.10:
                twist.linear.x = 0.15
            elif distance > 0.05:
                twist.linear.x = 0.07
            else:
                twist.linear.x = 0.03
            rospy.loginfo(f"[FORWARD] Distance: {distance*100:.1f}cm, Speed: {twist.linear.x:.2f}")

        self.cmd_pub.publish(twist)

    def execute_search(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.4 * self.search_direction
        rospy.loginfo("[SEARCHING] Rotating to find marker...")
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

        color = (0, 0, 255) if distance <= 0.02 else (0, 255, 0)

        cv2.putText(image, f"Dist: {distance*100:.1f}cm", (center_x, center_y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Yaw: {math.degrees(filtered_yaw):.1f}°", (center_x, center_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, "TRACKING", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Aruco Docking", image)
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
