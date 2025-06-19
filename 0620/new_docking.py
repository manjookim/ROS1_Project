#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import tf.transformations as tf_trans

class ArucoDocking:
    def __init__(self):
        rospy.init_node("aruco_docking_node")

        # camera intrinsic parameter
        self.camera_matrix = np.array([
            [506.73737097, 0, 316.26249958],
            [0, 506.68959373, 235.44052887],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([0.146345454, 0.04371783, 0.00114179444,
                                     0.00140841683, -1.19683513])

        # Publisher / Subscriber
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/camera/image", Image, self.image_callback)

        # Bridge & ArUco setting
        self.bridge = CvBridge()
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        # Robot pose
        self.robot_x = 0
        self.robot_y = 0
        self.robot_yaw = 0

        rospy.spin()

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and 1 in ids:
            idx = list(ids.flatten()).index(1)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[idx], 0.05, self.camera_matrix, self.dist_coeffs)

            # distance and angle to marker
            x, y, z = tvec[0][0]
            theta_a = np.arctan2(x, z)

            # 시각화
            aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
            cv2.imshow("ArUco Detection", frame)
            cv2.waitKey(1)

            # 제어 로직: 두 단계로 나눠서 이동
            if abs(y) > 0.15:
                self.move_straight(y)
            else:
                self.rotate_to(theta_a)
                self.move_straight(z - 0.10)  # 10cm 앞에서 멈춤

    def move_straight(self, distance):
        twist = Twist()
        twist.linear.x = 0.15 if distance > 0 else -0.15
        duration = abs(distance / twist.linear.x)
        self.publish_cmd(twist, duration)

    def rotate_to(self, angle):
        twist = Twist()
        twist.angular.z = 0.5 if angle > 0 else -0.5
        duration = abs(angle / twist.angular.z)
        self.publish_cmd(twist, duration)

    def publish_cmd(self, twist, duration):
        rate = rospy.Rate(10)
        ticks = int(duration * 10)
        for _ in range(ticks):
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    try:
        ArucoDocking()
    except rospy.ROSInterruptException:
        pass
