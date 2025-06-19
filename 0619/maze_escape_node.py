#!/usr/bin/env python3

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class MazeEscape:
    def __init__(self):
        rospy.init_node('maze_escape_node')

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.marker_actions = {
            1: 'left',
            2: 'right',
            3: 'forward',
            4: 'docking'  # 도킹용 마커 ID는 4번으로 가정
        }

        self.docking_triggered = False

        # 도킹 관련
        self.marker_length = 0.1  # meter
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix"))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for i, id_ in enumerate(ids.flatten()):
                action = self.marker_actions.get(id_, None)
                if action == 'docking':
                    self.handle_docking(corners[i], id_, gray)
                    return
                elif action:
                    self.execute_action(action)
                    return
        else:
            self.stop_robot()

    def handle_docking(self, corner, marker_id, gray):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers([corner], self.marker_length, self.camera_matrix, self.dist_coeffs)
        dx = tvec[0][0][0]
        dz = tvec[0][0][2]
        yaw = math.atan2(dx, dz)

        twist = Twist()
        if abs(yaw) > 0.05:
            twist.angular.z = 0.5 * yaw
        elif dz > 0.25:
            twist.linear.x = 0.3 * dz
        else:
            rospy.loginfo("도킹 완료!")
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

    def execute_action(self, action):
        twist = Twist()
        if action == 'left':
            twist.angular.z = 0.5
        elif action == 'right':
            twist.angular.z = -0.5
        elif action == 'forward':
            twist.linear.x = 0.2
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        MazeEscape()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
