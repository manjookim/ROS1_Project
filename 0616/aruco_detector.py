#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.pose_pub = rospy.Publisher('/aruco_pose', PoseStamped, queue_size=10)

        # 내부 파라미터 및 왜곡계수
        self.camera_matrix = np.array([
            [506.73737097, 0, 316.26249958],
            [0, 506.68959373, 235.44052887],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([
            0.146345454, 0.04371783, 0.00114179444, 0.00140841683, -1.19683513
        ], dtype=np.float32)

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position.x = tvecs[i][0][0]
                pose.pose.position.y = tvecs[i][0][1]
                pose.pose.position.z = tvecs[i][0][2]
                pose.pose.orientation.x = rvecs[i][0][0]
                pose.pose.orientation.y = rvecs[i][0][1]
                pose.pose.orientation.z = rvecs[i][0][2]
                pose.pose.orientation.w = 0.0  # 회전벡터 그대로 사용 중
                self.pose_pub.publish(pose)

if __name__ == '__main__':
    ArucoDetector()
    rospy.spin()
