# docking_node.py

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class ArucoDocking:
    def __init__(self):
        rospy.init_node('aruco_docking_node', anonymous=True)

        # 파라미터
        self.marker_length = 0.1  # meter
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix"))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))

        # ROS 통신
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.rate = rospy.Rate(10)
        rospy.loginfo("Aruco Docking Node Started")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()


        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            # 첫 번째 마커만 사용
            dx = tvec[0][0][0]
            dz = tvec[0][0][2]
            yaw = math.atan2(dx, dz)

            self.control_robot(dz, yaw)
        else:
            self.stop_robot()

    def control_robot(self, distance, angle):
        twist = Twist()

        if abs(angle) > 0.05:
            twist.angular.z = 0.5 * angle  # 회전 먼저
        elif distance > 0.25:
            twist.linear.x = 0.3 * distance  # 정면 접근
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("도킹 완료!")

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        ArucoDocking()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
