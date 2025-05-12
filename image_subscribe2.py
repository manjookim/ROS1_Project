#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
    
    def image_callback(self, data):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))
            return

        # 이미지 분할 (R, G, B)
        B, G, R = cv2.split(cv_image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 각각을 컬러로 보이게 하기 위해 채널 병합
        R_img = cv2.merge([np.zeros_like(R), np.zeros_like(R), R])
        G_img = cv2.merge([np.zeros_like(G), G, np.zeros_like(G)])
        B_img = cv2.merge([B, np.zeros_like(B), np.zeros_like(B)])

        # 이미지 보여주기
        cv2.imshow("Original RGB", cv_image)
        cv2.imshow("Red Channel", R_img)
        cv2.imshow("Green Channel", G_img)
        cv2.imshow("Blue Channel", B_img)
        cv2.imshow("Grayscale", gray)

        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ImageSubscriber()
        node.run()
    except rospy.ROSInterruptException:
        pass
