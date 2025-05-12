#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/image", Image, self.image_callback)
        rospy.loginfo("✅ image_subscriber 노드 시작됨")
        rospy.spin()

    def image_callback(self, msg):
        rospy.loginfo("✅ subscribe 중~")  # C++의 std::cout 대체

        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # OpenCV로 이미지 출력
            cv2.imshow("Received Image", img)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr("cv_bridge exception: %s", e)

if __name__ == '__main__':
    try:
        ImageSubscriber()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
