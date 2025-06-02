#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class CameraCapture:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None

        rospy.init_node('camera_capture')
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        print("[INFO] Press 's' to save image, 'q' to quit.")

        os.makedirs("captures", exist_ok=True)
        self.count = 0

        self.loop()

    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(f"Error converting image: {e}")

    def loop(self):
        while not rospy.is_shutdown():
            if self.image is not None:
                q = self.image.copy()
                s = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
                r = cv2.resize(q, (320, 240))

                cv2.imshow("q: Raw", q)
                cv2.imshow("s: Grayscale", s)
                cv2.imshow("r: Resized", r)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    # Save all 3 images
                    cv2.imwrite(f"captures/q_{self.count}.jpg", q)
                    cv2.imwrite(f"captures/s_{self.count}.jpg", s)
                    cv2.imwrite(f"captures/r_{self.count}.jpg", r)
                    print(f"[Saved] Image set {self.count}")
                    self.count += 1
                elif key == ord('q'):
                    print("[EXIT] Quit capture.")
                    break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    CameraCapture()
