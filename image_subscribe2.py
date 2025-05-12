#!/usr/bin/env python3  # 이 스크립트를 Python 3 인터프리터로 실행하라는 의미

import rospy  # ROS Python 인터페이스
from sensor_msgs.msg import Image  # 이미지 메시지 타입
from cv_bridge import CvBridge, CvBridgeError  # ROS 이미지 ↔ OpenCV 이미지 변환 도구
import cv2  # OpenCV
import numpy as np  # 배열 및 수치 연산용

class ImageSubscriber:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('image_subscriber_node', anonymous=True)

        # CvBridge 객체 생성 (ROS 이미지 ↔ OpenCV 변환용)
        self.bridge = CvBridge()

        # '/camera/image' 토픽을 구독하고, 메시지가 올 때마다 image_callback 호출
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)

    def image_callback(self, data):
        try:
            # ROS Image 메시지를 OpenCV BGR 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            # 변환 에러 발생 시 ROS 로그에 에러 메시지 출력
            rospy.logerr("CvBridge Error: {}".format(e))
            return

        # BGR 이미지를 각각 채널로 분리 (OpenCV는 BGR 순서임)
        B, G, R = cv2.split(cv_image)

        # Grayscale 이미지로 변환
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 각 채널 이미지를 강조한 컬러 이미지 생성
        # R 채널만 강조: (0, 0, R)
        R_img = cv2.merge([np.zeros_like(R), np.zeros_like(R), R])

        # G 채널만 강조: (0, G, 0)
        G_img = cv2.merge([np.zeros_like(G), G, np.zeros_like(G)])

        # B 채널만 강조: (B, 0, 0)
        B_img = cv2.merge([B, np.zeros_like(B), np.zeros_like(B)])

        # 원본 이미지와 각 채널을 시각적으로 출력
        cv2.imshow("Original RGB", cv_image)
        cv2.imshow("Red Channel", R_img)
        cv2.imshow("Green Channel", G_img)
        cv2.imshow("Blue Channel", B_img)
        cv2.imshow("Grayscale", gray)

        # OpenCV 이벤트 처리 및 잠깐 멈춤 (없으면 창이 안 뜰 수 있음)
        cv2.waitKey(1)

    def run(self):
        # ROS 노드가 종료될 때까지 콜백 대기
        rospy.spin()
        # 종료 시 OpenCV 창 모두 닫기
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ImageSubscriber()  # 객체 생성 (생성과 동시에 구독 시작됨)
        node.run()  # 콜백 대기 진입
    except rospy.ROSInterruptException:
        pass  # Ctrl+C 등으로 종료 시 예외 무시
