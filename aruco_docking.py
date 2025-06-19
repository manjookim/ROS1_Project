#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class ArucoDockingNode:
    def __init__(self):
        rospy.init_node('aruco_docking_node', anonymous=True)
        
        # 카메라 파라미터 (실제 캘리브레이션 값으로 교체)
        self.marker_length = rospy.get_param("~marker_length", 0.1)  # 마커 길이 [m]
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix", 
            [506.73737097, 0.0, 316.26249958,
             0.0, 506.68959373, 235.44052887,
            0.0, 0.0, 1.0])).reshape((3,3))

        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs", 
            [0.146345454, 0.04371783, 0.00114179444, 0.00140841683, -1.19683513]))
        
        # 타겟 마커 ID 설정 (ID=1)
        self.target_id = 1
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        rospy.loginfo(f"ArUco Docking Node Started (Targeting Marker ID={self.target_id})")

    def image_callback(self, msg):
        try:
            # ROS 이미지 → OpenCV 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 1. 왜곡 보정
            undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            
            # 2. 마커 검출
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            # 3. 타겟 마커(ID=1) 검출 처리
            target_detected = False
            if ids is not None:
                # 모든 감지된 마커 중에서 ID=1 찾기
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        
                        # 포즈 추정 (내부파라미터와 왜곡계수 사용)
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # 대상 마커 정보 추출
                        dx = tvec[0][0][0]  # x축(좌우) 거리
                        dz = tvec[0][0][2]  # z축(전방) 거리
                        yaw = math.atan2(dx, dz)  # 회전 각도
                        
                        # 도킹 제어
                        self.control_robot(dz, yaw)
                        
                        # 디버깅 시각화 (옵션)
                        self.visualize(undistorted, [corners[i]], [ids[i]], rvec, tvec, dz, yaw)
                        break  # 타겟 마커 찾았으므로 중단
            
            # 타겟 마커 미감지 시 처리
            if not target_detected:
                # 5초 이상 타겟 마커 미감지 시 정지
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 5.0:
                    self.stop_robot()
                    
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Processing Error: {e}")

    def visualize(self, image, corners, ids, rvec, tvec, distance, angle):
        """디버깅용 시각화 함수"""
        # 마커 경계 및 축 표시
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(image, corners, np.array(ids))
        for i in range(len(ids)):
            if rvec is not None and tvec is not None and len(rvec) > i:
                cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec[i], tvec[i], 0.05)
        
        # 거리/각도 정보 표시 (ID=1 마커 위에 표시)
        corner = corners[i][0]
        center_x = int(np.mean(corner[:, 0]))
        center_y = int(np.mean(corner[:, 1]))
        
        cv2.putText(image, f"Target ID: {ids[i][0]}", (center_x, center_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Dist: {distance:.2f}m", (center_x, center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Angle: {math.degrees(angle):.1f}deg", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 결과 이미지 표시
        cv2.imshow("ArUco Docking (ID=1)", image)
        cv2.waitKey(1)

    def control_robot(self, distance, angle):
        twist = Twist()
        
        # 각도 보정 (0.1 rad ≈ 5.7°)
        if abs(angle) > 0.1:
            twist.angular.z = 0.5 * angle
            rospy.loginfo(f"Angle correction: {math.degrees(angle):.1f}°")
        
        # 전진 접근 (0.25m 이상)
        elif distance > 0.25:
            twist.linear.x = 0.2 * min(distance, 0.5)
            rospy.loginfo(f"Approaching: {distance:.2f}m")
        
        # 도킹 완료
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("Docking completed!")
        
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        rospy.loginfo("Stopping: Target marker (ID=1) not detected")

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ArucoDockingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
