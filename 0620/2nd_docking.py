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
        
        # 카메라 파라미터
        self.marker_length = rospy.get_param("~marker_length", 0.1)
        self.camera_matrix = np.array(rospy.get_param("~camera_matrix")).reshape((3,3))
        self.dist_coeffs = np.array(rospy.get_param("~dist_coeffs"))
        
        self.target_id = 1
        self.target_distance = 0.1  # 10cm
        
        # 2단계 도킹 파라미터
        self.docking_state = "SEARCH"
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.approach_tolerance = 0.05
        
        # 안정성 개선 파라미터
        self.last_safe_image = None  # 마지막 안전한 이미지 저장
        
        # ROS 인터페이스
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 상태 변수
        self.last_marker_time = rospy.Time.now()
        self.odom_yaw = 0.0
        self.odom_received = False
        self.initial_y_position = None
        
        rospy.loginfo("2-Stage ArUco Docking Node Started (Stability Enhanced)")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True
        self.current_position = msg.pose.pose.position

    def image_callback(self, msg):
        try:
            # 이미지 변환 (항상 수행)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_safe_image = cv_image.copy()  # 안전한 이미지 백업
            
            undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            
            # ArUco 검출 파라미터 (안정성 개선)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.03  # 더 큰 마커만 검출
            parameters.polygonalApproxAccuracyRate = 0.05
            
            # 마커 검출 시도
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            target_detected = False
            
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == self.target_id:
                        self.last_marker_time = rospy.Time.now()
                        target_detected = True
                        
                        try:
                            # 포즈 추정 (예외 처리 강화)
                            rvec, tvec, _ = aruco.estimatePoseSingleMarkers([corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs)
                            
                            # 마커 위치 정보 추출
                            self.target_x = tvec[0][0][0]
                            self.target_y = tvec[0][0][1]
                            self.target_z = tvec[0][0][2]
                            
                            # 2단계 도킹 제어
                            self.execute_2stage_docking()
                            
                            # 시각화
                            self.visualize_2stage(undistorted, corners[i], ids[i], rvec, tvec)
                        except Exception as e:
                            rospy.logerr(f"Pose estimation error: {e}")
                            self.docking_state = "SEARCH"
                        break
            
            # 마커 미감지 시 처리
            if not target_detected:
                if (rospy.Time.now() - self.last_marker_time).to_sec() > 2.0:
                    if self.docking_state != "SEARCH":
                        rospy.logwarn("Marker lost! Returning to search mode")
                        self.docking_state = "SEARCH"
                    self.search_for_marker()
                
                # 마커 없어도 화면 갱신
                self.visualize_2stage(undistorted, None, None, None, None)
                    
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            # 이전 안전한 이미지 사용
            if self.last_safe_image is not None:
                cv2.imshow("2-Stage ArUco Docking", self.last_safe_image)
                cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"General Processing Error: {e}")

    def execute_2stage_docking(self):
        twist = Twist()
        
        if self.docking_state == "SEARCH":
            if self.initial_y_position is None and self.odom_received:
                self.initial_y_position = self.current_position.y
            self.docking_state = "APPROACH_Y"
            rospy.loginfo(f"Target found! X:{self.target_x:.2f}, Z:{self.target_z:.2f}")
        
        elif self.docking_state == "APPROACH_Y":
            if abs(self.target_z) > 0.3:
                twist.linear.x = 0.2
                twist.angular.z = 0.1 * self.target_x
            else:
                self.docking_state = "TURN_TO_CENTER"
        
        elif self.docking_state == "TURN_TO_CENTER":
            angle_to_center = math.atan2(self.target_x, self.target_z)
            if abs(angle_to_center) > 0.1:
                twist.angular.z = 0.8 * angle_to_center
                twist.linear.x = 0.0
            else:
                self.docking_state = "FINAL_APPROACH"
        
        elif self.docking_state == "FINAL_APPROACH":
            distance_to_marker = math.sqrt(self.target_x**2 + self.target_z**2)
            if distance_to_marker > self.target_distance:
                # 동적 속도 제어: 거리에 따른 가변 이득
                speed_gain = 0.5 * min(1.0, distance_to_marker / 0.5)  # 50cm 이내에서 감소
                speed = speed_gain * (distance_to_marker - self.target_distance)
                
                # 안전한 속도 범위 설정 (검색결과 2,3 반영)
                twist.linear.x = np.clip(speed, 0.1, 0.22)  # TurtleBot3 최대 속도 0.26m/s의 85%
                
                # 각도 오차 보정 (속도에 비례한 이득)
                angle_error = math.atan2(self.target_x, self.target_z)
                angular_gain = 0.4 + 0.2 * (twist.linear.x / 0.22)  # 속도↑ → 이득↑
                twist.angular.z = angular_gain * angle_error
                
                rospy.loginfo(f"FINAL_APPROACH: Speed={twist.linear.x:.2f}m/s")
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.docking_state = "COMPLETED"

        
        self.cmd_pub.publish(twist)

    def search_for_marker(self):
        twist = Twist()
        
        # 현재 시간 계산
        elapsed = (rospy.Time.now() - self.search_start_time).to_sec()
        
        # 4초 주기로 좌우 왕복 탐색
        if elapsed < 4.0:  # 첫 4초: 우회전
            twist.angular.z = -0.8
        elif elapsed < 8.0:  # 다음 4초: 좌회전
            twist.angular.z = 0.8
        else:  # 8초 경과 시 초기화
            self.search_start_time = rospy.Time.now()
        
        # 탐색 중 느린 전진 추가 (시야 확보)
        twist.linear.x = 0.1
        self.cmd_pub.publish(twist)
        rospy.loginfo(f"Searching: {elapsed:.1f}s, Angular: {twist.angular.z:.1f}")


    def visualize_2stage(self, image, corners, marker_id, rvec, tvec):
        display_image = image.copy()
        
        # 상태 정보 표시
        state_colors = {
            "SEARCH": (0, 0, 255),
            "APPROACH_Y": (0, 255, 255),
            "TURN_TO_CENTER": (255, 255, 0),
            "FINAL_APPROACH": (255, 165, 0),
            "COMPLETED": (0, 255, 0)
        }
        color = state_colors.get(self.docking_state, (255, 255, 255))
        cv2.putText(display_image, f"STATE: {self.docking_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 마커 감지된 경우에만 추가 정보 표시
        if corners is not None and marker_id is not None:
            # 마커 및 축 표시
            aruco.drawDetectedMarkers(display_image, [corners], np.array([[marker_id]]))
            
            if rvec is not None and tvec is not None:
                try:
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
                except Exception as e:
                    rospy.logwarn(f"Axis drawing failed: {e}")
            
            # 중심점 표시
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            cv2.circle(display_image, (center_x, center_y), 10, (0, 0, 255), 2)
            cv2.putText(display_image, "TARGET", (center_x-30, center_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 좌표 정보
            cv2.putText(display_image, f"X: {self.target_x*100:.1f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, f"Z: {self.target_z*100:.1f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_image, "NO MARKER DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 화면 갱신
        cv2.imshow("2-Stage ArUco Docking", display_image)
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
