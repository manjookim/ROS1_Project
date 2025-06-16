#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import pickle
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_matrix # rvec을 쿼터니언으로 변환하기 위함

# --- 카메라 내부 파라미터 로드 또는 정의 ---
try:
    with open('camera_calibration.pkl', 'rb') as f:
        camera_params = pickle.load(f)
    mtx = camera_params['camera_matrix']
    dist = camera_params['dist_coeffs']
    rospy.loginfo("카메라 내부 파라미터를 'camera_calibration.pkl'에서 로드했습니다.")
except FileNotFoundError:
    rospy.logwarn("경고: 'camera_calibration.pkl' 파일을 찾을 수 없습니다. 기본 카메라 파라미터를 사용합니다.")
    rospy.logwarn("정확한 위치 추정을 위해 카메라 캘리브레이션이 필수적입니다.")
    # 예시 값 (실제 카메라 캘리브레이션으로 얻은 값을 사용해야 함)
    mtx = np.array([[506.73737097, 0, 316.26249958],
                    [0, 506.68959373, 235.44052887],
                    [0, 0, 1]], dtype=np.float32)
    dist = np.array([0.146345454, 0.04371783, 0.00114179444,0.00140841683, -1.19683513], dtype=np.float32) # 왜곡 계수 (보통 0에 가깝거나 작음)

# --- ArUco 마커 설정 ---
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
MARKER_ID = 1
MARKER_LENGTH = 0.05 # 마커의 실제 변 길이 (미터 단위) - 중요!

# --- ROS 노드 초기화 및 퍼블리셔 설정 ---
rospy.init_node('aruco_pose_publisher', anonymous=True)
pose_pub = rospy.Publisher('/aruco_pose', PoseStamped, queue_size=1)
rate = rospy.Rate(30) # 30 Hz로 퍼블리싱

# --- 카메라 초기화 ---
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

# 카메라 워밍업
time.sleep(0.1)

rospy.loginfo("ArUco 마커 인식 및 Pose 발행을 시작합니다. 'q'를 누르면 종료됩니다.")

# 마커의 3D 객체 포인트 정의 (solvePnP용)
# 마커 중심을 원점으로 둠
obj_points = np.array([[-MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
                       [MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
                       [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
                       [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]], dtype=np.float32)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_threaded_capture=True):
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ArUco 마커 탐지
    corners, ids, rejected = aruco.detectMarkers(gray, ARUCO_DICT)

    if ids is not None and MARKER_ID in ids:
        # 특정 마커 ID 찾기
        idx = np.where(ids == MARKER_ID)[0][0]
        marker_corners = corners[idx]

        # solvePnP를 사용하여 rvec (회전 벡터) 및 tvec (병진 벡터) 계산
        ret, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, mtx, dist)

        if ret:
            # 마커 위에 축 그리기 (시각화)
            cv2.drawFrameAxes(image, mtx, dist, rvec, tvec, MARKER_LENGTH * 0.5)

            # --- PoseStamped 메시지 생성 및 발행 ---
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "camera_link" # 카메라 프레임 ID

            # Translation Vector (위치) 설정
            pose_msg.pose.position.x = tvec[0][0]
            pose_msg.pose.position.y = tvec[1][0]
            pose_msg.pose.position.z = tvec[2][0]

            # Rotation Vector (자세)를 Quaternion으로 변환하여 설정
            # rvec을 회전 행렬로 변환
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            # 회전 행렬을 Quaternion으로 변환
            # tf.transformations.quaternion_from_matrix는 4x4 행렬을 기대하므로, 3x3 회전 행렬을 확장
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            q = quaternion_from_matrix(transform_matrix)

            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]

            pose_pub.publish(pose_msg)
            rospy.loginfo(f"Published ArUco Pose: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}")

    # 결과 이미지 표시
    cv2.imshow("ArUco Detection", image)

    # 'q'를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or rospy.is_shutdown():
        break

    rawCapture.truncate(0) # 다음 프레임을 위해 버퍼 비우기
    rate.sleep()

cv2.destroyAllWindows()
camera.close()
rospy.loginfo("ArUco 마커 인식 및 위치 추정 종료.")
