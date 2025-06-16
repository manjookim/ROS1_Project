#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, PoseStamped
import math
import numpy as np
from tf.transformations import euler_from_quaternion # Quaternion을 Euler 각도로 변환하기 위함

# --- ROS 노드 초기화 ---
rospy.init_node('aruco_navigator', anonymous=True)
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10) # TurtleBot3 이동 제어 토픽

# --- 이동 파라미터 설정 ---
LINEAR_SPEED = 0.1 # 선형 속도 (m/s)
ANGULAR_SPEED = 0.3 # 각속도 (rad/s)
DISTANCE_THRESHOLD = 0.05 # 목표 지점까지의 허용 오차 (m)
ANGLE_THRESHOLD = 0.05 # 목표 방향까지의 허용 오차 (rad)

# --- 현재 ArUco 마커의 Pose 정보를 저장할 변수 ---
# PoseStamped 메시지로부터 업데이트될 것입니다.
current_position_x = 0.0
current_position_y = 0.0
current_position_z = 0.0
current_orientation_yaw = 0.0 # 마커의 yaw 각도 (로봇의 정렬에 사용)

# --- 콜백 함수: /aruco_pose 토픽으로부터 PoseStamped 메시지 수신 ---
def aruco_pose_callback(msg):
    global current_position_x, current_position_y, current_position_z, current_orientation_yaw

    # PoseStamped 메시지에서 위치 정보 추출
    current_position_x = msg.pose.position.x
    current_position_y = msg.pose.position.y
    current_position_z = msg.pose.position.z

    # PoseStamped 메시지에서 자세(Quaternion) 정보 추출 및 Euler 각도로 변환
    orientation_q = msg.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    current_orientation_yaw = yaw # 마커의 yaw 각도 (z축 회전)

    # rospy.loginfo(f"Received ArUco Pose: x={current_position_x:.3f}, y={current_position_y:.3f}, z={current_position_z:.3f}, yaw={math.degrees(current_orientation_yaw):.2f}°")


def navigate_to_aruco():
    # /aruco_pose 토픽 구독
    rospy.Subscriber('/aruco_pose', PoseStamped, aruco_pose_callback)

    rate = rospy.Rate(10) # 10 Hz

    rospy.loginfo("ArUco 마커를 향한 내비게이션을 시작합니다. /aruco_pose 토픽을 기다립니다.")

    # 마커 위치 정보가 수신될 때까지 대기
    # 최소 한 번은 콜백이 호출되어 초기 위치를 받아야 함
    rospy.wait_for_message('/aruco_pose', PoseStamped, timeout=None)
    rospy.loginfo("ArUco 마커 위치 정보 수신 시작.")

    while not rospy.is_shutdown():
        twist = Twist()

        # 카메라의 Z축 (current_position_z)은 로봇의 전방(X축) 거리와 유사
        # 카메라의 X축 (current_position_x)은 로봇의 좌우(Y축) 위치와 유사
        # TurtleBot3는 X축 전진, Z축 회전으로 제어

        # 1. 목표 각도 계산 (마커가 로봇의 정면에서 얼마나 틀어져 있는지)
        # atan2(y, x)를 사용하여 각도 계산. 여기서 y는 카메라 X (좌우), x는 카메라 Z (앞뒤)
        angle_to_marker = math.atan2(current_position_x, current_position_z)

        # 2. 마커의 자세(Yaw)를 이용한 최종 목표 방향 보정 (선택 사항)
        # 로봇이 마커를 정면으로 바라보게 하려면 마커의 Yaw 각도만큼 로봇도 회전해야 할 수 있습니다.
        # 여기서는 단순히 마커 위치로 이동하므로 마커의 자체 회전은 큰 의미가 없을 수 있습니다.
        # 하지만 만약 마커에 특정 방향성이 있고 로봇이 그 방향을 바라봐야 한다면 current_orientation_yaw를 활용할 수 있습니다.
        # 예: angle_to_marker += current_orientation_yaw # 마커의 Yaw에 맞춰 로봇도 회전

        # 현재 로봇과 마커 사이의 2D 거리 계산 (X-Z 평면)
        distance_to_marker = math.sqrt(current_position_x**2 + current_position_z**2)


        rospy.loginfo(f"Distance: {distance_to_marker:.3f}m, Angle: {math.degrees(angle_to_marker):.2f}°")

        # 1. 목표 각도까지 회전 (마커를 바라보도록)
        if abs(angle_to_marker) > ANGLE_THRESHOLD:
            # 마커가 오른쪽에 있으면 반시계(음수) 회전, 왼쪽에 있으면 시계(양수) 회전
            twist.angular.z = -ANGULAR_SPEED if angle_to_marker > 0 else ANGULAR_SPEED
            twist.linear.x = 0.0 # 회전 중에는 전진하지 않음
            rospy.loginfo(f"Rotating: {twist.angular.z:.2f} rad/s")
        # 2. 목표 거리까지 전진
        elif distance_to_marker > DISTANCE_THRESHOLD:
            twist.linear.x = LINEAR_SPEED
            twist.angular.z = 0.0 # 전진 중에는 회전하지 않음
            rospy.loginfo(f"Moving forward: {twist.linear.x:.2f} m/s")
        # 3. 목표에 도달
        else:
            rospy.loginfo("Reached target!")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            cmd_vel_pub.publish(twist)
            break # 목표에 도달했으므로 루프 종료 (또는 다음 마커 탐지 대기)

        cmd_vel_pub.publish(twist)
        rate.sleep()

    rospy.loginfo("Navigation finished.")
    cmd_vel_pub.publish(Twist()) # 정지

if __name__ == '__main__':
    try:
        navigate_to_aruco()
    except rospy.ROSInterruptException:
        pass
