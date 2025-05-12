#!/usr/bin/env python3  # 이 파일을 Python 3 인터프리터로 실행하라는 뜻

import rospy  # ROS Python API
from sensor_msgs.msg import LaserScan  # 라이다 센서 메시지 타입

# LaserScan 메시지가 수신될 때마다 호출되는 콜백 함수
def lidar_callback(msg):
    # 필터링된 메시지를 저장할 새로운 LaserScan 메시지 생성
    filtered = LaserScan()

    # 원본 메시지의 메타데이터 복사
    filtered.header = msg.header
    filtered.angle_min = msg.angle_min
    filtered.angle_max = msg.angle_max
    filtered.angle_increment = msg.angle_increment
    filtered.time_increment = msg.time_increment
    filtered.scan_time = msg.scan_time
    filtered.range_min = msg.range_min
    filtered.range_max = msg.range_max

    # 전체 데이터 개수 (각도 수)
    total_angles = len(msg.ranges)

    # 90도에 해당하는 인덱스 수 계산 (라디안 기준 π/2 = 1.5708)
    angle_90_count = int((1.5708) / msg.angle_increment)

    # ranges 배열에서 "정면"의 인덱스 (전체 각도의 절반 지점)
    center_index = total_angles // 2

    # 앞쪽 90도를 자르기 위한 범위 인덱스 계산 (중심 기준 양옆 45도)
    start_cut = center_index - angle_90_count // 2
    end_cut = center_index + angle_90_count // 2

    # 원래 ranges를 복사해서 앞쪽 90도 영역만 float('inf')로 덮어쓰기
    new_ranges = list(msg.ranges)
    for i in range(start_cut, end_cut):
        new_ranges[i] = float('inf')  # RViz 등에서 무시되게 함

    # 필터링된 ranges 적용
    filtered.ranges = new_ranges

    # intensities는 비워도 무방 (필요 시 msg.intensities 복사 가능)
    filtered.intensities = []

    # 필터링된 LaserScan 메시지를 새 토픽으로 퍼블리시
    pub.publish(filtered)

# 노드 메인 함수
if __name__ == '__main__':
    # ROS 노드 초기화
    rospy.init_node('lidar_filter_node')

    # 퍼블리셔 설정: 필터링된 데이터 → '/scan_filtered' 토픽
    pub = rospy.Publisher('/scan_filtered', LaserScan, queue_size=10)

    # 서브스크라이버 설정: 원본 LiDAR 데이터 ← '/scan' 토픽
    sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)

    # 콜백을 계속 기다리며 노드 실행 유지
    rospy.spin()
