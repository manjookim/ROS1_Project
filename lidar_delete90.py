#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan

def lidar_callback(msg):
    filtered = LaserScan()
    filtered.header = msg.header
    filtered.angle_min = msg.angle_min
    filtered.angle_max = msg.angle_max
    filtered.angle_increment = msg.angle_increment
    filtered.time_increment = msg.time_increment
    filtered.scan_time = msg.scan_time
    filtered.range_min = msg.range_min
    filtered.range_max = msg.range_max

    total_angles = len(msg.ranges)
    angle_90_count = int((1.5708) / msg.angle_increment)  # 90도 = π/2
    center_index = total_angles // 2  # 정면 기준

    start_cut = center_index - angle_90_count // 2
    end_cut = center_index + angle_90_count // 2

    new_ranges = list(msg.ranges)
    for i in range(start_cut, end_cut):
        new_ranges[i] = float('inf')

    filtered.ranges = new_ranges
    filtered.intensities = []

    pub.publish(filtered)

if __name__ == '__main__':
    rospy.init_node('lidar_filter_node')
    pub = rospy.Publisher('/scan_filtered', LaserScan, queue_size=10)
    sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)
    rospy.spin()
