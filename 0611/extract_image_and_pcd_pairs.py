#!/usr/bin/env python3

import rospy
import rosbag
from sensor_msgs.msg import LaserScan, Image
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from pypcd import PointCloud
from cv_bridge import CvBridge
import cv2
import os

def convert_pointcloud(cloud_msg):
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.array(list(gen), dtype=np.float32)

def save_pcd(points, filename):
    if points.shape[0] == 0:
        print(f"⚠️ {filename}: 포인트 없음")
        return
    pc = PointCloud.from_array(points)
    pc.save_pcd(filename, compression='binary')

def main():
    rospy.init_node("extract_pairs", anonymous=True)

    bag_path = "../calibration.bag"
    bag = rosbag.Bag(bag_path)
    image_topic = "/camera/image"   # 원래는 /camera/image_raw
    scan_topic = "/scan"

    bridge = CvBridge()
    lp = LaserProjection()

    save_img_dir = "./pair_output/images"
    save_pcd_dir = "./pair_output/pcds"
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_pcd_dir, exist_ok=True)

    # 메시지 모으기
    image_msgs = []
    scan_msgs = []

    for topic, msg, t in bag.read_messages(topics=[image_topic, scan_topic]):
        if topic == image_topic:
            image_msgs.append((t, msg))
        elif topic == scan_topic:
            scan_msgs.append((t, msg))

    print(f"📸 이미지 수: {len(image_msgs)}, 🌐 스캔 수: {len(scan_msgs)}")

    # 이미지와 스캔 쌍 매칭
    used_scan = set()
    pair_count = 0

    for img_time, img_msg in image_msgs:
        # 가장 가까운 스캔 찾기
        closest_scan = None
        min_diff = float("inf")
        for scan_time, scan_msg in scan_msgs:
            if scan_time in used_scan:
                continue
            diff = abs((img_time - scan_time).to_sec())
            if diff < min_diff and diff < 0.1:  # 0.1초 이내만 허용
                min_diff = diff
                closest_scan = (scan_time, scan_msg)
        if closest_scan is None:
            continue

        scan_time, scan_msg = closest_scan
        used_scan.add(scan_time)

        # 저장
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            image_path = os.path.join(save_img_dir, f"frame_{pair_count:04d}.png")
            cv2.imwrite(image_path, cv_image)

            cloud_msg = lp.projectLaser(scan_msg)
            cloud_msg.header.frame_id = "base_scan"
            points = convert_pointcloud(cloud_msg)
            pcd_path = os.path.join(save_pcd_dir, f"frame_{pair_count:04d}.pcd")
            save_pcd(points, pcd_path)

            print(f"✅ 저장 완료: frame_{pair_count:04d}.png + .pcd")
            pair_count += 1
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            continue

    bag.close()
    print(f"🎉 총 {pair_count} 쌍 저장 완료")

if __name__ == "__main__":
    main()

