#!/bin/bash
# image_saver 노드를 실행해서 이미지 저장
rosrun image_view image_saver _filename_format:=calib_images/img%03d.jpg image:=/camera/image_raw

# 실행 전 반드시 실행 권한을 부여해야한다.
chmod +x save_image.sh
