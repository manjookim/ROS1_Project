#!/usr/bin/env python3
import os

# 이미지가 저장된 폴더 경로
image_folder = './calib_images/'

# make_list.txt 파일 생성
with open('make_list.txt', 'w') as f:
    for fname in sorted(os.listdir(image_folder)):
        if fname.endswith('.jpg'):
            f.write(os.path.join(image_folder, fname) + '\n')

print("[INFO] make_list.txt 파일 생성 완료")

# ./calib_images/ 폴더 내에 있는 .jpg 파일들의 경로를 make_list.txt에 기록
# OpenCV에서 이미지 경로 리스트로 사용 가능
# scripts/ 폴더에 넣고 실행한다 !
chmod +x make_list.py
rosrun camera_pose_pkg make_list.py
