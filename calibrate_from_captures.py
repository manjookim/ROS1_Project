#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (7, 5)
SQUARE_SIZE = 0.03  # 체커보드 한 칸 크기(m)

# 3D 좌표 설정
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

image_dir = "/home/s0415j/catkin_ws/src/school_classes/captures"
image_paths = sorted(glob.glob(os.path.join(image_dir, "q_*.jpg")))

if not image_paths:
    print("❌ 이미지가 없습니다: ./captures/q_*.jpg 확인하세요.")
    exit(1)

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # 서브픽셀 보정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"⚠️ 코너 인식 실패: {fname}")

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("📷 Camera Matrix (내부 파라미터):\n", mtx)
print("🎯 Distortion Coefficients (왜곡 계수):\n", dist)

np.savez("camera_calibration_result.npz", camera_matrix=mtx, dist_coeffs=dist)
print("💾 calibration 결과가 camera_calibration_result.npz로 저장되었습니다.")
