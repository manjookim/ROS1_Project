#!/usr/bin/env python3
import cv2
import numpy as np
import glob

CHECKERBOARD = (8, 6)  # 체커보드 코너 개수
SQUARE_SIZE = 0.025    # 한 칸 크기 (미터)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob('./calib_images/*.jpg')
print(f"[INFO] 총 {len(images)}장의 이미지에서 코너 검출 시도 중...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

print("\n[INFO] 카메라 내부 파라미터 계산 중...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n▶ [내부 파라미터] Camera Matrix:")
print(mtx)

print("\n▶ [왜곡 계수] Distortion Coefficients:")
print(dist)

print("\n[INFO] 첫 번째 이미지에서 Pose (R|t) 추정 중...")
img = cv2.imread(images[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret:
    success, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    R, _ = cv2.Rodrigues(rvec)
    print("\n▶ [회전 행렬] Rotation Matrix R:")
    print(R)
    print("\n▶ [이동 벡터] Translation Vector t:")
    print(tvec)
else:
    print("[ERROR] 첫 번째 이미지에서 체커보드 코너를 찾을 수 없습니다.")

# scripts/solve_pnp.py 위치
# 실행 전 chmod +x solve_pnp.py 해줘야 함

