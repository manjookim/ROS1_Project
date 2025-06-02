import cv2
import numpy as np
import glob

# 체커보드 설정
CHECKERBOARD = (9, 6)
square_size = 0.03  # 3cm

# 3D 점 준비
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 변수
objpoints = []  # 실제 3D 포인트
imgpoints = []  # 이미지 내 2D 포인트

# 이미지 불러오기
images = glob.glob("images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 시각화 (선택)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 카메라 보정
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("=== Camera Intrinsic Matrix ===")
print(K)
print("\n=== Distortion Coefficients ===")
print(dist)

# 저장
np.savez("intrinsics.npz", K=K, dist=dist)
