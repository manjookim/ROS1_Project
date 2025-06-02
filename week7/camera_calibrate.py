import cv2
import numpy as np
import glob
import os

# === 설정 ===
CHECKERBOARD = (9, 6)     # 내부 코너 수
square_size = 0.03        # 사각형 크기 (3cm)
image_folder = "captures"

# === 3D 체커보드 좌표 만들기 ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D 포인트
imgpoints = []  # 2D 포인트
image_shape = None

images = sorted(glob.glob(os.path.join(image_folder, "q_*.jpg")))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# === 카메라 보정 ===
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

print("✅ Calibration Finished!")
print("Intrinsic Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist)

# === 저장 ===
np.savez("intrinsics.npz", K=K, dist=dist)
cv2.FileStorage("intrinsics.yaml", cv2.FILE_STORAGE_WRITE).write("K", K)
