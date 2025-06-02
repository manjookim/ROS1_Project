import cv2
import numpy as np
import glob

# 체커보드 크기
CHECKERBOARD = (8, 6)

# 월드 좌표계 기준 포인트 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []  # 3D 점
imgpoints = []  # 2D 점

images = glob.glob('calib_images/*.jpg')  # 캘리브레이션용 이미지 폴더

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# 내부 파라미터 및 왜곡 계수 추정
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("카메라 행렬 (내부 파라미터):\n", mtx)
print("왜곡 계수:\n", dist)
