import cv2
import numpy as np

# 2D 이미지 상의 점들 (이미지 좌표계)
points_2D = np.array([(102, 269), (198, 262), (287, 301), (373, 301), (491, 291), (555, 288)], dtype="double")

# 3D 실제 공간의 점들 (월드 좌표계)
points_3D = np.array([
    (-1.0732, -0.17268, 0),
    (-1.6316, -0.17515, 0),
    (-1.6426,  0.39314, 0),
    (-1.6522,  0.68096, 0)
], dtype="double")

# 🔽 네가 이미 갖고 있는 intrinsic 값 불러오기
# 예: np.load("camera_intrinsics.npz") 또는 직접 행렬로 설정
cameraMatrix = np.array([[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]], dtype="double")

# 🔽 왜곡 계수도 이미 있다면 같이 사용
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype="double")

# ✅ solvePnP 실행해서 외부 파라미터 구하기
retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, cameraMatrix, dist_coeffs)

# 회전벡터 → 회전행렬 변환
R, _ = cv2.Rodrigues(rvec)

print("R = \n", R)
print("t = \n", tvec)
