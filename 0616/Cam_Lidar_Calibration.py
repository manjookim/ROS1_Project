import cv2
import numpy as np

# 2D 이미지 상의 점들 (이미지 좌표계)
points_2D = np.array([(102, 269), (198, 262), (287, 301), (373, 301), (491, 291), (555, 288)], dtype="double")

# 3D 실제 공간의 점들 (월드 좌표계)
points_3D = np.array([
    (0.829593, 0.378840, 0),
    (0.846180, 0.248804, 0),
    (1.753914,  0.259447, 0),
    (1.805419,  -0.045797, 0),
    (1.338860, -0.334447, 0),
    (1.337416, -0.471215, 0)
], dtype="double")

# 🔽 네가 이미 갖고 있는 intrinsic 값 불러오기
# 예: np.load("camera_intrinsics.npz") 또는 직접 행렬로 설정
cameraMatrix = np.array([
[[506.73737097, 0, 316.26249958],
              [0, 506.68959373, 235.44052887],
              [0, 0, 1]], dtype="double")

# 🔽 왜곡 계수도 이미 있다면 같이 사용
dist_coeffs = np.array([1.46345454e-01, 4.37178300e-02, 1.14179444e-03, 1.40841683e-03, -1.19683513e+00], dtype="double")

# ✅ solvePnP 실행해서 외부 파라미터 구하기
retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, cameraMatrix, dist_coeffs)

# 회전벡터 → 회전행렬 변환
R, _ = cv2.Rodrigues(rvec)

print("R = \n", R)
print("t = \n", tvec)
