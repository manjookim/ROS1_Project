import cv2
import numpy as np

# ▶️ 2D 이미지 상의 점들 (픽셀 좌표계)
points_2D = np.array([
    (102, 269), (198, 262), (287, 301),
    (373, 301), (491, 291), (555, 288)
], dtype="double")

# ▶️ 3D 실제 공간의 점들 (월드 좌표계, 단위: m)
points_3D = np.array([
    (0.829593, 0.378840, 0),
    (0.846180, 0.248804, 0),
    (1.753914, 0.259447, 0),
    (1.805419, -0.045797, 0),
    (1.338860, -0.334447, 0),
    (1.337416, -0.471215, 0)
], dtype="double")

# ▶️ 카메라 내부 파라미터 (Intrinsics)
cameraMatrix = np.array([
    [506.73737097, 0, 316.26249958],
    [0, 506.68959373, 235.44052887],
    [0, 0, 1]
], dtype="double")

# ▶️ 렌즈 왜곡 계수 (Distortion Coefficients)
dist_coeffs = np.array([
    0.146345454, 0.04371783, 0.00114179444,
    0.00140841683, -1.19683513
], dtype="double")

# ▶️ 외부 파라미터 계산 (solvePnP)
retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, cameraMatrix, dist_coeffs)

# ▶️ 회전 벡터 → 회전 행렬로 변환
R, _ = cv2.Rodrigues(rvec)

# ▶️ 결과 출력
if retval:
    print("✅ solvePnP 성공!")
    print("회전 행렬 R:\n", R)
    print("\n이동 벡터 t:\n", tvec)
else:
    print("❌ solvePnP 실패! 입력값을 확인하세요.")
