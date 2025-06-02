import cv2
import numpy as np

# === 3D LiDAR 포인트 (예시) ===
objectPoints = np.array([
    [0.0, 0.0, 0.0],
    [0.03, 0.0, 0.0],
    [0.06, 0.0, 0.0],
], dtype=np.float32)

# === SolvePnP 결과 ===
rvec = np.array([[0.1], [0.05], [-0.02]])   # 회전
tvec = np.array([[0.1], [0.0], [0.3]])      # 이동

# === Intrinsic Matrix ===
K = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# === 왜곡 계수 (보통 0으로 초기화) ===
dist = np.zeros((5, 1))  # 또는 실제 값

# === 3D → 2D 픽셀 변환 ===
imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, K, dist)

# === 결과 출력 ===
for pt in imagePoints:
    x, y = pt.ravel()
    print(f"Projected Pixel: ({x:.2f}, {y:.2f})")
