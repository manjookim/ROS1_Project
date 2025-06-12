import cv2
import numpy as np

# 1. 카메라 내부 파라미터 (K) 및 왜곡 계수 (D) - 캘리브레이션 결과로 얻음
# 예시 값입니다. 실제 값으로 대체해야 합니다.
K = np.array([[506.73737097, 0, 316.26249958],
              [0, 506.68959373, 235.44052887],
              [0, 0, 1]], dtype=np.float32)
D = np.array([1.46345454e-01, 4.37178300e-02, 1.14179444e-03, 1.40841683e-03, -1.19683513e+00], dtype=np.float32)

# 2. LiDAR-카메라 외부 파라미터 (R_L2C, t_L2C) - 캘리브레이션 결과로 얻음
# 예시 값입니다. 실제 캘리브레이션으로 얻은 R과 t로 대체해야 합니다.
R_L2C = np.array([[ 0.32384176,  0.85229072,  0.41076397],
                  [-0.9433179,   0.32420462,  0.07101193],
                  [-0.07264877, -0.41047764,  0.90897209]], dtype=np.float32)
t_L2C = np.array([[-3.63888619],
                  [-0.78565351],
                  [-14.79084121]], dtype=np.float32)
# LiDAR 포인트 클라우드 데이터 (N x 3 배열, [X, Y, Z])
# 예시: 가상의 LiDAR 포인트 (실제 .pcd 파일에서 로드)
lidar_points = np.array([
    [8.75663471, 9.58374691, 3.5191474],
    [7.55289745, 0.80543745, 8.64756012],
    [5.9053998, 3.0043745, 0.32570705],
    [1.29465199, 1.42124355, 8.7392416],
    [0.61943209, 0.82270032, 3.22240162],
    [0.13944514, 5.19370604, 1.9353683],
    [2.2159214, 1.62210917, 6.64135504],
    [3.93816376, 2.80119944, 6.25790548],
    [7.02986908, 0.02520386, 6.43059158],
    [5.36995506, 3.03169298, 4.45643711],
    [0.52064025, 1.90104187, 8.38218403],
    [6.06925058, 3.5757587, 6.06803417],
    [3.0392704, 2.84488726, 2.11785054],
    [0.13788325, 0.36342865, 7.30370474],
    [7.57766247, 0.86626297, 7.85523462],
    [0.29204103, 0.33210272, 7.84231472],
    [1.27124739, 5.7932806, 2.0282948],
    [7.20304441, 0.84584707, 0.51132756],
    [4.5179491, 9.79505348, 9.69806004],
    [5.4632597, 4.89277792, 4.55024242],
    [7.10431385, 9.94181824, 4.05857992],
    [2.38563323, 5.76271009, 7.28837872],
    [2.88896894, 3.85141945, 7.90769529],
    [0.96730042, 8.00149155, 7.70392609],
    [5.43766594, 8.21444321, 7.38852453],
    [6.66230917, 5.77349281, 7.79179001],
    [7.80080318, 3.26735687, 3.33648634],
    [7.67222834, 1.28683901, 3.81086302],
    [3.10222507, 5.88493919, 5.44401073],
    [7.42997837, 6.39513254, 4.63181734],
    [4.62828779, 9.79705048, 6.44642544],
    [8.22126579, 5.25574636, 5.76867867],
    [7.52207327, 1.52790022, 6.43996906],
    [7.67835903, 3.9143908, 8.78158474],
    [8.94163513, 9.82015038, 7.0096488]],
 dtype=np.float32)

# 원본 이미지 로드
image_path = '/home/user/Downloads/school_classes/scripts/pair_output/images/frame_0052.png' # 실제 이미지 경로로 변경
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# 3. LiDAR 포인트 투영
projected_points = []
colors = [] # 시각화를 위한 색상 (예: 깊이에 따라)

# 회전 벡터 (Rodrigues)와 이동 벡터 형태로 변환
# cv2.projectPoints는 회전 행렬 대신 회전 벡터를 받습니다.
rvec_L2C, _ = cv2.Rodrigues(R_L2C)

# 각 LiDAR 포인트를 처리
for point_3d_lidar in lidar_points:
    # 1. LiDAR 좌표계 -> 카메라 좌표계
    point_3d_camera = np.dot(R_L2C, point_3d_lidar.reshape(3, 1)) + t_L2C
    
    # 카메라 앞에 있는지 확인 (Z > 0)
    if point_3d_camera[2] > 0:
        # 2. 3D 점을 2D 이미지 평면에 투영
        # cv2.projectPoints는 n x 3 배열을 받으므로 reshape 필요
        # point_3d_lidar (1,3) -> point_3d_lidar.reshape(1,1,3)
        img_points, _ = cv2.projectPoints(point_3d_lidar.reshape(1, 1, 3), 
                                          rvec_L2C, 
                                          t_L2C, 
                                          K, 
                                          D)
        
        # 픽셀 좌표 추출
        u, v = int(img_points[0][0][0]), int(img_points[0][0][1])

        # 이미지 경계 내에 있는지 확인
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            projected_points.append((u, v))
            # 깊이(Z 값)에 따른 색상 (선택 사항)
            depth_normalized = np.clip(point_3d_camera[2] / 20.0, 0.0, 1.0) # 최대 깊이 20m 가정
            color = (0, int(255 * (1 - depth_normalized)), int(255 * depth_normalized)) # BGR
            colors.append(color)

# 이미지에 투영된 포인트 그리기
output_image = image.copy()
for i, (u, v) in enumerate(projected_points):
    cv2.circle(output_image, (u, v), 2, colors[i], -1) # 반지름 2, 색상, 채우기

# 결과 이미지 저장 또는 표시
output_image_path = 'projected_lidar_on_image.jpg'
cv2.imwrite(output_image_path, output_image)
cv2.imshow('LiDAR Projected on Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"LiDAR-Camera 외부 파라미터 (R):\n{R_L2C}")
print(f"LiDAR-Camera 외부 파라미터 (t):\n{t_L2C}")
print(f"투영된 이미지가 '{output_image_path}' 에 저장되었습니다.")
