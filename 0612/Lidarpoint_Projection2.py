import numpy as np
import cv2
import open3d as o3d

def improved_lidar_camera_projection(pcd_file, image_file, camera_params, extrinsic_params):
    """
    개선된 LiDAR-Camera 투영 함수
    
    Args:
        pcd_file: PCD 파일 경로
        image_file: 이미지 파일 경로
        camera_params: 카메라 내부 파라미터 딕셔너리
        extrinsic_params: 외부 파라미터 딕셔너리
    """
    
    # 카메라 파라미터 추출 및 검증
    K = camera_params['camera_matrix']
    D = camera_params['distortion_coeffs']
    
    # 외부 파라미터 추출
    R_L2C = extrinsic_params['rotation_matrix']
    t_L2C = extrinsic_params['translation_vector']
    
    # 파라미터 유효성 검증
    if not validate_camera_parameters(K, D, R_L2C, t_L2C):
        raise ValueError("유효하지 않은 카메라 파라미터입니다.")
    
    # 회전 행렬을 회전 벡터로 변환
    rvec_L2C, _ = cv2.Rodrigues(R_L2C)
    
    try:
        # PCD 파일 로드
        pcd = o3d.io.read_point_cloud(pcd_file)
        lidar_points = np.asarray(pcd.points)
        
        if len(lidar_points) == 0:
            raise ValueError("PCD 파일에 포인트가 없습니다.")
        
        # 이미지 로드
        image = cv2.imread(image_file)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_file}")
        
        # 배치 투영 수행
        projected_points, valid_indices = batch_project_points(
            lidar_points, rvec_L2C, t_L2C, K, D, image.shape[:2]
        )
        
        # 결과 시각화
        result_image = visualize_projection(
            image, projected_points, lidar_points[valid_indices]
        )
        
        return result_image, projected_points, valid_indices
        
    except Exception as e:
        print(f"투영 중 오류 발생: {e}")
        return None, None, None

def batch_project_points(lidar_points, rvec, tvec, camera_matrix, dist_coeffs, image_shape):
    """
    배치 방식으로 LiDAR 포인트를 카메라 이미지에 투영
    
    Returns:
        projected_points: 투영된 2D 포인트들
        valid_indices: 유효한 포인트들의 인덱스
    """
    
    # 먼저 카메라 좌표계로 변환하여 깊이 확인
    R, _ = cv2.Rodrigues(rvec)
    points_camera = np.dot(R, lidar_points.T).T + tvec.T
    
    # 카메라 앞에 있는 포인트들만 선택 (Z > 0)
    front_mask = points_camera[:, 2] > 0
    
    if not np.any(front_mask):
        return np.array([]), np.array([])
    
    valid_lidar_points = lidar_points[front_mask]
    
    # OpenCV projectPoints를 사용하여 배치 투영
    # 중복 변환 제거: projectPoints가 내부적으로 변환 수행
    img_points, _ = cv2.projectPoints(
        valid_lidar_points.reshape(-1, 1, 3).astype(np.float32),
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        camera_matrix.astype(np.float32),
        dist_coeffs.astype(np.float32)
    )
    
    # 결과를 2D 배열로 변환
    img_points = img_points.reshape(-1, 2)
    
    # 이미지 경계 내 포인트들만 선택
    h, w = image_shape
    boundary_mask = (
        (img_points[:, 0] >= 0) & (img_points[:, 0] < w) &
        (img_points[:, 1] >= 0) & (img_points[:, 1] < h)
    )
    
    final_points = img_points[boundary_mask]
    
    # 원래 인덱스 계산
    front_indices = np.where(front_mask)[0]
    valid_indices = front_indices[boundary_mask]
    
    return final_points, valid_indices

def validate_camera_parameters(K, D, R, t):
    """카메라 파라미터 유효성 검증"""
    
    # 카메라 매트릭스 검증
    if K.shape != (3, 3):
        print("카메라 매트릭스 크기가 잘못되었습니다.")
        return False
    
    if K[2, 2] != 1.0:
        print("카메라 매트릭스의 (2,2) 요소는 1이어야 합니다.")
        return False
    
    # 왜곡 계수 검증
    if len(D) not in [4, 5, 8, 12, 14]:
        print(f"왜곡 계수 개수가 잘못되었습니다: {len(D)}")
        return False
    
    # 극도로 큰 왜곡 계수 확인
    if np.any(np.abs(D) > 10):
        print("왜곡 계수가 비정상적으로 큽니다.")
        return False
    
    # 회전 행렬 검증 (직교성)
    if R.shape != (3, 3):
        print("회전 행렬 크기가 잘못되었습니다.")
        return False
    
    # 직교성 확인 (R * R.T = I)
    identity_check = np.dot(R, R.T)
    if not np.allclose(identity_check, np.eye(3), atol=1e-6):
        print("회전 행렬이 직교 행렬이 아닙니다.")
        return False
    
    # 행렬식 확인 (det(R) = 1)
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        print("회전 행렬의 행렬식이 1이 아닙니다.")
        return False
    
    return True

def visualize_projection(image, projected_points, lidar_points_3d):
    """투영 결과 시각화"""
    
    result_image = image.copy()
    
    if len(projected_points) == 0:
        print("투영할 포인트가 없습니다.")
        return result_image
    
    # 깊이에 따른 색상 매핑
    depths = lidar_points_3d[:, 2]  # Z 좌표 (깊이)
    min_depth, max_depth = np.min(depths), np.max(depths)
    
    for i, (point_2d, depth) in enumerate(zip(projected_points, depths)):
        # 깊이에 따른 색상 계산 (가까우면 빨강, 멀면 파랑)
        normalized_depth = (depth - min_depth) / (max_depth - min_depth) if max_depth > min_depth else 0
        color = (
            int(255 * (1 - normalized_depth)),  # Blue
            int(255 * normalized_depth * 0.5),  # Green
            int(255 * normalized_depth)         # Red
        )
        
        # 깊이에 따른 포인트 크기 조절
        radius = max(1, int(5 - 3 * normalized_depth))
        
        cv2.circle(result_image, 
                  (int(point_2d[0]), int(point_2d[1])), 
                  radius, color, -1)
    
    return result_image

def calculate_reprojection_error(projected_points, ground_truth_points):
    """재투영 오차 계산"""
    
    if len(projected_points) != len(ground_truth_points):
        raise ValueError("포인트 수가 일치하지 않습니다.")
    
    errors = np.linalg.norm(projected_points - ground_truth_points, axis=1)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'individual_errors': errors
    }
