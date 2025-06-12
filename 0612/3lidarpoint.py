import open3d as o3d
import numpy as np

def get_lidar_points_from_pcd(pcd_file_path):
    """
    PCD 파일을 읽고 LiDAR 포인트의 3D (X, Y, Z) 좌표를 추출합니다.

    Args:
        pcd_file_path (str): .pcd 파일의 경로.

    Returns:
        numpy.ndarray: N x 3 NumPy 배열 (N은 포인트 수, 각 행은 [X, Y, Z]).
                       파일을 로드할 수 없으면 None을 반환합니다.
    """
    print(f"PCD 파일 로드 중: {pcd_file_path}")
    
    # PCD 파일 읽기
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    if not pcd.has_points():
        print(f"'{pcd_file_path}'에서 포인트를 찾을 수 없습니다.")
        return None

    # 3D 좌표 (X, Y, Z)를 NumPy 배열로 가져옵니다.
    # pcd.points는 open3d.utility.Vector3dVector 객체이며, NumPy로 변환할 수 있습니다.
    points = np.asarray(pcd.points)
    
    print(f"'{points.shape[0]}'개의 포인트를 성공적으로 로드했습니다.")
    
    # PCD 파일에 다른 속성(예: 강도, 색상)이 있는 경우에도 접근할 수 있습니다.
    # 예를 들어, 강도 또는 색상:
    # if pcd.has_colors():
    #     colors = np.asarray(pcd.colors)
    #     print(f"색상 데이터도 로드되었습니다. 형태: {colors.shape}")
    # if pcd.has_normals():
    #     normals = np.asarray(pcd.normals)
    #     print(f"노멀 데이터도 로드되었습니다. 형태: {normals.shape}")

    return points

# 사용 예시:
if __name__ == "__main__":
    # 여기에 실제 PCD 파일 경로를 입력하세요.
    # 예시: 'data/sample.pcd' 또는 'C:/Users/YourUser/Documents/point_cloud.pcd'
    pcd_file = '/home/user/Downloads/school_classes/scripts/pair_output/pcds/frmae_0052.pcd' 
    
    # 더미 PCD 파일 생성 (테스트용)
    # 실제 환경에서는 이 부분 대신 실제 PCD 파일을 사용하세요.
    try:
        dummy_points = np.random.rand(100, 3).astype(np.float32) * 10 # 100개의 랜덤 3D 포인트
        dummy_pcd = o3d.geometry.PointCloud()
        dummy_pcd.points = o3d.utility.Vector3dVector(dummy_points)
        o3d.io.write_point_cloud(pcd_file, dummy_pcd)
        print(f"테스트용 더미 PCD 파일 '{pcd_file}'을 생성했습니다.")
    except Exception as e:
        print(f"더미 PCD 파일 생성 실패: {e}")
        print("실제 PCD 파일 경로를 사용하거나 수동으로 파일을 제공해야 합니다.")
        exit()


    lidar_3d_points = get_lidar_points_from_pcd(pcd_file)

    if lidar_3d_points is not None:
        print("\n추출된 LiDAR 3D 포인트의 처음 5개:")
        print(lidar_3d_points[:100])
        print(f"\n총 포인트 수: {lidar_3d_points.shape[0]}")
        print(f"포인트 배열 형태: {lidar_3d_points.shape}")

        # (선택 사항) 포인트 클라우드 시각화 (테스트용)
        # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(pcd_file)])
