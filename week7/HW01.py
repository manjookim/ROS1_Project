#1 원격 PC 접속
ssh ubuntu@<TurtleBot3_IP>

#2 turtlebot에서 ros 마스터 실행
export TURTLEBOT3_MODEL=burger  # 또는 waffle
roscore

#3 카메라 드라이버 실행
roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch
#3-1 /camera/image_raw 또는 /image_raw 토픽이 퍼블리시되는지 확인
rostopic list

#4 카메라 이미지 확인 -> 로컬 PC에서 .. ROS_MASTER_URI 설정 후
  # 1. TurtleBot3의 IP를 ROS 마스터로 지정
  export ROS_MASTER_URI=http://192.168.0.10:11311
  # 2. 로컬 PC 자신의 IP를 ROS_HOSTNAME으로 지정
  export ROS_HOSTNAME=192.168.0.20
  # 3. 그리고 저장 후 
  source ~/.bashrc
  # 4. 로컬 PC에서 다음과 같이 실행 -> turtlebot 카메라 토픽을 구독해서 사용 가능
  rqt_image_view
  rosrun image_view image_view image:=/camera/image_raw
  rosrun camera_pose_pkg solve_pnp.py

#5 체커보드 이미지 저장 -> calib_images/ 폴더에 저장
rosrun image_view image_saver _filename_format:=calib_images/img%03d.jpg image:=/camera/image_raw

#6 이미지 리스트 파일 생성
ls ./calib_images/*.jpg > make_list.txt

#7 내부 파라미터 + Pose(R|t) 계산
#!/usr/bin/env python3
import cv2
import numpy as np
import glob

# 체커보드 설정: 내부 코너 개수 (행, 열)
CHECKERBOARD = (8, 6)  # 8 x 6 코너
SQUARE_SIZE = 0.025    # 1칸 크기 (m)

# 3D 객체 포인트 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 3D, 2D 포인트 저장 리스트
objpoints = []  # 실제 체커보드 3D 좌표
imgpoints = []  # 이미지 상의 2D 코너

# 저장된 이미지 불러오기
images = glob.glob('./calib_images/*.jpg')

print(f"[INFO] 총 {len(images)}장의 이미지에서 코너 검출 시도 중...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 디버깅용 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 내부 파라미터 계산
print("\n[INFO] 카메라 내부 파라미터 계산 중...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n▶ [내부 파라미터] Camera Matrix:")
print(mtx)

print("\n▶ [왜곡 계수] Distortion Coefficients:")
print(dist)

# 한 장의 이미지로 Pose (R|t) 추정
print("\n[INFO] 첫 번째 이미지에서 Pose (R|t) 추정 중...")
img = cv2.imread(images[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret:
    success, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    R, _ = cv2.Rodrigues(rvec)

    print("\n▶ [회전 행렬] Rotation Matrix R:")
    print(R)

    print("\n▶ [이동 벡터] Translation Vector t:")
    print(tvec)

else:
    print("[ERROR] 첫 번째 이미지에서 체커보드 코너를 찾을 수 없습니다.")

#8 코드 실행
rosrun camera_pose_pkg solve_pnp.py

                                              
                                              
                                              
