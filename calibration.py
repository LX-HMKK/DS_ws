import cv2
import numpy as np
import os
import yaml
from datetime import datetime
from HIK_CAM import HikIndustrialCamera

# 设置棋盘格参数 (内角点数量)
CHESSBOARD_SIZE = (11, 8)  # 根据实际棋盘格修改 (宽度, 高度)
SQUARE_SIZE = 30.0       # 棋盘格方块实际尺寸(毫米)

# 创建保存图像的文件夹
if not os.path.exists('D:\StudyWorks\Study_python\DS_ws\calibration_results'):
    os.makedirs('calibration_images')

# 准备物体点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 存储物体点和图像点的数组
obj_points = []  # 真实世界3D点
img_points = []  # 图像中的2D点

# 初始化摄像头
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("无法打开摄像头！")
#     exit()

# # 设置摄像头分辨率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera = HikIndustrialCamera()
camera.init()
camera.open()

print("相机标定程序")
print("操作说明:")
print("1. 将棋盘格放在摄像头前，确保完整可见")
print("2. 当检测到棋盘格时，程序会自动捕捉图像")
print("3. 按 's' 键手动保存当前帧")
print("4. 按 'c' 键开始标定 (至少需要5张有效图像)")
print("5. 按 'q' 键退出程序")

image_count = 0
last_capture_time = 0
min_capture_interval = 2  # 最小捕捉间隔(秒)

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     print("无法获取帧")
    #     break
    frame = camera.get_frame()
    # frame = frame[664:1384,864:1584]
    frame = frame[124:1924,324:2124]
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret_chess, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    # 如果找到棋盘格，绘制角点
    if ret_chess:
        # 亚像素级精确化
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # 绘制棋盘格角点
        frame_display = frame.copy()
        cv2.drawChessboardCorners(frame_display, CHESSBOARD_SIZE, corners_refined, ret_chess)
        
        # 自动捕捉图像 (每2秒最多捕捉一次)
        current_time = datetime.now().timestamp()
        if current_time - last_capture_time > min_capture_interval:
            last_capture_time = current_time
            image_count += 1
            filename = f"calibration_images/calib_{image_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"have got {filename}")
            
            # 存储物体点和图像点
            obj_points.append(objp)
            img_points.append(corners_refined)
    else:
        frame_display = frame.copy()
    
    # 显示状态信息
    status_text = f"have got: {image_count} mast>=25"
    cv2.putText(frame_display, status_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_display, "'s':to save, 'c':calibration, 'q':quit", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 显示图像
    cv2.namedWindow("Camera Calibration - Press q to quit",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Calibration - Press q to quit",1000,1000)
    cv2.imshow('Camera Calibration - Press q to quit', frame_display)
    
    # 键盘控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 退出
        break
    elif key == ord('s'):  # 手动保存当前帧
        image_count += 1
        filename = f"calibration_images/calib_{image_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"手动保存图像: {filename}")
        
        # 如果检测到棋盘格，保存点数据
        if ret_chess:
            obj_points.append(objp)
            img_points.append(corners_refined)
    elif key == ord('c') and len(obj_points) >= 15:  # 开始标定
        print("\n开始相机标定...")
        break

# 释放摄像头
# cap.release()
camera.close()
cv2.destroyAllWindows()

# 检查是否有足够的图像进行标定
if len(obj_points) < 15:
    print(f"\n有效标定图像不足: {len(obj_points)}张。需要至少5张图像进行标定。")
    exit()

# 获取图像尺寸
img_shape = gray.shape[::-1]

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img_shape, None, None)

# 打印标定结果
print("\n标定结果:")
print(f"重投影误差: {ret:.5f}")
print(f"相机内参矩阵:\n {mtx}")
print(f"畸变系数: {dist.ravel()}")

# 计算平均重投影误差
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(
        obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error
print(f"平均重投影误差: {mean_error/len(obj_points):.5f} 像素")

# 保存标定结果到YAML文件
calibration_data = {
    'camera_matrix': mtx.tolist(),
    'distortion_coefficients': dist.tolist(),
    'reprojection_error': float(ret),
    'image_size': list(img_shape),
    'chessboard_size': list(CHESSBOARD_SIZE),
    'square_size': SQUARE_SIZE,
    'calibration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'num_calibration_images': len(obj_points)
}

with open('camera_calibration.yaml', 'w') as f:
    yaml.dump(calibration_data, f, default_flow_style=False)

print("\n标定结果已保存到 camera_calibration.yaml")

# 测试去畸变效果
camera = HikIndustrialCamera()
camera.init()
camera.open()

# 计算优化后的相机矩阵
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, img_shape, 1, img_shape)

print("\n按任意键退出预览...")
while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break
    frame = camera.get_frame()
    # frame = frame[664:1384,864:1584]
    frame = frame[124:1924,324:2124]
    # 去畸变
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    # 并排显示原始和去畸变图像
    combined = np.hstack((frame, dst))
    
    # 添加标题
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined, "Undistorted", (img_shape[0] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Original vs Undistorted (Press any key to exit)', combined)
    
    if cv2.waitKey(1) != -1:
        break

# 释放资源
camera.close()
cv2.destroyAllWindows()