from tkinter import NO
import cv2
import numpy as np
import yaml
from detect import RectangleDetector
from HIK_CAM import HikIndustrialCamera
from find_shape import ShapeDetector
# from new_find_minSquare import find_smallest_square,draw_results
from find_shape3 import detect_smallest_square
from find_shape2 import CircleDetector
import os
from datetime import datetime

# 读取相机内外参
def read_camera_params(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
    distortion_coeffs = np.array(params['distortion_coefficients'], dtype=np.float32).flatten()
    return camera_matrix, distortion_coeffs

def get_3d_points(rect_width, rect_height):
    # 假设矩形位于Z=0平面，原点位于矩形的一个角点
    obj_points = np.array([
        [0, 0, 0],
        [rect_width, 0, 0],
        [rect_width, rect_height, 0],
        [0, rect_height, 0]
    ], dtype=np.float32)
    return obj_points

# 实现PnP测距
def pnp_distance(camera_matrix, distortion_coeffs, obj_points, img_points):
    # 检查 img_points 是否为空
    if img_points.size == 0:
        return None, None
    # 将 img_points 转换为合适的形状
    img_points = img_points.reshape(-1, 2)
    try:
        # 求解PnP问题
        # success, rotation_vector, translation_vector = cv2.solvePnP(obj_points, img_points, camera_matrix, distortion_coeffs, flags= cv2.SOLVEPNP_IPPE)
        success, rotation_vector, translation_vector = cv2.solvePnP(obj_points, img_points, camera_matrix, distortion_coeffs, flags= cv2.SOLVEPNP_ITERATIVE)
        if success:
            # 提取Z轴分量作为距离
            distance = translation_vector[2][0]
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            return distance,rotation_matrix
        else:
            return None, None
    except cv2.error as e:
        print(f"OpenCV error in solvePnP: {e}")
        return None,None

# 主函数
def main():
    save_dir = "D:/StudyWorks/squares_images/image"
    os.makedirs(save_dir, exist_ok=True)


    camera_matrix, distortion_coeffs = read_camera_params('D:/StudyWorks/Study_python/DS_ws/calibration_results/camera_calibration.yaml')
    
    # rect_width = 200  # mm
    # rect_height = 287  # mm
    rect_width = 170  # mm
    rect_height = 267  # mm
    camera_length = -90.0 #mm
    
    
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    
    # 获取真实世界中的3D点
    obj_points = get_3d_points(rect_width, rect_height)
    
    detector = RectangleDetector()
    detector_shape = ShapeDetector(
        area_ratio_threshold=(0.05, 0.95),
        frame_real_width=rect_width,
        frame_real_height=rect_height
    )
    find_minS = CircleDetector(
        min_area_ratio=0.1,  # 轮廓面积至少占画面10%
        max_area_ratio=0.9,  # 轮廓面积最大占画面90%
        # min_contact_points=36,  # 至少36个接触点
        # min_contact_regions=6,  # 至少6个区域有接触点
        # max_angle_gap=np.pi/2  # 最大间隔90度
    )
    while True:
        frame = camera.get_frame()
        # frame = frame[664:1384,864:1584]
        if frame is not None:
            # frame = frame[124:1924,324:2124]
            corners_list = detector.detect(frame)

            # 遍历每个检测到的矩形
            for corners in corners_list:
                # # # 步骤1: 确定仿射变换的源点和目标点
                src_points = corners  # 检测到的矩形的四个角点

                """左上 左下 右下 右上"""
                dst_points = np.array([
                    [0, 0],
                    [rect_width, 0],
                    [rect_width, rect_height],
                    [0, rect_height]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(src_points, dst_points)
                warped = cv2.warpPerspective(frame, M, (rect_width,rect_height))
                # result, output_image = detector_shape.detect_shape(warped)
                # output_image, size=detect_smallest_square(warped)
                output_image, min_side = find_minS.find_min_square_side(warped)
            
                # 绘制结果
                # output_image = draw_results(warped, square_points, square_size)
                # print(f"形状: {result['shape']}, 尺寸: {result['size']:.1f}mm")

                # 实现PnP测距
                distance,rotation_matrix = pnp_distance(camera_matrix, distortion_coeffs, obj_points, corners)


                            # 显示校正后的图像
                cv2.namedWindow("Warped",cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Warped",rect_width*3,rect_height*3)
                cv2.imshow("Warped", output_image)

                for (x, y) in corners:
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
                if distance is not None and rotation_matrix is not None:
                    # 计算矩形中心位置
                    center = np.mean(corners, axis=0).astype(int)
                    
                    distance_text = f"X: {camera_length + distance:.1f}mm"
                    
                    cv2.putText(frame, distance_text, (center[0], center[1]+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
                    
                    print(f"物体尺寸: {rect_width}mm x {rect_height}mm, 物体到相机的距离: {distance+camera_length} mm")
                    
                else:
                    print("PnP求解失败")

            cv2.namedWindow("demo",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("demo",1000,1000)
            cv2.imshow("demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键保存图像
                # 生成带时间戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"warped_{timestamp}.png")
                # 保存原始warped图像（不是output_image）
                cv2.imwrite(save_path, warped)
                print(f"已保存warped图像到: {save_path}")
            else :
                continue
    
    camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()