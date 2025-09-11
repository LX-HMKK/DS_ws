from tkinter import NO
import cv2
import numpy as np
import yaml
from detect import RectangleDetector
from HIK_CAM import HikIndustrialCamera
from find_all import ShapeDetector
from find_minSquare import CircleDetector
from find_numSquare import YOLO11_Detector
import os
from datetime import datetime
from send_data import SerialPort
import threading
import queue
import time

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
        success, rotation_vector, translation_vector = cv2.solvePnP(obj_points, img_points, camera_matrix, distortion_coeffs, flags= cv2.SOLVEPNP_ITERATIVE)
        if success:
            # 提取Z轴分量作为距离
            distance = translation_vector[2][0]
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            return distance, rotation_matrix
        else:
            return None, None
    except cv2.error as e:
        print(f"OpenCV error in solvePnP: {e}")
        return None, None

# 串口接收线程函数
def serial_receiver(serial_port, data_queue, port_name):
    while True:
        try:
            data = serial_port.read_data()
            if data:
                data_queue.put((port_name, data))
        except Exception as e:
            print(f"Error reading from {port_name}: {e}")
        time.sleep(0.01)  # 避免CPU占用过高

# 主函数
def main():
    camera_matrix, distortion_coeffs = read_camera_params('/home/sunrise/AAA_DS_WS/calibration_results/camera_calibration.yaml')
    
    rect_width = 170  # mm
    rect_height = 267  # mm
    camera_length = -90.0  # mm
    
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    
    # 获取真实世界中的3D点
    obj_points = get_3d_points(rect_width, rect_height)
    
    detector = RectangleDetector()
    find_all = ShapeDetector(
        area_ratio_threshold=(0.05, 0.95),
        frame_real_width=rect_width,
        frame_real_height=rect_height
    )
    find_minS = CircleDetector(
        min_area_ratio=0.05,  # 轮廓面积至少占画面10%
        max_area_ratio=0.95,  # 轮廓面积最大占画面90%
    )

    find_numS = YOLO11_Detector(
        model_path='/home/sunrise/AAA_DS_WS/DS_NUM.bin',
        conf_thres=0.30,
        iou_thres=0.6
    ) 

    # 创建串口对象
    serial_port_choice = SerialPort(
        port='/dev/ttyS1', 
        baudrate=115200,
        send_format='str',  
        recv_format='str'  
    )

    serial_port_power = SerialPort(
        port='/dev/ttyS3',  # 使用不同的串口
        baudrate=115200,
        send_format='str',  
        recv_format='str'  
    )

    # 创建消息队列
    serial_queue = queue.Queue()

    # 启动串口接收线程
    choice_thread = threading.Thread(
        target=serial_receiver, 
        args=(serial_port_choice, serial_queue, "choice"),
        daemon=True
    )
    power_thread = threading.Thread(
        target=serial_receiver, 
        args=(serial_port_power, serial_queue, "power"),
        daemon=True
    )
    choice_thread.start()
    power_thread.start()

    # 当前检测器状态
    current_detector = None
    current_detector_name = None
    # output_image = None
    power_message = ""
    middle_data_power = ""
    middle_data = ""
    show_output = False
    last_choice_time = 0
    min_choice_interval = 0.5  # 最小选择间隔时间（秒）

    # 创建输出窗口
    cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined View", 1000, 600)  # 设置初始窗口大小
    # cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("demo", 500, 500)
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Output", rect_width * 3, rect_height * 3)
    
    while True:
        # 处理串口消息（非阻塞）
        while not serial_queue.empty():
            port_type, data = serial_queue.get_nowait()
            
            if port_type == "power":
                power_message = data
                # print(f"Power received: {power_message}")

                # if power_message.startswith('I') and power_message.endswith('Z'):
                if  len(power_message) >= 6 and power_message[5] == 'A' and power_message.endswith('W'):
                    # middle_data_power = power_message[1:-1]
                    middle_data_power ="I:"+ power_message[:8] +"P:"+ power_message[8:16] + "PM:" + power_message[16:]
                    print(f"Power received: {middle_data_power}")
            
            elif port_type == "choice":
                current_time = time.time()
                if current_time - last_choice_time < min_choice_interval:
                    print(f"Ignoring choice message (too frequent): {data}")
                    continue
                
                last_choice_time = current_time
                print(f"Choice received: {data}")
                
                # 解析choice消息
                if data.startswith('g') and data.endswith('l'):
                    middle_data = data[1:-1]
                    if 'as' in middle_data:
                        current_detector = find_all.detect_shape
                        current_detector_name = "Shape"
                        show_output = True
                    elif 'bs' in middle_data:
                        current_detector = find_minS.find_min_square_side
                        current_detector_name = "Min Square"
                        show_output = True
                    elif 'd' in middle_data:
                        c_index = middle_data.find('d')
                        if c_index != -1 and c_index + 1 < len(middle_data):
                            num_char = middle_data[c_index + 1]
                            if num_char.isdigit():
                                current_detector = find_numS.process_frame
                                find_numS.set_selected_digit(num_char)
                                current_detector_name = f"Digit ({num_char})"
                                show_output = True
                # 重置输出图像
                # output_image = None
        
        # 视频处理
        frame = camera.get_frame()
        if frame is None: continue

        # frame = frame[124:1924, 324:2124]
        corners_list = detector.detect(frame)

        # current_frame_output = None  
        
        # 在左上角显示power消息
        if middle_data_power:
            cv2.putText(frame, f"Power: {middle_data_power}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # 在右上角显示当前检测器状态
        if current_detector_name:
            cv2.putText(frame, f"Detector: {current_detector_name}", 
                        (frame.shape[1] - 700, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

        size_info = ""
        
        # 初始化输出图像（黑色占位符）
        output_img = np.zeros((600, 400, 3), dtype=np.uint8)
        has_output = False
        
        # 遍历每个检测到的矩形
        for corners in corners_list:
            src_points = corners  # 检测到的矩形的四个角点

            """左上 左下 右下 右上"""
            dst_points = np.array([
                [0, 0],
                [rect_width, 0],
                [rect_width, rect_height],
                [0, rect_height]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(frame, M, (rect_width, rect_height))
            
            # 使用当前检测器处理图像（如果已选择）
            if show_output and current_detector:
                try:
                    # 根据检测器类型调用不同的处理函数
                    if current_detector == find_all.detect_shape:
                        size_val,_ = current_detector(warped)
                        size_info = f"D: {size_val:.1f}mm"
                    elif current_detector == find_minS.find_min_square_side:
                        _, size_min = current_detector(warped)
                        if size_min is not None:
                            size_info = f"D: {size_min:.1f}mm"
                        else:
                            size_info = "D: N/A"  # 未检测到时的默认文本
                    elif current_detector == find_numS.process_frame:
                        _, size_dight = current_detector(warped)
                        if size_dight is not None:
                            size_info = f"D: {size_dight:.1f}mm"
                        else:
                            size_info = "D: Not detected"  # 未检测到时的提示

                except Exception as e:
                    print(f"Error in detector: {e}")
                    size_info = f"Error: {str(e)}"
                   
            
             # 在warped图像上显示尺寸信息
            if size_info:
                cv2.putText(warped, size_info, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 实现PnP测距
            distance, rotation_matrix = pnp_distance(
                camera_matrix, distortion_coeffs, obj_points, corners
            )

            for (x, y) in corners:
                cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
                
            if  current_detector is not None and distance is not None and rotation_matrix is not None:
                center = np.mean(corners, axis=0).astype(int)
                distance_text = f"X: {camera_length + distance:.1f}mm"
                cv2.putText(frame, distance_text, (center[0], center[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 7)
                print(f"物体到相机的距离: {distance + camera_length} mm")

            # 显示主画面
             # 将处理后的输出图像保存
            if show_output:
                # 将warped图像放大并转换为彩色（如果是灰度）
                if len(warped.shape) == 2:
                    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                output_img = cv2.resize(warped, (400, 600))
                has_output = True

        # 如果没有有效的输出图像，创建占位符
        if show_output and not has_output:
            output_img = np.zeros((600,400, 3), dtype=np.uint8)
            cv2.putText(output_img, "No output available", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            
            # # 显示输出图像（如果已选择检测器）
            # # 显示warped图像（如果有矩形检测到）
            # if show_output and len(corners_list) > 0:
            #     cv2.imshow("Output", warped)
            # elif show_output:
            #     # 没有检测到矩形时显示占位符
            #     placeholder = np.zeros((rect_height, rect_width, 3), dtype=np.uint8)
            #     cv2.putText(placeholder, "No rectangle detected", (10, rect_height//2), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #     cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            #     cv2.resizeWindow("Output", rect_width,rect_height)
            #     cv2.imshow("Output", placeholder)
            # else:
            #     # 关闭输出窗口如果不需要显示
            #     if cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) > 0:                    
            #         cv2.destroyWindow("Output")

        # cv2.imshow("demo", frame)
        # 调整主画面大小以匹配输出画面高度
        frame_resized = cv2.resize(frame, (600, 600))
        
        # 创建并排显示的合并图像
        combined = np.hstack((frame_resized, output_img))
        
        # 显示合并后的图像
        cv2.imshow("Combined View", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # 空格键手动切换输出显示
            show_output = not show_output
            if not show_output:
                size_info = ""

    camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()