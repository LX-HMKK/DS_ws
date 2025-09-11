#!/user/bin/env python

import cv2
import numpy as np
import math
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn
from time import time
import argparse
import logging
import sys
import signal
import os
import serial
import serial.tools.list_ports
from HIK_CAM import HikIndustrialCamera

# 日志模块配置
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

coco_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
]

def draw_detection(img: np.array, 
                   bbox: tuple[int, int, int, int],
                   score: float, 
                   class_id: int) -> None:
    """
    Draws a detection bounding box and label on the image.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# ========== 正方形检测函数 ==========
def detect_squares(frame, area_ratio_threshold=(0.01, 0.8)):
    """
    检测图像中的所有正方形
    """
    # 预处理图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        # 跳过太小的轮廓
        if cv2.contourArea(contour) < 100:
            continue

        area = cv2.contourArea(contour)
        shape_area_ratio = area / frame_area if frame_area > 0 else 0

        # 面积占比检查
        if not (area_ratio_threshold[0] <= shape_area_ratio <= area_ratio_threshold[1]):
            continue

        # 检测正方形
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        if vertices != 4:  # 不是四边形
            continue

        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        (rx, ry), (rw, rh), angle = rect
        rect_width, rect_height = sorted([rw, rh], reverse=True)

        # 计算面积比
        rect_area = rect_width * rect_height
        area_ratio = area / rect_area if rect_area > 0 else 0

        # 检查是否为正方形（宽高比接近1，面积比接近1）
        if 0.85 < area_ratio < 1.05 and 0.9 < rect_height / rect_width < 1.1:
            # 计算中心点和尺寸
            center = (int(rx), int(ry))
            size = (rect_width + rect_height) / 2

            squares.append({
                "center": center,
                "size": size,
                "contour": contour,
                "box_points": cv2.boxPoints(rect).astype(np.int32)  # 存储边界点
            })

    return squares

class YOLO11_Detector:
    def __init__(self, model_path: str, conf_thres: float = 0.30, iou_thres: float = 0.6):
        """
        初始化YOLO检测器
        
        参数:
            model_path: 模型文件路径
            conf_thres: 置信度阈值 (默认 0.30)
            iou_thres: IoU阈值 (默认 0.6)
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.selected_digit = None  # 添加类变量存储选择的数字
        
        # 加载模型
        try:
            self.model = dnn.load(model_path)
        except Exception as e:
            logger.error(f"❌ Failed to load model file: {model_path}")
            logger.error(e)
            raise RuntimeError("Model loading failed") from e
        
        # 获取模型输入尺寸
        self.input_height, self.input_width = self.model[0].inputs[0].properties.shape[2:4]
        logger.info(f"Model input size: {self.input_width}x{self.input_height}")
        
        # 准备量化参数
        self.s_bboxes_scale = self.model[0].outputs[0].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.model[0].outputs[2].properties.scale_data[np.newaxis, :]
        
        # DFL系数
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        
        # anchors
        self.s_anchor = self._generate_anchors(80)
        self.m_anchor = self._generate_anchors(40)
        self.l_anchor = self._generate_anchors(20)
        
        # 置信度阈值转换
        self.conf_inverse = -np.log(1/conf_thres - 1)
        logger.info(f"Confidence threshold: {conf_thres}, IOU threshold: {iou_thres}")

    def set_selected_digit(self, digit: str):
        """
        设置要显示的数字
        
        参数:
            digit: 要显示的数字 (0-9)
        """
        if digit is None or (digit and digit in coco_names):
            self.selected_digit = digit
            logger.info(f"设置选择的数字: {digit if digit else '自动选择'}")
        else:
            logger.warning(f"无效的数字选择: {digit}")

    def _generate_anchors(self, size: int) -> np.ndarray:
        """生成锚点"""
        x = np.tile(np.linspace(0.5, size - 0.5, size), size)
        y = np.repeat(np.arange(0.5, size + 0.5, 1), size)
        return np.stack([x, y], axis=1)

    def _bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        将BGR图像转换为NV12格式
        """
        # 调整尺寸到模型输入尺寸
        resized_img = cv2.resize(bgr_img, (self.input_width, self.input_height))
        
        # 转换颜色空间
        yuv420p = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YUV_I420)
        yuv420p = yuv420p.flatten()
        
        # 重组为NV12格式
        height, width = self.input_height, self.input_width
        area = height * width
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        
        nv12 = np.concatenate([y, uv_packed])
        return nv12

    def _post_process(self, outputs: list[np.ndarray], orig_shape: tuple) -> tuple:
        """
        后处理函数，解析模型输出
        
        返回:
            (ids, scores, bboxes)
        """
        # 解析模型输出
        s_bboxes = outputs[0].reshape(-1, 64)
        m_bboxes = outputs[1].reshape(-1, 64)
        l_bboxes = outputs[2].reshape(-1, 64)
        s_clses = outputs[3].reshape(-1, 10)
        m_clses = outputs[4].reshape(-1, 10)
        l_clses = outputs[5].reshape(-1, 10)
        
        # 处理小特征层
        s_max_scores = np.max(s_clses, axis=1)
        s_valid = s_max_scores >= self.conf_inverse
        s_ids = np.argmax(s_clses[s_valid], axis=1)
        s_scores = s_max_scores[s_valid]
        s_scores = 1 / (1 + np.exp(-s_scores))
        s_dbboxes = self._process_bboxes(s_bboxes[s_valid], self.s_anchor[s_valid], 8)
        
        # 处理中特征层
        m_max_scores = np.max(m_clses, axis=1)
        m_valid = m_max_scores >= self.conf_inverse
        m_ids = np.argmax(m_clses[m_valid], axis=1)
        m_scores = m_max_scores[m_valid]
        m_scores = 1 / (1 + np.exp(-m_scores))
        m_dbboxes = self._process_bboxes(m_bboxes[m_valid], self.m_anchor[m_valid], 16)
        
        # 处理大特征层
        l_max_scores = np.max(l_clses, axis=1)
        l_valid = l_max_scores >= self.conf_inverse
        l_ids = np.argmax(l_clses[l_valid], axis=1)
        l_scores = l_max_scores[l_valid]
        l_scores = 1 / (1 + np.exp(-l_scores))
        l_dbboxes = self._process_bboxes(l_bboxes[l_valid], self.l_anchor[l_valid], 32)
        
        # 合并结果
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes))
        scores = np.concatenate((s_scores, m_scores, l_scores))
        ids = np.concatenate((s_ids, m_ids, l_ids))
        
        # NMS处理
        indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf_thres, self.iou_thres)
        if len(indices) == 0:
            return [], [], []
        
        # 映射回原始尺寸
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        
        bboxes = dbboxes[indices] * np.array([scale_x, scale_y, scale_x, scale_y])
        bboxes = bboxes.astype(np.int32)
        
        return ids[indices], scores[indices], bboxes

    def _process_bboxes(self, bboxes: np.ndarray, anchors: np.ndarray, scale: int) -> np.ndarray:
        """处理边界框"""
        bboxes_float = bboxes.reshape(-1, 4, 16)
        softmax_result = softmax(bboxes_float, axis=2)
        ltrb_indices = np.sum(softmax_result * self.weights_static, axis=2)
        
        x1y1 = anchors - ltrb_indices[:, 0:2]
        x2y2 = anchors + ltrb_indices[:, 2:4]
        
        return np.hstack([x1y1, x2y2]) * scale

    def _find_nearest_square(self, point: tuple, squares: list) -> dict:
        """找到距离点最近的正方形"""
        min_distance = float('inf')
        nearest_square = None
        
        for square in squares:
            sq_center = square["center"]
            distance = math.sqrt((point[0] - sq_center[0])**2 + (point[1] - sq_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_square = square
                
        return nearest_square

    def _find_smallest_square(self, squares: list) -> dict:
        """找到最小的正方形"""
        if not squares:
            return None
            
        smallest_square = min(squares, key=lambda x: x["size"])
        return smallest_square

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float or None]:
        """
        处理一帧图像，返回带有检测结果的图像和目标正方形尺寸
        
        参数:
            frame: 输入图像 (BGR格式)
        
        返回:
            带有检测结果的图像和目标正方形尺寸（若未检测到则为None）
        """
        # 记录原始尺寸
        orig_shape = frame.shape
        
        # 预处理 (BGR -> NV12)
        nv12_data = self._bgr2nv12(frame)
        
        # 模型推理
        outputs = self.model[0].forward(nv12_data)
        outputs = [tensor.buffer for tensor in outputs]
        
        # 后处理
        ids, scores, bboxes = self._post_process(outputs, orig_shape)
        
        # 绘制结果
        result_frame = frame.copy()
        
        # 检测所有正方形
        squares = detect_squares(result_frame)
        
        # 绘制所有检测到的正方形（浅灰色细线）
        for square in squares:
            cv2.drawContours(result_frame, [square["box_points"]], 0, (180, 180, 180), 1)
        
        # 存储数字检测结果和关联的正方形
        digit_squares = {}
        
        # 处理每个检测到的数字
        for class_id, score, bbox in zip(ids, scores, bboxes):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            digit = coco_names[class_id]
            
            # 绘制数字检测框
            draw_detection(result_frame, bbox, score, class_id)
            logger.info(f"检测到: {digit} (置信度: {score:.2f})")
            
            # 找到最近的正方形
            nearest_square = self._find_nearest_square((center_x, center_y), squares)
            if nearest_square:
                digit_squares[digit] = nearest_square
        
        # 处理用户选择的数字或自动选择最小正方形
        target_square = None
        target_digit = None
        size = None  # 初始化尺寸变量
        
        if self.selected_digit and self.selected_digit in digit_squares:
            # 显示用户选择的数字对应的正方形
            target_square = digit_squares[self.selected_digit]
            target_digit = self.selected_digit
        elif digit_squares:
            # 没有选择数字时，找到所有正方形中最小的一个
            target_square = self._find_smallest_square(list(digit_squares.values()))
            # 找到与最小正方形关联的数字
            target_digit = next((d for d, sq in digit_squares.items() if sq is target_square), None)
        
        # 高亮显示目标正方形并获取尺寸
        if target_square:
            size = target_square["size"]  # 获取尺寸
            # 高亮显示正方形（绿色粗线）
            cv2.drawContours(result_frame, [target_square["box_points"]], 0, (0, 255, 0), 2)
            
            # 在正方形中心显示尺寸
            size_text = f"Size: {size:.1f}px"
            cv2.putText(
                result_frame, size_text,
                (target_square["center"][0] - 50, target_square["center"][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            # 标注各边长度
            points = target_square["box_points"]
            for i in range(4):
                start = tuple(points[i])
                end = tuple(points[(i + 1) % 4])
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                edge_len = np.linalg.norm(points[i] - points[(i + 1) % 4])
                cv2.putText(
                    result_frame, f"{edge_len:.1f}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
                )
            
            # 在画面顶部显示结果信息
            info_text = f"num: {target_digit}  size: {size:.1f}px"
            cv2.putText(
                result_frame, info_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
        else:
            # 没有检测到任何正方形
            cv2.putText(
                result_frame, "no square",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        
        # 显示当前选择的数字
        if self.selected_digit:
            cv2.putText(
                result_frame, f"choice: {self.selected_digit}",
                (result_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                result_frame, "auto",
                (result_frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
        
        return result_frame, size  # 返回图像和尺寸

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/sunrise/AAA_DS_WS/DS_NUM.bin',
                        help='Path to BPU Quantized *.bin Model')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU threshold')
    opt = parser.parse_args()
    logger.info(opt)
    
    # 初始化检测器
    try:
        detector = YOLO11_Detector(
            model_path=opt.model_path,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres
        )
    except Exception as e:
        logger.error(f"初始化检测器失败: {e}")
        return
    
    # 初始化摄像头
    camera = HikIndustrialCamera()
    
    try:
        camera.init()
        camera.open()
        
        logger.info("开始实时检测...")
        logger.info("使用说明:")
        logger.info("  - 按0-9键选择要显示的数字")
        logger.info("  - 按'c'键清除选择")
        logger.info("  - 按'q'键退出程序")
        
        while True:
            start_time = time()
            
            # 获取帧
            frame = camera.get_frame()
            if frame is None:
                continue
                
            # 处理帧
            result_frame,size= detector.process_frame(frame)

            if size is not None:
                print(f"检测到正方形，尺寸: {size:.2f}mm")
            
            # 计算FPS
            fps = 1 / (time() - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, result_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示结果
            cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection", 800, 600)
            cv2.imshow("Detection", result_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key in range(ord('0'), ord('9') + 1):  # 数字键0-9
                selected_digit = chr(key)
                detector.set_selected_digit(selected_digit)  # 使用新方法设置数字
                logger.info(f"选择了数字: {selected_digit}")
            elif key == ord('c'):  # 清除选择
                detector.set_selected_digit(None)  # 使用新方法清除选择
                logger.info("已清除选择")
            elif key == ord('q'):  # 退出
                break
                
    finally:
        # 释放资源
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()