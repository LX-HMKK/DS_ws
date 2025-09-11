import cv2
import numpy as np
import math
from HIK_CAM import HikIndustrialCamera

class ShapeDetector:
    def __init__(self, area_ratio_threshold=(0.01, 0.8), 
                 frame_real_width=None, frame_real_height=None):
        """
        形状识别器初始化
        
        参数:
            area_ratio_threshold: 面积占比阈值范围
            frame_real_width: 截取区域的真实宽度（单位自定义）
            frame_real_height: 截取区域的真实高度（单位自定义）
        """
        self.area_ratio_threshold = area_ratio_threshold
        self.frame_real_width = frame_real_width
        self.frame_real_height = frame_real_height
        
    def set_real_frame_size(self, width, height):
        """设置截取区域的真实尺寸"""
        self.frame_real_width = width
        self.frame_real_height = height
        
    def detect_shape(self, frame):
        """
        执行形状识别
        
        参数:
            frame: 输入图像帧
            
        返回:
            result: 包含形状和真实尺寸的字典 {"shape": str, "size": float}
            output_frame: 带标注的输出图像
        """
        if frame is None:
            return {"shape": "unknown", "size": 0.0}, None
            
        # 预处理图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 初始化结果
        result = {"shape": "unknown", "size": 0.0}
        output_frame = frame.copy()
        
        if not contours:
            return result, output_frame
        
        frame_area = frame.shape[0] * frame.shape[1]
        valid_contours = []  # 存储满足面积要求的轮廓
        
        # 步骤1: 先筛选出满足面积要求的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            shape_area_ratio = area / frame_area if frame_area > 0 else 0
            
            # 检查面积是否在有效范围内
            if (self.area_ratio_threshold[0] <= shape_area_ratio <= self.area_ratio_threshold[1]):
                valid_contours.append(contour)
        
        # 没有有效轮廓时显示提示信息
        if not valid_contours:
            # 计算最小轮廓的面积占比
            min_contour = min(contours, key=cv2.contourArea)
            min_area = cv2.contourArea(min_contour)
            min_ratio = min_area / frame_area
            
            # 显示面积超出范围提示
            cv2.putText(output_frame, f"Area out of range ({min_ratio:.2%})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            return result, output_frame
        
        # 步骤2: 在有效轮廓中找面积最小的轮廓
        target_contour = min(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(target_contour)
        shape_area_ratio = area / frame_area
        
        # 形状识别核心逻辑
        shape_detected = False
        pixel_size = 0.0
        shape_type = "unknown"
        
        perimeter = cv2.arcLength(target_contour, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(target_contour, epsilon, True)
        vertices = len(approx)
        (x, y), radius = cv2.minEnclosingCircle(target_contour)
        rect = cv2.minAreaRect(target_contour)
        (rx, ry), (rw, rh), angle = rect
        rect_width, rect_height = sorted([rw, rh], reverse=True)
        circularity = (4 * math.pi * area) / (perimeter **2) if perimeter > 0 else 0
        rect_area = rect_width * rect_height
        area_ratio = area / rect_area if rect_area > 0 else 0
        
        # 三角形验证
        triangle_valid = False
        if vertices == 3:
            dists = [np.linalg.norm(approx[i][0] - approx[(i+1)%3][0]) for i in range(3)]
            triangle_valid = np.std(dists) < 0.1 * np.mean(dists)
        
        # 形状判断与尺寸计算
        if vertices >= 6 and circularity > 0.80:  # 圆形
            shape_type = "circle"
            pixel_size = 2 * radius  # 直径
            cv2.circle(output_frame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
            cv2.putText(output_frame, f"{shape_type}", 
                       (int(x)-40, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(output_frame, f"D={pixel_size:.1f}px", 
                       (int(x)-40, int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            shape_detected = True
            
        elif vertices == 4:  # 四边形
            # 计算宽高比和面积比
            aspect_ratio = min(rect_width, rect_height) / max(rect_width, rect_height)
            
            # 正方形判断条件
            if 0.9 <= aspect_ratio <= 1.0 and 0.85 <= area_ratio <= 1.05:
                shape_type = "square"
                pixel_size = (rect_width + rect_height) / 2
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 2)
                cv2.putText(output_frame, f"{shape_type}", 
                           (int(rx)-40, int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(output_frame, f"D={pixel_size:.1f}px", 
                           (int(rx)-40, int(ry)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                shape_detected = True
                
        elif vertices == 3 and triangle_valid and 0.45 < area_ratio < 0.65:  # 正三角形
            shape_type = "triangle"
            dists = [np.linalg.norm(approx[i][0] - approx[(i+1)%3][0]) for i in range(3)]
            pixel_size = np.mean(dists)
            cv2.drawContours(output_frame, [approx], -1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"{shape_type}", 
                       (int(rx)-40, int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),1)
            cv2.putText(output_frame, f"D={pixel_size:.1f}px", 
                       (int(rx)-40, int(ry)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),1)
            shape_detected = True
    
        # 计算真实尺寸（如果设置了真实帧尺寸）
        real_size = 0.0
        if shape_detected and self.frame_real_width and self.frame_real_height:
            frame_pixel_height, frame_pixel_width = frame.shape[:2]
            pixel_ratio = (self.frame_real_width / frame_pixel_width + 
                          self.frame_real_height / frame_pixel_height) / 2
            real_size = pixel_size * pixel_ratio
            result["size"] = real_size  # 返回真实尺寸
            result["shape"] = shape_type
            cv2.putText(output_frame, f"Real Size: {real_size:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif shape_detected:
            # 如果检测到形状但未设置真实尺寸，返回0.0
            result["size"] = 0.0
            result["shape"] = shape_type
        
        # 显示当前检测轮廓的面积占比
        cv2.putText(output_frame, f"Area Ratio: {shape_area_ratio:.2%}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return result, output_frame


# 使用示例
if __name__ == "__main__":
    # 初始化识别器（设置真实尺寸：100x80毫米）
    detector = ShapeDetector(
        area_ratio_threshold=(0.1, 0.8),
        frame_real_width=100.0,
        frame_real_height=80.0
    )
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    
    while True:
        frame = camera.get_frame()
        # 如果需要对特定区域进行检测，可以取消下面的注释
        # frame = frame[664:1384, 864:1584]
        
        # 执行识别
        result, output_image = detector.detect_shape(frame)
    
        print(f"识别结果 - 形状: {result['shape']}, 尺寸: {result['size']:.1f}mm")
        
        # 显示阈值图像和结果
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        cv2.imshow("Shape thresh", thresh)
        cv2.imshow("Shape Detection", output_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    camera.close()
    cv2.destroyAllWindows()