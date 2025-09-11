import cv2
import numpy as np
import math
from HIK_CAM import HikIndustrialCamera

class MinSquareDetector:
    def __init__(self, world_width, world_height):
        """
        初始化最小正方形检测器
        
        参数:
            world_width: 画面在世界坐标系中的宽度（单位：mm）
            world_height: 画面在世界坐标系中的高度（单位：mm）
        """
        self.world_width = world_width
        self.world_height = world_height
        self.min_side_length = 20  # 最小边长阈值（像素）
        self.max_corners = 20      # 最大角点数
        self.quality_level = 0.1   # 角点质量等级
        self.min_distance = 10     # 角点间最小距离
        self.gaussian_size = (31, 31)  # 高斯模糊核大小
    
    def find_square(self, p1, p3):
        """
        根据对角线上的两个点计算正方形的四个顶点
        
        参数:
            p1, p3: 对角线上的两个点坐标 (x, y)
        
        返回:
            正方形四个顶点的列表 [p1, p2, p3, p4]
        """
        # 计算公式中的中间点
        center_x = (p1[0] + p3[0]) / 2
        center_y = (p1[1] + p3[1]) / 2
        dx = (p3[0] - p1[0]) / 2
        dy = (p3[1] - p1[1]) / 2
        
        # 计算另外两个顶点
        p2 = (center_x - dy, center_y + dx)
        p4 = (center_x + dy, center_y - dx)
        
        return [p1, p2, p3, p4]
    
    def calculate_area(self, corners):
        """
        计算正方形面积（像素面积）
        """
        p1, p2 = corners[0], corners[1]
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        length = math.sqrt(dx*dx + dy*dy)
        return length * length
    
    def detect(self, frame):
        """
        在图像中检测最小面积正方形并返回结果
        
        参数:
            frame: 输入图像帧 (BGR格式)
        
        返回:
            result_image: 绘制了检测结果的图像
            min_square: 检测到的最小正方形信息（包含四个顶点和实际边长）
        """
        # 1. 图像预处理
        working_frame = frame.copy()
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        blurred = cv2.GaussianBlur(binary, self.gaussian_size, 0)
        
        # 2. 角点检测
        corners = cv2.goodFeaturesToTrack(
            blurred, 
            self.max_corners, 
            self.quality_level, 
            self.min_distance
        )
        
        if corners is None:
            return frame, None
        
        corners = corners.astype(np.int32).reshape(-1, 2)
        
        # 3. 计算像素到实际尺寸的转换比例
        h, w = frame.shape[:2]
        pixel_to_mm_x = self.world_width / w
        pixel_to_mm_y = self.world_height / h
        pixel_to_mm = (pixel_to_mm_x + pixel_to_mm_y) / 2  # 取平均值
        
        # 4. 检测最小面积正方形
        result_image = frame.copy()
        min_square = None
        min_area = float('inf')
        
        # 遍历所有可能的角点对
        for i in range(len(corners)):
            for j in range(i+1, len(corners)):
                p1 = tuple(corners[i])
                p3 = tuple(corners[j])
                
                # 计算可能的正方形
                square_pts = self.find_square(p1, p3)
                
                # 计算正方形面积
                area = self.calculate_area(square_pts)
                
                # 跳过面积过小的正方形
                if area < self.min_side_length * self.min_side_length:
                    continue
                
                # 创建掩模检查正方形区域
                mask = np.zeros_like(binary)
                pts_array = np.array([square_pts], dtype=np.int32)
                cv2.fillPoly(mask, pts_array, 255)
                
                # 计算掩模区域内的白色像素数量
                masked = cv2.bitwise_and(binary, binary, mask=mask)
                white_pixel_count = cv2.countNonZero(masked)
                
                # 如果白色像素太少，则认为是有效的正方形
                if white_pixel_count < 100:
                    # 更新最小正方形
                    if area < min_area:
                        min_area = area
                        
                        # 计算实际边长（mm）
                        p1, p2 = square_pts[0], square_pts[1]
                        dx = p1[0] - p2[0]
                        dy = p1[1] - p2[1]
                        pixel_length = math.sqrt(dx*dx + dy*dy)
                        real_length = round(pixel_length * pixel_to_mm, 1)
                        
                        min_square = {
                            'corners': square_pts,
                            'real_length': real_length
                        }
        
        # 5. 绘制最小正方形（如果找到）
        if min_square is not None:
            # 绘制正方形轮廓
            pts_array = np.array([min_square['corners']], dtype=np.int32)
            cv2.polylines(result_image, pts_array, True, (0, 0, 255), 2)
            
            # 标注边长
            corners = min_square['corners']
            center_x = int((corners[0][0] + corners[2][0]) / 2)
            center_y = int((corners[0][1] + corners[2][1]) / 2)
            text = f"{min_square['real_length']}mm"
            cv2.putText(result_image, text, (center_x-50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return result_image, real_length

# 使用示例
if __name__ == "__main__":
    # 初始化摄像头
    # 注意：这里需要导入HIK_CAM模块
    # from HIK_CAM import HikIndustrialCamera
    
    # 创建摄像头对象
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    
    # 创建正方形检测器（假设画面尺寸为300mm x 200mm）
    detector = MinSquareDetector(world_width=300, world_height=200)
    
    while True:
        frame = camera.get_frame()
        if frame is None: 
            continue
            
        # 裁剪ROI区域
        frame = frame[124:1924, 324:2124]
        
        # 检测最小正方形
        result_frame, _ = detector.detect(frame)
        
        # 显示结果
        cv2.namedWindow("Min Square Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Min Square Detection", 1000, 1000)
        cv2.imshow("Min Square Detection", result_frame)
        
        # 输出最小正方形信息
        # if min_square is not None:
        #     print(f"Min Square Found - Side Length: {min_square['real_length']}mm")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    camera.close()
    cv2.destroyAllWindows()