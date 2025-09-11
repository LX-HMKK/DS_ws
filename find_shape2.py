import cv2
import numpy as np
import math
from HIK_CAM import HikIndustrialCamera
from skimage.feature import peak_local_max

class CircleDetector:
    def __init__(self, min_area_ratio=0.05, max_area_ratio=0.9, debug_mode=False):
        """
        初始化圆检测器
        Args:
            min_area_ratio: 轮廓最小面积比例
            max_area_ratio: 轮廓最大面积比例
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.debug_mode = debug_mode
        self.last_diameter = None
        self.smooth_factor = 0.7  # 平滑因子

    def enable_debug(self, enable=True):
        """启用或禁用调试模式"""
        self.debug_mode = enable

    def is_valid_contour(self, contour, image_area):
        """检查轮廓面积是否在有效范围内"""
        area = cv2.contourArea(contour)
        ratio = area / image_area
        return self.min_area_ratio <= ratio <= self.max_area_ratio

    def find_min_square_side(self, frame):
        """通过内切圆方法寻找最小正方形边长"""
        # 转换为灰度图并二值化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.bitwise_not(binary)
        
        # 创建调试图像
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if self.debug_mode else None
        
        # 查找外轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, None
            
        # 找到符合面积要求的最大轮廓
        valid_contours = [cnt for cnt in contours if self.is_valid_contour(cnt, frame.shape[0] * frame.shape[1])]
        if not valid_contours:
            return frame, None
            
        main_contour = max(valid_contours, key=cv2.contourArea)
        
        # 创建距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        if self.debug_mode:
            dist_viz = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow("Distance Transform", dist_viz)
        
        # 找到所有局部最大值点
        dist_threshold = np.max(dist_transform) * 0.3
        local_max = peak_local_max(dist_transform, min_distance=5, threshold_abs=dist_threshold,
                             exclude_border=False)
        
        # 验证内切圆
        valid_circles = []
        for center in local_max:
            x, y = center[1], center[0]
            radius = dist_transform[y, x]
            
            if self.is_valid_inscribed_circle(binary, (x, y), radius):
                valid_circles.append((x, y, radius))
                if self.debug_mode:
                    cv2.circle(debug_img, (x, y), int(radius), (0, 255, 0), 2)
        
        if self.debug_mode:
            cv2.imshow("All Circles", debug_img)
        
        if not valid_circles:
            return frame, None
        
        # 找到最小有效内切圆
        min_circle = min(valid_circles, key=lambda x: x[2])
        min_diameter = min_circle[2] * 2
        
        # 绘制结果
        output = frame.copy()
        # 绘制最小内切圆
        cv2.circle(output, (min_circle[0], min_circle[1]), int(min_circle[2]), (0, 255, 0), 2)
        # 绘制轮廓
        cv2.drawContours(output, [main_contour], -1, (0, 0, 255), 2)
        # 标注直径
        text = f"D={min_diameter:.1f}px"
        cv2.putText(output, text, 
                   (min_circle[0] - 40, min_circle[1] - int(min_circle[2]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 在返回结果前进行平滑处理
        if self.last_diameter is not None:
            min_diameter = (self.smooth_factor * self.last_diameter + 
                          (1 - self.smooth_factor) * min_diameter)
        self.last_diameter = min_diameter
        
        return output, min_diameter

    # 将静态方法改为实例方法，添加self参数
    def is_valid_inscribed_circle(self, binary, center, radius):
        """验证圆是否为有效的内切圆，确保与每条边都相切"""
        x, y = center
        r = int(radius)
        
        # 1. 检查圆是否完全在图形内部
        mask = np.zeros_like(binary)
        cv2.circle(mask, (x, y), r, 255, -1)
        if not np.all(binary[mask == 255] == 255):
            return False
        
        # 2. 生成密集的采样点
        angles = np.linspace(0, 2*np.pi, 360)  # 每1度采样一个点
        contact_regions = np.zeros(8, dtype=int)  # 将圆周分成8个区域
        valid_points = []
        
        for angle in angles:
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            
            if 0 <= px < binary.shape[1] and 0 <= py < binary.shape[0]:
                # 检查3x3邻域
                neighborhood = binary[max(0, py-1):min(binary.shape[0], py+2),
                                    max(0, px-1):min(binary.shape[1], px+2)]
                
                # 判断是否为边界点（同时包含前景和背景）
                if np.any(neighborhood == 0) and np.any(neighborhood == 255):
                    valid_points.append((px, py))
                    # 记录接触点所在区域
                    region = int((angle / (2*np.pi) * 8)) % 8
                    contact_regions[region] = 1
        
        # 3. 验证条件：
        # a. 至少有足够多的接触点
        min_contact_points = 18  # 至少每10度有一个接触点
        if len(valid_points) < min_contact_points:
            return False

        # b. 接触点要分布在不同区域（至少4个区域有接触点）
        if np.sum(contact_regions) < 3:
            return False
        
        # c. 接触点之间不应该有太大间隔
        if len(valid_points) > 0:
            valid_points = np.array(valid_points)
            # 计算相邻接触点之间的最大角度间隔
            angles = np.arctan2(valid_points[:,1] - y, valid_points[:,0] - x)
            angles = np.sort(angles)
            angle_gaps = np.diff(angles)
            max_gap = max(angle_gaps)
            if max_gap > np.pi:  # 不允许超过180度的间隔
                return False
        
        return True

def main():
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    
    # 创建检测器实例
    detector = CircleDetector(min_area_ratio=0.2, max_area_ratio=0.8, debug_mode=False)
    # detector.enable_debug(True)  # 启用调试模式
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # 处理图像
        processed, min_side = detector.find_min_square_side(frame)
        
        if min_side is not None:
            print(f"最小正方形边长: {min_side:.2f}像素")
            
        # 显示结果
        cv2.imshow("Result", processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()