import cv2
import numpy as np
from HIK_CAM import HikIndustrialCamera
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class SquareResult:
    corners: np.ndarray
    size: float
    confidence: float

def preprocess_image(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """预处理图像"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 自适应二值化
    # binary = cv2.adaptiveThreshold(
    #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV, 11, 2
    # )
    _,binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    # 边缘检测
    edges = cv2.Canny(binary, 30, 100)
    
    return binary, edges

def detect_lines(edges: np.ndarray) -> Optional[np.ndarray]:
    """检测直线"""
    lines = cv2.HoughLinesP(
        edges, 
        rho=1,
        theta=np.pi/180,
        threshold=30,  # 降低阈值以检测更多线段
        minLineLength=50,  # 增加最小线长
        maxLineGap=20  # 增加最大间隙
    )
    return lines

def find_corners(lines: np.ndarray) -> List[Tuple[float, float]]:
    """查找角点"""
    corners = []
    max_corners = 12  # 限制最大角点数量
    
    for i in range(len(lines)):
        if len(corners) >= max_corners:  # 添加数量限制
            break
            
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            
            # 计算两线段的方向向量
            vec1 = np.array([x2 - x1, y2 - y1])
            vec2 = np.array([x4 - x3, y4 - y3])
            
            # 计算夹角
            dot = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norms == 0:
                continue
                
            cos_angle = dot / norms
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            # 如果接近直角
            if 80 <= angle <= 100:
                corner = line_intersection((x1,y1,x2,y2), (x3,y3,x4,y4))
                if corner is not None:
                    # 检查是否与已有角点重复（聚类）
                    is_duplicate = False
                    for existing_corner in corners:
                        if np.sqrt(((corner[0] - existing_corner[0]) ** 2) + 
                                 ((corner[1] - existing_corner[1]) ** 2)) < 15:  # 10像素距离内认为是同一个角点
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        corners.append(corner)
    
    return corners

def line_intersection(line1: Tuple[float, float, float, float],
                     line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    """计算两条线的交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None
        
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    
    if 0 <= t <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None

def find_potential_squares(corners: List[Tuple[float, float]], binary: np.ndarray) -> List[np.ndarray]:
    """查找潜在的正方形"""
    potential_squares = []
    max_squares = 4  # 限制最大正方形数量
    min_area = 100  # 最小面积
    max_area = 100000  # 最大面积
    
    for i in range(len(corners)):
        if len(potential_squares) >= max_squares:
            break
            
        for j in range(i + 1, len(corners)):
            for k in range(j + 1, len(corners)):
                p1 = np.array(corners[i])
                p2 = np.array(corners[j])
                p3 = np.array(corners[k])
                
                # 验证三点不共线
                v1 = p2 - p1
                v2 = p3 - p1
                if abs(np.cross(v1, v2)) < 100:  # 如果三点接近共线则跳过
                    continue
                
                # 计算三条边的长度
                d12 = np.linalg.norm(v1)
                d13 = np.linalg.norm(v2)
                d23 = np.linalg.norm(p3 - p2)
                
                # 使用长度值进行排序
                edge_lengths = [(d12, 0), (d13, 1), (d23, 2)]
                edge_lengths.sort(key=lambda x: x[0])  # 按照边长排序
                
                # 获取最短的两条边的索引
                shortest_edges = edge_lengths[:2]
                
                # 验证最短的两条边长度相近
                if abs(shortest_edges[0][0] - shortest_edges[1][0]) / shortest_edges[0][0] > 0.15:
                    continue
                
                # 根据最短边的索引获取对应的点
                points = [(p1, p2), (p1, p3), (p2, p3)]
                edge1_points = points[shortest_edges[0][1]]
                edge2_points = points[shortest_edges[1][1]]
                
                # 找到共同点
                common_point = None
                vec1 = None
                vec2 = None
                
                for pt1 in edge1_points:
                    for pt2 in edge2_points:
                        if np.array_equal(pt1, pt2):
                            common_point = pt1
                            vec1 = edge1_points[1] - edge1_points[0] if np.array_equal(pt1, edge1_points[0]) else edge1_points[0] - edge1_points[1]
                            vec2 = edge2_points[1] - edge2_points[0] if np.array_equal(pt1, edge2_points[0]) else edge2_points[0] - edge2_points[1]
                            break
                    if common_point is not None:
                        break
                
                if common_point is None:
                    continue
                
                # 构建第四个点
                p4 = common_point + vec1 + vec2
                
                # 计算面积
                area = abs(np.cross(vec1, vec2))
                if area < min_area or area > max_area:
                    continue
                
                # 验证构建的是否为正方形（检查对角线是否相等）
                diag1 = np.linalg.norm(p4 - common_point)
                diag2 = np.linalg.norm((common_point + vec1) - (common_point + vec2))
                if abs(diag1 - diag2) / diag1 > 0.15:
                    continue
                
                square = np.array([common_point, 
                                 common_point + vec1,
                                 p4,
                                 common_point + vec2])
                # 验证正方形区域
                if not verify_square_region(binary, square):
                    continue
                
                
                potential_squares.append(square)
    
    return potential_squares
def verify_square_region(binary: np.ndarray, square_points: np.ndarray) -> bool:
    """验证正方形区域内是否都为黑色"""
    # 创建掩码
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [square_points.astype(np.int32)], 255)
    
    # 获取正方形区域
    roi = cv2.bitwise_and(binary, mask)
    
    # 计算区域内白色像素的比例
    white_ratio = np.sum(roi == 255) / np.sum(mask == 255)
    
    # 如果白色像素比例过高，则认为不是有效的正方形
    return white_ratio < 0.1  # 允许10%的误差

def predict_fourth_corner(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """预测第四个角点"""
    # 计算两个最短边的向量
    v1 = p2 - p1
    v2 = p2 - p3
    
    # 如果v1更短，使用v1作为基准边
    if np.linalg.norm(v1) < np.linalg.norm(v2):
        p4 = p3 + v1
    else:
        p4 = p1 + v2
    
    return p4

def detect_smallest_square(frame: np.ndarray) -> Tuple[np.ndarray, Optional[SquareResult]]:
    """主检测函数"""
    output_frame = frame.copy()
    
    # 预处理
    binary, edges = preprocess_image(frame)
    
    # 检测直线
    lines = detect_lines(edges)
    if lines is None:
        return output_frame, None
    
    # 查找角点
    corners = find_corners(lines)
    if len(corners) < 3:
        return output_frame, None
    
    # 查找潜在的正方形
    potential_squares = find_potential_squares(corners, binary)
    if not potential_squares:
        return output_frame, None
    
    # 找出最小的正方形
    min_square = None
    min_size = float('inf')
    
    for square in potential_squares:
        size = np.min([
            np.linalg.norm(square[i] - square[(i+1)%4])
            for i in range(4)
        ])
        if size < min_size:
            min_size = size
            min_square = square
    
    if min_square is not None:
        # 绘制结果
        cv2.drawContours(output_frame, [min_square.astype(np.int32)],
                        0, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Size: {min_size:.1f}px",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 255), 2)
        
        result = SquareResult(
            corners=min_square,
            size=min_size,
            confidence=1.0
        )
        return output_frame, result

    return output_frame, None

def main():
    # 初始化相机
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    
    try:
        while True:
            # 获取帧
            frame = camera.get_frame()
            if frame is None:
                continue
            
            # 处理帧
            binary,edges= preprocess_image(frame)
            output_frame, result = detect_smallest_square(frame)
            
            # 显示结果
            cv2.imshow('Square Detection', output_frame)
            cv2.imshow('Binary', binary)
            cv2.imshow('Edges', edges)
    
            
            # 如果检测到正方形，打印信息
            if result is not None:
                print(f"检测到正方形，边长：{result.size:.1f}像素")
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 清理资源
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()