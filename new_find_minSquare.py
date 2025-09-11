import cv2
import numpy as np
from HIK_CAM import HikIndustrialCamera

def verify_square_by_area(points, frame_area):
    """
    验证正方形面积是否在合理范围内
    params:
        points: 正方形的四个顶点
        frame_area: 整个画面的面积
    returns:
        bool: 是否是有效的正方形
    """
    if points is None:
        return False
    
    area = cv2.contourArea(points)
    area_ratio = area / frame_area
    
    MIN_AREA_RATIO = 0.05  # 最小面积比例 1%
    MAX_AREA_RATIO = 0.95  # 最大面积比例 25%
    
    if MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
        perimeter = cv2.arcLength(points, True)
        ratio = perimeter * perimeter / area if area > 0 else 0
        return 15 < ratio < 17
    
    return False

def is_right_angle(pt1, pt2, pt3, tolerance=5):
    """
    判断三个点形成的角是否为直角
    params:
        pt1, pt2, pt3: 三个点的坐标，pt2为角点
        tolerance: 角度误差容许值（度）
    returns:
        bool: 是否为直角
    """
    # 计算两个向量
    vector1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vector2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
    
    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)
    # 计算向量的模
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # 计算夹角（弧度）
    if norm1 * norm2 == 0:
        return False
    cos_angle = dot_product / (norm1 * norm2)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    # 判断是否接近90度
    return abs(angle_deg - 90) <= tolerance

def find_smallest_square(frame):
    """
    在图像中查找最小的正方形
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    
    # 使用Harris角点检测找到角点和交点
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    _, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)
    
    # 找到所有角点的坐标
    corners = cv2.goodFeaturesToTrack(dst, maxCorners=50, 
                                    qualityLevel=0.01, 
                                    minDistance=10)
    
    if corners is None:
        return None, None
    
    corners = np.int32(corners)
    frame_area = frame.shape[0] * frame.shape[1]
    
    min_square_size = float('inf')
    min_square_points = None
    
    # 遍历所有可能的四点组合
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            for k in range(j + 1, len(corners)):
                for l in range(k + 1, len(corners)):
                    pts = [corners[i].ravel(), corners[j].ravel(), 
                          corners[k].ravel(), corners[l].ravel()]
                    
                    # 按顺时针或逆时针排序四个点
                    center = np.mean(pts, axis=0)
                    sorted_pts = sorted(pts, 
                        key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
                    pts = np.array(sorted_pts)
                    
                    # 验证四个角是否都是直角
                    is_square = True
                    for m in range(4):
                        if not is_right_angle(pts[m-1], pts[m], pts[(m+1)%4]):
                            is_square = False
                            break
                    
                    if not is_square:
                        continue
                    
                    # 验证四条边长是否相等
                    edges = []
                    for m in range(4):
                        edge = np.sqrt(np.sum((pts[m] - pts[(m+1)%4])**2))
                        edges.append(edge)
                    
                    mean_edge = np.mean(edges)
                    if max(abs(edge - mean_edge) for edge in edges) > 5:
                        continue
                    
                    # 验证面积比例
                    square_points = pts.reshape((-1,1,2))
                    if verify_square_by_area(square_points, frame_area):
                        if mean_edge < min_square_size:
                            min_square_size = mean_edge
                            min_square_points = square_points
    
    return min_square_size, min_square_points
def draw_results(frame, square_points, square_size):
    """
    在图像上绘制结果
    """
    if square_points is not None:
        cv2.polylines(frame, [square_points], True, (0, 255, 0), 2)
        
        frame_area = frame.shape[0] * frame.shape[1]
        square_area = cv2.contourArea(square_points)
        area_ratio = (square_area / frame_area) * 100
        
        cv2.putText(frame, f"Size: {square_size:.1f}px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Area Ratio: {area_ratio:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    # 初始化相机
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()  
    
    try:
        while True:
            # 取流
            frame = camera.get_frame()
            if frame is None:continue
         
            # 处理图像
            square_size, square_points = find_smallest_square(frame)
            
            # 绘制结果
            frame = draw_results(frame, square_points, square_size)
            
            # 显示图像
            cv2.imshow('Smallest Square Detection', frame)
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 释放资源
        camera.close()
        cv2.destroyAllWindows()