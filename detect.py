import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from HIK_CAM import HikIndustrialCamera

class RectangleDetector:
    """
    在原图中寻找满足以下条件的矩形：
        1. 外轮廓为四边形
        2. 面积 100000 ~ 3000000
        3. 宽高比在 [1.2, 3.0] 之间
        4. 内部恰好有一个四边形内轮廓
    返回内轮廓的 4 个角点，顺时针排序。
    """

    def __init__(self):
        self.ratio_min = 1.2
        self.ratio_max = 3.0
        self.EPSILON_RATIO = 0.1
        # 亚像素优化参数
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.05)
        self.subpix_window = (1, 1)  # 搜索窗口大小
        self.subpix_zero_zone = (-1, -1)  # 死区大小（通常设为-1,-1表示禁用）
        self.prev_corners = None  # 存储上一帧角点
        self.stabilization_threshold = 1.0  # 稳定阈值(像素距离)
        self.min_contour_dist = 5  # 轮廓间最小距离(避免重复检测)
        self.kalman_filters = []  # 存储每个矩形的卡尔曼滤波器

    @staticmethod
    def _order_corners_clockwise(pts):
        """更稳定的顺时针排序：左上→右上→右下→左下"""
        # 1. 计算中心点
        center = np.mean(pts, axis=0)
        
        # 2. 计算各点相对于中心的角度（考虑图像坐标系y轴向下）
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        
        # 3. 按角度从小到大排序（图像坐标系中为顺时针）
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        
        # 4. 确保左上角（x+y最小）作为起点
        dist_to_origin = np.linalg.norm(sorted_pts, axis=1)
        start_idx = np.argmin(dist_to_origin)
        
        # 5. 旋转数组使左上角成为第一个点
        return np.roll(sorted_pts, -start_idx, axis=0) 

    def detect(self, frame):
        """
        :param frame: BGR 图像 (H, W, 3)
        :return: list[np.ndarray]，每个元素为 shape=(4,2) 的 float32 数组
                 若未检测到，返回 []
        """
        # 1. ROI & 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. 轮廓提取
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []

        hierarchy = hierarchy[0]
        results = []

        # 3. 遍历顶层轮廓
        for idx, cnt in enumerate(contours):
            parent_idx = hierarchy[idx][3]
            if parent_idx != -1:
                continue

            area = cv2.contourArea(cnt)
            if area < 150000 or area > 3000000:
                continue

            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            ratio = float(w) / h if h != 0 else 0
            ratio = max(ratio, 1.0 / ratio) if ratio else 0
            if not (self.ratio_min <= ratio <= self.ratio_max):
                continue

            outer_pts = approx.reshape(4, 2).astype(np.float32)

            # 3.1 找内轮廓
            inner_pts_list = []
            child = hierarchy[idx][2]
            while child != -1:
                inner = contours[child]
                inner_area = cv2.contourArea(inner)
                if inner_area > 20000:
                    inner_peri = cv2.arcLength(inner, True)
                    inner_approx = cv2.approxPolyDP(inner,
                                                    self.EPSILON_RATIO * inner_peri,
                                                    True)
                    if len(inner_approx) == 4:
                        inner_pts_list.append(inner_approx.reshape(4, 2).astype(np.float32))
                child = hierarchy[child][0]

            if len(inner_pts_list) != 1:
                continue

            # 3.2 获取内轮廓点
            inner_pts = inner_pts_list[0]

            # 3.3 内轮廓顺时针排序
            ordered_inner = self._order_corners_clockwise(inner_pts)

            # 3.4 亚像素级精度优化（应用于内轮廓）
            corners_sp = ordered_inner.reshape(-1, 1, 2).astype(np.float32)
            cv2.cornerSubPix(
                gray,
                corners_sp,
                self.subpix_window,
                self.subpix_zero_zone,
                self.subpix_criteria
            )
            ordered_inner = corners_sp.reshape(4, 2)

            #3.5 卡尔曼滤波处理（针对内轮廓）
            if len(self.kalman_filters) < len(results) + 1:
                kf = []
                for _ in range(4):
                    kf.append(cv2.KalmanFilter(4, 2))
                    kf[-1].transitionMatrix = np.array([[1, 0, 1, 0],
                                                        [0, 1, 0, 1],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32)
                    kf[-1].measurementMatrix = np.array([[1, 0, 0, 0],
                                                         [0, 1, 0, 0]], np.float32)
                    kf[-1].processNoiseCov = 1e-1 * np.eye(4, dtype=np.float32)
                    kf[-1].measurementNoiseCov = 1e-4 * np.eye(2, dtype=np.float32)
                    kf[-1].errorCovPost = 1. * np.eye(4, dtype=np.float32)
                self.kalman_filters.append(kf)

            filtered_corners = []
            for i, corner in enumerate(ordered_inner):
                kf = self.kalman_filters[len(results)][i]
                measurement = np.array([[corner[0]], [corner[1]]], dtype=np.float32)
                prediction = kf.predict()
                corrected = kf.correct(measurement)
                filtered_corners.append(corrected[:2].flatten())

            # 将内轮廓结果添加到返回列表
            results.append(np.array(filtered_corners, dtype=np.float32))

        return results

    def _is_similar_contour(self, cnt1, cnt2):
        """检查两个轮廓是否相似"""
        M1 = cv2.moments(cnt1)
        M2 = cv2.moments(cnt2)
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])
        dist = np.sqrt((cx1 - cx2) **2 + (cy1 - cy2)** 2)

        area1 = cv2.contourArea(cnt1)
        area2 = cv2.contourArea(cnt2)
        area_ratio = min(area1, area2) / max(area1, area2)

        return dist < self.min_contour_dist and area_ratio > 0.8


# ----------------- 主函数调用示例 -----------------
if __name__ == "__main__":
    camera = HikIndustrialCamera()
    camera.init()
    camera.open()
    detector = RectangleDetector()

    while True:
        frame = camera.get_frame()
        frame = frame[124:1924, 324:2124]
        corners_list = detector.detect(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # 绘制内轮廓结果
        for corners in corners_list:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            for idx, (x, y) in enumerate(corners):
                cv2.circle(frame, (int(round(x)), int(round(y))), 8, colors[idx], -1)
                cv2.putText(frame, str(idx), (int(round(x)) + 10, int(round(y)) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, colors[idx], 3)

            # 绘制内四边形
            cv2.polylines(frame, [corners.astype(np.int32)], True, (255, 0, 0), 2)  # 用蓝色表示内轮廓

        # 显示结果
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("demo", 1000, 1000)
        cv2.imshow("demo", frame)
        cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary", 1000, 1000)
        cv2.imshow("binary", binary)


        if corners_list:
            pts_str = ', '.join(f'({pt[0]:.8f}, {pt[1]:.8f})' for pt in corners)
            print('内轮廓亚像素角点：', pts_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()
    cv2.destroyAllWindows()