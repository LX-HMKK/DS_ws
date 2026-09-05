"""纯几何 / 位姿辅助函数（依赖 OpenCV + numpy，不依赖相机与模型）。

集中放置相机标定参数读取、PnP 位姿解算、欧拉角提取与 A4 比例适宜性判断，
供 UI 与调试复用。均为纯函数，便于单独核对。
"""

import math

import cv2
import numpy as np
import yaml

# A4 纸张真实尺寸 mm = 210 × 297 -> 长宽比（宽高比取 max/min）
A4_ASPECT = 297.0 / 210.0  # ≈ 1.414

# A4 适宜性判定的英文缩写与对应显示颜色（OpenCV 画不了中文，用英文标注）
VERDICT_EN = {"适宜": "FIT", "偏长": "TALL", "偏宽": "WIDE"}
VERDICT_COLOR = {"适宜": (0, 255, 0), "偏长": (0, 200, 255), "偏宽": (255, 160, 0)}

# solvePnP 求解方法 名称 -> OpenCV 常量
PNP_FLAGS = {
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "EPNP": cv2.SOLVEPNP_EPNP,
    "IPPE": cv2.SOLVEPNP_IPPE,
    "DLS": cv2.SOLVEPNP_DLS,
    "UPNP": cv2.SOLVEPNP_UPNP,
}


def read_camera_params(file_path):
    """从标定结果 yaml 读取内参矩阵与畸变系数。

    参数:
        file_path: 标定文件路径（如 configs/camera_calibration.yaml）。

    返回:
        (camera_matrix: np.ndarray float32 3x3, distortion_coeffs: np.ndarray float32)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    camera_matrix = np.array(params["camera_matrix"], dtype=np.float32)
    distortion_coeffs = np.array(params["distortion_coefficients"], dtype=np.float32).flatten()
    return camera_matrix, distortion_coeffs


def adjust_camera_matrix_for_crop(camera_matrix, crop):
    """把裁剪区域标定的内参矩阵平移回整帧坐标系。

    标定时对帧做了 crop[y1:y2, x1:x2]（见 `scripts/calibration.py` 的 apply_crop）
    后再算内参，因此主点(cx,cy)在裁剪坐标系下。整帧像素(u,v)对应裁剪像素
    (u - x1, v - y1)，故整帧主点 = 裁剪主点 + (x1, y1)。畸变系数在归一化坐标下
    保持不变。

    crop 格式为 [y1, y2, x1, x2]；缺失或长度不足 4 时原样返回。
    """
    if not crop or len(crop) < 4:
        return camera_matrix
    y1, y2, x1, x2 = [int(v) for v in crop]
    m = camera_matrix.copy()
    m[0, 2] += x1  # cx
    m[1, 2] += y1  # cy
    return m


def get_3d_points(rect_width, rect_height):
    """纸板四个角的世界坐标（单位与 rect_width/rect_height 一致，默认 mm）。

    顺序与透视变换的目标点一致：左上、右上、右下、左下，z=0 置于纸板平面。
    """
    return np.array(
        [
            [0, 0, 0],
            [rect_width, 0, 0],
            [rect_width, rect_height, 0],
            [0, rect_height, 0],
        ],
        dtype=np.float32,
    )


def pnp_pose(camera_matrix, distortion_coeffs, obj_points, img_points,
             flags=cv2.SOLVEPNP_ITERATIVE):
    """对一组 2D 角点做 PnP 解算，返回位姿信息。

    参数:
        camera_matrix: 内参矩阵 (3x3)。
        distortion_coeffs: 畸变系数。
        obj_points: 世界坐标 (N,3)。
        img_points: 图像坐标 (N,2)；为空时返回 (False, None, None, None)。
        flags: cv2.solvePnP 的求解方法。

    返回:
        (success, distance, rotation_matrix, rvec)
        - success: bool，解算是否成功。
        - distance: 沿相机 z 轴的平移（与 obj_points 同单位，默认 mm）。
        - rotation_matrix: 3x3 旋转矩阵（世界->相机）。
        - rvec: 旋转向量。
        失败时相应项为 None。
    """
    if img_points is None or len(img_points) == 0:
        return False, None, None, None
    img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            distortion_coeffs,
            flags=flags,
        )
    except cv2.error as e:
        print(f"OpenCV error in solvePnP: {e}")
        return False, None, None, None
    if not success:
        return False, None, None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return success, translation_vector[2][0], rotation_matrix, rotation_vector


def extract_ypr(rvec):
    """从旋转向量提取欧拉角（yaw / pitch / roll，单位：度）。

    使用 cv2.RQDecomp3x3 把旋转矩阵分解为绕 x/y/z 轴的角度：
        roll  = Qx（绕相机光轴方向的平面内旋转）
        pitch = Qy（俯仰，绕图像横轴的倾斜）
        yaw   = Qz（偏航，绕图像纵轴的摆动）

    注意：轴向与符号约定依赖相机安装姿态，操作者应先把纸板摆正（放平、
    正对相机）记录此时读数作为零点，再观察相对偏差。本函数仅返回原始
    解算角度，不叠加任何偏移。
    """
    R = cv2.Rodrigues(rvec)[0]
    # 旋转矩阵 L = Rz(yaw)·Ry(pitch)·Rx(roll)（ZYX 顺序）标准欧拉角提取。
    # 纸板正对相机、放平时 R≈I，三个角≈0，作为角度零位。
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    s = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-R[2, 0], s))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    # 返回顺序: (yaw, pitch, roll)，单位度
    return yaw, pitch, roll


def a4_suitability(aspect, area):
    """根据检测四边形的宽高比判断其是否接近 A4 纸张比例。

    参数:
        aspect: 检测四边形外接矩形的宽高比（max/min，恒 >= 1）。
        area:   检测四边形面积（像素或任意单位），仅用于排查，不参与判定。

    返回:
        (verdict, aspect):
        - verdict: '适宜' / '偏长' / '偏宽'
        - aspect:  传入的宽高比。
    """
    if aspect <= 0:
        return "未知", aspect
    diff = abs(aspect - A4_ASPECT)
    if diff <= 0.15:
        verdict = "适宜"
    elif aspect > A4_ASPECT:
        verdict = "偏长"
    else:
        verdict = "偏宽"
    return verdict, aspect


def warp_to_plane(frame, corners, rect_width, rect_height):
    """把纸板内框透视矫正为矩形平面图。

    目标尺寸直接用 (rect_width, rect_height)（单位 mm），故 1px≈1mm，
    三个识别器返回的尺寸即 mm。corners 为 4 个内角点 (4,2)。
    """
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [[0, 0], [rect_width, 0], [rect_width, rect_height], [0, rect_height]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, transform, (rect_width, rect_height))


def annotate_rect(frame, corners, obj_points, camera_matrix, distortion_coeffs,
                  pnp_flag, camera_offset):
    """在 frame 上画内角点、距离、YPR 与 A4 适宜性，并返回元信息。

    纯函数（不依赖 Tk / 相机），所有输入均为参数；供 UI 与调试复用。
    返回 dict：{'distance', 'ypr':(yaw,pitch,roll), 'verdict', 'aspect'}。
    PnP 失败时返回各项为 None 的 dict，并在图上标 "PnP failed"。
    """
    meta = {"distance": None, "ypr": None, "verdict": None, "aspect": None}
    corners32 = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    for (x, y) in corners32:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

    success, distance, _rot_matrix, rvec = pnp_pose(
        camera_matrix, distortion_coeffs, obj_points, corners32, flags=pnp_flag,
    )
    if not success:
        cv2.putText(frame, "PnP failed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return meta

    distance_adj = camera_offset + distance
    yaw, pitch, roll = extract_ypr(rvec)
    meta["distance"] = distance_adj
    meta["ypr"] = (yaw, pitch, roll)

    x, y, w, h = cv2.boundingRect(corners32.astype(np.int32))
    aspect = float(w) / h if h else 0.0
    if aspect:
        aspect = max(aspect, 1.0 / aspect)
    verdict = a4_suitability(aspect, w * h)[0]
    meta["verdict"] = verdict
    meta["aspect"] = aspect

    cx, cy = np.mean(corners32, axis=0).astype(int)
    cv2.putText(frame, f"X:{distance_adj:.0f}mm", (cx, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}",
                (cx - 90, cy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    verdict_en = VERDICT_EN.get(verdict, verdict)
    verdict_color = VERDICT_COLOR.get(verdict, (0, 255, 255))
    cv2.putText(frame, f"A4:{verdict_en} {aspect:.2f}",
                (cx - 90, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2)
    return meta
