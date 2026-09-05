"""在帧上叠加 A4 比例的面积筛选参照矩形（纯 OpenCV 绘制，不依赖 Tk）。

供原始/设置页作为图例，与检测到的纸板外框对比当前最小/最大面积梯度。
"""

import math

import cv2

from modules.geometry import A4_ASPECT


def draw_filter_template(img, min_area, max_area, frame_area, a4_aspect=A4_ASPECT):
    """在 img 上按当前面积筛选范围画两个竖向 A4 参考矩形（内=min，外=max）。

    保持 A4 竖向比例(长/短≈1.414)。按「面积占帧面积的 √比例」换算边长并以
    画布中心摆放，随滑块连续增长；仅当 A4 超过画布高度（物理上限）时才贴到
    画布边界，属正常现象。
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = a4_aspect
    for area, color, label in (
        (min_area, (255, 160, 0), f"MIN {min_area}"),
        (max_area, (0, 255, 255), f"MAX {max_area}"),
    ):
        if area <= 0 or not frame_area:
            continue
        f = min(1.0, area / frame_area)
        wr = math.sqrt(f * w * h / r)
        hr = wr * r
        if wr > w or hr > h:
            s = min(w / wr, h / hr)
            wr *= s
            hr *= s
        x0 = int(cx - wr / 2)
        y0 = int(cy - hr / 2)
        cv2.rectangle(img, (x0, y0), (int(x0 + wr), int(y0 + hr)), color, 2)
        cv2.putText(img, label, (x0 + 4, y0 + 18), font, 0.5, color, 2)
