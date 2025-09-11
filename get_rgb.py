# import cv2
# from pathlib import Path
# from datetime import datetime
# from HIK_CAM import HikIndustrialCamera

# # ---------------- 保存路径 ----------------
# SAVE_DIR = Path(r"D:/StudyWorks/Study_python/DS_ws/rgb_save")      # 可自行修改
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# camera = HikIndustrialCamera()
# camera.init()
# camera.open()

# while True:
#     frame = camera.get_frame()
#     # roi = frame[664:1384, 864:1584]
#     if frame is not None:
#         roi = frame[124:1924,324:2124]

#         # —— 原来的二值处理流程 ——
#         # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         # gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
#         # gray = cv2.GaussianBlur(gray, (5, 5), 0)
#         # # _, binary = cv2.threshold(gray,50, 255,
#         # #                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         # _, binary = cv2.threshold(gray,100, 255,
#         #                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
#         # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆核更平滑
#         # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#         # —— 显示 ——
#         cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("binary", 1200, 1200)
#         cv2.imshow("binary", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord(' '):                    # 空格保存
#             # 用时间戳生成几乎不会重复的文件名
#             ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
#             color_path = SAVE_DIR / f"frame_{ts}.png"
#             cv2.imwrite(str(color_path), frame)
#             print(f"Saved: {color_path}")
#     else:
#         continue

# camera.close()
# cv2.destroyAllWindows()

import cv2
from pathlib import Path
from datetime import datetime
from HIK_CAM import HikIndustrialCamera

# ---------------- 保存路径 ----------------
SAVE_DIR = Path(r"D:/StudyWorks/Study_python/DS_ws/rgb_save")      # 可自行修改
SAVE_DIR.mkdir(parents=True, exist_ok=True)

camera = HikIndustrialCamera()
camera.init()
camera.open()

while True:
    frame = camera.get_frame()
    if frame is not None:
        # 提取感兴趣区域
        roi = frame[124:1924, 324:2124]
        
        # 将图像转为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 黑白颜色取反（不进行二值化，保留灰度层次）
        inverted = 255 - gray  # 核心操作：用255减去每个像素值实现反转
        
        # —— 显示 ——
        cv2.namedWindow("Inverted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Inverted", 1200, 1200)
        cv2.imshow("Inverted", inverted)
        
        # 同时显示原始彩色图像（可选）
        # cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Original", 1200, 1200)
        # cv2.imshow("Original", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):                    # 空格保存
            # 用时间戳生成几乎不会重复的文件名
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            # 保存反转后的图像
            inverted_path = SAVE_DIR / f"inverted_{ts}.png"
            cv2.imwrite(str(inverted_path), inverted)
            # 也可以同时保存原始图像
            # color_path = SAVE_DIR / f"original_{ts}.png"
            # cv2.imwrite(str(color_path), roi)
            print(f"Saved inverted image: {inverted_path}")
    else:
        continue

camera.close()
cv2.destroyAllWindows()
