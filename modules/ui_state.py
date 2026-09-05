"""测量界面的可变状态：配置值、页面、任务、面积阈值与到各检测器的同步。

不依赖 Tk，可被 UI 与其它纯逻辑直接驱动（含串口解析处）。
"""

import cv2

from modules.geometry import PNP_FLAGS, get_3d_points

PAGE_ORIGINAL = "original"
PAGE_TRANSFORMED = "transformed"
PAGE_SETTINGS = "settings"
PAGES = (PAGE_ORIGINAL, PAGE_TRANSFORMED, PAGE_SETTINGS)


class UIState:
    """持有测量/识别所需的全部可变数值，并提供加载、同步与回写配置。"""

    def __init__(self, cfg=None, frame_size=None, debug=False):
        self.cfg = cfg or {}
        meas = self.cfg.get("measurement", {})
        self.camera_offset = float(meas.get("camera_offset_mm", -90.0))
        self.rect_width = int(meas.get("rect_width_mm", 170))
        self.rect_height = int(meas.get("rect_height_mm", 267))

        ui_cfg = self.cfg.get("ui", {})
        self.page = ui_cfg.get("page", PAGE_ORIGINAL)
        if self.page not in PAGES:
            self.page = PAGE_ORIGINAL
        self.pnp_name = ui_cfg.get("pnp_type", "ITERATIVE")
        if self.pnp_name not in PNP_FLAGS:
            self.pnp_name = "ITERATIVE"

        self.debug_mode = bool(debug) or bool(ui_cfg.get("debug", False))

        # 整帧分辨率：有相机用实际尺寸；无相机用 CS050 约 5MP 参考值，
        # 保证面积百分比滑块始终可调（px 仅为换算展示）。
        if frame_size and frame_size[0] and frame_size[1]:
            self.frame_w = int(frame_size[0])
            self.frame_h = int(frame_size[1])
            self._frame_estimated = False
        else:
            self.frame_w, self.frame_h = 2592, 1944
            self._frame_estimated = True
        self.frame_area = self.frame_w * self.frame_h

        # 面积筛选：以「帧面积百分比」为权威值，随框架相对偏移、跨相机可迁移。
        # 兼容历史绝对值键(rect_min_area/rect_max_area，像素)：仅当非默认值
        # (150000/3000000)时按当前帧换算成百分比，否则用帧相对默认 5%/60%。
        pct_min = ui_cfg.get("rect_min_area_pct")
        pct_max = ui_cfg.get("rect_max_area_pct")
        if pct_min is not None and pct_max is not None:
            self.min_area_pct = float(pct_min)
            self.max_area_pct = float(pct_max)
        else:
            cfg_min = int(ui_cfg.get("rect_min_area", 150000))
            cfg_max = int(ui_cfg.get("rect_max_area", 3000000))
            is_legacy = (cfg_min == 150000 and cfg_max == 3000000)
            if self.frame_area and not is_legacy:
                self.max_area_pct = min(100.0, cfg_max / self.frame_area * 100.0)
                self.min_area_pct = min(self.max_area_pct, cfg_min / self.frame_area * 100.0)
            else:
                self.min_area_pct = 5.0
                self.max_area_pct = 60.0
        self.min_area_pct = max(0.0, min(100.0, self.min_area_pct))
        self.max_area_pct = max(self.min_area_pct, min(100.0, self.max_area_pct))
        if self.frame_area:
            self.min_area = int(round(self.min_area_pct / 100.0 * self.frame_area))
            self.max_area = int(round(self.max_area_pct / 100.0 * self.frame_area))
        else:
            self.min_area = int(self.min_area_pct)
            self.max_area = int(self.max_area_pct)

        # 运行时状态（非配置派生）
        self.current_task = None
        self.middle_data_power = ""

    @property
    def obj_points(self):
        return get_3d_points(self.rect_width, self.rect_height)

    @property
    def pnp_flag(self):
        return PNP_FLAGS.get(self.pnp_name, cv2.SOLVEPNP_ITERATIVE)

    def apply_geometry(self, shape_detector=None, min_square_detector=None,
                       rectangle_detector=None):
        """把当前补偿/尺寸同步到各识别器与面积筛选。"""
        if shape_detector is not None:
            shape_detector.frame_real_width = float(self.rect_width)
            shape_detector.frame_real_height = float(self.rect_height)
        if min_square_detector is not None:
            min_square_detector.world_width = float(self.rect_width)
            min_square_detector.world_height = float(self.rect_height)
        if rectangle_detector is not None:
            rectangle_detector.min_area = int(self.min_area)
            rectangle_detector.max_area = int(self.max_area)
