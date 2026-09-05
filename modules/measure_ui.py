"""Tkinter 多页测量界面（对齐 crane 的设计风格）。

结构参考:
    crane/models/ui/viewer.py     - CameraViewer 基础视频渲染
    crane/models/ui/calibration   - CalibrationUI 模式按钮切页 + tk.Scale 滑块
    crane/models/ui/competition   - CompetitionUI DEBUG 强制推理 + 状态栏

页面:
    原始   -> 相机原帧 + 矩形外框 + PnP 距离 / YPR / A4 适宜性
    变换   -> 透视矫正后的内框图 + 识别信息

滑块补偿: 距离 offset + 内框真实宽高(rect_width/rect_height_mm)。
面积筛选: min/max 面积（写入 RectangleDetector.min_area/max_area）。
"""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

from modules.geometry import (
    A4_ASPECT,
    a4_suitability,
    extract_ypr,
    get_3d_points,
    pnp_pose,
)

# solvePnP 求解方法 名称 -> OpenCV 常量
PNP_FLAGS = {
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "EPNP": cv2.SOLVEPNP_EPNP,
    "IPPE": cv2.SOLVEPNP_IPPE,
    "DLS": cv2.SOLVEPNP_DLS,
    "UPNP": cv2.SOLVEPNP_UPNP,
}

# 页面 / 任务常量
PAGE_ORIGINAL = "original"
PAGE_TRANSFORMED = "transformed"
PAGE_SETTINGS = "settings"
TASK_SHAPE = "shape"
TASK_MIN_SQUARE = "min_square"
TASK_DIGIT = "digit"

# 画面上 OpenCV 无法渲染中文，A4 判定用英文缩写 + 颜色
VERDICT_EN = {"适宜": "FIT", "偏长": "TALL", "偏宽": "WIDE"}
VERDICT_COLOR = {"适宜": (0, 255, 0), "偏长": (0, 200, 255), "偏宽": (255, 160, 0)}

SLIDER_GEOMETRY = {
    # key: (标题, 默认值, 最小值, 最大值)  —— 按页面关联分组
    "camera_offset": ("距离补偿(mm)", -90.0, -500.0, 500.0),
    "rect_width": ("内框宽度(mm)", 170, 100, 280),
    "rect_height": ("内框高度(mm)", 267, 150, 320),
    "min_area": ("最小面积(px)", 150000, 0, 500000),
    "max_area": ("最大面积(px)", 3000000, 500000, 4000000),
}
# 原始页：日常要调的（距离补偿、面积筛选）
ORIGINAL_SLIDERS = ("camera_offset", "min_area", "max_area")
# 设置页：不常改的（内框宽高）
SETTINGS_SLIDERS = ("rect_width", "rect_height")


class MeasurementUI(tk.Tk):
    """单相机单目测距/识别 GUI。"""

    def __init__(
        self,
        cfg,
        cfg_path,
        camera,
        rectangle_detector,
        shape_detector,
        min_square_detector,
        digit_detector,
        camera_matrix,
        distortion_coeffs,
        serial_queue=None,
        debug=False,
        frame_size=None,
    ):
        super().__init__()
        self.cfg = cfg or {}
        self.cfg_path = cfg_path
        self.camera = camera
        self.rectangle_detector = rectangle_detector
        self.shape_detector = shape_detector
        self.min_square_detector = min_square_detector
        self.digit_detector = digit_detector
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.serial_queue = serial_queue
        self.debug_mode = bool(debug)
        # 整帧分辨率：有相机用实际尺寸；无相机用 CS050 约 5MP 参考值，
        # 保证面积百分比滑块始终可调（px 仅为换算展示）。
        if frame_size and frame_size[0] and frame_size[1]:
            self.frame_w, self.frame_h = int(frame_size[0]), int(frame_size[1])
            self._frame_estimated = False
        else:
            self.frame_w, self.frame_h = 2592, 1944
            self._frame_estimated = True
        self.frame_area = self.frame_w * self.frame_h

        # ------- 可变状态 -------
        self.obj_points = None
        self.pnp_flag = cv2.SOLVEPNP_ITERATIVE
        self.page = PAGE_ORIGINAL
        self.current_task = None       # None/TASK_SHAPE/TASK_MIN_SQUARE/TASK_DIGIT
        self.selected_digit = None     # None=自动
        self.middle_data_power = ""
        self.last_choice_time = 0.0
        self.min_choice_interval = 0.5

        self._running = False
        self._result_lock = threading.Lock()
        self._latest_result = None     # {'annotated','size','tag'}
        self._digit_queue = queue.Queue(maxsize=1)
        self._digit_worker = None

        self._load_state()
        self._build_ui()
        self._apply_geometry()

    # ---------- 状态 / 配置 ----------
    def _load_state(self):
        meas = self.cfg.get("measurement", {})
        self.camera_offset = float(meas.get("camera_offset_mm", -90.0))
        self.rect_width = int(meas.get("rect_width_mm", 170))
        self.rect_height = int(meas.get("rect_height_mm", 267))

        ui_cfg = self.cfg.get("ui", {})
        page = ui_cfg.get("page", PAGE_ORIGINAL)
        self.page = page if page in (PAGE_ORIGINAL, PAGE_TRANSFORMED, PAGE_SETTINGS) else PAGE_ORIGINAL
        self.pnp_name = ui_cfg.get("pnp_type", "ITERATIVE")
        if self.pnp_name not in PNP_FLAGS:
            self.pnp_name = "ITERATIVE"

        # 面积筛选：与整帧尺寸挂钩（默认下限 5%、上限 60% 帧面积），
        # 仅当配置里仍是旧的硬编码默认值(150000/3000000)时才用帧相对默认；
        # 操作者手动调整并 SAVE 过的值会被保留。
        cfg_min = int(ui_cfg.get("rect_min_area", 150000))
        cfg_max = int(ui_cfg.get("rect_max_area", 3000000))
        is_legacy = (cfg_min == 150000 and cfg_max == 3000000)
        if self.frame_area:
            # 面积以“帧面积百分比”为滑块单位，并换算成像素供过滤使用。
            # 仅当配置仍是旧硬编码时用 5%~60% 默认，操作者手调并 SAVE 过的保留。
            if is_legacy:
                self.min_area_pct = 5.0
                self.max_area_pct = 60.0
            else:
                self.max_area_pct = min(100.0, cfg_max / self.frame_area * 100.0)
                self.min_area_pct = min(self.max_area_pct, cfg_min / self.frame_area * 100.0)
            self.min_area = int(round(self.min_area_pct / 100.0 * self.frame_area))
            self.max_area = int(round(self.max_area_pct / 100.0 * self.frame_area))
        else:
            self.min_area_pct = None
            self.max_area_pct = None
            self.min_area = cfg_min
            self.max_area = cfg_max
            if self.min_area > self.max_area:
                self.min_area, self.max_area = self.max_area, self.min_area

        if not self.debug_mode:
            self.debug_mode = bool(ui_cfg.get("debug", False))

    def _apply_geometry(self):
        """把当前补偿/尺寸同步到 PnP 目标点、warp 目标与各识别器 world 尺寸。"""
        self.obj_points = get_3d_points(self.rect_width, self.rect_height)
        self.pnp_flag = PNP_FLAGS.get(self.pnp_name, cv2.SOLVEPNP_ITERATIVE)
        if self.shape_detector is not None:
            self.shape_detector.frame_real_width = float(self.rect_width)
            self.shape_detector.frame_real_height = float(self.rect_height)
        if self.min_square_detector is not None:
            self.min_square_detector.world_width = float(self.rect_width)
            self.min_square_detector.world_height = float(self.rect_height)
        if self.rectangle_detector is not None:
            self.rectangle_detector.min_area = int(self.min_area)
            self.rectangle_detector.max_area = int(self.max_area)

    # ---------- UI 构建 ----------
    def _build_ui(self):
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            pass

        self.title("单目测量装置 - NUEDC2025")
        self.geometry("1024x680")
        self.minsize(800, 560)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)  # 视频行弹性
        main_frame.grid_columnconfigure(0, weight=1)

        # 顶栏（始终可见）：页面切换 + DEBUG/SAVE/EXIT
        top_bar = tk.Frame(main_frame)
        top_bar.grid(row=0, column=0, sticky="ew")
        self._build_top_bar(top_bar)

        # 视频区
        self.video_container = tk.Frame(main_frame, bg="black")
        self.video_container.grid(row=1, column=0, sticky="nsew")
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.grid(row=0, column=0)

        # 与当前页相关联的控件区（随页面切换）
        controls_frame = tk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, sticky="ew")
        self._controls_frame = controls_frame
        self._orig_controls = tk.Frame(controls_frame)
        self._trans_controls = tk.Frame(controls_frame)
        self._settings_controls = tk.Frame(controls_frame)
        self._build_original_controls(self._orig_controls)
        self._build_transformed_controls(self._trans_controls)
        self._build_settings_controls(self._settings_controls)

        # 状态栏（始终可见）
        status_frame = tk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky="ew")
        self._build_status_bar(status_frame)

        self.bind("<KeyPress-1>", lambda _e: self._set_page(PAGE_ORIGINAL))
        self.bind("<KeyPress-2>", lambda _e: self._set_page(PAGE_TRANSFORMED))
        self.bind("<KeyPress-3>", lambda _e: self._set_page(PAGE_SETTINGS))
        self.bind("<KeyPress-d>", self._toggle_debug)
        self.bind("<KeyPress-q>", lambda _e: self._quit())
        self.bind("<KeyPress-space>", self._toggle_page)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._show_page_controls(self.page)

    def _build_top_bar(self, parent):
        font = ("Helvetica", 10)
        self._page_buttons = {}
        for page, text in (
            (PAGE_ORIGINAL, "原始(1)"),
            (PAGE_TRANSFORMED, "变换(2)"),
            (PAGE_SETTINGS, "设置(3)"),
        ):
            btn = tk.Button(
                parent, text=text, command=lambda p=page: self._set_page(p),
                bg="#555555", fg="white", font=font, width=8,
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self._page_buttons[page] = btn

        self._exit_btn = tk.Button(
            parent, text="EXIT", command=self._quit, bg="#aa3333", fg="white", font=font, width=6,
        )
        self._exit_btn.pack(side=tk.RIGHT, padx=2, pady=2)
        self._save_btn = tk.Button(
            parent, text="SAVE", command=self._save_cfg, bg="#0066cc", fg="white", font=font, width=6,
        )
        self._save_btn.pack(side=tk.RIGHT, padx=2, pady=2)
        self._debug_btn = tk.Button(
            parent, text="DEBUG", command=self._toggle_debug, bg="#555555", fg="white", font=font, width=6,
        )
        self._debug_btn.pack(side=tk.RIGHT, padx=2, pady=2)

    def _build_original_controls(self, parent):
        """原始页关联控件：日常要调的——距离补偿、面积筛选 + A4 图例。"""
        self.sliders_frame = tk.Frame(parent)
        self.sliders_frame.pack(fill=tk.X)
        self._build_sliders(self.sliders_frame, ORIGINAL_SLIDERS)

    def _build_transformed_controls(self, parent):
        """变换页关联控件：识别任务与数字选择（影响识别结果）。"""
        self._build_task_controls(parent)

    def _build_settings_controls(self, parent):
        """设置页关联控件：不常改的——内框宽高、PnP 类型、帧分辨率。"""
        font = ("Helvetica", 10)
        tk.Label(parent, text=self._frame_size_text(), font=font, anchor=tk.W).pack(
            fill=tk.X, padx=(2, 0)
        )
        self.settings_sliders_frame = tk.Frame(parent)
        self.settings_sliders_frame.pack(fill=tk.X)
        self._build_sliders(self.settings_sliders_frame, SETTINGS_SLIDERS)

        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="PnP类型", width=11, anchor=tk.W, font=font).pack(side=tk.LEFT, padx=(2, 0))

    def _frame_size_text(self):
        suffix = "（CS050 5MP 估算）" if getattr(self, "_frame_estimated", False) else ""
        return f"帧分辨率: {self.frame_w} × {self.frame_h}  ({self.frame_area:,}px²){suffix}"
        self._pnp_var = tk.StringVar(value=self.pnp_name)
        self._pnp_combo = ttk.Combobox(
            row, textvariable=self._pnp_var, state="readonly", width=9,
            values=list(PNP_FLAGS.keys()), font=font,
        )
        self._pnp_combo.pack(side=tk.LEFT, padx=2)
        self._pnp_combo.bind("<<ComboboxSelected>>", self._on_pnp_selected)

    def _show_page_controls(self, page):
        """只显示与当前页相关联的控件。"""
        frames = (self._orig_controls, self._trans_controls, self._settings_controls)
        for frame in frames:
            frame.pack_forget()
        if page == PAGE_ORIGINAL:
            self._orig_controls.pack(fill=tk.X)
        elif page == PAGE_TRANSFORMED:
            self._trans_controls.pack(fill=tk.X)
        else:
            self._settings_controls.pack(fill=tk.X)

    def _build_sliders(self, parent, keys):
        font = ("Helvetica", 9)
        for key in keys:
            title, default, lo, hi = SLIDER_GEOMETRY[key]
            if key in ("min_area", "max_area"):
                is_min = key == "min_area"
                if self.frame_area:
                    # 面积用“帧面积百分比”为单位，直接可拖
                    title = "最小面积(%)" if is_min else "最大面积(%)"
                    lo, hi = 0.0, 100.0
                    default = self.min_area_pct if is_min else self.max_area_pct
                else:
                    # 无帧尺寸（空跑）退化为像素
                    default = self.min_area if is_min else self.max_area
            row = tk.Frame(parent)
            row.pack(fill=tk.X, pady=0)
            tk.Label(row, text=title, width=11, anchor=tk.W, font=font).pack(
                side=tk.LEFT, padx=(2, 0)
            )
            var = tk.DoubleVar(value=default)
            tk.Scale(
                row, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var,
                command=lambda value, k=key: self._on_slider(k, value),
                showvalue=True, length=300, width=6, font=font, bd=0,
                highlightthickness=0, troughcolor="#cccccc",
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def _build_task_controls(self, parent):
        font = ("Helvetica", 10)
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=1)

        tk.Label(row, text="识别任务", width=11, anchor=tk.W, font=font).pack(
            side=tk.LEFT, padx=(2, 0)
        )
        self._task_var = tk.StringVar(value=self.current_task or TASK_SHAPE)
        self._task_radios = {}
        for text, val in (("形状", TASK_SHAPE), ("最小方块", TASK_MIN_SQUARE), ("数字", TASK_DIGIT)):
            rb = tk.Radiobutton(
                row, text=text, value=val, variable=self._task_var,
                command=self._on_task_radio, font=font,
            )
            rb.pack(side=tk.LEFT, padx=4)
            self._task_radios[val] = rb

        tk.Label(row, text="数字", font=font).pack(side=tk.LEFT, padx=(8, 0))
        self._digit_var = tk.StringVar(value="auto")
        self._digit_combo = ttk.Combobox(
            row, textvariable=self._digit_var, state="readonly", width=5,
            values=["auto"] + [str(i) for i in range(10)], font=font,
        )
        self._digit_combo.pack(side=tk.LEFT, padx=2)
        self._digit_combo.bind("<<ComboboxSelected>>", self._on_digit_selected)

        self._refresh_task_controls()

    def _build_status_bar(self, parent):
        self._status_labels = {}
        font = ("Helvetica", 10)
        for key, title in (("cam", "相机"), ("serial", "串口"), ("task", "任务"),
                           ("pnp", "PnP"), ("dist", "读数")):
            label = tk.Label(parent, text=f"{title}: -", font=font, fg="white", bg="#555555")
            label.pack(side=tk.LEFT, padx=3)
            self._status_labels[key] = label

    # ---------- 滑块 / 控件回调 ----------
    def _on_slider(self, key, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if key == "camera_offset":
            self.camera_offset = value
        elif key == "rect_width":
            self.rect_width = int(round(value))
        elif key == "rect_height":
            self.rect_height = int(round(value))
        elif key in ("min_area", "max_area"):
            if self.frame_area:
                if key == "min_area":
                    self.min_area_pct = value
                    if self.min_area_pct > self.max_area_pct:
                        self.min_area_pct = self.max_area_pct
                else:
                    self.max_area_pct = value
                    if self.max_area_pct < self.min_area_pct:
                        self.max_area_pct = self.min_area_pct
                self.min_area = int(round(self.min_area_pct / 100.0 * self.frame_area))
                self.max_area = int(round(self.max_area_pct / 100.0 * self.frame_area))
            else:
                if key == "min_area":
                    self.min_area = int(round(value))
                else:
                    self.max_area = int(round(value))
                if self.min_area > self.max_area:
                    self.min_area = self.max_area
        self._apply_geometry()

    def _on_pnp_selected(self, _event=None):
        self.pnp_name = self._pnp_var.get()
        self._apply_geometry()
        self._update_status()

    def _on_task_radio(self):
        task = self._task_var.get()
        self.current_task = task
        self._update_status()

    def _on_digit_selected(self, _event=None):
        val = self._digit_var.get()
        self.selected_digit = None if val == "auto" else val

    def _set_task(self, task):
        if task == self.current_task:
            return
        self.current_task = task
        self._task_var.set(task)
        self._update_status()

    def _set_page(self, page):
        if page == self.page:
            return
        self.page = page
        self.cfg.setdefault("ui", {})["page"] = page
        self._update_page_buttons()
        self._show_page_controls(page)

    def _update_page_buttons(self):
        colors = {PAGE_ORIGINAL: "#00aa00", PAGE_TRANSFORMED: "#0066cc", PAGE_SETTINGS: "#cc6600"}
        for page, btn in getattr(self, "_page_buttons", {}).items():
            if page == self.page:
                btn.config(bg=colors.get(page, "#555555"), relief=tk.SUNKEN)
            else:
                btn.config(bg="#555555", relief=tk.RAISED)

    def _toggle_page(self, _event=None):
        self._set_page(PAGE_TRANSFORMED if self.page == PAGE_ORIGINAL else PAGE_ORIGINAL)

    def _toggle_debug(self, _event=None):
        self.debug_mode = not self.debug_mode
        self._refresh_task_controls()
        self._update_debug_button()
        self._update_status()

    def _update_debug_button(self):
        if self.debug_mode:
            self._debug_btn.config(text="DEBUG:ON", bg="#cc6600", relief=tk.SUNKEN)
        else:
            self._debug_btn.config(text="DEBUG", bg="#555555", relief=tk.RAISED)

    def _refresh_task_controls(self):
        # 无串口或 debug 时，允许 UI 自由选择任务；否则由串口驱动，置灰。
        ui_controlled = self.debug_mode or self.serial_queue is None
        state = tk.NORMAL if ui_controlled else tk.DISABLED
        for rb in self._task_radios.values():
            rb.config(state=state)
        self._digit_combo.config(state="readonly" if ui_controlled else tk.DISABLED)

    def _update_status(self):
        try:
            self._set_status("cam", self.camera is not None)
            self._set_status("serial", self.serial_queue is not None)
            task_names = {TASK_SHAPE: "形状", TASK_MIN_SQUARE: "最小方块", TASK_DIGIT: "数字"}
            task = task_names.get(self.current_task, "-")
            digit = "" if self.selected_digit is None else self.selected_digit
            self._status_labels["task"].config(text=f"任务: {task}{digit}")
            self._status_labels["pnp"].config(text=f"PnP: {self.pnp_name}")
        except Exception:
            pass

    def _set_status(self, key, ok):
        label = self._status_labels.get(key)
        if label is None:
            return
        if ok is None:
            text, bg = f"{key}: -", "#666666"
        elif ok:
            text, bg = f"{key}: OK", "#006600"
        else:
            text, bg = f"{key}: 无", "#555555"
        label.config(text=text, bg=bg)

    # ---------- 串口 ----------
    def _process_serial_queue(self):
        if self.serial_queue is None:
            return
        while True:
            try:
                port_type, data = self.serial_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if self.debug_mode:
                continue  # debug 模式丢弃串口数据，避免队列无限增长
            try:
                if port_type == "power":
                    if len(data) >= 6 and data[5] == "A" and data.endswith("W"):
                        self.middle_data_power = "I:" + data[:8] + "P:" + data[8:16] + "PM:" + data[16:]
                        print(f"Power received: {self.middle_data_power}")
                elif port_type == "choice":
                    now = time.time()
                    if now - self.last_choice_time < self.min_choice_interval:
                        print(f"Ignoring choice message (too frequent): {data}")
                        continue
                    self.last_choice_time = now
                    print(f"Choice received: {data}")
                    if data.startswith("g") and data.endswith("l"):
                        mid = data[1:-1]
                        if "as" in mid:
                            self._set_task(TASK_SHAPE)
                        elif "bs" in mid:
                            self._set_task(TASK_MIN_SQUARE)
                        elif "d" in mid:
                            ci = mid.find("d")
                            if ci != -1 and ci + 1 < len(mid) and mid[ci + 1].isdigit():
                                self._set_task(TASK_DIGIT)
                                self.selected_digit = mid[ci + 1]
                                self._digit_var.set(self.selected_digit)
                                self._update_status()
            except Exception as e:
                print(f"serial parse error: {e}")

    # ---------- 帧处理 ----------
    def _update_frame(self):
        if not self._running:
            return
        try:
            self._process_serial_queue()
            self._process_frame_live()
        except Exception as e:
            print(f"frame loop error: {e}")
        if self._running:
            self.after(30, self._update_frame)

    def _process_frame_live(self):
        frame = None
        if self.camera is None:
            self._show_empty_page("相机未连接 (空跑模式)")
            self._update_status()
            return
        try:
            frame = self.camera.get_frame()
        except Exception:
            frame = None
        if frame is None:
            self._show_empty_page("相机无帧 / 未连接")
            return

        corners_list = []
        try:
            corners_list = self.rectangle_detector.detect(frame)
        except Exception as e:
            print(f"rect detect error: {e}")

        page1 = frame.copy()
        warped = None
        last_meta = None

        for corners in corners_list:
            meta = self._annotate_rect(page1, corners)
            if warped is None:
                warped = self._warp(frame, corners)
                last_meta = meta

        self._run_recognition(warped, last_meta)

        # 在原始帧上叠加面积筛选 A4 模板，便于对比检测框
        self._draw_filter_template(page1)

        if self.page in (PAGE_ORIGINAL, PAGE_SETTINGS):
            self._show_frame(self._contain(page1))
        elif self.page == PAGE_TRANSFORMED:
            self._show_frame(self._build_page2())

        self._update_status_dist(last_meta)
        self._update_status()

    def _annotate_rect(self, page1, corners):
        """在 page1 上画角点/距离/YPR/A4 判定，返回元信息 dict。"""
        meta = {"distance": None, "ypr": None, "verdict": None, "aspect": None}
        corners32 = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        for (x, y) in corners32:
            cv2.circle(page1, (int(x), int(y)), 6, (0, 0, 255), -1)

        success, distance, rot_matrix, rvec = pnp_pose(
            self.camera_matrix, self.distortion_coeffs, self.obj_points, corners32,
            flags=self.pnp_flag,
        )
        if not success:
            cv2.putText(page1, "PnP failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return meta

        distance_adj = self.camera_offset + distance
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
        cv2.putText(page1, f"X:{distance_adj:.0f}mm", (cx, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(page1, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}",
                    (cx - 90, cy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        verdict_en = VERDICT_EN.get(verdict, verdict)
        verdict_color = VERDICT_COLOR.get(verdict, (0, 255, 255))
        cv2.putText(page1, f"A4:{verdict_en} {aspect:.2f}",
                    (cx - 90, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2)
        return meta

    def _warp(self, frame, corners):
        dst = np.array(
            [[0, 0], [self.rect_width, 0], [self.rect_width, self.rect_height], [0, self.rect_height]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        return cv2.warpPerspective(frame, transform, (self.rect_width, self.rect_height))

    def _run_recognition(self, warped, meta):
        """根据当前任务跑识别器；数字推理走后台 worker，其余同步。"""
        if warped is None:
            self._set_latest_result(None, None, "未检测到矩形", meta)
            return
        if self.current_task is None:
            self._set_latest_result(None, None, "等待任务选择", meta)
            return
        try:
            if self.current_task == TASK_SHAPE:
                result, annotated = self.shape_detector.detect_shape(warped)
                size = result.get("size", 0.0) or 0.0
                label = result.get("shape", "unknown")
                self._set_latest_result(annotated, size, f"形状:{label}  D:{size:.1f}mm", meta)
            elif self.current_task == TASK_MIN_SQUARE:
                annotated, size = self.min_square_detector.detect(warped)
                tag = f"最小方块  D:{size:.1f}mm" if size is not None else "最小方块  D:N/A"
                self._set_latest_result(annotated, size, tag, meta)
            elif self.current_task == TASK_DIGIT:
                self._submit_digit(warped, meta)
        except Exception as e:
            print(f"recognition error: {e}")
            self._set_latest_result(None, None, f"识别错误:{e}", meta)

    def _submit_digit(self, warped, meta):
        try:
            self._digit_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass
        try:
            self._digit_queue.put_nowait((warped.copy(), meta))
        except queue.Full:
            pass

    def _digit_worker_loop(self):
        while self._running:
            try:
                warped, meta = self._digit_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                continue
            if self.digit_detector is None:
                self._set_latest_result(None, None, "数字模型不可用", meta)
                continue
            try:
                # 始终同步当前选择（含 None=自动），避免切换后残留旧数字
                self.digit_detector.set_selected_digit(self.selected_digit)
                annotated, size = self.digit_detector.process_frame(warped)
                tag = f"数字:{self.selected_digit or 'auto'}  D:{size:.1f}mm" if size is not None else "数字:未检测到"
                self._set_latest_result(annotated, size, tag, meta)
            except Exception as e:
                print(f"digit worker error: {e}")
                self._set_latest_result(None, None, f"数字推理错误:{e}", meta)

    def _set_latest_result(self, annotated, size, tag, meta=None):
        with self._result_lock:
            self._latest_result = {"annotated": annotated, "size": size, "tag": tag}

    def _build_page2(self):
        with self._result_lock:
            res = self._latest_result
        if res is None or res.get("annotated") is None:
            return self._placeholder("未检测到矩形 / 等待任务")
        disp = res["annotated"].copy()
        cv2.putText(disp, res.get("tag", ""), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return self._contain(disp)

    def _update_status_dist(self, meta):
        label = self._status_labels.get("dist")
        if label is None:
            return
        if meta and meta.get("distance") is not None:
            yaw, pitch, roll = meta.get("ypr", (0, 0, 0))
            label.config(text=f"读数: X:{meta['distance']:.0f}mm Y:{yaw:.0f}° P:{pitch:.0f}° R:{roll:.0f}°")
        else:
            label.config(text="读数: -")

    # ---------- 渲染工具 ----------
    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = ImageTk.PhotoImage(img)
        self.video_label.imgtk = img
        self.video_label.config(image=img)

    def _container_size(self):
        """读视频区当前大小（随窗口自适应，非固定）。"""
        try:
            w = int(self.video_container.winfo_width())
            h = int(self.video_container.winfo_height())
        except Exception:
            w, h = 1000, 560
        return max(1, w), max(1, h)

    def _contain(self, frame):
        avail_w, avail_h = self._container_size()
        h, w = frame.shape[:2]
        scale = min(avail_w / w, avail_h / h)
        if scale <= 0 or scale == 1:
            return frame
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    def _show_placeholder(self, text):
        self._show_frame(self._placeholder(text))

    def _placeholder(self, text):
        w, h = self._container_size()
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (max(20, w // 10), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
        return img

    def _draw_filter_template(self, img):
        """按当前面积筛选范围画两个竖向 A4 参考矩形（内=min，外=max）。

        保持 A4 竖向比例(长/短≈1.414)。按「面积占帧面积的 √比例」换算边长并
        以画布中心摆放，随滑块连续增长；仅当 A4 超过画布高度（物理上限）时才
        贴到画布边界，属正常现象。
        """
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        r = A4_ASPECT
        for area, color, label in (
            (self.min_area, (255, 160, 0), f"MIN {self.min_area}"),
            (self.max_area, (0, 255, 255), f"MAX {self.max_area}"),
        ):
            if area <= 0 or not self.frame_area:
                continue
            f = min(1.0, area / self.frame_area)
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

    def _show_empty_page(self, text):
        """无帧时显示占位图；A4 面积模板只在原始/设置页叠加，变换页干净显示。"""
        ph = self._placeholder(text)
        if self.page in (PAGE_ORIGINAL, PAGE_SETTINGS):
            self._draw_filter_template(ph)
        self._show_frame(ph)

    # ---------- 保存 / 退出 ----------
    def _save_cfg(self):
        if not self.cfg_path:
            return
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("measurement", {}).update(
                camera_offset_mm=self.camera_offset,
                rect_width_mm=self.rect_width,
                rect_height_mm=self.rect_height,
            )
            cfg.setdefault("ui", {}).update(
                page=self.page,
                pnp_type=self.pnp_name,
                rect_min_area=self.min_area,
                rect_max_area=self.max_area,
                debug=self.debug_mode,
            )
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            self._save_btn.config(text="SAVED ✓", bg="#006600")
            self.after(1500, self._restore_save_btn)
        except Exception as e:
            print(f"save config error: {e}")
            self._save_btn.config(text="ERROR", bg="#aa0000")

    def _restore_save_btn(self):
        self._save_btn.config(text="SAVE", bg="#0066cc")

    def _quit(self, _event=None):
        self._running = False
        try:
            self.destroy()
        except Exception:
            pass

    # ---------- 生命周期 ----------
    def run(self):
        self._running = True
        self._digit_worker = threading.Thread(target=self._digit_worker_loop, daemon=True)
        self._digit_worker.start()
        self._update_status()
        self._update_frame()
        try:
            self.mainloop()
        finally:
            self._running = False
            try:
                if self.camera is not None:
                    self.camera.close()
            except Exception:
                pass
