"""Tkinter 多页测量界面（对齐 crane 的设计风格）。

结构参考:
    crane/models/ui/viewer.py     - CameraViewer 基础视频渲染
    crane/models/ui/calibration   - CalibrationUI 模式按钮切页 + tk.Scale 滑块
    crane/models/ui/competition   - CompetitionUI DEBUG 强制推理 + 状态栏

测量状态 / 识别编排 / 串口解析 / 几何标注均已下沉到对应模块，本类只负责：
    - Tk 控件组装与页面切换
    - 每帧调用 state 与 recognizer 完成 取帧->检测->标注->识别->渲染
    - 串口队列的节流 / 分发
"""

import queue
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

from modules import serial_protocol
from modules.filter_template import draw_filter_template
from modules.geometry import PNP_FLAGS, annotate_rect, warp_to_plane
from modules.recognizer import (
    Recognizer,
    TASK_DIGIT,
    TASK_MIN_SQUARE,
    TASK_SHAPE,
)
from modules.ui_state import (
    PAGE_ORIGINAL,
    PAGE_SETTINGS,
    PAGE_TRANSFORMED,
    UIState,
)

# 任务 -> 显示名
TASK_NAMES = {TASK_SHAPE: "形状", TASK_MIN_SQUARE: "最小方块", TASK_DIGIT: "数字"}

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

_PAGE_COLORS = {
    PAGE_ORIGINAL: "#00aa00",
    PAGE_TRANSFORMED: "#0066cc",
    PAGE_SETTINGS: "#cc6600",
}


class MeasurementUI(tk.Tk):
    """单相机单目测距/识别 GUI（薄控制器）。"""

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
        self.cfg_path = cfg_path
        self.camera = camera
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.serial_queue = serial_queue
        self.rectangle_detector = rectangle_detector
        self.shape_detector = shape_detector
        self.min_square_detector = min_square_detector

        self.state = UIState(cfg, frame_size=frame_size, debug=debug)
        self.state.apply_geometry(
            shape_detector, min_square_detector, rectangle_detector
        )
        self.recognizer = Recognizer(
            shape_detector, min_square_detector, digit_detector
        )

        self.selected_digit = None  # None=自动
        self.last_choice_time = 0.0
        self.min_choice_interval = 0.5
        self._running = False

        self._build_ui()

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

        self._show_page_controls(self.state.page)
        self._update_debug_button()

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
        """原始页关联控件：日常要调的——距离补偿、面积筛选。"""
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
        tk.Label(row, text="PnP类型", width=11, anchor=tk.W, font=font).pack(
            side=tk.LEFT, padx=(2, 0)
        )
        self._pnp_var = tk.StringVar(value=self.state.pnp_name)
        self._pnp_combo = ttk.Combobox(
            row, textvariable=self._pnp_var, state="readonly", width=9,
            values=list(PNP_FLAGS.keys()), font=font,
        )
        self._pnp_combo.pack(side=tk.LEFT, padx=2)
        self._pnp_combo.bind("<<ComboboxSelected>>", self._on_pnp_selected)

    def _frame_size_text(self):
        st = self.state
        suffix = "（CS050 5MP 估算）" if getattr(st, "_frame_estimated", False) else ""
        return f"帧分辨率: {st.frame_w} × {st.frame_h}  ({st.frame_area:,}px²){suffix}"

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
                if self.state.frame_area:
                    # 面积用“帧面积百分比”为单位，直接可拖
                    title = "最小面积(%)" if is_min else "最大面积(%)"
                    lo, hi = 0.0, 100.0
                    default = self.state.min_area_pct if is_min else self.state.max_area_pct
                else:
                    # 无帧尺寸（空跑）退化为像素
                    default = self.state.min_area if is_min else self.state.max_area
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
        self._task_var = tk.StringVar(value=self.state.current_task or TASK_SHAPE)
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
        st = self.state
        if key == "camera_offset":
            st.camera_offset = value
        elif key == "rect_width":
            st.rect_width = int(round(value))
        elif key == "rect_height":
            st.rect_height = int(round(value))
        elif key in ("min_area", "max_area"):
            if st.frame_area:
                if key == "min_area":
                    st.min_area_pct = value
                    if st.min_area_pct > st.max_area_pct:
                        st.min_area_pct = st.max_area_pct
                else:
                    st.max_area_pct = value
                    if st.max_area_pct < st.min_area_pct:
                        st.max_area_pct = st.min_area_pct
                st.min_area = int(round(st.min_area_pct / 100.0 * st.frame_area))
                st.max_area = int(round(st.max_area_pct / 100.0 * st.frame_area))
            else:
                if key == "min_area":
                    st.min_area = int(round(value))
                else:
                    st.max_area = int(round(value))
                if st.min_area > st.max_area:
                    st.min_area = st.max_area
        st.apply_geometry(
            self.shape_detector, self.min_square_detector, self.rectangle_detector
        )

    def _on_pnp_selected(self, _event=None):
        self.state.pnp_name = self._pnp_var.get()
        self.state.apply_geometry(
            self.shape_detector, self.min_square_detector, self.rectangle_detector
        )
        self._update_status()

    def _on_task_radio(self):
        self.state.current_task = self._task_var.get()
        self._update_status()

    def _on_digit_selected(self, _event=None):
        val = self._digit_var.get()
        self.selected_digit = None if val == "auto" else val
        self.recognizer.set_selected_digit(self.selected_digit)

    def _set_task(self, task):
        if task == self.state.current_task:
            return
        self.state.current_task = task
        if self._task_var.get() != task:
            self._task_var.set(task)
        self._update_status()

    def _set_page(self, page):
        if page == self.state.page:
            return
        self.state.page = page
        self._update_page_buttons()
        self._show_page_controls(page)

    def _update_page_buttons(self):
        for page, btn in getattr(self, "_page_buttons", {}).items():
            if page == self.state.page:
                btn.config(bg=_PAGE_COLORS.get(page, "#555555"), relief=tk.SUNKEN)
            else:
                btn.config(bg="#555555", relief=tk.RAISED)

    def _toggle_page(self, _event=None):
        page = PAGE_TRANSFORMED if self.state.page == PAGE_ORIGINAL else PAGE_ORIGINAL
        self._set_page(page)

    def _toggle_debug(self, _event=None):
        self.state.debug_mode = not self.state.debug_mode
        self._refresh_task_controls()
        self._update_debug_button()
        self._update_status()

    def _update_debug_button(self):
        if self.state.debug_mode:
            self._debug_btn.config(text="DEBUG:ON", bg="#cc6600", relief=tk.SUNKEN)
        else:
            self._debug_btn.config(text="DEBUG", bg="#555555", relief=tk.RAISED)

    def _refresh_task_controls(self):
        # 无串口或 debug 时，允许 UI 自由选择任务；否则由串口驱动，置灰。
        ui_controlled = self.state.debug_mode or self.serial_queue is None
        state = tk.NORMAL if ui_controlled else tk.DISABLED
        for rb in self._task_radios.values():
            rb.config(state=state)
        self._digit_combo.config(state="readonly" if ui_controlled else tk.DISABLED)

    def _update_status(self):
        try:
            self._set_status("cam", self.camera is not None)
            self._set_status("serial", self.serial_queue is not None)
            task = TASK_NAMES.get(self.state.current_task, "-")
            digit = "" if self.selected_digit is None else self.selected_digit
            self._status_labels["task"].config(text=f"任务: {task}{digit}")
            self._status_labels["pnp"].config(text=f"PnP: {self.state.pnp_name}")
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
            if self.state.debug_mode:
                continue  # debug 模式丢弃串口数据，避免队列无限增长
            try:
                if port_type == "power":
                    text = serial_protocol.parse_power(data)
                    if text:
                        self.state.middle_data_power = text
                        print(f"Power received: {text}")
                elif port_type == "choice":
                    now = time.time()
                    if now - self.last_choice_time < self.min_choice_interval:
                        print(f"Ignoring choice message (too frequent): {data}")
                        continue
                    self.last_choice_time = now
                    print(f"Choice received: {data}")
                    evt = serial_protocol.parse_choice(data)
                    if evt:
                        self._set_task(evt["task"])
                        if evt.get("digit") is not None:
                            self.selected_digit = evt["digit"]
                            self._digit_var.set(self.selected_digit)
                            self.recognizer.set_selected_digit(self.selected_digit)
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
        st = self.state

        for corners in corners_list:
            meta = annotate_rect(
                page1, corners, st.obj_points, self.camera_matrix,
                self.distortion_coeffs, st.pnp_flag, st.camera_offset,
            )
            if warped is None:
                warped = warp_to_plane(frame, corners, st.rect_width, st.rect_height)
                last_meta = meta

        self.recognizer.recognize(warped, st.current_task, last_meta)

        # 在原始帧上叠加面积筛选 A4 模板，便于对比检测框
        draw_filter_template(page1, st.min_area, st.max_area, st.frame_area)

        if st.page in (PAGE_ORIGINAL, PAGE_SETTINGS):
            self._show_frame(self._contain(page1))
        elif st.page == PAGE_TRANSFORMED:
            self._show_frame(self._build_page2())

        self._update_status_dist(last_meta)
        self._update_status()

    def _build_page2(self):
        res = self.recognizer.result()
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

    def _show_empty_page(self, text):
        """无帧时显示占位图；A4 面积模板只在原始/设置页叠加，变换页干净显示。"""
        ph = self._placeholder(text)
        if self.state.page in (PAGE_ORIGINAL, PAGE_SETTINGS):
            draw_filter_template(
                ph, self.state.min_area, self.state.max_area, self.state.frame_area
            )
        self._show_frame(ph)

    # ---------- 保存 / 退出 ----------
    def _save_cfg(self):
        if not self.cfg_path:
            return
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("measurement", {}).update(
                camera_offset_mm=self.state.camera_offset,
                rect_width_mm=self.state.rect_width,
                rect_height_mm=self.state.rect_height,
            )
            cfg.setdefault("ui", {}).update(
                page=self.state.page,
                pnp_type=self.state.pnp_name,
                rect_min_area=self.state.min_area,
                rect_max_area=self.state.max_area,
                debug=self.state.debug_mode,
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
        self.recognizer.stop()
        try:
            self.destroy()
        except Exception:
            pass

    # ---------- 生命周期 ----------
    def run(self):
        self._running = True
        self.recognizer.start()
        self.recognizer.set_selected_digit(self.selected_digit)
        self._update_status()
        self._update_frame()
        try:
            self.mainloop()
        finally:
            self._running = False
            self.recognizer.stop()
            try:
                if self.camera is not None:
                    self.camera.close()
            except Exception:
                pass
