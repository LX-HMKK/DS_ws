"""识别编排：三个检测器 + 数字后台推理 worker + 结果状态。

把线程 / 队列 / 锁的细节隔离在这里，MeasurementUI 只按帧调用 recognize() 并
轮询 result()。数字识别走后台线程，避免阻塞 Tk 主循环。
"""

import queue
import threading

# 任务标识（串口协议与 UI 均引用）
TASK_SHAPE = "shape"
TASK_MIN_SQUARE = "min_square"
TASK_DIGIT = "digit"


class Recognizer:
    """按当前任务对透视矫正图跑识别，缓存最新结果供 UI 渲染。"""

    def __init__(self, shape_detector, min_square_detector, digit_detector):
        self.shape_detector = shape_detector
        self.min_square_detector = min_square_detector
        self.digit_detector = digit_detector
        self.selected_digit = None  # None=自动

        self._running = False
        self._result_lock = threading.Lock()
        self._latest_result = None     # {'annotated','size','tag'}
        self._digit_queue = queue.Queue(maxsize=1)
        self._digit_worker = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._digit_worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._digit_worker.start()

    def stop(self):
        self._running = False

    def set_selected_digit(self, digit):
        self.selected_digit = digit

    def recognize(self, warped, task, meta=None):
        """按任务跑识别器；数字任务写入后台队列并继续刷新结果。"""
        if warped is None:
            self._set(None, None, "未检测到矩形")
            return
        if task is None:
            self._set(None, None, "等待任务选择")
            return
        try:
            if task == TASK_SHAPE:
                result, annotated = self.shape_detector.detect_shape(warped)
                label = result.get("shape", "unknown")
                raw = result.get("size")
                if raw:
                    size = float(raw)
                    tag = f"形状:{label}  D:{size:.1f}mm"
                else:
                    size = None
                    tag = f"形状:{label}  D:N/A"
                self._set(annotated, size, tag)
            elif task == TASK_MIN_SQUARE:
                annotated, size = self.min_square_detector.detect(warped)
                tag = f"最小方块  D:{size:.1f}mm" if size is not None else "最小方块  D:N/A"
                self._set(annotated, size, tag)
            elif task == TASK_DIGIT:
                self._submit_digit(warped, meta)
                # 立即显示数码推理中，避免在 worker 出结果前残留上个任务的结果
                cur = self.result()
                if cur is None or not cur.get("tag", "").startswith("数字"):
                    self._set(warped, None, "数字推理中...")
        except Exception as e:
            print(f"recognition error: {e}")
            self._set(None, None, f"识别错误:{e}")

    def result(self):
        with self._result_lock:
            return self._latest_result

    # ---------- 内部 ----------
    def _submit_digit(self, warped, meta=None):
        try:
            self._digit_queue.get_nowait()
        except Exception:
            pass
        try:
            self._digit_queue.put_nowait((warped.copy(), meta))
        except queue.Full:
            pass

    def _worker_loop(self):
        while self._running:
            try:
                warped, _meta = self._digit_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                continue
            if self.digit_detector is None:
                self._set(None, None, "数字模型不可用")
                continue
            try:
                # 始终同步当前选择（含 None=自动），避免切换后残留旧数字
                self.digit_detector.set_selected_digit(self.selected_digit)
                annotated, size = self.digit_detector.process_frame(warped)
                tag = (
                    f"数字:{self.selected_digit or 'auto'}  D:{size:.1f}mm"
                    if size is not None else "数字:未检测到"
                )
                self._set(annotated, size, tag)
            except Exception as e:
                print(f"digit worker error: {e}")
                self._set(None, None, f"数字推理错误:{e}")

    def _set(self, annotated, size, tag):
        with self._result_lock:
            self._latest_result = {"annotated": annotated, "size": size, "tag": tag}
