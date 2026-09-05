from _bootstrap import configure_paths

configure_paths()

import argparse
import os
import queue
import threading
import time
from pathlib import Path

from drivers.hikrobot.HIK_CAM import HikIndustrialCamera
from drivers.send_data import SerialPort
from modules.detect import RectangleDetector
from modules.find_all import ShapeDetector
from modules.find_minSquare import MinSquareDetector
from modules.geometry import adjust_camera_matrix_for_crop, read_camera_params
from modules.measure_ui import MeasurementUI
from tools.config_loader import load_app_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"


def serial_receiver(serial_port, data_queue, port_name):
    """串口接收线程：把收到的数据放入队列，由 UI 主循环解析。"""
    while True:
        try:
            data = serial_port.read_data()
            if data:
                data_queue.put((port_name, data))
        except Exception as e:
            print(f"Error reading from {port_name}: {e}")
        time.sleep(0.01)


def build_digit_detector(model_config):
    """懒构造数字识别器。

    find_numSquare 顶层依赖 hobot_dnn（仅 RDK BPU 可用）。在无 BPU 的机器上
    构造失败不影响其它识别器，数字任务将显示「模型不可用」。
    """
    try:
        from modules.find_numSquare import YOLO11_Detector

        return YOLO11_Detector(
            model_path=model_config.get("digit_model_path", "DS_NUM.bin"),
            conf_thres=float(model_config.get("conf_thres", 0.30)),
            iou_thres=float(model_config.get("iou_thres", 0.60)),
        )
    except Exception as e:
        print(f"YOLO 数字模型不可用（可能无 BPU）: {e}")
        return None


def _peek_frame_size(camera):
    """启动时取一帧得到整帧宽度/高度，用于把面积筛选上限与帧挂钩。

    取不到（无相机/无帧）时返回 None，界面退回配置文件里的默认值。
    """
    if camera is None:
        return None
    try:
        for _ in range(30):
            frame = camera.get_frame()
            if frame is not None:
                return (frame.shape[1], frame.shape[0])
            time.sleep(0.02)
    except Exception as e:
        print(f"无法获取帧尺寸: {e}")
    return None


def build_serial_queue(serial_config):
    """构建串口接收队列；失败返回 None（表示进入无串口模式）。"""
    try:
        data_queue = queue.Queue()
        choice_cfg = serial_config.get("choice", {})
        power_cfg = serial_config.get("power", {})
        port_choice = SerialPort(
            port=choice_cfg.get("port", "/dev/ttyS1"),
            baudrate=int(choice_cfg.get("baudrate", 115200)),
            send_format="str",
            recv_format="str",
        )
        port_power = SerialPort(
            port=power_cfg.get("port", "/dev/ttyS3"),
            baudrate=int(power_cfg.get("baudrate", 115200)),
            send_format="str",
            recv_format="str",
        )
        threading.Thread(
            target=serial_receiver,
            args=(port_choice, data_queue, "choice"),
            daemon=True,
        ).start()
        threading.Thread(
            target=serial_receiver,
            args=(port_power, data_queue, "power"),
            daemon=True,
        ).start()
        return data_queue
    except Exception as e:
        print(f"串口初始化失败，转为无串口模式: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="单目测量装置 GUI")
    parser.add_argument(
        "--debug", action="store_true",
        help="无下位机（关串口），识别任务由 UI 自由选择",
    )
    args = parser.parse_args()

    config_path = os.environ.get("DS_CONFIG_PATH")
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    app_config = load_app_config(config_path)

    camera_config = app_config.get("camera", {})
    calibration_config = app_config.get("calibration", {})
    measurement_config = app_config.get("measurement", {})
    model_config = app_config.get("model", {})
    serial_config = app_config.get("serial", {})

    camera_matrix, distortion_coeffs = read_camera_params(
        calibration_config.get("result_file", "configs/camera_calibration.yaml")
    )
    # 内参是在裁剪区域标定的，把主点平移回整帧坐标系，供整帧检测/PnP 使用
    camera_matrix = adjust_camera_matrix_for_crop(
        camera_matrix, calibration_config.get("crop")
    )

    rect_width = int(measurement_config.get("rect_width_mm", 170))
    rect_height = int(measurement_config.get("rect_height_mm", 267))

    camera = HikIndustrialCamera(
        exposure_time=camera_config.get("exposure_time"),
        exposure_auto=camera_config.get("exposure_auto", False),
        frame_timeout_ms=camera_config.get("frame_timeout_ms", 1000),
    )
    try:
        camera.init()
        camera.open()
    except Exception as e:
        print(f"相机初始化失败（空跑模式，仅 UI）：{e}")
        try:
            camera.close()
        except Exception:
            pass
        camera = None

    frame_size = _peek_frame_size(camera)

    rectangle_detector = RectangleDetector()
    shape_detector = ShapeDetector(
        area_ratio_threshold=(0.05, 0.95),
        frame_real_width=rect_width,
        frame_real_height=rect_height,
    )
    min_square_detector = MinSquareDetector(world_width=rect_width, world_height=rect_height)
    digit_detector = build_digit_detector(model_config)

    debug = args.debug or bool(app_config.get("ui", {}).get("debug", False))
    serial_queue = None if debug else build_serial_queue(serial_config)

    ui = MeasurementUI(
        cfg=app_config,
        cfg_path=config_path,
        camera=camera,
        rectangle_detector=rectangle_detector,
        shape_detector=shape_detector,
        min_square_detector=min_square_detector,
        digit_detector=digit_detector,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        serial_queue=serial_queue,
        debug=debug,
        frame_size=frame_size,
    )
    ui.run()


if __name__ == "__main__":
    main()
