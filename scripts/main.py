from _bootstrap import configure_paths

configure_paths()

import queue
import threading
import time

import cv2
import numpy as np
import yaml

from drivers.hikrobot.HIK_CAM import HikIndustrialCamera
from drivers.send_data import SerialPort
from modules.detect import RectangleDetector
from modules.find_all import ShapeDetector
from modules.find_minSquare import CircleDetector
from modules.find_numSquare import YOLO11_Detector
from tools.config_loader import load_app_config


def read_camera_params(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    camera_matrix = np.array(params["camera_matrix"], dtype=np.float32)
    distortion_coeffs = np.array(params["distortion_coefficients"], dtype=np.float32).flatten()
    return camera_matrix, distortion_coeffs


def get_3d_points(rect_width, rect_height):
    return np.array(
        [
            [0, 0, 0],
            [rect_width, 0, 0],
            [rect_width, rect_height, 0],
            [0, rect_height, 0],
        ],
        dtype=np.float32,
    )


def pnp_distance(camera_matrix, distortion_coeffs, obj_points, img_points):
    if img_points.size == 0:
        return None, None
    img_points = img_points.reshape(-1, 2)
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return translation_vector[2][0], rotation_matrix
    except cv2.error as e:
        print(f"OpenCV error in solvePnP: {e}")
        return None, None


def serial_receiver(serial_port, data_queue, port_name):
    while True:
        try:
            data = serial_port.read_data()
            if data:
                data_queue.put((port_name, data))
        except Exception as e:
            print(f"Error reading from {port_name}: {e}")
        time.sleep(0.01)


def main():
    app_config = load_app_config()
    camera_config = app_config.get("camera", {})
    calibration_config = app_config.get("calibration", {})
    measurement_config = app_config.get("measurement", {})
    model_config = app_config.get("model", {})
    serial_config = app_config.get("serial", {})

    camera_matrix, distortion_coeffs = read_camera_params(
        calibration_config.get("result_file", "configs/camera_calibration.yaml")
    )

    rect_width = int(measurement_config.get("rect_width_mm", 170))
    rect_height = int(measurement_config.get("rect_height_mm", 267))
    camera_length = float(measurement_config.get("camera_offset_mm", -90.0))

    camera = HikIndustrialCamera(
        exposure_time=camera_config.get("exposure_time"),
        exposure_auto=camera_config.get("exposure_auto", False),
        frame_timeout_ms=camera_config.get("frame_timeout_ms", 1000),
    )
    camera.init()
    camera.open()

    obj_points = get_3d_points(rect_width, rect_height)

    detector = RectangleDetector()
    find_all = ShapeDetector(
        area_ratio_threshold=(0.05, 0.95),
        frame_real_width=rect_width,
        frame_real_height=rect_height,
    )
    find_minS = CircleDetector(min_area_ratio=0.05, max_area_ratio=0.95)
    find_numS = YOLO11_Detector(
        model_path=model_config.get("digit_model_path", "DS_NUM.bin"),
        conf_thres=float(model_config.get("conf_thres", 0.30)),
        iou_thres=float(model_config.get("iou_thres", 0.6)),
    )

    choice_serial_config = serial_config.get("choice", {})
    serial_port_choice = SerialPort(
        port=choice_serial_config.get("port", "/dev/ttyS1"),
        baudrate=int(choice_serial_config.get("baudrate", 115200)),
        send_format="str",
        recv_format="str",
    )

    power_serial_config = serial_config.get("power", {})
    serial_port_power = SerialPort(
        port=power_serial_config.get("port", "/dev/ttyS3"),
        baudrate=int(power_serial_config.get("baudrate", 115200)),
        send_format="str",
        recv_format="str",
    )

    serial_queue = queue.Queue()
    threading.Thread(
        target=serial_receiver,
        args=(serial_port_choice, serial_queue, "choice"),
        daemon=True,
    ).start()
    threading.Thread(
        target=serial_receiver,
        args=(serial_port_power, serial_queue, "power"),
        daemon=True,
    ).start()

    current_detector = None
    current_detector_name = None
    middle_data_power = ""
    show_output = False
    last_choice_time = 0
    min_choice_interval = 0.5

    cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined View", 1000, 600)

    try:
        while True:
            while not serial_queue.empty():
                port_type, data = serial_queue.get_nowait()
                if port_type == "power":
                    if len(data) >= 6 and data[5] == "A" and data.endswith("W"):
                        middle_data_power = "I:" + data[:8] + "P:" + data[8:16] + "PM:" + data[16:]
                        print(f"Power received: {middle_data_power}")
                elif port_type == "choice":
                    current_time = time.time()
                    if current_time - last_choice_time < min_choice_interval:
                        print(f"Ignoring choice message (too frequent): {data}")
                        continue

                    last_choice_time = current_time
                    print(f"Choice received: {data}")
                    if data.startswith("g") and data.endswith("l"):
                        middle_data = data[1:-1]
                        if "as" in middle_data:
                            current_detector = find_all.detect_shape
                            current_detector_name = "Shape"
                            show_output = True
                        elif "bs" in middle_data:
                            current_detector = find_minS.find_min_square_side
                            current_detector_name = "Min Square"
                            show_output = True
                        elif "d" in middle_data:
                            c_index = middle_data.find("d")
                            if c_index != -1 and c_index + 1 < len(middle_data):
                                num_char = middle_data[c_index + 1]
                                if num_char.isdigit():
                                    current_detector = find_numS.process_frame
                                    find_numS.set_selected_digit(num_char)
                                    current_detector_name = f"Digit ({num_char})"
                                    show_output = True

            frame = camera.get_frame()
            if frame is None:
                continue

            corners_list = detector.detect(frame)

            if middle_data_power:
                cv2.putText(
                    frame,
                    f"Power: {middle_data_power}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3,
                )

            if current_detector_name:
                cv2.putText(
                    frame,
                    f"Detector: {current_detector_name}",
                    (frame.shape[1] - 700, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 255),
                    4,
                )

            size_info = ""
            output_img = np.zeros((600, 400, 3), dtype=np.uint8)
            has_output = False

            for corners in corners_list:
                dst_points = np.array(
                    [[0, 0], [rect_width, 0], [rect_width, rect_height], [0, rect_height]],
                    dtype=np.float32,
                )
                transform = cv2.getPerspectiveTransform(corners, dst_points)
                warped = cv2.warpPerspective(frame, transform, (rect_width, rect_height))

                if show_output and current_detector:
                    try:
                        if current_detector == find_all.detect_shape:
                            size_val, _ = current_detector(warped)
                            size_info = f"D: {size_val:.1f}mm"
                        elif current_detector == find_minS.find_min_square_side:
                            _, size_min = current_detector(warped)
                            size_info = f"D: {size_min:.1f}mm" if size_min is not None else "D: N/A"
                        elif current_detector == find_numS.process_frame:
                            _, size_digit = current_detector(warped)
                            size_info = (
                                f"D: {size_digit:.1f}mm"
                                if size_digit is not None
                                else "D: Not detected"
                            )
                    except Exception as e:
                        print(f"Error in detector: {e}")
                        size_info = f"Error: {e}"

                if size_info:
                    cv2.putText(
                        warped,
                        size_info,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )

                distance, rotation_matrix = pnp_distance(
                    camera_matrix, distortion_coeffs, obj_points, corners
                )

                for (x, y) in corners:
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)

                if current_detector is not None and distance is not None and rotation_matrix is not None:
                    center = np.mean(corners, axis=0).astype(int)
                    distance_text = f"X: {camera_length + distance:.1f}mm"
                    cv2.putText(
                        frame,
                        distance_text,
                        (center[0], center[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        7,
                    )
                    print(f"Object distance: {distance + camera_length} mm")

                if show_output:
                    if len(warped.shape) == 2:
                        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                    output_img = cv2.resize(warped, (400, 600))
                    has_output = True

            if show_output and not has_output:
                cv2.putText(
                    output_img,
                    "No output available",
                    (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            frame_resized = cv2.resize(frame, (600, 600))
            combined = np.hstack((frame_resized, output_img))
            cv2.imshow("Combined View", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                show_output = not show_output
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
