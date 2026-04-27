import os
from datetime import datetime

import cv2
import numpy as np
import yaml

from _bootstrap import configure_paths

configure_paths()

from drivers.hikrobot.HIK_CAM import HikIndustrialCamera
from tools.config_loader import load_app_config


def apply_crop(frame, crop):
    if not crop:
        return frame
    y1, y2, x1, x2 = [int(value) for value in crop]
    return frame[y1:y2, x1:x2]


def create_camera(camera_config):
    return HikIndustrialCamera(
        exposure_time=camera_config.get("exposure_time"),
        exposure_auto=camera_config.get("exposure_auto", False),
        frame_timeout_ms=camera_config.get("frame_timeout_ms", 1000),
    )


app_config = load_app_config()
camera_config = app_config.get("camera", {})
calibration_config = app_config.get("calibration", {})

CHESSBOARD_SIZE = tuple(calibration_config.get("chessboard_size", [11, 8]))
SQUARE_SIZE = float(calibration_config.get("square_size", 30.0))
MIN_CALIB_IMAGES = int(calibration_config.get("min_images", 15))
CROP = calibration_config.get("crop")

CALIB_IMAGES_DIR = calibration_config.get("image_output_dir", "calibration_images")
CALIB_RESULT_FILE = calibration_config.get("result_file", "configs/camera_calibration.yaml")
os.makedirs(CALIB_IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CALIB_RESULT_FILE), exist_ok=True)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []
img_points = []

camera = create_camera(camera_config)
camera.init()
camera.open()

print("Camera calibration")
print("Controls:")
print("1. Place the chessboard in front of the camera.")
print("2. Press space to capture a detected chessboard frame.")
print("3. Press c to calibrate after enough valid images.")
print("4. Press q to quit.")

image_count = 0
gray = None

while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    frame = apply_crop(frame, CROP)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_chess, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret_chess:
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        frame_display = frame.copy()
        cv2.drawChessboardCorners(frame_display, CHESSBOARD_SIZE, corners_refined, ret_chess)
    else:
        corners_refined = None
        frame_display = frame.copy()

    status_text = f"captured: {image_count} min>={MIN_CALIB_IMAGES}"
    cv2.putText(frame_display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(
        frame_display,
        "'space': capture, 'c': calibration, 'q': quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.namedWindow("Camera Calibration - Press q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Calibration - Press q to quit", 1000, 1000)
    cv2.imshow("Camera Calibration - Press q to quit", frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        if ret_chess:
            image_count += 1
            filename = os.path.join(CALIB_IMAGES_DIR, f"calib_{image_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved image: {filename}")
            obj_points.append(objp)
            img_points.append(corners_refined)
        else:
            print("Chessboard not detected; frame not saved.")
    elif key == ord("c") and len(obj_points) >= MIN_CALIB_IMAGES:
        print("\nStarting camera calibration...")
        break

camera.close()
cv2.destroyAllWindows()

if len(obj_points) < MIN_CALIB_IMAGES:
    print(f"\nNot enough valid calibration images: {len(obj_points)}. Need at least {MIN_CALIB_IMAGES}.")
    raise SystemExit(1)

img_shape = gray.shape[::-1]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

print("\nCalibration result:")
print(f"Reprojection error: {ret:.5f}")
print(f"Camera matrix:\n {mtx}")
print(f"Distortion coefficients: {dist.ravel()}")

mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error
print(f"Mean reprojection error: {mean_error / len(obj_points):.5f} pixels")

calibration_data = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "reprojection_error": float(ret),
    "image_size": list(img_shape),
    "chessboard_size": list(CHESSBOARD_SIZE),
    "square_size": SQUARE_SIZE,
    "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_calibration_images": len(obj_points),
}

with open(CALIB_RESULT_FILE, "w", encoding="utf-8") as f:
    yaml.dump(calibration_data, f, default_flow_style=False)

print(f"\nCalibration result saved to {CALIB_RESULT_FILE}")

camera = create_camera(camera_config)
camera.init()
camera.open()

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

print("\nPress any key to exit preview...")
while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    frame = apply_crop(frame, CROP)
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    combined = np.hstack((frame, dst))

    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(
        combined,
        "Undistorted",
        (img_shape[0] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Original vs Undistorted (Press any key to exit)", combined)
    if cv2.waitKey(1) != -1:
        break

camera.close()
cv2.destroyAllWindows()
