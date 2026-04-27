import os
import sys
import time
from ctypes import *
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from tools.hikrobot_paths import (
    mvs_sdk_roots,
    select_bundled_library,
    windows_python_import_dirs,
    windows_runtime_dirs,
)


BASE_DIR = Path(__file__).resolve().parent


def _prepend_env_path(env_key, path):
    path = os.fspath(path)
    current = os.environ.get(env_key, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if path not in entries:
        os.environ[env_key] = path + (os.pathsep + current if current else "")


def _prepare_mvs_sdk_paths():
    include_dir = BASE_DIR / "include"
    if os.path.isdir(include_dir) and include_dir not in sys.path:
        sys.path.insert(0, os.fspath(include_dir))

    bundled_library = select_bundled_library(PROJECT_ROOT)
    if bundled_library is not None:
        os.environ.setdefault("HIK_MVS_LIBRARY", os.fspath(bundled_library))
        _prepend_env_path("LD_LIBRARY_PATH", bundled_library.parent)
        _prepend_env_path("PATH", bundled_library.parent)

    sdk_roots = []
    for env_key in ("HIK_MVS_SDK_PATH", "MVS_SDK_PATH", "MVCAM_SDK_PATH"):
        env_val = os.environ.get(env_key)
        if env_val:
            sdk_roots.append(env_val)
    sdk_roots.extend(mvs_sdk_roots())

    for root in sdk_roots:
        for path in windows_python_import_dirs(root):
            path = os.fspath(path)
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)

        if hasattr(os, "add_dll_directory"):
            for path in windows_runtime_dirs(root):
                path = os.fspath(path)
                if os.path.isdir(path):
                    try:
                        os.add_dll_directory(path)
                    except OSError:
                        pass

    explicit_library = os.environ.get("HIK_MVS_LIBRARY")
    if explicit_library and hasattr(os, "add_dll_directory"):
        library_dir = os.path.dirname(os.path.abspath(explicit_library))
        if os.path.isdir(library_dir):
            try:
                os.add_dll_directory(library_dir)
            except OSError:
                pass


_prepare_mvs_sdk_paths()

try:
    from MvCameraControl_class import *
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import MvCameraControl_class. Expected hikrobot/include in this repository "
        "or set HIK_MVS_SDK_PATH to an installed MVS SDK."
    ) from exc


class HikIndustrialCamera:
    def __init__(self, exposure_time=None, exposure_auto=False, frame_timeout_ms=1000):
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.is_opened = False
        self.pData = None
        self.nDataSize = 0
        self.exposure_time = exposure_time
        self.exposure_auto = exposure_auto
        self.frame_timeout_ms = int(frame_timeout_ms)

    def init(self):
        ret = self.cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != 0:
            raise Exception(f"Enum devices failed: 0x{ret:x}")

        if self.device_list.nDeviceNum == 0:
            raise Exception("No camera device found")

        st_device_info = cast(self.device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(st_device_info)
        if ret != 0:
            raise Exception(f"Create camera handle failed: 0x{ret:x}")

    def open(self):
        if self.is_opened:
            return True

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"Open device failed: 0x{ret:x}")

        self._set_enum("TriggerMode", MV_TRIGGER_MODE_OFF)

        exposure_auto_value = (
            MV_EXPOSURE_AUTO_MODE_CONTINUOUS
            if self.exposure_auto
            else MV_EXPOSURE_AUTO_MODE_OFF
        )
        self._set_enum("ExposureAuto", exposure_auto_value)
        if not self.exposure_auto and self.exposure_time is not None:
            self._set_float("ExposureTime", float(self.exposure_time))

        st_param = MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(st_param))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            raise Exception(f"Get PayloadSize failed: 0x{ret:x}")

        self.nDataSize = st_param.nCurValue
        self.pData = (c_ubyte * self.nDataSize)()

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"Start grabbing failed: 0x{ret:x}")

        self.is_opened = True
        return True

    def get_frame(self):
        if not self.is_opened:
            raise Exception("Camera is not opened")

        st_frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(st_frame_info), 0, sizeof(st_frame_info))

        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.pData, self.nDataSize, st_frame_info, self.frame_timeout_ms
        )
        if ret != 0:
            return None

        st_convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(st_convert_param), 0, sizeof(st_convert_param))
        st_convert_param.nWidth = st_frame_info.nWidth
        st_convert_param.nHeight = st_frame_info.nHeight
        st_convert_param.pSrcData = cast(self.pData, POINTER(c_ubyte))
        st_convert_param.nSrcDataLen = st_frame_info.nFrameLen
        st_convert_param.enSrcPixelType = st_frame_info.enPixelType

        if self._is_color(st_frame_info.enPixelType):
            st_convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            n_convert_size = st_frame_info.nWidth * st_frame_info.nHeight * 3
        else:
            st_convert_param.enDstPixelType = PixelType_Gvsp_Mono8
            n_convert_size = st_frame_info.nWidth * st_frame_info.nHeight

        p_dst_buffer = (c_ubyte * n_convert_size)()
        st_convert_param.pDstBuffer = p_dst_buffer
        st_convert_param.nDstBufferSize = n_convert_size

        ret = self.cam.MV_CC_ConvertPixelType(st_convert_param)
        if ret != 0:
            return None

        if self._is_color(st_frame_info.enPixelType):
            frame = np.frombuffer(p_dst_buffer, dtype=np.uint8).reshape(
                (st_frame_info.nHeight, st_frame_info.nWidth, 3)
            )
        else:
            frame = np.frombuffer(p_dst_buffer, dtype=np.uint8).reshape(
                (st_frame_info.nHeight, st_frame_info.nWidth)
            )

        return frame

    def close(self):
        if not self.is_opened:
            return

        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        self.is_opened = False

    def _set_enum(self, node_name, value):
        ret = self.cam.MV_CC_SetEnumValue(node_name, value)
        if ret != 0:
            raise Exception(f"Set camera enum node {node_name} failed: 0x{ret:x}")

    def _set_float(self, node_name, value):
        ret = self.cam.MV_CC_SetFloatValue(node_name, value)
        if ret != 0:
            raise Exception(f"Set camera float node {node_name} failed: 0x{ret:x}")

    def _is_color(self, pixel_type):
        color_types = {
            PixelType_Gvsp_RGB8_Packed,
            PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_YUV422_Packed,
            PixelType_Gvsp_YUV422_YUYV_Packed,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGB10,
            PixelType_Gvsp_BayerGB10_Packed,
            PixelType_Gvsp_BayerBG10,
            PixelType_Gvsp_BayerBG10_Packed,
            PixelType_Gvsp_BayerRG10,
            PixelType_Gvsp_BayerRG10_Packed,
            PixelType_Gvsp_BayerGR10,
            PixelType_Gvsp_BayerGR10_Packed,
            PixelType_Gvsp_BayerGB12,
            PixelType_Gvsp_BayerGB12_Packed,
            PixelType_Gvsp_BayerBG12,
            PixelType_Gvsp_BayerBG12_Packed,
            PixelType_Gvsp_BayerRG12,
            PixelType_Gvsp_BayerRG12_Packed,
            PixelType_Gvsp_BayerGR12,
            PixelType_Gvsp_BayerGR12_Packed,
        }
        return pixel_type in color_types


if __name__ == "__main__":
    from tools.config_loader import load_app_config

    app_config = load_app_config()
    camera_config = app_config.get("camera", {})
    camera = HikIndustrialCamera(
        exposure_time=camera_config.get("exposure_time"),
        exposure_auto=camera_config.get("exposure_auto", False),
        frame_timeout_ms=camera_config.get("frame_timeout_ms", 1000),
    )
    try:
        camera.init()
        camera.open()
        print("Camera started. Press q to quit.")

        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.namedWindow("Hik Industrial Camera", cv2.WINDOW_NORMAL)
                cv2.imshow("Hik Industrial Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.close()
        cv2.destroyAllWindows()
