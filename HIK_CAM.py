from os import name
import sys
import time
import cv2
import numpy as np
from ctypes import *
sys.path.append("your SDK path/MvImport")  # 修改为你的SDK路径
from MvCameraControl_class import *

class HikIndustrialCamera:
    def __init__(self):
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.is_opened = False
        self.pData = None
        self.nDataSize = 0

    def init(self):
        """初始化相机，枚举设备并准备资源"""
        # 枚举设备
        ret = self.cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != 0:
            raise Exception(f"枚举设备失败! 错误码: 0x{ret:x}")
        
        if self.device_list.nDeviceNum == 0:
            raise Exception("未发现设备")

        # 创建句柄
        st_device_info = cast(self.device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(st_device_info)
        if ret != 0:
            raise Exception(f"创建句柄失败! 错误码: 0x{ret:x}")

    def open(self):
        """打开相机并配置参数"""
        if self.is_opened:
            return True

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"打开设备失败! 错误码: 0x{ret:x}")

        # 设置触发模式为关闭（连续取流）
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise Exception(f"设置触发模式失败! 错误码: 0x{ret:x}")

        # 分配缓冲区
        st_param = MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(st_param))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            raise Exception(f"获取PayloadSize失败! 错误码: 0x{ret:x}")
        
        self.nDataSize = st_param.nCurValue
        self.pData = (c_ubyte * self.nDataSize)()

        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"开始取流失败! 错误码: 0x{ret:x}")

        self.is_opened = True
        return True

    def get_frame(self):
        """获取一帧图像并转换为OpenCV格式"""
        if not self.is_opened:
            raise Exception("相机未打开")

        st_frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(st_frame_info), 0, sizeof(st_frame_info))

        # 获取图像
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            byref(self.pData), self.nDataSize, st_frame_info, 1000
        )
        if ret != 0:
            return None

        # 转换像素格式
        st_convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(st_convert_param), 0, sizeof(st_convert_param))
        st_convert_param.nWidth = st_frame_info.nWidth
        st_convert_param.nHeight = st_frame_info.nHeight
        st_convert_param.pSrcData = cast(self.pData, POINTER(c_ubyte))
        st_convert_param.nSrcDataLen = st_frame_info.nFrameLen
        st_convert_param.enSrcPixelType = st_frame_info.enPixelType

        # 判断图像类型
        if self._is_color(st_frame_info.enPixelType):
            st_convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            n_convert_size = st_frame_info.nWidth * st_frame_info.nHeight * 3
        else:
            st_convert_param.enDstPixelType = PixelType_Gvsp_Mono8
            n_convert_size = st_frame_info.nWidth * st_frame_info.nHeight

        # 执行转换
        p_dst_buffer = (c_ubyte * n_convert_size)()
        st_convert_param.pDstBuffer = p_dst_buffer
        st_convert_param.nDstBufferSize = n_convert_size

        ret = self.cam.MV_CC_ConvertPixelType(st_convert_param)
        if ret != 0:
            return None

        # 转换为numpy数组
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
        """关闭相机并释放资源"""
        if not self.is_opened:
            return

        # 停止取流
        self.cam.MV_CC_StopGrabbing()
        # 关闭设备
        self.cam.MV_CC_CloseDevice()
        # 销毁句柄
        self.cam.MV_CC_DestroyHandle()
        self.is_opened = False

    def _is_color(self, pixel_type):
        """判断像素类型是否为彩色"""
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
            PixelType_Gvsp_BayerGR12_Packed
        }
        return pixel_type in color_types



if __name__ == "__main__":

    
    camera = HikIndustrialCamera()
    try:
        camera.init()
        camera.open()
        print("相机启动成功，按q退出...")
        
        while True:
            frame = camera.get_frame()
            # frame = frame[664:1384,864:1584]
            # frame = frame[124:1924,324:2124]
            if frame is not None:
                cv2.namedWindow("Hik Industrial Camera",cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Hik Industrial Camera",1000,1000)
                cv2.imshow("Hik Industrial Camera", frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        camera.close()
        cv2.destroyAllWindows()