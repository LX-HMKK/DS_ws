# DS_ws

基于单目视觉的目标物测量工作空间（25 电赛 C 题实现），在 RDKx5 上可调用其 BPU 加速模型。

## 主要功能
- 调用海康相机采集图像
- 单目相机标定与畸变校正
- 稳定识别黑色矩形框（A4 + 2 cm 边缘），并使用简化卡尔曼滤波提升动态稳定性
- 三种识别器：
  - 边长识别器：根据边长判断并计算直径（支持三角形 / 正方形 / 圆形）
  - 数字识别器：先检测矩形，再用 YOLO/分类识别数字，同时计算数字外的正方形边长
  - 内切圆识别器：识别黑色图形并拟合内切圆，取满足条件的最小直径
- 主程序支持串口通信与识别器切换

## 项目结构

```text
DS_ws/
├── README.md
├── DS_start.sh
├── calibration.py
├── detect.py
├── find_all.py
├── find_minSquare.py
├── find_numSquare.py
├── get_rgb.py
├── HIK_CAM.py
├── main.py
├── send_data.py
├── __pycache__/
│   ├── detect.cpython-312.pyc
│   └── ...
├── calibration_results/
│   └── camera_calibration.yaml
└── include/
    └── CameraParams_header.py
```

## 环境与依赖（建议）
- Python 3.8+
- OpenCV 4.x
- numpy
- pyserial（串口通信）
- HIK SDK（海康相机驱动，需单独安装）
- 使用YoloV11预训练数字识别模型，将pt转onnx再转bin（RDK_BIN专用工具链格式）

示例安装（仅基础库）：
```bash
python -m pip install --upgrade pip
pip install opencv-python numpy pyserial pyyaml
```

备注：HIK_SDK 与相机驱动需按照厂商说明单独安装配置；在 RDKx5 上使用 BPU 时需根据设备提供的 SDK / 接口配置。

## 配置
- 标定文件：calibration_results/camera_calibration.yaml（程序会读取用于去畸变与内参）
- 串口、相机参数在 main.py 或 HIK_CAM.py 中设置，建议先检查并修改对应的端口号与分辨率参数

//
