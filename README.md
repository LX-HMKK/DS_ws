# MonoMeasure-NUEDC2025

基于单目视觉的目标物测量系统，面向 2025 年全国大学生电子设计竞赛 C 题。      
用于海康工业相机采集、相机标定、矩形框检测、形状/数字/内切圆测量，以及串口通信。

## 目录结构

```text
DS_ws/
├── DS_start.sh
├── README.md
├── configs/
│   ├── app.yaml
│   └── camera_calibration.yaml
├── drivers/
│   ├── send_data.py
│   └── hikrobot/
│       ├── HIK_CAM.py
│       ├── include/
│       └── lib/
│           ├── amd64/libMvCameraControl.so
│           └── arm64/libMvCameraControl.so
├── modules/
│   ├── detect.py
│   ├── find_all.py
│   ├── find_minSquare.py
│   ├── find_numSquare.py
│   └── get_rgb.py
├── scripts/
│   ├── _bootstrap.py
│   ├── calibration.py
│   └── main.py
├── tools/
│   ├── config_loader.py
│   └── hikrobot_paths.py
└── tests/
```

## 配置

- 运行配置：`configs/app.yaml`
- 标定参数：`configs/camera_calibration.yaml`
- Linux 海康运行库：`drivers/hikrobot/lib/<arch>/libMvCameraControl.so`
- 如需覆盖海康库路径，可设置 `HIK_MVS_LIBRARY`
- 如需覆盖运行配置文件，可设置 `DS_CONFIG_PATH`

## 启动

```bash
python3 scripts/main.py
```

或使用：

```bash
bash DS_start.sh
```

## 依赖

```bash
python -m pip install --upgrade pip
pip install opencv-python numpy pyserial pyyaml
```

数字识别依赖 RDK/BPU 环境中的 `hobot_dnn`，需要在目标设备上验证。
