# MonocularRangefinder-NUEDC2025

基于单目视觉的目标物测量系统，面向 2025 年全国大学生电子设计竞赛 C 题。      
运行平台为地平线 RDK X5 开发板，相机采用海康机器人 CS050 系列工业相机，用于相机标定、矩形框检测、形状/数字/内切圆测量，以及串口通信。

## 目录结构

```text
MonocularRangefinder-NUEDC2025/
├── DS_start.sh
├── README.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
├── DOCS/
│   └── C题_基于单目视觉的目标物测量装置.pdf
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
│   ├── filter_template.py       # 帧上叠加 A4 面积筛选参照矩形
│   ├── find_all.py
│   ├── find_minSquare.py
│   ├── find_numSquare.py
│   ├── geometry.py              # 纯几何/位姿/帧标注函数
│   ├── get_rgb.py
│   ├── measure_ui.py            # Tkinter 界面（薄控制器）
│   ├── recognizer.py            # 识别编排 + 数字后台 worker
│   ├── serial_protocol.py       # 串口帧解析
│   └── ui_state.py              # 界面可变状态与配置同步
├── scripts/
│   ├── _bootstrap.py
│   ├── calibration.py
│   └── main.py
├── tools/
│   ├── config_loader.py
│   └── hikrobot_paths.py
└── tests/
    ├── test_config_loader.py
    └── test_hikrobot_paths.py
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

调试模式（无下位机 / 关串口，识别任务由界面自由选择）：

```bash
python3 scripts/main.py --debug
```

或使用：

```bash
bash DS_start.sh
```

## 界面说明

Tkinter 多页界面，随相机实时刷新：

- **原始(1)**：相机原帧 + 检测到的矩形外框，标注 PnP 解算的距离（含补偿）、YPR 角度（偏离水平/竖直面）以及 A4 比例适宜性。
- **变换(2)**：透视矫正后的内框图 + 识别结果（形状/最小方块/数字 与尺寸，单位 mm）。
- 滑块补偿：距离偏移 `camera_offset_mm`、内框真实宽高 `rect_width_mm`/`rect_height_mm`，以及面积筛选上下限。
- **PnP** 下拉框可选 `ITERATIVE/EPNP/IPPE/DLS/UPNP`。
- **DEBUG** 按钮切换无串口模式：关闭串口收发，识别任务用顶部单选按钮自由选择（数字可在旁边指定 0-9 或自动）。
- `SAVE` 保存当前补偿/尺寸/PnP 类型到 `configs/app.yaml`；`EXIT` 或 `q`/`ESC` 退出。

## 依赖

```bash
python -m pip install --upgrade pip
pip install opencv-python numpy pyserial pyyaml scipy pillow
```

- 界面使用 Tkinter，Linux 需 `apt-get install python3-tk`；`pillow` 用于在 Tk 中显示图像。
- 数字识别依赖 RDK X5 的 BPU 环境中的 `hobot_dnn`，需要在目标设备（RDK X5）上验证；无 BPU 时数字任务显示「模型不可用」，其余识别不受影响。

## 许可证

本项目以 [MIT 许可证](LICENSE) 协议授权发布。
