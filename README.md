# DS_ws

25电赛C题的视觉工作空间
在RDKx5上实现，调用其BPU加速模型

## 项目概述

实现了：
1. 海康相机的调用
2. 单目相机的标定&畸变
3. 稳定识别黑色矩形框(A4+2厘米边缘)，引入简化的卡尔曼滤波优化动态效果
4. 针对基础题、拓展题的内容，降框内识别分为三个识别器：
    4.1. 边长识别器：识别边长计算直径（三角形、正方形、圆形）
    4.2. 数字识别器：先识别矩形，再yolo识别数字，计算数字外的正方形边长
    4.3. 内切圆识别器：先识别黑色图形，再画出内切圆，每个内切圆满足切两条边以上，获得最小直径
5. 主函数实现串口通信，切换识别器

## 项目结构
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
│   ├── find_shape.cpython-312.pyc
│   ├── find_shape2.cpython-312.pyc
│   ├── find_shape3.cpython-312.pyc
│   ├── HIK_CAM.cpython-312.pyc
│   ├── new_find_minSquare.cpython-312.pyc
│   └── UKF.cpython-312.pyc
├── calibration_results/
│   └── camera_calibration.yaml
├── DOCS/
│   └── C题_基于单目视觉的目标物测量装置.pdf
└── include/
    └── CameraParams_header.py

## 环境配置
python==3
    openCV==4
    Serial
HIK_SDK