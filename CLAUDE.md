# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概况

2025 年全国大学生电子设计竞赛 C 题「基于单目视觉的目标物测量装置」。在**地平线 RDK X5**（Linux aarch64）上运行，用**海康机器人 CS050 工业相机**采集，完成相机标定、矩形框检测、形状/数字/内切圆测量，并通过串口输出。

- 语言：Python 3；依赖：`opencv-python numpy pyserial pyyaml scipy`（基于 README，另补 `scipy`，见下）。
- 数字识别模块 `modules/find_numSquare.py` 依赖 RDK X5 的 BPU 环境里的 `hobot_dnn`（YOLO11 量化 `.bin`），这部分无法在没有 BPU 的机器上运行。
- 注：`scipy` 在 `modules/detect.py` 与 `modules/find_numSquare.py` 顶部被 import，是运行时依赖，但 README 的依赖列表漏掉了它。

## 常用命令

```bash
# 安装依赖（注意补上 scipy）
pip install opencv-python numpy pyserial pyyaml scipy

# 运行主程序（实时测量）
python3 scripts/main.py          # 或 bash DS_start.sh

# 相机标定（交互：空格采图 / c 标定 / q 退出）
python3 scripts/calibration.py

# 跑全部测试（必须在仓库根目录执行）
python3 -m unittest discover -s tests

# 跑单个测试文件
python3 -m unittest discover -s tests -p "test_config_loader.py"

# 跑单个测试用例（用 -k 按名字过滤）
python3 -m unittest discover -s tests -k test_load_app_config_uses_environment_override
```

- **没有构建/打包系统**：无 `pyproject.toml`、`requirements.txt`、`setup.py`；依赖只写在 README。也没有 lint/格式化配置。
- 测试是标准 `unittest`（非 pytest）。`tests/` 没有 `__init__.py`，所以不能用 `python -m unittest tests.xxx` 按包名调用；需用 `discover -s tests`（`python -m` 会把仓库根目录放进 `sys.path`，使 `tools`/`modules` 可导入）。单个用例用 `-k` 过滤。

## 提交规范

- 遵循 Angular 规范，消息用中文：`<type>(<scope>): <简述>`。
- `type` 取值：`feat` / `fix` / `docs` / `style` / `refactor` / `test` / `chore` / `perf`。
- **禁止**在 commit message 中添加 `Co-Authored-By: Claude` 等 Claude 协作者署名。

## 运行环境与硬件约束

- **目标平台是 Linux（RDK X5）**。相机库按平台选取：`drivers/hikrobot/lib/arm64`（aarch64）、`lib/amd64`（x86_64）。`tools/hikrobot_paths.py` 的 `select_bundled_library()` 在 Windows 上返回 `None`（不选 `.so`），纯 Windows 需要额外安装海康 MVS SDK。
- 环境变量：
  - `HIK_MVS_LIBRARY` — 覆盖海康运行库路径。
  - `DS_CONFIG_PATH` — 覆盖运行配置文件的路径。
- 串口：`configs/app.yaml` 里 `serial.choice`（`/dev/ttyS1`）与 `serial.power`（`/dev/ttyS3`），默认 115200。
- 数字识别模型：`configs/app.yaml` 的 `model.digit_model_path`（默认 `DS_NUM.bin`）。`find_numSquare.YOLO11_Detector` 在模型加载失败时抛 `RuntimeError`。

## 结构（大图）

仓库根目录就是包根，模块之间用根相对导入（`from tools.config_loader import ...`、`from drivers.hikrobot.HIK_CAM import ...`）。**项目没有被安装为 package**，运行入口 `scripts/*` 会先调用 `scripts/_bootstrap.py: configure_paths()` 把仓库根目录、`modules/`、`tools/` 加进 `sys.path`，之后才 import 项目模块；`find_numSquare.py` 也自己做了同样的路径注入。因此单独写脚本时，要「重根目录执行 + 先 configure_paths」，否则 `tools`/`modules`/`drivers` 找不到。

**配置层**：`tools/config_loader.load_app_config()` 读取 `configs/app.yaml`，并把若干项目相对路径（`calibration.result_file`、`calibration.image_output_dir`、`model.digit_model_path`）统一解析为绝对路径。`configs/camera_calibration.yaml` 是 `scripts/calibration.py` 的输出（内参、畸变、棋盘格尺寸、标定图像数），**与具体相机相关**——换相机或改采集区域后要重新标定。

**每帧主循环（`scripts/main.py`）**：
1. `HikIndustrialCamera.get_frame()` 取帧；
2. `modules.detect.RectangleDetector.detect(frame)` 找目标矩形内轮廓的 4 个角点；
3. 对每个矩形：透视变换到已知真实尺寸（`measurement.rect_width_mm/rect_height_mm`），再运行「当前激活检测器」得到某尺寸；同时用 `cv2.solvePnP`（对象为矩形真实三维点，`measurement.camera_offset_mm` 做补偿）算出距离；
4. 检测器：`find_all.ShapeDetector`（形状+真实尺寸）、`modules.find_minSquare`（最小内切方块）、`find_numSquare.YOLO11_Detector`（数字+方块，跑 BPU）；
5. 用哪个检测器由**串口命令**决定：`choice` 串口的 `g...l` 帧，内含 `as`（形状）、`bs`（最小方块）、`d<数字>`（指定数字）；`power` 串口数据被解析转发；窗口按键 `q` 退出、空格切换输出。

**检测器契约**：`main.py` 期望各检测器「调用即返回可计算尺寸的对象 + 标注图」，三者现均已对齐：

- `find_minSquare`：`MinSquareDetector(world_width, world_height).detect()` → `(标注图, size)`，未检测到时 `size=None`。
- `find_all`：`ShapeDetector.detect_shape()` → `(result_dict, 标注图)`，`main.py` 读取 `result['size']`。
- `find_numSquare`：`YOLO11_Detector.process_frame()` → `(标注图, size)`，未检测到时 `size=None`。另外每个检测模块各自带有 `if __name__ == "__main__"` 的可独立运行示例（会直接 `HikIndustrialCamera()` 打开相机），可作为单模块调试入口。

## 测试与验证约束

本项目是**应用型工程**，测试不是默认交付物。除非确有验证价值，否则**不得新增** `tests/`、单元/冒烟测试、fixture、mock 等测试基础设施。

**默认原则**

- 优先用现有生产代码、入口、脚本、launch、仿真或实际运行链路完成验证。
- 修改已有模块时，在现有模块与既有执行路径中验证；不得为验证复制出一套独立测试逻辑。
- 不得以「工程规范」「最佳实践」「覆盖率」「测试完整性」为由主动加测试。
- 小型项目允许没有 `tests/`；无实际测试需求时不得创建该目录。

**仅在满足下列之一时允许新增独立测试**

1. 纯算法 / 数学计算 / 解析器 / 状态机等可独立验证的核心逻辑；
2. 已出现、且需要防止复发的 bug；
3. 现有运行链路无法稳定、快速、确定地验证某个关键行为；
4. 项目已有明确的 CI 自动化测试要求；
5. 用户明确要求增加测试。

**禁止这些测试**：仅为了调用一次生产函数；仅为增加覆盖率；把现有函数复制/提取成 `test_xxx()`；没有明确输入/输出/失败条件的 smoke test；不可独立执行且无自动化价值；与生产代码重复实现同一逻辑；仅验证 getter/setter、参数转发、ROS 2 胶水、简单 callback 或简单封装；为测试方便引入 mock/fake/fixture/interface/factory/DI 等额外架构。

**测试位置**：`tests/` 只放真正需要自动执行的测试。用户可直接运行的实例、调试程序、实验脚本、模块验证入口不得放入 `tests/`，优先复用现有入口，或放到 `scripts/`、`examples/`、`tools/`。

**新增前自问**：现有运行入口能否验证该行为？为何现有模块不能直接验证？该测试具体防止什么错误？失败时能提供什么实际信息？答不清就不建测试。

**生产侧**：测试必须调用现有生产接口，不得复制生产逻辑；不得为可测试性而改变生产架构；测试基础设施不得增加复杂度；测试数量不是工程质量指标。

**验证优先级**（实际运行优先于自动化）：现有程序入口 → 现有脚本/launch → 模块集成 → 仿真/实际硬件 → integration test → unit test。不得默认从最后一项开始。

## 测试覆盖

- `tests/test_config_loader.py` — 配置相对路径解析、`DS_CONFIG_PATH` 覆盖。
- `tests/test_hikrobot_paths.py` — 内置 `.so` 按平台选取（arm64/amd64/Windows）。通过给 `select_bundled_library(repo_root, system_name, machine)` 传参来测。
- `tests/test_path_setup.py` — 用子进程验证 `configure_paths()` 后各模块能被找到（`find_spec`，并非真 import）。

这些测试都不依赖相机或 `hobot_dnn`；但 `tools/config_loader` 用到 `yaml`，裸环境需先 `pip install pyyaml`。
