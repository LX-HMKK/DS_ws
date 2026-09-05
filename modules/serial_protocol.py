"""串口帧解析：把下位机发来的 choice / power 数据解析成语义事件。

纯函数，不依赖 Tk 与状态；由 MeasurementUI 负责节流与分发。
"""

from modules.recognizer import TASK_SHAPE, TASK_MIN_SQUARE, TASK_DIGIT


def parse_choice(data):
    """解析 choice 串口的 `g...l` 帧。

    不是 choice 帧或无法解析时返回 None；否则返回 dict：
        {"task": "shape"}  /  {"task": "min_square"}
        {"task": "digit", "digit": "0".."9"}
    """
    if not (data.startswith("g") and data.endswith("l")):
        return None
    mid = data[1:-1]
    if "as" in mid:
        return {"task": TASK_SHAPE}
    if "bs" in mid:
        return {"task": TASK_MIN_SQUARE}
    if "d" in mid:
        ci = mid.find("d")
        if ci != -1 and ci + 1 < len(mid) and mid[ci + 1].isdigit():
            return {"task": TASK_DIGIT, "digit": mid[ci + 1]}
    return None


def parse_power(data):
    """解析 power 串口数据为显示文本；格式不匹配返回 None。"""
    if len(data) >= 6 and data[5] == "A" and data.endswith("W"):
        return "I:" + data[:8] + "P:" + data[8:16] + "PM:" + data[16:]
    return None
