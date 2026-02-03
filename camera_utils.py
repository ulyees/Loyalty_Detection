import cv2
from config import MAX_CAMERA_INDEX


def list_available_cameras() -> list[int]:
    """
    跨平台查找所有可用摄像头设备，返回可用摄像头索引列表
    :return: 可用摄像头索引列表（如 [0, 1]）
    """
    available_cameras = []
    print(f"正在查找摄像头设备（遍历索引 0 ~ {MAX_CAMERA_INDEX}）...")

    for index in range(MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            success, _ = cap.read()
            if success:
                available_cameras.append(index)
                print(f"✅ 找到可用摄像头：索引 {index}")
            cap.release()
        cv2.destroyAllWindows()

    if not available_cameras:
        print("❌ 未找到任何可用摄像头设备")
    else:
        print(f"✅ 摄像头查找完成，可用摄像头索引列表：{available_cameras}")

    return available_cameras


def init_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    初始化摄像头并设置分辨率
    :param camera_index: 摄像头索引
    :param width: 宽度
    :param height: 高度
    :return: 初始化后的VideoCapture对象
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception(f"无法打开摄像头（索引：{camera_index}），请检查摄像头是否被占用或索引是否正确")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    except Exception as e:
        print(f"提示：部分设备不支持设置摄像头分辨率，忽略该错误：{e}")

    return cap