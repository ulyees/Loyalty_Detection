from camera_utils import list_available_cameras
from pose_hand_detector import HandUpHeadWithOpenDetector


def main():
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("❌ 无可用摄像头，程序退出")
        return

    default_index = available_cameras[0]
    try:
        user_input = input(
            f"\n请输入要使用的摄像头索引（默认：{default_index}，可用列表：{available_cameras}）："
        ).strip()
        selected_index = int(user_input) if user_input else default_index

        # 验证索引有效性
        if selected_index not in available_cameras:
            print(f"⚠️  索引 {selected_index} 不可用，使用默认索引 {default_index}")
            selected_index = default_index
    except ValueError:
        print(f"⚠️  输入无效，使用默认索引 {default_index}")
        selected_index = default_index

    try:
        detector = HandUpHeadWithOpenDetector(camera_index=selected_index)
        detector.start_detection()
    except Exception as e:
        print(f"❌ 检测器初始化失败：{e}")


if __name__ == "__main__":
    main()