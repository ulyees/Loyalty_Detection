import cv2
import mediapipe as mp
import platform
import os
import sys
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import logging
import math
import audio_player

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    POSE_DETECTION_CONFIDENCE, POSE_TRACKING_CONFIDENCE,
    HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE,
    STABLE_FRAME_COUNT,
    COLOR_KEYPOINT, COLOR_CONNECTION, COLOR_SUCCESS, COLOR_NORMAL, COLOR_HAND,
    FONT_SCALE, FONT_THICKNESS, FONT_SIZE,
    HAND_Y_TOLERANCE, FINGER_OPEN_THRESHOLD,
    ENABLE_AUDIO,
    DEBUG_MODE_LOG
)
from camera_utils import init_camera

if ENABLE_AUDIO:
    player = audio_player.AudioPlayer()

class HandUpHeadWithOpenDetector:
    """
    🤖 高精度双手举过头顶检测器 - 可调试版
    使用MediaPipe库进行人体姿态和手部关键点检测
    """

    def __init__(self, camera_index: int = 0, debug_mode: bool = True):
        """
        初始化检测器
        :param camera_index: 摄像头索引
        :param debug_mode: 调试模式，显示详细检测信息
        """
        self.debug_mode = debug_mode
        self.camera_index = camera_index

        # MediaPipe库初始化
        self._init_mediapipe()

        # 摄像头初始化
        self._init_camera()

        # 状态变量
        self._init_state_variables()

        # 字体初始化
        self.font = self._init_font()

        logger.info(f"🎯 检测器初始化完成 (摄像头: {camera_index}, 调试模式: {debug_mode})")
        if ENABLE_AUDIO:
            self.audio_player = audio_player.AudioPlayer()
            self.audio_played = False
            self.last_audio_time = 0

            logger.info(f"音频初始化完成")

    def _init_mediapipe(self):
        """初始化MediaPipe组件"""
        # MediaPipe绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # MediaPipe解决方案
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        # 姿态检测器配置
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # 视频流模式
            model_complexity=2,  # 高精度模型
            smooth_landmarks=True,  # 平滑关键点
            enable_segmentation=False,  # 不需要分割
            smooth_segmentation=True,  # 平滑分割
            min_detection_confidence=POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE
        )

        # 手部检测器配置
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # 视频流模式
            max_num_hands=2,  # 最多检测2只手
            model_complexity=1,  # 平衡精度和性能
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )

        if self.debug_mode:
            logger.info("✅ MediaPipe组件初始化完成")

    def _init_camera(self):
        """初始化摄像头"""
        self.cap = init_camera(self.camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头索引 {self.camera_index}")

        if self.debug_mode:
            # 测试摄像头
            ret, frame = self.cap.read()
            if ret:
                logger.info(f"✅ 摄像头测试成功 - 分辨率: {frame.shape[1]}x{frame.shape[0]}")
            else:
                logger.warning("⚠️ 摄像头读取测试失败")
            self.cap.release()
            # 重新初始化
            self.cap = init_camera(self.camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)

    def _init_state_variables(self):
        """初始化状态变量"""
        self.stable_count = 0
        self.final_result = False
        self.is_running = False
        self.frame_count = 0

        # 检测详情记录
        self.detection_details = {
            "hands_above_head": False,
            "left_hand_open": False,
            "right_hand_open": False,
            "both_hands_detected": False,
            "all_conditions_met": False
        }

        # 调试数据
        self.debug_data = {
            "last_pose_time": 0,
            "last_hand_time": 0,
            "detection_stats": {"success": 0, "total": 0}
        }

    def _init_font(self):
        """初始化中文字体"""
        try:
            system = platform.system()
            if system == "Windows":
                return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", FONT_SIZE)
            elif system == "Darwin":
                return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", FONT_SIZE)
            else:
                return ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        处理单帧图像 - 核心检测逻辑
        :param frame: 输入图像帧
        :return: (检测结果, 处理后的图像帧)
        """
        self.frame_count += 1
        original_frame = frame.copy()

        # 1. 图像预处理
        frame = cv2.flip(frame, 1)  # 镜像翻转
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # 2. MediaPipe检测
        pose_results, hand_results = self._run_mediapipe_detection(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # 3. 绘制检测结果
        frame = self._draw_detection_results(frame, pose_results, hand_results)

        # 4. 核心逻辑判断
        current_result = self._analyze_detection_results(pose_results, hand_results)

        # 5. 显示调试信息
        frame = self._draw_debug_info(frame, current_result)

        return current_result, frame

    def _run_mediapipe_detection(self, rgb_frame: np.ndarray):
        """执行MediaPipe检测"""
        pose_results, hand_results = None, None

        try:
            # 姿态检测
            pose_start = time.time()
            pose_results = self.pose.process(rgb_frame)
            pose_time = time.time() - pose_start

            # 手部检测
            hand_start = time.time()
            hand_results = self.hands.process(rgb_frame)
            hand_time = time.time() - hand_start

            if self.debug_mode and self.frame_count % 30 == 0:  # 每30帧输出一次性能数据
                logger.debug(f"⏱️ 检测耗时 - 姿态: {pose_time * 1000:.1f}ms, 手部: {hand_time * 1000:.1f}ms")

        except Exception as e:
            logger.error(f"MediaPipe检测错误: {e}")

        return pose_results, hand_results

    def _draw_detection_results(self, frame: np.ndarray, pose_results, hand_results):
        """绘制检测结果到图像"""
        try:
            # 绘制姿态关键点
            if pose_results and pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=COLOR_KEYPOINT, thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=COLOR_CONNECTION, thickness=2
                    )
                )

            # 绘制手部关键点
            if hand_results and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=COLOR_HAND, thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=COLOR_HAND, thickness=2
                        )
                    )

        except Exception as e:
            logger.error(f"绘制检测结果错误: {e}")

        return frame

    def _analyze_detection_results(self, pose_results, hand_results) -> bool:
        """分析检测结果并判断是否满足条件"""
        # 重置检测详情
        self.detection_details = {key: False for key in self.detection_details}

        has_pose = pose_results and pose_results.pose_landmarks
        has_hands = hand_results and hand_results.multi_hand_landmarks
        hand_count = len(hand_results.multi_hand_landmarks) if has_hands else 0

        if self.debug_mode:
            logger.debug(f"🔍 检测统计 - 姿态: {has_pose}, 手部: {hand_count}只手")

        # 基本条件检查
        if not has_pose or hand_count < 2:
            self._update_stable_count(False)
            return False

        try:
            # 1. 检查双手是否举过头顶
            hands_above_head = self._check_hands_above_head(pose_results.pose_landmarks.landmark)
            self.detection_details["hands_above_head"] = hands_above_head

            # 2. 识别左右手并检查是否张开
            left_hand, right_hand = self._identify_hands(hand_results.multi_hand_landmarks,
                                                         pose_results.pose_landmarks.landmark)
            both_hands_detected = left_hand is not None and right_hand is not None
            self.detection_details["both_hands_detected"] = both_hands_detected

            left_hand_open = self._check_hand_open(left_hand) if left_hand else False
            right_hand_open = self._check_hand_open(right_hand) if right_hand else False
            left_hand_open = not left_hand_open
            right_hand_open = not right_hand_open
            self.detection_details["left_hand_open"] = left_hand_open
            self.detection_details["right_hand_open"] = right_hand_open

            # 3. 最终判断：所有条件必须同时满足
            all_conditions_met = (
                    hands_above_head and
                    both_hands_detected and
                    left_hand_open and
                    right_hand_open
            )
            self.detection_details["all_conditions_met"] = all_conditions_met

            if self.debug_mode and all_conditions_met:
                if ENABLE_AUDIO:
                    current_time = time.time()
                    if not self.audio_played or (current_time - self.last_audio_time > 3.0):
                        self.audio_player.play_success_sound()  # 使用成员变量
                        self.audio_played = True
                        self.last_audio_time = current_time
                        logger.info("🎉 所有检测条件满足！音频已播放")
                logger.info("🎉 所有检测条件满足！")

            # 更新稳定计数
            self._update_stable_count(all_conditions_met)

            return self.final_result

        except Exception as e:
            logger.error(f"检测分析错误: {e}")
            self._update_stable_count(False)
            return False

    def _check_hands_above_head(self, pose_landmarks) -> bool:
        """检查双手是否举过头顶"""
        try:
            # MediaPipe姿态关键点索引
            LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LEFT_WRIST = self.mp_pose.PoseLandmark.LEFT_WRIST.value
            RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST.value
            LEFT_ELBOW = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
            RIGHT_ELBOW = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
            NOSE = self.mp_pose.PoseLandmark.NOSE.value

            landmarks = pose_landmarks

            # 条件1: 手腕在鼻子以上
            wrist_above_nose = (
                    landmarks[LEFT_WRIST].y < landmarks[NOSE].y and
                    landmarks[RIGHT_WRIST].y < landmarks[NOSE].y
            )

            # 条件2: 手腕在肩膀以上
            wrist_above_shoulder = (
                    landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y and
                    landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y
            )

            # 条件3: 肘部在肩膀以上（确保手臂举起）
            elbow_above_shoulder = (
                    landmarks[LEFT_ELBOW].y < landmarks[LEFT_SHOULDER].y and
            landmarks[RIGHT_ELBOW].y < landmarks[RIGHT_SHOULDER].y
            )

            # 条件4: 可见度检查
            visibility_ok = (
                    landmarks[LEFT_WRIST].visibility > 0.5 and
                    landmarks[RIGHT_WRIST].visibility > 0.5 and
                    landmarks[LEFT_SHOULDER].visibility > 0.5 and
                    landmarks[RIGHT_SHOULDER].visibility > 0.5
            )

            result = wrist_above_nose and wrist_above_shoulder and elbow_above_shoulder and visibility_ok

            if self.debug_mode:
                logger.debug(
                    f"🙌 举过头顶检查: {result} (手腕在鼻上: {wrist_above_nose}, 手腕在肩上: {wrist_above_shoulder}, 肘在肩上: {elbow_above_shoulder}, 可见度: {visibility_ok})")

            return result

        except Exception as e:
            logger.error(f"举过头顶检查错误: {e}")
            return False

    def _identify_hands(self, hand_landmarks_list, pose_landmarks):
        """识别左右手"""
        try:
            if len(hand_landmarks_list) != 2:
                return None, None

            # 使用鼻子位置作为参考点
            NOSE = self.mp_pose.PoseLandmark.NOSE.value
            nose_x = pose_landmarks[NOSE].x

            left_hand = None
            right_hand = None

            for hand_landmarks in hand_landmarks_list:
                if hand_landmarks and len(hand_landmarks.landmark) > 0:
                    wrist_x = hand_landmarks.landmark[0].x
                    if wrist_x < nose_x:
                        left_hand = hand_landmarks
                    else:
                        right_hand = hand_landmarks

            if self.debug_mode:
                logger.debug(f"👐 左右手识别: 左手{'✅' if left_hand else '❌'}, 右手{'✅' if right_hand else '❌'}")

            return left_hand, right_hand

        except Exception as e:
            logger.error(f"左右手识别错误: {e}")
            return None, None

    def _check_hand_open(self, hand_landmarks) -> bool:
        """检查手部是否张开"""
        try:
            if not hand_landmarks or len(hand_landmarks.landmark) < 21:
                return False

            landmarks = hand_landmarks.landmark

            # MediaPipe手部关键点索引
            WRIST = 0
            THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
            THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 3, 6, 10, 14, 18
            THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17

            def calculate_finger_extension(tip_idx, pip_idx, mcp_idx):
                """计算手指伸直程度"""
                tip = landmarks[tip_idx]
                pip = landmarks[pip_idx]
                mcp = landmarks[mcp_idx]

                # 计算向量
                vec1 = np.array([pip.x - mcp.x, pip.y - mcp.y])
                vec2 = np.array([tip.x - pip.x, tip.y - pip.y])

                # 计算夹角
                if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                    return 180

                dot_product = np.dot(vec1, vec2)
                cos_angle = dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                return angle

            # 检查每个手指的伸直角度
            thumb_angle = calculate_finger_extension(THUMB_TIP, THUMB_IP, THUMB_MCP)
            index_angle = calculate_finger_extension(INDEX_TIP, INDEX_PIP, INDEX_MCP)
            middle_angle = calculate_finger_extension(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
            ring_angle = calculate_finger_extension(RING_TIP, RING_PIP, RING_MCP)
            pinky_angle = calculate_finger_extension(PINKY_TIP, PINKY_PIP, PINKY_MCP)

            # 角度阈值：大于160度认为伸直
            angles = [thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle]
            straight_fingers = [angle > 160 for angle in angles]
            straight_count = sum(straight_fingers)

            # 至少4个手指伸直认为手掌张开
            result = straight_count >= 4

            if self.debug_mode and result:
                logger.debug(f"🖐️ 手部张开检查: 伸直{straight_count}/5指 (角度: {[f'{a:.1f}°' for a in angles]})")

            return result

        except Exception as e:
            logger.error(f"手部张开检查错误: {e}")
            return False

    def _update_stable_count(self, current_condition: bool):
        """更新稳定帧计数"""
        if current_condition:
            self.stable_count = min(self.stable_count + 1, STABLE_FRAME_COUNT + 10)
            if self.stable_count >= STABLE_FRAME_COUNT:
                self.final_result = True
                self.debug_data["detection_stats"]["success"] += 1
        else:
            self.stable_count = max(self.stable_count - 2, 0)
            if self.stable_count <= STABLE_FRAME_COUNT // 3:
                self.final_result = False

        self.debug_data["detection_stats"]["total"] += 1
        self.debug_data["success_rate"] = (
                self.debug_data["detection_stats"]["success"] /
                max(self.debug_data["detection_stats"]["total"], 1) * 100
        )

    def _draw_debug_info(self, frame: np.ndarray, current_result: bool) -> np.ndarray:
        """在图像上绘制调试信息"""
        try:
            # 使用PIL绘制中文文本
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # 调试信息显示
            if DEBUG_MODE_LOG:
                debug_info = [
                    f"帧号: {self.frame_count}",
                    f"检测结果: {'成功' if current_result else '失败'}",
                    f"稳定帧: {self.stable_count}/{STABLE_FRAME_COUNT}",
                    f"成功率: {self.debug_data['success_rate']:.1f}%",
                    "",
                    "=== 详细检测结果 ===",
                    f"举过头顶: {'是' if self.detection_details['hands_above_head'] else '否'}",
                    f"左手张开: {'是' if self.detection_details['left_hand_open'] else '否'}",
                    f"右手张开: {'是' if self.detection_details['right_hand_open'] else '否'}",
                    f"双手检测: {'是' if self.detection_details['both_hands_detected'] else '否'}",
                    f"所有条件: {'是' if self.detection_details['all_conditions_met'] else '否'}"
                ]
            else:
                debug_info = []

            y_offset = 30
            for i, line in enumerate(debug_info):
                color = (0, 255, 0) if "✅" in line else (255, 255, 255)
                if "成功" in line and current_result:
                    color = (0, 255, 0)  # 绿色
                elif "失败" in line and not current_result:
                    color = (0, 0, 255)  # 红色

                if self.font:
                    draw.text((10, y_offset + i * 25), line, font=self.font, fill=color)
                else:
                    draw.text((10, y_offset + i * 25), line, fill=color)

            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"绘制调试信息错误: {e}")
            # 备用方案：使用OpenCV绘制
            y_offset = 30
            for i, line in enumerate(debug_info):
                color = (0, 255, 0) if "✅" in line else (255, 255, 255)
                cv2.putText(frame, line, (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return frame

    def start_detection(self):
        """开始检测循环"""
        if not self.cap or not self.cap.isOpened():
            logger.error("❌ 摄像头未就绪")
            return

        self.is_running = True
        logger.info("🚀 开始检测循环...")

        print("\n" + "=" * 60)
        print("🎯 高精度双手举过头顶检测系统 - 已启动")
        print("📋 检测条件:")
        print("  1. 双手完全举过头顶（手腕在鼻子以上）")
        print("  2. 双手手掌完全张开（至少4指伸直）")
        print("  3. 双手同时被检测到")
        print("⚡ 调试模式: 已启用 - 查看控制台输出获取详细检测信息")
        print("🎮 操作: 按Q或ESC退出检测")
        print("=" * 60 + "\n")

        try:
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ 读取帧失败")
                    time.sleep(0.1)
                    continue

                # 处理帧
                result, processed_frame = self.process_frame(frame)

                # 显示结果
                cv2.imshow("🤖 智能姿势检测系统 (按Q退出)", processed_frame)

                # 按键检测
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                    break

        except KeyboardInterrupt:
            logger.info("检测被用户中断")
        except Exception as e:
            logger.error(f"检测循环错误: {e}")
        finally:
            self.release()

    def release(self):
        """释放资源"""
        self.is_running = False
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'pose') and self.pose:
                self.pose.close()
            if hasattr(self, 'hands') and self.hands:
                self.hands.close()
        except Exception as e:
            logger.error(f"资源释放错误: {e}")
        finally:
            logger.info("✅ 资源释放完成")

        # 输出最终统计信息
        if self.debug_mode:
            total = self.debug_data["detection_stats"]["total"]
            success = self.debug_data["detection_stats"]["success"]
            rate = self.debug_data["success_rate"]
            logger.info(f"📊 检测统计: 总帧数{total}, 成功{success}, 成功率{rate:.1f}%")

    def __del__(self):
        """析构函数"""
        self.release()


# 调试函数
def debug_mediapipe_detection():
    """MediaPipe检测调试函数"""
    print("🎯 MediaPipe库调试信息")
    print("=" * 50)

    # 检查MediaPipe版本
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe版本: {mp.__version__}")
    except ImportError:
        print("❌ MediaPipe未安装，请运行: pip install mediapipe")
        return

    # 检查依赖库
    libraries = ['cv2', 'numpy', 'PIL']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✅ {lib} 可用")
        except ImportError:
            print(f"❌ {lib} 未安装")

    print("=" * 50)


if __name__ == "__main__":
    debug_mediapipe_detection()

    try:
        detector = HandUpHeadWithOpenDetector(camera_index=0, debug_mode=True)
        detector.start_detection()
    except Exception as e:
        logger.error(f"测试运行失败: {e}")