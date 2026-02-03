import os
import platform
import threading
import time
import logging
import subprocess
from typing import Optional
import sys

logger = logging.getLogger(__name__)


class AudioPlayer:
    """跨平台音频播放器 - 修复路径问题版"""

    def __init__(self, audio_file: str = None):
        # 修复：使用智能路径检测
        self.audio_file = self._get_audio_file_path(audio_file)
        self.is_playing = False
        self.system = platform.system().lower()
        self._play_lock = threading.Lock()
        self._current_process: Optional[subprocess.Popen] = None
        self._last_play_time = 0
        self._min_play_interval = 3.0  # 最小播放间隔3秒

        # 检查音频文件是否存在
        if not os.path.exists(self.audio_file):
            logger.warning(f"⚠️ 音频文件不存在: {self.audio_file}，将使用系统提示音")
            # 尝试创建目录和默认音频文件
            self._create_default_audio_file()
        else:
            logger.info(f"✅ 找到音频文件: {self.audio_file}")

        # 检查系统兼容性
        self._check_system_compatibility()

    def _get_audio_file_path(self, audio_file: Optional[str]) -> str:
        """智能获取音频文件路径（跨平台兼容）"""
        # 如果提供了音频文件路径，直接使用
        if audio_file:
            return audio_file

        # 否则自动检测合适的路径
        possible_paths = self._get_possible_audio_paths()

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"🎵 自动发现音频文件: {path}")
                return path

        # 如果都没找到，返回最可能的一个路径
        default_path = possible_paths[0]
        logger.info(f"📁 使用默认音频路径: {default_path}")
        return default_path

    def _get_possible_audio_paths(self) -> list:
        """获取所有可能的音频文件路径（跨平台）"""
        # 获取当前脚本所在目录
        if getattr(sys, 'frozen', False):
            # 打包后的可执行文件
            base_dir = os.path.dirname(sys.executable)
        else:
            # 开发环境
            base_dir = os.path.dirname(os.path.abspath(__file__))

        possible_paths = []

        # 不同层级的可能路径
        search_dirs = [
            base_dir,  # 当前文件所在目录
            os.path.join(base_dir, "Music"),  # 当前目录下的Music文件夹
            os.path.join(os.path.dirname(base_dir), "Music"),  # 上级目录的Music文件夹
            os.getcwd(),  # 当前工作目录
            os.path.join(os.getcwd(), "Music"),  # 工作目录下的Music文件夹
        ]

        # 去重
        search_dirs = list(dict.fromkeys(search_dirs))

        # 为每个目录生成可能的音频文件路径
        for search_dir in search_dirs:
            possible_paths.extend([
                os.path.join(search_dir, "music.mp3"),
                os.path.join(search_dir, "Music", "music.mp3"),
                os.path.join(search_dir, "audio", "music.mp3"),
                os.path.join(search_dir, "sounds", "music.mp3"),
            ])

        # 去重并返回
        return list(dict.fromkeys(possible_paths))

    def _create_default_audio_file(self):
        """创建默认音频目录结构"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.audio_file), exist_ok=True)
            logger.info(f"📁 创建音频目录: {os.path.dirname(self.audio_file)}")

            # 可以在这里添加创建默认音频文件的逻辑
            # 目前只是创建目录，音频文件需要用户自己放置
            logger.info(f"💡 请将music.mp3文件放置在: {os.path.dirname(self.audio_file)}")

        except Exception as e:
            logger.warning(f"创建音频目录失败: {e}")

    def _check_system_compatibility(self):
        """检查系统兼容性"""
        logger.info(f"🎵 音频播放器初始化 - 系统: {self.system}")

        # 检查必要的命令行工具
        if self.system == "darwin":
            self._check_command("afplay")
        elif self.system == "linux":
            self._check_command("aplay")
        elif self.system == "windows":
            self._check_command("powershell")

    def _check_command(self, cmd: str):
        """检查命令行工具是否可用"""
        try:
            if self.system == "windows":
                result = subprocess.run(["where", cmd], capture_output=True, text=True, timeout=2)
            else:
                result = subprocess.run(["which", cmd], capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                logger.debug(f"✅ 命令行工具可用: {cmd}")
            else:
                logger.warning(f"⚠️ 命令行工具不可用: {cmd}")
        except:
            logger.warning(f"⚠️ 检查命令行工具失败: {cmd}")

    def play_audio(self, audio_file: Optional[str] = None):
        """播放音频 - 主要接口"""
        if self.is_playing:
            logger.debug("⏳ 音频正在播放中，跳过")
            return

        # 确定要播放的音频文件
        target_file = audio_file or self.audio_file

        def play_thread():
            self.is_playing = True
            try:
                current_time = time.time()
                if current_time - self._last_play_time < self._min_play_interval:
                    logger.debug("⏳ 音频播放过于频繁，跳过")
                    return

                self._last_play_time = current_time
                self._safe_play_audio(target_file)

            except Exception as e:
                logger.error(f"❌ 音频播放失败: {e}")
            finally:
                self.is_playing = False

        # 在新线程中播放
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
        logger.debug(f"🎵 开始播放音频: {target_file}")

    def _safe_play_audio(self, audio_file: str):
        """安全播放音频 - 跨平台兼容"""
        try:
            if os.path.exists(audio_file):
                logger.info(f"🔊 播放音频文件: {audio_file}")
                self._play_system_specific(audio_file)
            else:
                logger.warning(f"⚠️ 音频文件不存在: {audio_file}，使用系统提示音")
                self._play_system_beep()

        except Exception as e:
            logger.error(f"❌ 音频播放异常: {e}")
            self._play_system_beep()

    def _play_system_specific(self, audio_file: str):
        """系统特定的音频播放"""
        try:
            if self.system == "windows":
                self._play_windows(audio_file)
            elif self.system == "darwin":
                self._play_darwin(audio_file)
            else:  # linux
                self._play_linux(audio_file)
        except Exception as e:
            logger.error(f"系统音频播放失败: {e}")
            self._play_system_beep()

    def _play_windows(self, audio_file: str):
        """Windows系统音频播放"""
        try:
            # 方法1: 使用系统默认播放器
            os.startfile(audio_file)
        except:
            try:
                # 方法2: 使用Windows Media Player COM组件
                import comtypes
                import comtypes.client

                comtypes.CoInitialize()
                wmp = comtypes.client.CreateObject("WMPlayer.OCX")
                media = wmp.newMedia(audio_file)
                wmp.currentPlaylist.appendItem(media)
                wmp.controls.play()

                # 非阻塞播放
                def cleanup():
                    time.sleep(5)
                    try:
                        wmp.controls.stop()
                        wmp.close()
                        comtypes.CoUninitialize()
                    except:
                        pass

                threading.Thread(target=cleanup, daemon=True).start()
            except:
                # 方法3: 使用系统提示音
                import winsound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

    def _play_darwin(self, audio_file: str):
        """macOS系统音频播放"""
        try:
            subprocess.Popen(["afplay", audio_file],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except:
            # 备用方案
            subprocess.Popen(["osascript", "-e", "beep"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

    def _play_linux(self, audio_file: str):
        """Linux系统音频播放"""
        try:
            # 尝试多种播放器
            players = ["paplay", "aplay", "mpg123", "mpg321", "play"]
            for player in players:
                try:
                    result = subprocess.run(["which", player], capture_output=True, timeout=1)
                    if result.returncode == 0:
                        subprocess.Popen([player, audio_file],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                        return
                except:
                    continue

            # 如果没有找到播放器，使用系统蜂鸣
            print("\a")
        except:
            print("\a")

    def _play_system_beep(self):
        """播放系统蜂鸣声（最终备用方案）"""
        try:
            if self.system == "windows":
                import winsound
                winsound.Beep(1000, 200)  # 频率1000Hz，持续时间200ms
            else:
                print("\a")  # 终端蜂鸣
        except:
            print("\a")  # 终极备用方案

    def play_success_sound(self):
        """播放成功音效 - 兼容旧接口"""
        self.play_audio()

    def stop_audio(self):
        """停止当前音频播放"""
        try:
            if self._current_process and self._current_process.poll() is None:
                self._current_process.terminate()
                self._current_process.wait(timeout=1)
        except:
            pass
        finally:
            self.is_playing = False

    def cleanup(self):
        """清理资源"""
        self.stop_audio()
        logger.debug("✅ 音频播放器资源已清理")