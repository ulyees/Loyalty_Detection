import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import platform
import sys
import time
import os
import logging
from typing import Optional, Tuple

# 导入原有模块
from camera_utils import list_available_cameras, init_camera
from pose_hand_detector import HandUpHeadWithOpenDetector
from config import CAMERA_WIDTH, CAMERA_HEIGHT

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernPoseDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 忠诚度检测")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        self.root.configure(bg="#ffffff")

        # 设置现代化字体
        self._setup_modern_fonts()

        # 核心变量
        self.available_cameras = []
        self.detector: Optional[HandUpHeadWithOpenDetector] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_detection_result = False
        self.exit_requested = False

        # 现代化样式配置
        self._setup_modern_style()

        # 构建现代化UI
        self._create_modern_widgets()

        # 初始化摄像头搜索
        self.root.after(100, self._search_cameras)

        # 安全退出处理
        self.root.protocol("WM_DELETE_WINDOW", self._safe_exit)
        self.root.bind('<Escape>', lambda e: self._safe_exit())

        # 焦点设置
        self.root.focus_force()

    def _setup_modern_fonts(self):
        """设置现代化字体"""
        system = platform.system()
        if system == "Windows":
            self.title_font = ("微软雅黑", 16, "bold")
            self.heading_font = ("微软雅黑", 12, "bold")
            self.body_font = ("微软雅黑", 10)
            self.small_font = ("微软雅黑", 9)
        elif system == "Darwin":
            self.title_font = ("PingFang SC", 16, "bold")
            self.heading_font = ("PingFang SC", 12, "bold")
            self.body_font = ("PingFang SC", 11)
            self.small_font = ("PingFang SC", 10)
        else:  # Linux
            self.title_font = ("DejaVu Sans", 16, "bold")
            self.heading_font = ("DejaVu Sans", 12, "bold")
            self.body_font = ("DejaVu Sans", 10)
            self.small_font = ("DejaVu Sans", 9)

    def _setup_modern_style(self):
        """配置现代化白色系样式"""
        self.style = ttk.Style()

        # 尝试使用现代化主题
        try:
            if platform.system() == "Windows":
                self.style.theme_use("vista")
            elif platform.system() == "Darwin":
                self.style.theme_use("aqua")
            else:
                self.style.theme_use("clam")
        except:
            self.style.theme_use("clam")

        # 现代化颜色方案
        self.colors = {
            "primary": "#2563eb",
            "success": "#10b981",
            "danger": "#ef4444",
            "warning": "#f59e0b",
            "background": "#ffffff",
            "surface": "#f8fafc",
            "border": "#e2e8f0",
            "text_primary": "#1e293b",
            "text_secondary": "#64748b"
        }

        # 配置样式
        self.style.configure("Modern.TFrame", background=self.colors["background"])
        self.style.configure("Card.TFrame",
                             background=self.colors["surface"],
                             relief="flat",
                             borderwidth=1)

        # 按钮样式
        self.style.configure("Primary.TButton",
                             font=self.body_font,
                             padding=(20, 10),
                             background=self.colors["primary"],
                             foreground="white",
                             borderwidth=0,
                             focuscolor="none")
        self.style.map("Primary.TButton",
                       background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")])

        self.style.configure("Success.TButton",
                             font=self.body_font,
                             padding=(20, 10),
                             background=self.colors["success"],
                             foreground="white",
                             borderwidth=0)
        self.style.map("Success.TButton",
                       background=[("active", "#059669"), ("disabled", "#a7f3d0")])

        self.style.configure("Danger.TButton",
                             font=self.body_font,
                             padding=(20, 10),
                             background=self.colors["danger"],
                             foreground="white",
                             borderwidth=0)
        self.style.map("Danger.TButton",
                       background=[("active", "#dc2626"), ("disabled", "#fca5a5")])

    def _create_modern_widgets(self):
        """创建现代化布局（修复所有布局问题）"""
        # 主容器 - 使用grid布局确保稳定性
        main_container = ttk.Frame(self.root, style="Modern.TFrame", padding=0)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 配置网格权重，确保布局稳定
        main_container.grid_rowconfigure(1, weight=1)  # 视频区域可扩展
        main_container.grid_rowconfigure(3, weight=0)  # 日志区域固定高度
        main_container.grid_columnconfigure(0, weight=1)

        # 1. 标题区域（第0行）
        title_frame = self._create_title_section(main_container)
        title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        # 2. 控制面板区域（第1行）
        control_frame = self._create_control_section(main_container)
        control_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        # 3. 状态和信息区域（第2行）
        status_frame = self._create_status_section(main_container)
        status_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

        # 4. 日志区域（第3行）- 固定高度，支持滚动
        log_frame = self._create_log_section(main_container)
        log_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))

    def _create_title_section(self, parent):
        """创建标题区域"""
        frame = ttk.Frame(parent, style="Modern.TFrame")

        # 标题
        title_label = tk.Label(frame,
                               text="🤖 智能姿势检测系统",
                               font=self.title_font,
                               bg=self.colors["background"],
                               fg=self.colors["text_primary"])
        title_label.pack(side=tk.LEFT)

        # 版本信息
        version_label = tk.Label(frame,
                                 text="v2.0",
                                 font=self.small_font,
                                 bg=self.colors["background"],
                                 fg=self.colors["text_secondary"])
        version_label.pack(side=tk.RIGHT)

        return frame

    def _create_control_section(self, parent):
        """创建控制面板区域（修复布局挤压问题）"""
        container = ttk.Frame(parent, style="Modern.TFrame")

        # 左右分栏布局
        container.grid_columnconfigure(0, weight=2, minsize=600)  # 视频区域权重更大
        container.grid_columnconfigure(1, weight=1, minsize=300)  # 控制面板固定宽度
        container.grid_rowconfigure(0, weight=1)

        # 左侧：视频显示区域
        video_frame = self._create_video_section(container)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # 右侧：控制面板
        control_panel = self._create_control_panel(container)
        control_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        return container

    def _create_video_section(self, parent):
        """创建视频显示区域（修复尺寸问题）"""
        frame = ttk.LabelFrame(parent,
                               text="📹 实时视频画面",
                               padding=10,
                               style="Card.TFrame")

        # 视频显示容器 - 固定最小尺寸，避免挤压
        video_container = ttk.Frame(frame, style="Card.TFrame", relief="sunken", borderwidth=2)
        video_container.pack(fill=tk.BOTH, expand=True)

        # 设置固定最小尺寸
        video_container.pack_propagate(False)  # 禁止自动调整大小
        video_container.configure(width=640, height=480)

        self.video_label = tk.Label(video_container,
                                    bg="#000000",
                                    relief="flat")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        return frame

    def _create_control_panel(self, parent):
        """创建右侧控制面板"""
        frame = ttk.LabelFrame(parent,
                               text="⚙️ 控制面板",
                               padding=15,
                               style="Card.TFrame")

        # 摄像头选择区域
        camera_frame = ttk.Frame(frame, style="Card.TFrame")
        camera_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(camera_frame,
                 text="摄像头选择:",
                 font=self.body_font,
                 bg=self.colors["surface"]).pack(anchor="w", pady=(0, 5))

        # 摄像头选择行
        camera_select_frame = ttk.Frame(camera_frame, style="Card.TFrame")
        camera_select_frame.pack(fill=tk.X, pady=5)

        self.search_btn = ttk.Button(camera_select_frame,
                                     text="🔍 搜索设备",
                                     style="Primary.TButton",
                                     command=self._search_cameras)
        self.search_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.camera_var = tk.StringVar()
        self.camera_combobox = ttk.Combobox(camera_select_frame,
                                            textvariable=self.camera_var,
                                            state="readonly",
                                            width=12,
                                            font=self.body_font)
        self.camera_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 控制按钮区域
        btn_frame = ttk.Frame(frame, style="Card.TFrame")
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame,
                                    text="▶️ 开始检测",
                                    style="Success.TButton",
                                    command=self._start_detection)
        self.start_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_btn = ttk.Button(btn_frame,
                                   text="⏹️ 停止检测",
                                   style="Danger.TButton",
                                   command=self._stop_detection,
                                   state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)

        # 状态信息区域
        status_info_frame = ttk.LabelFrame(frame,
                                           text="📊 检测状态",
                                           padding=10,
                                           style="Card.TFrame")
        status_info_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="系统就绪")
        status_label = tk.Label(status_info_frame,
                                textvariable=self.status_var,
                                font=self.body_font,
                                bg=self.colors["surface"],
                                fg=self.colors["text_primary"],
                                wraplength=250,
                                justify=tk.LEFT)
        status_label.pack(anchor="w", fill=tk.X)

        self.result_var = tk.StringVar(value="等待检测开始...")
        result_label = tk.Label(status_info_frame,
                                textvariable=self.result_var,
                                font=self.heading_font,
                                bg=self.colors["surface"],
                                fg=self.colors["text_secondary"])
        result_label.pack(anchor="w", pady=(5, 0))

        return frame

    def _create_status_section(self, parent):
        """创建状态信息区域"""
        frame = ttk.Frame(parent, style="Modern.TFrame")

        # 使用网格布局确保稳定性
        for i in range(3):
            frame.grid_columnconfigure(i, weight=1)

        # 系统状态卡片
        status_card = self._create_status_card(frame, "系统状态", "🟢 运行正常")
        status_card.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        # 摄像头状态卡片
        camera_card = self._create_status_card(frame, "摄像头状态", "未连接")
        camera_card.grid(row=0, column=1, sticky="ew", padx=5)

        # 检测结果卡片
        result_card = self._create_status_card(frame, "检测结果", "未开始")
        result_card.grid(row=0, column=2, sticky="ew", padx=(10, 0))

        return frame

    def _create_status_card(self, parent, title, value):
        """创建状态卡片"""
        frame = ttk.Frame(parent, style="Card.TFrame", relief="raised", borderwidth=1)

        # 标题
        title_label = tk.Label(frame,
                               text=title,
                               font=self.small_font,
                               bg=self.colors["surface"],
                               fg=self.colors["text_secondary"])
        title_label.pack(pady=(8, 2))

        # 值
        value_label = tk.Label(frame,
                               text=value,
                               font=self.heading_font,
                               bg=self.colors["surface"],
                               fg=self.colors["text_primary"])
        value_label.pack(pady=(2, 8))

        # 存储引用以便更新
        if title == "系统状态":
            self.system_status_var = value_label
        elif title == "摄像头状态":
            self.camera_status_var = value_label
        elif title == "检测结果":
            self.detection_status_var = value_label

        return frame

    def _create_log_section(self, parent):
        """创建日志区域（支持滚动，固定高度）"""
        frame = ttk.LabelFrame(parent,
                               text="📝 系统日志",
                               padding=10,
                               style="Card.TFrame")

        # 固定高度，避免挤压其他元素
        frame.pack_propagate(False)
        frame.configure(height=120)

        # 创建带滚动条的文本框
        log_container = ttk.Frame(frame, style="Card.TFrame")
        log_container.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_container,
                                                  font=self.small_font,
                                                  bg=self.colors["surface"],
                                                  fg=self.colors["text_primary"],
                                                  wrap=tk.WORD,
                                                  borderwidth=0,
                                                  relief="flat")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        return frame

    def _log(self, message: str, level: str = "INFO"):
        """现代化日志记录"""

        def add_log():
            if self.exit_requested:
                return

            try:
                self.log_text.config(state=tk.NORMAL)

                # 添加时间戳和等级
                timestamp = time.strftime("%H:%M:%S")
                level_color = {
                    "INFO": "#2563eb",
                    "SUCCESS": "#10b981",
                    "ERROR": "#ef4444",
                    "WARNING": "#f59e0b"
                }.get(level, "#64748b")

                # 插入带格式的日志
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, f"{level}: ", level.lower())
                self.log_text.insert(tk.END, f"{message}\n")

                # 滚动到底部
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)

            except Exception as e:
                print(f"日志记录错误: {e}")

        if self.root and not self.exit_requested:
            self.root.after(0, add_log)

    def _search_cameras(self):
        """搜索摄像头设备"""

        def search_task():
            try:
                self.search_btn.config(state=tk.DISABLED)
                self._log("开始搜索摄像头设备...", "INFO")
                self.status_var.set("正在搜索摄像头...")

                cameras = list_available_cameras()
                self.available_cameras = cameras

                if cameras:
                    camera_list = [str(cam) for cam in cameras]
                    self.camera_combobox['values'] = camera_list
                    if camera_list:
                        self.camera_combobox.current(0)

                    self._log(f"找到 {len(cameras)} 个可用摄像头", "SUCCESS")
                    self.status_var.set(f"找到 {len(cameras)} 个摄像头")
                    self.camera_status_var.config(text=f"{len(cameras)}个可用")
                else:
                    self._log("未找到可用摄像头", "WARNING")
                    self.status_var.set("未找到摄像头")
                    self.camera_status_var.config(text="未连接")

            except Exception as e:
                self._log(f"摄像头搜索失败: {str(e)}", "ERROR")
                self.status_var.set("搜索失败")
            finally:
                if not self.exit_requested:
                    self.search_btn.config(state=tk.NORMAL)

        threading.Thread(target=search_task, daemon=True).start()

    def _start_detection(self):
        """启动检测"""
        if not self.available_cameras:
            messagebox.showwarning("警告", "请先搜索并选择摄像头")
            return

        if self.is_running:
            messagebox.showinfo("提示", "检测已在运行中")
            return

        try:
            camera_index = int(self.camera_var.get())
        except:
            messagebox.showerror("错误", "请选择有效的摄像头")
            return

        # 更新UI状态
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.search_btn.config(state=tk.DISABLED)
        self.camera_combobox.config(state="disabled")

        def start_task():
            try:
                self._log(f"正在启动摄像头 {camera_index}...", "INFO")
                self.status_var.set("正在启动摄像头...")
                self.camera_status_var.config(text="连接中...")

                # 初始化检测器
                self.detector = HandUpHeadWithOpenDetector(camera_index=camera_index)
                self.is_running = True

                self._log("摄像头启动成功，开始检测", "SUCCESS")
                self.status_var.set("检测运行中")
                self.camera_status_var.config(text="已连接")
                self.detection_status_var.config(text="检测中")

                # 启动检测线程
                self.camera_thread = threading.Thread(target=self._detection_loop, daemon=True)
                self.camera_thread.start()

            except Exception as e:
                self._log(f"启动失败: {str(e)}", "ERROR")
                self.status_var.set("启动失败")
                self._reset_ui_state()

        threading.Thread(target=start_task, daemon=True).start()

    def _stop_detection(self):
        """停止检测"""
        if not self.is_running:
            return

        self.is_running = False
        self._log("正在停止检测...", "INFO")
        self.status_var.set("正在停止...")

        def stop_task():
            # 等待检测线程结束
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=3.0)

            # 释放资源
            if self.detector:
                try:
                    self.detector.release()
                except Exception as e:
                    self._log(f"资源释放警告: {e}", "WARNING")
                finally:
                    self.detector = None

            # 更新UI
            self._reset_ui_state()
            self._log("检测已停止", "INFO")
            self.status_var.set("检测已停止")
            self.camera_status_var.config(text="已断开")
            self.detection_status_var.config(text="已停止")

        threading.Thread(target=stop_task, daemon=True).start()

    def _reset_ui_state(self):
        """重置UI状态"""
        if not self.exit_requested:
            self.root.after(0, lambda: [
                self.start_btn.config(state=tk.NORMAL),
                self.stop_btn.config(state=tk.DISABLED),
                self.search_btn.config(state=tk.NORMAL),
                self.camera_combobox.config(state="readonly"),
                self.video_label.config(image=""),
                setattr(self.video_label, 'image', None)
            ])

    def _detection_loop(self):
        """检测循环"""
        last_update = 0
        update_interval = 1.0 / 30  # 30 FPS

        while self.is_running and self.detector:
            try:
                current_time = time.time()
                if current_time - last_update < update_interval:
                    time.sleep(0.01)
                    continue

                success, frame = self.detector.cap.read()
                if not success:
                    time.sleep(0.1)
                    continue

                # 处理帧
                result, processed_frame = self.detector.process_frame(frame)

                # 更新检测结果
                if result:
                    self.detection_status_var.config(text="检测成功", fg="#10b981")
                    self.result_var.set("✅ 姿势正确")
                else:
                    self.detection_status_var.config(text="检测中", fg="#64748b")
                    self.result_var.set("❌ 请举双手过头顶")

                # 更新视频帧
                self._update_video_display(processed_frame)
                last_update = current_time

            except Exception as e:
                if self.is_running:  # 只在运行状态下记录错误
                    self._log(f"检测错误: {str(e)}", "ERROR")
                time.sleep(0.1)

        # 循环结束后的清理
        self.is_running = False

    def _update_video_display(self, frame):
        """更新视频显示（去除不必要的动画）"""
        if self.exit_requested or frame is None:
            return

        def update():
            try:
                # 直接缩放，不使用动画效果
                h, w = frame.shape[:2]
                target_width = 640
                target_height = 480

                # 保持宽高比缩放
                scale = min(target_width / w, target_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                if new_w > 0 and new_h > 0:
                    resized = cv2.resize(frame, (new_w, new_h))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb)
                    photo = ImageTk.PhotoImage(image=pil_image)

                    self.video_label.config(image=photo)
                    self.video_label.image = photo

            except Exception as e:
                if not self.exit_requested:
                    self._log(f"显示更新失败: {e}", "ERROR")

        if not self.exit_requested:
            self.root.after(0, update)

    def _safe_exit(self):
        """安全退出"""
        if self.exit_requested:
            return

        self.exit_requested = True
        self._log("系统正在安全退出...", "INFO")

        # 停止检测
        self.is_running = False

        # 禁用所有控件
        self._disable_all_controls()

        # 释放资源
        def cleanup():
            try:
                if self.detector:
                    self.detector.release()
                cv2.destroyAllWindows()
            except:
                pass

            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

            os._exit(0)

        self.root.after(100, cleanup)

    def _disable_all_controls(self):
        """禁用所有控件"""
        for widget in self.root.winfo_children():
            try:
                if hasattr(widget, 'config'):
                    widget.config(state=tk.DISABLED)
            except:
                pass


def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = ModernPoseDetectionGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        messagebox.showerror("错误", f"程序启动失败: {e}")
    finally:
        try:
            os._exit(0)
        except:
            pass


if __name__ == "__main__":
    main()