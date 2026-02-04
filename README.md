# 🤖 忠诚度检测系统

## 项目简介

基于MediaPipe的智能双手举过头顶检测系统，具备高精度姿态识别和现代化用户界面。

### 核心特性
- ✨ 实时人体姿态检测
- 👐 双手举过头顶识别
- 🖐 手掌张开状态判断
- 🎵 音频反馈提示
- 🎨 现代化GUI界面
- 🔧 跨平台兼容

## 系统要求

### 硬件要求
- 摄像头设备
- 支持OpenGL 3.0+的显卡（推荐）
- 至少2GB可用内存

### 软件要求
- Python 3.8+
- 以下操作系统之一：
  - Windows 10/11
  - macOS 10.14+
  - Ubuntu 18.04+

## 安装指南

### 1. 克隆项目
```bash
git clone https://github.com/your-username/loyalty-detection.git
cd loyalty-detection
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 依赖包列表
```
mediapipe==0.10.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.0
comtypes==1.1.14
```

## 快速开始

### 命令行版本
```bash
python main.py
```

### 图形界面版本
```bash
python gui_app.py
```

## 使用说明

### 1. 摄像头设置
- 系统会自动搜索可用摄像头
- 支持多摄像头切换
- 默认分辨率：640x480

### 2. 检测姿势要求
- 双手完全举过头顶
- 手腕位置高于鼻子
- 手掌完全张开（至少4指伸直）
- 双手同时可见

### 3. 操作控制
- **开始检测**：点击"开始检测"按钮
- **停止检测**：点击"停止检测"按钮或按ESC键
- **退出程序**：关闭窗口或按ESC键

## 项目结构

```
loyalty-detection/
├── audio_player.py          # 音频播放模块
├── camera_utils.py          # 摄像头工具
├── config.py                # 配置文件
├── pose_hand_detector.py    # 姿态检测核心
├── gui_app.py              # 图形界面应用
├── main.py                 # 命令行入口
└── requirements.txt        # 依赖列表
```

## 配置选项

### 主要配置参数
```python
# 摄像头设置
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 检测灵敏度
POSE_DETECTION_CONFIDENCE = 0.8
HAND_DETECTION_CONFIDENCE = 0.8

# 稳定帧数（防抖动）
STABLE_FRAME_COUNT = 15

# 音频反馈
ENABLE_AUDIO = True
```

## 故障排除

### 常见问题

**Q: 摄像头无法打开**
- 检查摄像头是否被其他程序占用
- 尝试不同的摄像头索引
- 确保摄像头驱动程序正常

**Q: 检测精度不佳**
- 确保光照充足
- 调整摄像头角度
- 检查背景是否过于复杂

**Q: 音频无法播放**
- 检查系统音量设置
- 确认音频文件存在
- 查看系统音频驱动

### 调试模式
启用调试日志：
```python
DEBUG_MODE_LOG = True
```

## 开发指南

### 代码规范
- 使用类型注解
- 遵循PEP8规范
- 添加适当的文档字符串

### 扩展功能
1. 添加新的检测姿势
2. 集成更多媒体格式支持
3. 开发网络传输功能

## 版本历史

### v2.0 (当前版本)
- 现代化GUI界面
- 增强的检测算法
- 跨平台音频支持
- 性能优化

### v1.0
- 基础检测功能
- 命令行界面
- 基本配置系统

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目！

### 开发流程
1. Fork本项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 技术支持

如有问题请联系：
- 📧 Email: 511700825@qq.com

## 致谢

感谢以下开源项目：
- https://mediapipe.dev/ - Google的跨平台ML解决方案
- https://opencv.org/ - 计算机视觉库
- https://docs.python.org/3/library/tkinter.html - Python GUI工具包

---

**注意**: 本项目仅供学习和研究使用。请遵守当地法律法规，尊重用户隐私。