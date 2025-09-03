# TriModalFusion Web Demo User Guide

![TriModalFusion Logo](https://img.shields.io/badge/TriModalFusion-Web%20Demo-blue.svg)

## Features

### Real-time Multimodal Detection
- **Camera Real-time Detection**: Real-time analysis of camera video streams
- **Audio Real-time Collection**: Real-time processing and visualization of microphone audio
- **Gesture Real-time Recognition**: MediaPipe-based gesture keypoint detection
- **Multimodal Fusion**: Real-time fusion analysis of three modal data types

### Visualization Interface
- **Real-time Dashboard**: Performance metrics including FPS, latency, and confidence
- **Prediction Results Display**: Classification results, Top-K predictions, and confidence visualization
- **Feature Visualization**: Real-time visualization of speech, image, and gesture features
- **Attention Heatmaps**: Cross-modal attention weight visualization
- **Detection History**: Recording and viewing of historical detection results

### Interactive Features
- **File Upload Detection**: Support for audio, image, and gesture file uploads
- **Parameter Adjustment**: Adjustment of detection interval, confidence threshold, and other parameters
- **Settings Persistence**: Local storage of user preference settings
- **Responsive Design**: Support for desktop and mobile devices

## Quick Start

### 1. Environment Check
```bash
# Check if environment is ready
python start_web_demo.py --check-only
```

### 2. Start Service
```bash
# Basic startup (local access)
python start_web_demo.py

# Allow external access
python start_web_demo.py --host 0.0.0.0

# Custom port
python start_web_demo.py --port 8080

# Development mode (hot reload)
python start_web_demo.py --dev
```

### 3. Access Interface
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## System Requirements

### Hardware Requirements
- **Camera**: WebRTC-compatible camera
- **Microphone**: Audio input device
- **GPU**: NVIDIA GPU recommended (optional, for accelerated inference)
- **Memory**: At least 4GB RAM

### Software Dependencies
```bash
# Core dependencies
pip install fastapi uvicorn jinja2 python-multipart websockets pillow

# Optional dependencies (for full functionality)
pip install torch torchvision torchaudio  # PyTorch
pip install opencv-python mediapipe       # Computer Vision
pip install librosa soundfile            # Audio Processing
```

### Browser Support
- Chrome 70+
- Firefox 65+
- Safari 12+
- Edge 79+

## Interface Usage Guide

### Main Interface Layout

```
┌─────────────────┬─────────────────┐
│   Real-time Input │   Detection Results │
├─────────────────┼─────────────────┤
│ Camera Video    │ Prediction Results │
│ Audio Visualization │ Confidence Display │
│ Gesture Detection │ Top-K Predictions │
│ File Upload     │ Modal Status    │
├─────────────────┼─────────────────┤
│   System Status │ Feature Visualization │
├─────────────────┼─────────────────┤
│ FPS Display     │ Speech Features │
│ Latency Monitor │ Image Features  │
│ Confidence Stats │ Gesture Features │
│ Prediction Count │ Attention Weights │
└─────────────────┴─────────────────┘
```

### Operation Steps

#### 1. Real-time Detection
1. Click the **"Start Detection"** button
2. Grant browser access to camera and microphone
3. Wait for model loading to complete
4. View real-time detection results

#### 2. File Upload Detection
1. Select files in the file upload area:
   - **Audio Files**: `.wav`, `.mp3`, `.flac`
   - **Image Files**: `.jpg`, `.png`, `.bmp`
   - **Gesture Data**: `.json`, `.csv`
2. Click the **"Upload Detection"** button
3. Wait for processing completion and view results

#### 3. Parameter Adjustment
1. Click the settings button to open configuration panel
2. Adjust the following parameters:
   - **Detection Interval**: 100-5000 milliseconds
   - **Confidence Threshold**: 0.1-1.0
   - **Display Options**: Feature visualization, attention weights
3. Click **"Save Settings"** to apply changes

## Advanced Features

### WebSocket实时通信
```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/detection');

// 发送检测请求
ws.send(JSON.stringify({
    type: 'detection_request',
    data: {
        timestamp: Date.now(),
        image: base64ImageData,
        audio: audioFrequencyData,
        gesture: keypointData
    }
}));

// 接收检测结果
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.type === 'detection_response') {
        handlePredictionResult(result);
    }
};
```

### API接口
```bash
# 健康检查
curl http://localhost:8000/health

# 文件上传检测
curl -X POST http://localhost:8000/predict/files \
  -F "speech_file=@audio.wav" \
  -F "image_file=@image.jpg" \
  -F "request={\"return_features\": true}"

# 模型信息
curl http://localhost:8000/model/info

# 性能基准测试
curl http://localhost:8000/model/benchmark?num_runs=50
```

## 📊 性能监控

### 实时指标
- **FPS**: 每秒帧数，反映处理速度
- **延迟**: 从输入到输出的时间延迟
- **置信度**: 当前预测的置信度分数
- **预测数**: 累计预测次数

### 系统状态
- **🟢 已连接**: WebSocket连接正常，模型可用
- **🟡 连接中**: 正在建立连接或加载模型
- **🔴 连接断开**: WebSocket断开或模型不可用
- **🟣 错误**: 发生系统错误

## Troubleshooting

### Common Issues

#### 1. Camera/Microphone Access Issues
```
解决方案:
- 检查浏览器权限设置
- 确保设备未被其他程序占用
- 使用HTTPS协议 (部分浏览器要求)
```

#### 2. WebSocket Connection Failure
```
解决方案:
- 确认服务器正在运行
- 检查防火墙设置
- 验证端口未被占用
```

#### 3. Model Loading Failure
```
解决方案:
- 确认模型文件路径正确
- 检查配置文件格式
- 验证GPU/CUDA环境 (如使用GPU)
```

#### 4. Abnormal Detection Results
```
解决方案:
- 调整置信度阈值
- 检查输入数据质量
- 重新训练模型 (如必要)
```

### Debug Mode
```bash
# 启用详细日志
python start_web_demo.py --dev

# 查看浏览器开发者工具
F12 -> Console/Network 选项卡
```

## Security Notes

### Data Privacy
- 所有音频/视频数据仅在本地处理
- 不会上传或存储个人数据
- WebSocket连接采用安全传输

### Access Control
- 默认仅允许本地访问 (127.0.0.1)
- 外部访问需显式指定 `--host 0.0.0.0`
- 建议在受信任的网络环境中使用

## Contributing Guide

### Development Environment
```bash
# 克隆仓库
git clone https://github.com/your-repo/TriModalFusion.git
cd TriModalFusion

# 安装开发依赖
pip install -r requirements-dev.txt

# 启动开发服务器
python start_web_demo.py --dev
```

### Code Structure
```
web/
├── templates/
│   └── index.html          # 主页面模板
├── static/
│   ├── style.css          # 样式文件
│   └── app.js             # 前端逻辑
└── README.md              # 本文档

deployment/
└── serve.py               # FastAPI服务器
```

### Adding New Features
1. 前端: 修改 `web/static/app.js`
2. 后端: 修改 `deployment/serve.py`
3. 样式: 修改 `web/static/style.css`
4. 模板: 修改 `web/templates/index.html`

## Technology Stack

### Backend Technologies
- **FastAPI**: Web框架和API
- **WebSocket**: 实时通信
- **PyTorch**: 深度学习推理
- **Uvicorn**: ASGI服务器

### Frontend Technologies
- **HTML5**: Web标准
- **CSS3**: 样式和动画
- **JavaScript ES6**: 交互逻辑
- **WebRTC**: 媒体流处理
- **Canvas API**: 可视化渲染

### Supporting Libraries
- **Bootstrap**: UI框架
- **Font Awesome**: 图标库
- **MediaPipe**: 手势检测 (可选)

## Support and Feedback

- **问题报告**: [GitHub Issues](https://github.com/your-repo/TriModalFusion/issues)
- **功能建议**: [GitHub Discussions](https://github.com/your-repo/TriModalFusion/discussions)
- **邮件联系**: guowei_ni@bit.edu.cn

---

**TriModalFusion Web Demo System** - Making multimodal AI accessible to everyone!