/**
 * TriModalFusion Web界面主要JavaScript文件
 */

class TriModalDetector {
    constructor() {
        this.isStreaming = false;
        this.websocket = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.audioAnalyzer = null;
        this.gestureDetector = null;
        
        // 性能统计
        this.stats = {
            predictions: 0,
            totalLatency: 0,
            fps: 0,
            lastFrameTime: Date.now()
        };
        
        // 设置
        this.settings = {
            detectionInterval: 500,
            confidenceThreshold: 0.5,
            showFeatures: true,
            showAttention: true
        };
        
        // 历史记录
        this.history = [];
        this.maxHistorySize = 50;
        
        this.initializeUI();
        this.initializeWebSocket();
        this.loadSettings();
    }
    
    /**
     * 初始化UI事件监听
     */
    initializeUI() {
        // 开始/停止检测按钮
        const toggleBtn = document.getElementById('toggleStream');
        toggleBtn.addEventListener('click', () => this.toggleDetection());
        
        // 音频录制按钮
        const audioBtn = document.getElementById('toggleAudio');
        audioBtn.addEventListener('click', () => this.toggleAudio());
        
        // 文件上传按钮
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.addEventListener('click', () => this.uploadFiles());
        
        // 设置相关
        const confidenceSlider = document.getElementById('confidenceThreshold');
        const thresholdValue = document.getElementById('thresholdValue');
        confidenceSlider.addEventListener('input', (e) => {
            thresholdValue.textContent = e.target.value;
            this.settings.confidenceThreshold = parseFloat(e.target.value);
        });
        
        const saveSettingsBtn = document.getElementById('saveSettings');
        saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        
        // 定期更新统计信息
        setInterval(() => this.updateStats(), 1000);
    }
    
    /**
     * 初始化WebSocket连接
     */
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/detection`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket连接已建立');
            this.updateStatus('connected', '已连接');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handlePredictionResult(data);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket连接已关闭');
            this.updateStatus('disconnected', '连接断开');
            // 尝试重连
            setTimeout(() => this.initializeWebSocket(), 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket错误:', error);
            this.updateStatus('error', '连接错误');
        };
    }
    
    /**
     * 切换检测状态
     */
    async toggleDetection() {
        if (!this.isStreaming) {
            await this.startDetection();
        } else {
            this.stopDetection();
        }
    }
    
    /**
     * 开始实时检测
     */
    async startDetection() {
        try {
            this.updateStatus('connecting', '正在启动...');
            
            // 获取摄像头和麦克风权限
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: true
            });
            
            // 设置视频元素
            const video = document.getElementById('videoInput');
            video.srcObject = this.mediaStream;
            
            // 初始化音频分析
            this.initializeAudioAnalysis();
            
            // 初始化手势检测
            this.initializeGestureDetection();
            
            this.isStreaming = true;
            this.updateToggleButton();
            this.updateStatus('connected', '检测中...');
            
            // 开始检测循环
            this.startDetectionLoop();
            
        } catch (error) {
            console.error('启动检测失败:', error);
            this.updateStatus('error', '启动失败');
            alert('无法访问摄像头或麦克风，请检查权限设置');
        }
    }
    
    /**
     * 停止实时检测
     */
    stopDetection() {
        this.isStreaming = false;
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.updateToggleButton();
        this.updateStatus('disconnected', '已停止');
    }
    
    /**
     * 初始化音频分析
     */
    initializeAudioAnalysis() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.audioAnalyzer = this.audioContext.createAnalyser();
        this.audioAnalyzer.fftSize = 256;
        
        source.connect(this.audioAnalyzer);
        
        // 开始音频可视化
        this.visualizeAudio();
    }
    
    /**
     * 初始化手势检测
     */
    initializeGestureDetection() {
        // 这里可以集成MediaPipe手势检测
        // 或者使用其他手势识别库
        console.log('手势检测初始化完成');
    }
    
    /**
     * 开始检测循环
     */
    startDetectionLoop() {
        if (!this.isStreaming) return;
        
        this.captureFrame()
            .then(() => {
                setTimeout(() => this.startDetectionLoop(), this.settings.detectionInterval);
            })
            .catch(error => {
                console.error('检测循环错误:', error);
                setTimeout(() => this.startDetectionLoop(), this.settings.detectionInterval);
            });
    }
    
    /**
     * 捕获当前帧进行检测
     */
    async captureFrame() {
        const video = document.getElementById('videoInput');
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        
        // 设置canvas尺寸
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // 绘制当前帧
        ctx.drawImage(video, 0, 0);
        
        // 获取图像数据
        const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
        
        // 获取音频数据
        const audioData = this.getAudioData();
        
        // 获取手势数据（如果可用）
        const gestureData = this.getGestureData();
        
        // 发送到服务器
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            const frameData = {
                timestamp: Date.now(),
                image: await this.blobToBase64(imageBlob),
                audio: audioData,
                gesture: gestureData
            };
            
            this.websocket.send(JSON.stringify({
                type: 'detection_request',
                data: frameData
            }));
        }
    }
    
    /**
     * 获取音频数据
     */
    getAudioData() {
        if (!this.audioAnalyzer) return null;
        
        const bufferLength = this.audioAnalyzer.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.audioAnalyzer.getByteFrequencyData(dataArray);
        
        return Array.from(dataArray);
    }
    
    /**
     * 获取手势数据
     */
    getGestureData() {
        // 这里应该从手势检测器获取关键点数据
        // 暂时返回模拟数据
        return null;
    }
    
    /**
     * 处理预测结果
     */
    handlePredictionResult(result) {
        const currentTime = Date.now();
        const latency = currentTime - result.timestamp;
        
        // 更新统计信息
        this.stats.predictions++;
        this.stats.totalLatency += latency;
        this.updateLatencyDisplay(latency);
        
        // 显示预测结果
        this.displayPredictionResult(result);
        
        // 可视化特征（如果启用）
        if (this.settings.showFeatures && result.features) {
            this.visualizeFeatures(result.features);
        }
        
        // 显示注意力权重（如果启用）
        if (this.settings.showAttention && result.attention_weights) {
            this.visualizeAttention(result.attention_weights);
        }
        
        // 添加到历史记录
        this.addToHistory(result);
    }
    
    /**
     * 显示预测结果
     */
    displayPredictionResult(result) {
        const container = document.getElementById('predictionResults');
        
        if (result.predictions && result.predictions.classification) {
            const classification = result.predictions.classification;
            const confidence = Math.max(...classification.probabilities) * 100;
            const predictedClass = classification.predicted_classes[0];
            
            // 更新置信度显示
            document.getElementById('confidence').textContent = `${confidence.toFixed(1)}%`;
            
            const resultHtml = `
                <div class="prediction-item pulse">
                    <div class="prediction-header">
                        <div class="prediction-class">类别: ${predictedClass}</div>
                        <div class="prediction-confidence ${this.getConfidenceClass(confidence)}">
                            ${confidence.toFixed(1)}%
                        </div>
                    </div>
                    <div class="topk-predictions">
                        ${classification.top_k.indices.slice(0, 3).map((idx, i) => `
                            <div class="topk-item">
                                ${idx}: ${(classification.top_k.probabilities[i] * 100).toFixed(1)}%
                            </div>
                        `).join('')}
                    </div>
                    <div class="modal-visualization">
                        <div class="modal-item ${result.modalities?.includes('speech') ? 'active' : ''}">
                            <div class="modal-icon"><i class="fas fa-microphone"></i></div>
                            <div class="modal-label">语音</div>
                        </div>
                        <div class="modal-item ${result.modalities?.includes('image') ? 'active' : ''}">
                            <div class="modal-icon"><i class="fas fa-image"></i></div>
                            <div class="modal-label">图像</div>
                        </div>
                        <div class="modal-item ${result.modalities?.includes('gesture') ? 'active' : ''}">
                            <div class="modal-icon"><i class="fas fa-hand-paper"></i></div>
                            <div class="modal-label">手势</div>
                        </div>
                    </div>
                </div>
            `;
            
            container.innerHTML = resultHtml;
        }
    }
    
    /**
     * 可视化特征
     */
    visualizeFeatures(features) {
        if (features.encoded) {
            // 语音特征可视化
            if (features.encoded.speech) {
                this.drawFeatureVisualization('speechFeatureCanvas', features.encoded.speech, '#ff6b6b');
            }
            
            // 图像特征可视化
            if (features.encoded.image) {
                this.drawFeatureVisualization('imageFeatureCanvas', features.encoded.image, '#4ecdc4');
            }
            
            // 手势特征可视化
            if (features.encoded.gesture) {
                this.drawFeatureVisualization('gestureFeatureCanvas', features.encoded.gesture, '#45b7d1');
            }
        }
    }
    
    /**
     * 绘制特征可视化
     */
    drawFeatureVisualization(canvasId, featureData, color) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 简化特征数据用于显示
        const simplified = this.simplifyFeatureData(featureData);
        const maxVal = Math.max(...simplified);
        
        ctx.fillStyle = color;
        const barWidth = canvas.width / simplified.length;
        
        simplified.forEach((value, index) => {
            const height = (value / maxVal) * canvas.height;
            ctx.fillRect(index * barWidth, canvas.height - height, barWidth - 1, height);
        });
    }
    
    /**
     * 可视化注意力权重
     */
    visualizeAttention(attentionWeights) {
        const canvas = document.getElementById('attentionCanvas');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 这里应该根据注意力权重的具体格式进行绘制
        // 暂时绘制一个简化的热力图
        this.drawAttentionHeatmap(ctx, attentionWeights, canvas.width, canvas.height);
    }
    
    /**
     * 音频可视化
     */
    visualizeAudio() {
        if (!this.audioAnalyzer || !this.isStreaming) return;
        
        const canvas = document.getElementById('audioCanvas');
        const ctx = canvas.getContext('2d');
        const bufferLength = this.audioAnalyzer.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            if (!this.isStreaming) return;
            
            requestAnimationFrame(draw);
            
            this.audioAnalyzer.getByteFrequencyData(dataArray);
            
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * canvas.height;
                
                const r = barHeight + 25 * (i / bufferLength);
                const g = 250 * (i / bufferLength);
                const b = 50;
                
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        };
        
        draw();
    }
    
    /**
     * 文件上传处理
     */
    async uploadFiles() {
        const speechFile = document.getElementById('speechFile').files[0];
        const imageFile = document.getElementById('imageFile').files[0];
        const gestureFile = document.getElementById('gestureFile').files[0];
        
        if (!speechFile && !imageFile && !gestureFile) {
            alert('请至少选择一个文件');
            return;
        }
        
        const formData = new FormData();
        if (speechFile) formData.append('speech_file', speechFile);
        if (imageFile) formData.append('image_file', imageFile);
        if (gestureFile) formData.append('gesture_file', gestureFile);
        
        const requestData = {
            return_features: this.settings.showFeatures,
            return_attention: this.settings.showAttention,
            temperature: 1.0
        };
        
        formData.append('request', JSON.stringify(requestData));
        
        try {
            this.updateStatus('connecting', '处理中...');
            
            const response = await fetch('/predict/files', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('上传失败');
            }
            
            const result = await response.json();
            this.handlePredictionResult({
                ...result,
                timestamp: Date.now(),
                modalities: [
                    speechFile ? 'speech' : null,
                    imageFile ? 'image' : null,
                    gestureFile ? 'gesture' : null
                ].filter(Boolean)
            });
            
            this.updateStatus('connected', '处理完成');
            
        } catch (error) {
            console.error('文件上传错误:', error);
            this.updateStatus('error', '处理失败');
            alert('文件处理失败，请重试');
        }
    }
    
    /**
     * 工具函数
     */
    
    updateStatus(status, text) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = text;
        statusElement.className = `badge status-${status} me-3`;
    }
    
    updateToggleButton() {
        const button = document.getElementById('toggleStream');
        if (this.isStreaming) {
            button.innerHTML = '<i class="fas fa-stop"></i> 停止检测';
            button.className = 'btn btn-danger btn-sm';
        } else {
            button.innerHTML = '<i class="fas fa-play"></i> 开始检测';
            button.className = 'btn btn-success btn-sm';
        }
    }
    
    updateStats() {
        // FPS计算
        const now = Date.now();
        this.stats.fps = 1000 / (now - this.stats.lastFrameTime);
        this.stats.lastFrameTime = now;
        
        document.getElementById('fps').textContent = this.stats.fps.toFixed(1);
        document.getElementById('predictions').textContent = this.stats.predictions;
    }
    
    updateLatencyDisplay(latency) {
        document.getElementById('latency').textContent = `${latency}ms`;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 80) return 'confidence-high';
        if (confidence >= 50) return 'confidence-medium';
        return 'confidence-low';
    }
    
    addToHistory(result) {
        const timestamp = new Date().toLocaleTimeString();
        const confidence = result.predictions?.classification ? 
            Math.max(...result.predictions.classification.probabilities) * 100 : 0;
        const predictedClass = result.predictions?.classification?.predicted_classes?.[0] || '未知';
        
        this.history.unshift({
            timestamp,
            class: predictedClass,
            confidence: confidence.toFixed(1)
        });
        
        if (this.history.length > this.maxHistorySize) {
            this.history.pop();
        }
        
        this.updateHistoryDisplay();
    }
    
    updateHistoryDisplay() {
        const container = document.getElementById('historyList');
        container.innerHTML = this.history.map(item => `
            <div class="history-item">
                <span class="history-result">${item.class}</span>
                <span class="history-time">${item.timestamp} (${item.confidence}%)</span>
            </div>
        `).join('');
    }
    
    simplifyFeatureData(data, targetSize = 20) {
        if (data.length <= targetSize) return data;
        
        const step = Math.floor(data.length / targetSize);
        const simplified = [];
        
        for (let i = 0; i < data.length; i += step) {
            simplified.push(data[i]);
        }
        
        return simplified.slice(0, targetSize);
    }
    
    drawAttentionHeatmap(ctx, weights, width, height) {
        // 简化的注意力权重可视化
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, width, height);
        
        // 这里应该根据实际的注意力权重格式进行绘制
        // 暂时绘制一些示例热力图
        const cellWidth = width / 10;
        const cellHeight = height / 3;
        
        for (let i = 0; i < 10; i++) {
            for (let j = 0; j < 3; j++) {
                const intensity = Math.random();
                ctx.fillStyle = `rgba(255, 0, 0, ${intensity})`;
                ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth - 1, cellHeight - 1);
            }
        }
    }
    
    async blobToBase64(blob) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(blob);
        });
    }
    
    loadSettings() {
        const saved = localStorage.getItem('trimodal_settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
            this.applySettings();
        }
    }
    
    saveSettings() {
        localStorage.setItem('trimodal_settings', JSON.stringify(this.settings));
        this.applySettings();
        
        // 关闭设置模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
        modal.hide();
    }
    
    applySettings() {
        document.getElementById('detectionInterval').value = this.settings.detectionInterval;
        document.getElementById('confidenceThreshold').value = this.settings.confidenceThreshold;
        document.getElementById('thresholdValue').textContent = this.settings.confidenceThreshold;
        document.getElementById('showFeatures').checked = this.settings.showFeatures;
        document.getElementById('showAttention').checked = this.settings.showAttention;
    }
    
    toggleAudio() {
        // 音频录制切换逻辑
        const button = document.getElementById('toggleAudio');
        if (button.textContent.includes('开始')) {
            button.innerHTML = '<i class="fas fa-microphone-slash"></i> 停止录音';
            button.className = 'btn btn-danger btn-sm';
        } else {
            button.innerHTML = '<i class="fas fa-microphone"></i> 开始录音';
            button.className = 'btn btn-outline-primary btn-sm';
        }
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    window.detector = new TriModalDetector();
});