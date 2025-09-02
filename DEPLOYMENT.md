# TriModalFusion 部署指南

## 部署概述

TriModalFusion提供了多种部署方式，从开发测试到生产环境的完整解决方案。本文档详细介绍了各种部署选项、配置和最佳实践。

## 🚀 快速部署

### 1. 本地开发部署

```bash
# 1. 克隆项目
git clone https://github.com/your-org/TriModalFusion.git
cd TriModalFusion

# 2. 安装依赖
pip install -r requirements.txt
pip install -r requirements-prod.txt

# 3. 下载预训练模型
mkdir -p models
# 将训练好的模型放到 models/best_model.pth

# 4. 启动API服务
python deployment/serve.py --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看API文档。

### 2. Docker 单容器部署

```bash
# 构建Docker镜像
docker build -t trimodal-fusion -f deployment/docker/Dockerfile .

# 运行容器
docker run -d \
  --name trimodal-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/configs:/app/configs:ro \
  trimodal-fusion
```

## 🏢 生产环境部署

### Docker Compose 部署（推荐）

完整的生产级部署，包含负载均衡、缓存、数据库和监控。

```bash
cd deployment/docker

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f trimodal-api

# 扩展API服务
docker-compose up -d --scale trimodal-api=3
```

#### 服务架构：
- **trimodal-api**: 主API服务
- **nginx**: 负载均衡和反向代理
- **redis**: 缓存和会话存储
- **postgres**: 数据库
- **prometheus + grafana**: 监控系统
- **elasticsearch**: 日志聚合
- **celery**: 异步任务队列

#### 访问地址：
- API服务: `http://localhost`
- API文档: `http://localhost/docs`
- 监控面板: `http://localhost:3000` (admin/admin)
- 任务监控: `http://localhost:5555`
- 指标监控: `http://localhost:9090`

### Kubernetes 部署（企业级）

适用于大规模生产环境的容器编排部署。

```bash
# 1. 创建命名空间
kubectl create namespace trimodal

# 2. 部署应用
kubectl apply -f deployment/kubernetes/

# 3. 检查部署状态
kubectl get pods -n trimodal
kubectl get services -n trimodal

# 4. 查看日志
kubectl logs -f deployment/trimodal-api -n trimodal

# 5. 端口转发（开发测试）
kubectl port-forward service/trimodal-api-service 8000:80 -n trimodal
```

#### 功能特性：
- **自动扩缩容**: 基于CPU/内存使用率
- **滚动更新**: 零停机更新
- **健康检查**: 自动重启故障容器
- **资源限制**: GPU和内存资源管理
- **负载均衡**: Ingress控制器
- **SSL终端**: 自动HTTPS证书

## ⚙️ 配置管理

### 环境配置

创建 `configs/production_config.yaml`:

```yaml
# 模型配置
model:
  name: "TriModalFusion"
  d_model: 512
  tasks: ["classification"]
  num_classes: 10

# 服务配置
serving:
  device: "cuda"
  batch_size: 16
  half_precision: true
  model_path: "/app/models/best_model.pth"
  max_concurrent_requests: 100
  request_timeout: 30

# 性能优化
optimization:
  enable_tensorrt: true
  enable_model_compilation: true
  memory_pool_size: "2GB"
  
# 监控配置
monitoring:
  enable_metrics: true
  log_level: "INFO"
  enable_request_logging: true
```

### 环境变量

```bash
# 模型配置
export TRIMODAL_MODEL_PATH="/path/to/best_model.pth"
export TRIMODAL_CONFIG_PATH="/path/to/config.yaml"

# 服务配置
export TRIMODAL_HOST="0.0.0.0"
export TRIMODAL_PORT="8000"
export TRIMODAL_WORKERS="4"

# 数据库连接
export DATABASE_URL="postgresql://user:pass@host:5432/trimodal"
export REDIS_URL="redis://host:6379/0"

# 安全配置
export API_KEY="your-secure-api-key"
export JWT_SECRET="your-jwt-secret"

# 存储配置
export MODEL_STORAGE_PATH="/models"
export LOG_STORAGE_PATH="/logs"
```

## 📊 性能优化

### 1. 模型优化

```bash
# 转换为TensorRT（NVIDIA GPU）
python scripts/optimize_model.py \
  --input-model models/best_model.pth \
  --output-model models/optimized_model.trt \
  --precision fp16

# 转换为ONNX（通用加速）
python scripts/convert_to_onnx.py \
  --input-model models/best_model.pth \
  --output-model models/model.onnx \
  --dynamic-batch
```

### 2. 服务优化

```python
# 启用并发处理
uvicorn deployment.serve:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker

# 启用异步处理
gunicorn deployment.serve:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --worker-connections 1000
```

### 3. 缓存配置

```yaml
# Redis缓存配置
cache:
  redis_url: "redis://localhost:6379/0"
  ttl: 3600  # 1小时
  enable_result_cache: true
  enable_model_cache: true
```

## 🔒 安全配置

### 1. API安全

```python
# API密钥认证
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != os.getenv("API_KEY"):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key"}
        )
    return await call_next(request)
```

### 2. 网络安全

```nginx
# Nginx安全配置
server {
    listen 443 ssl http2;
    server_name api.trimodal.example.com;
    
    # SSL配置
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # 限流
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://trimodal-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. 容器安全

```dockerfile
# 使用非root用户
RUN useradd -m -u 1000 trimodal && \
    chown -R trimodal:trimodal /app
USER trimodal

# 只安装必要包
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/*
```

## 📈 监控与日志

### 1. 应用监控

```python
# Prometheus指标
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    response = await call_next(request)
    
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response
```

### 2. 日志配置

```python
# 结构化日志
import structlog

logger = structlog.get_logger()

logger.info(
    "Prediction completed",
    request_id=request_id,
    inference_time=inference_time,
    model_version="1.0",
    input_modalities=["speech", "image"]
)
```

### 3. 健康检查

```python
@app.get("/health/deep")
async def deep_health_check():
    checks = {
        "model_loaded": predictor is not None,
        "gpu_available": torch.cuda.is_available(),
        "redis_connected": await check_redis_connection(),
        "database_connected": await check_db_connection(),
        "disk_space_ok": check_disk_space() > 0.1,  # >10% 剩余
        "memory_ok": check_memory_usage() < 0.9,    # <90% 使用
    }
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": status,
        "checks": checks,
        "timestamp": time.time()
    }
```

## 🔧 故障排除

### 常见问题

#### 1. CUDA内存不足
```bash
# 解决方案：
export CUDA_VISIBLE_DEVICES=0  # 只使用一个GPU
# 或在配置中减少batch_size
```

#### 2. 模型加载失败
```bash
# 检查模型文件
ls -la models/best_model.pth
# 检查模型兼容性
python -c "import torch; torch.load('models/best_model.pth', map_location='cpu')"
```

#### 3. API响应慢
```bash
# 检查系统资源
htop
nvidia-smi
# 启用性能分析
python -m cProfile deployment/serve.py
```

#### 4. 容器启动失败
```bash
# 查看容器日志
docker logs trimodal-api
# 进入容器调试
docker exec -it trimodal-api bash
```

### 性能基准

| 配置 | 吞吐量 (req/s) | 平均延迟 (ms) | GPU内存 (GB) |
|------|----------------|---------------|--------------|
| T4 + FP16 | 15 | 150 | 4.2 |
| V100 + FP16 | 45 | 60 | 6.8 |
| A100 + FP16 | 80 | 35 | 12.1 |

### 扩展建议

#### 1. 水平扩展
```bash
# Docker Compose扩展
docker-compose up -d --scale trimodal-api=5

# Kubernetes扩展
kubectl scale deployment trimodal-api --replicas=5
```

#### 2. 垂直扩展
```yaml
# 增加资源限制
resources:
  requests:
    memory: "8Gi"
    cpu: "2000m"
    nvidia.com/gpu: 2
  limits:
    memory: "16Gi" 
    cpu: "4000m"
    nvidia.com/gpu: 2
```

## 📚 更多资源

- [API文档](http://localhost:8000/docs)
- [监控面板](http://localhost:3000)
- [性能基准测试](./benchmark.md)
- [安全最佳实践](./security.md)
- [故障排除指南](./troubleshooting.md)

---

如需技术支持，请访问：
- GitHub Issues: https://github.com/your-org/TriModalFusion/issues
- 技术文档: https://trimodal-docs.example.com
- 社区讨论: https://discord.gg/trimodal