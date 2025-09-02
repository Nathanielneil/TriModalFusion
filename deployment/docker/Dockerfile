# TriModalFusion Production Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .
COPY requirements-prod.txt .

# 安装Python依赖
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install --no-cache-dir -r requirements.txt && \
    python3.9 -m pip install --no-cache-dir -r requirements-prod.txt

# 复制项目代码
COPY . .

# 安装项目包
RUN python3.9 -m pip install -e .

# 创建模型和日志目录
RUN mkdir -p /app/models /app/logs /app/data

# 设置权限
RUN useradd -m -u 1000 trimodal && \
    chown -R trimodal:trimodal /app
USER trimodal

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3.9 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 暴露端口
EXPOSE 8000 8001

# 启动命令
CMD ["python3.9", "deployment/serve.py", "--host", "0.0.0.0", "--port", "8000"]