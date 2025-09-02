#!/usr/bin/env python3
"""
TriModalFusion 生产环境服务器
提供HTTP API接口用于模型推理
"""

import asyncio
import logging
import time
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import numpy as np

# 导入项目模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import TriModalPredictor
from src.inference.batch_predictor import BatchPredictor
from src.utils.config import load_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
predictor = None
batch_predictor = None
app_config = None


class PredictionRequest(BaseModel):
    """预测请求模型"""
    return_features: bool = Field(default=False, description="是否返回特征向量")
    return_attention: bool = Field(default=False, description="是否返回注意力权重")
    temperature: float = Field(default=1.0, description="softmax温度参数", ge=0.1, le=10.0)
    batch_size: int = Field(default=1, description="批处理大小", ge=1, le=32)


class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool
    predictions: Dict[str, Any]
    inference_time: float
    model_version: str = "1.0"
    timestamp: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime: float


# 启动和关闭时的操作
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global predictor, batch_predictor, app_config
    
    # 启动时加载模型
    logger.info("正在启动TriModalFusion服务...")
    
    try:
        # 加载配置
        config_path = Path("configs/production_config.yaml")
        if not config_path.exists():
            config_path = Path("configs/default_config.yaml")
        
        app_config = load_config(str(config_path))
        
        # 加载模型
        model_path = app_config.serving.get("model_path", "checkpoints/best_model.pth")
        
        predictor = TriModalPredictor(
            model_path=model_path,
            config_path=str(config_path),
            device=app_config.serving.get("device", "auto"),
            batch_size=app_config.serving.get("batch_size", 8),
            half_precision=app_config.serving.get("half_precision", True)
        )
        
        batch_predictor = BatchPredictor(
            model_path=model_path,
            config_path=str(config_path),
            device=app_config.serving.get("device", "auto"),
            batch_size=app_config.serving.get("batch_size", 8),
            half_precision=app_config.serving.get("half_precision", True)
        )
        
        logger.info("TriModalFusion服务启动成功!")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("正在关闭TriModalFusion服务...")


# 创建FastAPI应用
app = FastAPI(
    title="TriModalFusion API",
    description="多模态AI推理服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 记录启动时间
start_time = time.time()


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "service": "TriModalFusion API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    import torch
    import psutil
    
    # 检查GPU内存
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    # 检查系统内存
    memory = psutil.virtual_memory()
    
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        gpu_available=torch.cuda.is_available(),
        memory_usage={
            "system_percent": memory.percent,
            "system_available_gb": memory.available / 1024**3,
            **gpu_memory
        },
        uptime=time.time() - start_time
    )


@app.post("/predict/files", response_model=PredictionResponse)
async def predict_from_files(
    request: PredictionRequest,
    speech_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    gesture_file: Optional[UploadFile] = File(None)
):
    """从上传文件进行预测"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 保存上传的文件
        temp_files = {}
        
        if speech_file:
            speech_path = f"/tmp/speech_{int(time.time())}.wav"
            with open(speech_path, "wb") as f:
                content = await speech_file.read()
                f.write(content)
            temp_files['speech'] = speech_path
        
        if image_file:
            image_path = f"/tmp/image_{int(time.time())}.jpg"
            with open(image_path, "wb") as f:
                content = await image_file.read()
                f.write(content)
            temp_files['image'] = image_path
        
        if gesture_file:
            gesture_path = f"/tmp/gesture_{int(time.time())}.json"
            with open(gesture_path, "wb") as f:
                content = await gesture_file.read()
                f.write(content)
            temp_files['gesture'] = gesture_path
        
        # 进行预测
        predictions = predictor.predict_from_files(
            temp_files,
            return_features=request.return_features,
            return_attention=request.return_attention,
            temperature=request.temperature
        )
        
        # 清理临时文件
        for file_path in temp_files.values():
            Path(file_path).unlink(missing_ok=True)
        
        return PredictionResponse(
            success=True,
            predictions=predictions,
            inference_time=predictions.get('inference_time', 0.0),
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=Dict[str, Any])
async def predict_batch(
    background_tasks: BackgroundTasks,
    request: PredictionRequest,
    files: List[UploadFile] = File(...)
):
    """批量预测"""
    if batch_predictor is None:
        raise HTTPException(status_code=503, detail="批量预测器未加载")
    
    try:
        # 这里简化实现，实际需要处理文件组织
        task_id = f"batch_{int(time.time())}"
        
        # 在后台执行批量预测
        background_tasks.add_task(
            process_batch_prediction,
            task_id,
            files,
            request
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "processing",
            "message": f"批量预测任务已启动，任务ID: {task_id}"
        }
        
    except Exception as e:
        logger.error(f"批量预测启动失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_batch_prediction(task_id: str, files: List[UploadFile], request: PredictionRequest):
    """处理批量预测任务"""
    try:
        logger.info(f"开始处理批量预测任务: {task_id}")
        
        # 这里需要实现具体的批量处理逻辑
        # 包括文件保存、数据组织、批量预测等
        
        # 模拟处理时间
        await asyncio.sleep(5)
        
        logger.info(f"批量预测任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"批量预测任务失败 {task_id}: {e}")


@app.get("/predict/batch/{task_id}/status")
async def get_batch_status(task_id: str):
    """获取批量预测任务状态"""
    # 这里需要实现任务状态跟踪
    return {
        "task_id": task_id,
        "status": "completed",  # pending, processing, completed, failed
        "progress": 100,
        "result_path": f"/results/{task_id}/predictions.json"
    }


@app.get("/model/info")
async def get_model_info():
    """获取模型信息"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_name": "TriModalFusion",
        "version": "1.0.0",
        "supported_modalities": ["speech", "gesture", "image"],
        "supported_tasks": app_config.model.tasks if app_config else [],
        "model_parameters": predictor.model.get_num_parameters() if hasattr(predictor.model, 'get_num_parameters') else "unknown",
        "device": str(predictor.device),
        "half_precision": predictor.half_precision
    }


@app.get("/model/benchmark")
async def benchmark_model(num_runs: int = 100, batch_size: int = 1):
    """模型性能基准测试"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        benchmark_results = predictor.benchmark(num_runs=num_runs, batch_size=batch_size)
        return {
            "success": True,
            "benchmark_results": benchmark_results,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    import torch
    import psutil
    
    metrics = {
        "service": {
            "uptime": time.time() - start_time,
            "requests_total": "unknown",  # 需要添加计数器
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    }
    
    if torch.cuda.is_available():
        metrics["gpu"] = {
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "utilization_percent": "unknown"  # 需要nvidia-ml-py
        }
    
    if predictor:
        stats = predictor.get_stats()
        metrics["model"] = {
            "total_predictions": stats['total_predictions'],
            "average_inference_time": stats['average_time'],
            "total_inference_time": stats['total_time']
        }
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TriModalFusion API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开启热重载")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    # 启动服务器
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()