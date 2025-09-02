# TriModalFusion 训练指南

## 目录
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [分布式训练](#分布式训练)
- [超参数调优](#超参数调优)
- [训练监控](#训练监控)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 快速开始

### 环境准备

```bash
# 1. 克隆代码库
git clone https://github.com/your-repo/TriModalFusion.git
cd TriModalFusion

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import mediapipe; print('MediaPipe安装成功')"
```

### 基础训练

```bash
# 使用默认配置开始训练
python train.py --config configs/default_config.yaml

# 使用自定义配置
python train.py --config configs/my_config.yaml --gpus 1
```

### 最小训练示例

```python
# minimal_train.py
import torch
from src.models.trimodal_fusion import TriModalFusionModel
from src.utils.config import load_config
from src.trainers.multimodal_trainer import MultiModalTrainer

# 加载配置
config = load_config("configs/default_config.yaml")

# 创建模型
model = TriModalFusionModel(config)

# 创建训练器
trainer = MultiModalTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader
)

# 开始训练
trainer.fit()
```

## 数据准备

### 数据格式要求

TriModalFusion支持多种数据格式，推荐使用HDF5格式进行高效存储和加载。

#### 1. 数据集结构

```
dataset/
├── train/
│   ├── speech/          # 语音文件 (.wav, .mp3, .flac)
│   ├── video/           # 视频文件 (.mp4, .avi, .mov)
│   ├── images/          # 图像文件 (.jpg, .png, .bmp)
│   └── annotations.json # 标注文件
├── val/
│   └── ...
└── test/
    └── ...
```

#### 2. 标注文件格式

```json
{
  "samples": [
    {
      "id": "sample_001",
      "speech_path": "speech/sample_001.wav",
      "video_path": "video/sample_001.mp4", 
      "image_path": "images/sample_001.jpg",
      "label": 5,
      "transcription": "Hello world",
      "bounding_boxes": [
        {
          "class": 1,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95
        }
      ],
      "metadata": {
        "duration": 3.5,
        "fps": 30,
        "resolution": [640, 480]
      }
    }
  ]
}
```

### 数据预处理

#### 自动预处理脚本

```python
# scripts/prepare_data.py
import argparse
from src.data.preprocessor import MultiModalPreprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', default='configs/preprocessing.yaml')
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = MultiModalPreprocessor(
        config_path=args.config
    )
    
    # 预处理数据
    preprocessor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1}
    )

if __name__ == "__main__":
    main()
```

#### 预处理配置

```yaml
# configs/preprocessing.yaml
preprocessing:
  # 语音预处理
  speech:
    target_sample_rate: 16000
    max_duration: 10.0
    trim_silence: true
    normalize: true
    
  # 手势预处理
  gesture:
    target_fps: 30
    max_frames: 300
    smooth_keypoints: true
    normalize_coordinates: true
    
  # 图像预处理
  image:
    target_size: [224, 224]
    normalize: true
    augmentation: true

# 数据增强
augmentation:
  speech:
    speed_perturbation: [0.9, 1.1]
    volume_perturbation: [0.8, 1.2]
    noise_injection: 0.1
    
  gesture:
    temporal_jitter: 0.1
    spatial_noise: 0.02
    
  image:
    horizontal_flip: 0.5
    rotation: [-15, 15]
    color_jitter: [0.1, 0.1, 0.1, 0.05]
```

### 自定义数据加载器

```python
# src/data/custom_dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
from pathlib import Path

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, config, split='train'):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # 加载标注
        with open(self.data_dir / f"{split}_annotations.json") as f:
            self.annotations = json.load(f)
        
        # 预计算的特征文件
        self.h5_file = h5py.File(
            self.data_dir / f"{split}_features.h5", 'r'
        )
        
    def __len__(self):
        return len(self.annotations['samples'])
    
    def __getitem__(self, idx):
        sample = self.annotations['samples'][idx]
        sample_id = sample['id']
        
        # 加载预计算特征
        features = {}
        
        # 语音特征
        if 'speech' in self.h5_file:
            features['speech'] = torch.from_numpy(
                self.h5_file['speech'][sample_id][:]
            ).float()
        
        # 手势特征  
        if 'gesture' in self.h5_file:
            features['gesture'] = torch.from_numpy(
                self.h5_file['gesture'][sample_id][:]
            ).float()
            
        # 图像特征
        if 'image' in self.h5_file:
            features['image'] = torch.from_numpy(
                self.h5_file['image'][sample_id][:]
            ).float()
        
        # 标签
        targets = {}
        if 'label' in sample:
            targets['classification'] = torch.tensor(sample['label'])
        if 'bounding_boxes' in sample:
            targets['detection'] = self._process_bboxes(sample['bounding_boxes'])
        
        return features, targets
    
    def _process_bboxes(self, bboxes):
        # 处理边界框标注
        boxes = []
        labels = []
        for bbox in bboxes:
            boxes.append(bbox['bbox'])
            labels.append(bbox['class'])
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
```

## 训练流程

### 训练器架构

```python
# src/trainers/multimodal_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MultiModalTrainer(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        
        # 损失函数
        self._setup_losses()
        
        # 评估指标
        self._setup_metrics()
        
    def _setup_losses(self):
        """初始化损失函数"""
        self.losses = nn.ModuleDict()
        
        # 分类损失
        if 'classification' in self.config.model.tasks:
            if self.config.training.loss_config.classification.type == 'cross_entropy':
                self.losses['classification'] = nn.CrossEntropyLoss(
                    label_smoothing=self.config.training.loss_config.classification.label_smoothing
                )
            elif self.config.training.loss_config.classification.type == 'focal':
                self.losses['classification'] = FocalLoss(
                    alpha=self.config.training.loss_config.focal_loss.alpha,
                    gamma=self.config.training.loss_config.focal_loss.gamma
                )
        
        # 检测损失
        if 'detection' in self.config.model.tasks:
            self.losses['detection'] = DETRLoss(
                num_classes=self.config.model.num_detection_classes,
                weight_dict=self.config.training.loss_config.detection.loss_weights
            )
        
        # 回归损失
        if 'regression' in self.config.model.tasks:
            if self.config.training.loss_config.regression.loss_function == 'mse':
                self.losses['regression'] = nn.MSELoss()
            elif self.config.training.loss_config.regression.loss_function == 'mae':
                self.losses['regression'] = nn.L1Loss()
            elif self.config.training.loss_config.regression.loss_function == 'huber':
                self.losses['regression'] = nn.HuberLoss(
                    delta=self.config.training.loss_config.regression.huber_delta
                )
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        inputs, targets = batch
        
        # 前向传播
        outputs = self.model(inputs)
        
        # 计算损失
        losses = self.model.compute_loss(outputs, targets)
        
        # 记录损失
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        # 计算训练指标
        if batch_idx % self.config.training.log_every_n_steps == 0:
            train_metrics = self._compute_metrics(outputs, targets, 'train')
            for metric_name, metric_value in train_metrics.items():
                self.log(f'train_{metric_name}', metric_value, on_step=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs, targets = batch
        
        # 前向传播
        outputs = self.model(inputs)
        
        # 计算损失
        losses = self.model.compute_loss(outputs, targets)
        
        # 记录验证损失
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_epoch=True, sync_dist=True)
        
        # 计算验证指标
        val_metrics = self._compute_metrics(outputs, targets, 'val')
        for metric_name, metric_value in val_metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, sync_dist=True)
        
        return {
            'val_loss': losses['total_loss'],
            'outputs': outputs,
            'targets': targets
        }
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 优化器
        if self.config.training.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
                nesterov=self.config.training.nesterov
            )
        
        # 学习率调度器
        scheduler_config = {}
        
        if self.config.training.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.cosine_config.T_max,
                eta_min=self.config.training.cosine_config.eta_min
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        elif self.config.training.scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.config.training.linear_config.end_factor,
                total_iters=self.config.training.linear_config.decay_steps
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        
        # 预热调度器
        if hasattr(self.config.training, 'warmup_steps') and self.config.training.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.training.warmup_steps
            )
            scheduler_config = {
                'scheduler': [warmup_scheduler, scheduler],
                'interval': 'step',
                'frequency': 1
            }
        
        if scheduler_config:
            return [optimizer], [scheduler_config]
        else:
            return optimizer
```

### 训练脚本

```python
# train.py
import argparse
import os
import logging
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    RichProgressBar, RichModelSummary
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from src.models.trimodal_fusion import TriModalFusionModel
from src.trainers.multimodal_trainer import MultiModalTrainer
from src.data.datamodule import MultiModalDataModule
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='TriModalFusion训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--fast_dev_run', action='store_true', help='快速开发运行')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(level='DEBUG' if args.debug else 'INFO')
    logger = logging.getLogger(__name__)
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 设置随机种子
    if hasattr(config.system, 'seed'):
        L.seed_everything(config.system.seed, workers=True)
    
    # 创建数据模块
    logger.info("创建数据模块...")
    datamodule = MultiModalDataModule(config)
    
    # 创建模型
    logger.info("创建模型...")
    model = TriModalFusionModel(config)
    
    # 创建训练器
    trainer_module = MultiModalTrainer(model, config)
    
    # 设置回调函数
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename='{epoch:02d}-{val_accuracy:.3f}',
        monitor=config.system.checkpoint.monitor,
        mode=config.system.checkpoint.mode,
        save_top_k=config.system.checkpoint.save_top_k,
        save_last=config.system.checkpoint.save_last,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    if config.training.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=config.training.early_stopping.monitor,
            min_delta=config.training.early_stopping.min_delta,
            patience=config.training.early_stopping.patience,
            mode=config.training.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(
        logging_interval=config.training.lr_monitor.logging_interval,
        log_momentum=config.training.lr_monitor.log_momentum
    )
    callbacks.append(lr_monitor)
    
    # 进度条
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # 模型摘要
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    # 设置日志记录器
    loggers = []
    
    # TensorBoard
    if config.system.logging.tensorboard.enabled:
        tb_logger = TensorBoardLogger(
            save_dir=config.paths.log_dir,
            name=config.experiment.name,
            version=config.experiment.version,
            log_graph=config.system.logging.tensorboard.log_graph
        )
        loggers.append(tb_logger)
    
    # Weights & Biases
    if config.system.logging.wandb.enabled:
        wandb_logger = WandbLogger(
            project=config.system.logging.wandb.project,
            entity=config.system.logging.wandb.entity,
            name=f"{config.experiment.name}-{config.experiment.version}",
            tags=config.system.logging.wandb.tags,
            config=config
        )
        loggers.append(wandb_logger)
    
    # 创建Lightning Trainer
    trainer = L.Trainer(
        # 基础配置
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps if hasattr(config.training, 'max_steps') else -1,
        
        # 硬件配置
        devices=args.gpus,
        accelerator='gpu' if torch.cuda.is_available() and args.gpus > 0 else 'cpu',
        strategy='ddp' if args.gpus > 1 else 'auto',
        precision=config.system.precision if hasattr(config.system, 'precision') else 32,
        
        # 日志和回调
        logger=loggers if loggers else True,
        callbacks=callbacks,
        log_every_n_steps=config.system.logging.log_every_n_steps,
        
        # 验证配置
        check_val_every_n_epoch=config.training.validate_every_n_epochs,
        val_check_interval=config.evaluation.val_check_interval,
        
        # 梯度相关
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        
        # 其他配置
        deterministic=config.system.deterministic if hasattr(config.system, 'deterministic') else False,
        benchmark=config.system.benchmark if hasattr(config.system, 'benchmark') else True,
        fast_dev_run=args.fast_dev_run,
        detect_anomaly=config.system.detect_anomaly if hasattr(config.system, 'detect_anomaly') else False,
        
        # 分析器
        profiler=config.system.profiler if hasattr(config.system, 'profiler') else None,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.fit(
        model=trainer_module,
        datamodule=datamodule,
        ckpt_path=args.resume
    )
    
    # 测试最佳模型
    logger.info("测试最佳模型...")
    trainer.test(
        model=trainer_module,
        datamodule=datamodule,
        ckpt_path='best'
    )
    
    logger.info("训练完成!")

if __name__ == '__main__':
    main()
```

## 分布式训练

### 多GPU训练

```bash
# 单机多GPU训练
python train.py --config configs/default_config.yaml --gpus 4

# 使用torchrun进行分布式训练
torchrun --nproc_per_node=4 train.py --config configs/default_config.yaml
```

### 分布式配置

```yaml
# configs/distributed_config.yaml
system:
  distributed:
    strategy: "ddp"                    # 分布式策略
    find_unused_parameters: false     # 查找未使用参数
    gradient_as_bucket_view: true      # 梯度桶视图优化
    
  # 混合精度训练
  mixed_precision:
    enabled: true
    opt_level: "O1"
    loss_scale: "dynamic"

training:
  # 根据GPU数量调整批次大小
  batch_size: 32                      # 每个GPU的批次大小
  accumulate_grad_batches: 2          # 梯度累积
  
  # 学习率根据总批次大小调整
  learning_rate: 4e-4                 # base_lr * num_gpus
```

### 分布式训练脚本

```python
# scripts/distributed_train.py
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
```

## 超参数调优

### Grid Search

```python
# scripts/hyperparameter_search.py
import itertools
from typing import Dict, List, Any

def grid_search(base_config: Dict[str, Any], 
                search_space: Dict[str, List[Any]]) -> List[Dict]:
    """网格搜索超参数"""
    
    # 生成所有参数组合
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    configs = []
    for combination in itertools.product(*values):
        config = base_config.copy()
        
        # 更新超参数
        for key, value in zip(keys, combination):
            # 支持嵌套键，如 'training.learning_rate'
            keys_path = key.split('.')
            current = config
            for k in keys_path[:-1]:
                current = current[k]
            current[keys_path[-1]] = value
        
        configs.append(config)
    
    return configs

# 使用示例
search_space = {
    'training.learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    'training.batch_size': [16, 32, 64],
    'model.dropout': [0.0, 0.1, 0.2],
    'fusion_config.num_heads': [4, 8, 12]
}

base_config = load_config('configs/default_config.yaml')
configs = grid_search(base_config, search_space)

# 运行所有配置
for i, config in enumerate(configs):
    print(f"运行配置 {i+1}/{len(configs)}")
    run_experiment(config, experiment_name=f"grid_search_{i}")
```

### Bayesian Optimization

```python
# scripts/bayesian_optimization.py
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# 定义搜索空间
dimensions = [
    Real(low=1e-5, high=1e-2, name='learning_rate', prior='log-uniform'),
    Integer(low=16, high=128, name='batch_size'),
    Real(low=0.0, high=0.3, name='dropout'),
    Integer(low=4, high=16, name='num_heads')
]

@use_named_args(dimensions)
def objective(**params):
    """目标函数"""
    # 更新配置
    config = base_config.copy()
    config['training']['learning_rate'] = params['learning_rate']
    config['training']['batch_size'] = params['batch_size']
    config['model']['dropout'] = params['dropout']
    config['fusion_config']['num_heads'] = params['num_heads']
    
    # 训练模型
    result = run_experiment(config)
    
    # 返回负的验证准确率（因为gp_minimize最小化目标函数）
    return -result['val_accuracy']

# 运行贝叶斯优化
result = gp_minimize(
    func=objective,
    dimensions=dimensions,
    n_calls=50,
    random_state=42
)

print(f"最佳参数: {result.x}")
print(f"最佳性能: {-result.fun}")
```

### Optuna集成

```python
# scripts/optuna_optimization.py
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    """Optuna目标函数"""
    # 建议超参数
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    num_heads = trial.suggest_int('num_heads', 4, 16, step=4)
    
    # 更新配置
    config = base_config.copy()
    config['training']['learning_rate'] = lr
    config['training']['batch_size'] = batch_size
    config['model']['dropout'] = dropout
    config['fusion_config']['num_heads'] = num_heads
    
    # 添加剪枝回调
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_accuracy')
    
    # 训练模型
    trainer = create_trainer(config, callbacks=[pruning_callback])
    trainer.fit(model, datamodule)
    
    return trainer.callback_metrics['val_accuracy'].item()

# 创建研究
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
)

# 运行优化
study.optimize(objective, n_trials=100)

print(f"最佳参数: {study.best_params}")
print(f"最佳性能: {study.best_value}")
```

## 训练监控

### 指标监控

```python
# src/utils/monitoring.py
import wandb
import matplotlib.pyplot as plt
from typing import Dict, List

class TrainingMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics_history = {}
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录训练指标"""
        # 记录到历史
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append((step, value))
        
        # 记录到WandB
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_groups = {
            'Loss': ['train_loss', 'val_loss'],
            'Accuracy': ['train_accuracy', 'val_accuracy'],
            'Learning Rate': ['lr'],
            'Other': []
        }
        
        for idx, (group_name, metric_names) in enumerate(metric_groups.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            for metric_name in metric_names:
                if metric_name in self.metrics_history:
                    steps, values = zip(*self.metrics_history[metric_name])
                    ax.plot(steps, values, label=metric_name)
            
            ax.set_title(group_name)
            ax.set_xlabel('Step')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        report = {
            'final_metrics': {},
            'best_metrics': {},
            'training_summary': {}
        }
        
        # 最终指标
        for metric_name, history in self.metrics_history.items():
            if history:
                report['final_metrics'][metric_name] = history[-1][1]
        
        # 最佳指标
        for metric_name, history in self.metrics_history.items():
            if 'accuracy' in metric_name.lower():
                best_value = max(history, key=lambda x: x[1])
            elif 'loss' in metric_name.lower():
                best_value = min(history, key=lambda x: x[1])
            else:
                continue
            
            report['best_metrics'][metric_name] = {
                'value': best_value[1],
                'step': best_value[0]
            }
        
        return report
```

### 实时监控仪表板

```python
# scripts/monitoring_dashboard.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path

def create_dashboard():
    st.title("TriModalFusion 训练监控")
    
    # 侧边栏配置
    st.sidebar.title("配置")
    log_dir = st.sidebar.text_input("日志目录", "./logs")
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", 1, 30, 5)
    
    # 自动刷新
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            # 读取最新日志
            metrics = load_latest_metrics(log_dir)
            
            if metrics:
                # 创建子图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('损失曲线', '准确率曲线', '学习率', '其他指标')
                )
                
                # 绘制损失曲线
                if 'train_loss' in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=metrics['train_loss']['steps'],
                            y=metrics['train_loss']['values'],
                            name='训练损失'
                        ),
                        row=1, col=1
                    )
                
                if 'val_loss' in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=metrics['val_loss']['steps'],
                            y=metrics['val_loss']['values'],
                            name='验证损失'
                        ),
                        row=1, col=1
                    )
                
                # 绘制准确率曲线
                if 'train_accuracy' in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=metrics['train_accuracy']['steps'],
                            y=metrics['train_accuracy']['values'],
                            name='训练准确率'
                        ),
                        row=1, col=2
                    )
                
                if 'val_accuracy' in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=metrics['val_accuracy']['steps'],
                            y=metrics['val_accuracy']['values'],
                            name='验证准确率'
                        ),
                        row=1, col=2
                    )
                
                # 显示图表
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示最新指标
                st.subheader("最新指标")
                cols = st.columns(4)
                
                latest_metrics = get_latest_values(metrics)
                for idx, (name, value) in enumerate(latest_metrics.items()):
                    with cols[idx % 4]:
                        st.metric(name, f"{value:.4f}")
            
            else:
                st.warning("未找到训练日志")
        
        time.sleep(refresh_interval)

def load_latest_metrics(log_dir: str) -> Dict:
    """加载最新的训练指标"""
    log_path = Path(log_dir) / "metrics.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {}

def get_latest_values(metrics: Dict) -> Dict[str, float]:
    """获取最新的指标值"""
    latest = {}
    for name, data in metrics.items():
        if 'values' in data and data['values']:
            latest[name] = data['values'][-1]
    return latest

if __name__ == "__main__":
    create_dashboard()
```

## 故障排除

### 常见问题

#### 1. 内存不足 (OOM)

```python
# 解决方案
config.training.batch_size = 16  # 减小批次大小
config.training.accumulate_grad_batches = 2  # 使用梯度累积
config.system.mixed_precision.enabled = True  # 启用混合精度
config.system.memory_optimization.gradient_checkpointing = True  # 梯度检查点
```

#### 2. 梯度爆炸/消失

```python
# 梯度裁剪
config.training.gradient_clip_val = 1.0
config.training.gradient_clip_algorithm = "norm"

# 学习率调整
config.training.learning_rate = 1e-5  # 降低学习率

# 权重初始化
config.model.initializer_range = 0.01  # 较小的初始化范围
```

#### 3. 训练不稳定

```python
# 标签平滑
config.training.loss_config.classification.label_smoothing = 0.1

# Dropout
config.model.dropout = 0.1

# 批归一化
config.speech_config.use_batch_norm = True
config.gesture_config.use_batch_norm = True
```

### 调试工具

```python
# src/utils/debug_utils.py
import torch
import matplotlib.pyplot as plt

def check_gradients(model):
    """检查模型梯度"""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[name] = grad_norm
        else:
            grad_norms[name] = 0.0
    
    return grad_norms

def visualize_attention_weights(attention_weights, save_path=None):
    """可视化注意力权重"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Weights')
    
    plt.colorbar(im)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def profile_model(model, inputs, num_runs=100):
    """性能分析"""
    model.eval()
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(inputs)
    
    # 计时
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(inputs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time
```

## 最佳实践

### 1. 训练策略

```python
# 渐进式训练
def progressive_training(config):
    """渐进式训练策略"""
    
    # 阶段1：单模态预训练
    for modality in ['speech', 'gesture', 'image']:
        single_config = config.copy()
        single_config.model.enabled_modalities = [modality]
        train_single_modality(single_config, modality)
    
    # 阶段2：双模态训练  
    modality_pairs = [
        ['speech', 'gesture'],
        ['speech', 'image'], 
        ['gesture', 'image']
    ]
    
    for pair in modality_pairs:
        dual_config = config.copy()
        dual_config.model.enabled_modalities = pair
        train_dual_modality(dual_config, pair)
    
    # 阶段3：三模态训练
    full_config = config.copy()
    full_config.model.enabled_modalities = ['speech', 'gesture', 'image']
    train_full_model(full_config)

# 课程学习
def curriculum_learning(config):
    """课程学习策略"""
    
    # 简单样本开始
    easy_config = config.copy()
    easy_config.data.difficulty_threshold = 0.3
    train_model(easy_config, epochs=10)
    
    # 逐渐增加难度
    for difficulty in [0.5, 0.7, 1.0]:
        curr_config = config.copy()
        curr_config.data.difficulty_threshold = difficulty
        train_model(curr_config, epochs=10)
```

### 2. 数据策略

```python
# 数据平衡
def balance_dataset(dataset):
    """数据集平衡"""
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    
    # 统计类别分布
    labels = [sample['label'] for sample in dataset]
    class_counts = Counter(labels)
    
    # 计算权重
    weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [weights[label] for label in labels]
    
    # 创建采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler

# 在线难例挖掘
def online_hard_negative_mining(outputs, targets, ratio=0.3):
    """在线难例挖掘"""
    with torch.no_grad():
        # 计算损失
        losses = F.cross_entropy(outputs, targets, reduction='none')
        
        # 选择最难的样本
        num_hard = int(len(losses) * ratio)
        _, hard_indices = torch.topk(losses, num_hard)
        
    return hard_indices
```

### 3. 模型优化

```python
# 知识蒸馏
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, inputs, targets):
        # 学生预测
        student_outputs = self.student(inputs)
        
        # 教师预测
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        # 蒸馏损失
        distill_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 标准损失
        standard_loss = F.cross_entropy(student_outputs, targets)
        
        # 组合损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * standard_loss
        
        return total_loss

# 模型剪枝
def prune_model(model, pruning_ratio=0.2):
    """模型剪枝"""
    import torch.nn.utils.prune as prune
    
    # 收集所有线性层
    modules_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            modules_to_prune.append((module, 'weight'))
    
    # 全局剪枝
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio
    )
    
    # 永久移除剪枝连接
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    return model
```

这个训练指南提供了完整的训练流程，从数据准备到模型部署的每个环节都有详细说明，确保能够成功训练和优化TriModalFusion模型。
