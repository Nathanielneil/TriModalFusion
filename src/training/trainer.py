import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import wandb

from ..models.trimodal_fusion import TriModalFusionModel
from ..evaluation.metrics import MetricsCalculator
from ..utils.config import Config
from .optimizers import get_optimizer, get_scheduler

logger = logging.getLogger(__name__)


class TriModalTrainer:
    """
    TriModalFusion模型训练器
    
    支持多GPU训练、混合精度、梯度累积等高级训练功能
    """
    
    def __init__(
        self,
        config: Config,
        model: TriModalFusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 设备和分布式训练
        self.device = self._setup_device()
        self.model = self._setup_model_parallel()
        
        # 优化器和调度器
        self.optimizer = get_optimizer(self.model, config.training)
        self.scheduler = get_scheduler(self.optimizer, config.training)
        
        # 损失函数
        self.criterion = self._setup_criterion()
        
        # 混合精度训练
        self.use_amp = config.training.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('-inf')
        self.patience_counter = 0
        
        # 监控和日志
        self.metrics_calculator = MetricsCalculator(config)
        self._setup_logging()
        
        # 模型保存
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 恢复训练
        if resume_from:
            self.resume_from_checkpoint(resume_from)
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if torch.cuda.is_available() and self.config.system.device != 'cpu':
            device = torch.device('cuda')
            logger.info(f"使用GPU训练: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU训练")
        
        return device
    
    def _setup_model_parallel(self) -> nn.Module:
        """设置模型并行"""
        model = self.model.to(self.device)
        
        # 数据并行
        if torch.cuda.device_count() > 1 and self.config.training.get('data_parallel', True):
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
            model = nn.DataParallel(model)
        
        # 模型编译 (PyTorch 2.0+)
        if self.config.system.get('compile_model', False) and hasattr(torch, 'compile'):
            logger.info("编译模型以加速训练")
            model = torch.compile(model)
        
        return model
    
    def _setup_criterion(self) -> Dict[str, nn.Module]:
        """设置损失函数"""
        criterion = {}
        
        # 分类损失
        if 'classification' in self.config.model.tasks:
            if self.config.training.get('label_smoothing', 0.0) > 0:
                criterion['classification'] = nn.CrossEntropyLoss(
                    label_smoothing=self.config.training.label_smoothing
                )
            else:
                criterion['classification'] = nn.CrossEntropyLoss()
        
        # 检测损失
        if 'detection' in self.config.model.tasks:
            # 这里需要实现YOLO损失或其他检测损失
            criterion['detection'] = self._get_detection_loss()
        
        # 回归损失
        if 'regression' in self.config.model.tasks:
            criterion['regression'] = nn.MSELoss()
        
        # 对比学习损失 (用于跨模态对齐)
        if self.config.fusion_config.get('use_contrastive_loss', False):
            criterion['contrastive'] = self._get_contrastive_loss()
        
        return criterion
    
    def _get_detection_loss(self):
        """获取检测损失函数"""
        # 简化的检测损失实现
        class DetectionLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bbox_loss = nn.SmoothL1Loss()
                self.cls_loss = nn.CrossEntropyLoss()
                self.obj_loss = nn.BCEWithLogitsLoss()
            
            def forward(self, predictions, targets):
                # 简化实现，实际需要根据检测架构调整
                total_loss = 0.0
                if 'bbox' in predictions and 'bbox' in targets:
                    total_loss += self.bbox_loss(predictions['bbox'], targets['bbox'])
                if 'classes' in predictions and 'classes' in targets:
                    total_loss += self.cls_loss(predictions['classes'], targets['classes'])
                if 'objectness' in predictions and 'objectness' in targets:
                    total_loss += self.obj_loss(predictions['objectness'], targets['objectness'])
                return total_loss
        
        return DetectionLoss()
    
    def _get_contrastive_loss(self):
        """获取对比学习损失"""
        class ContrastiveLoss(nn.Module):
            def __init__(self, temperature=0.07):
                super().__init__()
                self.temperature = temperature
                self.cosine_sim = nn.CosineSimilarity(dim=-1)
            
            def forward(self, features_a, features_b):
                batch_size = features_a.size(0)
                
                # 计算余弦相似度矩阵
                sim_matrix = torch.mm(features_a, features_b.t()) / self.temperature
                
                # 正样本标签
                labels = torch.arange(batch_size).to(features_a.device)
                
                # 计算对比损失
                loss = nn.CrossEntropyLoss()(sim_matrix, labels)
                return loss
        
        return ContrastiveLoss()
    
    def _setup_logging(self):
        """设置实验跟踪和日志"""
        if self.config.training.get('use_wandb', False):
            wandb.init(
                project=self.config.training.get('project_name', 'trimodal-fusion'),
                config=self.config.to_dict(),
                name=self.config.training.get('experiment_name', f'run_{int(time.time())}'),
                resume='allow' if self.config.training.get('resume_wandb', False) else None
            )
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        logger.info(f"批次大小: {self.config.training.batch_size}")
        logger.info(f"最大训练轮数: {self.config.training.max_epochs}")
        
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self.train_epoch()
            
            # 验证阶段
            val_metrics = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # 记录指标
            self.log_metrics(train_metrics, val_metrics, epoch)
            
            # 保存检查点
            is_best = val_metrics[self.config.training.get('monitor_metric', 'accuracy')] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics[self.config.training.get('monitor_metric', 'accuracy')]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            # 早停
            if (self.config.training.get('early_stopping_patience', 0) > 0 and 
                self.patience_counter >= self.config.training.early_stopping_patience):
                logger.info(f"在第{epoch+1}轮触发早停")
                break
        
        logger.info("训练完成!")
        
        # 最终测试
        if self.test_loader is not None:
            self.test()
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个轮次"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {}
        num_batches = len(self.train_loader)
        
        # 初始化任务损失跟踪
        for task in self.config.model.tasks:
            task_losses[task] = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'训练 Epoch {self.current_epoch + 1}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 数据移动到设备
            inputs = self.move_to_device(inputs)
            targets = self.move_to_device(targets)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss, task_loss_dict = self.compute_loss(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss, task_loss_dict = self.compute_loss(outputs, targets)
            
            # 梯度累积
            if self.config.training.get('gradient_accumulation_steps', 1) > 1:
                loss = loss / self.config.training.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 梯度累积检查
                if (batch_idx + 1) % self.config.training.get('gradient_accumulation_steps', 1) == 0:
                    # 梯度裁剪
                    if self.config.training.get('max_grad_norm', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.training.max_grad_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
            else:
                loss.backward()
                
                # 梯度累积检查
                if (batch_idx + 1) % self.config.training.get('gradient_accumulation_steps', 1) == 0:
                    # 梯度裁剪
                    if self.config.training.get('max_grad_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.training.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.global_step += 1
            
            # 损失累积
            total_loss += loss.item()
            for task, task_loss in task_loss_dict.items():
                task_losses[task] += task_loss.item()
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # 记录训练步骤
            if self.global_step % self.config.training.get('log_every', 100) == 0:
                self.log_training_step(loss.item(), task_loss_dict, current_lr)
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        return {
            'loss': avg_loss,
            **avg_task_losses
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='验证')
            
            for inputs, targets in progress_bar:
                # 数据移动到设备
                inputs = self.move_to_device(inputs)
                targets = self.move_to_device(targets)
                
                # 前向传播
                outputs = self.model(inputs)
                loss, _ = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # 收集预测和目标
                all_outputs.append(outputs)
                all_targets.append(targets)
                
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        # 计算验证指标
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calculator.compute_metrics(all_outputs, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """测试模型"""
        if self.test_loader is None:
            logger.warning("没有提供测试数据集")
            return {}
        
        logger.info("开始测试...")
        
        # 加载最佳模型
        best_checkpoint = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint.exists():
            self.load_checkpoint(str(best_checkpoint))
            logger.info("已加载最佳模型进行测试")
        
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='测试')
            
            for inputs, targets in progress_bar:
                # 数据移动到设备
                inputs = self.move_to_device(inputs)
                targets = self.move_to_device(targets)
                
                # 前向传播
                outputs = self.model(inputs)
                loss, _ = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # 收集预测和目标
                all_outputs.append(outputs)
                all_targets.append(targets)
                
                progress_bar.set_postfix({'test_loss': f'{loss.item():.4f}'})
        
        # 计算测试指标
        avg_loss = total_loss / len(self.test_loader)
        test_metrics = self.metrics_calculator.compute_metrics(all_outputs, all_targets)
        test_metrics['loss'] = avg_loss
        
        logger.info("测试完成!")
        for metric, value in test_metrics.items():
            logger.info(f"测试 {metric}: {value:.4f}")
        
        return test_metrics
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算损失"""
        total_loss = 0.0
        task_losses = {}
        
        # 任务特定损失
        task_outputs = outputs.get('task_outputs', {})
        
        for task in self.config.model.tasks:
            if task in task_outputs and task in targets:
                if task in self.criterion:
                    task_loss = self.criterion[task](task_outputs[task], targets[task])
                    task_weight = self.config.training.get(f'{task}_loss_weight', 1.0)
                    weighted_loss = task_loss * task_weight
                    
                    total_loss += weighted_loss
                    task_losses[task] = task_loss
        
        # 对比学习损失
        if 'contrastive' in self.criterion and 'encoded_features' in outputs:
            features = outputs['encoded_features']
            if len(features) >= 2:
                modalities = list(features.keys())
                contrastive_loss = 0.0
                pairs = 0
                
                for i, mod_a in enumerate(modalities):
                    for mod_b in modalities[i+1:]:
                        contrastive_loss += self.criterion['contrastive'](
                            features[mod_a], features[mod_b]
                        )
                        pairs += 1
                
                if pairs > 0:
                    contrastive_loss /= pairs
                    contrastive_weight = self.config.training.get('contrastive_loss_weight', 0.1)
                    total_loss += contrastive_loss * contrastive_weight
                    task_losses['contrastive'] = contrastive_loss
        
        return total_loss, task_losses
    
    def move_to_device(self, data):
        """将数据移动到设备"""
        if isinstance(data, dict):
            return {key: self.move_to_device(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self.move_to_device(item) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        else:
            return data
    
    def log_training_step(self, loss: float, task_losses: Dict[str, torch.Tensor], lr: float):
        """记录训练步骤"""
        log_dict = {
            'train/loss': loss,
            'train/learning_rate': lr,
            'train/global_step': self.global_step,
            'train/epoch': self.current_epoch
        }
        
        for task, task_loss in task_losses.items():
            log_dict[f'train/{task}_loss'] = task_loss.item()
        
        if self.config.training.get('use_wandb', False):
            wandb.log(log_dict, step=self.global_step)
    
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """记录训练和验证指标"""
        logger.info(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
        logger.info(f"训练损失: {train_metrics['loss']:.4f}")
        logger.info(f"验证损失: {val_metrics['loss']:.4f}")
        
        if self.config.training.get('use_wandb', False):
            log_dict = {}
            
            # 训练指标
            for key, value in train_metrics.items():
                log_dict[f'train/{key}'] = value
            
            # 验证指标
            for key, value in val_metrics.items():
                log_dict[f'val/{key}'] = value
            
            log_dict['epoch'] = epoch
            
            wandb.log(log_dict, step=self.global_step)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config.to_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None
        }
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型到: {best_path}")
        
        # 定期保存
        if (epoch + 1) % self.config.training.get('save_every', 10) == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch + 1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        logger.info(f"从 {checkpoint_path} 恢复训练")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载混合精度状态
        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复训练状态
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))
        
        logger.info(f"恢复训练从第 {self.current_epoch} 轮开始")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练"""
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            logger.warning(f"检查点文件不存在: {checkpoint_path}")