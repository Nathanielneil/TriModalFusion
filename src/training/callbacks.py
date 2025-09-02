import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TriModalModelCheckpoint(ModelCheckpoint):
    """
    自定义模型检查点回调，专门为TriModalFusion优化
    """
    
    def __init__(self, 
                 dirpath: Optional[str] = None,
                 filename: Optional[str] = None,
                 monitor: Optional[str] = 'val/loss',
                 verbose: bool = True,
                 save_last: bool = True,
                 save_top_k: int = 3,
                 mode: str = 'min',
                 save_weights_only: bool = False,
                 **kwargs):
        
        # 默认文件名模式
        if filename is None:
            filename = 'trimodal-{epoch:02d}-{val/loss:.4f}'
        
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            mode=mode,
            save_weights_only=save_weights_only,
            **kwargs
        )
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """保存检查点时添加额外信息"""
        # 添加模型配置
        checkpoint['model_config'] = pl_module.config
        
        # 添加训练统计信息
        checkpoint['training_stats'] = {
            'total_params': sum(p.numel() for p in pl_module.parameters()),
            'trainable_params': sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            'current_epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
        }
        
        # 记录最佳指标
        if hasattr(self, 'best_model_score'):
            checkpoint['best_score'] = self.best_model_score.item()


class MetricsLogger(Callback):
    """
    自定义指标记录器，记录详细的训练指标
    """
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics = []
        self.val_metrics = []
        self.lr_history = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        """训练轮次结束时记录指标"""
        # 获取当前轮次的指标
        train_metrics = trainer.callback_metrics
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # 记录学习率
        self.lr_history.append({
            'epoch': trainer.current_epoch,
            'lr': current_lr
        })
        
        # 记录训练指标
        epoch_train_metrics = {'epoch': trainer.current_epoch}
        for key, value in train_metrics.items():
            if key.startswith('train/') and torch.is_tensor(value):
                epoch_train_metrics[key] = value.item()
        
        if epoch_train_metrics:
            self.train_metrics.append(epoch_train_metrics)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证轮次结束时记录指标"""
        val_metrics = trainer.callback_metrics
        
        # 记录验证指标
        epoch_val_metrics = {'epoch': trainer.current_epoch}
        for key, value in val_metrics.items():
            if key.startswith('val/') and torch.is_tensor(value):
                epoch_val_metrics[key] = value.item()
        
        if epoch_val_metrics:
            self.val_metrics.append(epoch_val_metrics)
    
    def on_train_end(self, trainer, pl_module):
        """训练结束时保存所有指标"""
        # 保存指标到文件
        import json
        
        metrics_data = {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'lr_history': self.lr_history,
            'final_epoch': trainer.current_epoch,
            'total_steps': trainer.global_step
        }
        
        metrics_file = self.log_dir / f'metrics_{int(time.time())}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"训练指标已保存到: {metrics_file}")


class AttentionVisualizationCallback(Callback):
    """
    注意力权重可视化回调
    """
    
    def __init__(self, 
                 log_dir: str = 'attention_visualizations',
                 visualize_every_n_epochs: int = 10,
                 num_samples: int = 4):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.num_samples = num_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """在验证结束时可视化注意力"""
        if trainer.current_epoch % self.visualize_every_n_epochs != 0:
            return
        
        # 获取验证数据加载器
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            return
        
        # 获取一小批数据
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if batch_idx >= 1:  # 只处理第一批
                    break
                
                # 限制样本数量
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key][:self.num_samples].to(pl_module.device)
                
                # 前向传播获取注意力权重
                outputs = pl_module.forward(inputs)
                
                # 可视化注意力权重
                if 'attention_weights' in outputs:
                    self._visualize_attention_weights(
                        outputs['attention_weights'],
                        trainer.current_epoch,
                        inputs
                    )
                
                break
    
    def _visualize_attention_weights(self, attention_weights, epoch, inputs):
        """可视化注意力权重"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Attention Weights - Epoch {epoch}', fontsize=16)
            
            # 跨模态注意力
            if 'cross_modal' in attention_weights:
                cross_modal_attn = attention_weights['cross_modal']
                if torch.is_tensor(cross_modal_attn):
                    # 平均化头部和批次维度
                    attn_matrix = cross_modal_attn.mean(dim=0).mean(dim=0).cpu().numpy()
                    
                    sns.heatmap(attn_matrix, 
                              annot=True, 
                              fmt='.3f',
                              cmap='Blues',
                              ax=axes[0, 0])
                    axes[0, 0].set_title('Cross-Modal Attention')
                    axes[0, 0].set_xlabel('Key Position')
                    axes[0, 0].set_ylabel('Query Position')
            
            # 自注意力权重（如果有的话）
            modality_names = ['speech', 'gesture', 'image']
            plot_idx = 1
            
            for modality in modality_names:
                if f'{modality}_self_attention' in attention_weights and plot_idx < 4:
                    row, col = divmod(plot_idx, 2)
                    
                    self_attn = attention_weights[f'{modality}_self_attention']
                    if torch.is_tensor(self_attn):
                        # 获取第一个样本的第一个头
                        attn_matrix = self_attn[0, 0].cpu().numpy()
                        
                        sns.heatmap(attn_matrix,
                                  cmap='Reds',
                                  ax=axes[row, col])
                        axes[row, col].set_title(f'{modality.title()} Self-Attention')
                    
                    plot_idx += 1
            
            # 隐藏未使用的子图
            for i in range(plot_idx, 4):
                row, col = divmod(i, 2)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            save_path = self.log_dir / f'attention_epoch_{epoch}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 如果使用wandb，也记录到wandb
            if wandb.run is not None:
                wandb.log({
                    'attention_visualization': wandb.Image(str(save_path)),
                    'epoch': epoch
                })
            
            logger.info(f"注意力可视化已保存到: {save_path}")
            
        except Exception as e:
            logger.warning(f"注意力可视化失败: {e}")


class ModelStatsCallback(Callback):
    """
    模型统计信息回调
    """
    
    def __init__(self, log_every_n_epochs: int = 5):
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        """记录模型统计信息"""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # 计算梯度统计
        grad_norm = self._compute_grad_norm(pl_module)
        
        # 计算参数统计
        param_stats = self._compute_param_stats(pl_module)
        
        # 记录统计信息
        trainer.logger.log_metrics({
            'model_stats/grad_norm': grad_norm,
            'model_stats/param_mean': param_stats['mean'],
            'model_stats/param_std': param_stats['std'],
            'model_stats/param_min': param_stats['min'],
            'model_stats/param_max': param_stats['max']
        }, step=trainer.global_step)
    
    def _compute_grad_norm(self, model):
        """计算梯度范数"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _compute_param_stats(self, model):
        """计算参数统计信息"""
        all_params = []
        for param in model.parameters():
            all_params.extend(param.data.cpu().numpy().flatten())
        
        all_params = np.array(all_params)
        return {
            'mean': float(np.mean(all_params)),
            'std': float(np.std(all_params)),
            'min': float(np.min(all_params)),
            'max': float(np.max(all_params))
        }


class WandbCallback(Callback):
    """
    Weights & Biases集成回调
    """
    
    def __init__(self, project: str, name: Optional[str] = None, tags: Optional[list] = None):
        self.project = project
        self.name = name or f"trimodal_run_{int(time.time())}"
        self.tags = tags or []
    
    def on_train_start(self, trainer, pl_module):
        """训练开始时初始化wandb"""
        if wandb.run is None:
            wandb.init(
                project=self.project,
                name=self.name,
                tags=self.tags,
                config=pl_module.config if hasattr(pl_module, 'config') else {}
            )
            
            # 记录模型架构
            wandb.watch(pl_module, log='all', log_freq=100)
    
    def on_train_end(self, trainer, pl_module):
        """训练结束时完成wandb运行"""
        if wandb.run is not None:
            # 记录最终指标
            final_metrics = trainer.callback_metrics
            wandb.log({
                'final/' + k: v for k, v in final_metrics.items() 
                if torch.is_tensor(v)
            })
            
            wandb.finish()


def get_default_callbacks(config: Dict[str, Any]) -> list:
    """
    获取默认的回调函数列表
    
    Args:
        config: 训练配置
    
    Returns:
        回调函数列表
    """
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = TriModalModelCheckpoint(
        dirpath=config.training.get('checkpoint_dir', 'checkpoints'),
        monitor=config.training.get('monitor_metric', 'val/loss'),
        mode=config.training.get('monitor_mode', 'min'),
        save_top_k=config.training.get('save_top_k', 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    if config.training.get('early_stopping_patience', 0) > 0:
        early_stopping = EarlyStopping(
            monitor=config.training.get('monitor_metric', 'val/loss'),
            mode=config.training.get('monitor_mode', 'min'),
            patience=config.training.early_stopping_patience,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 指标记录
    metrics_logger = MetricsLogger(
        log_dir=config.training.get('log_dir', 'logs')
    )
    callbacks.append(metrics_logger)
    
    # 模型统计
    model_stats = ModelStatsCallback(
        log_every_n_epochs=config.training.get('log_stats_every', 5)
    )
    callbacks.append(model_stats)
    
    # 注意力可视化
    if config.training.get('visualize_attention', False):
        attention_viz = AttentionVisualizationCallback(
            visualize_every_n_epochs=config.training.get('visualize_every', 10)
        )
        callbacks.append(attention_viz)
    
    # Wandb集成
    if config.training.get('use_wandb', False):
        wandb_callback = WandbCallback(
            project=config.training.get('wandb_project', 'trimodal-fusion'),
            name=config.training.get('experiment_name'),
            tags=config.training.get('wandb_tags', [])
        )
        callbacks.append(wandb_callback)
    
    return callbacks