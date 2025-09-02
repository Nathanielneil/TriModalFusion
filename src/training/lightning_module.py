import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, List, Optional, Any, Tuple
import torchmetrics

from ..models.trimodal_fusion import TriModalFusionModel
from ..evaluation.metrics import MetricsCalculator
from .optimizers import get_optimizer, get_scheduler


class TriModalLightningModule(L.LightningModule):
    """
    TriModalFusion的PyTorch Lightning模块
    
    这个模块封装了训练、验证、测试逻辑，并自动处理
    分布式训练、检查点、日志记录等功能
    """
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # 创建模型
        self.model = TriModalFusionModel(config)
        
        # 损失函数
        self.criterion = self._setup_criterion()
        
        # 评估指标
        self.metrics_calculator = MetricsCalculator(config)
        
        # 设置torchmetrics指标
        self._setup_torchmetrics()
        
        # 训练状态
        self.validation_outputs = []
        self.test_outputs = []
    
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
            criterion['detection'] = self._get_detection_loss()
        
        # 回归损失
        if 'regression' in self.config.model.tasks:
            criterion['regression'] = nn.MSELoss()
        
        # 对比学习损失
        if self.config.fusion_config.get('use_contrastive_loss', False):
            criterion['contrastive'] = self._get_contrastive_loss()
        
        return criterion
    
    def _get_detection_loss(self):
        """获取检测损失函数"""
        class DetectionLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bbox_loss = nn.SmoothL1Loss()
                self.cls_loss = nn.CrossEntropyLoss()
                self.obj_loss = nn.BCEWithLogitsLoss()
            
            def forward(self, predictions, targets):
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
            
            def forward(self, features_a, features_b):
                batch_size = features_a.size(0)
                sim_matrix = torch.mm(features_a, features_b.t()) / self.temperature
                labels = torch.arange(batch_size).to(features_a.device)
                loss = nn.CrossEntropyLoss()(sim_matrix, labels)
                return loss
        
        return ContrastiveLoss()
    
    def _setup_torchmetrics(self):
        """设置torchmetrics指标"""
        num_classes = self.config.model.get('num_classes', 10)
        
        # 分类指标
        if 'classification' in self.config.model.tasks:
            self.train_accuracy = torchmetrics.Accuracy(
                task='multiclass', num_classes=num_classes
            )
            self.val_accuracy = torchmetrics.Accuracy(
                task='multiclass', num_classes=num_classes
            )
            self.test_accuracy = torchmetrics.Accuracy(
                task='multiclass', num_classes=num_classes
            )
            
            self.train_f1 = torchmetrics.F1Score(
                task='multiclass', num_classes=num_classes, average='macro'
            )
            self.val_f1 = torchmetrics.F1Score(
                task='multiclass', num_classes=num_classes, average='macro'
            )
            self.test_f1 = torchmetrics.F1Score(
                task='multiclass', num_classes=num_classes, average='macro'
            )
        
        # 回归指标
        if 'regression' in self.config.model.tasks:
            self.train_mse = torchmetrics.MeanSquaredError()
            self.val_mse = torchmetrics.MeanSquaredError()
            self.test_mse = torchmetrics.MeanSquaredError()
            
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.test_mae = torchmetrics.MeanAbsoluteError()
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
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
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        inputs, targets = batch
        outputs = self.forward(inputs)
        
        loss, task_losses = self.compute_loss(outputs, targets)
        
        # 记录损失
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for task, task_loss in task_losses.items():
            self.log(f'train/{task}_loss', task_loss, on_step=True, on_epoch=True)
        
        # 更新指标
        if 'classification' in self.config.model.tasks and 'classification' in targets:
            logits = outputs['task_outputs']['classification']
            preds = torch.argmax(logits, dim=-1)
            self.train_accuracy(preds, targets['classification'])
            self.train_f1(preds, targets['classification'])
            
            self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train/f1', self.train_f1, on_step=False, on_epoch=True)
        
        if 'regression' in self.config.model.tasks and 'regression' in targets:
            predictions = outputs['task_outputs']['regression']
            self.train_mse(predictions, targets['regression'])
            self.train_mae(predictions, targets['regression'])
            
            self.log('train/mse', self.train_mse, on_step=False, on_epoch=True)
            self.log('train/mae', self.train_mae, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs, targets = batch
        outputs = self.forward(inputs)
        
        loss, task_losses = self.compute_loss(outputs, targets)
        
        # 记录损失
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for task, task_loss in task_losses.items():
            self.log(f'val/{task}_loss', task_loss, on_step=False, on_epoch=True)
        
        # 更新指标
        if 'classification' in self.config.model.tasks and 'classification' in targets:
            logits = outputs['task_outputs']['classification']
            preds = torch.argmax(logits, dim=-1)
            self.val_accuracy(preds, targets['classification'])
            self.val_f1(preds, targets['classification'])
            
            self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val/f1', self.val_f1, on_step=False, on_epoch=True)
        
        if 'regression' in self.config.model.tasks and 'regression' in targets:
            predictions = outputs['task_outputs']['regression']
            self.val_mse(predictions, targets['regression'])
            self.val_mae(predictions, targets['regression'])
            
            self.log('val/mse', self.val_mse, on_step=False, on_epoch=True)
            self.log('val/mae', self.val_mae, on_step=False, on_epoch=True)
        
        # 保存输出用于额外计算
        self.validation_outputs.append({
            'outputs': outputs,
            'targets': targets,
            'loss': loss
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """验证轮次结束"""
        if self.validation_outputs:
            # 计算额外的验证指标
            all_outputs = [item['outputs'] for item in self.validation_outputs]
            all_targets = [item['targets'] for item in self.validation_outputs]
            
            # 使用自定义指标计算器
            additional_metrics = self.metrics_calculator.compute_metrics(all_outputs, all_targets)
            
            # 记录额外指标
            for metric_name, metric_value in additional_metrics.items():
                if metric_name != 'loss':  # 损失已经记录过了
                    self.log(f'val/{metric_name}', metric_value)
            
            # 清空输出
            self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        inputs, targets = batch
        outputs = self.forward(inputs)
        
        loss, task_losses = self.compute_loss(outputs, targets)
        
        # 记录损失
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        for task, task_loss in task_losses.items():
            self.log(f'test/{task}_loss', task_loss, on_step=False, on_epoch=True)
        
        # 更新指标
        if 'classification' in self.config.model.tasks and 'classification' in targets:
            logits = outputs['task_outputs']['classification']
            preds = torch.argmax(logits, dim=-1)
            self.test_accuracy(preds, targets['classification'])
            self.test_f1(preds, targets['classification'])
            
            self.log('test/accuracy', self.test_accuracy, on_step=False, on_epoch=True)
            self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        
        if 'regression' in self.config.model.tasks and 'regression' in targets:
            predictions = outputs['task_outputs']['regression']
            self.test_mse(predictions, targets['regression'])
            self.test_mae(predictions, targets['regression'])
            
            self.log('test/mse', self.test_mse, on_step=False, on_epoch=True)
            self.log('test/mae', self.test_mae, on_step=False, on_epoch=True)
        
        # 保存输出用于额外计算
        self.test_outputs.append({
            'outputs': outputs,
            'targets': targets,
            'loss': loss
        })
        
        return loss
    
    def on_test_epoch_end(self):
        """测试轮次结束"""
        if self.test_outputs:
            # 计算额外的测试指标
            all_outputs = [item['outputs'] for item in self.test_outputs]
            all_targets = [item['targets'] for item in self.test_outputs]
            
            # 使用自定义指标计算器
            additional_metrics = self.metrics_calculator.compute_metrics(all_outputs, all_targets)
            
            # 记录额外指标
            for metric_name, metric_value in additional_metrics.items():
                if metric_name != 'loss':
                    self.log(f'test/{metric_name}', metric_value)
            
            # 清空输出
            self.test_outputs.clear()
    
    def configure_optimizers(self):
        """配置优化器和调度器"""
        # 获取优化器
        optimizer = get_optimizer(self.model, self.config.training)
        
        # 获取调度器
        scheduler = get_scheduler(optimizer, self.config.training)
        
        if scheduler is None:
            return optimizer
        
        # 根据调度器类型配置
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',  # 'epoch' 或 'step'
        }
        
        # 特殊调度器配置
        if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in str(type(scheduler)):
            scheduler_config.update({
                'monitor': self.config.training.get('monitor_metric', 'val/loss'),
                'frequency': 1
            })
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """预测步骤"""
        inputs, _ = batch
        outputs = self.forward(inputs)
        
        # 返回预测结果
        predictions = {}
        
        if 'classification' in self.config.model.tasks:
            logits = outputs['task_outputs']['classification']
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(logits, dim=-1)
            
            predictions['classification'] = {
                'logits': logits,
                'probabilities': probabilities,
                'predicted_classes': predicted_classes
            }
        
        if 'detection' in self.config.model.tasks:
            predictions['detection'] = outputs['task_outputs']['detection']
        
        if 'regression' in self.config.model.tasks:
            predictions['regression'] = outputs['task_outputs']['regression']
        
        # 包含特征信息
        predictions['features'] = {
            'fused_features': outputs.get('fused_features'),
            'encoded_features': outputs.get('encoded_features')
        }
        
        return predictions
    
    def on_train_epoch_start(self):
        """训练轮次开始"""
        # 记录学习率
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr)
    
    def on_save_checkpoint(self, checkpoint):
        """保存检查点时的回调"""
        # 添加自定义信息到检查点
        checkpoint['config'] = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
        checkpoint['model_summary'] = str(self.model)
    
    def on_load_checkpoint(self, checkpoint):
        """加载检查点时的回调"""
        # 可以在这里进行一些恢复操作
        pass