import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional
import math


def get_optimizer(model: torch.nn.Module, training_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    根据配置创建优化器
    
    Args:
        model: 要优化的模型
        training_config: 训练配置
    
    Returns:
        优化器实例
    """
    optimizer_type = training_config.get('optimizer', 'adamw').lower()
    learning_rate = training_config.get('learning_rate', 1e-4)
    weight_decay = training_config.get('weight_decay', 1e-2)
    
    # 获取模型参数，可以设置不同的学习率
    param_groups = _get_parameter_groups(model, training_config)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=training_config.get('adam_betas', (0.9, 0.999)),
            eps=training_config.get('adam_eps', 1e-8)
        )
    
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=training_config.get('adamw_betas', (0.9, 0.999)),
            eps=training_config.get('adamw_eps', 1e-8)
        )
    
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=training_config.get('sgd_momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=training_config.get('sgd_nesterov', True)
        )
    
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=training_config.get('rmsprop_alpha', 0.99),
            eps=training_config.get('rmsprop_eps', 1e-8)
        )
    
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=training_config.get('adagrad_eps', 1e-10)
        )
    
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, training_config: Dict[str, Any]) -> Optional[object]:
    """
    根据配置创建学习率调度器
    
    Args:
        optimizer: 优化器
        training_config: 训练配置
    
    Returns:
        学习率调度器实例或None
    """
    scheduler_type = training_config.get('scheduler', None)
    
    if scheduler_type is None:
        return None
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=training_config.get('step_size', 10),
            gamma=training_config.get('step_gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=training_config.get('exp_gamma', 0.95)
        )
    
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('cosine_t_max', training_config.get('max_epochs', 100)),
            eta_min=training_config.get('cosine_eta_min', 1e-6)
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=training_config.get('plateau_mode', 'min'),
            factor=training_config.get('plateau_factor', 0.5),
            patience=training_config.get('plateau_patience', 5),
            verbose=training_config.get('plateau_verbose', True),
            min_lr=training_config.get('plateau_min_lr', 1e-6)
        )
    
    elif scheduler_type == 'warmup':
        return WarmupScheduler(
            optimizer,
            warmup_epochs=training_config.get('warmup_epochs', 5),
            total_epochs=training_config.get('max_epochs', 100),
            warmup_method=training_config.get('warmup_method', 'linear')
        )
    
    elif scheduler_type == 'cosine_warmup':
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs=training_config.get('warmup_epochs', 10),
            total_epochs=training_config.get('max_epochs', 100),
            eta_min=training_config.get('cosine_eta_min', 1e-6)
        )
    
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def _get_parameter_groups(model: torch.nn.Module, training_config: Dict[str, Any]) -> list:
    """
    获取模型参数组，支持不同层设置不同学习率
    
    Args:
        model: 模型
        training_config: 训练配置
    
    Returns:
        参数组列表
    """
    # 默认参数组
    param_groups = [{'params': list(model.parameters())}]
    
    # 检查是否有特殊的学习率设置
    if 'parameter_groups' in training_config:
        param_groups = []
        used_params = set()
        
        # 处理特殊参数组
        for group_config in training_config['parameter_groups']:
            group_params = []
            
            # 根据名称匹配参数
            for name, param in model.named_parameters():
                if any(pattern in name for pattern in group_config.get('name_patterns', [])):
                    group_params.append(param)
                    used_params.add(id(param))
            
            if group_params:
                group = {'params': group_params}
                
                # 设置组特定的超参数
                if 'lr' in group_config:
                    group['lr'] = group_config['lr']
                if 'weight_decay' in group_config:
                    group['weight_decay'] = group_config['weight_decay']
                
                param_groups.append(group)
        
        # 添加剩余参数到默认组
        remaining_params = [param for param in model.parameters() if id(param) not in used_params]
        if remaining_params:
            param_groups.append({'params': remaining_params})
    
    return param_groups


class WarmupScheduler:
    """
    学习率预热调度器
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_method='linear'):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_method = warmup_method
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段
            if self.warmup_method == 'linear':
                warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            elif self.warmup_method == 'exponential':
                warmup_factor = self.current_epoch / self.warmup_epochs
                warmup_factor = warmup_factor ** 2
            else:
                warmup_factor = 1.0
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        
        self.current_epoch += 1


class CosineWarmupScheduler:
    """
    带预热的余弦退火调度器
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # 余弦退火阶段
            cosine_epoch = self.current_epoch - self.warmup_epochs
            cosine_total = self.total_epochs - self.warmup_epochs
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                                   (1 + math.cos(math.pi * cosine_epoch / cosine_total)) / 2
        
        self.current_epoch += 1


class LinearWarmupCosineAnnealingLR:
    """
    线性预热 + 余弦退火学习率调度器
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
    
    def state_dict(self):
        return {'last_epoch': self.last_epoch}
    
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
    
    def step(self):
        self.last_epoch += 1
        
        if self.last_epoch <= self.warmup_epochs:
            # 线性预热
            lr_scale = self.last_epoch / self.warmup_epochs
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.last_epoch <= self.warmup_epochs:
                param_group['lr'] = base_lr * lr_scale
            else:
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * lr_scale


def get_linear_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs, eta_min=0):
    """
    获取线性预热余弦退火调度器的工厂函数
    """
    return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, eta_min)