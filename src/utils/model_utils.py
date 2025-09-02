"""
Model utilities for TriModalFusion.

This module provides utility functions for model operations such as
parameter counting, activation functions, and model manipulation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_activation_function(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Name of activation function
        
    Returns:
        Activation function module
    """
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'swish': nn.SiLU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': lambda: nn.LeakyReLU(0.1),
        'elu': nn.ELU,
        'prelu': nn.PReLU,
        'mish': lambda: nn.Mish() if hasattr(nn, 'Mish') else nn.SiLU(),
        'none': nn.Identity,
        'identity': nn.Identity
    }
    
    if activation.lower() not in activations:
        logger.warning(f"Unknown activation function: {activation}. Using ReLU instead.")
        return nn.ReLU
    
    return activations[activation.lower()]


def freeze_model(model: nn.Module):
    """
    Freeze all parameters in a model.
    
    Args:
        model: PyTorch model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    logger.info(f"Frozen model: {model.__class__.__name__}")


def unfreeze_model(model: nn.Module):
    """
    Unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info(f"Unfrozen model: {model.__class__.__name__}")


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break
    
    logger.info(f"Frozen layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: list):
    """
    Unfreeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break
    
    logger.info(f"Unfrozen layers: {layer_names}")


def get_model_summary(model: nn.Module, input_size: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size for calculating forward pass (optional)
        
    Returns:
        Dictionary containing model summary information
    """
    summary = {
        'model_name': model.__class__.__name__,
        'total_params': count_parameters(model, trainable_only=False),
        'trainable_params': count_parameters(model, trainable_only=True),
        'non_trainable_params': count_parameters(model, trainable_only=False) - count_parameters(model, trainable_only=True),
    }
    
    # Layer information
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'params': count_parameters(module, trainable_only=False)
            }
            layers.append(layer_info)
    
    summary['layers'] = layers
    summary['num_layers'] = len(layers)
    
    return summary


def initialize_weights(model: nn.Module, init_type: str = 'xavier_uniform'):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """
    def init_func(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            else:
                logger.warning(f"Unknown initialization type: {init_type}")
                return
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_func)
    logger.info(f"Initialized model weights with {init_type}")


def calculate_receptive_field(model: nn.Module) -> Dict[str, int]:
    """
    Calculate receptive field for convolutional layers.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to receptive field sizes
    """
    receptive_fields = {}
    
    def calculate_rf(module, name=""):
        if isinstance(module, nn.Conv1d):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            
            # Simplified RF calculation (assumes no dilation)
            rf = kernel_size
            receptive_fields[name or module.__class__.__name__] = rf
    
    for name, module in model.named_modules():
        calculate_rf(module, name)
    
    return receptive_fields


def get_gradient_norm(model: nn.Module) -> float:
    """
    Calculate the norm of gradients in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def clip_gradients(model: nn.Module, max_norm: float = 1.0, norm_type: float = 2.0):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm for gradients
        norm_type: Type of norm to use
    """
    nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)


def get_learning_rate_schedule(optimizer: torch.optim.Optimizer) -> Optional[float]:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return None


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate for optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_device_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {'gpu_available': False}
    
    info = {
        'gpu_available': True,
        'num_gpus': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
    }
    
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        
        info[f'gpu_{i}'] = {
            'name': torch.cuda.get_device_name(i),
            'memory_allocated_gb': memory_allocated,
            'memory_cached_gb': memory_cached,
            'memory_total_gb': memory_total,
            'memory_usage_percent': (memory_allocated / memory_total) * 100
        }
    
    return info


def move_to_device(batch: Union[torch.Tensor, Dict, list, tuple], device: torch.device):
    """
    Recursively move tensors to device.
    
    Args:
        batch: Batch of data (tensor, dict, list, or tuple)
        device: Target device
        
    Returns:
        Batch moved to device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(item, device) for item in batch)
    else:
        return batch


def warmup_learning_rate(optimizer: torch.optim.Optimizer, 
                        current_step: int, 
                        warmup_steps: int, 
                        base_lr: float,
                        warmup_method: str = 'linear'):
    """
    Apply learning rate warmup.
    
    Args:
        optimizer: PyTorch optimizer
        current_step: Current training step
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate after warmup
        warmup_method: Warmup method ('linear' or 'cosine')
    """
    if current_step < warmup_steps:
        if warmup_method == 'linear':
            lr = base_lr * (current_step / warmup_steps)
        elif warmup_method == 'cosine':
            lr = base_lr * (1 - torch.cos(torch.tensor(current_step / warmup_steps * 3.14159))) / 2
        else:
            lr = base_lr
            
        set_learning_rate(optimizer, lr)
    else:
        set_learning_rate(optimizer, base_lr)


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}