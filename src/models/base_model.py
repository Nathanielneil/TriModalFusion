"""
Base model class for multimodal systems.

This module provides the abstract base class that all multimodal models should inherit from,
defining common interfaces and utilities.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BaseMultiModalModel(nn.Module, ABC):
    """
    Abstract base class for multimodal models.
    
    This class defines the common interface that all multimodal models should implement,
    including forward pass, loss computation, and evaluation methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base multimodal model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.modalities = config.get('modalities', ['speech', 'gesture', 'image'])
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multimodal model.
        
        Args:
            inputs: Dictionary containing input tensors for each modality
                   Keys: modality names ('speech', 'gesture', 'image')
                   Values: corresponding input tensors
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def compute_loss(self, 
                    outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.
        
        Args:
            outputs: Model outputs from forward pass
            targets: Ground truth targets
            
        Returns:
            Dictionary containing computed losses
        """
        pass
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including architecture details."""
        return {
            'model_name': self.__class__.__name__,
            'num_parameters': self.get_num_parameters(),
            'modalities': self.modalities,
            'device': self.device,
            'config': self.config
        }
    
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None, 
                       epoch: Optional[int] = None, metrics: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            optimizer_state: Optimizer state dict
            epoch: Current training epoch
            metrics: Training metrics
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_info': self.get_model_info()
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[str] = None):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            device: Device to load the model on
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        # Create model instance
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device:
            model = model.to(device)
            
        # Extract additional info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict')
        }
        
        logger.info(f"Model loaded from {filepath}")
        return model, checkpoint_info
    
    def freeze_encoder(self, modality: str):
        """Freeze parameters of a specific encoder."""
        encoder_name = f"{modality}_encoder"
        if hasattr(self, encoder_name):
            encoder = getattr(self, encoder_name)
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info(f"Frozen {modality} encoder parameters")
    
    def unfreeze_encoder(self, modality: str):
        """Unfreeze parameters of a specific encoder."""
        encoder_name = f"{modality}_encoder"
        if hasattr(self, encoder_name):
            encoder = getattr(self, encoder_name)
            for param in encoder.parameters():
                param.requires_grad = True
            logger.info(f"Unfrozen {modality} encoder parameters")
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if hasattr(self, 'gradient_checkpointing_enable'):
            self.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def get_parameter_groups(self, learning_rates: Dict[str, float]) -> List[Dict]:
        """
        Get parameter groups for different learning rates.
        
        Args:
            learning_rates: Dictionary mapping component names to learning rates
            
        Returns:
            List of parameter group dictionaries for optimizer
        """
        param_groups = []
        
        for component, lr in learning_rates.items():
            if hasattr(self, component):
                module = getattr(self, component)
                param_groups.append({
                    'params': module.parameters(),
                    'lr': lr,
                    'name': component
                })
        
        return param_groups