"""
TriModalFusion: Main multimodal fusion model.

This module implements the core multimodal fusion model that integrates speech, gesture,
and image modalities using cross-modal attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging

from .base_model import BaseMultiModalModel
from ..encoders import SpeechEncoder, GestureEncoder, ImageEncoder
from ..fusion import CrossModalFusion, HierarchicalFusion, TemporalAligner, SemanticAligner
from ..utils import get_activation_function

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """Multi-task output heads for different downstream tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.tasks = config.get('tasks', ['classification'])
        
        # Initialize task-specific heads
        self.task_heads = nn.ModuleDict()
        
        if 'classification' in self.tasks:
            self.task_heads['classification'] = nn.Sequential(
                nn.Linear(self.d_model, config.get('hidden_dim', 512)),
                get_activation_function(config.get('activation', 'relu'))(),
                nn.Dropout(config.get('dropout', 0.1)),
                nn.Linear(config.get('hidden_dim', 512), config.get('num_classes', 1000))
            )
            
        if 'detection' in self.tasks:
            self.task_heads['detection'] = self._build_detection_head(config)
            
        if 'generation' in self.tasks:
            self.task_heads['generation'] = self._build_generation_head(config)
            
        if 'regression' in self.tasks:
            self.task_heads['regression'] = nn.Sequential(
                nn.Linear(self.d_model, config.get('hidden_dim', 512)),
                get_activation_function(config.get('activation', 'relu'))(),
                nn.Dropout(config.get('dropout', 0.1)),
                nn.Linear(config.get('hidden_dim', 512), config.get('regression_dim', 1))
            )
    
    def _build_detection_head(self, config: Dict[str, Any]) -> nn.Module:
        """Build detection head for object detection tasks."""
        num_classes = config.get('num_detection_classes', 80)
        num_queries = config.get('num_detection_queries', 100)
        
        return nn.ModuleDict({
            'class_embed': nn.Linear(self.d_model, num_classes + 1),  # +1 for no-object
            'bbox_embed': nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 4)  # (x, y, w, h)
            ),
            'query_embed': nn.Embedding(num_queries, self.d_model)
        })
    
    def _build_generation_head(self, config: Dict[str, Any]) -> nn.Module:
        """Build generation head for text/sequence generation."""
        vocab_size = config.get('vocab_size', 50000)
        
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            get_activation_function(config.get('activation', 'gelu'))(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, vocab_size)
        )
    
    def forward(self, fused_features: torch.Tensor, task: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through task heads.
        
        Args:
            fused_features: Fused multimodal features [B, d_model]
            task: Specific task to run (if None, run all tasks)
            
        Returns:
            Dictionary of task outputs
        """
        outputs = {}
        
        if task and task in self.task_heads:
            outputs[task] = self.task_heads[task](fused_features)
        else:
            # Run all tasks
            for task_name, head in self.task_heads.items():
                if task_name == 'detection':
                    # Special handling for detection
                    B = fused_features.size(0)
                    query_embeds = head['query_embed'].weight.unsqueeze(0).repeat(B, 1, 1)
                    
                    # Use fused features as memory for detection transformer
                    detection_features = fused_features.unsqueeze(1).repeat(1, query_embeds.size(1), 1)
                    combined_features = query_embeds + detection_features
                    
                    outputs[task_name] = {
                        'class_logits': head['class_embed'](combined_features),
                        'bbox_coords': torch.sigmoid(head['bbox_embed'](combined_features))
                    }
                else:
                    outputs[task_name] = head(fused_features)
        
        return outputs


class TriModalFusionModel(BaseMultiModalModel):
    """
    Main TriModalFusion model that integrates speech, gesture, and image modalities.
    
    This model follows a hierarchical fusion approach:
    1. Modal-specific encoding
    2. Temporal and semantic alignment
    3. Cross-modal attention and fusion
    4. Task-specific output heads
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TriModalFusion model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)
        
        self.d_model = config['d_model']
        self.target_seq_length = config.get('target_seq_length', 512)
        
        # Initialize modality-specific encoders
        self.speech_encoder = SpeechEncoder(config.get('speech_config', config))
        self.gesture_encoder = GestureEncoder(config.get('gesture_config', config))
        self.image_encoder = ImageEncoder(config.get('image_config', config))
        
        # Initialize alignment modules
        self.temporal_aligner = TemporalAligner(config)
        self.semantic_aligner = SemanticAligner(config)
        
        # Initialize fusion modules
        self.cross_modal_fusion = CrossModalFusion(config)
        self.hierarchical_fusion = HierarchicalFusion(config)
        
        # Initialize task heads
        self.task_heads = MultiTaskHead(config)
        
        # Loss weights
        self.task_weights = config.get('task_weights', {})
        self.alignment_weight = config.get('alignment_weight', 0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"TriModalFusion model initialized with {self.get_num_parameters()} parameters")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def encode_modalities(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode each modality using respective encoders.
        
        Args:
            inputs: Dictionary containing input tensors for each modality
            
        Returns:
            Dictionary of encoded features for each modality
        """
        encoded_features = {}
        
        # Encode speech
        if 'speech' in inputs and inputs['speech'] is not None:
            encoded_features['speech'] = self.speech_encoder(inputs['speech'])
        
        # Encode gesture
        if 'gesture' in inputs and inputs['gesture'] is not None:
            encoded_features['gesture'] = self.gesture_encoder(inputs['gesture'])
        
        # Encode image
        if 'image' in inputs and inputs['image'] is not None:
            encoded_features['image'] = self.image_encoder(inputs['image'])
        
        return encoded_features
    
    def align_features(self, encoded_features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Align features temporally and semantically.
        
        Args:
            encoded_features: Dictionary of encoded features
            
        Returns:
            Tuple of (aligned_features, alignment_loss)
        """
        # Temporal alignment
        aligned_features = self.temporal_aligner(encoded_features)
        
        # Semantic alignment with contrastive learning
        projected_features = self.semantic_aligner(aligned_features)
        alignment_loss = self.semantic_aligner.contrastive_loss(projected_features)
        
        return aligned_features, alignment_loss
    
    def fuse_modalities(self, aligned_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse aligned multimodal features.
        
        Args:
            aligned_features: Dictionary of temporally aligned features
            
        Returns:
            Tuple of (fused_representation, intermediate_features)
        """
        # Cross-modal attention fusion
        cross_modal_features = self.cross_modal_fusion(aligned_features)
        
        # Hierarchical fusion
        fused_representation, intermediate_features = self.hierarchical_fusion(cross_modal_features)
        
        return fused_representation, intermediate_features
    
    def forward(self, inputs: Dict[str, torch.Tensor], task: Optional[str] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TriModalFusion model.
        
        Args:
            inputs: Dictionary containing input tensors for each modality
                   - 'speech': Audio tensor [B, T_audio] or [B, T_audio, F_audio]
                   - 'gesture': Video tensor [B, T_video, H, W, C] or keypoints [B, T_video, N_joints, 3]
                   - 'image': Image tensor [B, C, H, W]
            task: Specific task to run (optional)
            
        Returns:
            Dictionary containing model outputs and intermediate results
        """
        # Step 1: Encode each modality
        encoded_features = self.encode_modalities(inputs)
        
        if not encoded_features:
            raise ValueError("No valid input modalities provided")
        
        # Step 2: Align features temporally and semantically
        aligned_features, alignment_loss = self.align_features(encoded_features)
        
        # Step 3: Fuse multimodal features
        fused_representation, intermediate_features = self.fuse_modalities(aligned_features)
        
        # Step 4: Generate task-specific outputs
        task_outputs = self.task_heads(fused_representation, task)
        
        # Prepare return dictionary
        outputs = {
            'task_outputs': task_outputs,
            'fused_features': fused_representation,
            'encoded_features': encoded_features,
            'aligned_features': aligned_features,
            'intermediate_features': intermediate_features,
            'alignment_loss': alignment_loss
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including task-specific losses and alignment loss.
        
        Args:
            outputs: Model outputs from forward pass
            targets: Ground truth targets
            
        Returns:
            Dictionary containing computed losses
        """
        losses = {}
        total_loss = 0.0
        
        # Task-specific losses
        task_outputs = outputs['task_outputs']
        
        for task, task_output in task_outputs.items():
            if task in targets:
                task_weight = self.task_weights.get(task, 1.0)
                
                if task == 'classification':
                    task_loss = F.cross_entropy(task_output, targets[task])
                elif task == 'detection':
                    task_loss = self._compute_detection_loss(task_output, targets[task])
                elif task == 'generation':
                    task_loss = F.cross_entropy(
                        task_output.view(-1, task_output.size(-1)),
                        targets[task].view(-1),
                        ignore_index=-100
                    )
                elif task == 'regression':
                    task_loss = F.mse_loss(task_output, targets[task])
                else:
                    logger.warning(f"Unknown task: {task}, skipping loss computation")
                    continue
                
                losses[f'{task}_loss'] = task_loss
                total_loss += task_weight * task_loss
        
        # Alignment loss
        if 'alignment_loss' in outputs and self.alignment_weight > 0:
            alignment_loss = outputs['alignment_loss']
            losses['alignment_loss'] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_detection_loss(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> torch.Tensor:
        """Compute detection loss (simplified DETR-style loss)."""
        class_logits = outputs['class_logits']  # [B, N_queries, N_classes+1]
        bbox_coords = outputs['bbox_coords']    # [B, N_queries, 4]
        
        # This is a simplified version - in practice, you'd use Hungarian matching
        # For now, we'll use a basic classification loss
        B, N_queries, N_classes = class_logits.shape
        
        # Create dummy targets for no-object class
        dummy_targets = torch.full((B, N_queries), N_classes-1, dtype=torch.long, device=class_logits.device)
        
        class_loss = F.cross_entropy(class_logits.view(-1, N_classes), dummy_targets.view(-1))
        bbox_loss = torch.tensor(0.0, device=class_logits.device)  # Placeholder
        
        return class_loss + bbox_loss
    
    def inference(self, inputs: Dict[str, torch.Tensor], task: str = 'classification') -> torch.Tensor:
        """
        Inference mode with specific task.
        
        Args:
            inputs: Input dictionary
            task: Task to perform
            
        Returns:
            Task-specific output tensor
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, task=task)
            return outputs['task_outputs'][task]
    
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract fused multimodal features without task heads.
        
        Args:
            inputs: Input dictionary
            
        Returns:
            Dictionary of extracted features
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            return {
                'fused_features': outputs['fused_features'],
                'encoded_features': outputs['encoded_features'],
                'aligned_features': outputs['aligned_features']
            }