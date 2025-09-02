"""
Alignment modules for multimodal fusion.

This module provides temporal and semantic alignment mechanisms to synchronize
features from different modalities before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TemporalAligner(nn.Module):
    """
    Temporal alignment module that synchronizes sequences of different lengths
    and sampling rates across modalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.target_length = config.get('target_seq_length', 512)
        self.alignment_method = config.get('alignment_method', 'interpolation')  # 'interpolation', 'attention', 'learned'
        self.d_model = config['d_model']
        
        if self.alignment_method == 'attention':
            # Learnable attention-based alignment
            self.alignment_attention = nn.ModuleDict()
            for modality in ['speech', 'gesture', 'image']:
                self.alignment_attention[modality] = nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=config.get('alignment_heads', 8),
                    batch_first=True
                )
                
        elif self.alignment_method == 'learned':
            # Learned upsampling/downsampling
            self.alignment_layers = nn.ModuleDict()
            for modality in ['speech', 'gesture', 'image']:
                self.alignment_layers[modality] = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 2),
                    nn.GELU(),
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.Dropout(config.get('dropout', 0.1))
                )
        
        # Positional encoding for aligned sequences
        self.pos_encoding = self._create_positional_encoding(self.target_length, self.d_model)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def interpolate_sequence(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Interpolate sequence to target length.
        
        Args:
            features: Input features [B, T, D]
            target_length: Target sequence length
            
        Returns:
            Interpolated features [B, target_length, D]
        """
        B, T, D = features.shape
        
        if T == target_length:
            return features
        
        # Transpose for interpolation
        features = features.transpose(1, 2)  # [B, D, T]
        
        # Interpolate
        aligned_features = F.interpolate(
            features, 
            size=target_length, 
            mode='linear', 
            align_corners=False
        )
        
        # Transpose back
        aligned_features = aligned_features.transpose(1, 2)  # [B, target_length, D]
        
        return aligned_features
    
    def attention_align(self, features: torch.Tensor, modality: str, target_length: int) -> torch.Tensor:
        """
        Use attention mechanism for temporal alignment.
        
        Args:
            features: Input features [B, T, D]
            modality: Modality name
            target_length: Target sequence length
            
        Returns:
            Aligned features [B, target_length, D]
        """
        B, T, D = features.shape
        
        # Create target sequence with positional encoding
        target_pos = self.pos_encoding[:, :target_length, :].to(features.device)  # [1, target_length, D]
        target_seq = target_pos.expand(B, -1, -1)  # [B, target_length, D]
        
        # Use attention to align source to target
        aligned_features, _ = self.alignment_attention[modality](
            query=target_seq,
            key=features,
            value=features
        )
        
        return aligned_features
    
    def learned_align(self, features: torch.Tensor, modality: str, target_length: int) -> torch.Tensor:
        """
        Use learned transformation for temporal alignment.
        
        Args:
            features: Input features [B, T, D]
            modality: Modality name
            target_length: Target sequence length
            
        Returns:
            Aligned features [B, target_length, D]
        """
        # Apply learned transformation
        transformed_features = self.alignment_layers[modality](features)
        
        # Then interpolate to target length
        aligned_features = self.interpolate_sequence(transformed_features, target_length)
        
        return aligned_features
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Align temporal sequences across modalities.
        
        Args:
            features_dict: Dictionary of features for each modality
                          Keys: modality names ('speech', 'gesture', 'image')
                          Values: feature tensors [B, T_i, D]
                          
        Returns:
            Dictionary of temporally aligned features [B, target_length, D]
        """
        aligned_features = {}
        
        for modality, features in features_dict.items():
            if features.dim() == 2:
                # Add sequence dimension if missing
                features = features.unsqueeze(1)  # [B, 1, D]
            
            B, T, D = features.shape
            
            if T == self.target_length:
                aligned_features[modality] = features
                continue
            
            # Apply alignment method
            if self.alignment_method == 'interpolation':
                aligned = self.interpolate_sequence(features, self.target_length)
                
            elif self.alignment_method == 'attention':
                aligned = self.attention_align(features, modality, self.target_length)
                
            elif self.alignment_method == 'learned':
                aligned = self.learned_align(features, modality, self.target_length)
                
            else:
                raise ValueError(f"Unknown alignment method: {self.alignment_method}")
            
            aligned_features[modality] = aligned
        
        return aligned_features


class SemanticAligner(nn.Module):
    """
    Semantic alignment module that projects different modalities into a common
    semantic space using contrastive learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.projection_dim = config.get('projection_dim', 256)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Modality-specific projection heads
        self.projection_heads = nn.ModuleDict()
        for modality in ['speech', 'gesture', 'image']:
            self.projection_heads[modality] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(config.get('dropout', 0.1)),
                nn.Linear(self.d_model, self.projection_dim),
                nn.LayerNorm(self.projection_dim)
            )
        
        # Cross-modal similarity learning
        self.similarity_type = config.get('similarity_type', 'cosine')  # 'cosine', 'bilinear', 'mlp'
        
        if self.similarity_type == 'bilinear':
            # Bilinear similarity matrices
            self.bilinear_layers = nn.ModuleDict()
            modalities = ['speech', 'gesture', 'image']
            for i, mod1 in enumerate(modalities):
                for j, mod2 in enumerate(modalities):
                    if i < j:  # Only create upper triangular
                        layer_name = f"{mod1}_{mod2}"
                        self.bilinear_layers[layer_name] = nn.Bilinear(
                            self.projection_dim, self.projection_dim, 1
                        )
                        
        elif self.similarity_type == 'mlp':
            # MLP-based similarity
            self.similarity_mlp = nn.Sequential(
                nn.Linear(self.projection_dim * 2, self.projection_dim),
                nn.GELU(),
                nn.Linear(self.projection_dim, 1)
            )
    
    def global_pool_features(self, features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """
        Pool sequence features to global representation.
        
        Args:
            features: Input features [B, T, D]
            method: Pooling method ('mean', 'max', 'cls')
            
        Returns:
            Global features [B, D]
        """
        if method == 'mean':
            return features.mean(dim=1)
        elif method == 'max':
            return features.max(dim=1)[0]
        elif method == 'cls':
            return features[:, 0]  # Assume first token is CLS
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    
    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                          mod1: str, mod2: str) -> torch.Tensor:
        """
        Compute similarity between two modality features.
        
        Args:
            feat1: Features from modality 1 [B, D]
            feat2: Features from modality 2 [B, D] 
            mod1: Name of modality 1
            mod2: Name of modality 2
            
        Returns:
            Similarity scores [B, B]
        """
        if self.similarity_type == 'cosine':
            # Normalize features
            feat1_norm = F.normalize(feat1, dim=-1)
            feat2_norm = F.normalize(feat2, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.matmul(feat1_norm, feat2_norm.T)
            
        elif self.similarity_type == 'bilinear':
            # Use bilinear layer
            layer_name = f"{mod1}_{mod2}" if mod1 < mod2 else f"{mod2}_{mod1}"
            if layer_name in self.bilinear_layers:
                bilinear_layer = self.bilinear_layers[layer_name]
                
                # Compute pairwise bilinear similarity
                B1, B2 = feat1.size(0), feat2.size(0)
                similarity = torch.zeros(B1, B2, device=feat1.device)
                
                for i in range(B1):
                    for j in range(B2):
                        similarity[i, j] = bilinear_layer(feat1[i:i+1], feat2[j:j+1])
            else:
                # Fallback to cosine similarity
                return self.compute_similarity(feat1, feat2, mod1, mod2)
                
        elif self.similarity_type == 'mlp':
            # Use MLP for similarity
            B1, B2 = feat1.size(0), feat2.size(0)
            similarity = torch.zeros(B1, B2, device=feat1.device)
            
            for i in range(B1):
                for j in range(B2):
                    concat_feat = torch.cat([feat1[i:i+1], feat2[j:j+1]], dim=-1)
                    similarity[i, j] = self.similarity_mlp(concat_feat).squeeze()
        
        return similarity
    
    def contrastive_loss(self, projected_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss for semantic alignment.
        
        Args:
            projected_features: Dictionary of projected features
            
        Returns:
            Contrastive loss scalar
        """
        modalities = list(projected_features.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0, device=next(iter(projected_features.values())).device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Compute loss for all pairs of modalities
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                feat1 = projected_features[mod1]
                feat2 = projected_features[mod2]
                
                # Compute similarity matrix
                similarity = self.compute_similarity(feat1, feat2, mod1, mod2)
                
                # Scale by temperature
                logits = similarity / self.temperature.exp()
                
                # Contrastive loss (symmetric)
                batch_size = logits.size(0)
                labels = torch.arange(batch_size, device=logits.device)
                
                loss_12 = F.cross_entropy(logits, labels)
                loss_21 = F.cross_entropy(logits.T, labels)
                
                pair_loss = (loss_12 + loss_21) / 2
                total_loss += pair_loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def forward(self, aligned_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project features to common semantic space.
        
        Args:
            aligned_features: Dictionary of temporally aligned features [B, T, D]
            
        Returns:
            Dictionary of projected features [B, projection_dim]
        """
        projected_features = {}
        
        for modality, features in aligned_features.items():
            # Global pooling to get fixed-size representation
            global_features = self.global_pool_features(
                features, method=self.config.get('pooling_method', 'mean')
            )
            
            # Project to semantic space
            projected = self.projection_heads[modality](global_features)
            
            # L2 normalize if using cosine similarity
            if self.similarity_type == 'cosine':
                projected = F.normalize(projected, dim=-1)
            
            projected_features[modality] = projected
        
        return projected_features


class ModalityDropout(nn.Module):
    """
    Modality dropout for robust multimodal learning.
    Randomly drops entire modalities during training to improve robustness.
    """
    
    def __init__(self, modality_dropout_prob: float = 0.1):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
    
    def forward(self, features_dict: Dict[str, torch.Tensor], 
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Apply modality dropout.
        
        Args:
            features_dict: Dictionary of features
            training: Whether in training mode
            
        Returns:
            Dictionary with potentially dropped modalities
        """
        if not training or self.modality_dropout_prob == 0.0:
            return features_dict
        
        # Always keep at least one modality
        modalities = list(features_dict.keys())
        if len(modalities) <= 1:
            return features_dict
        
        # Randomly select modalities to keep
        kept_modalities = []
        for modality in modalities:
            if torch.rand(1).item() > self.modality_dropout_prob:
                kept_modalities.append(modality)
        
        # Ensure at least one modality is kept
        if not kept_modalities:
            kept_modalities = [np.random.choice(modalities)]
        
        # Return filtered dictionary
        return {mod: features_dict[mod] for mod in kept_modalities if mod in features_dict}