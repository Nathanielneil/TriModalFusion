"""
Cross-modal fusion module.

This module implements the main cross-modal fusion mechanisms that integrate
information from different modalities using attention and other fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import logging

from .attention_mechanisms import CrossModalAttention, CoAttentionLayer, AdaptiveFusionAttention

logger = logging.getLogger(__name__)


class CrossModalFusion(nn.Module):
    """
    Main cross-modal fusion module that integrates features from multiple modalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.fusion_strategy = config.get('fusion_strategy', 'attention')  # 'attention', 'concat', 'add', 'adaptive'
        self.num_heads = config.get('fusion_heads', 8)
        self.num_layers = config.get('fusion_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict()
        for modality in ['speech', 'gesture', 'image']:
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
        
        # Fusion mechanisms
        if self.fusion_strategy == 'attention':
            self.cross_attention = CrossModalAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            
        elif self.fusion_strategy == 'adaptive':
            self.adaptive_fusion = AdaptiveFusionAttention(
                d_model=self.d_model,
                num_modalities=3,
                num_heads=self.num_heads
            )
            
        elif self.fusion_strategy == 'concat':
            # Concatenation-based fusion
            self.concat_projection = nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.LayerNorm(self.d_model)
            )
            
        elif self.fusion_strategy == 'add':
            # Simple additive fusion with learned weights
            self.modality_weights = nn.Parameter(torch.ones(3) / 3)
            
        # Pairwise co-attention (optional)
        self.use_pairwise_attention = config.get('use_pairwise_attention', False)
        if self.use_pairwise_attention:
            self.pairwise_attention = nn.ModuleDict()
            modality_pairs = [('speech', 'gesture'), ('speech', 'image'), ('gesture', 'image')]
            for mod1, mod2 in modality_pairs:
                self.pairwise_attention[f"{mod1}_{mod2}"] = CoAttentionLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dropout=self.dropout
                )
    
    def apply_modality_projections(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality-specific projections."""
        projected_features = {}
        
        for modality, features in features_dict.items():
            if modality in self.modality_projections:
                projected_features[modality] = self.modality_projections[modality](features)
            else:
                projected_features[modality] = features
        
        return projected_features
    
    def pairwise_co_attention(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply pairwise co-attention between modalities."""
        enhanced_features = {mod: feat.clone() for mod, feat in features_dict.items()}
        
        modality_pairs = [('speech', 'gesture'), ('speech', 'image'), ('gesture', 'image')]
        
        for mod1, mod2 in modality_pairs:
            if mod1 in enhanced_features and mod2 in enhanced_features:
                pair_key = f"{mod1}_{mod2}"
                if pair_key in self.pairwise_attention:
                    enhanced_1, enhanced_2 = self.pairwise_attention[pair_key](
                        enhanced_features[mod1], enhanced_features[mod2]
                    )
                    enhanced_features[mod1] = enhanced_1
                    enhanced_features[mod2] = enhanced_2
        
        return enhanced_features
    
    def attention_fusion(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cross-modal attention fusion."""
        return self.cross_attention(features_dict)
    
    def adaptive_fusion(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply adaptive fusion attention."""
        return self.adaptive_fusion(features_dict)
    
    def concat_fusion(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply concatenation-based fusion."""
        modalities = ['speech', 'gesture', 'image']
        
        # Collect features in fixed order
        feature_list = []
        for modality in modalities:
            if modality in features_dict:
                # Global pooling to get fixed-size representation
                pooled_features = features_dict[modality].mean(dim=1)  # [B, D]
                feature_list.append(pooled_features)
            else:
                # Zero padding for missing modalities
                batch_size = next(iter(features_dict.values())).size(0)
                zero_features = torch.zeros(batch_size, self.d_model, 
                                          device=next(iter(features_dict.values())).device)
                feature_list.append(zero_features)
        
        # Concatenate features
        concat_features = torch.cat(feature_list, dim=-1)  # [B, 3*D]
        
        # Project to target dimension
        fused_features = self.concat_projection(concat_features)  # [B, D]
        
        return fused_features
    
    def additive_fusion(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply weighted additive fusion."""
        modalities = ['speech', 'gesture', 'image']
        weights = F.softmax(self.modality_weights, dim=0)
        
        # Collect and weight features
        weighted_features = []
        weight_idx = 0
        
        for modality in modalities:
            if modality in features_dict:
                # Global pooling
                pooled_features = features_dict[modality].mean(dim=1)  # [B, D]
                weighted_feat = pooled_features * weights[weight_idx]
                weighted_features.append(weighted_feat)
            weight_idx += 1
        
        # Sum weighted features
        if weighted_features:
            fused_features = torch.stack(weighted_features, dim=0).sum(dim=0)
        else:
            # Fallback: zero tensor
            batch_size = next(iter(features_dict.values())).size(0)
            fused_features = torch.zeros(batch_size, self.d_model,
                                       device=next(iter(features_dict.values())).device)
        
        return fused_features
    
    def forward(self, aligned_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of cross-modal fusion.
        
        Args:
            aligned_features: Dictionary of temporally aligned features
                             Keys: modality names
                             Values: feature tensors [B, T, D]
                             
        Returns:
            Dictionary containing fused features and intermediate results
        """
        if not aligned_features:
            raise ValueError("No aligned features provided for fusion")
        
        # Apply modality-specific projections
        projected_features = self.apply_modality_projections(aligned_features)
        
        # Apply pairwise co-attention if enabled
        if self.use_pairwise_attention:
            enhanced_features = self.pairwise_co_attention(projected_features)
        else:
            enhanced_features = projected_features
        
        # Apply main fusion strategy
        if self.fusion_strategy == 'attention':
            fused_features = self.attention_fusion(enhanced_features)
            
        elif self.fusion_strategy == 'adaptive':
            # Adaptive fusion returns a single tensor
            fused_tensor = self.adaptive_fusion(enhanced_features)
            fused_features = {'fused': fused_tensor}
            fused_features.update(enhanced_features)  # Include individual modality features
            
        elif self.fusion_strategy == 'concat':
            fused_tensor = self.concat_fusion(enhanced_features)
            fused_features = {'fused': fused_tensor}
            fused_features.update(enhanced_features)
            
        elif self.fusion_strategy == 'add':
            fused_tensor = self.additive_fusion(enhanced_features)
            fused_features = {'fused': fused_tensor}
            fused_features.update(enhanced_features)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return fused_features