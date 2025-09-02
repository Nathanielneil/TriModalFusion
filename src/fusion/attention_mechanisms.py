"""
Cross-modal attention mechanisms for multimodal fusion.

This module implements various attention mechanisms designed specifically for
cross-modal interaction and information fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import math
import logging

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [B, H, T_q, D_k]
            key: Key tensor [B, H, T_k, D_k]
            value: Value tensor [B, H, T_v, D_v]
            mask: Attention mask [B, H, T_q, T_k]
            
        Returns:
            Tuple of (attended_values, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention for cross-modal interaction."""
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query tensor [B, T_q, D]
            key: Key tensor [B, T_k, D]
            value: Value tensor [B, T_v, D]
            mask: Attention mask [B, T_q, T_k]
            
        Returns:
            Cross-attended output [B, T_q, D]
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multi-head attention
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Apply attention
        attended_values, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Output projection
        output = self.w_o(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(output))
        
        return output


class CoAttentionLayer(nn.Module):
    """
    Co-attention layer for bidirectional cross-modal attention.
    Allows two modalities to attend to each other simultaneously.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Cross-attention layers
        self.cross_attn_1_to_2 = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.cross_attn_2_to_1 = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # Self-attention for refinement
        self.self_attn_1 = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.self_attn_2 = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # Feed-forward networks
        self.ffn_1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.ffn_2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
    def forward(self, 
                features_1: torch.Tensor,
                features_2: torch.Tensor,
                mask_1: Optional[torch.Tensor] = None,
                mask_2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features_1: Features from modality 1 [B, T1, D]
            features_2: Features from modality 2 [B, T2, D]
            mask_1: Mask for modality 1 [B, T1]
            mask_2: Mask for modality 2 [B, T2]
            
        Returns:
            Tuple of enhanced features (enhanced_1, enhanced_2)
        """
        # Cross-attention: modality 1 attends to modality 2
        enhanced_1 = self.cross_attn_1_to_2(
            query=features_1,
            key=features_2,
            value=features_2,
            mask=mask_2.unsqueeze(1).expand(-1, features_1.size(1), -1) if mask_2 is not None else None
        )
        
        # Cross-attention: modality 2 attends to modality 1  
        enhanced_2 = self.cross_attn_2_to_1(
            query=features_2,
            key=features_1,
            value=features_1,
            mask=mask_1.unsqueeze(1).expand(-1, features_2.size(1), -1) if mask_1 is not None else None
        )
        
        # Self-attention for refinement
        enhanced_1 = self.self_attn_1(enhanced_1, enhanced_1, enhanced_1, 
                                     mask_1.unsqueeze(1).expand(-1, enhanced_1.size(1), -1) if mask_1 is not None else None)
        enhanced_2 = self.self_attn_2(enhanced_2, enhanced_2, enhanced_2,
                                     mask_2.unsqueeze(1).expand(-1, enhanced_2.size(1), -1) if mask_2 is not None else None)
        
        # Feed-forward networks
        enhanced_1 = self.layer_norm_1(enhanced_1 + self.ffn_1(enhanced_1))
        enhanced_2 = self.layer_norm_2(enhanced_2 + self.ffn_2(enhanced_2))
        
        return enhanced_1, enhanced_2


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for multiple modalities.
    Enables each modality to attend to all other modalities.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Multi-layer cross-modal attention
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': MultiHeadCrossAttention(d_model, num_heads, dropout),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                'layer_norm': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # Gating mechanism for modality importance
        self.use_gating = True
        if self.use_gating:
            self.gate_networks = nn.ModuleDict()
            modalities = ['speech', 'gesture', 'image']
            for target_mod in modalities:
                self.gate_networks[target_mod] = nn.Sequential(
                    nn.Linear(d_model * len(modalities), d_model),
                    nn.Sigmoid()
                )
    
    def apply_gating(self, 
                    target_features: torch.Tensor,
                    context_features: List[torch.Tensor],
                    target_modality: str) -> torch.Tensor:
        """
        Apply gating mechanism to control information flow.
        
        Args:
            target_features: Target modality features [B, T, D]
            context_features: List of context modality features
            target_modality: Name of target modality
            
        Returns:
            Gated features [B, T, D]
        """
        # Global pooling of all features
        target_global = target_features.mean(dim=1)  # [B, D]
        context_global = [feat.mean(dim=1) for feat in context_features]  # List of [B, D]
        
        # Concatenate all global features
        all_global = torch.cat([target_global] + context_global, dim=-1)  # [B, D * num_modalities]
        
        # Compute gate
        gate = self.gate_networks[target_modality](all_global)  # [B, D]
        gate = gate.unsqueeze(1)  # [B, 1, D]
        
        # Apply gate
        gated_features = target_features * gate
        
        return gated_features
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply cross-modal attention to all modalities.
        
        Args:
            features_dict: Dictionary of features for each modality
                          Keys: modality names ('speech', 'gesture', 'image')
                          Values: feature tensors [B, T, D]
                          
        Returns:
            Dictionary of cross-modal attended features
        """
        modalities = list(features_dict.keys())
        enhanced_features = {mod: feat.clone() for mod, feat in features_dict.items()}
        
        # Apply multiple layers of cross-modal attention
        for layer_idx in range(self.num_layers):
            layer_modules = self.attention_layers[layer_idx]
            
            # For each modality, attend to all other modalities
            for target_mod in modalities:
                target_feat = enhanced_features[target_mod]
                
                # Collect context from other modalities
                context_features = []
                for source_mod in modalities:
                    if source_mod != target_mod:
                        context_features.append(enhanced_features[source_mod])
                
                if context_features:
                    # Concatenate context modalities
                    context = torch.cat(context_features, dim=1)  # [B, T_context, D]
                    
                    # Cross-modal attention
                    attended = layer_modules['cross_attn'](
                        query=target_feat,
                        key=context,
                        value=context
                    )
                    
                    # Apply gating if enabled
                    if self.use_gating and target_mod in self.gate_networks:
                        attended = self.apply_gating(attended, context_features, target_mod)
                    
                    # Feed-forward network
                    ffn_output = layer_modules['ffn'](attended)
                    enhanced_features[target_mod] = layer_modules['layer_norm'](attended + ffn_output)
                else:
                    # No other modalities available, keep original
                    enhanced_features[target_mod] = target_feat
        
        return enhanced_features


class AdaptiveFusionAttention(nn.Module):
    """
    Adaptive fusion attention that dynamically weights modalities based on content.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_modalities: int = 3,
                 num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        
        # Modality importance estimation
        self.importance_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Fusion attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def compute_modality_importance(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute importance weights for each modality.
        
        Args:
            features_dict: Dictionary of modality features
            
        Returns:
            Dictionary of importance weights [B, 1]
        """
        importance_weights = {}
        
        for modality, features in features_dict.items():
            # Global pooling
            global_feat = features.mean(dim=1)  # [B, D]
            
            # Compute importance
            importance = self.importance_network(global_feat)  # [B, 1]
            importance_weights[modality] = importance
        
        # Normalize importance weights
        total_importance = sum(importance_weights.values())
        for modality in importance_weights:
            importance_weights[modality] = importance_weights[modality] / (total_importance + 1e-8)
        
        return importance_weights
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Adaptive fusion of multiple modalities.
        
        Args:
            features_dict: Dictionary of modality features [B, T, D]
            
        Returns:
            Fused representation [B, T, D]
        """
        if not features_dict:
            raise ValueError("No modalities provided for fusion")
        
        # Compute modality importance
        importance_weights = self.compute_modality_importance(features_dict)
        
        # Weight features by importance
        weighted_features = []
        for modality, features in features_dict.items():
            weight = importance_weights[modality].unsqueeze(1)  # [B, 1, 1]
            weighted_feat = features * weight
            weighted_features.append(weighted_feat)
        
        # Stack modalities
        stacked_features = torch.stack(weighted_features, dim=2)  # [B, T, M, D]
        B, T, M, D = stacked_features.shape
        
        # Reshape for attention
        stacked_features = stacked_features.view(B * T, M, D)  # [B*T, M, D]
        
        # Self-attention across modalities
        fused_features, _ = self.fusion_attention(
            stacked_features, stacked_features, stacked_features
        )  # [B*T, M, D]
        
        # Global pooling across modalities
        fused_features = fused_features.mean(dim=1)  # [B*T, D]
        
        # Reshape back
        fused_features = fused_features.view(B, T, D)  # [B, T, D]
        
        # Output projection
        output = self.output_projection(fused_features)
        
        return output