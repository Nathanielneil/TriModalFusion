"""
Hierarchical fusion module.

This module implements hierarchical fusion strategies that combine cross-modal
features at multiple levels of abstraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion module that combines multimodal features through
    multiple levels of abstraction and integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.dropout = config.get('dropout', 0.1)
        
        # Fusion levels
        self.num_fusion_levels = config.get('num_fusion_levels', 3)
        
        # Level 1: Feature-level fusion
        self.feature_fusion = FeatureLevelFusion(config)
        
        # Level 2: Semantic-level fusion  
        self.semantic_fusion = SemanticLevelFusion(config)
        
        # Level 3: Decision-level fusion
        self.decision_fusion = DecisionLevelFusion(config)
        
        # Progressive fusion with skip connections
        self.use_skip_connections = config.get('use_skip_connections', True)
        if self.use_skip_connections:
            self.skip_projections = nn.ModuleList([
                nn.Linear(self.d_model, self.d_model) 
                for _ in range(self.num_fusion_levels)
            ])
        
        # Final output projection
        fusion_dim = self.d_model * self.num_fusion_levels if self.use_skip_connections else self.d_model
        self.final_projection = nn.Sequential(
            nn.Linear(fusion_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
    
    def forward(self, cross_modal_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical fusion of cross-modal features.
        
        Args:
            cross_modal_features: Dictionary of cross-modal enhanced features
            
        Returns:
            Tuple of (final_representation, intermediate_features)
        """
        intermediate_features = {}
        fusion_outputs = []
        
        # Level 1: Feature-level fusion
        feature_fused, feature_intermediates = self.feature_fusion(cross_modal_features)
        intermediate_features['feature_level'] = feature_intermediates
        fusion_outputs.append(feature_fused)
        
        # Level 2: Semantic-level fusion
        semantic_input = {**cross_modal_features, 'feature_fused': feature_fused}
        semantic_fused, semantic_intermediates = self.semantic_fusion(semantic_input)
        intermediate_features['semantic_level'] = semantic_intermediates
        fusion_outputs.append(semantic_fused)
        
        # Level 3: Decision-level fusion
        decision_input = {**cross_modal_features, 'feature_fused': feature_fused, 'semantic_fused': semantic_fused}
        decision_fused, decision_intermediates = self.decision_fusion(decision_input)
        intermediate_features['decision_level'] = decision_intermediates
        fusion_outputs.append(decision_fused)
        
        # Combine all fusion levels
        if self.use_skip_connections:
            # Apply skip projections
            projected_outputs = []
            for i, output in enumerate(fusion_outputs):
                projected = self.skip_projections[i](output)
                projected_outputs.append(projected)
            
            # Concatenate all levels
            combined_representation = torch.cat(projected_outputs, dim=-1)
        else:
            # Use only the final level
            combined_representation = fusion_outputs[-1]
        
        # Final projection
        final_representation = self.final_projection(combined_representation)
        
        return final_representation, intermediate_features


class FeatureLevelFusion(nn.Module):
    """Feature-level fusion that operates on raw multimodal features."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model = config['d_model']
        self.dropout = config.get('dropout', 0.1)
        
        # Feature alignment and normalization
        self.feature_aligners = nn.ModuleDict()
        for modality in ['speech', 'gesture', 'image']:
            self.feature_aligners[modality] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            )
        
        # Multi-scale feature extraction
        self.multiscale_conv = nn.ModuleList([
            nn.Conv1d(self.d_model, self.d_model, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        # Feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(self.d_model * 4, self.d_model * 2),  # 4 scales
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # Global pooling
        self.pooling_method = config.get('pooling_method', 'attention')
        if self.pooling_method == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.Tanh(),
                nn.Linear(self.d_model // 4, 1)
            )
    
    def extract_multiscale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale temporal features."""
        # features: [B, T, D]
        features_transposed = features.transpose(1, 2)  # [B, D, T]
        
        multiscale_features = []
        for conv in self.multiscale_conv:
            scale_features = conv(features_transposed)  # [B, D, T]
            scale_features = scale_features.transpose(1, 2)  # [B, T, D]
            multiscale_features.append(scale_features)
        
        # Concatenate scales
        combined_features = torch.cat(multiscale_features, dim=-1)  # [B, T, 4*D]
        
        return combined_features
    
    def global_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Apply global pooling to sequence features."""
        if self.pooling_method == 'mean':
            return features.mean(dim=1)
        elif self.pooling_method == 'max':
            return features.max(dim=1)[0]
        elif self.pooling_method == 'attention':
            # Attention pooling
            attn_weights = self.attention_pooling(features)  # [B, T, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (features * attn_weights).sum(dim=1)  # [B, D]
            return pooled
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Feature-level fusion forward pass."""
        aligned_features = {}
        multiscale_features = {}
        
        # Align and extract multi-scale features for each modality
        for modality, features in features_dict.items():
            if modality in self.feature_aligners:
                # Align features
                aligned = self.feature_aligners[modality](features)
                aligned_features[modality] = aligned
                
                # Extract multi-scale features
                multiscale = self.extract_multiscale_features(aligned)
                multiscale_features[modality] = multiscale
        
        # Combine all modalities
        if multiscale_features:
            # Average multi-scale features across modalities
            combined_multiscale = torch.stack(list(multiscale_features.values()), dim=0).mean(dim=0)
            
            # Fusion network
            fused_features = self.fusion_network(combined_multiscale)  # [B, T, D]
            
            # Global pooling
            global_representation = self.global_pooling(fused_features)
        else:
            # Fallback
            batch_size = next(iter(features_dict.values())).size(0)
            global_representation = torch.zeros(batch_size, self.d_model, 
                                              device=next(iter(features_dict.values())).device)
        
        intermediates = {
            'aligned_features': aligned_features,
            'multiscale_features': multiscale_features,
            'fused_sequence': fused_features if 'fused_features' in locals() else None
        }
        
        return global_representation, intermediates


class SemanticLevelFusion(nn.Module):
    """Semantic-level fusion that operates on higher-level representations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model = config['d_model']
        self.dropout = config.get('dropout', 0.1)
        
        # Semantic encoding
        self.semantic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=config.get('semantic_heads', 8),
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=config.get('semantic_layers', 2)
        )
        
        # Semantic concepts extraction
        self.concept_extractor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Cross-semantic attention
        self.cross_semantic_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.get('semantic_heads', 8),
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Semantic-level fusion forward pass."""
        # Collect modality features
        modality_features = []
        modality_names = []
        
        for modality, features in features_dict.items():
            if features.dim() == 2:  # Global features
                features = features.unsqueeze(1)  # Add sequence dimension
            
            # Extract semantic concepts
            semantic_concepts = self.concept_extractor(features)
            modality_features.append(semantic_concepts)
            modality_names.append(modality)
        
        if not modality_features:
            batch_size = next(iter(features_dict.values())).size(0)
            return torch.zeros(batch_size, self.d_model), {}
        
        # Stack modality features
        stacked_features = torch.cat(modality_features, dim=1)  # [B, sum(T_i), D]
        
        # Semantic encoding
        encoded_semantics = self.semantic_encoder(stacked_features)  # [B, sum(T_i), D]
        
        # Cross-semantic attention
        attended_semantics, attn_weights = self.cross_semantic_attention(
            encoded_semantics, encoded_semantics, encoded_semantics
        )
        
        # Global semantic representation
        semantic_representation = attended_semantics.mean(dim=1)  # [B, D]
        
        intermediates = {
            'semantic_concepts': {name: feat for name, feat in zip(modality_names, modality_features)},
            'encoded_semantics': encoded_semantics,
            'attention_weights': attn_weights
        }
        
        return semantic_representation, intermediates


class DecisionLevelFusion(nn.Module):
    """Decision-level fusion that operates on high-level decisions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model = config['d_model']
        self.dropout = config.get('dropout', 0.1)
        self.num_decision_classes = config.get('num_decision_classes', 512)
        
        # Decision extractors for each input type
        self.decision_extractors = nn.ModuleDict()
        
        # Multi-level decision fusion
        self.decision_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model * 2),  # Assuming 3 inputs
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.num_decision_classes),
            nn.GELU(),
            nn.Linear(self.num_decision_classes, self.d_model)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Final decision integration
        self.final_integrator = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decision-level fusion forward pass."""
        # Collect decision inputs
        decision_inputs = []
        input_names = []
        
        for name, features in features_dict.items():
            if features.dim() > 2:
                features = features.mean(dim=1)  # Global pooling if needed
            decision_inputs.append(features)
            input_names.append(name)
        
        if not decision_inputs:
            batch_size = 1
            return torch.zeros(batch_size, self.d_model), {}
        
        # Pad to fixed number of inputs (3)
        while len(decision_inputs) < 3:
            zero_input = torch.zeros_like(decision_inputs[0])
            decision_inputs.append(zero_input)
        
        # Take only first 3 inputs
        decision_inputs = decision_inputs[:3]
        
        # Concatenate decision inputs
        concatenated_decisions = torch.cat(decision_inputs, dim=-1)
        
        # Decision fusion
        fused_decisions = self.decision_fusion(concatenated_decisions)
        
        # Confidence estimation
        confidence = self.confidence_estimator(fused_decisions)
        
        # Apply confidence weighting
        weighted_decisions = fused_decisions * confidence
        
        # Final integration
        final_representation = self.final_integrator(weighted_decisions)
        
        intermediates = {
            'decision_inputs': {name: inp for name, inp in zip(input_names, decision_inputs[:len(input_names)])},
            'fused_decisions': fused_decisions,
            'confidence': confidence
        }
        
        return final_representation, intermediates