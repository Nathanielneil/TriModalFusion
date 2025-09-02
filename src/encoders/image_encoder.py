"""
Image Encoder Module.

This module implements image encoders based on Vision Transformer (ViT) and 
convolutional architectures for image recognition and object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, Any, List
import math
import logging

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them."""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding via convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        
        # Check input size
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Extract patches and embed
        x = self.proj(x)  # [B, embed_dim, H_patch, W_patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, embed_dim]
            mask: Attention mask [B, N, N] (optional)
        Returns:
            Output tensor [B, N, embed_dim]
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]
        
        # Final projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class MLP(nn.Module):
    """Multi-layer perceptron used in Transformer blocks."""
    
    def __init__(self, 
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path: float = 0.0,
                 activation: str = 'gelu',
                 pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim, 
            int(embed_dim * mlp_ratio), 
            embed_dim, 
            dropout, 
            activation
        )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-norm variant
            x = x + self.drop_path(self.attn(self.norm1(x), mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # Post-norm variant
            x = self.norm1(x + self.drop_path(self.attn(x, mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        
        return output


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 use_cls_token: bool = True,
                 representation_size: Optional[int] = None):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            num_tokens = num_patches
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.has_logits = False
            self.num_features = embed_dim
            self.pre_logits = nn.Identity()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
            return_features: Whether to return all patch features
            
        Returns:
            Features tensor
        """
        x = self.forward_features(x)  # [B, N, embed_dim]
        
        if return_features:
            return x  # Return all features
        
        if self.use_cls_token:
            x = x[:, 0]  # Return CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        x = self.pre_logits(x)  # [B, num_features]
        
        return x


class ConvolutionalEncoder(nn.Module):
    """Convolutional encoder based on ResNet-like architecture."""
    
    def __init__(self, 
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 arch: str = 'resnet50'):
        super().__init__()
        
        if arch == 'resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=False)
            # Remove classification head
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Adaptive pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(backbone_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Encoded features [B, embed_dim]
        """
        # Extract features
        features = self.backbone(x)  # [B, backbone_dim, H', W']
        
        # Global pooling
        features = self.global_pool(features)  # [B, backbone_dim, 1, 1]
        features = features.flatten(1)  # [B, backbone_dim]
        
        # Project to target dimension
        features = self.projection(features)  # [B, embed_dim]
        
        return features


class ImageEncoder(nn.Module):
    """
    Image encoder that can use Vision Transformer or Convolutional architectures
    for image recognition and feature extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.architecture = config.get('image_architecture', 'vit')  # 'vit' or 'cnn'
        
        # Image preprocessing parameters
        self.img_size = config.get('img_size', 224)
        self.mean = config.get('image_mean', [0.485, 0.456, 0.406])
        self.std = config.get('image_std', [0.229, 0.224, 0.225])
        
        # Initialize encoder based on architecture
        if self.architecture == 'vit':
            self.encoder = VisionTransformer(
                img_size=self.img_size,
                patch_size=config.get('patch_size', 16),
                in_channels=config.get('in_channels', 3),
                embed_dim=config.get('vit_embed_dim', 768),
                depth=config.get('vit_depth', 12),
                num_heads=config.get('vit_num_heads', 12),
                mlp_ratio=config.get('mlp_ratio', 4.0),
                dropout=config.get('dropout', 0.1),
                drop_path_rate=config.get('drop_path_rate', 0.1),
                use_cls_token=config.get('use_cls_token', True),
                representation_size=config.get('representation_size', None)
            )
            encoder_dim = self.encoder.num_features
            
        elif self.architecture == 'cnn':
            self.encoder = ConvolutionalEncoder(
                in_channels=config.get('in_channels', 3),
                embed_dim=config.get('cnn_embed_dim', 2048),
                arch=config.get('cnn_arch', 'resnet50')
            )
            encoder_dim = config.get('cnn_embed_dim', 2048)
            
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Projection to target dimension if needed
        if encoder_dim != self.d_model:
            self.projection = nn.Sequential(
                nn.Linear(encoder_dim, self.d_model),
                nn.LayerNorm(self.d_model)
            )
        else:
            self.projection = nn.Identity()
        
        # Image preprocessing
        self.register_buffer('norm_mean', torch.tensor(self.mean).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor(self.std).view(1, 3, 1, 1))
        
        # Additional layers for object detection (optional)
        self.enable_detection = config.get('enable_detection', False)
        if self.enable_detection:
            self.detection_head = self._build_detection_head(config)
        
        logger.info(f"ImageEncoder initialized with {self.architecture} architecture")
    
    def _build_detection_head(self, config: Dict[str, Any]) -> nn.Module:
        """Build detection head for object detection tasks."""
        num_classes = config.get('num_detection_classes', 80)
        
        if self.architecture == 'vit':
            # For ViT, we need to handle patch features
            return nn.ModuleDict({
                'class_head': nn.Linear(self.d_model, num_classes),
                'box_head': nn.Linear(self.d_model, 4)
            })
        else:
            # For CNN, use standard detection head
            return nn.ModuleDict({
                'class_head': nn.Linear(self.d_model, num_classes),
                'box_head': nn.Linear(self.d_model, 4)
            })
    
    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images using ImageNet statistics."""
        return (images - self.norm_mean) / self.norm_std
    
    def forward(self, 
                images: torch.Tensor, 
                return_features: bool = False,
                enable_detection: bool = False) -> torch.Tensor:
        """
        Forward pass of the image encoder.
        
        Args:
            images: Input images [B, C, H, W]
            return_features: Whether to return patch-level features (for ViT)
            enable_detection: Whether to run detection head
            
        Returns:
            Encoded image features [B, d_model] or detection outputs
        """
        # Normalize images
        if images.max() > 1.0:  # Assume images are in [0, 255] range
            images = images / 255.0
        
        images = self.normalize_images(images)
        
        # Forward through encoder
        if self.architecture == 'vit':
            if return_features:
                # Return all patch features for dense tasks
                features = self.encoder.forward_features(images)  # [B, N, embed_dim]
                features = self.projection(features)  # [B, N, d_model]
                return features
            else:
                # Return global features
                features = self.encoder(images, return_features=False)  # [B, embed_dim]
                
        else:  # CNN
            features = self.encoder(images)  # [B, embed_dim]
        
        # Project to target dimension
        features = self.projection(features)  # [B, d_model]
        
        # Run detection head if requested
        if enable_detection and self.enable_detection:
            if self.architecture == 'vit' and not return_features:
                # Need patch features for detection
                patch_features = self.encoder.forward_features(images)  # [B, N, embed_dim]
                patch_features = self.projection(patch_features)  # [B, N, d_model]
                
                # Run detection on patch features
                class_logits = self.detection_head['class_head'](patch_features)  # [B, N, num_classes]
                box_coords = torch.sigmoid(self.detection_head['box_head'](patch_features))  # [B, N, 4]
                
                return {
                    'class_logits': class_logits,
                    'box_coords': box_coords,
                    'features': features
                }
            else:
                # Standard detection head
                class_logits = self.detection_head['class_head'](features)
                box_coords = torch.sigmoid(self.detection_head['box_head'](features))
                
                return {
                    'class_logits': class_logits.unsqueeze(1),  # [B, 1, num_classes]
                    'box_coords': box_coords.unsqueeze(1),      # [B, 1, 4]
                    'features': features
                }
        
        return features
    
    def extract_multiscale_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features for dense prediction tasks."""
        if self.architecture != 'vit':
            raise NotImplementedError("Multi-scale features only supported for ViT currently")
        
        # Get patch features at different scales
        features = self.encoder.forward_features(images)  # [B, N+1, embed_dim]
        
        if self.encoder.use_cls_token:
            patch_features = features[:, 1:]  # Remove CLS token
        else:
            patch_features = features
        
        B, N, D = patch_features.shape
        H = W = int(math.sqrt(N))  # Assume square patches
        
        # Reshape to spatial format
        spatial_features = patch_features.transpose(1, 2).view(B, D, H, W)
        
        # Apply projection
        spatial_features = self.projection(spatial_features.flatten(2).transpose(1, 2))
        spatial_features = spatial_features.transpose(1, 2).view(B, self.d_model, H, W)
        
        # Create multi-scale features by pooling
        scales = []
        for scale in [1, 2, 4]:
            if scale == 1:
                scales.append(spatial_features)
            else:
                pooled = F.avg_pool2d(spatial_features, kernel_size=scale, stride=scale)
                scales.append(pooled)
        
        return scales