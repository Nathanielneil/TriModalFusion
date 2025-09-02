"""
Speech Encoder Module.

This module implements a speech encoder based on Transformer architecture,
inspired by Whisper and wav2vec2.0 models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MelSpectrogramExtractor(nn.Module):
    """Mel-spectrogram feature extractor."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: Optional[int] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        
        # Create mel filterbank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect"
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: Audio tensor [B, T] or [B, 1, T]
            
        Returns:
            Mel-spectrogram [B, n_mels, T_mel]
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension if present
        
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        
        return mel_spec


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(1)].transpose(0, 1)


class TransformerEncoderLayer(nn.Module):
    """Custom Transformer encoder layer with optional pre-norm."""
    
    def __init__(self, 
                 d_model: int,
                 n_head: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, d_model]
            src_mask: Source mask
            src_key_padding_mask: Padding mask [B, T]
            
        Returns:
            Output tensor [B, T, d_model]
        """
        if self.pre_norm:
            # Pre-norm variant
            x_norm = self.norm1(x)
            attn_output, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
            x = x + self.dropout(attn_output)
            
            x_norm = self.norm2(x)
            ff_output = self.feed_forward(x_norm)
            x = x + ff_output
        else:
            # Post-norm variant
            attn_output, _ = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
            x = self.norm1(x + self.dropout(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)
        
        return x


class SpeechEncoder(nn.Module):
    """
    Speech encoder based on Transformer architecture.
    
    This encoder processes audio input and converts it to fixed-size feature representations
    suitable for multimodal fusion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Audio processing parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.n_fft = config.get('n_fft', 1024)
        self.hop_length = config.get('hop_length', 256)
        
        # Model parameters
        self.d_model = config['d_model']
        self.n_head = config.get('n_head', 8)
        self.n_layer = config.get('n_layer', 6)
        self.d_ff = config.get('d_ff', self.d_model * 4)
        self.dropout = config.get('dropout', 0.1)
        self.max_length = config.get('max_audio_length', 3000)
        
        # Feature extraction
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Input projection
        self.input_projection = nn.Linear(self.n_mels, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_length)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_head=self.n_head,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=config.get('activation', 'relu'),
                pre_norm=config.get('pre_norm', True)
            ) for _ in range(self.n_layer)
        ])
        
        # Output layer norm (for pre-norm variant)
        if config.get('pre_norm', True):
            self.output_norm = nn.LayerNorm(self.d_model)
        
        # Global pooling
        self.pooling_type = config.get('pooling', 'mean')  # 'mean', 'max', 'cls', 'attention'
        
        if self.pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        elif self.pooling_type == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.ReLU(),
                nn.Linear(self.d_model // 4, 1)
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create padding mask for variable length sequences."""
        B, T, _ = x.shape
        
        if lengths is None:
            # No padding mask needed
            return None
        
        # Create mask where True indicates padding positions
        mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)
        return mask
    
    def global_pooling(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply global pooling to sequence features."""
        if self.pooling_type == 'mean':
            if padding_mask is not None:
                # Mask out padding positions
                mask = ~padding_mask.unsqueeze(-1)  # [B, T, 1]
                x_masked = x * mask
                lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
                return x_masked.sum(dim=1) / lengths
            else:
                return x.mean(dim=1)
                
        elif self.pooling_type == 'max':
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            return x.max(dim=1)[0]
            
        elif self.pooling_type == 'cls':
            # Return CLS token (first token)
            return x[:, 0]
            
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attn_weights = self.attention_pooling(x)  # [B, T, 1]
            
            if padding_mask is not None:
                attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=1)
            return (x * attn_weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def forward(self, audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the speech encoder.
        
        Args:
            audio: Audio tensor [B, T_audio] or [B, 1, T_audio]
            lengths: Optional tensor of actual lengths [B]
            
        Returns:
            Encoded speech features [B, T_mel, d_model] if global pooling not used,
            or [B, d_model] if global pooling is applied
        """
        # Extract mel-spectrogram features
        mel_features = self.mel_extractor(audio)  # [B, n_mels, T_mel]
        
        # Transpose to [B, T_mel, n_mels]
        mel_features = mel_features.transpose(1, 2)
        B, T_mel, n_mels = mel_features.shape
        
        # Project to model dimension
        x = self.input_projection(mel_features)  # [B, T_mel, d_model]
        
        # Add CLS token if using CLS pooling
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T_mel+1, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create padding mask if needed
        if lengths is not None:
            # Convert audio lengths to mel lengths
            mel_lengths = torch.ceil(lengths.float() / self.hop_length).long()
            if self.pooling_type == 'cls':
                mel_lengths = mel_lengths + 1  # Account for CLS token
            padding_mask = self.create_padding_mask(x, mel_lengths)
        else:
            padding_mask = None
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        
        # Apply output norm if using pre-norm
        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)
        
        # Apply global pooling to get fixed-size representation
        if self.pooling_type in ['mean', 'max', 'cls', 'attention']:
            if self.pooling_type == 'cls':
                # Remove CLS token from padding mask for other tokens
                if padding_mask is not None:
                    padding_mask = padding_mask[:, 1:]
                x_pooled = self.global_pooling(x, padding_mask)
                return x_pooled  # [B, d_model]
            else:
                x_pooled = self.global_pooling(x, padding_mask)
                return x_pooled  # [B, d_model]
        
        # Return sequence features
        return x  # [B, T_mel, d_model]
    
    def get_mel_length(self, audio_length: int) -> int:
        """Calculate mel-spectrogram sequence length from audio length."""
        return math.ceil(audio_length / self.hop_length)