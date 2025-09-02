"""
Gesture Encoder Module.

This module implements a gesture encoder that processes hand/body keypoints
using Graph Convolutional Networks and temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MediaPipeHandExtractor:
    """MediaPipe hand keypoint extractor."""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark indices (21 points per hand)
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        # Define hand skeleton connections for graph structure
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
    
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract hand keypoints from a single frame.
        
        Args:
            frame: Input image [H, W, C]
            
        Returns:
            Keypoints array [N_hands, 21, 3] where last dimension is (x, y, z)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        keypoints_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                keypoints_list.append(landmarks)
        
        # Pad to max_num_hands if needed
        while len(keypoints_list) < self.max_num_hands:
            keypoints_list.append([[0.0, 0.0, 0.0]] * 21)
        
        # Convert to numpy array and take only max_num_hands
        keypoints = np.array(keypoints_list[:self.max_num_hands])
        
        return keypoints  # [N_hands, 21, 3]
    
    def extract_sequence(self, video_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract keypoints from a sequence of video frames.
        
        Args:
            video_frames: List of frames [H, W, C]
            
        Returns:
            Keypoints sequence [T, N_hands, 21, 3]
        """
        keypoints_sequence = []
        
        for frame in video_frames:
            keypoints = self.extract_keypoints(frame)
            keypoints_sequence.append(keypoints)
        
        return np.array(keypoints_sequence)


class GraphConvolution(nn.Module):
    """Graph convolution layer for skeleton data."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input features [B, N_nodes, in_features]
            adj: Adjacency matrix [N_nodes, N_nodes]
            
        Returns:
            Output features [B, N_nodes, out_features]
        """
        support = torch.matmul(input, self.weight)  # [B, N_nodes, out_features]
        output = torch.matmul(adj, support)  # [B, N_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class SpatialGraphConvNet(nn.Module):
    """Spatial Graph Convolutional Network for hand skeleton."""
    
    def __init__(self, 
                 in_channels: int = 3,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GraphConvolution(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GraphConvolution(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.gcn_layers.append(GraphConvolution(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Create adjacency matrix for hand skeleton
        self.register_buffer('adj_matrix', self._create_adjacency_matrix())
    
    def _create_adjacency_matrix(self) -> torch.Tensor:
        """Create normalized adjacency matrix for hand skeleton."""
        # Hand connections (21 joints)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        # Create adjacency matrix
        num_joints = 21
        adj = torch.zeros(num_joints, num_joints)
        
        # Add connections (undirected graph)
        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Add self-connections
        adj.fill_diagonal_(1)
        
        # Normalize adjacency matrix (D^-1/2 * A * D^-1/2)
        degree = adj.sum(dim=1)
        degree_sqrt_inv = torch.pow(degree, -0.5)
        degree_sqrt_inv[torch.isinf(degree_sqrt_inv)] = 0
        
        adj_normalized = degree_sqrt_inv.unsqueeze(1) * adj * degree_sqrt_inv.unsqueeze(0)
        
        return adj_normalized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input keypoints [B, N_hands, N_joints, in_channels]
            
        Returns:
            Spatial features [B, N_hands, N_joints, out_channels]
        """
        B, N_hands, N_joints, C = x.shape
        
        # Reshape to process all hands together
        x = x.view(B * N_hands, N_joints, C)  # [B*N_hands, N_joints, C]
        
        # Apply GCN layers
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn(x, self.adj_matrix)  # [B*N_hands, N_joints, channels]
            
            # Apply batch norm (need to reshape for 1D batch norm)
            x_reshape = x.transpose(1, 2).contiguous()  # [B*N_hands, channels, N_joints]
            x_reshape = bn(x_reshape.view(-1, x_reshape.size(1))).view(x_reshape.shape)
            x = x_reshape.transpose(1, 2)  # [B*N_hands, N_joints, channels]
            
            if i < len(self.gcn_layers) - 1:  # Don't apply activation to last layer
                x = self.activation(x)
                x = self.dropout(x)
        
        # Reshape back
        x = x.view(B, N_hands, N_joints, -1)
        
        return x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 256,
                 out_channels: int = 512,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.conv_layers.append(
            nn.Conv1d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)
        )
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, T, in_channels]
            
        Returns:
            Temporal features [B, T, out_channels]
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)  # [B, in_channels, T]
        
        # Apply temporal conv layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv(x)
            x = bn(x)
            
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # [B, T, out_channels]
        
        return x


class GestureEncoder(nn.Module):
    """
    Gesture encoder that processes hand keypoints using Graph Convolutional Networks
    and temporal modeling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.d_model = config['d_model']
        self.num_hands = config.get('num_hands', 2)
        self.num_joints = config.get('num_joints', 21)  # MediaPipe hand has 21 joints
        self.input_dim = config.get('input_dim', 3)  # x, y, z coordinates
        
        # Spatial GCN parameters
        self.spatial_hidden_dim = config.get('spatial_hidden_dim', 64)
        self.spatial_out_dim = config.get('spatial_out_dim', 128)
        self.spatial_layers = config.get('spatial_layers', 2)
        
        # Temporal CNN parameters
        self.temporal_hidden_dim = config.get('temporal_hidden_dim', 256)
        self.temporal_layers = config.get('temporal_layers', 3)
        self.temporal_kernel_size = config.get('temporal_kernel_size', 3)
        
        self.dropout = config.get('dropout', 0.1)
        
        # Initialize MediaPipe extractor (optional, for direct video input)
        self.use_mediapipe = config.get('use_mediapipe', True)
        if self.use_mediapipe:
            self.mediapipe_extractor = MediaPipeHandExtractor(
                max_num_hands=self.num_hands,
                min_detection_confidence=config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=config.get('min_tracking_confidence', 0.5)
            )
        
        # Spatial Graph Convolution
        self.spatial_gcn = SpatialGraphConvNet(
            in_channels=self.input_dim,
            hidden_channels=self.spatial_hidden_dim,
            out_channels=self.spatial_out_dim,
            num_layers=self.spatial_layers,
            dropout=self.dropout
        )
        
        # Aggregate features across hands and joints
        self.spatial_aggregation = config.get('spatial_aggregation', 'mean')  # 'mean', 'max', 'attention'
        
        if self.spatial_aggregation == 'attention':
            self.spatial_attention = nn.Sequential(
                nn.Linear(self.spatial_out_dim, self.spatial_out_dim // 4),
                nn.ReLU(),
                nn.Linear(self.spatial_out_dim // 4, 1)
            )
        
        # Temporal Convolution
        spatial_feature_dim = self.spatial_out_dim
        self.temporal_conv = TemporalConvNet(
            in_channels=spatial_feature_dim,
            hidden_channels=self.temporal_hidden_dim,
            out_channels=self.d_model,
            num_layers=self.temporal_layers,
            kernel_size=self.temporal_kernel_size,
            dropout=self.dropout
        )
        
        # Global pooling
        self.pooling_type = config.get('pooling', 'mean')  # 'mean', 'max', 'attention'
        
        if self.pooling_type == 'attention':
            self.temporal_attention = nn.Sequential(
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
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def extract_keypoints_from_video(self, video_frames: List[np.ndarray]) -> torch.Tensor:
        """
        Extract keypoints from video frames using MediaPipe.
        
        Args:
            video_frames: List of video frames [H, W, C]
            
        Returns:
            Keypoints tensor [1, T, N_hands, N_joints, 3]
        """
        if not self.use_mediapipe:
            raise ValueError("MediaPipe extraction not enabled. Set use_mediapipe=True")
        
        keypoints_sequence = self.mediapipe_extractor.extract_sequence(video_frames)
        
        # Convert to tensor and add batch dimension
        keypoints_tensor = torch.from_numpy(keypoints_sequence).float().unsqueeze(0)
        
        return keypoints_tensor
    
    def aggregate_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features across hands and joints.
        
        Args:
            x: Spatial features [B, T, N_hands, N_joints, spatial_out_dim]
            
        Returns:
            Aggregated features [B, T, spatial_out_dim]
        """
        B, T, N_hands, N_joints, D = x.shape
        
        # Reshape to [B*T, N_hands*N_joints, D]
        x_reshape = x.view(B * T, N_hands * N_joints, D)
        
        if self.spatial_aggregation == 'mean':
            aggregated = x_reshape.mean(dim=1)  # [B*T, D]
        elif self.spatial_aggregation == 'max':
            aggregated = x_reshape.max(dim=1)[0]  # [B*T, D]
        elif self.spatial_aggregation == 'attention':
            # Attention pooling
            attn_weights = self.spatial_attention(x_reshape)  # [B*T, N_hands*N_joints, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (x_reshape * attn_weights).sum(dim=1)  # [B*T, D]
        else:
            raise ValueError(f"Unknown spatial aggregation: {self.spatial_aggregation}")
        
        # Reshape back to [B, T, D]
        aggregated = aggregated.view(B, T, D)
        
        return aggregated
    
    def global_pooling(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply global pooling to temporal sequence."""
        if self.pooling_type == 'mean':
            if padding_mask is not None:
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
            
        elif self.pooling_type == 'attention':
            attn_weights = self.temporal_attention(x)  # [B, T, 1]
            
            if padding_mask is not None:
                attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=1)
            return (x * attn_weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def forward(self, gesture_input: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the gesture encoder.
        
        Args:
            gesture_input: Either keypoints [B, T, N_hands, N_joints, 3] or 
                          video frames for MediaPipe extraction
            lengths: Optional tensor of sequence lengths [B]
            
        Returns:
            Encoded gesture features [B, d_model] or [B, T, d_model] depending on pooling
        """
        # Handle input format
        if isinstance(gesture_input, list):
            # List of video frames - extract keypoints
            gesture_input = self.extract_keypoints_from_video(gesture_input)
        
        # gesture_input should now be [B, T, N_hands, N_joints, 3]
        B, T, N_hands, N_joints, _ = gesture_input.shape
        
        # Spatial Graph Convolution
        # Process each timestep separately
        spatial_features = []
        for t in range(T):
            frame_keypoints = gesture_input[:, t]  # [B, N_hands, N_joints, 3]
            frame_features = self.spatial_gcn(frame_keypoints)  # [B, N_hands, N_joints, spatial_out_dim]
            spatial_features.append(frame_features)
        
        # Stack temporal features
        spatial_features = torch.stack(spatial_features, dim=1)  # [B, T, N_hands, N_joints, spatial_out_dim]
        
        # Aggregate spatial features
        aggregated_features = self.aggregate_spatial_features(spatial_features)  # [B, T, spatial_out_dim]
        
        # Temporal Convolution
        temporal_features = self.temporal_conv(aggregated_features)  # [B, T, d_model]
        
        # Apply global pooling if needed
        if self.pooling_type in ['mean', 'max', 'attention']:
            # Create padding mask if lengths provided
            padding_mask = None
            if lengths is not None:
                padding_mask = torch.arange(T, device=gesture_input.device).expand(B, T) >= lengths.unsqueeze(1)
            
            pooled_features = self.global_pooling(temporal_features, padding_mask)
            return pooled_features  # [B, d_model]
        
        # Return sequence features
        return temporal_features  # [B, T, d_model]