# TriModalFusion 模型架构详解

## 目录
- [总体架构](#总体架构)
- [模态编码器详解](#模态编码器详解)
- [特征融合机制](#特征融合机制)
- [多任务输出头](#多任务输出头)
- [数据流分析](#数据流分析)
- [计算复杂度分析](#计算复杂度分析)
- [扩展性设计](#扩展性设计)

## 总体架构

TriModalFusion采用分层式多模态融合架构，将语音、手势、图像三种模态信息进行深度整合。整体架构遵循"编码-对齐-融合-输出"的设计理念。

### 架构层次

```
输入层 (Input Layer)
├── 语音输入: 16kHz 音频波形 [B, T_audio]
├── 手势输入: 关键点序列 [B, T_gesture, 2, 21, 3] 
└── 图像输入: RGB图像 [B, 3, 224, 224]

编码层 (Encoding Layer)  
├── SpeechEncoder: Transformer-based
├── GestureEncoder: MediaPipe + GCN + Temporal CNN
└── ImageEncoder: Vision Transformer / CNN

对齐层 (Alignment Layer)
├── TemporalAlignment: 时序对齐机制
└── SemanticAlignment: 语义对齐机制

融合层 (Fusion Layer)
├── CrossModalAttention: 跨模态注意力
└── HierarchicalFusion: 分层融合策略

输出层 (Output Layer)
├── ClassificationHead: 多类分类
├── DetectionHead: 目标检测  
├── RegressionHead: 回归预测
└── GenerationHead: 序列生成
```

### 核心设计原则

1. **模块化设计**: 每个模态编码器独立设计，便于替换和升级
2. **渐进式融合**: 从特征级到语义级到决策级的分层融合
3. **注意力机制**: 广泛使用自注意力和跨模态注意力
4. **可配置性**: 所有组件参数均可通过配置文件调整
5. **可扩展性**: 架构支持新模态和新任务的接入

## 模态编码器详解

### 语音编码器 (SpeechEncoder)

语音编码器基于Whisper架构设计，专门处理音频信号的特征提取。

#### 架构细节

```python
class SpeechEncoder(nn.Module):
    def __init__(self, config):
        # Mel频谱图提取器
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model=512)
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=2048
            ),
            num_layers=6
        )
        
        # 池化策略
        self.pooling = PoolingLayer(method="attention")
```

#### 数据流处理

1. **音频预处理**:
   - 输入: 原始音频波形 `[batch_size, audio_length]`
   - 重采样至16kHz (如需要)
   - 音频归一化: `audio = audio / torch.max(torch.abs(audio))`

2. **Mel频谱图提取**:
   ```python
   mel_spec = torch.stft(
       input=audio,
       n_fft=1024,
       hop_length=256,
       win_length=1024,
       window=torch.hann_window(1024)
   )
   mel_spec = mel_filter_bank @ torch.abs(mel_spec)
   mel_spec = torch.log(mel_spec + 1e-8)  # 对数变换
   ```
   输出: `[batch_size, n_mels, time_frames]` = `[B, 80, T]`

3. **特征投影**:
   - Mel频谱图转置为序列格式: `[B, T, 80]`
   - 线性投影到模型维度: `[B, T, d_model]`
   - 添加位置编码

4. **Transformer编码**:
   - 多头自注意力机制提取时序依赖
   - 残差连接和层归一化
   - 前馈网络进行非线性变换

5. **特征池化**:
   - **Mean Pooling**: `output = torch.mean(sequence, dim=1)`
   - **Attention Pooling**: 学习注意力权重进行加权平均
   - **CLS Token**: 添加分类令牌并提取其表示

#### 关键参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| sample_rate | 16000 | 音频采样率 |
| n_mels | 80 | Mel滤波器数量 |
| d_model | 512 | 模型隐藏维度 |
| n_head | 8 | 注意力头数 |
| n_layer | 6 | Transformer层数 |
| max_audio_length | 3000 | 最大音频帧数 |

### 手势编码器 (GestureEncoder)

手势编码器结合MediaPipe关键点检测和图神经网络，专门处理手部动作序列。

#### 架构细节

```python
class GestureEncoder(nn.Module):
    def __init__(self, config):
        # MediaPipe手部检测器
        self.mediapipe_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # 空间图卷积网络
        self.spatial_gcn = SpatialGCN(
            in_channels=3,
            hidden_channels=64,
            out_channels=128,
            num_layers=2
        )
        
        # 时序卷积网络
        self.temporal_cnn = TemporalCNN(
            in_channels=128,
            hidden_channels=256,
            kernel_sizes=[3, 5, 7],
            num_layers=3
        )
        
        # 特征聚合
        self.aggregation = SpatialAggregation(method="attention")
```

#### MediaPipe关键点检测

1. **手部检测**:
   - 输入视频帧: `[B, T, H, W, 3]`
   - MediaPipe检测每帧中的手部区域
   - 提取21个关键点坐标 `(x, y, z)`

2. **关键点结构**:
   ```
   手部关键点编号:
   0: WRIST (手腕)
   1-4: THUMB (拇指) 
   5-8: INDEX_FINGER (食指)
   9-12: MIDDLE_FINGER (中指)
   13-16: RING_FINGER (无名指)
   17-20: PINKY (小指)
   ```

3. **坐标归一化**:
   ```python
   # 手腕中心化
   keypoints_centered = keypoints - keypoints[:, :, :, 0:1, :]  # 以手腕为原点
   
   # 尺度归一化
   hand_size = torch.max(torch.abs(keypoints_centered), dim=-2)[0]
   keypoints_norm = keypoints_centered / (hand_size + 1e-8)
   ```

#### 空间图卷积网络 (Spatial GCN)

1. **图结构定义**:
   ```python
   # 手部骨架连接关系
   hand_skeleton = [
       (0, 1), (1, 2), (2, 3), (3, 4),    # 拇指
       (0, 5), (5, 6), (6, 7), (7, 8),    # 食指  
       (0, 9), (9, 10), (10, 11), (11, 12), # 中指
       (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
       (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
   ]
   ```

2. **图卷积操作**:
   ```python
   def graph_conv(x, adj_matrix):
       # x: [B, T, N, C] - 节点特征
       # adj_matrix: [N, N] - 邻接矩阵
       
       # 消息传递
       messages = torch.matmul(adj_matrix, x)  # [B, T, N, C]
       
       # 特征变换
       output = self.linear(messages)  # [B, T, N, C']
       
       return output
   ```

3. **多尺度特征**:
   - 1-hop邻居: 直接相邻关节
   - 2-hop邻居: 二度相邻关节
   - Global连接: 所有关节的全局信息

#### 时序卷积网络 (Temporal CNN)

1. **多尺度时序建模**:
   ```python
   # 不同时间窗口的卷积核
   kernel_sizes = [3, 5, 7]  # 捕获短期、中期、长期依赖
   
   temporal_features = []
   for kernel_size in kernel_sizes:
       conv = nn.Conv1d(
           in_channels=128,
           out_channels=256 // len(kernel_sizes),
           kernel_size=kernel_size,
           padding=kernel_size // 2
       )
       feat = conv(x.transpose(1, 2)).transpose(1, 2)
       temporal_features.append(feat)
   
   # 特征融合
   output = torch.cat(temporal_features, dim=-1)
   ```

2. **残差连接**:
   ```python
   def temporal_block(x):
       residual = x
       
       # 卷积 + 批归一化 + 激活
       x = self.conv1(x)
       x = self.bn1(x)
       x = F.relu(x)
       
       x = self.conv2(x)
       x = self.bn2(x)
       
       # 残差连接
       if self.downsample:
           residual = self.downsample(residual)
       
       return F.relu(x + residual)
   ```

#### 空间特征聚合

1. **注意力聚合**:
   ```python
   def attention_aggregation(hand_features):
       # hand_features: [B, T, 2, 21, C] - 双手特征
       
       # 计算注意力权重
       attention_scores = self.attention_linear(hand_features)  # [B, T, 2, 21, 1]
       attention_weights = F.softmax(attention_scores, dim=-2)
       
       # 加权聚合
       aggregated = torch.sum(hand_features * attention_weights, dim=-2)  # [B, T, 2, C]
       
       return aggregated
   ```

2. **双手融合**:
   - 单独处理: 左右手独立编码
   - 交互建模: 双手之间的协调关系
   - 自适应权重: 根据任务动态调整双手重要性

### 图像编码器 (ImageEncoder)

图像编码器支持Vision Transformer和CNN两种架构，可根据任务需求选择。

#### Vision Transformer (ViT) 架构

1. **图像分块 (Patch Embedding)**:
   ```python
   # 图像分割成patches
   patch_size = 16
   num_patches = (224 // patch_size) ** 2  # 196个patches
   
   # 线性投影
   x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
   x = x.contiguous().view(B, 3, -1, patch_size * patch_size)
   x = x.permute(0, 2, 1, 3).contiguous().view(B, num_patches, -1)
   
   # 投影到模型维度
   patch_embeddings = self.patch_projection(x)  # [B, 196, d_model]
   ```

2. **位置编码**:
   ```python
   # 可学习的位置编码
   pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
   
   # CLS token
   cls_token = nn.Parameter(torch.randn(1, 1, d_model))
   cls_tokens = cls_token.expand(B, -1, -1)
   
   # 拼接CLS token和patch embeddings
   x = torch.cat([cls_tokens, patch_embeddings], dim=1)
   x = x + pos_embedding
   ```

3. **Transformer编码**:
   ```python
   for layer in self.transformer_layers:
       # 多头自注意力
       attn_output = layer.self_attention(x)
       x = layer.norm1(x + attn_output)
       
       # 前馈网络
       ff_output = layer.feed_forward(x)
       x = layer.norm2(x + ff_output)
   ```

#### CNN架构 (可选)

1. **主干网络**:
   - ResNet50/101: 深度残差网络
   - EfficientNet: 高效卷积网络
   - DenseNet: 密集连接网络

2. **特征金字塔**:
   ```python
   # 多尺度特征提取
   features = {
       'stage1': self.stage1(x),  # [B, 256, 56, 56]
       'stage2': self.stage2(x),  # [B, 512, 28, 28] 
       'stage3': self.stage3(x),  # [B, 1024, 14, 14]
       'stage4': self.stage4(x)   # [B, 2048, 7, 7]
   }
   
   # FPN融合
   fused_features = self.fpn(features)
   ```

#### 目标检测模块 (可选)

基于DETR (Detection Transformer) 架构:

1. **对象查询 (Object Queries)**:
   ```python
   # 可学习的对象查询
   object_queries = nn.Parameter(torch.randn(num_queries, d_model))
   
   # 位置编码
   pos_encoding = self.position_encoding(image_features)
   ```

2. **解码器**:
   ```python
   for layer in self.decoder_layers:
       # 自注意力
       queries = layer.self_attention(queries)
       
       # 交叉注意力 (查询 -> 图像特征)
       queries = layer.cross_attention(queries, image_features + pos_encoding)
       
       # 前馈网络
       queries = layer.feed_forward(queries)
   ```

3. **检测头**:
   ```python
   # 分类预测
   class_logits = self.class_head(queries)  # [B, num_queries, num_classes]
   
   # 边框预测  
   bbox_coords = self.bbox_head(queries)    # [B, num_queries, 4]
   ```

## 特征融合机制

### 时序对齐 (Temporal Alignment)

不同模态具有不同的时间分辨率，需要进行时序对齐:

- **语音**: ~100 frames/second (mel-spectrogram)
- **手势**: ~30 frames/second (video)  
- **图像**: 1 frame (static image)

#### 1. 插值对齐 (Interpolation Alignment)

```python
def interpolation_alignment(features, target_length):
    """
    使用插值方法对齐时序特征
    """
    B, T, C = features.shape
    
    if T == target_length:
        return features
    
    # 线性插值
    features_interp = F.interpolate(
        features.transpose(1, 2),  # [B, C, T]
        size=target_length,
        mode='linear',
        align_corners=False
    ).transpose(1, 2)  # [B, target_length, C]
    
    return features_interp
```

#### 2. 注意力对齐 (Attention Alignment)

```python
class AttentionAlignment(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, source_features, target_features):
        """
        source_features: [B, T_source, C]
        target_features: [B, T_target, C]  
        """
        # 使用target作为query，source作为key和value
        aligned_features, attention_weights = self.multihead_attn(
            query=target_features.transpose(0, 1),
            key=source_features.transpose(0, 1),
            value=source_features.transpose(0, 1)
        )
        
        return aligned_features.transpose(0, 1), attention_weights
```

#### 3. 学习对齐 (Learned Alignment)

```python
class LearnedAlignment(nn.Module):
    def __init__(self, d_model, max_length=512):
        super().__init__()
        self.alignment_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_length),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, source_features, target_length):
        """学习源序列到目标长度的软对齐"""
        B, T_source, C = source_features.shape
        
        # 计算全局表示
        global_repr = torch.mean(source_features, dim=1)  # [B, C]
        
        # 为每个目标位置学习对齐权重
        aligned_features = []
        for t in range(target_length):
            # 位置嵌入
            pos_emb = self.pos_embedding(t).expand(B, -1)  # [B, C]
            
            # 计算对齐权重
            query = torch.cat([global_repr, pos_emb], dim=-1)
            alignment_weights = self.alignment_net(query)  # [B, T_source]
            
            # 加权聚合
            aligned_feat = torch.sum(
                source_features * alignment_weights.unsqueeze(-1), 
                dim=1
            )
            aligned_features.append(aligned_feat)
        
        return torch.stack(aligned_features, dim=1)
```

### 语义对齐 (Semantic Alignment)

确保不同模态的特征在相同语义空间中表示。

#### 对比学习对齐

```python
class ContrastiveAlignment(nn.Module):
    def __init__(self, d_model, projection_dim=256, temperature=0.07):
        super().__init__()
        self.projectors = nn.ModuleDict({
            'speech': nn.Linear(d_model, projection_dim),
            'gesture': nn.Linear(d_model, projection_dim), 
            'image': nn.Linear(d_model, projection_dim)
        })
        self.temperature = temperature
    
    def forward(self, features_dict):
        # 投影到共同语义空间
        projected_features = {}
        for modality, features in features_dict.items():
            projected = self.projectors[modality](features)
            projected = F.normalize(projected, dim=-1)
            projected_features[modality] = projected
        
        return projected_features
    
    def compute_alignment_loss(self, projected_features):
        """计算跨模态对比损失"""
        modalities = list(projected_features.keys())
        total_loss = 0
        num_pairs = 0
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                feat1, feat2 = projected_features[mod1], projected_features[mod2]
                
                # 相似度矩阵
                similarity = torch.matmul(feat1, feat2.T) / self.temperature
                
                # 对比损失
                labels = torch.arange(feat1.size(0)).to(feat1.device)
                loss = F.cross_entropy(similarity, labels)
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs
```

### 跨模态注意力 (Cross-Modal Attention)

#### 成对注意力 (Pairwise Attention)

```python
class PairwiseAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attention_modules = nn.ModuleDict({
            'speech_to_gesture': nn.MultiheadAttention(d_model, num_heads),
            'speech_to_image': nn.MultiheadAttention(d_model, num_heads),
            'gesture_to_speech': nn.MultiheadAttention(d_model, num_heads),
            'gesture_to_image': nn.MultiheadAttention(d_model, num_heads),
            'image_to_speech': nn.MultiheadAttention(d_model, num_heads),
            'image_to_gesture': nn.MultiheadAttention(d_model, num_heads)
        })
    
    def forward(self, features_dict):
        enhanced_features = {}
        
        for target_mod in features_dict.keys():
            target_feat = features_dict[target_mod]
            attentions = []
            
            for source_mod in features_dict.keys():
                if source_mod != target_mod:
                    source_feat = features_dict[source_mod]
                    
                    # 跨模态注意力
                    enhanced_feat, _ = self.attention_modules[f'{source_mod}_to_{target_mod}'](
                        query=target_feat.transpose(0, 1),
                        key=source_feat.transpose(0, 1),
                        value=source_feat.transpose(0, 1)
                    )
                    attentions.append(enhanced_feat.transpose(0, 1))
            
            # 融合增强特征
            if attentions:
                enhanced_features[target_mod] = target_feat + torch.mean(torch.stack(attentions), dim=0)
            else:
                enhanced_features[target_mod] = target_feat
        
        return enhanced_features
```

#### 全局注意力 (Global Attention)

```python
class GlobalCrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
    
    def forward(self, features_dict):
        # 拼接所有模态特征
        all_features = []
        modality_masks = []
        
        for modality, features in features_dict.items():
            all_features.append(features)
            # 创建模态掩码
            mask = torch.zeros(features.size(1), dtype=torch.long)
            mask.fill_(hash(modality) % 1000)  # 模态ID
            modality_masks.append(mask)
        
        # 拼接序列 [B, T_total, C]
        concatenated_features = torch.cat(all_features, dim=1)
        concatenated_masks = torch.cat(modality_masks, dim=0)
        
        # 全局自注意力处理
        x = concatenated_features.transpose(0, 1)  # [T_total, B, C]
        for layer in self.layers:
            x = layer(x)
        
        enhanced_features = x.transpose(0, 1)  # [B, T_total, C]
        
        # 分离各模态特征
        start_idx = 0
        separated_features = {}
        for modality, original_features in features_dict.items():
            seq_len = original_features.size(1)
            separated_features[modality] = enhanced_features[:, start_idx:start_idx+seq_len, :]
            start_idx += seq_len
        
        return separated_features
```

### 分层融合 (Hierarchical Fusion)

#### 三层融合策略

```python
class HierarchicalFusion(nn.Module):
    def __init__(self, d_model, num_fusion_levels=3):
        super().__init__()
        self.fusion_levels = nn.ModuleList()
        
        for level in range(num_fusion_levels):
            fusion_layer = FusionLayer(
                d_model=d_model,
                fusion_type=self._get_fusion_type(level),
                attention_heads=8 if level > 0 else 4
            )
            self.fusion_levels.append(fusion_layer)
    
    def _get_fusion_type(self, level):
        """根据层级确定融合策略"""
        fusion_types = ['concat', 'attention', 'adaptive']
        return fusion_types[min(level, len(fusion_types) - 1)]
    
    def forward(self, features_dict, skip_connections=True):
        current_features = features_dict
        skip_features = [] if skip_connections else None
        
        for level, fusion_layer in enumerate(self.fusion_levels):
            # 融合当前层特征
            fused = fusion_layer(current_features)
            
            # 跳跃连接
            if skip_connections and level > 0:
                # 与之前层的特征结合
                prev_fused = skip_features[-1]
                fused = fused + self.skip_projections[level-1](prev_fused)
            
            if skip_connections:
                skip_features.append(fused)
            
            # 更新特征用于下一层
            current_features = {
                modality: fused for modality in features_dict.keys()
            }
        
        return fused, skip_features
```

#### 自适应融合权重

```python
class AdaptiveFusion(nn.Module):
    def __init__(self, d_model, num_modalities):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features_list):
        # 计算全局特征表示
        global_features = torch.cat([
            torch.mean(feat, dim=1) for feat in features_list
        ], dim=-1)
        
        # 生成自适应权重
        fusion_weights = self.weight_generator(global_features)  # [B, num_modalities]
        
        # 加权融合
        weighted_features = []
        for i, feat in enumerate(features_list):
            weight = fusion_weights[:, i:i+1].unsqueeze(1)  # [B, 1, 1]
            weighted_features.append(feat * weight)
        
        fused_features = sum(weighted_features)
        
        return fused_features, fusion_weights
```

## 多任务输出头

### 分类头 (Classification Head)

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, fused_features):
        # 全局池化
        if fused_features.dim() > 2:
            pooled_features = torch.mean(fused_features, dim=1)
        else:
            pooled_features = fused_features
        
        logits = self.classifier(pooled_features)
        return logits
```

### 检测头 (Detection Head)

```python
class DetectionHead(nn.Module):
    def __init__(self, d_model, num_classes, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        
        # 对象查询
        self.object_queries = nn.Parameter(torch.randn(num_queries, d_model))
        
        # 解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, 8, d_model * 4),
            num_layers=6
        )
        
        # 预测头
        self.class_head = nn.Linear(d_model, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )
    
    def forward(self, fused_features):
        B = fused_features.size(0)
        
        # 扩展对象查询
        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1)
        
        # 解码
        decoded_queries = self.decoder(
            tgt=queries.transpose(0, 1),
            memory=fused_features.transpose(0, 1)
        ).transpose(0, 1)
        
        # 预测
        class_logits = self.class_head(decoded_queries)
        bbox_coords = self.bbox_head(decoded_queries)
        
        return {
            'class_logits': class_logits,  # [B, num_queries, num_classes]
            'bbox_coords': bbox_coords     # [B, num_queries, 4]
        }
```

### 生成头 (Generation Head)

```python
class GenerationHead(nn.Module):
    def __init__(self, d_model, vocab_size, max_length=512):
        super().__init__()
        self.max_length = max_length
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, 8, d_model * 4),
            num_layers=6
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, fused_features, target_sequence=None, mode='train'):
        if mode == 'train':
            return self._forward_training(fused_features, target_sequence)
        else:
            return self._forward_inference(fused_features)
    
    def _forward_training(self, memory, target_sequence):
        # 目标序列嵌入
        tgt_emb = self.embedding(target_sequence)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # 生成因果掩码
        seq_len = target_sequence.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # 解码
        decoded = self.decoder(
            tgt=tgt_emb.transpose(0, 1),
            memory=memory.transpose(0, 1),
            tgt_mask=causal_mask
        ).transpose(0, 1)
        
        # 输出投影
        logits = self.output_projection(decoded)
        
        return logits
```

## 数据流分析

### 前向传播流程

1. **输入预处理**:
   ```
   Speech: [B, audio_len] → Mel-spectrogram → [B, T_audio, 80]
   Gesture: [B, T_gesture, 2, 21, 3] → Keypoint normalization
   Image: [B, 3, 224, 224] → Patch embedding → [B, 196, d_model]
   ```

2. **模态编码**:
   ```
   Speech: [B, T_audio, 80] → Transformer → [B, T_audio, d_model]
   Gesture: [B, T_gesture, 2, 21, 3] → GCN+CNN → [B, T_gesture, d_model]  
   Image: [B, 196, d_model] → ViT → [B, 196, d_model]
   ```

3. **特征对齐**:
   ```
   Target length: T_target = 256
   Speech: [B, T_audio, d_model] → [B, T_target, d_model]
   Gesture: [B, T_gesture, d_model] → [B, T_target, d_model]
   Image: [B, 196, d_model] → [B, T_target, d_model] (repeat/interpolate)
   ```

4. **跨模态融合**:
   ```
   Individual features → Cross-modal attention → Enhanced features
   Enhanced features → Hierarchical fusion → [B, T_target, d_model]
   ```

5. **任务输出**:
   ```
   Classification: [B, T_target, d_model] → Global pooling → [B, num_classes]
   Detection: [B, T_target, d_model] → Object queries → [B, num_queries, ...]
   Generation: [B, T_target, d_model] → Decoder → [B, seq_len, vocab_size]
   ```

### 内存使用分析

假设批次大小B=32, d_model=512:

| 组件 | 输入形状 | 输出形状 | 内存使用 (MB) |
|-----|----------|----------|---------------|
| SpeechEncoder | [32, 16000] | [32, 256, 512] | 16.8 |
| GestureEncoder | [32, 30, 2, 21, 3] | [32, 256, 512] | 16.8 |
| ImageEncoder | [32, 3, 224, 224] | [32, 256, 512] | 16.8 |
| CrossModalAttention | 3×[32, 256, 512] | 3×[32, 256, 512] | 50.3 |
| HierarchicalFusion | 3×[32, 256, 512] | [32, 256, 512] | 33.6 |
| ClassificationHead | [32, 256, 512] | [32, 10] | 0.001 |

**总内存使用**: ~134 MB (仅前向传播，不包括梯度)

### 计算复杂度分析

#### 时间复杂度

| 组件 | 复杂度 | 主要操作 |
|-----|--------|----------|
| Mel-spectrogram | O(N log N) | FFT变换 |
| Transformer (Speech) | O(T² × d) | 自注意力 |
| GCN (Gesture) | O(T × N × d) | 图卷积 |
| ViT (Image) | O(P² × d) | 自注意力 (P=patches) |
| Cross-modal Attention | O(T² × d × M) | M=模态数 |
| Hierarchical Fusion | O(T × d² × L) | L=融合层数 |

其中：
- N: 音频采样点数
- T: 序列长度  
- d: 模型维度
- P: 图像patch数量
- M: 模态数量

#### 空间复杂度

主要存储需求：
- **模型参数**: ~50M parameters ≈ 200MB (float32)
- **激活值**: ~150MB per batch
- **梯度**: ~200MB (训练时)
- **优化器状态**: ~400MB (Adam)

**总计**: ~950MB GPU内存 (batch_size=32)

## 扩展性设计

### 新模态接入

```python
class NewModalityEncoder(nn.Module):
    """新模态编码器接口"""
    def __init__(self, config):
        super().__init__()
        # 实现新模态的特征提取
    
    def forward(self, inputs):
        """
        Returns:
            features: [B, T, d_model] - 统一的特征格式
            attention_mask: [B, T] - 有效位置掩码
        """
        pass

# 在主模型中注册新模态
class TriModalFusionModel(nn.Module):
    def register_modality(self, modality_name, encoder_class, config):
        """动态注册新模态"""
        self.encoders[modality_name] = encoder_class(config)
        
        # 更新融合组件
        self.fusion_module.add_modality(modality_name)
```

### 新任务接入

```python
class CustomTaskHead(nn.Module):
    """自定义任务头接口"""
    def __init__(self, d_model, task_config):
        super().__init__()
        # 实现任务特定的输出层
    
    def forward(self, fused_features):
        """
        Args:
            fused_features: [B, T, d_model]
        Returns:
            task_output: 任务特定的输出格式
        """
        pass

# 注册新任务
model.add_task_head('custom_task', CustomTaskHead, task_config)
```

### 配置驱动扩展

```yaml
# 扩展配置示例
modalities:
  speech:
    encoder: "SpeechEncoder"
    config: {...}
  gesture: 
    encoder: "GestureEncoder"
    config: {...}
  image:
    encoder: "ImageEncoder" 
    config: {...}
  # 新模态
  lidar:
    encoder: "LidarEncoder"
    config: {...}

tasks:
  - name: "classification"
    head: "ClassificationHead"
    loss: "cross_entropy"
    weight: 1.0
  # 新任务
  - name: "segmentation"
    head: "SegmentationHead"
    loss: "dice_loss"
    weight: 0.5
```

这种架构设计确保了：
1. **模块化**: 每个组件独立实现，易于测试和维护
2. **可扩展**: 支持新模态和新任务的无缝接入
3. **配置驱动**: 通过配置文件控制模型行为
4. **高性能**: 优化的注意力机制和并行计算
5. **灵活性**: 支持不同的融合策略和输出任务