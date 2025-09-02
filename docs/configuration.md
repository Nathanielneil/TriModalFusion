# TriModalFusion 配置指南

## 目录
- [配置文件概述](#配置文件概述)
- [模型配置](#模型配置)
- [模态配置详解](#模态配置详解)
- [融合配置](#融合配置)
- [训练配置](#训练配置)
- [评估配置](#评估配置)
- [系统配置](#系统配置)
- [高级配置](#高级配置)
- [配置验证](#配置验证)
- [配置示例](#配置示例)

## 配置文件概述

TriModalFusion使用YAML格式的配置文件来控制模型架构、训练参数、数据处理等各个方面。配置系统采用分层设计，支持配置继承、环境变量替换和动态验证。

### 配置文件结构

```yaml
# 基础配置结构
model:              # 模型架构配置
  name: str         # 模型名称
  d_model: int      # 模型隐藏维度
  tasks: list       # 任务列表
  
speech_config:      # 语音模态配置
  sample_rate: int  # 采样率
  n_mels: int       # Mel滤波器数量
  
gesture_config:     # 手势模态配置
  num_hands: int    # 手部数量
  num_joints: int   # 关节点数量
  
image_config:       # 图像模态配置
  img_size: int     # 图像尺寸
  architecture: str # 架构类型
  
fusion_config:      # 融合配置
  strategy: str     # 融合策略
  alignment: str    # 对齐方法
  
training:           # 训练配置
  optimizer: str    # 优化器类型
  learning_rate: float # 学习率
  
evaluation:         # 评估配置
  metrics: list     # 评估指标
  
system:             # 系统配置
  device: str       # 计算设备
  precision: str    # 数值精度
```

## 模型配置

### 基础模型配置

```yaml
model:
  # 模型标识
  name: "TriModalFusion"           # 模型名称
  version: "1.0"                   # 模型版本
  description: "Unified multimodal recognition system"
  
  # 架构参数
  d_model: 512                     # 模型隐藏维度 [128, 256, 512, 768, 1024]
  target_seq_length: 256           # 目标序列长度 [64, 128, 256, 512]
  max_sequence_length: 1024        # 最大序列长度
  
  # 任务配置
  tasks: ["classification"]        # 支持的任务类型
  num_classes: 10                  # 分类任务类别数
  num_detection_classes: 80        # 检测任务类别数 (COCO)
  regression_dim: 1                # 回归任务输出维度
  vocab_size: 50000               # 生成任务词汇表大小
  
  # 正则化
  dropout: 0.1                     # Dropout概率 [0.0, 0.5]
  attention_dropout: 0.1           # 注意力Dropout概率
  activation_dropout: 0.0          # 激活函数Dropout概率
  
  # 初始化
  initializer_range: 0.02          # 参数初始化范围
  layer_norm_eps: 1e-12           # LayerNorm epsilon值
```

### 任务特定配置

```yaml
tasks:
  classification:
    enabled: true
    num_classes: 10
    loss_function: "cross_entropy"
    class_weights: null              # 类别权重 [null, "balanced", list]
    label_smoothing: 0.1             # 标签平滑系数
    
  detection:
    enabled: false
    num_classes: 80
    num_queries: 100                 # DETR查询数量
    loss_weights:
      class_loss: 2.0                # 分类损失权重
      bbox_loss: 5.0                 # 边框损失权重
      giou_loss: 2.0                 # GIoU损失权重
    
  regression:
    enabled: false
    output_dim: 1
    loss_function: "mse"             # 'mse', 'mae', 'huber'
    huber_delta: 1.0                 # Huber损失参数
    
  generation:
    enabled: false
    vocab_size: 50000
    max_length: 512
    beam_size: 4                     # 束搜索宽度
    length_penalty: 0.6              # 长度惩罚系数
    repetition_penalty: 1.2          # 重复惩罚系数
```

## 模态配置详解

### 语音配置 (speech_config)

```yaml
speech_config:
  # 音频处理
  sample_rate: 16000               # 采样率 [8000, 16000, 22050, 44100]
  audio_duration: 10.0             # 最大音频时长(秒)
  normalize_audio: true            # 是否归一化音频
  audio_gain: 1.0                  # 音频增益
  
  # Mel频谱图
  n_mels: 80                       # Mel滤波器数量 [40, 80, 128]
  n_fft: 1024                      # FFT窗口大小 [512, 1024, 2048]
  hop_length: 256                  # 跳跃长度
  win_length: 1024                 # 窗口长度
  window: "hann"                   # 窗口函数类型
  center: true                     # 是否中心化
  pad_mode: "reflect"              # 填充模式
  power: 2.0                       # 功率谱指数
  
  # 频谱增强
  freq_masking: false              # 频率掩码
  time_masking: false              # 时间掩码
  max_freq_mask: 27                # 最大频率掩码长度
  max_time_mask: 40                # 最大时间掩码长度
  num_freq_masks: 2                # 频率掩码数量
  num_time_masks: 2                # 时间掩码数量
  
  # Transformer配置
  encoder_layers: 6                # 编码器层数 [3, 6, 12]
  encoder_attention_heads: 8       # 注意力头数 [4, 8, 12, 16]
  encoder_ffn_dim: 2048           # 前馈网络维度
  
  # 位置编码
  positional_encoding: "sinusoidal"  # 'sinusoidal', 'learned'
  max_position_embeddings: 3000    # 最大位置嵌入数
  
  # 特征池化
  pooling: "mean"                  # 'mean', 'max', 'cls', 'attention'
  pooling_kernel_size: 2           # 池化核大小
  
  # 正则化
  dropout: 0.1                     # Dropout概率
  attention_dropout: 0.1           # 注意力Dropout
  activation_dropout: 0.0          # 激活Dropout
  layerdrop: 0.0                   # 层级Dropout
```

### 手势配置 (gesture_config)

```yaml
gesture_config:
  # 手部检测
  num_hands: 2                     # 检测手数 [1, 2]
  use_mediapipe: true              # 使用MediaPipe检测
  min_detection_confidence: 0.5    # 最小检测置信度 [0.1, 0.9]
  min_tracking_confidence: 0.5     # 最小跟踪置信度 [0.1, 0.9]
  model_complexity: 1              # MediaPipe模型复杂度 [0, 1]
  
  # 关键点配置
  num_joints: 21                   # 手部关键点数量
  input_dim: 3                     # 坐标维度 (x, y, z)
  keypoint_threshold: 0.5          # 关键点置信度阈值
  
  # 数据预处理
  normalize_keypoints: true        # 关键点归一化
  center_on_wrist: true           # 以手腕为中心
  scale_to_unit: true             # 缩放到单位尺度
  flip_augmentation: false         # 水平翻转增强
  rotation_augmentation: false     # 旋转增强
  noise_augmentation: false        # 噪声增强
  
  # 空间图卷积网络 (GCN)
  spatial_gcn_layers: 2            # GCN层数 [1, 2, 3]
  spatial_hidden_dim: 64           # 隐藏维度 [32, 64, 128]
  spatial_out_dim: 128             # 输出维度
  gcn_activation: "relu"           # 激活函数
  gcn_normalize: true              # 图归一化
  adjacency_type: "physical"       # 邻接矩阵类型 'physical', 'adaptive'
  
  # 手部骨架连接关系
  hand_skeleton:
    - [0, 1, 2, 3, 4]              # 拇指
    - [0, 5, 6, 7, 8]              # 食指
    - [0, 9, 10, 11, 12]           # 中指
    - [0, 13, 14, 15, 16]          # 无名指
    - [0, 17, 18, 19, 20]          # 小指
  
  # 时序卷积网络
  temporal_cnn_layers: 3           # CNN层数 [1, 2, 3]
  temporal_hidden_dim: 256         # 隐藏维度
  temporal_kernel_sizes: [3, 5, 7] # 多尺度卷积核
  temporal_stride: 1               # 卷积步长
  temporal_padding: "same"         # 填充方式
  temporal_dilation: 1             # 膨胀率
  
  # 特征聚合
  spatial_aggregation: "attention" # 'mean', 'max', 'attention'
  temporal_pooling: "mean"         # 时序池化方式
  hand_fusion: "concat"            # 双手融合方式 'concat', 'attention', 'mean'
  
  # 序列处理
  max_sequence_length: 128         # 最大序列长度
  sequence_stride: 1               # 序列步长
  sequence_overlap: 0.5            # 序列重叠率
  
  # 正则化
  dropout: 0.1                     # Dropout概率
  spatial_dropout: 0.1             # 空间Dropout
  temporal_dropout: 0.1            # 时序Dropout
```

### 图像配置 (image_config)

```yaml
image_config:
  # 图像预处理
  img_size: 224                    # 输入图像尺寸 [224, 256, 384, 448]
  in_channels: 3                   # 输入通道数
  interpolation: "bicubic"         # 插值方法
  
  # 数据增强
  random_crop: true                # 随机裁剪
  random_flip: true                # 随机翻转
  color_jitter: false              # 颜色抖动
  random_rotation: false           # 随机旋转
  gaussian_blur: false             # 高斯模糊
  
  # 归一化
  image_mean: [0.485, 0.456, 0.406]  # ImageNet均值
  image_std: [0.229, 0.224, 0.225]   # ImageNet标准差
  normalize: true                  # 是否归一化
  
  # 架构选择
  image_architecture: "vit"        # 'vit', 'cnn', 'hybrid'
  
  # Vision Transformer (ViT) 配置
  vit_config:
    patch_size: 16                 # Patch大小 [8, 16, 32]
    embed_dim: 768                 # 嵌入维度 [384, 768, 1024]
    depth: 12                      # 层数 [6, 12, 24]
    num_heads: 12                  # 注意力头数 [6, 12, 16]
    mlp_ratio: 4.0                 # MLP扩展比例
    qkv_bias: true                 # QKV偏置
    drop_rate: 0.0                 # Dropout率
    attn_drop_rate: 0.0            # 注意力Dropout率
    drop_path_rate: 0.1            # DropPath率
    use_cls_token: true            # 使用CLS token
    use_abs_pos_emb: true          # 绝对位置编码
    use_rel_pos_emb: false         # 相对位置编码
    
  # CNN配置
  cnn_config:
    architecture: "resnet50"       # 'resnet50', 'resnet101', 'efficientnet'
    pretrained: true               # 使用预训练权重
    frozen_stages: 1               # 冻结阶段数
    norm_layer: "batch_norm"       # 归一化层类型
    activation: "relu"             # 激活函数
    
    # ResNet特定配置
    replace_stride_with_dilation: [false, false, false]
    zero_init_residual: false
    groups: 1                      # 分组卷积组数
    width_per_group: 64           # 每组宽度
    
  # 特征提取
  feature_extraction_layers: ["layer4"]  # 特征提取层
  feature_pyramid: false          # 特征金字塔
  fpn_channels: 256              # FPN通道数
  
  # 目标检测配置 (DETR-style)
  detection_config:
    enabled: false                 # 启用检测
    num_queries: 100              # 查询数量
    num_classes: 80               # 检测类别数
    
    # 损失权重
    class_loss_coef: 2.0          # 分类损失系数
    bbox_loss_coef: 5.0           # 边框损失系数
    giou_loss_coef: 2.0           # GIoU损失系数
    
    # 匈牙利匹配
    cost_class: 2.0               # 分类匹配代价
    cost_bbox: 5.0                # 边框匹配代价
    cost_giou: 2.0                # GIoU匹配代价
  
  # 正则化
  dropout: 0.1                    # Dropout概率
  stochastic_depth: 0.1           # 随机深度概率
  layer_scale: false              # LayerScale
  layer_scale_init_value: 1e-6    # LayerScale初始值
```

## 融合配置

```yaml
fusion_config:
  # 时序对齐
  alignment_method: "interpolation"  # 'interpolation', 'attention', 'learned'
  target_sequence_length: 256     # 目标序列长度
  
  # 插值对齐配置
  interpolation_config:
    mode: "linear"                 # 插值模式 'linear', 'cubic'
    align_corners: false           # 对齐角点
    
  # 注意力对齐配置
  attention_alignment_config:
    num_heads: 8                   # 注意力头数
    dropout: 0.1                   # Dropout率
    temperature: 1.0               # 注意力温度
    
  # 学习对齐配置
  learned_alignment_config:
    hidden_dim: 256                # 隐藏维度
    num_layers: 2                  # 网络层数
    activation: "relu"             # 激活函数
  
  # 语义对齐
  semantic_alignment:
    enabled: true                  # 启用语义对齐
    projection_dim: 256            # 投影维度
    similarity_type: "cosine"      # 'cosine', 'bilinear', 'mlp'
    temperature: 0.07              # 对比学习温度
    
  # 跨模态注意力
  cross_modal_attention:
    strategy: "pairwise"           # 'pairwise', 'global', 'hierarchical'
    num_heads: 8                   # 注意力头数
    num_layers: 2                  # 注意力层数
    use_residual: true             # 残差连接
    
  # 融合策略
  fusion_strategy: "attention"     # 'concat', 'add', 'attention', 'adaptive'
  
  # 拼接融合配置
  concat_config:
    projection_dim: 512            # 投影维度
    
  # 注意力融合配置
  attention_fusion_config:
    num_heads: 8                   # 注意力头数
    key_dim: 64                    # Key维度
    value_dim: 64                  # Value维度
    
  # 自适应融合配置
  adaptive_fusion_config:
    gating_dim: 256               # 门控维度
    num_experts: 3                # 专家数量
    
  # 分层融合
  hierarchical_fusion:
    num_levels: 3                 # 融合层级数
    level_configs:
      - fusion_type: "concat"     # 特征级融合
        hidden_dim: 256
      - fusion_type: "attention"  # 语义级融合  
        hidden_dim: 512
      - fusion_type: "adaptive"   # 决策级融合
        hidden_dim: 512
        
  # 跳跃连接
  skip_connections: true          # 启用跳跃连接
  skip_projection_dim: 256        # 跳跃连接投影维度
  
  # 正则化
  alignment_weight: 0.1           # 对齐损失权重
  diversity_weight: 0.01          # 多样性损失权重
  dropout: 0.1                    # Dropout概率
  layer_norm: true                # 层归一化
```

## 训练配置

```yaml
training:
  # 基础训练参数
  max_epochs: 100                 # 最大训练轮数
  max_steps: 100000              # 最大训练步数 (优先级高于epochs)
  validate_every_n_epochs: 1     # 验证频率
  
  # 优化器配置
  optimizer: "adamw"             # 'adam', 'adamw', 'sgd', 'rmsprop'
  learning_rate: 1e-4            # 学习率
  weight_decay: 1e-2             # 权重衰减
  betas: [0.9, 0.999]           # Adam beta参数
  eps: 1e-8                      # Adam epsilon
  momentum: 0.9                  # SGD动量 (仅SGD)
  nesterov: false                # Nesterov动量 (仅SGD)
  
  # 学习率调度
  scheduler: "cosine"            # 'none', 'cosine', 'linear', 'polynomial'
  
  # 余弦退火配置
  cosine_config:
    T_max: 100000                # 最大步数
    eta_min: 1e-7                # 最小学习率
    warmup_steps: 4000           # 预热步数
    warmup_method: "linear"      # 预热方法 'linear', 'constant'
    
  # 线性调度配置
  linear_config:
    warmup_steps: 4000
    decay_steps: 96000
    end_factor: 0.01
    
  # 梯度处理
  gradient_clip_val: 1.0         # 梯度裁剪阈值
  gradient_clip_algorithm: "norm" # 'norm', 'value'
  accumulate_grad_batches: 1     # 梯度累积批次
  
  # 批次配置
  batch_size: 32                 # 训练批次大小
  eval_batch_size: 64            # 评估批次大小
  
  # 数据加载
  num_workers: 4                 # 数据加载进程数
  pin_memory: true               # 固定内存
  persistent_workers: true       # 持久化工作进程
  prefetch_factor: 2             # 预取因子
  
  # 多任务学习
  task_weights:                  # 任务损失权重
    classification: 1.0
    detection: 1.0
    regression: 1.0
    generation: 1.0
    
  # 课程学习
  curriculum_learning:
    enabled: false               # 启用课程学习
    start_epoch: 0               # 开始轮数
    schedule: "linear"           # 进度安排
    
  # 损失函数
  loss_config:
    classification:
      type: "cross_entropy"      # 'cross_entropy', 'focal'
      label_smoothing: 0.1       # 标签平滑
      class_weights: null        # 类别权重
      
    focal_loss:                  # Focal Loss配置
      alpha: 1.0                 # 平衡参数
      gamma: 2.0                 # 聚焦参数
      
  # 正则化
  dropout: 0.1                   # 全局Dropout率
  mixup_alpha: 0.0               # Mixup参数
  cutmix_alpha: 0.0              # CutMix参数
  
  # 早停
  early_stopping:
    enabled: true                # 启用早停
    patience: 10                 # 耐心值
    min_delta: 1e-4             # 最小改进量
    mode: "max"                  # 'min', 'max'
    monitor: "val_accuracy"      # 监控指标
    
  # 学习率监控
  lr_monitor:
    logging_interval: "step"     # 'step', 'epoch'
    log_momentum: false          # 记录动量
```

## 评估配置

```yaml
evaluation:
  # 基础评估设置
  compute_on_step: false         # 每步计算指标
  compute_on_epoch: true         # 每轮计算指标
  
  # 指标配置
  metrics:
    classification:
      - accuracy
      - precision_macro
      - recall_macro  
      - f1_macro
      - confusion_matrix
      
    detection:
      - map_50              # mAP@0.5
      - map_75              # mAP@0.75
      - map_50_95           # mAP@0.5:0.95
      
    speech:
      - wer                 # Word Error Rate
      - cer                 # Character Error Rate
      - bleu                # BLEU Score
      
    gesture:
      - gesture_accuracy
      - keypoint_error
      - temporal_consistency
      
    multimodal:
      - fusion_effectiveness
      - modality_contribution
      - cross_modal_similarity
  
  # 详细评估
  save_predictions: true         # 保存预测结果
  save_attention_maps: false     # 保存注意力图
  save_embeddings: false         # 保存嵌入向量
  
  # 验证设置
  val_check_interval: 1000       # 验证检查间隔(步数)
  validation_step_outputs: "all" # 'all', 'minimal'
  
  # 测试配置
  test_batch_size: 64           # 测试批次大小
  test_time_augmentation: false # 测试时数据增强
  num_test_samples: null        # 测试样本数量限制
  
  # 可视化
  visualization:
    enabled: true               # 启用可视化
    save_every_n_epochs: 5     # 保存频率
    max_samples_per_class: 10  # 每类最大样本数
    
    # 注意力可视化
    attention_visualization:
      enabled: false
      layers_to_visualize: ["last"]  # 'first', 'middle', 'last', 'all'
      heads_to_visualize: [0, 1, 2]  # 注意力头索引
      
  # 基准测试
  benchmark:
    enabled: false              # 启用基准测试
    datasets: []                # 基准数据集列表
    compute_flops: false        # 计算FLOPs
    compute_latency: false      # 计算延迟
    
  # 错误分析
  error_analysis:
    enabled: false              # 启用错误分析
    save_error_cases: true      # 保存错误案例
    max_error_samples: 100      # 最大错误样本数
```

## 系统配置

```yaml
system:
  # 硬件配置
  device: "auto"                # 'auto', 'cuda', 'cpu', 'cuda:0'
  num_gpus: 1                   # GPU数量 (分布式训练)
  precision: 16                 # 数值精度 [16, 32, 64]
  
  # 混合精度训练
  mixed_precision:
    enabled: true               # 启用混合精度
    opt_level: "O1"            # 优化级别 'O0', 'O1', 'O2', 'O3'
    loss_scale: "dynamic"       # 损失缩放 'dynamic', float
    
  # 模型编译
  compile_model: false          # 编译模型 (PyTorch 2.0+)
  compile_backend: "inductor"   # 编译后端
  compile_mode: "default"       # 编译模式
  
  # 检查点
  checkpoint:
    save_top_k: 3              # 保存最好的K个模型
    save_last: true            # 保存最后一个模型
    monitor: "val_accuracy"    # 监控指标
    mode: "max"                # 'min', 'max'
    save_weights_only: false   # 仅保存权重
    every_n_epochs: 1          # 保存频率
    
  # 日志配置
  logging:
    level: "INFO"              # 日志级别
    log_every_n_steps: 100     # 记录频率
    log_model: false           # 记录模型结构
    
    # WandB配置
    wandb:
      enabled: false           # 启用WandB
      project: "trimodal-fusion"
      entity: "your-team"
      tags: ["multimodal", "fusion"]
      
    # TensorBoard配置  
    tensorboard:
      enabled: true            # 启用TensorBoard
      log_graph: false         # 记录计算图
      
  # 分布式训练
  distributed:
    strategy: "ddp"            # 'ddp', 'ddp_spawn', 'fsdp'
    find_unused_parameters: false
    gradient_as_bucket_view: false
    
  # 内存优化
  memory_optimization:
    gradient_checkpointing: false  # 梯度检查点
    cpu_offload: false             # CPU卸载
    pin_memory: true               # 固定内存
    
  # 随机种子
  seed: 42                    # 随机种子
  deterministic: false        # 确定性模式
  benchmark: true             # 基准模式
  
  # 异常处理
  detect_anomaly: false       # 检测异常
  profiler: null              # 性能分析器 'simple', 'advanced'
```

## 高级配置

### 实验跟踪配置

```yaml
experiment:
  # 实验信息
  name: "trimodal_fusion_baseline"
  version: "1.0"
  description: "Baseline experiment with default configuration"
  tags: ["baseline", "multimodal", "fusion"]
  notes: "Initial experiment to establish baseline performance"
  
  # 实验组织
  group: "baseline_experiments"
  job_type: "train"
  
  # 超参数搜索
  hyperparameter_search:
    enabled: false
    method: "grid"             # 'grid', 'random', 'bayes'
    num_trials: 20
    
    # 搜索空间
    search_space:
      learning_rate:
        type: "loguniform"
        min: 1e-5
        max: 1e-2
      batch_size:
        type: "choice"
        values: [16, 32, 64]
      dropout:
        type: "uniform"  
        min: 0.0
        max: 0.3
```

### 数据配置

```yaml
data:
  # 数据集路径
  train_data: "./data/train"
  val_data: "./data/val"
  test_data: "./data/test"
  
  # 数据格式
  data_format: "hdf5"          # 'json', 'hdf5', 'tfrecord'
  
  # 预处理
  preprocessing:
    speech:
      trim_silence: true       # 删除静音
      normalize_volume: true   # 音量归一化
      
    gesture:
      smooth_keypoints: true   # 关键点平滑
      fill_missing: "interpolation"  # 缺失值填充
      
    image:
      resize_method: "bilinear"
      crop_method: "center"
  
  # 数据增强
  augmentation:
    enabled: true
    prob: 0.5                 # 增强概率
    
    speech_augmentation:
      speed_perturb: true     # 速度扰动
      volume_perturb: true    # 音量扰动
      noise_addition: false   # 噪声添加
      
    gesture_augmentation:  
      temporal_jitter: true   # 时间抖动
      spatial_jitter: false  # 空间抖动
      
    image_augmentation:
      random_brightness: true
      random_contrast: true
      random_saturation: false
  
  # 缓存配置
  cache:
    enabled: true             # 启用缓存
    cache_dir: "./cache"      # 缓存目录
    max_cache_size: "10GB"    # 最大缓存大小
```

### 部署配置

```yaml
deployment:
  # 模型导出
  export:
    format: "pytorch"         # 'pytorch', 'onnx', 'tensorrt'
    dynamic_axes: true        # 动态轴
    opset_version: 11         # ONNX opset版本
    
  # 推理优化
  inference:
    batch_size: 1             # 推理批次大小
    use_fp16: true            # 使用FP16
    use_tensorrt: false       # 使用TensorRT
    max_workspace_size: "1GB" # TensorRT工作空间
    
  # 服务配置
  serving:
    host: "0.0.0.0"          # 服务主机
    port: 8080               # 服务端口
    workers: 4               # 工作进程数
    timeout: 30              # 超时时间
```

## 配置验证

配置系统包含自动验证机制，确保配置的有效性：

```python
# 配置验证示例
from src.utils.config import ConfigValidator

def validate_config(config_path: str):
    validator = ConfigValidator()
    
    # 加载配置
    config = validator.load_config(config_path)
    
    # 验证配置
    is_valid, errors = validator.validate(config)
    
    if not is_valid:
        for error in errors:
            print(f"配置错误: {error}")
        return False
    
    return True
```

### 验证规则

1. **类型检查**: 确保参数类型正确
2. **范围检查**: 验证数值参数在合理范围内
3. **依赖检查**: 验证相关参数的一致性
4. **路径检查**: 验证文件和目录路径有效性
5. **设备检查**: 验证计算设备可用性

## 配置示例

### 小型模型配置

```yaml
# configs/small_model.yaml
model:
  d_model: 256
  target_seq_length: 128

speech_config:
  n_mels: 40
  encoder_layers: 3
  encoder_attention_heads: 4

gesture_config:
  spatial_hidden_dim: 32
  temporal_hidden_dim: 128

image_config:
  img_size: 224
  vit_config:
    embed_dim: 384
    depth: 6
    num_heads: 6

training:
  batch_size: 64
  learning_rate: 5e-4
```

### 高性能配置

```yaml
# configs/high_performance.yaml
model:
  d_model: 1024
  target_seq_length: 512

speech_config:
  n_mels: 128
  encoder_layers: 12
  encoder_attention_heads: 16

gesture_config:
  spatial_hidden_dim: 128
  temporal_hidden_dim: 512

image_config:
  img_size: 384
  vit_config:
    embed_dim: 1024
    depth: 24
    num_heads: 16

training:
  batch_size: 16
  learning_rate: 1e-4
  gradient_clip_val: 0.5

system:
  precision: 16
  mixed_precision:
    enabled: true
```

### 研究配置

```yaml
# configs/research.yaml
# 启用所有高级功能进行研究实验

evaluation:
  save_predictions: true
  save_attention_maps: true
  save_embeddings: true
  
  visualization:
    enabled: true
    attention_visualization:
      enabled: true
      layers_to_visualize: ["all"]
      
  error_analysis:
    enabled: true
    save_error_cases: true

fusion_config:
  semantic_alignment:
    enabled: true
  cross_modal_attention:
    strategy: "hierarchical"
  hierarchical_fusion:
    num_levels: 4

experiment:
  hyperparameter_search:
    enabled: true
    method: "bayes"
    num_trials: 50
```

这个配置系统确保了：
1. **灵活性**: 支持各种实验配置需求
2. **可维护性**: 清晰的配置结构和文档
3. **可扩展性**: 容易添加新的配置选项
4. **安全性**: 配置验证防止错误设置
5. **可重现性**: 完整记录实验配置