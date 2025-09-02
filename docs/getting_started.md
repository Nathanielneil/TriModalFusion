# TriModalFusion 快速开始指南

## 目录
- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [基础概念](#基础概念)
- [第一个示例](#第一个示例)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [下一步](#下一步)

## 系统要求

### 硬件要求

**最低配置:**
- CPU: Intel i5 或 AMD Ryzen 5 (4核心)
- 内存: 8GB RAM
- 存储: 10GB 可用空间
- GPU: 可选，推荐NVIDIA GTX 1060或更高

**推荐配置:**
- CPU: Intel i7 或 AMD Ryzen 7 (8核心)
- 内存: 16GB RAM
- 存储: 50GB 可用空间 (SSD推荐)
- GPU: NVIDIA RTX 3070 或更高，8GB+ 显存

**生产环境:**
- CPU: Intel Xeon 或 AMD EPYC (16+核心)
- 内存: 32GB+ RAM
- 存储: 100GB+ NVMe SSD
- GPU: NVIDIA A100, V100 或 RTX 4090

### 软件要求

**操作系统:**
- Ubuntu 18.04+ / CentOS 7+
- macOS 10.15+
- Windows 10+

**Python环境:**
- Python 3.8-3.11
- pip 21.0+
- conda (推荐)

**核心依赖:**
- PyTorch 2.0+
- torchvision 0.15+
- torchaudio 2.0+
- CUDA 11.8+ (GPU支持)

## 安装指南

### 方法1: Conda环境 (推荐)

```bash
# 1. 创建conda环境
conda create -n trimodal python=3.9
conda activate trimodal

# 2. 安装PyTorch (CUDA版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或者安装CPU版本
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 3. 克隆项目
git clone https://github.com/your-org/TriModalFusion.git
cd TriModalFusion

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装项目包 (开发模式)
pip install -e .
```

### 方法2: Docker环境

```bash
# 1. 拉取预构建镜像
docker pull trimodal/trimodal-fusion:latest

# 或者构建本地镜像
docker build -t trimodal-fusion .

# 2. 运行容器
docker run -it --gpus all -v $(pwd):/workspace trimodal-fusion:latest

# 3. 在容器中运行
cd /workspace
python examples/basic_usage.py
```

### 方法3: pip安装

```bash
# 1. 创建虚拟环境
python -m venv trimodal_env
source trimodal_env/bin/activate  # Linux/Mac
# trimodal_env\Scripts\activate    # Windows

# 2. 升级pip
pip install --upgrade pip

# 3. 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 克隆并安装项目
git clone https://github.com/your-org/TriModalFusion.git
cd TriModalFusion
pip install -r requirements.txt
pip install -e .
```

### 验证安装

```python
# test_installation.py
import torch
import sys
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 测试导入
try:
    from src.models.trimodal_fusion import TriModalFusionModel
    print("✓ TriModalFusion导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")

# 测试MediaPipe
try:
    import mediapipe as mp
    print("✓ MediaPipe导入成功")
except ImportError as e:
    print(f"✗ MediaPipe导入失败: {e}")

# 测试基础功能
try:
    from src.utils.config import load_config
    config = load_config("configs/default_config.yaml")
    model = TriModalFusionModel(config)
    print(f"✓ 模型创建成功，参数数量: {model.get_num_parameters():,}")
except Exception as e:
    print(f"✗ 模型创建失败: {e}")
```

## 快速开始

### 5分钟快速体验

```python
# quick_start.py
import torch
from src.models.trimodal_fusion import TriModalFusionModel
from src.utils.config import load_config

# 1. 加载配置
print("1. 加载配置...")
config = load_config("configs/default_config.yaml")

# 2. 创建模型
print("2. 创建模型...")
model = TriModalFusionModel(config)
print(f"   模型参数数量: {model.get_num_parameters():,}")

# 3. 准备示例数据
print("3. 准备示例数据...")
batch_size = 2
inputs = {
    'speech': torch.randn(batch_size, 16000),        # 1秒音频 @16kHz
    'gesture': torch.randn(batch_size, 30, 2, 21, 3), # 30帧手势，双手21个关键点
    'image': torch.randn(batch_size, 3, 224, 224)    # 224x224 RGB图像
}

# 4. 运行推理
print("4. 运行推理...")
model.eval()
with torch.no_grad():
    outputs = model(inputs)

# 5. 查看结果
print("5. 输出结果:")
print(f"   - 分类logits形状: {outputs['task_outputs']['classification'].shape}")
print(f"   - 融合特征形状: {outputs['fused_features'].shape}")
print("✓ 快速体验完成!")
```

### 使用预训练模型

```python
# pretrained_example.py
import torch
from src.models.trimodal_fusion import TriModalFusionModel
from src.utils.config import load_config

# 加载预训练模型
def load_pretrained_model(checkpoint_path):
    """加载预训练模型"""
    config = load_config("configs/pretrained_config.yaml")
    model = TriModalFusionModel(config)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

# 使用示例
model, config = load_pretrained_model("checkpoints/pretrained_model.pth")

# 准备真实数据 (替换为您的数据)
inputs = {
    'speech': torch.randn(1, 16000),
    'gesture': torch.randn(1, 30, 2, 21, 3),
    'image': torch.randn(1, 3, 224, 224)
}

# 推理
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    predictions = torch.softmax(outputs['task_outputs']['classification'], dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)

print(f"预测类别: {predicted_class.item()}")
print(f"置信度: {predictions.max().item():.4f}")
```

## 基础概念

### 核心组件

**1. 多模态编码器**
```
TriModalFusion 包含三个专门的编码器:
├── SpeechEncoder: 处理音频信号
│   └── 基于Transformer的语音特征提取
├── GestureEncoder: 处理手势序列  
│   └── MediaPipe + GCN + 时序CNN
└── ImageEncoder: 处理图像数据
    └── Vision Transformer 或 CNN
```

**2. 特征融合**
```
融合机制处理跨模态信息整合:
├── 时序对齐: 同步不同模态的时间信息
├── 语义对齐: 将特征映射到共同语义空间
└── 跨模态注意力: 学习模态间的相互关系
```

**3. 多任务输出**
```
支持多种下游任务:
├── 分类任务: 多类别分类
├── 检测任务: 目标检测 (可选)
├── 回归任务: 数值预测 (可选)
└── 生成任务: 序列生成 (可选)
```

### 数据格式

**输入数据格式:**
```python
inputs = {
    'speech': torch.Tensor,    # [B, audio_length]
    'gesture': torch.Tensor,   # [B, time, hands, joints, coords]
    'image': torch.Tensor      # [B, channels, height, width]
}

# 具体形状示例:
# speech: [2, 16000] - 2个样本，每个1秒@16kHz
# gesture: [2, 30, 2, 21, 3] - 2个样本，30帧，2只手，21个关节，3个坐标(x,y,z)
# image: [2, 3, 224, 224] - 2个样本，RGB图像224x224
```

**输出数据格式:**
```python
outputs = {
    'task_outputs': {
        'classification': torch.Tensor,  # [B, num_classes]
        'detection': dict,               # 检测结果 (如果启用)
        'regression': torch.Tensor,      # 回归结果 (如果启用)
    },
    'fused_features': torch.Tensor,      # [B, seq_len, d_model] 融合特征
    'encoded_features': {                # 各模态编码特征
        'speech': torch.Tensor,
        'gesture': torch.Tensor,
        'image': torch.Tensor
    },
    'attention_weights': dict            # 注意力权重 (可选)
}
```

## 第一个示例

让我们通过一个完整的示例来了解整个流程:

### 示例1: 基础分类任务

```python
# example_classification.py
import torch
import torch.nn.functional as F
from src.models.trimodal_fusion import TriModalFusionModel
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

# 设置日志
setup_logging()

def create_sample_data():
    """创建示例数据"""
    batch_size = 4
    
    # 模拟真实数据的分布
    inputs = {
        'speech': torch.randn(batch_size, 16000) * 0.5,  # 音频数据
        'gesture': torch.randn(batch_size, 30, 2, 21, 3) * 0.1,  # 手势关键点
        'image': torch.randn(batch_size, 3, 224, 224) * 2 - 1  # 图像数据 [-1, 1]
    }
    
    # 模拟分类标签
    targets = torch.randint(0, 10, (batch_size,))
    
    return inputs, targets

def main():
    print("=== TriModalFusion 分类示例 ===\n")
    
    # 1. 加载配置
    print("1. 加载配置...")
    config = load_config("configs/default_config.yaml")
    print(f"   配置文件加载成功，任务: {config.model.tasks}")
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    model = TriModalFusionModel(config)
    print(f"   模型创建成功")
    print(f"   - 总参数数量: {model.get_num_parameters():,}")
    print(f"   - 模型大小: {model.get_num_parameters() * 4 / 1024**2:.1f} MB")
    
    # 3. 准备数据
    print("\n3. 准备数据...")
    inputs, targets = create_sample_data()
    print(f"   - 语音输入: {inputs['speech'].shape}")
    print(f"   - 手势输入: {inputs['gesture'].shape}")
    print(f"   - 图像输入: {inputs['image'].shape}")
    print(f"   - 目标标签: {targets.shape}")
    
    # 4. 前向传播
    print("\n4. 前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    print("   前向传播完成!")
    print(f"   - 分类输出形状: {outputs['task_outputs']['classification'].shape}")
    print(f"   - 融合特征形状: {outputs['fused_features'].shape}")
    
    # 5. 预测结果
    print("\n5. 预测结果...")
    logits = outputs['task_outputs']['classification']
    probabilities = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    print("   样本预测结果:")
    for i in range(len(targets)):
        true_label = targets[i].item()
        pred_label = predictions[i].item()
        confidence = probabilities[i, pred_label].item()
        
        status = "✓" if pred_label == true_label else "✗"
        print(f"   {status} 样本{i+1}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.3f}")
    
    # 6. 特征分析
    print("\n6. 特征分析...")
    if 'encoded_features' in outputs:
        for modality, features in outputs['encoded_features'].items():
            print(f"   - {modality}编码特征: {features.shape}")
    
    # 7. 模态贡献度分析 (简化版)
    print("\n7. 模态重要性分析...")
    baseline_logits = outputs['task_outputs']['classification']
    baseline_confidence = F.softmax(baseline_logits, dim=-1).max(dim=-1)[0].mean()
    
    print(f"   - 完整模型置信度: {baseline_confidence:.4f}")
    
    # 测试单模态性能
    for modality in ['speech', 'gesture', 'image']:
        single_input = {modality: inputs[modality]}
        with torch.no_grad():
            single_output = model(single_input)
            single_logits = single_output['task_outputs']['classification']
            single_confidence = F.softmax(single_logits, dim=-1).max(dim=-1)[0].mean()
        
        print(f"   - 仅{modality}置信度: {single_confidence:.4f}")
    
    print("\n✓ 示例运行完成!")

if __name__ == "__main__":
    main()
```

### 示例2: 训练简单模型

```python
# example_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.trimodal_fusion import TriModalFusionModel
from src.utils.config import load_config

def create_synthetic_dataset(num_samples=1000):
    """创建合成数据集"""
    # 创建输入数据
    speech_data = torch.randn(num_samples, 16000) * 0.5
    gesture_data = torch.randn(num_samples, 30, 2, 21, 3) * 0.1
    image_data = torch.randn(num_samples, 3, 224, 224) * 2 - 1
    
    # 创建标签 (基于数据的某些简单规律)
    labels = ((speech_data.mean(dim=1) + 
               gesture_data.mean(dim=(1,2,3,4)) + 
               image_data.mean(dim=(1,2,3))) > 0).long()
    
    # 转换为字典格式
    inputs_list = []
    for i in range(num_samples):
        inputs_list.append({
            'speech': speech_data[i],
            'gesture': gesture_data[i],
            'image': image_data[i]
        })
    
    return inputs_list, labels

def collate_fn(batch):
    """自定义数据整理函数"""
    inputs_batch = {}
    targets_batch = []
    
    for inputs, target in batch:
        targets_batch.append(target)
        for modality, data in inputs.items():
            if modality not in inputs_batch:
                inputs_batch[modality] = []
            inputs_batch[modality].append(data)
    
    # 堆叠数据
    for modality in inputs_batch:
        inputs_batch[modality] = torch.stack(inputs_batch[modality])
    
    targets_batch = torch.stack(targets_batch)
    
    return inputs_batch, targets_batch

def train_simple_model():
    """训练简单模型示例"""
    print("=== TriModalFusion 训练示例 ===\n")
    
    # 1. 准备数据
    print("1. 准备数据...")
    inputs_list, labels = create_synthetic_dataset(num_samples=200)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(inputs_list))
    train_inputs = inputs_list[:train_size]
    train_labels = labels[:train_size]
    val_inputs = inputs_list[train_size:]
    val_labels = labels[train_size:]
    
    # 创建数据加载器
    train_dataset = list(zip(train_inputs, train_labels))
    val_dataset = list(zip(val_inputs, val_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print(f"   训练集: {len(train_dataset)}个样本")
    print(f"   验证集: {len(val_dataset)}个样本")
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    config = load_config("configs/default_config.yaml")
    # 调整为二分类
    config.model.num_classes = 2
    
    model = TriModalFusionModel(config)
    print(f"   模型参数数量: {model.get_num_parameters():,}")
    
    # 3. 设置训练
    print("\n3. 设置训练...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环
    print("\n4. 开始训练...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            logits = outputs['task_outputs']['classification']
            
            # 计算损失
            loss = criterion(logits, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"   Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                logits = outputs['task_outputs']['classification']
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # 输出结果
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"\n   Epoch {epoch+1}/{num_epochs}:")
        print(f"   训练 - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"   验证 - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    print("\n✓ 训练完成!")
    
    # 5. 保存模型
    print("\n5. 保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_val_acc': val_acc
    }, 'simple_trained_model.pth')
    
    print("   模型已保存至: simple_trained_model.pth")

if __name__ == "__main__":
    train_simple_model()
```

## 配置说明

### 基础配置结构

```yaml
# configs/getting_started.yaml
model:
  name: "TriModalFusion"
  d_model: 256                    # 较小的模型用于快速实验
  tasks: ["classification"]
  num_classes: 10

# 简化的模态配置
speech_config:
  sample_rate: 16000
  n_mels: 40                      # 减少计算量
  encoder_layers: 3               # 较少的层数
  encoder_attention_heads: 4

gesture_config:
  num_hands: 2
  spatial_hidden_dim: 32          # 较小的隐藏层
  temporal_hidden_dim: 64

image_config:
  img_size: 224
  image_architecture: "vit"
  vit_config:
    embed_dim: 256                # 较小的嵌入维度
    depth: 6                      # 较少的层数
    num_heads: 4

# 简单的融合配置
fusion_config:
  alignment_method: "interpolation"
  fusion_strategy: "attention"
  fusion_heads: 4

# 训练配置
training:
  batch_size: 16                  # 较小的批次大小
  learning_rate: 1e-4
  max_epochs: 10
  
# 系统配置
system:
  device: "auto"
  precision: 32                   # 使用float32确保稳定性
```

### 配置文件解释

1. **model**: 定义模型的基本架构参数
2. **speech_config**: 语音模态的特定配置
3. **gesture_config**: 手势模态的特定配置  
4. **image_config**: 图像模态的特定配置
5. **fusion_config**: 多模态融合的配置
6. **training**: 训练相关的配置
7. **system**: 系统和硬件相关的配置

## 常见问题

### Q1: 导入错误 "No module named 'src'"

**解决方案:**
```bash
# 确保在项目根目录下
cd TriModalFusion

# 安装为可编辑包
pip install -e .

# 或者添加到Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Q2: CUDA内存不足

**解决方案:**
```python
# 1. 减小批次大小
config.training.batch_size = 8  # 或更小

# 2. 启用混合精度训练
config.system.precision = 16

# 3. 启用梯度检查点
config.system.memory_optimization.gradient_checkpointing = True

# 4. 减小模型大小
config.model.d_model = 256
config.speech_config.encoder_layers = 3
```

### Q3: MediaPipe安装失败

**解决方案:**
```bash
# 方法1: 使用conda安装
conda install -c conda-forge mediapipe

# 方法2: 更新pip并重新安装
pip install --upgrade pip
pip install mediapipe

# 方法3: 如果仍然失败，禁用MediaPipe
# 在配置中设置:
# gesture_config:
#   use_mediapipe: false
```

### Q4: 模型训练很慢

**优化建议:**
```python
# 1. 使用更小的数据尺寸
config.speech_config.max_audio_length = 1000  # 减少音频长度
config.gesture_config.max_sequence_length = 64  # 减少手势序列长度
config.image_config.img_size = 128  # 减少图像尺寸

# 2. 减少模型复杂度
config.model.d_model = 256
config.fusion_config.fusion_heads = 4

# 3. 启用编译加速 (PyTorch 2.0+)
config.system.compile_model = True
```

### Q5: 多GPU训练出错

**解决方案:**
```bash
# 1. 确保所有GPU可见
nvidia-smi

# 2. 使用正确的启动方式
python -m torch.distributed.launch --nproc_per_node=2 train.py

# 3. 或使用torchrun
torchrun --nproc_per_node=2 train.py --config configs/default_config.yaml
```

### Q6: 预测结果不合理

**排查步骤:**
```python
# 1. 检查数据预处理
print("Input ranges:")
for modality, data in inputs.items():
    print(f"{modality}: min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}")

# 2. 检查模型输出
with torch.no_grad():
    outputs = model(inputs)
    logits = outputs['task_outputs']['classification']
    print(f"Logits range: min={logits.min():.3f}, max={logits.max():.3f}")
    
# 3. 检查梯度
model.train()
outputs = model(inputs)
loss = criterion(outputs['task_outputs']['classification'], targets)
loss.backward()

grad_norm = 0
for param in model.parameters():
    if param.grad is not None:
        grad_norm += param.grad.data.norm(2).item() ** 2
grad_norm = grad_norm ** 0.5
print(f"Gradient norm: {grad_norm:.3f}")
```

## 下一步

完成了基础设置，下一步可以继续学习:

1. **[训练指南](training_guide.md)** - 学习如何训练自己的模型
2. **[配置指南](configuration.md)** - 深入了解配置选项
3. **[模型架构](model_architecture.md)** - 理解模型内部结构
4. **[评估指南](evaluation.md)** - 学习如何评估模型性能

### 进阶示例

```python
# 运行更复杂的示例
python examples/multimodal_classification.py
python examples/attention_visualization.py
python examples/feature_extraction.py
```

### 自定义数据集

```python
# 准备您自己的数据
from src.data.custom_dataset import MultiModalDataset
from src.data.preprocessor import MultiModalPreprocessor

# 预处理数据
preprocessor = MultiModalPreprocessor(config)
processed_data = preprocessor.process_directory("path/to/your/data")

# 创建数据集
dataset = MultiModalDataset(processed_data, config)
```

### 社区和支持

- **GitHub Issues**: [报告问题](https://github.com/your-org/TriModalFusion/issues)
- **GitHub Discussions**: [社区讨论](https://github.com/your-org/TriModalFusion/discussions)
- **文档**: [完整文档](https://trimodal-fusion.readthedocs.io/)
- **示例**: [examples/](../examples/) 目录

使用愉快！
