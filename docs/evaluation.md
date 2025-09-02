# TriModalFusion 评估指南

## 目录
- [评估概述](#评估概述)
- [评估指标体系](#评估指标体系)
- [评估工具使用](#评估工具使用)
- [基准测试](#基准测试)
- [错误分析](#错误分析)
- [可视化分析](#可视化分析)
- [性能分析](#性能分析)
- [评估报告](#评估报告)

## 评估概述

TriModalFusion的评估系统提供了全面的多模态模型性能分析框架，涵盖单模态评估、跨模态评估和融合效果评估三个层面。

### 评估架构

```
评估系统架构
├── 单模态评估
│   ├── 语音识别评估 (WER, CER, BLEU)
│   ├── 手势识别评估 (准确率, 关键点误差)
│   └── 图像识别评估 (mAP, Top-k准确率)
├── 跨模态评估  
│   ├── 模态贡献度分析
│   ├── 跨模态相似性
│   └── 时序对齐质量
├── 融合效果评估
│   ├── 融合有效性
│   ├── 信息增益
│   └── 鲁棒性分析
└── 综合性能评估
    ├── 计算效率
    ├── 内存使用
    └── 推理速度
```

## 评估指标体系

### 语音识别指标

#### 1. Word Error Rate (WER)

```python
def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    计算词错误率
    
    WER = (S + D + I) / N
    其中：
    S = 替换错误数
    D = 删除错误数  
    I = 插入错误数
    N = 参考词汇总数
    """
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        
        # 使用编辑距离算法
        errors = editdistance.eval(pred_words, ref_words)
        total_errors += errors
        total_words += len(ref_words)
    
    return total_errors / total_words if total_words > 0 else 0.0

# 使用示例
predictions = ["hello world", "how are you"]
references = ["hello word", "how old are you"] 
wer_score = calculate_wer(predictions, references)
print(f"WER: {wer_score:.3f}")  # 输出: WER: 0.333
```

#### 2. Character Error Rate (CER)

```python
def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """计算字符错误率"""
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.lower())
        ref_chars = list(ref.lower())
        
        errors = editdistance.eval(pred_chars, ref_chars)
        total_errors += errors
        total_chars += len(ref_chars)
    
    return total_errors / total_chars if total_chars > 0 else 0.0
```

#### 3. BLEU Score

```python
def calculate_bleu(predictions: List[str], references: List[str], n: int = 4) -> float:
    """
    计算BLEU分数
    
    BLEU = BP × exp(∑(1/n) × log(p_n))
    其中：
    BP = 简洁性惩罚
    p_n = n-gram精确率
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        if len(tokens) < n:
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def calculate_bleu_for_pair(pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # 计算n-gram精确率
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = get_ngrams(pred_tokens, i)
            ref_ngrams = get_ngrams(ref_tokens, i)
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                         for ngram in pred_ngrams)
            total = sum(pred_ngrams.values())
            
            precisions.append(matches / total if total > 0 else 0.0)
        
        # 计算简洁性惩罚
        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)
        
        if pred_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / pred_len)
        
        # 计算几何平均
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
        else:
            geo_mean = 0.0
        
        return bp * geo_mean
    
    scores = [calculate_bleu_for_pair(pred, ref) 
              for pred, ref in zip(predictions, references)]
    
    return sum(scores) / len(scores) if scores else 0.0
```

### 手势识别指标

#### 1. 手势分类准确率

```python
def gesture_classification_metrics(predictions: torch.Tensor, 
                                 targets: torch.Tensor) -> Dict[str, float]:
    """计算手势分类指标"""
    if predictions.dim() == 2:
        predicted_classes = torch.argmax(predictions, dim=1)
    else:
        predicted_classes = predictions
    
    # 准确率
    accuracy = (predicted_classes == targets).float().mean().item()
    
    # 转换为numpy进行sklearn计算
    pred_np = predicted_classes.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # 精确率、召回率、F1分数
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_np, pred_np, average='weighted', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(target_np, pred_np)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
```

#### 2. 关键点定位误差

```python
def keypoint_localization_error(predicted_keypoints: torch.Tensor,
                              target_keypoints: torch.Tensor,
                              valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    计算关键点定位误差
    
    参数:
        predicted_keypoints: [B, T, N, 2/3] 预测关键点
        target_keypoints: [B, T, N, 2/3] 真实关键点
        valid_mask: [B, T, N] 有效关键点掩码
    """
    # 计算欧式距离
    diff = predicted_keypoints - target_keypoints
    distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [B, T, N]
    
    if valid_mask is not None:
        distances = distances * valid_mask
        valid_points = valid_mask.sum()
    else:
        valid_points = torch.numel(distances)
    
    if valid_points == 0:
        return {'mse': 0.0, 'mae': 0.0, 'pck_0.1': 0.0, 'pck_0.2': 0.0}
    
    # 均方误差 (MSE)
    mse = torch.mean(distances ** 2).item()
    
    # 平均绝对误差 (MAE)
    mae = torch.mean(distances).item()
    
    # Percentage of Correct Keypoints (PCK)
    pck_01 = (distances < 0.1).float().mean().item()  # 阈值0.1
    pck_02 = (distances < 0.2).float().mean().item()  # 阈值0.2
    
    return {
        'mse': mse,
        'mae': mae, 
        'pck_0.1': pck_01,
        'pck_0.2': pck_02
    }
```

#### 3. 时序一致性

```python
def temporal_consistency(predictions: torch.Tensor, window_size: int = 5) -> float:
    """
    计算时序预测的一致性
    
    参数:
        predictions: [B, T, ...] 时序预测
        window_size: 滑动窗口大小
    """
    if predictions.dim() == 3:  # 分类logits
        predictions = torch.argmax(predictions, dim=-1)  # [B, T]
    
    B, T = predictions.shape
    consistency_scores = []
    
    for b in range(B):
        for t in range(T - window_size + 1):
            window = predictions[b, t:t + window_size]
            
            # 计算窗口内最频繁类别的比例
            unique_values, counts = torch.unique(window, return_counts=True)
            max_count = torch.max(counts).item()
            consistency = max_count / window_size
            
            consistency_scores.append(consistency)
    
    return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
```

### 图像识别指标

#### 1. 图像分类指标

```python
def image_classification_metrics(predictions: torch.Tensor,
                               targets: torch.Tensor) -> Dict[str, float]:
    """计算图像分类指标"""
    
    # Top-1准确率
    if predictions.dim() == 2:
        top1_acc = (torch.argmax(predictions, dim=1) == targets).float().mean().item()
        
        # Top-5准确率
        _, top5_pred = torch.topk(predictions, 5, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top5_pred)
        top5_acc = torch.any(top5_pred == targets_expanded, dim=1).float().mean().item()
    else:
        top1_acc = (predictions == targets).float().mean().item()
        top5_acc = top1_acc  # 对于已经是类别的预测
    
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc
    }
```

#### 2. 目标检测指标 (mAP)

```python
def calculate_map(predictions: List[Dict], 
                  targets: List[Dict],
                  iou_thresholds: List[float] = None) -> Dict[str, float]:
    """
    计算平均精度均值 (mAP)
    
    参数:
        predictions: 预测结果列表，每个元素包含 'boxes', 'scores', 'labels'
        targets: 真实标签列表，每个元素包含 'boxes', 'labels'
        iou_thresholds: IoU阈值列表
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """计算两个边界框的IoU"""
        # box格式: [x1, y1, x2, y2]
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return (intersection / union).item() if union > 0 else 0.0
    
    def calculate_ap_for_class_and_threshold(class_id: int, iou_threshold: float) -> float:
        """计算特定类别和IoU阈值下的AP"""
        # 收集该类别的所有预测和真实标注
        all_predictions = []
        all_targets = []
        
        for pred_dict, target_dict in zip(predictions, targets):
            # 预测
            if 'labels' in pred_dict:
                class_mask = pred_dict['labels'] == class_id
                if torch.any(class_mask):
                    boxes = pred_dict['boxes'][class_mask]
                    scores = pred_dict['scores'][class_mask]
                    for box, score in zip(boxes, scores):
                        all_predictions.append({
                            'box': box,
                            'score': score.item(),
                            'matched': False
                        })
            
            # 真实标注
            if 'labels' in target_dict:
                class_mask = target_dict['labels'] == class_id
                if torch.any(class_mask):
                    boxes = target_dict['boxes'][class_mask]
                    for box in boxes:
                        all_targets.append({
                            'box': box,
                            'matched': False
                        })
        
        if not all_predictions:
            return 0.0
        
        # 按置信度排序
        all_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算TP和FP
        tp = []
        fp = []
        
        for pred in all_predictions:
            best_iou = 0.0
            best_target_idx = -1
            
            for idx, target in enumerate(all_targets):
                if target['matched']:
                    continue
                
                iou = calculate_iou(pred['box'], target['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = idx
            
            if best_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                all_targets[best_target_idx]['matched'] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        num_targets = len(all_targets)
        recalls = tp_cumsum / num_targets if num_targets > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # 计算AP (使用11点插值)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            precision_at_t = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
            ap += precision_at_t
        
        return ap / 11.0
    
    # 收集所有类别
    all_classes = set()
    for pred_dict in predictions:
        if 'labels' in pred_dict:
            all_classes.update(pred_dict['labels'].tolist())
    for target_dict in targets:
        if 'labels' in target_dict:
            all_classes.update(target_dict['labels'].tolist())
    
    if not all_classes:
        return {'mAP': 0.0}
    
    # 计算每个类别在每个IoU阈值下的AP
    results = {}
    
    for iou_thresh in iou_thresholds:
        class_aps = []
        for class_id in all_classes:
            ap = calculate_ap_for_class_and_threshold(class_id, iou_thresh)
            class_aps.append(ap)
        
        results[f'mAP@{iou_thresh:.2f}'] = sum(class_aps) / len(class_aps)
    
    # 计算平均mAP
    results['mAP'] = sum(results.values()) / len(results)
    
    return results
```

### 多模态融合指标

#### 1. 模态贡献度分析

```python
def modality_contribution_analysis(model: nn.Module,
                                 inputs: Dict[str, torch.Tensor],
                                 num_ablations: int = 10) -> Dict[str, float]:
    """
    分析各模态的贡献度
    
    使用留一法 (Leave-One-Out) 和随机遮蔽评估各模态重要性
    """
    model.eval()
    contributions = {}
    
    # 获取完整模型的基线性能
    with torch.no_grad():
        baseline_output = model(inputs)
        if isinstance(baseline_output, dict) and 'task_outputs' in baseline_output:
            baseline_logits = baseline_output['task_outputs']['classification']
        else:
            baseline_logits = baseline_output
        
        baseline_confidence = torch.softmax(baseline_logits, dim=-1).max(dim=-1)[0].mean()
    
    # 测试去除每个模态的影响
    for modality in inputs.keys():
        # 创建缺失该模态的输入
        ablated_inputs = {k: v for k, v in inputs.items() if k != modality}
        
        if not ablated_inputs:  # 如果只有一个模态
            contributions[modality] = 1.0
            continue
        
        # 评估性能下降
        with torch.no_grad():
            ablated_output = model(ablated_inputs)
            if isinstance(ablated_output, dict) and 'task_outputs' in ablated_output:
                ablated_logits = ablated_output['task_outputs']['classification']
            else:
                ablated_logits = ablated_output
            
            ablated_confidence = torch.softmax(ablated_logits, dim=-1).max(dim=-1)[0].mean()
        
        # 计算贡献度（性能差异）
        contribution = (baseline_confidence - ablated_confidence).item()
        contributions[modality] = max(0.0, contribution)  # 确保非负
    
    # 归一化贡献度
    total_contribution = sum(contributions.values())
    if total_contribution > 0:
        contributions = {k: v / total_contribution for k, v in contributions.items()}
    
    return contributions
```

#### 2. 跨模态相似性

```python
def cross_modal_similarity(features_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    计算跨模态特征相似性
    
    使用余弦相似度度量不同模态特征的相关性
    """
    import itertools
    
    similarities = {}
    
    # 对特征进行全局池化
    pooled_features = {}
    for modality, features in features_dict.items():
        if features.dim() > 2:
            pooled = torch.mean(features, dim=1)  # [B, D]
        else:
            pooled = features
        
        # L2归一化
        pooled_features[modality] = F.normalize(pooled, dim=-1)
    
    # 计算所有模态对的相似性
    modalities = list(pooled_features.keys())
    for mod1, mod2 in itertools.combinations(modalities, 2):
        feat1 = pooled_features[mod1]
        feat2 = pooled_features[mod2]
        
        # 余弦相似度
        similarity = torch.mean(torch.sum(feat1 * feat2, dim=-1)).item()
        similarities[f"{mod1}_{mod2}"] = similarity
    
    return similarities
```

#### 3. 融合有效性评估

```python
def fusion_effectiveness_analysis(unimodal_results: Dict[str, float],
                                multimodal_result: float) -> Dict[str, float]:
    """
    评估多模态融合的有效性
    
    参数:
        unimodal_results: 单模态结果字典 {'speech': 0.75, 'gesture': 0.68, ...}
        multimodal_result: 多模态融合结果
    """
    if not unimodal_results:
        return {'improvement': 0.0, 'relative_improvement': 0.0}
    
    best_unimodal = max(unimodal_results.values())
    mean_unimodal = sum(unimodal_results.values()) / len(unimodal_results)
    
    # 绝对改进
    improvement = multimodal_result - best_unimodal
    
    # 相对改进
    relative_improvement = improvement / best_unimodal if best_unimodal > 0 else 0.0
    
    # 融合效率 (相对于简单平均的改进)
    fusion_efficiency = multimodal_result - mean_unimodal
    
    return {
        'improvement': improvement,
        'relative_improvement': relative_improvement,
        'fusion_efficiency': fusion_efficiency,
        'best_unimodal': best_unimodal,
        'mean_unimodal': mean_unimodal,
        'multimodal': multimodal_result
    }
```

## 评估工具使用

### 评估器主类

```python
# src/evaluation/evaluator.py
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path

class MultiModalEvaluator:
    """多模态评估器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # 初始化各种评估指标
        self.speech_metrics = SpeechRecognitionMetrics()
        self.gesture_metrics = GestureRecognitionMetrics()
        self.image_metrics = ImageRecognitionMetrics()
        self.fusion_metrics = MultiModalFusionMetrics()
        
        # 存储评估结果
        self.results = {}
        
    def evaluate(self, dataloader: DataLoader, 
                save_predictions: bool = False,
                save_attention: bool = False) -> Dict[str, Any]:
        """
        完整评估流程
        
        参数:
            dataloader: 数据加载器
            save_predictions: 是否保存预测结果
            save_attention: 是否保存注意力权重
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_outputs = []
        all_attention_weights = []
        
        # 收集所有预测结果
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # 前向传播
                outputs = self.model(inputs)
                
                # 存储结果
                all_predictions.append(outputs)
                all_targets.append(targets)
                all_outputs.append(outputs)
                
                # 保存注意力权重（如果需要）
                if save_attention and hasattr(outputs, 'attention_weights'):
                    all_attention_weights.append(outputs.attention_weights)
                
                # 记录进度
                if batch_idx % 100 == 0:
                    print(f"评估进度: {batch_idx}/{len(dataloader)}")
        
        # 计算各类指标
        results = {}
        
        # 1. 基础任务指标
        results.update(self._evaluate_classification(all_predictions, all_targets))
        results.update(self._evaluate_detection(all_predictions, all_targets))
        results.update(self._evaluate_regression(all_predictions, all_targets))
        
        # 2. 模态特定指标
        results.update(self._evaluate_speech_recognition(all_predictions, all_targets))
        results.update(self._evaluate_gesture_recognition(all_predictions, all_targets))
        results.update(self._evaluate_image_recognition(all_predictions, all_targets))
        
        # 3. 多模态融合指标
        results.update(self._evaluate_multimodal_fusion(all_predictions, all_targets))
        
        # 4. 模型分析
        results.update(self._analyze_model_performance(all_predictions, all_targets))
        
        # 保存结果
        self.results = results
        
        if save_predictions:
            self._save_predictions(all_predictions, all_targets)
        
        if save_attention and all_attention_weights:
            self._save_attention_weights(all_attention_weights)
        
        return results
    
    def _evaluate_classification(self, predictions: List, targets: List) -> Dict[str, float]:
        """评估分类任务"""
        if 'classification' not in self.config.model.tasks:
            return {}
        
        all_preds = []
        all_targets = []
        
        for pred_batch, target_batch in zip(predictions, targets):
            if 'task_outputs' in pred_batch and 'classification' in pred_batch['task_outputs']:
                preds = pred_batch['task_outputs']['classification']
                targets_cls = target_batch.get('classification', target_batch)
                
                all_preds.append(preds)
                all_targets.append(targets_cls)
        
        if not all_preds:
            return {}
        
        # 拼接所有批次
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算分类指标
        metrics = ClassificationMetrics.compute_all_metrics(
            all_preds, all_targets, 
            num_classes=self.config.model.num_classes
        )
        
        # 添加前缀
        return {f'classification_{k}': v for k, v in metrics.items()}
    
    def _evaluate_detection(self, predictions: List, targets: List) -> Dict[str, float]:
        """评估检测任务"""
        if 'detection' not in self.config.model.tasks:
            return {}
        
        pred_dicts = []
        target_dicts = []
        
        for pred_batch, target_batch in zip(predictions, targets):
            if 'task_outputs' in pred_batch and 'detection' in pred_batch['task_outputs']:
                detection_output = pred_batch['task_outputs']['detection']
                
                # 转换为检测格式
                for i in range(detection_output['class_logits'].size(0)):
                    pred_dicts.append({
                        'boxes': detection_output['bbox_coords'][i],
                        'scores': torch.softmax(detection_output['class_logits'][i], dim=-1).max(dim=-1)[0],
                        'labels': torch.argmax(detection_output['class_logits'][i], dim=-1)
                    })
                
                # 目标格式
                if 'detection' in target_batch:
                    for i in range(len(target_batch['detection']['boxes'])):
                        target_dicts.append({
                            'boxes': target_batch['detection']['boxes'][i],
                            'labels': target_batch['detection']['labels'][i]
                        })
        
        if not pred_dicts or not target_dicts:
            return {}
        
        # 计算检测指标
        detection_metrics = DetectionMetrics.compute_detection_metrics(pred_dicts, target_dicts)
        
        return {f'detection_{k}': v for k, v in detection_metrics.items()}
    
    def _evaluate_multimodal_fusion(self, predictions: List, targets: List) -> Dict[str, float]:
        """评估多模态融合效果"""
        fusion_results = {}
        
        # 1. 模态贡献度分析
        if len(predictions) > 0:
            sample_inputs = predictions[0].get('inputs', {})
            if sample_inputs:
                contributions = modality_contribution_analysis(self.model, sample_inputs)
                fusion_results.update({f'contribution_{k}': v for k, v in contributions.items()})
        
        # 2. 跨模态相似性
        all_features = []
        for pred_batch in predictions:
            if 'encoded_features' in pred_batch:
                all_features.append(pred_batch['encoded_features'])
        
        if all_features:
            # 使用第一个批次计算相似性
            similarities = cross_modal_similarity(all_features[0])
            fusion_results.update({f'similarity_{k}': v for k, v in similarities.items()})
        
        return fusion_results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成评估报告"""
        if not self.results:
            return "无评估结果可报告"
        
        report = []
        report.append("=" * 60)
        report.append("TriModalFusion 评估报告")
        report.append("=" * 60)
        report.append("")
        
        # 1. 总体性能
        report.append("总体性能:")
        report.append("-" * 30)
        
        key_metrics = ['classification_accuracy', 'detection_mAP@0.5', 'fusion_effectiveness']
        for metric in key_metrics:
            if metric in self.results:
                report.append(f"{metric}: {self.results[metric]:.4f}")
        report.append("")
        
        # 2. 各模态性能
        report.append("各模态性能:")
        report.append("-" * 30)
        
        modalities = ['speech', 'gesture', 'image']
        for modality in modalities:
            modality_metrics = {k: v for k, v in self.results.items() 
                              if k.startswith(modality)}
            if modality_metrics:
                report.append(f"{modality.capitalize()}:")
                for k, v in modality_metrics.items():
                    report.append(f"  {k}: {v:.4f}")
        report.append("")
        
        # 3. 融合分析
        report.append("融合效果分析:")
        report.append("-" * 30)
        
        fusion_metrics = {k: v for k, v in self.results.items() 
                         if k.startswith('fusion') or k.startswith('contribution')}
        for k, v in fusion_metrics.items():
            report.append(f"{k}: {v:.4f}")
        report.append("")
        
        # 4. 性能总结
        report.append("性能总结:")
        report.append("-" * 30)
        
        if 'classification_accuracy' in self.results:
            acc = self.results['classification_accuracy']
            if acc > 0.9:
                report.append("✓ 分类性能优秀")
            elif acc > 0.8:
                report.append("✓ 分类性能良好")
            else:
                report.append("⚠ 分类性能需要改进")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
```

### 使用示例

```python
# evaluate_model.py
import torch
from torch.utils.data import DataLoader
from src.models.trimodal_fusion import TriModalFusionModel
from src.evaluation.evaluator import MultiModalEvaluator
from src.utils.config import load_config

def main():
    # 加载配置和模型
    config = load_config('configs/default_config.yaml')
    model = TriModalFusionModel(config)
    
    # 加载预训练权重
    checkpoint = torch.load('checkpoints/best_model.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    
    # 创建测试数据加载器
    test_dataset = MultiModalDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建评估器
    evaluator = MultiModalEvaluator(model, config)
    
    # 运行评估
    results = evaluator.evaluate(
        dataloader=test_loader,
        save_predictions=True,
        save_attention=True
    )
    
    # 生成报告
    report = evaluator.generate_report(save_path='evaluation_report.txt')
    print(report)
    
    # 保存详细结果
    import json
    with open('detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
```

## 基准测试

### 标准数据集评估

```python
# scripts/benchmark_evaluation.py
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

class BenchmarkEvaluator:
    """基准测试评估器"""
    
    def __init__(self):
        self.benchmark_datasets = {
            'speech': ['LibriSpeech', 'CommonVoice', 'VCTK'],
            'gesture': ['NTU-RGB+D', 'MSR-Action3D', 'JHMDB'],
            'image': ['ImageNet', 'COCO', 'Pascal VOC'],
            'multimodal': ['AVA', 'Kinetics', 'How2']
        }
        
    def run_benchmarks(self, model, config) -> Dict[str, Dict[str, float]]:
        """运行所有基准测试"""
        results = {}
        
        for modality, datasets in self.benchmark_datasets.items():
            results[modality] = {}
            
            for dataset_name in datasets:
                print(f"评估 {modality} - {dataset_name}")
                
                # 加载数据集
                dataset = self._load_benchmark_dataset(dataset_name, config)
                if dataset is None:
                    continue
                
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
                
                # 运行评估
                evaluator = MultiModalEvaluator(model, config)
                dataset_results = evaluator.evaluate(dataloader)
                
                results[modality][dataset_name] = dataset_results
        
        return results
    
    def compare_with_baselines(self, our_results: Dict, 
                             baseline_results: Dict) -> Dict[str, float]:
        """与基线模型比较"""
        comparisons = {}
        
        for metric in our_results:
            if metric in baseline_results:
                our_score = our_results[metric]
                baseline_score = baseline_results[metric]
                
                improvement = (our_score - baseline_score) / baseline_score * 100
                comparisons[f"{metric}_improvement"] = improvement
        
        return comparisons
```

### 性能基准

```python
# scripts/performance_benchmark.py
import time
import torch
import numpy as np
from typing import Dict, List

class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def benchmark_inference_speed(self, inputs: Dict[str, torch.Tensor], 
                                num_runs: int = 100) -> Dict[str, float]:
        """基准测试推理速度"""
        self.model.eval()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(inputs)
        
        # 同步GPU
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # 计时
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(inputs)
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }
    
    def benchmark_memory_usage(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基准测试内存使用"""
        if not self.device.startswith('cuda'):
            return {'error': '仅支持CUDA设备'}
        
        torch.cuda.reset_peak_memory_stats()
        
        # 测量推理内存
        with torch.no_grad():
            outputs = self.model(inputs)
        
        inference_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # 测量训练内存
        torch.cuda.reset_peak_memory_stats()
        
        outputs = self.model(inputs)
        dummy_loss = outputs['task_outputs']['classification'].sum()
        dummy_loss.backward()
        
        training_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'inference_memory_mb': inference_memory,
            'training_memory_mb': training_memory,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        }
    
    def benchmark_throughput(self, batch_sizes: List[int]) -> Dict[int, Dict[str, float]]:
        """基准测试不同批次大小的吞吐量"""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"测试批次大小: {batch_size}")
            
            # 创建测试数据
            inputs = {
                'speech': torch.randn(batch_size, 16000).to(self.device),
                'gesture': torch.randn(batch_size, 30, 2, 21, 3).to(self.device),
                'image': torch.randn(batch_size, 3, 224, 224).to(self.device)
            }
            
            try:
                # 测试推理速度
                speed_results = self.benchmark_inference_speed(inputs, num_runs=50)
                
                # 测试内存使用
                memory_results = self.benchmark_memory_usage(inputs)
                
                results[batch_size] = {
                    'throughput_samples_per_sec': batch_size * speed_results['fps'],
                    'latency_ms': speed_results['mean_time'] * 1000,
                    'memory_mb': memory_results['inference_memory_mb']
                }
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    results[batch_size] = {'error': 'OOM'}
                    break
                else:
                    raise e
        
        return results
```

## 错误分析

### 混淆矩阵分析

```python
def analyze_confusion_matrix(predictions: torch.Tensor, 
                           targets: torch.Tensor,
                           class_names: List[str] = None) -> Dict[str, Any]:
    """分析混淆矩阵"""
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if predictions.dim() == 2:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
    
    pred_np = pred_classes.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(target_np, pred_np)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 详细分类报告
    report = classification_report(target_np, pred_np, 
                                 target_names=class_names,
                                 output_dict=True)
    
    # 错误分析
    error_analysis = {}
    
    # 最容易混淆的类别对
    np.fill_diagonal(cm, 0)  # 移除对角线
    max_confusion_idx = np.unravel_index(np.argmax(cm), cm.shape)
    
    error_analysis['most_confused_pair'] = {
        'true_class': class_names[max_confusion_idx[0]] if class_names else max_confusion_idx[0],
        'pred_class': class_names[max_confusion_idx[1]] if class_names else max_confusion_idx[1],
        'count': cm[max_confusion_idx]
    }
    
    # 性能最差的类别
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
    worst_class_idx = np.argmin(class_accuracies)
    
    error_analysis['worst_performing_class'] = {
        'class': class_names[worst_class_idx] if class_names else worst_class_idx,
        'accuracy': class_accuracies[worst_class_idx]
    }
    
    return {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'error_analysis': error_analysis
    }
```

### 失败案例分析

```python
def analyze_failure_cases(model, dataloader, num_cases: int = 50) -> Dict[str, Any]:
    """分析失败案例"""
    model.eval()
    
    failure_cases = []
    correct_cases = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            
            if 'task_outputs' in outputs and 'classification' in outputs['task_outputs']:
                predictions = outputs['task_outputs']['classification']
                pred_classes = torch.argmax(predictions, dim=1)
                
                # 找出错误预测
                incorrect_mask = pred_classes != targets
                correct_mask = pred_classes == targets
                
                # 收集失败案例
                for i in range(len(inputs['speech'])):
                    case_data = {
                        'inputs': {k: v[i] for k, v in inputs.items()},
                        'predicted': pred_classes[i].item(),
                        'target': targets[i].item(),
                        'confidence': torch.softmax(predictions[i], dim=0).max().item(),
                        'outputs': outputs
                    }
                    
                    if incorrect_mask[i] and len(failure_cases) < num_cases:
                        failure_cases.append(case_data)
                    elif correct_mask[i] and len(correct_cases) < num_cases:
                        correct_cases.append(case_data)
            
            if len(failure_cases) >= num_cases and len(correct_cases) >= num_cases:
                break
    
    # 分析失败原因
    failure_analysis = {
        'low_confidence_errors': [],
        'high_confidence_errors': [],
        'common_error_patterns': {}
    }
    
    for case in failure_cases:
        if case['confidence'] < 0.6:
            failure_analysis['low_confidence_errors'].append(case)
        else:
            failure_analysis['high_confidence_errors'].append(case)
    
    return {
        'failure_cases': failure_cases,
        'correct_cases': correct_cases,
        'failure_analysis': failure_analysis
    }
```

## 可视化分析

### 注意力可视化

```python
def visualize_attention_maps(attention_weights: torch.Tensor,
                           input_sequence_length: int,
                           layer_names: List[str] = None,
                           save_path: str = None):
    """可视化注意力图"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # attention_weights: [num_layers, num_heads, seq_len, seq_len]
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # 为每一层创建子图
    fig, axes = plt.subplots(num_layers, num_heads, 
                           figsize=(num_heads * 3, num_layers * 3))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    if num_heads == 1:
        axes = axes.reshape(-1, 1)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head]
            
            # 绘制注意力热图
            attn_map = attention_weights[layer, head].cpu().numpy()
            
            sns.heatmap(attn_map, ax=ax, cmap='Blues', 
                       xticklabels=False, yticklabels=False,
                       cbar=head == num_heads - 1)
            
            # 设置标题
            layer_name = layer_names[layer] if layer_names else f'Layer {layer}'
            ax.set_title(f'{layer_name}, Head {head}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

### 特征分布可视化

```python
def visualize_feature_distributions(features_dict: Dict[str, torch.Tensor],
                                  labels: torch.Tensor = None,
                                  method: str = 'tsne',
                                  save_path: str = None):
    """可视化特征分布"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # 合并所有模态特征
    all_features = []
    modality_labels = []
    
    for modality, features in features_dict.items():
        # 全局池化
        if features.dim() > 2:
            pooled_features = torch.mean(features, dim=1)
        else:
            pooled_features = features
        
        all_features.append(pooled_features.cpu().numpy())
        modality_labels.extend([modality] * len(pooled_features))
    
    features_array = np.vstack(all_features)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    features_2d = reducer.fit_transform(features_array)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 按模态着色
    unique_modalities = list(features_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_modalities)))
    
    for i, modality in enumerate(unique_modalities):
        mask = np.array(modality_labels) == modality
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=modality, alpha=0.6)
    
    plt.legend()
    plt.title(f'特征分布可视化 ({method.upper()})')
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

## 性能分析

### 计算效率分析

```python
def analyze_computational_efficiency(model, input_shape_dict: Dict[str, tuple]) -> Dict[str, Any]:
    """分析计算效率"""
    import torch.profiler
    
    # 创建示例输入
    inputs = {}
    for modality, shape in input_shape_dict.items():
        inputs[modality] = torch.randn(shape)
    
    # 使用PyTorch Profiler分析
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            outputs = model(inputs)
    
    # 分析结果
    efficiency_stats = {
        'total_time_ms': 0,
        'gpu_time_ms': 0,
        'cpu_time_ms': 0,
        'memory_usage_mb': 0,
        'flops': 0
    }
    
    # 解析profiler结果
    events = prof.events()
    for event in events:
        if event.name == "model_inference":
            efficiency_stats['total_time_ms'] = event.cuda_time_total / 1000
            efficiency_stats['gpu_time_ms'] = event.cuda_time / 1000
            efficiency_stats['cpu_time_ms'] = event.cpu_time / 1000
    
    # 计算FLOPs (需要额外工具)
    try:
        from fvcore.nn import FlopCountMode, flop_count
        flops_dict, _ = flop_count(model, inputs)
        efficiency_stats['flops'] = sum(flops_dict.values())
    except ImportError:
        efficiency_stats['flops'] = 'N/A (需要安装fvcore)'
    
    return efficiency_stats
```

## 评估报告

### 自动报告生成

```python
class EvaluationReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any]):
        self.results = results
        self.config = config
    
    def generate_html_report(self, save_path: str = 'evaluation_report.html'):
        """生成HTML格式的评估报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TriModalFusion 评估报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric-table { border-collapse: collapse; width: 100%; }
                .metric-table th, .metric-table td { 
                    border: 1px solid #ddd; padding: 8px; text-align: left; 
                }
                .metric-table th { background-color: #f2f2f2; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>TriModalFusion 评估报告</h1>
                <p>生成时间: {timestamp}</p>
                <p>模型配置: {model_config}</p>
            </div>
            
            <div class="section">
                <h2>总体性能</h2>
                <table class="metric-table">
                    <tr><th>指标</th><th>值</th></tr>
                    {overall_metrics}
                </table>
            </div>
            
            <div class="section">
                <h2>各模态性能</h2>
                {modality_performance}
            </div>
            
            <div class="section">
                <h2>融合效果分析</h2>
                {fusion_analysis}
            </div>
            
            <div class="section">
                <h2>错误分析</h2>
                {error_analysis}
            </div>
        </body>
        </html>
        """
        
        # 填充模板
        import datetime
        
        content = html_template.format(
            timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            model_config=str(self.config.get('model', {})),
            overall_metrics=self._format_overall_metrics(),
            modality_performance=self._format_modality_performance(),
            fusion_analysis=self._format_fusion_analysis(),
            error_analysis=self._format_error_analysis()
        )
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"HTML报告已保存至: {save_path}")
    
    def _format_overall_metrics(self) -> str:
        """格式化总体指标"""
        key_metrics = [
            'classification_accuracy',
            'classification_f1_macro',
            'detection_mAP@0.5',
            'speech_wer',
            'gesture_accuracy',
            'image_top1_accuracy'
        ]
        
        rows = []
        for metric in key_metrics:
            if metric in self.results:
                value = self.results[metric]
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                rows.append(f"<tr><td>{metric}</td><td>{value_str}</td></tr>")
        
        return "\n".join(rows)
    
    def _format_modality_performance(self) -> str:
        """格式化各模态性能"""
        modalities = ['speech', 'gesture', 'image']
        content = []
        
        for modality in modalities:
            modality_metrics = {k: v for k, v in self.results.items() 
                              if k.startswith(modality)}
            
            if modality_metrics:
                content.append(f"<h3>{modality.capitalize()}</h3>")
                content.append("<table class='metric-table'>")
                content.append("<tr><th>指标</th><th>值</th></tr>")
                
                for metric, value in modality_metrics.items():
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    content.append(f"<tr><td>{metric}</td><td>{value_str}</td></tr>")
                
                content.append("</table>")
        
        return "\n".join(content)
    
    def generate_pdf_report(self, save_path: str = 'evaluation_report.pdf'):
        """生成PDF格式的评估报告"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            
            doc = SimpleDocTemplate(save_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 标题
            title = Paragraph("TriModalFusion 评估报告", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # 总体性能
            story.append(Paragraph("总体性能", styles['Heading2']))
            
            # 创建性能表格
            performance_data = [['指标', '值']]
            key_metrics = ['classification_accuracy', 'detection_mAP@0.5', 'fusion_effectiveness']
            
            for metric in key_metrics:
                if metric in self.results:
                    value = f"{self.results[metric]:.4f}" if isinstance(self.results[metric], float) else str(self.results[metric])
                    performance_data.append([metric, value])
            
            performance_table = Table(performance_data)
            story.append(performance_table)
            story.append(Spacer(1, 12))
            
            # 生成PDF
            doc.build(story)
            print(f"PDF报告已保存至: {save_path}")
            
        except ImportError:
            print("生成PDF报告需要安装reportlab库: pip install reportlab")
```

这个评估指南提供了全面的多模态模型评估框架，涵盖了从基础指标计算到高级分析可视化的所有方面，确保用户能够全面了解模型的性能表现。