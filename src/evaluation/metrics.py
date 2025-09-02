"""
Evaluation metrics for multimodal systems.

This module provides comprehensive metrics for evaluating speech recognition,
gesture recognition, image recognition, and multimodal fusion performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import editdistance
import logging

logger = logging.getLogger(__name__)


class SpeechRecognitionMetrics:
    """Metrics for speech recognition evaluation."""
    
    @staticmethod
    def word_error_rate(predictions: List[str], references: List[str]) -> float:
        """
        Calculate Word Error Rate (WER).
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            WER score (0.0 = perfect, higher = worse)
        """
        total_words = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Calculate edit distance
            errors = editdistance.eval(pred_words, ref_words)
            total_errors += errors
            total_words += len(ref_words)
        
        if total_words == 0:
            return 0.0
            
        return total_errors / total_words
    
    @staticmethod
    def character_error_rate(predictions: List[str], references: List[str]) -> float:
        """
        Calculate Character Error Rate (CER).
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            CER score (0.0 = perfect, higher = worse)
        """
        total_chars = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred.lower())
            ref_chars = list(ref.lower())
            
            # Calculate edit distance
            errors = editdistance.eval(pred_chars, ref_chars)
            total_errors += errors
            total_chars += len(ref_chars)
        
        if total_chars == 0:
            return 0.0
            
        return total_errors / total_chars
    
    @staticmethod
    def bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
        """
        Calculate BLEU score for speech recognition.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            n: Maximum n-gram order
            
        Returns:
            BLEU score (1.0 = perfect, 0.0 = worst)
        """
        from collections import Counter
        import math
        
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Get n-grams from tokens."""
            if len(tokens) < n:
                return Counter()
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        
        def calculate_bleu_for_pair(pred: str, ref: str) -> float:
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            # Calculate precision for each n-gram order
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
            
            # Calculate brevity penalty
            pred_len = len(pred_tokens)
            ref_len = len(ref_tokens)
            
            if pred_len >= ref_len:
                bp = 1.0
            else:
                bp = math.exp(1 - ref_len / pred_len)
            
            # Calculate geometric mean of precisions
            if all(p > 0 for p in precisions):
                geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
            else:
                geo_mean = 0.0
            
            return bp * geo_mean
        
        scores = [calculate_bleu_for_pair(pred, ref) 
                 for pred, ref in zip(predictions, references)]
        
        return sum(scores) / len(scores) if scores else 0.0


class GestureRecognitionMetrics:
    """Metrics for gesture recognition evaluation."""
    
    @staticmethod
    def gesture_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate gesture classification accuracy.
        
        Args:
            predictions: Predicted gesture classes [B] or [B, C]
            targets: Target gesture classes [B]
            
        Returns:
            Accuracy score
        """
        if predictions.dim() == 2:
            predictions = torch.argmax(predictions, dim=1)
        
        correct = (predictions == targets).float().sum()
        total = targets.size(0)
        
        return (correct / total).item()
    
    @staticmethod
    def gesture_f1_score(predictions: torch.Tensor, targets: torch.Tensor, 
                        num_classes: int, average: str = 'weighted') -> float:
        """
        Calculate F1 score for gesture recognition.
        
        Args:
            predictions: Predicted gesture classes
            targets: Target gesture classes
            num_classes: Number of gesture classes
            average: Averaging method ('micro', 'macro', 'weighted')
            
        Returns:
            F1 score
        """
        if predictions.dim() == 2:
            predictions = torch.argmax(predictions, dim=1)
        
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        return f1_score(targets_np, predictions_np, average=average, 
                       labels=list(range(num_classes)), zero_division=0)
    
    @staticmethod
    def keypoint_error(predicted_keypoints: torch.Tensor, 
                      target_keypoints: torch.Tensor,
                      valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate keypoint localization errors.
        
        Args:
            predicted_keypoints: Predicted keypoints [B, T, N, 2/3]
            target_keypoints: Target keypoints [B, T, N, 2/3]
            valid_mask: Valid keypoints mask [B, T, N]
            
        Returns:
            Dictionary of error metrics
        """
        # Calculate Euclidean distance
        diff = predicted_keypoints - target_keypoints
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [B, T, N]
        
        if valid_mask is not None:
            distances = distances * valid_mask
            valid_points = valid_mask.sum()
        else:
            valid_points = torch.numel(distances)
        
        if valid_points == 0:
            return {'mse': 0.0, 'mae': 0.0, 'pck': 0.0}
        
        # Mean Square Error
        mse = torch.mean(distances ** 2).item()
        
        # Mean Absolute Error
        mae = torch.mean(distances).item()
        
        # Percentage of Correct Keypoints (PCK) with threshold 0.1
        pck_threshold = 0.1
        correct_keypoints = (distances < pck_threshold).float()
        pck = torch.mean(correct_keypoints).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'pck': pck
        }
    
    @staticmethod
    def temporal_consistency(predictions: torch.Tensor, 
                           window_size: int = 5) -> float:
        """
        Calculate temporal consistency of gesture predictions.
        
        Args:
            predictions: Temporal predictions [B, T, ...]
            window_size: Size of temporal window
            
        Returns:
            Temporal consistency score
        """
        if predictions.dim() == 3:  # Classification logits
            predictions = torch.argmax(predictions, dim=-1)  # [B, T]
        
        B, T = predictions.shape
        consistency_scores = []
        
        for b in range(B):
            for t in range(T - window_size + 1):
                window = predictions[b, t:t + window_size]
                # Calculate consistency as ratio of most frequent class
                unique_values, counts = torch.unique(window, return_counts=True)
                max_count = torch.max(counts).item()
                consistency = max_count / window_size
                consistency_scores.append(consistency)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0


class ImageRecognitionMetrics:
    """Metrics for image recognition and object detection."""
    
    @staticmethod
    def classification_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate image classification accuracy.
        
        Args:
            predictions: Predicted class logits [B, C] or indices [B]
            targets: Target class indices [B]
            
        Returns:
            Accuracy score
        """
        if predictions.dim() == 2:
            predictions = torch.argmax(predictions, dim=1)
        
        correct = (predictions == targets).float().sum()
        total = targets.size(0)
        
        return (correct / total).item()
    
    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            predictions: Predicted class logits [B, C]
            targets: Target class indices [B]
            k: Top-k parameter
            
        Returns:
            Top-k accuracy score
        """
        _, top_k_predictions = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_predictions)
        
        correct = torch.any(top_k_predictions == targets_expanded, dim=1).float()
        
        return torch.mean(correct).item()
    
    @staticmethod
    def mean_average_precision(predictions: List[Dict], targets: List[Dict], 
                              iou_threshold: float = 0.5) -> float:
        """
        Calculate mean Average Precision (mAP) for object detection.
        
        Args:
            predictions: List of prediction dictionaries with 'boxes', 'scores', 'labels'
            targets: List of target dictionaries with 'boxes', 'labels'
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            mAP score
        """
        def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
            """Calculate IoU between two boxes."""
            # box format: [x1, y1, x2, y2]
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
        
        def calculate_ap_for_class(class_predictions: List, class_targets: List) -> float:
            """Calculate AP for a specific class."""
            if not class_predictions:
                return 0.0
            
            # Sort predictions by confidence score
            class_predictions.sort(key=lambda x: x['score'], reverse=True)
            
            tp = []
            fp = []
            
            for pred in class_predictions:
                best_iou = 0.0
                best_target_idx = -1
                
                for i, target in enumerate(class_targets):
                    if target.get('matched', False):
                        continue
                    
                    iou = calculate_iou(pred['box'], target['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = i
                
                if best_iou >= iou_threshold:
                    tp.append(1)
                    fp.append(0)
                    class_targets[best_target_idx]['matched'] = True
                else:
                    tp.append(0)
                    fp.append(1)
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(class_targets) if len(class_targets) > 0 else np.zeros_like(tp_cumsum)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
            # Calculate AP using 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                precision_at_t = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
                ap += precision_at_t
            
            return ap / 11.0
        
        # Collect predictions and targets by class
        all_classes = set()
        for pred_dict in predictions:
            if 'labels' in pred_dict:
                all_classes.update(pred_dict['labels'].tolist())
        for target_dict in targets:
            if 'labels' in target_dict:
                all_classes.update(target_dict['labels'].tolist())
        
        if not all_classes:
            return 0.0
        
        class_aps = []
        
        for class_id in all_classes:
            class_predictions = []
            class_targets = []
            
            # Collect predictions for this class
            for pred_dict in predictions:
                if 'labels' not in pred_dict:
                    continue
                class_mask = pred_dict['labels'] == class_id
                if torch.any(class_mask):
                    boxes = pred_dict['boxes'][class_mask]
                    scores = pred_dict['scores'][class_mask]
                    for box, score in zip(boxes, scores):
                        class_predictions.append({'box': box, 'score': score.item()})
            
            # Collect targets for this class
            for target_dict in targets:
                if 'labels' not in target_dict:
                    continue
                class_mask = target_dict['labels'] == class_id
                if torch.any(class_mask):
                    boxes = target_dict['boxes'][class_mask]
                    for box in boxes:
                        class_targets.append({'box': box, 'matched': False})
            
            # Calculate AP for this class
            ap = calculate_ap_for_class(class_predictions, class_targets)
            class_aps.append(ap)
        
        return sum(class_aps) / len(class_aps) if class_aps else 0.0


class MultiModalFusionMetrics:
    """Metrics for evaluating multimodal fusion performance."""
    
    @staticmethod
    def modality_contribution(features_dict: Dict[str, torch.Tensor], 
                            fused_features: torch.Tensor) -> Dict[str, float]:
        """
        Calculate the contribution of each modality to the fused representation.
        
        Args:
            features_dict: Dictionary of individual modality features
            fused_features: Fused multimodal features
            
        Returns:
            Dictionary of contribution scores for each modality
        """
        contributions = {}
        
        for modality, features in features_dict.items():
            # Calculate cosine similarity between individual and fused features
            if features.dim() > 2:
                features = features.mean(dim=1)  # Global pooling
            if fused_features.dim() > 2:
                fused_pooled = fused_features.mean(dim=1)
            else:
                fused_pooled = fused_features
            
            # Normalize features
            features_norm = F.normalize(features, dim=-1)
            fused_norm = F.normalize(fused_pooled, dim=-1)
            
            # Calculate similarity
            similarity = torch.mean(torch.sum(features_norm * fused_norm, dim=-1))
            contributions[modality] = similarity.item()
        
        return contributions
    
    @staticmethod
    def fusion_effectiveness(unimodal_performances: Dict[str, float],
                           multimodal_performance: float) -> Dict[str, float]:
        """
        Calculate fusion effectiveness metrics.
        
        Args:
            unimodal_performances: Performance scores for each individual modality
            multimodal_performance: Performance score for multimodal fusion
            
        Returns:
            Dictionary of fusion effectiveness metrics
        """
        if not unimodal_performances:
            return {'improvement': 0.0, 'relative_improvement': 0.0}
        
        best_unimodal = max(unimodal_performances.values())
        mean_unimodal = sum(unimodal_performances.values()) / len(unimodal_performances)
        
        improvement = multimodal_performance - best_unimodal
        relative_improvement = improvement / best_unimodal if best_unimodal > 0 else 0.0
        
        return {
            'improvement': improvement,
            'relative_improvement': relative_improvement,
            'best_unimodal': best_unimodal,
            'mean_unimodal': mean_unimodal,
            'multimodal': multimodal_performance
        }
    
    @staticmethod
    def cross_modal_similarity(features1: torch.Tensor, features2: torch.Tensor) -> float:
        """
        Calculate cross-modal similarity between two modalities.
        
        Args:
            features1: Features from first modality
            features2: Features from second modality
            
        Returns:
            Cross-modal similarity score
        """
        # Global pooling if needed
        if features1.dim() > 2:
            features1 = features1.mean(dim=1)
        if features2.dim() > 2:
            features2 = features2.mean(dim=1)
        
        # Normalize features
        features1_norm = F.normalize(features1, dim=-1)
        features2_norm = F.normalize(features2, dim=-1)
        
        # Calculate cosine similarity
        similarity = torch.mean(torch.sum(features1_norm * features2_norm, dim=-1))
        
        return similarity.item()
    
    @staticmethod
    def information_gain(joint_entropy: float, individual_entropies: List[float]) -> float:
        """
        Calculate information gain from multimodal fusion.
        
        Args:
            joint_entropy: Entropy of joint multimodal distribution
            individual_entropies: List of individual modality entropies
            
        Returns:
            Information gain score
        """
        sum_individual = sum(individual_entropies)
        information_gain = sum_individual - joint_entropy
        
        return information_gain


class ClassificationMetrics:
    """General classification metrics."""
    
    @staticmethod
    def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                          num_classes: Optional[int] = None) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            predictions: Predicted class logits [B, C] or indices [B]
            targets: Target class indices [B]
            num_classes: Number of classes
            
        Returns:
            Dictionary of all classification metrics
        """
        if predictions.dim() == 2:
            pred_probs = F.softmax(predictions, dim=1)
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
            pred_probs = None
        
        predictions_np = pred_classes.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets_np, predictions_np)
        metrics['precision_macro'] = precision_score(targets_np, predictions_np, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(targets_np, predictions_np, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(targets_np, predictions_np, average='macro', zero_division=0)
        
        metrics['precision_micro'] = precision_score(targets_np, predictions_np, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(targets_np, predictions_np, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(targets_np, predictions_np, average='micro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(targets_np, predictions_np, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(targets_np, predictions_np, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets_np, predictions_np, average='weighted', zero_division=0)
        
        # AUC (if probabilities available)
        if pred_probs is not None and num_classes is not None:
            try:
                if num_classes == 2:
                    metrics['auc'] = roc_auc_score(targets_np, pred_probs[:, 1].cpu().numpy())
                else:
                    metrics['auc'] = roc_auc_score(targets_np, pred_probs.cpu().numpy(), 
                                                 multi_class='ovr', average='macro')
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics


class DetectionMetrics:
    """Object detection metrics."""
    
    @staticmethod
    def compute_detection_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """
        Compute comprehensive detection metrics.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary of detection metrics
        """
        metrics = {}
        
        # mAP at different IoU thresholds
        iou_thresholds = [0.5, 0.75]
        for iou_thresh in iou_thresholds:
            map_score = ImageRecognitionMetrics.mean_average_precision(
                predictions, targets, iou_threshold=iou_thresh
            )
            metrics[f'mAP@{iou_thresh}'] = map_score
        
        # Average mAP across IoU thresholds
        iou_range = np.arange(0.5, 1.0, 0.05)
        map_scores = []
        for iou_thresh in iou_range:
            map_score = ImageRecognitionMetrics.mean_average_precision(
                predictions, targets, iou_threshold=iou_thresh
            )
            map_scores.append(map_score)
        
        metrics['mAP@0.5:0.95'] = sum(map_scores) / len(map_scores)
        
        return metrics


class RegressionMetrics:
    """Regression metrics."""
    
    @staticmethod
    def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            predictions: Predicted values
            targets: Target values
            
        Returns:
            Dictionary of regression metrics
        """
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        metrics = {}
        
        metrics['mse'] = mean_squared_error(targets_np, predictions_np)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets_np, predictions_np)
        metrics['r2'] = r2_score(targets_np, predictions_np)
        
        # Additional metrics
        residuals = targets_np - predictions_np
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        return metrics