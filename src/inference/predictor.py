import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
import time
import json

from ..models.trimodal_fusion import TriModalFusionModel
from ..utils.config import load_config, Config
from ..data.preprocessor import AudioPreprocessor, ImagePreprocessor, GesturePreprocessor

logger = logging.getLogger(__name__)


class TriModalPredictor:
    """
    TriModalFusion模型预测器
    
    提供单样本和批量预测功能，支持不同的推理模式和输出格式
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        half_precision: bool = False
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径，如果None则从checkpoint加载
            device: 推理设备，None为自动选择
            batch_size: 批处理大小
            half_precision: 是否使用半精度推理
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.batch_size = batch_size
        self.half_precision = half_precision
        
        # 设置设备
        self.device = self._setup_device(device)
        
        # 加载配置和模型
        self.config = self._load_config()
        self.model = self._load_model()
        
        # 设置预处理器
        self.preprocessors = self._setup_preprocessors()
        
        # 性能统计
        self.inference_stats = {
            'total_predictions': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """设置推理设备"""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"使用GPU推理: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("使用CPU推理")
        
        return torch.device(device)
    
    def _load_config(self) -> Config:
        """加载配置"""
        if self.config_path and self.config_path.exists():
            return load_config(str(self.config_path))
        else:
            # 尝试从checkpoint加载配置
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'config' in checkpoint:
                return Config(checkpoint['config'])
            else:
                raise ValueError("无法找到模型配置，请提供config_path")
    
    def _load_model(self) -> TriModalFusionModel:
        """加载模型"""
        logger.info(f"加载模型从: {self.model_path}")
        
        # 创建模型
        model = TriModalFusionModel(self.config)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DataParallel包装的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # 半精度推理
        if self.half_precision and self.device.type == 'cuda':
            model = model.half()
            logger.info("启用半精度推理")
        
        logger.info(f"模型加载完成，参数数量: {model.get_num_parameters():,}")
        
        return model
    
    def _setup_preprocessors(self) -> Dict[str, Any]:
        """设置预处理器"""
        preprocessors = {}
        
        # 音频预处理器
        preprocessors['speech'] = AudioPreprocessor(self.config.speech_config)
        
        # 图像预处理器
        preprocessors['image'] = ImagePreprocessor(self.config.image_config)
        
        # 手势预处理器
        preprocessors['gesture'] = GesturePreprocessor(self.config.gesture_config)
        
        return preprocessors
    
    def predict(
        self,
        inputs: Dict[str, Any],
        return_features: bool = False,
        return_attention: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        单样本预测
        
        Args:
            inputs: 输入数据字典
            return_features: 是否返回特征表示
            return_attention: 是否返回注意力权重
            temperature: softmax温度参数
        
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        # 预处理输入
        processed_inputs = self._preprocess_inputs(inputs)
        
        # 推理
        with torch.no_grad():
            if self.half_precision and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(processed_inputs)
            else:
                outputs = self.model(processed_inputs)
        
        # 后处理输出
        predictions = self._postprocess_outputs(
            outputs, temperature, return_features, return_attention
        )
        
        # 更新统计
        inference_time = time.time() - start_time
        self._update_stats(inference_time)
        
        predictions['inference_time'] = inference_time
        
        return predictions
    
    def predict_batch(
        self,
        batch_inputs: List[Dict[str, Any]],
        return_features: bool = False,
        return_attention: bool = False,
        temperature: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            batch_inputs: 输入数据列表
            return_features: 是否返回特征表示
            return_attention: 是否返回注意力权重
            temperature: softmax温度参数
        
        Returns:
            预测结果列表
        """
        predictions = []
        
        # 分批处理
        for i in range(0, len(batch_inputs), self.batch_size):
            batch = batch_inputs[i:i + self.batch_size]
            
            # 合并批次数据
            batched_inputs = self._batch_inputs(batch)
            
            # 批量推理
            batch_predictions = self.predict(
                batched_inputs, return_features, return_attention, temperature
            )
            
            # 拆分批次结果
            split_predictions = self._split_batch_outputs(batch_predictions, len(batch))
            predictions.extend(split_predictions)
        
        return predictions
    
    def predict_from_files(
        self,
        file_paths: Dict[str, str],
        return_features: bool = False,
        return_attention: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        从文件路径预测
        
        Args:
            file_paths: 文件路径字典 {'speech': '...', 'image': '...', 'gesture': '...'}
            return_features: 是否返回特征表示
            return_attention: 是否返回注意力权重
            temperature: softmax温度参数
        
        Returns:
            预测结果字典
        """
        # 从文件加载数据
        inputs = {}
        
        if 'speech' in file_paths:
            inputs['speech'] = self.preprocessors['speech'].load_from_file(
                file_paths['speech']
            )
        
        if 'image' in file_paths:
            inputs['image'] = self.preprocessors['image'].load_from_file(
                file_paths['image']
            )
        
        if 'gesture' in file_paths:
            inputs['gesture'] = self.preprocessors['gesture'].load_from_file(
                file_paths['gesture']
            )
        
        return self.predict(inputs, return_features, return_attention, temperature)
    
    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """预处理输入数据"""
        processed = {}
        
        for modality, data in inputs.items():
            if modality in self.preprocessors:
                # 如果数据已经是tensor，直接使用
                if isinstance(data, torch.Tensor):
                    processed_data = data
                else:
                    # 否则使用预处理器处理
                    processed_data = self.preprocessors[modality].preprocess(data)
                
                # 确保有batch维度
                if processed_data.dim() == len(processed_data.shape) - 1:
                    processed_data = processed_data.unsqueeze(0)
                
                # 移动到设备
                processed_data = processed_data.to(self.device)
                
                # 半精度转换
                if self.half_precision and self.device.type == 'cuda':
                    processed_data = processed_data.half()
                
                processed[modality] = processed_data
        
        return processed
    
    def _postprocess_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """后处理模型输出"""
        predictions = {}
        task_outputs = outputs.get('task_outputs', {})
        
        # 分类任务
        if 'classification' in task_outputs:
            logits = task_outputs['classification']
            
            # 应用温度缩放
            if temperature != 1.0:
                logits = logits / temperature
            
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(logits, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
            
            predictions['classification'] = {
                'logits': logits.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'predicted_classes': predicted_classes.cpu().numpy(),
                'confidence_scores': confidence_scores.cpu().numpy()
            }
            
            # Top-K预测
            top_k = min(5, probabilities.size(-1))
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
            predictions['classification']['top_k'] = {
                'indices': top_k_indices.cpu().numpy(),
                'probabilities': top_k_probs.cpu().numpy()
            }
        
        # 检测任务
        if 'detection' in task_outputs:
            detection_outputs = task_outputs['detection']
            predictions['detection'] = self._process_detection_outputs(detection_outputs)
        
        # 回归任务
        if 'regression' in task_outputs:
            regression_outputs = task_outputs['regression']
            predictions['regression'] = {
                'predictions': regression_outputs.cpu().numpy()
            }
        
        # 特征表示
        if return_features:
            features = {}
            if 'fused_features' in outputs:
                features['fused'] = outputs['fused_features'].cpu().numpy()
            
            if 'encoded_features' in outputs:
                encoded_features = {}
                for modality, feat in outputs['encoded_features'].items():
                    encoded_features[modality] = feat.cpu().numpy()
                features['encoded'] = encoded_features
            
            predictions['features'] = features
        
        # 注意力权重
        if return_attention and 'attention_weights' in outputs:
            attention_weights = {}
            for key, weights in outputs['attention_weights'].items():
                if torch.is_tensor(weights):
                    attention_weights[key] = weights.cpu().numpy()
            predictions['attention_weights'] = attention_weights
        
        return predictions
    
    def _process_detection_outputs(self, detection_outputs: Dict) -> Dict:
        """处理检测任务输出"""
        processed = {}
        
        # 这里需要根据具体的检测架构实现
        # 简化示例：假设输出包含boxes, scores, classes
        
        if 'boxes' in detection_outputs:
            processed['boxes'] = detection_outputs['boxes'].cpu().numpy()
        
        if 'scores' in detection_outputs:
            scores = detection_outputs['scores']
            processed['scores'] = scores.cpu().numpy()
            
            # 置信度阈值过滤
            confidence_threshold = 0.5
            valid_detections = scores > confidence_threshold
            processed['valid_detections'] = valid_detections.cpu().numpy()
        
        if 'classes' in detection_outputs:
            processed['classes'] = detection_outputs['classes'].cpu().numpy()
        
        return processed
    
    def _batch_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将输入列表合并为批次"""
        batched = {}
        
        # 获取所有模态
        all_modalities = set()
        for inputs in batch:
            all_modalities.update(inputs.keys())
        
        for modality in all_modalities:
            modality_data = []
            for inputs in batch:
                if modality in inputs:
                    data = inputs[modality]
                    if not isinstance(data, torch.Tensor):
                        data = self.preprocessors[modality].preprocess(data)
                    modality_data.append(data)
                else:
                    # 如果某个样本缺少某个模态，创建零填充
                    # 这里需要根据具体模态确定零填充的形状
                    zero_shape = self._get_zero_shape(modality)
                    modality_data.append(torch.zeros(zero_shape))
            
            # 堆叠为批次
            batched[modality] = torch.stack(modality_data)
        
        return batched
    
    def _get_zero_shape(self, modality: str) -> Tuple[int, ...]:
        """获取指定模态的零填充形状"""
        if modality == 'speech':
            return (self.config.speech_config.get('max_audio_length', 16000),)
        elif modality == 'image':
            size = self.config.image_config.get('img_size', 224)
            return (3, size, size)
        elif modality == 'gesture':
            max_len = self.config.gesture_config.get('max_sequence_length', 30)
            num_hands = self.config.gesture_config.get('num_hands', 2)
            return (max_len, num_hands, 21, 3)
        else:
            raise ValueError(f"未知模态: {modality}")
    
    def _split_batch_outputs(self, batch_predictions: Dict, batch_size: int) -> List[Dict]:
        """将批次输出拆分为单独的预测"""
        predictions = []
        
        for i in range(batch_size):
            prediction = {}
            
            for task, task_outputs in batch_predictions.items():
                if task == 'inference_time':
                    # 推理时间平分
                    prediction[task] = task_outputs / batch_size
                    continue
                
                if isinstance(task_outputs, dict):
                    prediction[task] = {}
                    for key, value in task_outputs.items():
                        if isinstance(value, np.ndarray) and value.ndim > 0:
                            prediction[task][key] = value[i]
                        else:
                            prediction[task][key] = value
                elif isinstance(task_outputs, np.ndarray) and task_outputs.ndim > 0:
                    prediction[task] = task_outputs[i]
                else:
                    prediction[task] = task_outputs
            
            predictions.append(prediction)
        
        return predictions
    
    def _update_stats(self, inference_time: float):
        """更新性能统计"""
        self.inference_stats['total_predictions'] += 1
        self.inference_stats['total_time'] += inference_time
        self.inference_stats['average_time'] = (
            self.inference_stats['total_time'] / self.inference_stats['total_predictions']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理性能统计"""
        return self.inference_stats.copy()
    
    def reset_stats(self):
        """重置性能统计"""
        self.inference_stats = {
            'total_predictions': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
    
    def benchmark(self, num_runs: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            num_runs: 运行次数
            batch_size: 批次大小
        
        Returns:
            基准测试结果
        """
        logger.info(f"开始性能基准测试：{num_runs} 次运行，批次大小 {batch_size}")
        
        # 创建随机测试数据
        test_inputs = self._create_test_inputs(batch_size)
        
        # 热身运行
        for _ in range(5):
            _ = self.predict(test_inputs)
        
        # 基准测试
        times = []
        for i in range(num_runs):
            start_time = time.time()
            _ = self.predict(test_inputs)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"完成 {i + 1}/{num_runs} 次运行")
        
        # 计算统计信息
        times = np.array(times)
        results = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'throughput': batch_size / float(np.mean(times)),  # samples per second
            'total_runs': num_runs,
            'batch_size': batch_size
        }
        
        logger.info(f"基准测试完成:")
        logger.info(f"  平均推理时间: {results['mean_time']:.4f}s")
        logger.info(f"  吞吐量: {results['throughput']:.2f} samples/s")
        
        return results
    
    def _create_test_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """创建测试输入数据"""
        inputs = {}
        
        # 语音数据
        audio_length = self.config.speech_config.get('max_audio_length', 16000)
        inputs['speech'] = torch.randn(batch_size, audio_length).to(self.device)
        
        # 图像数据
        img_size = self.config.image_config.get('img_size', 224)
        inputs['image'] = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # 手势数据
        max_len = self.config.gesture_config.get('max_sequence_length', 30)
        num_hands = self.config.gesture_config.get('num_hands', 2)
        inputs['gesture'] = torch.randn(batch_size, max_len, num_hands, 21, 3).to(self.device)
        
        # 半精度转换
        if self.half_precision and self.device.type == 'cuda':
            for key in inputs:
                inputs[key] = inputs[key].half()
        
        return inputs