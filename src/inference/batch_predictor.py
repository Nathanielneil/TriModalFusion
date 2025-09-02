import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Iterator
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .predictor import TriModalPredictor

logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    大批量数据预测器
    
    优化大规模数据集的预测性能，支持多线程数据加载、
    内存管理和进度跟踪
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_memory_usage: float = 0.8,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        half_precision: bool = False
    ):
        """
        初始化批量预测器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径
            device: 推理设备
            batch_size: 批次大小
            max_memory_usage: 最大内存使用率（0-1）
            num_workers: 数据加载线程数
            prefetch_factor: 预取因子
            half_precision: 是否使用半精度
        """
        self.predictor = TriModalPredictor(
            model_path=model_path,
            config_path=config_path,
            device=device,
            batch_size=1,  # 单个预测器使用batch_size=1
            half_precision=half_precision
        )
        
        self.batch_size = batch_size
        self.max_memory_usage = max_memory_usage
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # 内存监控
        self.memory_monitor = MemoryMonitor(max_memory_usage)
        
        # 性能统计
        self.batch_stats = {
            'total_batches': 0,
            'total_samples': 0,
            'total_time': 0.0,
            'average_batch_time': 0.0,
            'throughput': 0.0
        }
    
    def predict_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        save_features: bool = False,
        save_attention: bool = False,
        checkpoint_every: int = 100
    ) -> List[Dict[str, Any]]:
        """
        预测整个数据集
        
        Args:
            dataset_path: 数据集路径
            output_path: 输出文件路径
            save_features: 是否保存特征
            save_attention: 是否保存注意力权重
            checkpoint_every: 每多少批次保存一次检查点
        
        Returns:
            预测结果列表
        """
        logger.info(f"开始预测数据集: {dataset_path}")
        
        # 加载数据集信息
        dataset_info = self._load_dataset_info(dataset_path)
        total_samples = dataset_info['total_samples']
        
        logger.info(f"数据集包含 {total_samples} 个样本")
        logger.info(f"批次大小: {self.batch_size}, 预计批次数: {(total_samples + self.batch_size - 1) // self.batch_size}")
        
        all_predictions = []
        batch_count = 0
        
        # 创建数据加载器
        data_loader = self._create_data_loader(dataset_path)
        
        # 进度条
        progress_bar = tqdm(
            total=total_samples,
            desc="批量预测",
            unit="samples"
        )
        
        try:
            for batch_inputs in data_loader:
                # 内存检查
                if not self.memory_monitor.check_memory():
                    logger.warning("内存使用率过高，等待垃圾回收")
                    torch.cuda.empty_cache()
                    time.sleep(1)
                
                # 批量预测
                batch_start_time = time.time()
                batch_predictions = self._predict_batch_optimized(
                    batch_inputs, save_features, save_attention
                )
                batch_time = time.time() - batch_start_time
                
                # 更新统计信息
                self._update_batch_stats(len(batch_inputs), batch_time)
                
                # 添加到结果
                all_predictions.extend(batch_predictions)
                
                # 更新进度条
                progress_bar.update(len(batch_inputs))
                progress_bar.set_postfix({
                    'batch_time': f'{batch_time:.2f}s',
                    'throughput': f'{len(batch_inputs)/batch_time:.1f} samples/s',
                    'memory': f'{self.memory_monitor.get_memory_usage():.1%}'
                })
                
                batch_count += 1
                
                # 定期保存检查点
                if output_path and batch_count % checkpoint_every == 0:
                    self._save_checkpoint(all_predictions, output_path, batch_count)
                
        finally:
            progress_bar.close()
        
        # 保存最终结果
        if output_path:
            self._save_predictions(all_predictions, output_path)
        
        logger.info(f"批量预测完成: {len(all_predictions)} 个样本")
        logger.info(f"总时间: {self.batch_stats['total_time']:.2f}s")
        logger.info(f"平均吞吐量: {self.batch_stats['throughput']:.1f} samples/s")
        
        return all_predictions
    
    def predict_file_list(
        self,
        file_list: List[Dict[str, str]],
        output_path: Optional[str] = None,
        save_features: bool = False,
        save_attention: bool = False
    ) -> List[Dict[str, Any]]:
        """
        预测文件列表
        
        Args:
            file_list: 文件路径列表，每个元素是包含模态文件路径的字典
            output_path: 输出文件路径
            save_features: 是否保存特征
            save_attention: 是否保存注意力权重
        
        Returns:
            预测结果列表
        """
        logger.info(f"开始预测文件列表: {len(file_list)} 个文件")
        
        all_predictions = []
        
        # 分批处理
        batches = [
            file_list[i:i + self.batch_size] 
            for i in range(0, len(file_list), self.batch_size)
        ]
        
        progress_bar = tqdm(batches, desc="文件批量预测", unit="batches")
        
        for batch_files in progress_bar:
            # 加载批次数据
            batch_inputs = self._load_batch_from_files(batch_files)
            
            # 预测
            batch_predictions = self._predict_batch_optimized(
                batch_inputs, save_features, save_attention
            )
            
            all_predictions.extend(batch_predictions)
            
            # 更新进度条
            progress_bar.set_postfix({
                'completed': len(all_predictions),
                'memory': f'{self.memory_monitor.get_memory_usage():.1%}'
            })
        
        # 保存结果
        if output_path:
            self._save_predictions(all_predictions, output_path)
        
        return all_predictions
    
    def _predict_batch_optimized(
        self,
        batch_inputs: List[Dict[str, Any]],
        save_features: bool = False,
        save_attention: bool = False
    ) -> List[Dict[str, Any]]:
        """优化的批量预测"""
        # 预处理所有输入
        processed_inputs = []
        for inputs in batch_inputs:
            processed = self.predictor._preprocess_inputs(inputs)
            processed_inputs.append(processed)
        
        # 合并为单个批次张量
        batched_tensors = self._merge_batch_tensors(processed_inputs)
        
        # 单次前向传播
        with torch.no_grad():
            if self.predictor.half_precision and self.predictor.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.predictor.model(batched_tensors)
            else:
                outputs = self.predictor.model(batched_tensors)
        
        # 拆分输出
        batch_predictions = []
        for i in range(len(batch_inputs)):
            # 提取第i个样本的输出
            sample_outputs = self._extract_sample_outputs(outputs, i)
            
            # 后处理
            prediction = self.predictor._postprocess_outputs(
                sample_outputs, 1.0, save_features, save_attention
            )
            
            batch_predictions.append(prediction)
        
        return batch_predictions
    
    def _merge_batch_tensors(self, processed_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """合并批次张量"""
        if not processed_inputs:
            return {}
        
        batched = {}
        modalities = processed_inputs[0].keys()
        
        for modality in modalities:
            # 收集所有样本的该模态数据
            modality_tensors = []
            for inputs in processed_inputs:
                if modality in inputs:
                    tensor = inputs[modality]
                    if tensor.dim() > 0:
                        # 移除可能存在的批次维度
                        if tensor.size(0) == 1:
                            tensor = tensor.squeeze(0)
                    modality_tensors.append(tensor)
            
            if modality_tensors:
                # 堆叠为批次
                try:
                    batched[modality] = torch.stack(modality_tensors)
                except RuntimeError as e:
                    logger.warning(f"无法堆叠 {modality} 模态数据: {e}")
                    # 使用零填充使张量形状一致
                    batched[modality] = self._pad_and_stack_tensors(modality_tensors)
        
        return batched
    
    def _pad_and_stack_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """填充并堆叠不同形状的张量"""
        if not tensors:
            return torch.empty(0)
        
        # 找到每个维度的最大尺寸
        max_shape = list(tensors[0].shape)
        for tensor in tensors[1:]:
            for i, size in enumerate(tensor.shape):
                max_shape[i] = max(max_shape[i], size)
        
        # 填充所有张量到相同形状
        padded_tensors = []
        for tensor in tensors:
            # 计算填充量
            padding = []
            for i in reversed(range(len(max_shape))):
                diff = max_shape[i] - tensor.shape[i]
                padding.extend([0, diff])
            
            # 执行填充
            if any(p > 0 for p in padding):
                padded = torch.nn.functional.pad(tensor, padding)
            else:
                padded = tensor
            
            padded_tensors.append(padded)
        
        return torch.stack(padded_tensors)
    
    def _extract_sample_outputs(self, batch_outputs: Dict, sample_idx: int) -> Dict:
        """从批次输出中提取单个样本的输出"""
        sample_outputs = {}
        
        for key, value in batch_outputs.items():
            if isinstance(value, dict):
                sample_outputs[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor) and sub_value.dim() > 0:
                        sample_outputs[key][sub_key] = sub_value[sample_idx:sample_idx+1]
                    else:
                        sample_outputs[key][sub_key] = sub_value
            elif isinstance(value, torch.Tensor) and value.dim() > 0:
                sample_outputs[key] = value[sample_idx:sample_idx+1]
            else:
                sample_outputs[key] = value
        
        return sample_outputs
    
    def _load_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """加载数据集信息"""
        # 这里需要根据实际数据集格式实现
        # 简化实现：假设有一个info.json文件
        info_path = Path(dataset_path) / 'info.json'
        if info_path.exists():
            import json
            with open(info_path, 'r') as f:
                return json.load(f)
        else:
            # 扫描目录获取文件数量
            dataset_dir = Path(dataset_path)
            if dataset_dir.is_dir():
                # 假设每个子目录代表一个样本
                total_samples = len([d for d in dataset_dir.iterdir() if d.is_dir()])
            else:
                total_samples = 1000  # 默认值
            
            return {'total_samples': total_samples}
    
    def _create_data_loader(self, dataset_path: str) -> Iterator[List[Dict[str, Any]]]:
        """创建数据加载器"""
        # 这里需要根据实际数据集格式实现
        # 简化实现：生成模拟数据
        dataset_info = self._load_dataset_info(dataset_path)
        total_samples = dataset_info['total_samples']
        
        for i in range(0, total_samples, self.batch_size):
            batch_size = min(self.batch_size, total_samples - i)
            batch = []
            
            for j in range(batch_size):
                # 创建模拟输入数据
                sample = self._create_sample_data()
                batch.append(sample)
            
            yield batch
    
    def _create_sample_data(self) -> Dict[str, Any]:
        """创建样本数据（模拟）"""
        config = self.predictor.config
        
        sample = {}
        
        # 语音数据
        audio_length = config.speech_config.get('max_audio_length', 16000)
        sample['speech'] = torch.randn(audio_length)
        
        # 图像数据
        img_size = config.image_config.get('img_size', 224)
        sample['image'] = torch.randn(3, img_size, img_size)
        
        # 手势数据
        max_len = config.gesture_config.get('max_sequence_length', 30)
        num_hands = config.gesture_config.get('num_hands', 2)
        sample['gesture'] = torch.randn(max_len, num_hands, 21, 3)
        
        return sample
    
    def _load_batch_from_files(self, batch_files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """从文件批量加载数据"""
        batch_inputs = []
        
        # 使用多线程加载文件
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(self._load_sample_from_files, files): files 
                for files in batch_files
            }
            
            for future in as_completed(future_to_file):
                try:
                    sample_data = future.result()
                    batch_inputs.append(sample_data)
                except Exception as e:
                    logger.error(f"加载文件失败: {e}")
        
        return batch_inputs
    
    def _load_sample_from_files(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """从文件加载单个样本"""
        sample = {}
        
        for modality, file_path in file_paths.items():
            if modality in self.predictor.preprocessors:
                try:
                    data = self.predictor.preprocessors[modality].load_from_file(file_path)
                    sample[modality] = data
                except Exception as e:
                    logger.warning(f"加载 {modality} 文件 {file_path} 失败: {e}")
        
        return sample
    
    def _update_batch_stats(self, batch_size: int, batch_time: float):
        """更新批次统计信息"""
        self.batch_stats['total_batches'] += 1
        self.batch_stats['total_samples'] += batch_size
        self.batch_stats['total_time'] += batch_time
        
        if self.batch_stats['total_batches'] > 0:
            self.batch_stats['average_batch_time'] = (
                self.batch_stats['total_time'] / self.batch_stats['total_batches']
            )
        
        if self.batch_stats['total_time'] > 0:
            self.batch_stats['throughput'] = (
                self.batch_stats['total_samples'] / self.batch_stats['total_time']
            )
    
    def _save_checkpoint(self, predictions: List[Dict], output_path: str, batch_count: int):
        """保存检查点"""
        checkpoint_path = f"{output_path}_checkpoint_{batch_count}.json"
        self._save_predictions(predictions, checkpoint_path)
        logger.info(f"保存检查点到: {checkpoint_path}")
    
    def _save_predictions(self, predictions: List[Dict], output_path: str):
        """保存预测结果"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_predictions = []
        for pred in predictions:
            serializable_pred = self._make_serializable(pred)
            serializable_predictions.append(serializable_pred)
        
        # 保存为JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(serializable_predictions, f, indent=2)
        
        logger.info(f"预测结果已保存到: {output_file}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """获取批量预测统计信息"""
        return self.batch_stats.copy()


class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, max_usage: float = 0.8):
        self.max_usage = max_usage
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用率"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            import psutil
            return psutil.virtual_memory().percent / 100.0
    
    def check_memory(self) -> bool:
        """检查内存使用是否在安全范围内"""
        usage = self.get_memory_usage()
        return usage < self.max_usage
    
    def clear_cache(self):
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()