import torch
import torchaudio
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import mediapipe as mp

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    音频预处理器
    
    负责音频文件的加载、预处理和特征提取
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.max_audio_length = config.get('max_audio_length', 16000)
        self.normalize = config.get('normalize', True)
        self.add_noise = config.get('add_noise', False)
        self.noise_factor = config.get('noise_factor', 0.005)
        
        # 音频增强参数
        self.pitch_shift_range = config.get('pitch_shift_range', 0.0)
        self.time_stretch_range = config.get('time_stretch_range', 0.0)
        self.volume_range = config.get('volume_range', (0.8, 1.2))
        
        # 频谱参数
        self.n_mels = config.get('n_mels', 80)
        self.n_fft = config.get('n_fft', 512)
        self.hop_length = config.get('hop_length', 160)
        self.win_length = config.get('win_length', 400)
        
        # 创建mel-spectrogram变换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
    
    def load_from_file(self, file_path: str) -> torch.Tensor:
        """从文件加载音频"""
        try:
            # 使用soundfile加载音频
            audio, sr = sf.read(file_path)
            
            # 转换为torch张量
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            audio = torch.from_numpy(audio)
            
            # 处理立体声
            if audio.dim() > 1:
                audio = torch.mean(audio, dim=-1)
            
            # 重采样
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            return self.preprocess(audio)
            
        except Exception as e:
            logger.error(f"加载音频文件失败 {file_path}: {e}")
            # 返回零张量作为回退
            return torch.zeros(self.max_audio_length)
    
    def preprocess(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """预处理音频数据"""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
        
        # 确保是1D张量
        if audio.dim() > 1:
            audio = torch.mean(audio, dim=-1)
        
        # 长度处理
        audio = self._handle_length(audio)
        
        # 归一化
        if self.normalize:
            audio = self._normalize_audio(audio)
        
        # 数据增强（训练时）
        if self.config.get('training', False):
            audio = self._augment_audio(audio)
        
        return audio
    
    def _handle_length(self, audio: torch.Tensor) -> torch.Tensor:
        """处理音频长度"""
        current_length = audio.size(0)
        
        if current_length > self.max_audio_length:
            # 裁剪
            start_idx = torch.randint(0, current_length - self.max_audio_length + 1, (1,)).item()
            audio = audio[start_idx:start_idx + self.max_audio_length]
        elif current_length < self.max_audio_length:
            # 填充
            padding = self.max_audio_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding), 'constant', 0)
        
        return audio
    
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """归一化音频"""
        # RMS归一化
        rms = torch.sqrt(torch.mean(audio**2))
        if rms > 1e-6:
            audio = audio / rms * 0.1
        
        # 限制幅值
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """音频数据增强"""
        # 添加噪声
        if self.add_noise and torch.rand(1) < 0.3:
            noise = torch.randn_like(audio) * self.noise_factor
            audio = audio + noise
        
        # 音量变化
        if self.volume_range != (1.0, 1.0) and torch.rand(1) < 0.5:
            volume_factor = torch.uniform(self.volume_range[0], self.volume_range[1])
            audio = audio * volume_factor
        
        # 音调变化（简化实现）
        if self.pitch_shift_range > 0 and torch.rand(1) < 0.3:
            # 这里需要更复杂的音调变换实现
            pass
        
        return audio
    
    def extract_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取音频特征"""
        features = {}
        
        # Mel频谱图
        mel_spec = self.mel_transform(audio)
        features['mel_spectrogram'] = torch.log(mel_spec + 1e-8)
        
        # MFCC特征
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_mels': self.n_mels,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length
            }
        )
        features['mfcc'] = mfcc(audio)
        
        # 零交叉率
        zcr = self._compute_zcr(audio)
        features['zcr'] = zcr
        
        # 光谱质心
        spectral_centroid = self._compute_spectral_centroid(audio)
        features['spectral_centroid'] = spectral_centroid
        
        return features
    
    def _compute_zcr(self, audio: torch.Tensor) -> torch.Tensor:
        """计算零交叉率"""
        # 简化实现
        diff = torch.diff(torch.sign(audio))
        zcr = torch.sum(torch.abs(diff)) / (2 * len(audio))
        return zcr.unsqueeze(0)
    
    def _compute_spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
        """计算光谱质心"""
        # 使用FFT计算频谱
        fft = torch.fft.fft(audio)
        magnitude = torch.abs(fft)
        
        # 频率轴
        freqs = torch.fft.fftfreq(len(audio), 1/self.sample_rate)
        freqs = freqs[:len(freqs)//2]  # 只取正频率
        magnitude = magnitude[:len(magnitude)//2]
        
        # 计算质心
        if torch.sum(magnitude) > 0:
            centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)
        else:
            centroid = torch.tensor(0.0)
        
        return centroid.unsqueeze(0)


class ImagePreprocessor:
    """
    图像预处理器
    
    负责图像的加载、预处理和增强
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.img_size = config.get('img_size', 224)
        self.normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # 训练时的增强变换
        self.train_transform = transforms.Compose([
            transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
            transforms.RandomCrop((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def load_from_file(self, file_path: str) -> torch.Tensor:
        """从文件加载图像"""
        try:
            image = Image.open(file_path).convert('RGB')
            return self.preprocess(image)
        except Exception as e:
            logger.error(f"加载图像文件失败 {file_path}: {e}")
            # 返回零张量作为回退
            return torch.zeros(3, self.img_size, self.img_size)
    
    def preprocess(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """预处理图像数据"""
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                # CHW格式转换为HWC
                image = image.permute(1, 2, 0)
            image_np = image.numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
        
        # 应用变换
        if self.config.get('training', False):
            return self.train_transform(image)
        else:
            return self.base_transform(image)
    
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取图像特征"""
        features = {}
        
        # 如果图像已经标准化，需要反标准化进行特征提取
        if image.min() < 0:  # 假设已标准化
            # 反标准化
            mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
            std = torch.tensor(self.normalize_std).view(3, 1, 1)
            denorm_image = image * std + mean
        else:
            denorm_image = image
        
        # 颜色直方图
        features['color_histogram'] = self._compute_color_histogram(denorm_image)
        
        # 纹理特征
        features['texture_features'] = self._compute_texture_features(denorm_image)
        
        return features
    
    def _compute_color_histogram(self, image: torch.Tensor) -> torch.Tensor:
        """计算颜色直方图"""
        # 简化实现：计算RGB通道的均值和标准差
        mean_rgb = torch.mean(image, dim=(1, 2))
        std_rgb = torch.std(image, dim=(1, 2))
        return torch.cat([mean_rgb, std_rgb])
    
    def _compute_texture_features(self, image: torch.Tensor) -> torch.Tensor:
        """计算纹理特征"""
        # 简化实现：计算梯度统计
        gray = torch.mean(image, dim=0)
        
        # Sobel滤波器
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 应用滤波器
        grad_x = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        # 梯度幅值
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 统计特征
        features = torch.tensor([
            torch.mean(grad_magnitude),
            torch.std(grad_magnitude),
            torch.max(grad_magnitude),
            torch.min(grad_magnitude)
        ])
        
        return features


class GesturePreprocessor:
    """
    手势预处理器
    
    负责手势关键点数据的处理和增强
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_sequence_length = config.get('max_sequence_length', 30)
        self.num_hands = config.get('num_hands', 2)
        self.num_keypoints = config.get('num_keypoints', 21)
        self.coordinate_dim = config.get('coordinate_dim', 3)
        self.normalize_coordinates = config.get('normalize_coordinates', True)
        self.use_mediapipe = config.get('use_mediapipe', True)
        
        # MediaPipe手部检测器
        if self.use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.num_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def load_from_file(self, file_path: str) -> torch.Tensor:
        """从文件加载手势数据"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix == '.json':
                # 加载JSON格式的关键点数据
                with open(file_path, 'r') as f:
                    data = json.load(f)
                keypoints = np.array(data['keypoints'])
                return self.preprocess(keypoints)
            
            elif file_path.suffix in ['.mp4', '.avi', '.mov']:
                # 从视频中提取手势关键点
                return self._extract_from_video(str(file_path))
            
            else:
                # 尝试作为numpy数组加载
                data = np.load(file_path)
                return self.preprocess(data)
                
        except Exception as e:
            logger.error(f"加载手势文件失败 {file_path}: {e}")
            # 返回零张量作为回退
            return torch.zeros(self.max_sequence_length, self.num_hands, self.num_keypoints, self.coordinate_dim)
    
    def _extract_from_video(self, video_path: str) -> torch.Tensor:
        """从视频中提取手势关键点"""
        if not self.use_mediapipe:
            logger.warning("MediaPipe未启用，无法从视频提取手势")
            return torch.zeros(self.max_sequence_length, self.num_hands, self.num_keypoints, self.coordinate_dim)
        
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测手部关键点
                results = self.hands.process(rgb_frame)
                
                # 提取关键点
                frame_keypoints = self._extract_keypoints_from_results(results, frame.shape)
                keypoints_sequence.append(frame_keypoints)
                
                if len(keypoints_sequence) >= self.max_sequence_length:
                    break
            
        finally:
            cap.release()
        
        # 转换为numpy数组
        if keypoints_sequence:
            keypoints_array = np.array(keypoints_sequence)
        else:
            keypoints_array = np.zeros((1, self.num_hands, self.num_keypoints, self.coordinate_dim))
        
        return self.preprocess(keypoints_array)
    
    def _extract_keypoints_from_results(self, results, frame_shape) -> np.ndarray:
        """从MediaPipe结果中提取关键点"""
        h, w = frame_shape[:2]
        frame_keypoints = np.zeros((self.num_hands, self.num_keypoints, self.coordinate_dim))
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= self.num_hands:
                    break
                
                for kp_idx, landmark in enumerate(hand_landmarks.landmark):
                    if kp_idx >= self.num_keypoints:
                        break
                    
                    # 转换为像素坐标
                    frame_keypoints[hand_idx, kp_idx, 0] = landmark.x * w
                    frame_keypoints[hand_idx, kp_idx, 1] = landmark.y * h
                    frame_keypoints[hand_idx, kp_idx, 2] = landmark.z
        
        return frame_keypoints
    
    def preprocess(self, keypoints: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """预处理手势关键点数据"""
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints.astype(np.float32))
        
        # 确保形状正确
        keypoints = self._handle_shape(keypoints)
        
        # 长度处理
        keypoints = self._handle_sequence_length(keypoints)
        
        # 坐标归一化
        if self.normalize_coordinates:
            keypoints = self._normalize_coordinates(keypoints)
        
        # 数据增强（训练时）
        if self.config.get('training', False):
            keypoints = self._augment_keypoints(keypoints)
        
        return keypoints
    
    def _handle_shape(self, keypoints: torch.Tensor) -> torch.Tensor:
        """处理关键点形状"""
        # 期望形状: [T, num_hands, num_keypoints, coordinate_dim]
        if keypoints.dim() == 2:
            # [num_points, coordinate_dim] -> [1, 1, num_points, coordinate_dim]
            keypoints = keypoints.unsqueeze(0).unsqueeze(0)
        elif keypoints.dim() == 3:
            # [T, num_points, coordinate_dim] -> [T, 1, num_points, coordinate_dim]
            keypoints = keypoints.unsqueeze(1)
        
        # 调整手部数量
        current_hands = keypoints.size(1)
        if current_hands > self.num_hands:
            keypoints = keypoints[:, :self.num_hands]
        elif current_hands < self.num_hands:
            padding = torch.zeros(
                keypoints.size(0), 
                self.num_hands - current_hands,
                keypoints.size(2),
                keypoints.size(3)
            )
            keypoints = torch.cat([keypoints, padding], dim=1)
        
        # 调整关键点数量
        current_keypoints = keypoints.size(2)
        if current_keypoints > self.num_keypoints:
            keypoints = keypoints[:, :, :self.num_keypoints]
        elif current_keypoints < self.num_keypoints:
            padding = torch.zeros(
                keypoints.size(0),
                keypoints.size(1),
                self.num_keypoints - current_keypoints,
                keypoints.size(3)
            )
            keypoints = torch.cat([keypoints, padding], dim=2)
        
        # 调整坐标维度
        current_dim = keypoints.size(3)
        if current_dim > self.coordinate_dim:
            keypoints = keypoints[:, :, :, :self.coordinate_dim]
        elif current_dim < self.coordinate_dim:
            padding = torch.zeros(
                keypoints.size(0),
                keypoints.size(1),
                keypoints.size(2),
                self.coordinate_dim - current_dim
            )
            keypoints = torch.cat([keypoints, padding], dim=3)
        
        return keypoints
    
    def _handle_sequence_length(self, keypoints: torch.Tensor) -> torch.Tensor:
        """处理序列长度"""
        current_length = keypoints.size(0)
        
        if current_length > self.max_sequence_length:
            # 随机裁剪或均匀采样
            if self.config.get('training', False):
                # 训练时随机裁剪
                start_idx = torch.randint(0, current_length - self.max_sequence_length + 1, (1,)).item()
                keypoints = keypoints[start_idx:start_idx + self.max_sequence_length]
            else:
                # 推理时均匀采样
                indices = torch.linspace(0, current_length - 1, self.max_sequence_length).long()
                keypoints = keypoints[indices]
        
        elif current_length < self.max_sequence_length:
            # 零填充
            padding = torch.zeros(
                self.max_sequence_length - current_length,
                keypoints.size(1),
                keypoints.size(2),
                keypoints.size(3)
            )
            keypoints = torch.cat([keypoints, padding], dim=0)
        
        return keypoints
    
    def _normalize_coordinates(self, keypoints: torch.Tensor) -> torch.Tensor:
        """归一化坐标"""
        # 对每个手部分别归一化
        for hand_idx in range(keypoints.size(1)):
            hand_keypoints = keypoints[:, hand_idx]  # [T, num_keypoints, coordinate_dim]
            
            # 检查是否有有效关键点
            valid_mask = torch.any(hand_keypoints != 0, dim=(0, 2))  # [num_keypoints]
            
            if torch.any(valid_mask):
                valid_keypoints = hand_keypoints[:, valid_mask]  # [T, valid_keypoints, coordinate_dim]
                
                # 计算边界框
                min_coords = torch.min(valid_keypoints, dim=1, keepdim=True)[0]  # [T, 1, coordinate_dim]
                max_coords = torch.max(valid_keypoints, dim=1, keepdim=True)[0]  # [T, 1, coordinate_dim]
                
                # 避免除零
                range_coords = max_coords - min_coords
                range_coords = torch.where(range_coords == 0, torch.ones_like(range_coords), range_coords)
                
                # 归一化到[0, 1]
                normalized = (hand_keypoints - min_coords) / range_coords
                
                # 处理无效点
                invalid_mask = torch.all(hand_keypoints == 0, dim=2, keepdim=True)  # [T, num_keypoints, 1]
                normalized = torch.where(invalid_mask, torch.zeros_like(normalized), normalized)
                
                keypoints[:, hand_idx] = normalized
        
        return keypoints
    
    def _augment_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """关键点数据增强"""
        # 随机噪声
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(keypoints) * 0.02
            keypoints = keypoints + noise
        
        # 随机缩放
        if torch.rand(1) < 0.3:
            scale_factor = torch.uniform(0.8, 1.2, (1,)).item()
            keypoints = keypoints * scale_factor
        
        # 随机时序抖动
        if torch.rand(1) < 0.2:
            keypoints = self._temporal_jitter(keypoints)
        
        return keypoints
    
    def _temporal_jitter(self, keypoints: torch.Tensor) -> torch.Tensor:
        """时序抖动增强"""
        seq_len = keypoints.size(0)
        if seq_len <= 2:
            return keypoints
        
        # 创建轻微的时序扰动
        jitter_strength = min(2, seq_len // 10)
        if jitter_strength > 0:
            for i in range(1, seq_len - 1):
                if torch.rand(1) < 0.1:
                    # 随机交换相邻帧
                    swap_idx = i + torch.randint(-jitter_strength, jitter_strength + 1, (1,)).item()
                    swap_idx = max(0, min(seq_len - 1, swap_idx))
                    if swap_idx != i:
                        keypoints[i], keypoints[swap_idx] = keypoints[swap_idx].clone(), keypoints[i].clone()
        
        return keypoints
    
    def extract_features(self, keypoints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取手势特征"""
        features = {}
        
        # 速度特征
        features['velocity'] = self._compute_velocity(keypoints)
        
        # 加速度特征
        features['acceleration'] = self._compute_acceleration(keypoints)
        
        # 手部形状特征
        features['hand_shape'] = self._compute_hand_shape_features(keypoints)
        
        # 时序统计特征
        features['temporal_stats'] = self._compute_temporal_stats(keypoints)
        
        return features
    
    def _compute_velocity(self, keypoints: torch.Tensor) -> torch.Tensor:
        """计算速度特征"""
        if keypoints.size(0) < 2:
            return torch.zeros_like(keypoints)
        
        velocity = torch.diff(keypoints, dim=0)
        # 添加零填充使维度匹配
        velocity = torch.cat([velocity, torch.zeros_like(velocity[-1:])], dim=0)
        
        return velocity
    
    def _compute_acceleration(self, keypoints: torch.Tensor) -> torch.Tensor:
        """计算加速度特征"""
        velocity = self._compute_velocity(keypoints)
        if velocity.size(0) < 2:
            return torch.zeros_like(velocity)
        
        acceleration = torch.diff(velocity, dim=0)
        # 添加零填充
        acceleration = torch.cat([acceleration, torch.zeros_like(acceleration[-1:])], dim=0)
        
        return acceleration
    
    def _compute_hand_shape_features(self, keypoints: torch.Tensor) -> torch.Tensor:
        """计算手部形状特征"""
        # 计算手指间的距离
        features_list = []
        
        for hand_idx in range(keypoints.size(1)):
            hand_kp = keypoints[:, hand_idx]  # [T, 21, 3]
            
            # 计算指尖到手腕的距离
            wrist = hand_kp[:, 0]  # 手腕点 [T, 3]
            fingertips = [4, 8, 12, 16, 20]  # 指尖点索引
            
            distances = []
            for tip_idx in fingertips:
                tip = hand_kp[:, tip_idx]  # [T, 3]
                dist = torch.norm(tip - wrist, dim=1)  # [T]
                distances.append(dist)
            
            hand_features = torch.stack(distances, dim=1)  # [T, 5]
            features_list.append(hand_features)
        
        # 合并所有手部特征
        all_features = torch.stack(features_list, dim=1)  # [T, num_hands, 5]
        
        return all_features
    
    def _compute_temporal_stats(self, keypoints: torch.Tensor) -> torch.Tensor:
        """计算时序统计特征"""
        # 计算每个关键点的时序统计
        mean_pos = torch.mean(keypoints, dim=0)  # [num_hands, num_keypoints, 3]
        std_pos = torch.std(keypoints, dim=0)    # [num_hands, num_keypoints, 3]
        
        # 展平并合并
        temporal_features = torch.cat([
            mean_pos.flatten(),
            std_pos.flatten()
        ])
        
        return temporal_features