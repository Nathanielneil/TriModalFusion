"""
Basic usage example for TriModalFusion.

This example demonstrates how to use the TriModalFusion model for multimodal
classification with speech, gesture, and image inputs.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = f"{project_root}:{project_root / 'src'}:{os.environ.get('PYTHONPATH', '')}"

try:
    from src.models.trimodal_fusion import TriModalFusionModel
    from src.utils.config import load_config
    from src.utils.logging_utils import setup_logging
except ImportError:
    # Fallback for relative imports
    from models.trimodal_fusion import TriModalFusionModel
    from utils.config import load_config
    from utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def create_sample_data(batch_size: int = 4):
    """Create sample multimodal data for demonstration."""
    
    # Sample speech data (audio waveform)
    speech_data = torch.randn(batch_size, 16000)  # 1 second at 16kHz
    
    # Sample gesture data (keypoints sequence)
    # [batch_size, sequence_length, num_hands, num_joints, coordinates]
    gesture_data = torch.randn(batch_size, 30, 2, 21, 3)  # 30 frames, 2 hands, 21 joints, (x,y,z)
    
    # Sample image data
    image_data = torch.randn(batch_size, 3, 224, 224)  # RGB images
    
    # Sample targets for classification
    targets = torch.randint(0, 10, (batch_size,))  # 10 classes
    
    return {
        'speech': speech_data,
        'gesture': gesture_data, 
        'image': image_data
    }, targets


def main():
    """Main function demonstrating basic TriModalFusion usage."""
    
    logger.info("Starting TriModalFusion basic usage example")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    config = load_config(config_path)
    
    # Create model
    logger.info("Creating TriModalFusion model...")
    model = TriModalFusionModel(config)
    
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Print model info
    model_info = model.get_model_info()
    for key, value in model_info.items():
        if key != 'config':  # Skip printing full config
            logger.info(f"{key}: {value}")
    
    # Create sample data
    logger.info("Creating sample data...")
    inputs, targets = create_sample_data(batch_size=2)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        outputs = model(inputs)
    
    # Display results
    logger.info("Forward pass completed!")
    logger.info(f"Available outputs: {list(outputs.keys())}")
    
    # Task outputs
    task_outputs = outputs['task_outputs']
    logger.info(f"Task outputs: {list(task_outputs.keys())}")
    
    if 'classification' in task_outputs:
        classification_logits = task_outputs['classification']
        logger.info(f"Classification logits shape: {classification_logits.shape}")
        
        # Get predictions
        predictions = torch.argmax(classification_logits, dim=1)
        logger.info(f"Predictions: {predictions.tolist()}")
        logger.info(f"Targets: {targets.tolist()}")
    
    # Feature shapes
    logger.info(f"Fused features shape: {outputs['fused_features'].shape}")
    
    if 'encoded_features' in outputs:
        for modality, features in outputs['encoded_features'].items():
            logger.info(f"{modality} encoded features shape: {features.shape}")
    
    # Training mode example
    logger.info("\n" + "="*50)
    logger.info("Training mode example")
    
    # Set to training mode
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Forward pass with loss computation
    outputs = model(inputs)
    
    # Compute loss
    target_dict = {'classification': targets}
    losses = model.compute_loss(outputs, target_dict)
    
    logger.info(f"Computed losses: {list(losses.keys())}")
    for loss_name, loss_value in losses.items():
        logger.info(f"{loss_name}: {loss_value.item():.4f}")
    
    # Backward pass
    total_loss = losses['total_loss']
    total_loss.backward()
    
    logger.info("Backward pass completed")
    
    # Check gradients
    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() 
                   if p.grad is not None) ** 0.5
    logger.info(f"Gradient norm: {grad_norm:.4f}")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    logger.info("Optimization step completed")
    
    # Feature extraction example
    logger.info("\n" + "="*50)
    logger.info("Feature extraction example")
    
    features = model.extract_features(inputs)
    for feature_type, feature_tensor in features.items():
        logger.info(f"{feature_type} shape: {feature_tensor.shape}")
    
    # Inference example
    logger.info("\n" + "="*50)
    logger.info("Inference example")
    
    predictions = model.inference(inputs, task='classification')
    logger.info(f"Inference predictions shape: {predictions.shape}")
    
    predicted_classes = torch.argmax(predictions, dim=1)
    logger.info(f"Predicted classes: {predicted_classes.tolist()}")
    
    # Single modality example
    logger.info("\n" + "="*50)
    logger.info("Single modality example")
    
    # Test with only speech input
    speech_only_inputs = {'speech': inputs['speech']}
    speech_outputs = model(speech_only_inputs)
    logger.info(f"Speech-only classification shape: {speech_outputs['task_outputs']['classification'].shape}")
    
    # Test with only image input
    image_only_inputs = {'image': inputs['image']}
    image_outputs = model(image_only_inputs)
    logger.info(f"Image-only classification shape: {image_outputs['task_outputs']['classification'].shape}")
    
    logger.info("Basic usage example completed successfully!")


if __name__ == "__main__":
    main()