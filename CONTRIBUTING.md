# Contributing to TriModalFusion

We welcome contributions to TriModalFusion! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please treat all contributors with respect and create an inclusive environment for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a feature branch from `main`
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (optional, for accelerated training/inference)

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/TriModalFusion.git
cd TriModalFusion

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .

# Run tests to verify installation
python -m pytest tests/
```

### Additional Dependencies

For full functionality, install optional dependencies:

```bash
# Web demo dependencies
pip install fastapi uvicorn jinja2 python-multipart websockets pillow

# Computer vision dependencies
pip install opencv-python mediapipe

# Audio processing dependencies
pip install librosa soundfile

# Development tools
pip install black flake8 isort pytest pytest-cov
```

## Making Changes

### Branch Naming

Use descriptive branch names that indicate the type of change:

- `feature/add-new-encoder` for new features
- `fix/memory-leak-in-training` for bug fixes
- `docs/update-api-documentation` for documentation updates
- `refactor/simplify-fusion-logic` for code refactoring

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for custom gesture vocabularies

- Implement GestureVocabulary class for managing gesture sets
- Add configuration options for custom gesture definitions
- Update preprocessing pipeline to handle custom vocabularies
- Add tests for vocabulary management functionality

Fixes #123
```

### Types of Contributions

We welcome the following types of contributions:

#### 1. Bug Fixes
- Fix existing functionality that is not working correctly
- Address performance issues
- Resolve compatibility problems

#### 2. New Features
- Add new modality encoders
- Implement new fusion mechanisms
- Create additional evaluation metrics
- Enhance web interface functionality

#### 3. Documentation
- Improve API documentation
- Add tutorials and examples
- Update configuration guides
- Translate documentation

#### 4. Testing
- Add unit tests for new functionality
- Improve test coverage
- Add integration tests
- Create performance benchmarks

#### 5. Performance Improvements
- Optimize model architectures
- Improve training efficiency
- Reduce memory usage
- Accelerate inference

## Submitting Changes

### Pull Request Process

1. **Before Creating a PR:**
   - Ensure your code follows the coding standards
   - Run the test suite and ensure all tests pass
   - Update documentation if necessary
   - Add tests for new functionality

2. **Creating the PR:**
   - Use a descriptive title that summarizes the changes
   - Provide a detailed description of what was changed and why
   - Reference any related issues using `Fixes #issue-number`
   - Include screenshots for UI changes

3. **PR Description Template:**

```markdown
## Summary
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally with my changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

### Review Process

1. All PRs require at least one review from a maintainer
2. Address review feedback promptly
3. Keep PRs focused and reasonably sized
4. Be responsive to questions and suggestions

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

```bash
# Format code with Black
black src/ tests/

# Check style with Flake8
flake8 src/ tests/

# Sort imports with isort
isort src/ tests/
```

### Code Quality Guidelines

1. **Docstrings:** Use Google-style docstrings for all public functions and classes
2. **Type Hints:** Add type hints to function signatures
3. **Error Handling:** Use appropriate exception types and provide meaningful error messages
4. **Logging:** Use the logging module instead of print statements
5. **Constants:** Define constants at module level in UPPERCASE

### Example Code Style

```python
"""Module for multimodal fusion operations."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Constants
DEFAULT_FUSION_DIM = 512
SUPPORTED_MODALITIES = ["speech", "image", "gesture"]


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module using multi-head attention.
    
    This module implements cross-modal attention mechanisms to fuse
    features from multiple modalities.
    
    Args:
        d_model: Model dimension for all modalities.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        
    Example:
        >>> fusion = CrossModalFusion(d_model=512, num_heads=8)
        >>> features = {"speech": speech_tensor, "image": image_tensor}
        >>> fused = fusion(features)
    """
    
    def __init__(
        self,
        d_model: int = DEFAULT_FUSION_DIM,
        num_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        logger.info(f"Initialized CrossModalFusion with d_model={d_model}, num_heads={num_heads}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion module.
        
        Args:
            features: Dictionary mapping modality names to feature tensors.
            
        Returns:
            Fused feature tensor of shape (batch_size, d_model).
            
        Raises:
            ValueError: If no features are provided or unsupported modality.
        """
        if not features:
            raise ValueError("No features provided for fusion")
            
        for modality in features.keys():
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(f"Unsupported modality: {modality}")
        
        # Implementation details...
        return fused_features
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_fusion.py

# Run with verbose output
python -m pytest tests/ -v
```

### Writing Tests

1. **Test Organization:**
   - Place tests in the `tests/` directory
   - Mirror the source code structure
   - Use descriptive test names

2. **Test Types:**
   - Unit tests for individual functions/classes
   - Integration tests for module interactions
   - End-to-end tests for complete workflows

3. **Test Example:**

```python
"""Tests for cross-modal fusion functionality."""

import pytest
import torch

from src.fusion.cross_modal_fusion import CrossModalFusion


class TestCrossModalFusion:
    """Test cases for CrossModalFusion class."""
    
    @pytest.fixture
    def fusion_module(self):
        """Create a fusion module for testing."""
        return CrossModalFusion(d_model=512, num_heads=8)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature tensors."""
        return {
            "speech": torch.randn(2, 100, 512),
            "image": torch.randn(2, 196, 512),
            "gesture": torch.randn(2, 30, 512)
        }
    
    def test_initialization(self):
        """Test module initialization."""
        fusion = CrossModalFusion(d_model=256, num_heads=4)
        assert fusion.d_model == 256
        assert fusion.num_heads == 4
    
    def test_forward_pass(self, fusion_module, sample_features):
        """Test forward pass with valid inputs."""
        output = fusion_module(sample_features)
        assert output.shape == (2, 512)
    
    def test_empty_features_raises_error(self, fusion_module):
        """Test that empty features dict raises ValueError."""
        with pytest.raises(ValueError, match="No features provided"):
            fusion_module({})
    
    @pytest.mark.parametrize("d_model,num_heads", [
        (512, 7),  # Not divisible
        (256, 3),  # Not divisible
    ])
    def test_invalid_dimensions_raise_error(self, d_model, num_heads):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            CrossModalFusion(d_model=d_model, num_heads=num_heads)
```

## Documentation

### API Documentation

- Use Google-style docstrings for all public APIs
- Include parameter types and return types
- Provide usage examples
- Document exceptions that may be raised

### User Documentation

- Update relevant documentation files in `docs/`
- Add tutorials for new features
- Update configuration examples
- Include performance benchmarks when relevant

### Web Demo Documentation

- Update `WEB_DEMO_README.md` for interface changes
- Add screenshots for new UI features
- Document new API endpoints
- Update troubleshooting guides

## Release Process

### Version Numbering

We follow semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release notes
6. Tag the release

## Getting Help

- **Questions:** Open a GitHub Discussion
- **Bug Reports:** Create a GitHub Issue
- **Feature Requests:** Create a GitHub Issue with the "enhancement" label
- **Security Issues:** Email the maintainers directly

## Recognition

Contributors are recognized in several ways:
- Listed in the AUTHORS file
- Mentioned in release notes
- GitHub contributor statistics
- Special recognition for significant contributions

Thank you for contributing to TriModalFusion!