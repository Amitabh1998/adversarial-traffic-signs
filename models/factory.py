"""
Model factory for creating and managing different architectures.
Supports: ResNet-50, EfficientNet-B0, Vision Transformer (ViT)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ViTLogitsWrapper(nn.Module):
    """
    Wrapper for HuggingFace ViT model to extract logits directly.
    
    This ensures consistent output format across all model architectures.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if hasattr(output, 'logits'):
            return output.logits
        return output


def get_model(
    model_name: str,
    num_classes: int = 43,
    pretrained: bool = True,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create different model architectures.
    
    Args:
        model_name: Model architecture name ('resnet50', 'efficientnet_b0', 'vit')
        num_classes: Number of output classes (default: 43 for GTSRB)
        pretrained: Whether to use pretrained weights
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Model loaded on specified device
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        logger.info(f"Created ResNet-50 with {count_parameters(model):,} parameters")
    
    elif model_name == 'efficientnet_b0':
        try:
            import timm
            model = timm.create_model(
                'efficientnet_b0',
                pretrained=pretrained,
                num_classes=num_classes
            )
            logger.info(f"Created EfficientNet-B0 with {count_parameters(model):,} parameters")
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
    
    elif model_name == 'vit':
        try:
            from transformers import ViTForImageClassification
            model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            logger.info(f"Created ViT-Base with {count_parameters(model):,} parameters")
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: resnet50, efficientnet_b0, vit"
        )
    
    return model.to(device)


def load_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int = 43,
    device: str = 'cuda',
    wrap_vit: bool = True
) -> nn.Module:
    """
    Load a model from checkpoint.
    
    Args:
        model_name: Model architecture name
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        device: Device to load model on
        wrap_vit: Whether to wrap ViT models for consistent output
    
    Returns:
        Loaded model in evaluation mode
    """
    model = get_model(model_name, num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    if model_name.lower() == 'vit' and wrap_vit:
        model = ViTLogitsWrapper(model)
    
    logger.info(f"Loaded {model_name} from {checkpoint_path}")
    return model


def load_all_models(
    checkpoint_dir: str,
    model_names: list = ['resnet50', 'efficientnet_b0', 'vit'],
    num_classes: int = 43,
    device: str = 'cuda'
) -> Dict[str, nn.Module]:
    """
    Load all models from checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_names: List of model names to load
        num_classes: Number of output classes
        device: Device to load models on
    
    Returns:
        Dictionary mapping model names to loaded models
    """
    from pathlib import Path
    
    models_dict = {}
    checkpoint_dir = Path(checkpoint_dir)
    
    for model_name in model_names:
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        
        if checkpoint_path.exists():
            models_dict[model_name] = load_model(
                model_name,
                str(checkpoint_path),
                num_classes=num_classes,
                device=device
            )
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    return models_dict


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
