"""
Configuration management for the adversarial traffic sign recognition project.
"""

import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class DatasetConfig:
    name: str = "GTSRB"
    num_classes: int = 43
    image_size: int = 224
    val_split: float = 0.1
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001


@dataclass
class AttackConfig:
    # FGSM
    fgsm_eps: float = 0.03
    # PGD
    pgd_eps: float = 0.03
    pgd_alpha: float = 0.01
    pgd_steps: int = 10
    # CW
    cw_c: float = 1
    cw_kappa: float = 0
    cw_steps: int = 100
    cw_lr: float = 0.01


@dataclass
class DiffusionConfig:
    model: str = "runwayml/stable-diffusion-v1-5"
    strength: float = 0.7
    guidance_scale: float = 7.5
    num_samples: int = 1000


@dataclass
class Config:
    """Main configuration class."""
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = None
    results_dir: Path = None
    checkpoints_dir: Path = None
    figures_dir: Path = None
    logs_dir: Path = None
    
    # Sub-configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 8
    pin_memory: bool = True
    
    # Models to train
    models: List[str] = field(default_factory=lambda: ["resnet50", "efficientnet_b0", "vit"])
    
    # Evaluation
    eval_max_batches: Optional[int] = 50
    
    def __post_init__(self):
        """Initialize paths after creation."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.results_dir / "checkpoints"
        if self.figures_dir is None:
            self.figures_dir = self.results_dir / "figures"
        if self.logs_dir is None:
            self.logs_dir = self.results_dir / "logs"
        
        # Create directories
        for dir_path in [self.data_dir, self.results_dir, self.checkpoints_dir, 
                         self.figures_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        config = cls()
        
        # Update dataset config
        if 'dataset' in yaml_config:
            for key, value in yaml_config['dataset'].items():
                if hasattr(config.dataset, key):
                    setattr(config.dataset, key, value)
        
        # Update training config
        if 'training' in yaml_config:
            tc = yaml_config['training']
            config.training.batch_size = tc.get('batch_size', config.training.batch_size)
            config.training.num_epochs = tc.get('num_epochs', config.training.num_epochs)
            config.training.learning_rate = tc.get('learning_rate', config.training.learning_rate)
            config.training.weight_decay = tc.get('weight_decay', config.training.weight_decay)
            if 'early_stopping' in tc:
                config.training.early_stopping_enabled = tc['early_stopping'].get('enabled', True)
                config.training.early_stopping_patience = tc['early_stopping'].get('patience', 5)
        
        # Update attack config
        if 'attacks' in yaml_config:
            ac = yaml_config['attacks']
            if 'fgsm' in ac:
                config.attack.fgsm_eps = ac['fgsm'].get('eps', config.attack.fgsm_eps)
            if 'pgd' in ac:
                config.attack.pgd_eps = ac['pgd'].get('eps', config.attack.pgd_eps)
                config.attack.pgd_alpha = ac['pgd'].get('alpha', config.attack.pgd_alpha)
                config.attack.pgd_steps = ac['pgd'].get('steps', config.attack.pgd_steps)
            if 'cw' in ac:
                config.attack.cw_c = ac['cw'].get('c', config.attack.cw_c)
                config.attack.cw_steps = ac['cw'].get('steps', config.attack.cw_steps)
        
        # Update diffusion config
        if 'diffusion' in yaml_config:
            dc = yaml_config['diffusion']
            config.diffusion.model = dc.get('model', config.diffusion.model)
            config.diffusion.strength = dc.get('strength', config.diffusion.strength)
            config.diffusion.num_samples = dc.get('num_samples', config.diffusion.num_samples)
        
        # Update models list
        if 'models' in yaml_config:
            config.models = yaml_config['models']
        
        # Update hardware settings
        if 'hardware' in yaml_config:
            config.num_workers = yaml_config['hardware'].get('num_workers', config.num_workers)
            config.pin_memory = yaml_config['hardware'].get('pin_memory', config.pin_memory)
        
        # Update evaluation settings
        if 'evaluation' in yaml_config:
            config.eval_max_batches = yaml_config['evaluation'].get('max_batches', config.eval_max_batches)
        
        return config


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration, optionally from a YAML file."""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config()


# Default configuration instance
config = get_config()
