"""
Data loading utilities for GTSRB dataset.
"""

import os
import glob
import logging
from typing import Tuple, Optional, List
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GTSRBDataset(Dataset):
    """
    Custom PyTorch Dataset for German Traffic Sign Recognition Benchmark (GTSRB).
    
    Supports the official GTSRB folder structure with separate train/test splits.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        num_classes: int = 43
    ):
        """
        Initialize GTSRBDataset.
        
        Args:
            root_dir: Path to the GTSRB dataset directory
            split: 'train' or 'test'
            transform: Torchvision transforms to apply
            num_classes: Number of traffic sign classes (default: 43)
        """
        if split not in ['train', 'test']:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.images: List[str] = []
        self.labels: List[int] = []
        
        if split == 'train':
            self._load_train_data()
        else:
            self._load_test_data()
    
    def _load_train_data(self):
        """Load training data from GTSRB folder structure."""
        train_dir = self.root_dir / 'GTSRB_final_training_images'
        
        if not train_dir.exists():
            # Try alternative path
            train_dir = self.root_dir / 'Train'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        for class_id in range(self.num_classes):
            class_folder = train_dir / f'{class_id:05d}'
            
            if not class_folder.exists():
                continue
            
            image_files = list(class_folder.glob('*.ppm')) + list(class_folder.glob('*.png'))
            
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(class_id)
        
        logger.info(f"Loaded {len(self.images)} training images from {self.num_classes} classes")
    
    def _load_test_data(self):
        """Load test data using ground truth CSV."""
        test_dir = self.root_dir / 'GTSRB_final_test_images'
        csv_path = self.root_dir / 'GTSRB_Final_Test_GT' / 'GT-final_test.csv'
        
        if not test_dir.exists():
            test_dir = self.root_dir / 'Test'
        
        if not csv_path.exists():
            csv_path = self.root_dir / 'GT-final_test.csv'
        
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=';')
            
            for _, row in df.iterrows():
                img_name = row['Filename']
                class_id = row['ClassId']
                img_path = test_dir / img_name
                
                if img_path.exists():
                    self.images.append(str(img_path))
                    self.labels.append(class_id)
        else:
            # Fallback: try to load from folder structure
            for img_path in test_dir.glob('*.ppm'):
                self.images.append(str(img_path))
                self.labels.append(0)  # Unknown label
            logger.warning("Test CSV not found, labels may be incorrect")
        
        logger.info(f"Loaded {len(self.images)} test images")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DiffusedTrafficSignDataset(Dataset):
    """
    Dataset for diffusion-perturbed traffic sign images.
    
    Returns both original and adversarial images for comparison.
    """
    
    def __init__(
        self,
        metadata: List[Tuple[str, str, int, str]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        """
        Initialize DiffusedTrafficSignDataset.
        
        Args:
            metadata: List of (orig_path, adv_path, label, prompt) tuples
            transform: Torchvision transforms to apply
            image_size: Target image size
        """
        self.metadata = metadata
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        orig_path, adv_path, label, prompt = self.metadata[idx]
        
        orig_img = Image.open(orig_path).convert("RGB").resize((self.image_size, self.image_size))
        adv_img = Image.open(adv_path).convert("RGB").resize((self.image_size, self.image_size))
        
        if self.transform:
            orig_img = self.transform(orig_img)
            adv_img = self.transform(adv_img)
        
        return orig_img, adv_img, int(label), prompt


def get_transforms(
    mode: str = 'train',
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    """
    Get data augmentation and normalization transforms.
    
    Args:
        mode: 'train' for training augmentations, 'test' for inference
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_dataloaders(
    root_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 8,
    pin_memory: bool = True,
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Path to GTSRB dataset
        batch_size: Batch size
        val_split: Fraction of training data for validation
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_transforms('train', image_size, mean, std)
    test_transform = get_transforms('test', image_size, mean, std)
    
    logger.info("Loading training dataset...")
    train_full = GTSRBDataset(root_dir, split='train', transform=train_transform)
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(train_full))
    val_size = len(train_full) - train_size
    
    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info("Loading test dataset...")
    test_dataset = GTSRBDataset(root_dir, split='test', transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_diffusion_dataloader(
    metadata: List[Tuple[str, str, int, str]],
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> DataLoader:
    """
    Create dataloader for diffusion-perturbed images.
    
    Args:
        metadata: List of (orig_path, adv_path, label, prompt) tuples
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        DataLoader for diffused images
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = DiffusedTrafficSignDataset(metadata, transform=transform, image_size=image_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
