#!/usr/bin/env python3
"""
Train traffic sign recognition models on GTSRB dataset.

Usage:
    python scripts/train.py --data_dir /path/to/GTSRB --models resnet50 efficientnet_b0 vit
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.config import get_config
from data.dataloader import get_dataloaders
from models.factory import get_model
from src.trainer import ModelTrainer, train_all_models


def parse_args():
    parser = argparse.ArgumentParser(description='Train traffic sign recognition models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GTSRB dataset')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'efficientnet_b0', 'vit'],
                       help='Models to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRAFFIC SIGN RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print()
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Train models
    results = train_all_models(
        model_names=args.models,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Train Acc: {result['train_metrics']['accuracy']:.2f}%")
        print(f"  Val Acc:   {result['val_metrics']['accuracy']:.2f}%")
        print(f"  Test Acc:  {result['test_metrics']['accuracy']:.2f}%")
    
    print(f"\nCheckpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
