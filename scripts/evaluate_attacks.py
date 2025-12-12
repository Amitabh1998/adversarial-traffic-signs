#!/usr/bin/env python3
"""
Evaluate adversarial attacks on traffic sign recognition models.

Usage:
    python scripts/evaluate_attacks.py --data_dir /path/to/GTSRB --checkpoint_dir results/checkpoints
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

from data.dataloader import get_dataloaders, get_diffusion_dataloader
from models.factory import load_all_models
from attacks.classical import evaluate_classical_attacks
from attacks.diffusion import load_metadata
from evaluation.metrics import (
    MetricsCalculator,
    evaluate_diffusion_attacks,
    evaluate_classical_attacks_with_metrics,
    summarize_results
)
from evaluation.visualization import save_all_figures


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate adversarial attacks')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GTSRB dataset')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to model checkpoints')
    parser.add_argument('--diffusion_metadata', type=str, default=None,
                       help='Path to diffusion attack metadata CSV')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--max_batches', type=int, default=50,
                       help='Maximum batches for classical attacks (None for full)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    models = load_all_models(args.checkpoint_dir, device=device)
    model_names = list(models.keys())
    print(f"Loaded models: {model_names}")
    
    # Load test data
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(args.data_dir)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(device=device)
    
    # Use first model as surrogate for classical attacks
    surrogate_model = models[model_names[0]]
    
    # Evaluate classical attacks
    print("\n" + "=" * 60)
    print("Evaluating Classical Attacks (FGSM, PGD, CW)")
    print("=" * 60)
    
    classical_results = evaluate_classical_attacks_with_metrics(
        models=models,
        test_loader=test_loader,
        surrogate_model=surrogate_model,
        metrics_calculator=metrics_calc,
        device=device,
        max_batches=args.max_batches
    )
    
    # Evaluate diffusion attacks if metadata provided
    diffusion_results = None
    if args.diffusion_metadata and Path(args.diffusion_metadata).exists():
        print("\n" + "=" * 60)
        print("Evaluating Diffusion Attacks")
        print("=" * 60)
        
        metadata = load_metadata(args.diffusion_metadata)
        diffusion_loader = get_diffusion_dataloader(metadata)
        
        diffusion_results = evaluate_diffusion_attacks(
            models=models,
            diffusion_loader=diffusion_loader,
            metrics_calculator=metrics_calc,
            device=device
        )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Classical attacks summary
    print("\nClassical Attacks - Adversarial Accuracy (%):")
    for attack in ['FGSM', 'PGD', 'CW']:
        print(f"\n{attack}:")
        for model_name in model_names:
            acc = classical_results[attack]['adv_acc'][model_name]
            print(f"  {model_name}: {acc:.2f}%")
        print(f"  LPIPS: {np.mean(classical_results[attack]['lpips']):.4f}")
        print(f"  SSIM: {np.mean(classical_results[attack]['ssim']):.4f}")
    
    if diffusion_results:
        print("\nDiffusion Attacks:")
        for model_name in model_names:
            acc = diffusion_results['adv_acc'][model_name]
            print(f"  {model_name}: {acc:.2f}%")
        print(f"  LPIPS: {np.mean(diffusion_results['lpips']):.4f}")
        print(f"  SSIM: {np.mean(diffusion_results['ssim']):.4f}")
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    
    # Save to JSON
    export_data = {
        'classical_attacks': {
            attack: {
                'adv_acc': classical_results[attack]['adv_acc'],
                'lpips_mean': float(np.mean(classical_results[attack]['lpips'])),
                'lpips_std': float(np.std(classical_results[attack]['lpips'])),
                'ssim_mean': float(np.mean(classical_results[attack]['ssim'])),
                'ssim_std': float(np.std(classical_results[attack]['ssim']))
            }
            for attack in ['FGSM', 'PGD', 'CW']
        },
        'clean_acc': classical_results['clean_acc']
    }
    
    if diffusion_results:
        export_data['diffusion_attacks'] = {
            'adv_acc': diffusion_results['adv_acc'],
            'lpips_mean': float(np.mean(diffusion_results['lpips'])),
            'lpips_std': float(np.std(diffusion_results['lpips'])),
            'ssim_mean': float(np.mean(diffusion_results['ssim'])),
            'ssim_std': float(np.std(diffusion_results['ssim']))
        }
    
    with open(output_dir / 'attack_results.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    # Generate visualizations if we have diffusion results
    if diffusion_results:
        print("Generating visualizations...")
        save_all_figures(
            classical_results=classical_results,
            diffusion_results=diffusion_results,
            model_names=model_names,
            output_dir=str(output_dir / 'figures')
        )
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
