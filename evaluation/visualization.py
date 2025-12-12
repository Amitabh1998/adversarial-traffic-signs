"""
Visualization utilities for adversarial attack analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_imperceptibility_comparison(
    classical_results: Dict,
    diffusion_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot LPIPS and SSIM comparison across attack types."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    attack_names = ['FGSM', 'PGD', 'CW', 'Diffusion']
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    all_lpips = [
        classical_results['FGSM']['lpips'],
        classical_results['PGD']['lpips'],
        classical_results['CW']['lpips'],
        diffusion_results['lpips']
    ]
    all_ssim = [
        classical_results['FGSM']['ssim'],
        classical_results['PGD']['ssim'],
        classical_results['CW']['ssim'],
        diffusion_results['ssim']
    ]
    
    # LPIPS Box Plot
    bp1 = axes[0, 0].boxplot(all_lpips, labels=attack_names, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_ylabel('LPIPS Score', fontsize=12)
    axes[0, 0].set_title('LPIPS Distribution (Lower = More Similar)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # SSIM Box Plot
    bp2 = axes[0, 1].boxplot(all_ssim, labels=attack_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_ylabel('SSIM Score', fontsize=12)
    axes[0, 1].set_title('SSIM Distribution (Higher = More Similar)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Histograms
    for data, name, color in zip(all_lpips, attack_names, colors):
        axes[1, 0].hist(data, bins=30, alpha=0.5, label=name, color=color)
    axes[1, 0].set_xlabel('LPIPS Score', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('LPIPS Histogram', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    
    for data, name, color in zip(all_ssim, attack_names, colors):
        axes[1, 1].hist(data, bins=30, alpha=0.5, label=name, color=color)
    axes[1, 1].set_xlabel('SSIM Score', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('SSIM Histogram', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    
    plt.suptitle('Imperceptibility Metrics: Classical vs Diffusion Attacks', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_transferability_heatmap(
    classical_results: Dict,
    diffusion_results: Dict,
    model_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot transferability heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    attack_names = ['FGSM', 'PGD', 'CW', 'Diffusion']
    
    adv_acc_matrix = np.array([
        [classical_results['FGSM']['adv_acc'][m] for m in model_names],
        [classical_results['PGD']['adv_acc'][m] for m in model_names],
        [classical_results['CW']['adv_acc'][m] for m in model_names],
        [diffusion_results['adv_acc'][m] for m in model_names]
    ])
    
    asr_matrix = 100 - adv_acc_matrix
    
    # Adversarial Accuracy
    im1 = axes[0].imshow(adv_acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0].set_xticks(np.arange(len(model_names)))
    axes[0].set_yticks(np.arange(len(attack_names)))
    axes[0].set_xticklabels(model_names, rotation=15)
    axes[0].set_yticklabels(attack_names)
    axes[0].set_title('Adversarial Accuracy (%)\nHigher = More Robust', fontsize=14, fontweight='bold')
    
    for i in range(len(attack_names)):
        for j in range(len(model_names)):
            color = "white" if adv_acc_matrix[i, j] < 50 else "black"
            axes[0].text(j, i, f'{adv_acc_matrix[i, j]:.1f}%', ha="center", va="center", 
                        fontsize=12, fontweight='bold', color=color)
    plt.colorbar(im1, ax=axes[0], label='Accuracy (%)')
    
    # Attack Success Rate
    im2 = axes[1].imshow(asr_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=100)
    axes[1].set_xticks(np.arange(len(model_names)))
    axes[1].set_yticks(np.arange(len(attack_names)))
    axes[1].set_xticklabels(model_names, rotation=15)
    axes[1].set_yticklabels(attack_names)
    axes[1].set_title('Attack Success Rate (%)\nHigher = Stronger Attack', fontsize=14, fontweight='bold')
    
    for i in range(len(attack_names)):
        for j in range(len(model_names)):
            color = "white" if asr_matrix[i, j] > 50 else "black"
            axes[1].text(j, i, f'{asr_matrix[i, j]:.1f}%', ha="center", va="center",
                        fontsize=12, fontweight='bold', color=color)
    plt.colorbar(im2, ax=axes[1], label='Success Rate (%)')
    
    plt.suptitle('Attack Transferability Across Model Architectures', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comprehensive_comparison(
    classical_results: Dict,
    diffusion_results: Dict,
    model_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    attack_names = ['FGSM', 'PGD', 'CW', 'Diffusion']
    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    # Attack Success Rate by Model
    for i, (attack, color) in enumerate(zip(attack_names, colors)):
        if attack == 'Diffusion':
            values = [100 - diffusion_results['adv_acc'][m] for m in model_names]
        else:
            values = [100 - classical_results[attack]['adv_acc'][m] for m in model_names]
        axes[0].bar(x + i*width, values, width, label=attack, color=color, alpha=0.8)
    
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Attack Success Rate (%)', fontsize=12)
    axes[0].set_title('Attack Success Rate by Model', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].set_ylim(0, 100)
    
    # LPIPS
    lpips_means = [np.mean(classical_results[a]['lpips']) for a in ['FGSM', 'PGD', 'CW']] + [np.mean(diffusion_results['lpips'])]
    lpips_stds = [np.std(classical_results[a]['lpips']) for a in ['FGSM', 'PGD', 'CW']] + [np.std(diffusion_results['lpips'])]
    bars = axes[1].bar(attack_names, lpips_means, yerr=lpips_stds, color=colors, alpha=0.8, capsize=5)
    axes[1].set_ylabel('LPIPS Score', fontsize=12)
    axes[1].set_title('LPIPS (Lower = More Imperceptible)', fontsize=14, fontweight='bold')
    
    # SSIM
    ssim_means = [np.mean(classical_results[a]['ssim']) for a in ['FGSM', 'PGD', 'CW']] + [np.mean(diffusion_results['ssim'])]
    ssim_stds = [np.std(classical_results[a]['ssim']) for a in ['FGSM', 'PGD', 'CW']] + [np.std(diffusion_results['ssim'])]
    bars = axes[2].bar(attack_names, ssim_means, yerr=ssim_stds, color=colors, alpha=0.8, capsize=5)
    axes[2].set_ylabel('SSIM Score', fontsize=12)
    axes[2].set_title('SSIM (Higher = More Imperceptible)', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    
    plt.suptitle('Comprehensive Attack Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_all_figures(
    classical_results: Dict,
    diffusion_results: Dict,
    model_names: List[str],
    output_dir: str
) -> None:
    """Generate and save all visualization figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_imperceptibility_comparison(
        classical_results, diffusion_results,
        save_path=str(output_dir / 'imperceptibility_comparison.png')
    )
    plt.close()
    
    plot_transferability_heatmap(
        classical_results, diffusion_results, model_names,
        save_path=str(output_dir / 'transferability_heatmap.png')
    )
    plt.close()
    
    plot_comprehensive_comparison(
        classical_results, diffusion_results, model_names,
        save_path=str(output_dir / 'comprehensive_comparison.png')
    )
    plt.close()
    
    logger.info(f"All figures saved to {output_dir}")
