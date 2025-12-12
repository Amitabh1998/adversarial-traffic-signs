"""
Evaluation metrics for adversarial attacks.
Includes LPIPS, SSIM, and transferability analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate imperceptibility and transferability metrics for adversarial attacks.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        lpips_net: str = 'alex',
        imagenet_mean: List[float] = [0.485, 0.456, 0.406],
        imagenet_std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize metrics calculator.
        
        Args:
            device: Computation device
            lpips_net: Network backbone for LPIPS ('alex', 'vgg', 'squeeze')
            imagenet_mean: ImageNet normalization mean
            imagenet_std: ImageNet normalization std
        """
        self.device = device
        self.mean = torch.tensor(imagenet_mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(imagenet_std).view(1, 3, 1, 1).to(device)
        
        # Initialize LPIPS
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
            self.lpips_fn.eval()
            logger.info(f"Initialized LPIPS with {lpips_net} backbone")
        except ImportError:
            logger.warning("LPIPS not available. Install with: pip install lpips")
            self.lpips_fn = None
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor from ImageNet stats to [0, 1]."""
        return tensor * self.std + self.mean
    
    @torch.no_grad()
    def compute_lpips(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> np.ndarray:
        """
        Compute LPIPS perceptual similarity.
        
        Args:
            img1: First batch of images (ImageNet normalized)
            img2: Second batch of images (ImageNet normalized)
        
        Returns:
            Array of LPIPS scores (lower = more similar)
        """
        if self.lpips_fn is None:
            raise RuntimeError("LPIPS not initialized")
        
        # Denormalize to [0, 1] then scale to [-1, 1] for LPIPS
        img1_denorm = self.denormalize(img1)
        img2_denorm = self.denormalize(img2)
        
        img1_lpips = img1_denorm * 2 - 1
        img2_lpips = img2_denorm * 2 - 1
        
        scores = self.lpips_fn(img1_lpips, img2_lpips)
        
        return scores.squeeze().cpu().numpy()
    
    def compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> np.ndarray:
        """
        Compute SSIM structural similarity.
        
        Args:
            img1: First batch of images (ImageNet normalized)
            img2: Second batch of images (ImageNet normalized)
        
        Returns:
            Array of SSIM scores (higher = more similar)
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            raise ImportError("Please install scikit-image: pip install scikit-image")
        
        # Denormalize to [0, 1]
        img1_denorm = self.denormalize(img1).cpu().numpy()
        img2_denorm = self.denormalize(img2).cpu().numpy()
        
        ssim_scores = []
        for i in range(img1_denorm.shape[0]):
            # Convert from CHW to HWC
            img1_np = np.transpose(img1_denorm[i], (1, 2, 0))
            img2_np = np.transpose(img2_denorm[i], (1, 2, 0))
            
            # Clip to valid range
            img1_np = np.clip(img1_np, 0, 1)
            img2_np = np.clip(img2_np, 0, 1)
            
            score = ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)
        
        return np.array(ssim_scores)
    
    def compute_batch_metrics(
        self,
        orig_images: torch.Tensor,
        adv_images: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Compute all imperceptibility metrics for a batch.
        
        Args:
            orig_images: Original images
            adv_images: Adversarial images
        
        Returns:
            Dictionary with 'lpips' and 'ssim' arrays
        """
        metrics = {}
        
        if self.lpips_fn is not None:
            lpips_scores = self.compute_lpips(orig_images, adv_images)
            if np.ndim(lpips_scores) == 0:
                lpips_scores = np.array([lpips_scores])
            metrics['lpips'] = lpips_scores
        
        ssim_scores = self.compute_ssim(orig_images, adv_images)
        metrics['ssim'] = ssim_scores
        
        return metrics


def evaluate_diffusion_attacks(
    models: Dict[str, nn.Module],
    diffusion_loader: DataLoader,
    metrics_calculator: MetricsCalculator,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate diffusion-based attacks with all metrics.
    
    Args:
        models: Dictionary of models to evaluate
        diffusion_loader: DataLoader returning (orig, adv, label, prompt)
        metrics_calculator: MetricsCalculator instance
        device: Computation device
    
    Returns:
        Dictionary with comprehensive results
    """
    results = {
        'lpips': [],
        'ssim': [],
        'preds': {m: [] for m in models},
        'correct': {m: 0 for m in models},
        'clean_correct': {m: 0 for m in models}
    }
    
    all_labels = []
    all_prompts = []
    total = 0
    
    logger.info("Evaluating diffusion attacks...")
    
    for orig_imgs, adv_imgs, labels, prompts in tqdm(diffusion_loader, desc="Diffusion Eval"):
        orig_imgs = orig_imgs.to(device)
        adv_imgs = adv_imgs.to(device)
        labels = labels.to(device)
        
        batch_size = labels.size(0)
        total += batch_size
        all_labels.extend(labels.cpu().numpy())
        all_prompts.extend(prompts)
        
        # Compute imperceptibility metrics
        batch_metrics = metrics_calculator.compute_batch_metrics(orig_imgs, adv_imgs)
        results['lpips'].extend(batch_metrics.get('lpips', []).tolist())
        results['ssim'].extend(batch_metrics['ssim'].tolist())
        
        # Evaluate on all models
        with torch.no_grad():
            for model_name, model in models.items():
                model.eval()
                
                # Clean accuracy
                out_clean = model(orig_imgs)
                if hasattr(out_clean, 'logits'):
                    out_clean = out_clean.logits
                _, pred_clean = out_clean.max(1)
                results['clean_correct'][model_name] += (pred_clean == labels).sum().item()
                
                # Adversarial accuracy
                out_adv = model(adv_imgs)
                if hasattr(out_adv, 'logits'):
                    out_adv = out_adv.logits
                _, pred_adv = out_adv.max(1)
                results['correct'][model_name] += (pred_adv == labels).sum().item()
                results['preds'][model_name].extend(pred_adv.cpu().numpy())
    
    # Compute final metrics
    results['clean_acc'] = {m: 100.0 * results['clean_correct'][m] / total for m in models}
    results['adv_acc'] = {m: 100.0 * results['correct'][m] / total for m in models}
    results['lpips'] = np.array(results['lpips'])
    results['ssim'] = np.array(results['ssim'])
    results['total_samples'] = total
    results['all_labels'] = np.array(all_labels)
    results['all_prompts'] = all_prompts
    results['preds'] = {m: np.array(results['preds'][m]) for m in models}
    
    return results


def evaluate_classical_attacks_with_metrics(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    surrogate_model: nn.Module,
    metrics_calculator: MetricsCalculator,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    attack_params: Optional[Dict] = None
) -> Dict:
    """
    Evaluate classical attacks with imperceptibility metrics.
    
    Args:
        models: Dictionary of models to evaluate
        test_loader: Test data loader
        surrogate_model: Model to generate attacks
        metrics_calculator: MetricsCalculator instance
        device: Computation device
        max_batches: Maximum batches to process
        attack_params: Attack parameters
    
    Returns:
        Dictionary with comprehensive results
    """
    from attacks.classical import ClassicalAttacks
    
    if attack_params is None:
        attack_params = {}
    
    attacker = ClassicalAttacks(surrogate_model, **attack_params)
    
    results = {
        'FGSM': {'lpips': [], 'ssim': [], 'preds': {m: [] for m in models}, 'correct': {m: 0 for m in models}},
        'PGD': {'lpips': [], 'ssim': [], 'preds': {m: [] for m in models}, 'correct': {m: 0 for m in models}},
        'CW': {'lpips': [], 'ssim': [], 'preds': {m: [] for m in models}, 'correct': {m: 0 for m in models}},
    }
    
    clean_correct = {m: 0 for m in models}
    all_labels = []
    total = 0
    
    num_batches = min(max_batches, len(test_loader)) if max_batches else len(test_loader)
    
    logger.info(f"Evaluating classical attacks on {num_batches} batches...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, total=num_batches)):
        if max_batches and batch_idx >= max_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        
        # Clean accuracy
        with torch.no_grad():
            for model_name, model in models.items():
                model.eval()
                out = model(images)
                if hasattr(out, 'logits'):
                    out = out.logits
                _, pred = out.max(1)
                clean_correct[model_name] += (pred == labels).sum().item()
        
        # Evaluate each attack
        for attack_name in ['FGSM', 'PGD', 'CW']:
            adv_images = attacker.generate(attack_name, images, labels)
            
            # Imperceptibility metrics
            batch_metrics = metrics_calculator.compute_batch_metrics(images, adv_images)
            results[attack_name]['lpips'].extend(batch_metrics.get('lpips', []).tolist())
            results[attack_name]['ssim'].extend(batch_metrics['ssim'].tolist())
            
            # Transferability evaluation
            with torch.no_grad():
                for model_name, model in models.items():
                    model.eval()
                    out = model(adv_images)
                    if hasattr(out, 'logits'):
                        out = out.logits
                    _, pred = out.max(1)
                    results[attack_name]['correct'][model_name] += (pred == labels).sum().item()
                    results[attack_name]['preds'][model_name].extend(pred.cpu().numpy())
    
    # Compute final metrics
    results['clean_acc'] = {m: 100.0 * clean_correct[m] / total for m in models}
    
    for attack_name in ['FGSM', 'PGD', 'CW']:
        results[attack_name]['adv_acc'] = {
            m: 100.0 * results[attack_name]['correct'][m] / total for m in models
        }
        results[attack_name]['lpips'] = np.array(results[attack_name]['lpips'])
        results[attack_name]['ssim'] = np.array(results[attack_name]['ssim'])
        results[attack_name]['preds'] = {
            m: np.array(results[attack_name]['preds'][m]) for m in models
        }
    
    results['total_samples'] = total
    results['all_labels'] = np.array(all_labels)
    
    return results


def compute_transferability_metrics(
    preds: Dict[str, np.ndarray],
    labels: np.ndarray,
    model_names: List[str]
) -> Dict:
    """
    Compute transferability metrics from predictions.
    
    Args:
        preds: Dictionary mapping model names to prediction arrays
        labels: True labels
        model_names: List of model names
    
    Returns:
        Dictionary with agreement matrix and all-fooled rate
    """
    n = len(model_names)
    
    # Agreement matrix
    agreement_matrix = np.zeros((n, n))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            agreement_matrix[i, j] = np.mean(preds[m1] == preds[m2]) * 100
    
    # All fooled rate
    all_wrong = np.ones(len(labels), dtype=bool)
    for m in model_names:
        all_wrong &= (preds[m] != labels)
    all_fooled_rate = np.mean(all_wrong) * 100
    
    return {
        'agreement_matrix': agreement_matrix,
        'all_fooled_rate': all_fooled_rate
    }


def summarize_results(
    classical_results: Dict,
    diffusion_results: Dict,
    model_names: List[str]
) -> Dict:
    """
    Create summary comparison of all attacks.
    
    Args:
        classical_results: Results from classical attacks
        diffusion_results: Results from diffusion attacks
        model_names: List of model names
    
    Returns:
        Summary dictionary
    """
    summary = []
    
    for attack in ['FGSM', 'PGD', 'CW']:
        avg_asr = np.mean([100 - classical_results[attack]['adv_acc'][m] for m in model_names])
        
        trans_metrics = compute_transferability_metrics(
            classical_results[attack]['preds'],
            classical_results['all_labels'],
            model_names
        )
        
        summary.append({
            'attack': attack,
            'avg_asr': avg_asr,
            'lpips_mean': np.mean(classical_results[attack]['lpips']),
            'lpips_std': np.std(classical_results[attack]['lpips']),
            'ssim_mean': np.mean(classical_results[attack]['ssim']),
            'ssim_std': np.std(classical_results[attack]['ssim']),
            'all_fooled_rate': trans_metrics['all_fooled_rate']
        })
    
    # Diffusion
    avg_asr_diff = np.mean([100 - diffusion_results['adv_acc'][m] for m in model_names])
    trans_metrics_diff = compute_transferability_metrics(
        diffusion_results['preds'],
        diffusion_results['all_labels'],
        model_names
    )
    
    summary.append({
        'attack': 'Diffusion',
        'avg_asr': avg_asr_diff,
        'lpips_mean': np.mean(diffusion_results['lpips']),
        'lpips_std': np.std(diffusion_results['lpips']),
        'ssim_mean': np.mean(diffusion_results['ssim']),
        'ssim_std': np.std(diffusion_results['ssim']),
        'all_fooled_rate': trans_metrics_diff['all_fooled_rate']
    })
    
    return summary
