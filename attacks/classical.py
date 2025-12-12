"""
Classical adversarial attack implementations.
Supports: FGSM, PGD, C&W attacks using torchattacks library.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

try:
    import torchattacks
except ImportError:
    raise ImportError("Please install torchattacks: pip install torchattacks")

logger = logging.getLogger(__name__)


class ClassicalAttacks:
    """
    Wrapper class for classical adversarial attacks.
    
    Provides a unified interface for FGSM, PGD, and C&W attacks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        fgsm_eps: float = 0.03,
        pgd_eps: float = 0.03,
        pgd_alpha: float = 0.01,
        pgd_steps: int = 10,
        cw_c: float = 1,
        cw_kappa: float = 0,
        cw_steps: int = 100,
        cw_lr: float = 0.01
    ):
        """
        Initialize attack methods.
        
        Args:
            model: Target model for attacks
            fgsm_eps: FGSM perturbation bound
            pgd_eps: PGD perturbation bound
            pgd_alpha: PGD step size
            pgd_steps: Number of PGD iterations
            cw_c: C&W confidence parameter
            cw_kappa: C&W margin parameter
            cw_steps: Number of C&W optimization steps
            cw_lr: C&W learning rate
        """
        self.model = model
        
        # Initialize attacks
        self.attacks = {
            'FGSM': torchattacks.FGSM(model, eps=fgsm_eps),
            'PGD': torchattacks.PGD(
                model, eps=pgd_eps, alpha=pgd_alpha,
                steps=pgd_steps, random_start=True
            ),
            'CW': torchattacks.CW(
                model, c=cw_c, kappa=cw_kappa,
                steps=cw_steps, lr=cw_lr
            )
        }
        
        # Store parameters for reference
        self.params = {
            'FGSM': {'eps': fgsm_eps},
            'PGD': {'eps': pgd_eps, 'alpha': pgd_alpha, 'steps': pgd_steps},
            'CW': {'c': cw_c, 'kappa': cw_kappa, 'steps': cw_steps, 'lr': cw_lr}
        }
    
    def generate(
        self,
        attack_name: str,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Args:
            attack_name: Name of attack ('FGSM', 'PGD', 'CW')
            images: Input images
            labels: True labels
        
        Returns:
            Adversarial images
        """
        if attack_name not in self.attacks:
            raise ValueError(f"Unknown attack: {attack_name}. Available: {list(self.attacks.keys())}")
        
        return self.attacks[attack_name](images, labels)
    
    def evaluate_single_attack(
        self,
        attack_name: str,
        loader: DataLoader,
        device: str = 'cuda',
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single attack on a dataset.
        
        Args:
            attack_name: Name of attack
            loader: Data loader
            device: Device for computation
            max_batches: Maximum batches to process (None for all)
        
        Returns:
            Dictionary with clean and adversarial accuracy
        """
        self.model.eval()
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        num_batches = min(max_batches, len(loader)) if max_batches else len(loader)
        
        for i, (images, labels) in enumerate(tqdm(loader, total=num_batches, desc=f"{attack_name}")):
            if max_batches and i >= max_batches:
                break
            
            images, labels = images.to(device), labels.to(device)
            adv_images = self.generate(attack_name, images, labels)
            
            with torch.no_grad():
                # Clean predictions
                outputs_clean = self.model(images)
                if hasattr(outputs_clean, 'logits'):
                    outputs_clean = outputs_clean.logits
                _, pred_clean = torch.max(outputs_clean, 1)
                
                # Adversarial predictions
                outputs_adv = self.model(adv_images)
                if hasattr(outputs_adv, 'logits'):
                    outputs_adv = outputs_adv.logits
                _, pred_adv = torch.max(outputs_adv, 1)
            
            correct_clean += (pred_clean == labels).sum().item()
            correct_adv += (pred_adv == labels).sum().item()
            total += labels.size(0)
        
        return {
            'clean_acc': 100 * correct_clean / total,
            'adv_acc': 100 * correct_adv / total,
            'attack_success_rate': 100 * (1 - correct_adv / total)
        }


def evaluate_classical_attacks(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    surrogate_model: nn.Module,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    attack_params: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Evaluate all classical attacks on multiple models.
    
    Args:
        models: Dictionary of models to evaluate
        test_loader: Test data loader
        surrogate_model: Model used to generate attacks
        device: Computation device
        max_batches: Maximum batches to process
        attack_params: Attack parameters dictionary
    
    Returns:
        Dictionary with results for each attack and model
    """
    if attack_params is None:
        attack_params = {}
    
    # Initialize attacks on surrogate model
    attacker = ClassicalAttacks(surrogate_model, **attack_params)
    
    results = {
        'FGSM': {'preds': {m: [] for m in models}, 'correct': {m: 0 for m in models}},
        'PGD': {'preds': {m: [] for m in models}, 'correct': {m: 0 for m in models}},
        'CW': {'preds': {m: [] for m in models}, 'correct': {m: 0 for m in models}},
    }
    
    clean_correct = {m: 0 for m in models}
    all_labels = []
    total = 0
    
    num_batches = min(max_batches, len(test_loader)) if max_batches else len(test_loader)
    
    print(f"Evaluating classical attacks on {num_batches} batches...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, total=num_batches)):
        if max_batches and batch_idx >= max_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        total += batch_size
        all_labels.extend(labels.cpu().numpy())
        
        # Evaluate clean accuracy on all models
        with torch.no_grad():
            for model_name, model in models.items():
                model.eval()
                outputs = model(images)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                _, pred = outputs.max(1)
                clean_correct[model_name] += (pred == labels).sum().item()
        
        # Generate and evaluate each attack
        for attack_name in ['FGSM', 'PGD', 'CW']:
            adv_images = attacker.generate(attack_name, images, labels)
            
            with torch.no_grad():
                for model_name, model in models.items():
                    model.eval()
                    outputs = model(adv_images)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    _, pred = outputs.max(1)
                    results[attack_name]['correct'][model_name] += (pred == labels).sum().item()
                    results[attack_name]['preds'][model_name].extend(pred.cpu().numpy())
    
    # Compute accuracies
    clean_acc = {m: 100.0 * clean_correct[m] / total for m in models}
    
    for attack_name in results:
        results[attack_name]['adv_acc'] = {
            m: 100.0 * results[attack_name]['correct'][m] / total for m in models
        }
        results[attack_name]['preds'] = {
            m: np.array(results[attack_name]['preds'][m]) for m in models
        }
    
    results['clean_acc'] = clean_acc
    results['total_samples'] = total
    results['all_labels'] = np.array(all_labels)
    
    return results


def run_attack_comparison(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda',
    max_batches: Optional[int] = None,
    **attack_params
) -> Dict[str, Dict[str, float]]:
    """
    Quick comparison of all attacks on a single model.
    
    Args:
        model: Model to attack
        loader: Data loader
        device: Computation device
        max_batches: Maximum batches to process
        **attack_params: Attack parameters
    
    Returns:
        Dictionary with metrics for each attack
    """
    attacker = ClassicalAttacks(model, **attack_params)
    
    results = {}
    for attack_name in ['FGSM', 'PGD', 'CW']:
        print(f"\n[{attack_name}]")
        results[attack_name] = attacker.evaluate_single_attack(
            attack_name, loader, device, max_batches
        )
        print(f"  Clean Acc: {results[attack_name]['clean_acc']:.2f}%")
        print(f"  Adv Acc: {results[attack_name]['adv_acc']:.2f}%")
    
    return results
