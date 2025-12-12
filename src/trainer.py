"""
Training utilities for traffic sign recognition models.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer class for traffic sign recognition models.
    
    Supports training with early stopping, learning rate scheduling,
    and comprehensive metric logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 10,
        checkpoint_dir: Optional[Path] = None,
        early_stopping: bool = True,
        patience: int = 5
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            model_name: Name for saving checkpoints
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Maximum number of epochs
            checkpoint_dir: Directory to save checkpoints
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir or Path('results/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle ViT output format
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
        
        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Tuple of (trained model, training history)
        """
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"Device: {self.device} | Epochs: {self.num_epochs}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                checkpoint_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.early_stopping and self.patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.model, self.history
    
    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: str = 'cuda',
        description: str = "Evaluation"
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            loader: Data loader
            device: Evaluation device
            description: Description for progress bar
        
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_labels = []
        all_preds = []
        
        for images, labels in tqdm(loader, desc=description, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return metrics


def train_all_models(
    model_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cuda',
    checkpoint_dir: Optional[Path] = None,
    **training_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Train all specified models.
    
    Args:
        model_names: List of model architectures to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Training device
        checkpoint_dir: Directory for checkpoints
        **training_kwargs: Additional training parameters
    
    Returns:
        Dictionary with results for each model
    """
    from models.factory import get_model, load_model
    
    results = {}
    
    for model_name in model_names:
        try:
            print(f"\n{'='*60}")
            print(f"Loading {model_name}...")
            print(f"{'='*60}")
            
            model = get_model(model_name, device=device)
            
            trainer = ModelTrainer(
                model=model,
                model_name=model_name,
                device=device,
                checkpoint_dir=checkpoint_dir,
                **training_kwargs
            )
            
            trained_model, history = trainer.train(train_loader, val_loader)
            
            # Load best checkpoint for evaluation
            checkpoint_path = checkpoint_dir / f'{model_name}_best.pth'
            best_model = load_model(model_name, str(checkpoint_path), device=device)
            
            # Evaluate on all splits
            print("\nEvaluating best checkpoint...")
            train_metrics = ModelTrainer.evaluate(best_model, train_loader, device, "Train Eval")
            val_metrics = ModelTrainer.evaluate(best_model, val_loader, device, "Val Eval")
            test_metrics = ModelTrainer.evaluate(best_model, test_loader, device, "Test Eval")
            
            results[model_name] = {
                'history': history,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_epoch': trainer.best_epoch
            }
            
            print(f"\n{model_name} Results:")
            print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Test Acc: {test_metrics['accuracy']:.2f}%")
            
            # Clean up
            del model, trained_model, best_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results
