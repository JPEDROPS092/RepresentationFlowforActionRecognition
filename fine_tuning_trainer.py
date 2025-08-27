import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime
from tqdm import tqdm

from gesture_recognition_model import GestureRecognitionModel, create_gesture_model
from gesture_dataset import create_gesture_dataloaders


class GestureFinetuner:
    """
    Fine-tuning trainer for gesture recognition models with Representation Flow.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_classes: int,
                 device: torch.device,
                 save_dir: str = "./checkpoints",
                 log_dir: str = "./logs"):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging setup
        self.setup_logging()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizers with different learning rates for different components
        self.setup_optimizers()
        
        # Learning rate schedulers
        self.setup_schedulers()
        
        # Model complexity analysis
        self.analyze_model()
    
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizers(self):
        """Setup optimizers with different learning rates for different components."""
        # Separate parameters based on component
        flow_params = []
        attention_params = []
        classifier_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'flow_layer' in name:
                flow_params.append(param)
            elif 'attention' in name:
                attention_params.append(param)
            elif 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Create optimizer with parameter groups
        self.optimizer = optim.AdamW([
            {'params': flow_params, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': attention_params, 'lr': 5e-4, 'weight_decay': 1e-4},
            {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 1e-3},
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4}
        ])
        
        self.logger.info(f"Optimizer setup:")
        self.logger.info(f"  Flow parameters: {len(flow_params)} (lr=1e-3)")
        self.logger.info(f"  Attention parameters: {len(attention_params)} (lr=5e-4)")
        self.logger.info(f"  Classifier parameters: {len(classifier_params)} (lr=1e-3)")
        self.logger.info(f"  Backbone parameters: {len(backbone_params)} (lr=1e-5)")
    
    def setup_schedulers(self):
        """Setup learning rate schedulers."""
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,  # Initial restart period
            T_mult=2,  # Restart period multiplier
            eta_min=1e-6
        )
        
        # Plateau scheduler as backup
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def analyze_model(self):
        """Analyze model complexity."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Analysis:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
        self.logger.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (videos, labels) in enumerate(progress_bar):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # Log batch metrics
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
            if batch_idx % 50 == 0:
                # Log learning rates
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'LR/Group_{i}', param_group['lr'], global_step)
        
        # Update scheduler
        self.scheduler.step()
        
        avg_loss = running_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> Tuple[float, float, Dict[str, Any]]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for videos, labels in progress_bar:
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                acc = 100 * correct / total
                progress_bar.set_postfix({'Acc': f'{acc:.2f}%'})
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        # Calculate detailed metrics
        detailed_metrics = self.calculate_detailed_metrics(all_predictions, all_labels)
        detailed_metrics['accuracy'] = accuracy
        detailed_metrics['loss'] = avg_loss
        
        self.val_accuracies.append(accuracy)
        return avg_loss, accuracy, detailed_metrics
    
    def calculate_detailed_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, Any]:
        """Calculate detailed metrics including per-class accuracy."""
        from collections import defaultdict
        
        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for pred, label in zip(predictions, labels):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
        
        per_class_acc = {}
        for class_id in range(self.num_classes):
            if class_total[class_id] > 0:
                per_class_acc[class_id] = 100 * class_correct[class_id] / class_total[class_id]
            else:
                per_class_acc[class_id] = 0.0
        
        # Overall metrics
        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        
        return {
            'per_class_accuracy': per_class_acc,
            'mean_class_accuracy': np.mean(list(per_class_acc.values())),
            'total_samples': total_samples,
            'total_correct': total_correct
        }
    
    def save_checkpoint(self, 
                       is_best: bool = False, 
                       extra_info: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'model_config': {
                'num_classes': self.num_classes,
                'model_type': type(self.model).__name__
            }
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # Save latest checkpoint
        latest_path = self.save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with accuracy: {self.best_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        
        self.logger.info(f"Checkpoint loaded from epoch {self.epoch}")
    
    def train(self, 
              num_epochs: int, 
              save_frequency: int = 5,
              early_stopping_patience: int = 15):
        """Main training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_acc, detailed_metrics = self.validate()
            
            # Update plateau scheduler
            self.plateau_scheduler.step(val_acc)
            
            # Check for best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {self.epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            self.writer.add_scalar('Train/Loss', train_loss, self.epoch)
            self.writer.add_scalar('Val/Loss', val_loss, self.epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, self.epoch)
            self.writer.add_scalar('Val/MeanClassAccuracy', 
                                 detailed_metrics['mean_class_accuracy'], self.epoch)
            
            # Save checkpoint
            if self.epoch % save_frequency == 0 or is_best:
                self.save_checkpoint(is_best, {'detailed_metrics': detailed_metrics})
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.epoch} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        # Save final metrics
        self.save_training_summary()
        self.writer.close()
    
    def save_training_summary(self):
        """Save training summary to JSON."""
        summary = {
            'training_completed': datetime.now().isoformat(),
            'total_epochs': self.epoch,
            'best_accuracy': self.best_acc,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
            'model_config': {
                'num_classes': self.num_classes,
                'model_type': type(self.model).__name__,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Example usage of the fine-tuning trainer."""
    # Configuration
    config = {
        'data_path': '/path/to/gesture/videos',
        'train_annotation': '/path/to/train_annotations.json',
        'val_annotation': '/path/to/val_annotations.json',
        'num_classes': 20,
        'batch_size': 8,
        'num_epochs': 50,
        'clip_len': 16,
        'crop_size': 112,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Create data loaders
    train_loader, val_loader = create_gesture_dataloaders(
        data_path=config['data_path'],
        train_annotation=config['train_annotation'],
        val_annotation=config['val_annotation'],
        batch_size=config['batch_size'],
        clip_len=config['clip_len'],
        crop_size=config['crop_size']
    )
    
    # Create model
    model = create_gesture_model(
        model_type='full',
        num_classes=config['num_classes'],
        use_adaptive_flow=True,
        freeze_backbone=True
    )
    
    # Create trainer
    trainer = GestureFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['num_classes'],
        device=torch.device(config['device']),
        save_dir='./gesture_checkpoints',
        log_dir='./gesture_logs'
    )
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        save_frequency=5,
        early_stopping_patience=15
    )


if __name__ == "__main__":
    main()