#!/usr/bin/env python3
"""
Complete training pipeline for gesture recognition with Representation Flow.
This script demonstrates the entire workflow from data loading to evaluation.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

from gesture_recognition_model import create_gesture_model
from gesture_dataset import create_gesture_dataloaders
from fine_tuning_trainer import GestureFinetuner
from evaluation_metrics import GestureEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def create_config_template(save_path: str):
    """Create a configuration template file."""
    config = {
        'data': {
            'data_path': '/path/to/gesture/videos',
            'train_annotation': '/path/to/train_annotations.json',
            'val_annotation': '/path/to/val_annotations.json',
            'test_annotation': '/path/to/test_annotations.json',
            'class_names': [
                'wave', 'point', 'thumbs_up', 'thumbs_down', 'peace',
                'fist', 'open_palm', 'swipe_left', 'swipe_right', 'swipe_up',
                'swipe_down', 'pinch', 'spread', 'rotate_cw', 'rotate_ccw',
                'tap', 'double_tap', 'hold', 'zoom_in', 'zoom_out'
            ]
        },
        'model': {
            'type': 'full',  # 'full' or 'lightweight'
            'num_classes': 20,
            'flow_channels': 64,
            'use_adaptive_flow': True,
            'freeze_backbone': True,
            'dropout_rate': 0.5
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 50,
            'learning_rates': {
                'flow_layers': 1e-3,
                'attention_layers': 5e-4,
                'classifier': 1e-3,
                'backbone': 1e-5
            },
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'early_stopping_patience': 15,
            'save_frequency': 5
        },
        'data_preprocessing': {
            'clip_len': 16,
            'crop_size': 112,
            'temporal_stride': 1,
            'normalize': True,
            'num_workers': 4
        },
        'paths': {
            'checkpoint_dir': './gesture_checkpoints',
            'log_dir': './gesture_logs',
            'evaluation_dir': './evaluation_results'
        },
        'evaluation': {
            'metrics_to_compute': ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'],
            'save_predictions': True,
            'benchmark_inference': True,
            'realtime_evaluation': False,
            'realtime_duration': 60
        }
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration template saved to: {save_path}")


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('GestureTrainingPipeline')
    return logger


def train_model(config: dict, logger: logging.Logger):
    """Train the gesture recognition model."""
    logger.info("Starting model training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_gesture_dataloaders(
        data_path=config['data']['data_path'],
        train_annotation=config['data']['train_annotation'],
        val_annotation=config['data']['val_annotation'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data_preprocessing']['num_workers'],
        clip_len=config['data_preprocessing']['clip_len'],
        crop_size=config['data_preprocessing']['crop_size']
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_gesture_model(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        flow_channels=config['model']['flow_channels'],
        use_adaptive_flow=config['model']['use_adaptive_flow'],
        freeze_backbone=config['model']['freeze_backbone'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Create trainer
    trainer = GestureFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['model']['num_classes'],
        device=device,
        save_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir']
    )
    
    # Start training
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_frequency=config['training']['save_frequency'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    logger.info("Training completed!")
    return trainer.save_dir / "best_checkpoint.pth"


def evaluate_model(config: dict, checkpoint_path: str, logger: logging.Logger):
    """Evaluate the trained model."""
    logger.info("Starting model evaluation...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_gesture_model(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        flow_channels=config['model']['flow_channels'],
        use_adaptive_flow=config['model']['use_adaptive_flow'],
        freeze_backbone=False,  # Unfreeze for evaluation
        dropout_rate=0.0  # Disable dropout for evaluation
    )
    
    # Load trained weights
    if Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Proceeding with randomly initialized model for demonstration")
    
    # Create evaluator
    evaluator = GestureEvaluator(
        model=model,
        device=device,
        class_names=config['data']['class_names']
    )
    
    # Test set evaluation
    if 'test_annotation' in config['data']:
        from gesture_dataset import GestureVideoDataset
        from torch.utils.data import DataLoader
        
        test_dataset = GestureVideoDataset(
            data_path=config['data']['data_path'],
            annotation_file=config['data']['test_annotation'],
            clip_len=config['data_preprocessing']['clip_len'],
            crop_size=config['data_preprocessing']['crop_size'],
            is_training=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data_preprocessing']['num_workers'],
            pin_memory=True
        )
        
        logger.info("Evaluating on test set...")
        test_results = evaluator.evaluate_dataset(
            test_loader,
            save_dir=config['paths']['evaluation_dir']
        )
        
        logger.info(f"Test Accuracy: {test_results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {test_results['overall_metrics']['macro_f1']:.4f}")
    
    # Benchmark inference speed
    if config['evaluation']['benchmark_inference']:
        logger.info("Benchmarking inference speed...")
        
        shapes = [
            (1, 3, config['data_preprocessing']['clip_len'], 
             config['data_preprocessing']['crop_size'], 
             config['data_preprocessing']['crop_size']),
            (config['training']['batch_size'], 3, 
             config['data_preprocessing']['clip_len'],
             config['data_preprocessing']['crop_size'], 
             config['data_preprocessing']['crop_size'])
        ]
        
        benchmark_results = evaluator.benchmark_inference_speed(shapes, num_runs=100)
        
        for shape_key, metrics in benchmark_results.items():
            logger.info(f"Shape {shape_key}: {metrics['mean_time']*1000:.2f}ms, {metrics['fps']:.1f} FPS")
    
    # Real-time evaluation
    if config['evaluation']['realtime_evaluation']:
        logger.info("Starting real-time evaluation...")
        try:
            realtime_results = evaluator.evaluate_real_time(
                duration_seconds=config['evaluation']['realtime_duration'],
                save_dir=config['paths']['evaluation_dir']
            )
            logger.info(f"Real-time FPS: {realtime_results['frame_rate_metrics']['mean_fps']:.1f}")
        except Exception as e:
            logger.warning(f"Real-time evaluation failed: {e}")
    
    logger.info("Evaluation completed!")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Gesture Recognition Training Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--create-config', type=str,
                       help='Create configuration template at specified path')
    parser.add_argument('--train-only', action='store_true',
                       help='Only perform training')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only perform evaluation')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Create config template if requested
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['paths']['log_dir'])
    logger.info("Starting Gesture Recognition Training Pipeline")
    logger.info(f"Configuration loaded from: {args.config}")
    
    # Print configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Model: {config['model']['type']}")
    logger.info(f"  Classes: {config['model']['num_classes']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Clip length: {config['data_preprocessing']['clip_len']}")
    
    checkpoint_path = None
    
    # Training
    if not args.eval_only:
        try:
            checkpoint_path = train_model(config, logger)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    # Evaluation
    if not args.train_only:
        eval_checkpoint = args.checkpoint or checkpoint_path
        if eval_checkpoint:
            try:
                evaluate_model(config, eval_checkpoint, logger)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                raise
        else:
            logger.warning("No checkpoint specified for evaluation")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()