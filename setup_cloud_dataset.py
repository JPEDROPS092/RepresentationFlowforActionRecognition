#!/usr/bin/env python3
"""
Quick setup script to download and configure cloud gesture datasets.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml

def setup_huggingface_dataset(dataset_name: str, config_path: str):
    """Setup HuggingFace dataset configuration."""
    print(f"Setting up HuggingFace dataset: {dataset_name}")
    
    # Install required packages
    try:
        import datasets
    except ImportError:
        print("Installing datasets package...")
        os.system("pip install datasets")
    
    # Update config for HuggingFace dataset
    config = {
        'data': {
            'dataset_type': 'huggingface',
            'dataset_name': dataset_name,
            'data_path': f'./hf_cache/{dataset_name}',
            'class_names': []  # Will be auto-detected
        },
        'model': {
            'type': 'lightweight',  # Start with lightweight for faster testing
            'num_classes': 'auto',  # Will be auto-detected
            'flow_channels': 32,
            'use_adaptive_flow': True,
            'freeze_backbone': True,
            'dropout_rate': 0.3
        },
        'training': {
            'batch_size': 4,  # Small batch for testing
            'num_epochs': 10,
            'learning_rates': {
                'flow_layers': 1e-3,
                'attention_layers': 5e-4,
                'classifier': 1e-3,
                'backbone': 1e-5
            },
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'early_stopping_patience': 5,
            'save_frequency': 2
        },
        'data_preprocessing': {
            'clip_len': 8,  # Shorter clips for faster processing
            'crop_size': 64,  # Smaller crops for testing
            'temporal_stride': 1,
            'normalize': True,
            'num_workers': 2,
            'max_train_samples': 1000,  # Limit for testing
            'max_val_samples': 200
        },
        'paths': {
            'checkpoint_dir': f'./checkpoints_{dataset_name.replace("/", "_")}',
            'log_dir': f'./logs_{dataset_name.replace("/", "_")}',
            'evaluation_dir': f'./evaluation_{dataset_name.replace("/", "_")}'
        },
        'evaluation': {
            'metrics_to_compute': ['accuracy', 'precision', 'recall', 'f1'],
            'save_predictions': True,
            'benchmark_inference': True,
            'realtime_evaluation': False
        }
    }
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config

def setup_manual_dataset(dataset_name: str, config_path: str):
    """Setup manual dataset configuration."""
    from cloud_dataset_downloader import CloudDatasetDownloader
    
    downloader = CloudDatasetDownloader()
    
    if dataset_name == "jester":
        paths = downloader.download_jester("small")
        num_classes = 27
    elif dataset_name == "egogesture":
        paths = downloader.download_ego_gesture()
        num_classes = 19
    elif dataset_name == "chalearn":
        paths = downloader.download_chalearn_isogd()
        num_classes = 20
    elif dataset_name == "something_something":
        paths = downloader.download_something_something_v2()
        num_classes = 20  # Subset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = {
        'data': {
            'dataset_type': 'manual',
            'data_path': paths['data_path'],
            'train_annotation': paths['train_annotation'],
            'val_annotation': paths['val_annotation'],
            'test_annotation': paths.get('test_annotation'),
            'class_names': []  # Would need to be filled from actual data
        },
        'model': {
            'type': 'lightweight',
            'num_classes': num_classes,
            'flow_channels': 32,
            'use_adaptive_flow': True,
            'freeze_backbone': True,
            'dropout_rate': 0.3
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 10,
            'learning_rates': {
                'flow_layers': 1e-3,
                'attention_layers': 5e-4,
                'classifier': 1e-3,
                'backbone': 1e-5
            }
        },
        'data_preprocessing': {
            'clip_len': 8,
            'crop_size': 64,
            'temporal_stride': 1,
            'normalize': True,
            'num_workers': 2
        },
        'paths': {
            'checkpoint_dir': f'./checkpoints_{dataset_name}',
            'log_dir': f'./logs_{dataset_name}',
            'evaluation_dir': f'./evaluation_{dataset_name}'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    print(f"Dataset structure created at: {paths['data_path']}")
    print("Note: You'll need to download actual video files from the dataset website")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Setup cloud gesture datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., microsoft/kinetics-400, jester, egogesture)')
    parser.add_argument('--config', type=str, default='dataset_config.yaml',
                       help='Output configuration file')
    parser.add_argument('--type', type=str, choices=['huggingface', 'manual'], 
                       default='huggingface',
                       help='Dataset source type')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gesture Recognition Dataset Setup")
    print("=" * 60)
    
    if args.type == 'huggingface':
        # Recommended HuggingFace datasets
        recommended_hf = [
            "microsoft/kinetics-400",
            "google/something-something-v2", 
            "facebook/egogesture"
        ]
        
        if args.dataset in recommended_hf or '/' in args.dataset:
            config = setup_huggingface_dataset(args.dataset, args.config)
        else:
            print(f"Dataset {args.dataset} might not be available on HuggingFace Hub")
            print("Recommended HuggingFace datasets:")
            for dataset in recommended_hf:
                print(f"  - {dataset}")
            sys.exit(1)
    
    else:  # manual
        manual_datasets = ["jester", "egogesture", "chalearn", "something_something"]
        
        if args.dataset not in manual_datasets:
            print(f"Unknown manual dataset: {args.dataset}")
            print("Available manual datasets:")
            for dataset in manual_datasets:
                print(f"  - {dataset}")
            sys.exit(1)
        
        config = setup_manual_dataset(args.dataset, args.config)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print(f"1. Review the configuration in: {args.config}")
    
    if args.type == 'huggingface':
        print("2. Test the dataset loading:")
        print(f"   python huggingface_dataset_loader.py")
        print("3. Start training:")
        print(f"   python complete_training_pipeline.py --config {args.config}")
    else:
        print("2. Download actual video files from dataset website")
        print("3. Update video paths in the configuration")
        print("4. Start training:")
        print(f"   python complete_training_pipeline.py --config {args.config}")
    
    print("\nFor quick testing with minimal resources:")
    print("- Small batch size (4)")
    print("- Short clips (8 frames)")
    print("- Small crops (64x64)")
    print("- Limited samples (1000 train, 200 val)")


if __name__ == "__main__":
    main()