#!/usr/bin/env python3
"""
Setup checker for gesture recognition project.
Run this inside your virtual environment to verify setup.
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (not installed)")
        return False

def install_missing_packages():
    """Install missing packages."""
    missing_packages = []
    
    # Core packages
    packages_to_check = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('opencv-python', 'cv2'),
        ('datasets', 'datasets'),
        ('huggingface-hub', 'huggingface_hub'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('pandas', 'pandas'),
        ('PyYAML', 'yaml'),
        ('tensorboard', 'tensorboard'),
        ('numpy', 'numpy'),
        ('Pillow', 'PIL')
    ]
    
    print("Checking packages...")
    for package, import_name in packages_to_check:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All packages are installed!")
        return True

def test_model_creation():
    """Test if we can create a model."""
    try:
        import torch
        from gesture_recognition_model import create_gesture_model
        
        print("\nTesting model creation...")
        model = create_gesture_model('lightweight', num_classes=10)
        print("✓ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 8, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_dataset_structure():
    """Test if dataset configuration works."""
    try:
        import yaml
        
        print("\nTesting configuration loading...")
        with open('test_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
        
        print(f"Dataset: {config['data']['dataset_name']}")
        print(f"Model type: {config['model']['type']}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Clip length: {config['data_preprocessing']['clip_len']}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Gesture Recognition Project Setup Check")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Running in virtual environment")
    else:
        print("⚠ Not in virtual environment (recommended to use venv)")
    
    print("\n" + "=" * 60)
    
    # Check packages
    packages_ok = install_missing_packages()
    
    if packages_ok:
        print("\n" + "=" * 60)
        # Test model
        model_ok = test_model_creation()
        
        print("\n" + "=" * 60)
        # Test config
        config_ok = test_dataset_structure()
        
        if model_ok and config_ok:
            print("\n🎉 Setup is complete! You can now:")
            print("1. Train with: python complete_training_pipeline.py --config test_config.yaml")
            print("2. Test HF dataset: python huggingface_dataset_loader.py")
            print("3. Run imports test: python test_imports.py")
        else:
            print("\n⚠ Some components failed. Check errors above.")
    else:
        print("\n⚠ Install missing packages first, then run this script again.")

if __name__ == "__main__":
    main()