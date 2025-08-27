#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import torchvision
    print(f"✓ TorchVision {torchvision.__version__}")
except ImportError as e:
    print(f"✗ TorchVision: {e}")

try:
    from rep_flow_layer import RepFlowLayer, AdaptiveRepFlowLayer
    print("✓ RepFlowLayer imports")
except ImportError as e:
    print(f"✗ RepFlowLayer: {e}")

try:
    from gesture_recognition_model import GestureRecognitionModel, create_gesture_model
    print("✓ GestureRecognitionModel imports")
except ImportError as e:
    print(f"✗ GestureRecognitionModel: {e}")

try:
    from jester_dataset_loader import JesterDataset, create_jester_dataloaders
    print("✓ JesterDataset imports")
except ImportError as e:
    print(f"✗ JesterDataset: {e}")

try:
    from fine_tuning_trainer import GestureFinetuner
    print("✓ GestureFinetuner imports")
except ImportError as e:
    print(f"✗ GestureFinetuner: {e}")

try:
    from evaluation_metrics import GestureEvaluator
    print("✓ GestureEvaluator imports")
except ImportError as e:
    print(f"✗ GestureEvaluator: {e}")

try:
    from complete_training_pipeline import load_config, create_config_template
    print("✓ TrainingPipeline imports")
except ImportError as e:
    print(f"✗ TrainingPipeline: {e}")

print("\nTesting model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_gesture_model('lightweight', num_classes=10)
    print("✓ Model created successfully")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 8, 64, 64)  # Small input for testing
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Model creation/forward pass: {e}")

print("\nAll tests completed!")