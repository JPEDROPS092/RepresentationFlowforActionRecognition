#!/usr/bin/env python3
"""
Create synthetic gesture dataset for testing the pipeline.
"""

import os
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import yaml


def create_synthetic_gesture_video(output_path: str, 
                                 frames: int = 16, 
                                 width: int = 112, 
                                 height: int = 112,
                                 gesture_type: str = "wave"):
    """Create a synthetic gesture video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    for frame_idx in range(frames):
        # Create base frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark background
        
        # Add some random noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Simulate gesture motion based on type
        t = frame_idx / frames  # Normalized time [0, 1]
        
        if gesture_type == "wave":
            # Horizontal wave motion
            center_x = int(width * 0.5 + 0.2 * width * np.sin(t * 4 * np.pi))
            center_y = int(height * 0.5)
            
        elif gesture_type == "swipe_right":
            # Right swipe motion
            center_x = int(width * 0.2 + 0.6 * width * t)
            center_y = int(height * 0.5)
            
        elif gesture_type == "swipe_left":
            # Left swipe motion
            center_x = int(width * 0.8 - 0.6 * width * t)
            center_y = int(height * 0.5)
            
        elif gesture_type == "zoom_in":
            # Zoom in motion
            radius = int(10 + 30 * t)
            center_x = int(width * 0.5)
            center_y = int(height * 0.5)
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
            out.write(frame)
            continue
            
        elif gesture_type == "zoom_out":
            # Zoom out motion
            radius = int(40 - 30 * t)
            center_x = int(width * 0.5)
            center_y = int(height * 0.5)
            if radius > 0:
                cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
            out.write(frame)
            continue
            
        elif gesture_type == "rotate":
            # Rotation motion
            angle = t * 360
            center_x = int(width * 0.5)
            center_y = int(height * 0.5)
            
            # Draw rotating line
            end_x = int(center_x + 30 * np.cos(np.radians(angle)))
            end_y = int(center_y + 30 * np.sin(np.radians(angle)))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (255, 255, 255), 3)
            out.write(frame)
            continue
            
        else:  # static gesture like "thumbs_up"
            center_x = int(width * 0.5)
            center_y = int(height * 0.5)
        
        # Draw hand-like object (circle with smaller circle for thumb)
        cv2.circle(frame, (center_x, center_y), 20, (200, 150, 100), -1)  # Hand
        if gesture_type == "thumbs_up":
            cv2.circle(frame, (center_x - 15, center_y - 25), 8, (200, 150, 100), -1)  # Thumb
        elif gesture_type == "thumbs_down":
            cv2.circle(frame, (center_x - 15, center_y + 25), 8, (200, 150, 100), -1)  # Thumb
        
        out.write(frame)
    
    out.release()


def create_synthetic_dataset(output_dir: str = "./synthetic_gestures", 
                           num_classes: int = 10,
                           samples_per_class: int = 50,
                           val_split: float = 0.2):
    """Create a complete synthetic gesture dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define gesture classes
    gesture_classes = [
        "wave", "swipe_right", "swipe_left", "swipe_up", "swipe_down",
        "zoom_in", "zoom_out", "rotate", "thumbs_up", "thumbs_down",
        "point", "grab", "release", "tap", "double_tap"
    ][:num_classes]
    
    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    train_annotations = []
    val_annotations = []
    
    print(f"Creating synthetic dataset with {num_classes} classes...")
    
    for class_idx, gesture_class in enumerate(tqdm(gesture_classes, desc="Classes")):
        class_dir = videos_dir / gesture_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_idx in tqdm(range(samples_per_class), desc=f"Creating {gesture_class}", leave=False):
            # Create video filename
            video_filename = f"{gesture_class}_{sample_idx:04d}.mp4"
            video_path = class_dir / video_filename
            relative_path = f"{gesture_class}/{video_filename}"
            
            # Create synthetic video
            create_synthetic_gesture_video(
                str(video_path),
                frames=np.random.randint(12, 20),  # Variable length
                gesture_type=gesture_class
            )
            
            # Add to annotations
            annotation = {
                "video_path": relative_path,
                "label": gesture_class,
                "start_frame": 0,
                "end_frame": -1
            }
            
            # Split into train/val
            if sample_idx < int(samples_per_class * (1 - val_split)):
                train_annotations.append(annotation)
            else:
                val_annotations.append(annotation)
    
    # Save annotations
    with open(output_path / "train_annotations.json", "w") as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(output_path / "val_annotations.json", "w") as f:
        json.dump(val_annotations, f, indent=2)
    
    # Save class info
    with open(output_path / "classes.json", "w") as f:
        json.dump({"classes": gesture_classes}, f, indent=2)
    
    # Create dataset config
    config = {
        'data': {
            'dataset_type': 'manual',
            'data_path': str(videos_dir),
            'train_annotation': str(output_path / "train_annotations.json"),
            'val_annotation': str(output_path / "val_annotations.json"),
            'class_names': gesture_classes
        },
        'model': {
            'type': 'lightweight',
            'num_classes': len(gesture_classes),
            'flow_channels': 32,
            'use_adaptive_flow': True,
            'freeze_backbone': False,  # No pre-training benefit for synthetic data
            'dropout_rate': 0.3
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 15,
            'learning_rates': {
                'flow_layers': 1e-3,
                'attention_layers': 5e-4,
                'classifier': 1e-3,
                'backbone': 1e-4
            },
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'early_stopping_patience': 8,
            'save_frequency': 3
        },
        'data_preprocessing': {
            'clip_len': 12,
            'crop_size': 64,
            'temporal_stride': 1,
            'normalize': True,
            'num_workers': 2
        },
        'paths': {
            'checkpoint_dir': './checkpoints_synthetic',
            'log_dir': './logs_synthetic',
            'evaluation_dir': './evaluation_synthetic'
        },
        'evaluation': {
            'metrics_to_compute': ['accuracy', 'precision', 'recall', 'f1'],
            'save_predictions': True,
            'benchmark_inference': True,
            'realtime_evaluation': False
        }
    }
    
    config_path = output_path / "synthetic_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ Synthetic dataset created!")
    print(f"📁 Location: {output_path}")
    print(f"📊 Classes: {len(gesture_classes)}")
    print(f"🎬 Training videos: {len(train_annotations)}")
    print(f"🎬 Validation videos: {len(val_annotations)}")
    print(f"⚙️  Configuration: {config_path}")
    
    print(f"\n🚀 To start training:")
    print(f"python complete_training_pipeline.py --config {config_path}")
    
    return str(config_path)


def main():
    """Create synthetic dataset with different options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create synthetic gesture dataset')
    parser.add_argument('--output_dir', default='./synthetic_gestures',
                       help='Output directory for synthetic dataset')
    parser.add_argument('--num_classes', type=int, default=8,
                       help='Number of gesture classes')
    parser.add_argument('--samples_per_class', type=int, default=30,
                       help='Number of samples per class')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    config_path = create_synthetic_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        val_split=args.val_split
    )
    
    return config_path


if __name__ == "__main__":
    main()