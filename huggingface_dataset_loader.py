import torch
from datasets import load_dataset
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
import os
from tqdm import tqdm


class HuggingFaceGestureDataset(Dataset):
    """
    Dataset loader for gesture datasets from Hugging Face Hub.
    """
    
    def __init__(self,
                 dataset_name: str,
                 split: str = 'train',
                 clip_len: int = 16,
                 crop_size: int = 112,
                 cache_dir: str = "./hf_cache",
                 max_samples: Optional[int] = None):
        """
        Initialize HF gesture dataset.
        
        Args:
            dataset_name: Name of the dataset on HF Hub
            split: Dataset split ('train', 'validation', 'test')
            clip_len: Number of frames per clip
            crop_size: Size of spatial crop
            cache_dir: Directory to cache downloaded data
            max_samples: Limit number of samples (for testing)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset from HF Hub
        self.dataset = self._load_hf_dataset(max_samples)
        self.classes = self._get_classes()
        self.num_classes = len(self.classes)
        
        print(f"Loaded {len(self.dataset)} samples from {dataset_name}:{split}")
        print(f"Classes: {self.classes}")
    
    def _load_hf_dataset(self, max_samples: Optional[int]):
        """Load dataset from Hugging Face Hub."""
        try:
            # Try to load the dataset
            dataset = load_dataset(
                self.dataset_name, 
                split=self.split,
                cache_dir=str(self.cache_dir)
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            return dataset
            
        except Exception as e:
            print(f"Error loading {self.dataset_name}: {e}")
            print("Available gesture datasets on HF Hub:")
            self._list_available_datasets()
            raise
    
    def _list_available_datasets(self):
        """List some popular gesture datasets."""
        popular_datasets = [
            "google/jester",
            "google/something-something-v2", 
            "facebook/egogesture",
            "laion/hand-gesture-recognition",
            "nielsr/ChaLearn-IsoGD",
            "microsoft/kinetics-400"  # Action recognition (includes gestures)
        ]
        
        print("Popular gesture/action datasets:")
        for dataset in popular_datasets:
            print(f"  - {dataset}")
    
    def _get_classes(self) -> List[str]:
        """Extract class labels from dataset."""
        # Try different common label field names
        label_fields = ['label', 'class', 'category', 'gesture', 'action']
        
        for field in label_fields:
            if field in self.dataset.features:
                if hasattr(self.dataset.features[field], 'names'):
                    return self.dataset.features[field].names
                elif hasattr(self.dataset.features[field], '_str2int'):
                    return list(self.dataset.features[field]._str2int.keys())
        
        # Fallback: extract unique labels from data
        labels = set()
        for item in self.dataset:
            for field in label_fields:
                if field in item:
                    labels.add(item[field])
                    break
        
        return sorted(list(labels))
    
    def _process_video(self, video_data) -> np.ndarray:
        """Process video data from different formats."""
        if isinstance(video_data, list):
            # List of PIL images
            frames = []
            for img in video_data[:self.clip_len]:
                frame = np.array(img)
                if frame.shape[-1] == 4:  # RGBA
                    frame = frame[:, :, :3]  # Remove alpha
                frames.append(frame)
            return np.array(frames)
        
        elif hasattr(video_data, 'path'):
            # Video file path
            return self._load_video_from_path(video_data.path)
        
        elif isinstance(video_data, str):
            # Video file path as string
            return self._load_video_from_path(video_data)
        
        else:
            raise ValueError(f"Unsupported video format: {type(video_data)}")
    
    def _load_video_from_path(self, video_path: str) -> np.ndarray:
        """Load video from file path."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")
        
        # Pad with last frame if needed
        while len(frames) < self.clip_len:
            frames.append(frames[-1])
        
        return np.array(frames)
    
    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess frames for model input."""
        processed_frames = []
        
        for frame in frames:
            # Resize and crop
            h, w = frame.shape[:2]
            
            # Resize smaller dimension to crop_size + margin
            scale = max(self.crop_size / h, self.crop_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Center crop
            start_h = (new_h - self.crop_size) // 2
            start_w = (new_w - self.crop_size) // 2
            
            frame_cropped = frame_resized[
                start_h:start_h + self.crop_size,
                start_w:start_w + self.crop_size
            ]
            
            # Normalize to [0, 1]
            frame_normalized = frame_cropped.astype(np.float32) / 255.0
            
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame_tensor = (frame_tensor - mean) / std
            
            processed_frames.append(frame_tensor)
        
        # Stack and reorder to (C, T, H, W)
        video_tensor = torch.stack(processed_frames)  # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return video_tensor
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        try:
            item = self.dataset[idx]
            
            # Extract video data
            video_fields = ['video', 'frames', 'clip', 'sequence']
            video_data = None
            
            for field in video_fields:
                if field in item:
                    video_data = item[field]
                    break
            
            if video_data is None:
                raise ValueError(f"No video data found in item keys: {list(item.keys())}")
            
            # Process video
            frames = self._process_video(video_data)
            
            # Sample frames temporally
            if len(frames) > self.clip_len:
                # Uniform sampling
                indices = np.linspace(0, len(frames) - 1, self.clip_len).astype(int)
                frames = frames[indices]
            
            # Preprocess frames
            video_tensor = self._preprocess_frames(frames)
            
            # Get label
            label_fields = ['label', 'class', 'category', 'gesture', 'action']
            label = None
            
            for field in label_fields:
                if field in item:
                    label = item[field]
                    break
            
            if label is None:
                raise ValueError(f"No label found in item keys: {list(item.keys())}")
            
            # Convert label to int if needed
            if isinstance(label, str):
                label = self.classes.index(label)
            
            return video_tensor, label
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a random sample as fallback
            return self.__getitem__(np.random.randint(0, len(self.dataset)))


def download_jester_dataset(data_dir: str = "./data/jester") -> Dict[str, str]:
    """
    Download and prepare the Jester dataset.
    Returns paths to annotation files.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Jester dataset URLs (These are examples - check actual availability)
    urls = {
        'train': 'https://example.com/jester/train_annotations.json',
        'validation': 'https://example.com/jester/val_annotations.json', 
        'videos': 'https://example.com/jester/videos.tar.gz'
    }
    
    print("Note: Jester dataset requires manual download from:")
    print("https://developer.qualcomm.com/software/ai-datasets/jester")
    print("\nAlternatively, try these Hugging Face datasets:")
    print("- google/something-something-v2")
    print("- facebook/egogesture") 
    print("- microsoft/kinetics-400")
    
    # Return expected paths
    return {
        'train_annotation': str(data_dir / 'train_annotations.json'),
        'val_annotation': str(data_dir / 'val_annotations.json'),
        'data_path': str(data_dir / 'videos')
    }


def create_hf_gesture_dataloaders(
    dataset_name: str,
    batch_size: int = 8,
    clip_len: int = 16,
    crop_size: int = 112,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create data loaders for HuggingFace gesture datasets.
    
    Args:
        dataset_name: Name of dataset on HF Hub
        batch_size: Batch size
        clip_len: Number of frames per clip
        crop_size: Spatial crop size
        max_train_samples: Limit training samples (for testing)
        max_val_samples: Limit validation samples (for testing)
    
    Returns:
        train_loader, val_loader, class_names
    """
    
    # Load training dataset
    train_dataset = HuggingFaceGestureDataset(
        dataset_name=dataset_name,
        split='train',
        clip_len=clip_len,
        crop_size=crop_size,
        max_samples=max_train_samples
    )
    
    # Load validation dataset
    try:
        val_dataset = HuggingFaceGestureDataset(
            dataset_name=dataset_name,
            split='validation',
            clip_len=clip_len,
            crop_size=crop_size,
            max_samples=max_val_samples
        )
    except:
        # Use test split if validation not available
        try:
            val_dataset = HuggingFaceGestureDataset(
                dataset_name=dataset_name,
                split='test',
                clip_len=clip_len,
                crop_size=crop_size,
                max_samples=max_val_samples
            )
        except:
            # Split training set if no validation available
            print("No validation/test split found, using 20% of training data")
            train_size = int(0.8 * len(train_dataset.dataset))
            val_size = len(train_dataset.dataset) - train_size
            
            train_data, val_data = torch.utils.data.random_split(
                train_dataset.dataset, [train_size, val_size]
            )
            
            train_dataset.dataset = train_data
            val_dataset = train_dataset
            val_dataset.dataset = val_data
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.classes


if __name__ == "__main__":
    # Example usage
    print("Testing HuggingFace gesture dataset loader...")
    
    # List of datasets to try (some might not exist)
    test_datasets = [
        "microsoft/kinetics-400",  # Large action recognition dataset
        "google/something-something-v2",  # Temporal reasoning dataset
        "facebook/egogesture",  # Egocentric gestures
    ]
    
    for dataset_name in test_datasets:
        try:
            print(f"\nTrying dataset: {dataset_name}")
            
            # Create small test dataset
            dataset = HuggingFaceGestureDataset(
                dataset_name=dataset_name,
                split='train',
                clip_len=8,
                crop_size=64,
                max_samples=10  # Small for testing
            )
            
            print(f"Successfully loaded {len(dataset)} samples")
            print(f"Classes: {dataset.classes[:5]}...")  # Show first 5 classes
            
            # Test one sample
            video, label = dataset[0]
            print(f"Sample shape: {video.shape}, Label: {label}")
            
            break  # Stop at first successful dataset
            
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue
    
    else:
        print("\nNo datasets could be loaded. You may need to:")
        print("1. Install datasets: pip install datasets")
        print("2. Check dataset availability on Hugging Face Hub")
        print("3. Use manual dataset download functions provided")