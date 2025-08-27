import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Optional
import urllib.request


class CloudDatasetDownloader:
    """
    Downloader for popular gesture recognition datasets from various cloud sources.
    """
    
    def __init__(self, base_dir: str = "./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def download_jester(self, subset: str = "small") -> Dict[str, str]:
        """
        Download Jester dataset (20BN-JESTER).
        
        Args:
            subset: "small" (~1GB) or "full" (~22GB)
        
        Returns:
            Dictionary with paths to dataset components
        """
        jester_dir = self.base_dir / "jester"
        jester_dir.mkdir(parents=True, exist_ok=True)
        
        print("Downloading Jester dataset...")
        print("Note: This requires registration at https://developer.qualcomm.com/software/ai-datasets/jester")
        
        # These URLs would need to be updated with actual download links
        if subset == "small":
            files = {
                "videos": "jester_small_videos.tar.gz",
                "train_labels": "jester_train_labels.csv",
                "val_labels": "jester_val_labels.csv",
                "test_labels": "jester_test_labels.csv"
            }
        else:
            files = {
                "videos": "jester_videos.tar.gz", 
                "train_labels": "jester_train_labels.csv",
                "val_labels": "jester_val_labels.csv",
                "test_labels": "jester_test_labels.csv"
            }
        
        # Create mock annotations for demo
        self._create_jester_demo_annotations(jester_dir)
        
        return {
            "data_path": str(jester_dir / "videos"),
            "train_annotation": str(jester_dir / "train_annotations.json"),
            "val_annotation": str(jester_dir / "val_annotations.json"),
            "test_annotation": str(jester_dir / "test_annotations.json")
        }
    
    def download_ego_gesture(self) -> Dict[str, str]:
        """Download EgoGesture dataset from cloud storage."""
        ego_dir = self.base_dir / "egogesture"
        ego_dir.mkdir(parents=True, exist_ok=True)
        
        print("EgoGesture dataset download...")
        print("Note: Available at http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html")
        
        # Create demo structure
        self._create_egogesture_demo_annotations(ego_dir)
        
        return {
            "data_path": str(ego_dir / "videos"),
            "train_annotation": str(ego_dir / "train_annotations.json"),
            "val_annotation": str(ego_dir / "val_annotations.json")
        }
    
    def download_chalearn_isogd(self) -> Dict[str, str]:
        """Download ChaLearn IsoGD dataset."""
        chalearn_dir = self.base_dir / "chalearn_isogd"
        chalearn_dir.mkdir(parents=True, exist_ok=True)
        
        print("ChaLearn IsoGD dataset download...")
        print("Note: Available at http://chalearnlap.cvc.uab.es/dataset/21/description/")
        
        # Create demo structure
        self._create_chalearn_demo_annotations(chalearn_dir)
        
        return {
            "data_path": str(chalearn_dir / "videos"),
            "train_annotation": str(chalearn_dir / "train_annotations.json"),
            "val_annotation": str(chalearn_dir / "val_annotations.json")
        }
    
    def download_something_something_v2(self) -> Dict[str, str]:
        """Download Something-Something V2 dataset."""
        ss_dir = self.base_dir / "something_something_v2"
        ss_dir.mkdir(parents=True, exist_ok=True)
        
        print("Something-Something V2 dataset...")
        print("Note: Available at https://developer.qualcomm.com/software/ai-datasets/something-something")
        
        # Create demo structure
        self._create_something_something_demo_annotations(ss_dir)
        
        return {
            "data_path": str(ss_dir / "videos"),
            "train_annotation": str(ss_dir / "train_annotations.json"),
            "val_annotation": str(ss_dir / "val_annotations.json")
        }
    
    def _create_jester_demo_annotations(self, jester_dir: Path):
        """Create demo annotations for Jester dataset."""
        # Jester has 27 gesture classes
        classes = [
            "Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
            "Pushing Hand Away", "Pulling Hand In", "Sliding Two Fingers Left",
            "Sliding Two Fingers Right", "Sliding Two Fingers Down", "Sliding Two Fingers Up",
            "Pushing Two Fingers Away", "Pulling Two Fingers In", "Rolling Hand Forward",
            "Rolling Hand Backward", "Turning Hand Clockwise", "Turning Hand Counterclockwise",
            "Zooming In With Full Hand", "Zooming Out With Full Hand", "Zooming In With Two Fingers",
            "Zooming Out With Two Fingers", "Thumb Up", "Thumb Down", "Shaking Hand",
            "Stop Sign", "Drumming Fingers", "No gesture", "Doing other things"
        ]
        
        # Create sample annotations
        train_annotations = []
        val_annotations = []
        
        for i, class_name in enumerate(classes):
            # Create training samples
            for j in range(100):  # 100 samples per class
                train_annotations.append({
                    "video_path": f"{class_name.replace(' ', '_').lower()}/video_{j:04d}.mp4",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
            
            # Create validation samples
            for j in range(20):  # 20 samples per class
                val_annotations.append({
                    "video_path": f"{class_name.replace(' ', '_').lower()}/val_video_{j:04d}.mp4",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
        
        # Save annotations
        with open(jester_dir / "train_annotations.json", "w") as f:
            json.dump(train_annotations, f, indent=2)
        
        with open(jester_dir / "val_annotations.json", "w") as f:
            json.dump(val_annotations, f, indent=2)
        
        # Create class info
        with open(jester_dir / "classes.json", "w") as f:
            json.dump({"classes": classes}, f, indent=2)
    
    def _create_egogesture_demo_annotations(self, ego_dir: Path):
        """Create demo annotations for EgoGesture dataset."""
        classes = [
            "grab", "tap", "expand", "pinch", "rotate_cw", "rotate_ccw",
            "swipe_right", "swipe_left", "swipe_up", "swipe_down",
            "push", "pull", "shake", "thumb_up", "thumb_down",
            "index_point", "ok", "c_shape", "peace"
        ]
        
        train_annotations = []
        val_annotations = []
        
        for i, class_name in enumerate(classes):
            for j in range(80):
                train_annotations.append({
                    "video_path": f"subject_{j%20:02d}/{class_name}/video_{j:04d}.mp4",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
            
            for j in range(20):
                val_annotations.append({
                    "video_path": f"subject_{j%5:02d}/{class_name}/val_video_{j:04d}.mp4",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
        
        with open(ego_dir / "train_annotations.json", "w") as f:
            json.dump(train_annotations, f, indent=2)
        
        with open(ego_dir / "val_annotations.json", "w") as f:
            json.dump(val_annotations, f, indent=2)
    
    def _create_chalearn_demo_annotations(self, chalearn_dir: Path):
        """Create demo annotations for ChaLearn IsoGD dataset."""
        classes = [
            "vattene", "vieniqui", "perfetto", "furbo", "cheduepalle",
            "chevuoi", "daccordo", "seipazzo", "combinato", "freganiente",
            "ok", "cosatifarei", "basta", "prendere", "noncenepiu",
            "fame", "tantotempo", "buonissimo", "messidaccordo", "sonostufo"
        ]
        
        train_annotations = []
        val_annotations = []
        
        for i, class_name in enumerate(classes):
            for j in range(60):
                train_annotations.append({
                    "video_path": f"{class_name}/video_{j:04d}.mp4",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
            
            for j in range(15):
                val_annotations.append({
                    "video_path": f"{class_name}/val_video_{j:04d}.mp4",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
        
        with open(chalearn_dir / "train_annotations.json", "w") as f:
            json.dump(train_annotations, f, indent=2)
        
        with open(chalearn_dir / "val_annotations.json", "w") as f:
            json.dump(val_annotations, f, indent=2)
    
    def _create_something_something_demo_annotations(self, ss_dir: Path):
        """Create demo annotations for Something-Something V2 dataset."""
        classes = [
            "Approaching something with your camera",
            "Attaching something to something",
            "Bending something so that it deforms",
            "Bending something until it breaks",
            "Burying something in something",
            "Closing something",
            "Covering something with something",
            "Digging something out of something",
            "Dropping something behind something",
            "Dropping something in front of something",
            "Dropping something into something",
            "Dropping something next to something",
            "Dropping something onto something",
            "Failing to put something into something because something does not fit",
            "Folding something",
            "Hitting something with something",
            "Inserting something into something",
            "Letting something roll along a flat surface",
            "Letting something roll down a slanted surface",
            "Letting something roll up a slanted surface"
        ]
        
        train_annotations = []
        val_annotations = []
        
        for i, class_name in enumerate(classes):
            for j in range(200):  # More samples for temporal reasoning
                train_annotations.append({
                    "video_path": f"videos/{i*1000+j:06d}.webm",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
            
            for j in range(50):
                val_annotations.append({
                    "video_path": f"videos/val_{i*200+j:06d}.webm",
                    "label": class_name,
                    "start_frame": 0,
                    "end_frame": -1
                })
        
        with open(ss_dir / "train_annotations.json", "w") as f:
            json.dump(train_annotations, f, indent=2)
        
        with open(ss_dir / "val_annotations.json", "w") as f:
            json.dump(val_annotations, f, indent=2)
    
    def list_available_datasets(self):
        """List all available datasets and their characteristics."""
        datasets_info = {
            "Jester (20BN-JESTER)": {
                "size": "~22GB (full), ~1GB (subset)",
                "samples": "148,092 videos",
                "classes": 27,
                "description": "Hand gestures for human-computer interaction",
                "url": "https://developer.qualcomm.com/software/ai-datasets/jester"
            },
            "Something-Something V2": {
                "size": "~20GB",
                "samples": "220,847 videos", 
                "classes": 174,
                "description": "Temporal reasoning with everyday objects",
                "url": "https://developer.qualcomm.com/software/ai-datasets/something-something"
            },
            "EgoGesture": {
                "size": "~3GB",
                "samples": "24,161 videos",
                "classes": 83,
                "description": "Egocentric hand gestures",
                "url": "http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html"
            },
            "ChaLearn IsoGD": {
                "size": "~8GB",
                "samples": "47,933 videos",
                "classes": 249,
                "description": "Isolated gesture recognition",
                "url": "http://chalearnlap.cvc.uab.es/dataset/21/description/"
            },
            "Kinetics-400": {
                "size": "~500GB",
                "samples": "400,000 videos",
                "classes": 400,
                "description": "Human action recognition (includes gestures)",
                "url": "https://deepmind.com/research/open-source/kinetics"
            }
        }
        
        print("Available Gesture Recognition Datasets:")
        print("=" * 50)
        
        for name, info in datasets_info.items():
            print(f"\n{name}:")
            print(f"  Size: {info['size']}")
            print(f"  Samples: {info['samples']}")
            print(f"  Classes: {info['classes']}")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")


def main():
    """Example usage of cloud dataset downloader."""
    downloader = CloudDatasetDownloader()
    
    # List available datasets
    downloader.list_available_datasets()
    
    print("\n" + "="*50)
    print("Creating demo dataset structures...")
    
    # Create demo dataset structures
    datasets = {
        "jester": downloader.download_jester("small"),
        "egogesture": downloader.download_ego_gesture(),
        "chalearn": downloader.download_chalearn_isogd(),
        "something_something": downloader.download_something_something_v2()
    }
    
    print("\nDemo dataset structures created:")
    for name, paths in datasets.items():
        print(f"\n{name.upper()}:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
    
    print("\nTo use with the training pipeline:")
    print("1. Download actual videos from the provided URLs")
    print("2. Update the config.yaml with the correct paths")
    print("3. Or use HuggingFace datasets for easier access")


if __name__ == "__main__":
    main()