import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import json
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
from collections import defaultdict
import time

from gesture_recognition_model import create_gesture_model
from jester_dataset_loader import JesterDataset


class GestureEvaluator:
    """
    Comprehensive evaluation suite for gesture recognition models.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 class_names: Optional[List[str]] = None):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def evaluate_dataset(self, 
                        dataloader, 
                        save_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation dataset
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        all_predictions = []
        all_labels = []
        all_probs = []
        inference_times = []
        
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")
            
            for videos, labels in progress_bar:
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.cpu().numpy()
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(videos)
                inference_time = time.time() - start_time
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels)
                all_probs.extend(probs.cpu().numpy())
                inference_times.append(inference_time / len(videos))  # Per sample
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_predictions, all_labels, all_probs, inference_times
        )
        
        # Save results
        self._save_evaluation_results(results, save_path)
        
        # Generate plots
        self._generate_evaluation_plots(
            all_predictions, all_labels, all_probs, save_path
        )
        
        return results
    
    def _calculate_comprehensive_metrics(self, 
                                       predictions: List[int], 
                                       labels: List[int],
                                       probabilities: List[np.ndarray],
                                       inference_times: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic accuracy metrics
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro'
        )
        
        # Per-class accuracy
        per_class_acc = self._calculate_per_class_accuracy(predictions, labels)
        
        # Confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            predictions, labels, probabilities
        )
        
        # Timing metrics
        timing_metrics = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1
            },
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'support': support.tolist(),
                'accuracy': per_class_acc
            },
            'confidence_metrics': confidence_metrics,
            'timing_metrics': timing_metrics,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(labels),
            'num_classes': len(set(labels))
        }
        
        return results
    
    def _calculate_per_class_accuracy(self, 
                                    predictions: List[int], 
                                    labels: List[int]) -> List[float]:
        """Calculate accuracy for each class."""
        num_classes = len(set(labels))
        per_class_acc = []
        
        for class_id in range(num_classes):
            class_mask = np.array(labels) == class_id
            if np.sum(class_mask) > 0:
                class_predictions = np.array(predictions)[class_mask]
                class_labels = np.array(labels)[class_mask]
                acc = np.mean(class_predictions == class_labels)
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        return per_class_acc
    
    def _calculate_confidence_metrics(self, 
                                    predictions: List[int], 
                                    labels: List[int],
                                    probabilities: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate confidence-based metrics."""
        probs_array = np.array(probabilities)
        predictions_array = np.array(predictions)
        labels_array = np.array(labels)
        
        # Max probability (confidence) for each prediction
        max_probs = np.max(probs_array, axis=1)
        
        # Entropy (uncertainty measure)
        entropy = -np.sum(probs_array * np.log(probs_array + 1e-8), axis=1)
        
        # Correct vs incorrect predictions confidence
        correct_mask = predictions_array == labels_array
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        # Top-k accuracy
        top_k_accuracies = {}
        for k in [1, 3, 5]:
            if k <= probs_array.shape[1]:
                top_k_pred = np.argsort(probs_array, axis=1)[:, -k:]
                top_k_acc = np.mean([label in pred_k for label, pred_k in zip(labels_array, top_k_pred)])
                top_k_accuracies[f'top_{k}_accuracy'] = top_k_acc
        
        confidence_metrics = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'mean_entropy': np.mean(entropy),
            'std_entropy': np.std(entropy),
            'correct_predictions_confidence': {
                'mean': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0.0,
                'std': np.std(correct_confidences) if len(correct_confidences) > 0 else 0.0
            },
            'incorrect_predictions_confidence': {
                'mean': np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0.0,
                'std': np.std(incorrect_confidences) if len(incorrect_confidences) > 0 else 0.0
            },
            **top_k_accuracies
        }
        
        return confidence_metrics
    
    def _save_evaluation_results(self, results: Dict[str, Any], save_path: Path):
        """Save evaluation results to JSON and CSV files."""
        
        # Save complete results as JSON
        with open(save_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save per-class metrics as CSV
        if self.class_names and len(self.class_names) == len(results['per_class_metrics']['precision']):
            class_names = self.class_names
        else:
            class_names = [f"Class_{i}" for i in range(len(results['per_class_metrics']['precision']))]
        
        per_class_df = pd.DataFrame({
            'Class': class_names,
            'Precision': results['per_class_metrics']['precision'],
            'Recall': results['per_class_metrics']['recall'],
            'F1-Score': results['per_class_metrics']['f1_score'],
            'Support': results['per_class_metrics']['support'],
            'Accuracy': results['per_class_metrics']['accuracy']
        })
        
        per_class_df.to_csv(save_path / "per_class_metrics.csv", index=False)
        
        # Save summary metrics
        summary_df = pd.DataFrame([results['overall_metrics']])
        summary_df.to_csv(save_path / "overall_metrics.csv", index=False)
    
    def _generate_evaluation_plots(self, 
                                 predictions: List[int], 
                                 labels: List[int],
                                 probabilities: List[np.ndarray], 
                                 save_path: Path):
        """Generate evaluation plots."""
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        
        if self.class_names:
            class_names = self.class_names[:len(set(labels))]
        else:
            class_names = [f"Class {i}" for i in range(len(set(labels)))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confidence distribution
        probs_array = np.array(probabilities)
        max_probs = np.max(probs_array, axis=1)
        correct_mask = np.array(predictions) == np.array(labels)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='Correct', color='green')
        plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence (Max Probability)')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        
        # Entropy distribution
        entropy = -np.sum(probs_array * np.log(probs_array + 1e-8), axis=1)
        plt.subplot(1, 2, 2)
        plt.hist(entropy[correct_mask], bins=50, alpha=0.7, label='Correct', color='green')
        plt.hist(entropy[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Entropy (Uncertainty)')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_real_time(self, 
                          duration_seconds: int = 60, 
                          save_dir: str = "./realtime_evaluation") -> Dict[str, Any]:
        """
        Evaluate model performance on real-time webcam input.
        
        Args:
            duration_seconds: How long to run the evaluation
            save_dir: Directory to save results
            
        Returns:
            Real-time evaluation metrics
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement webcam dataset for real-time evaluation
        print("Real-time evaluation não implementado ainda para datasets reais")
        return {'error': 'Real-time evaluation not implemented for real datasets'}
        
        predictions = []
        confidences = []
        inference_times = []
        frame_rates = []
        
        print(f"Starting real-time evaluation for {duration_seconds} seconds...")
        print("Press 'q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                frame_start = time.time()
                
                # Get clip from webcam
                clip = webcam_dataset.get_clip()
                if clip is None:
                    continue
                
                clip = clip.to(self.device)
                
                # Model inference
                inference_start = time.time()
                with torch.no_grad():
                    outputs = self.model(clip)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                
                inference_time = time.time() - inference_start
                
                # Store results
                predictions.append(predicted.item())
                confidences.append(torch.max(probs).item())
                inference_times.append(inference_time)
                
                # Calculate frame rate
                frame_time = time.time() - frame_start
                frame_rates.append(1.0 / frame_time if frame_time > 0 else 0.0)
                
                frame_count += 1
                
                # Display progress
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_count} clips in {elapsed:.1f}s, "
                          f"Avg FPS: {np.mean(frame_rates[-10:]):.1f}")
        
        except KeyboardInterrupt:
            print("Evaluation interrupted by user")
        
        finally:
            webcam_dataset.release()
        
        # Calculate real-time metrics
        results = {
            'total_clips_processed': frame_count,
            'total_duration': time.time() - start_time,
            'inference_metrics': {
                'mean_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times)
            },
            'frame_rate_metrics': {
                'mean_fps': np.mean(frame_rates),
                'std_fps': np.std(frame_rates),
                'min_fps': np.min(frame_rates),
                'max_fps': np.max(frame_rates)
            },
            'confidence_metrics': {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            },
            'prediction_distribution': dict(zip(*np.unique(predictions, return_counts=True)))
        }
        
        # Save results
        with open(save_path / "realtime_evaluation.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def benchmark_inference_speed(self, 
                                input_shapes: List[Tuple[int, int, int, int, int]],
                                num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark inference speed for different input shapes.
        
        Args:
            input_shapes: List of input shapes (N, C, T, H, W)
            num_runs: Number of inference runs per shape
            
        Returns:
            Benchmarking results
        """
        results = {}
        
        self.model.eval()
        
        for shape in input_shapes:
            N, C, T, H, W = shape
            shape_key = f"{N}x{C}x{T}x{H}x{W}"
            
            print(f"Benchmarking shape: {shape_key}")
            
            # Warmup
            dummy_input = torch.randn(shape).to(self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            inference_times = []
            for _ in tqdm(range(num_runs), desc=f"Shape {shape_key}"):
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                inference_times.append(time.time() - start_time)
            
            results[shape_key] = {
                'shape': shape,
                'mean_time': np.mean(inference_times),
                'std_time': np.std(inference_times),
                'min_time': np.min(inference_times),
                'max_time': np.max(inference_times),
                'fps': 1.0 / np.mean(inference_times),
                'samples_per_second': N / np.mean(inference_times)
            }
        
        return results


def main():
    """Example usage of the gesture evaluator."""
    
    # Load model (example)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_gesture_model('full', num_classes=20)
    
    # Load trained weights (uncomment if you have a checkpoint)
    # checkpoint = torch.load('path/to/checkpoint.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize evaluator
    class_names = [f"Gesture_{i}" for i in range(20)]  # Replace with actual class names
    evaluator = GestureEvaluator(model, device, class_names)
    
    # Benchmark inference speed
    print("Benchmarking inference speed...")
    shapes = [
        (1, 3, 16, 112, 112),  # Single sample
        (4, 3, 16, 112, 112),  # Small batch
        (8, 3, 16, 112, 112),  # Medium batch
    ]
    
    benchmark_results = evaluator.benchmark_inference_speed(shapes, num_runs=50)
    
    for shape_key, metrics in benchmark_results.items():
        print(f"\nShape {shape_key}:")
        print(f"  Mean inference time: {metrics['mean_time']*1000:.2f} ms")
        print(f"  FPS: {metrics['fps']:.1f}")
        print(f"  Samples/sec: {metrics['samples_per_second']:.1f}")


if __name__ == "__main__":
    main()