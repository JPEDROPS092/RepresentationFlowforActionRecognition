#!/usr/bin/env python3
"""
Find real gesture/action datasets on HuggingFace Hub.
"""

try:
    from huggingface_hub import list_datasets
    from datasets import load_dataset_builder
    import requests
    
    def search_gesture_datasets():
        """Search for real gesture/action datasets on HF Hub."""
        print("Searching for real gesture/action datasets on HuggingFace Hub...")
        
        # Keywords to search for
        keywords = ['gesture', 'action', 'video', 'kinetics', 'jester', 'hand', 'sign']
        
        try:
            # Get all datasets (this might take a while)
            datasets = list_datasets(filter="dataset_info:*video*")
            
            gesture_datasets = []
            for dataset in datasets[:50]:  # Limit to first 50 for speed
                dataset_name = dataset.id
                tags = getattr(dataset, 'tags', [])
                
                # Check if dataset might contain gestures/actions
                if any(keyword in dataset_name.lower() or 
                      any(keyword in tag.lower() for tag in tags if isinstance(tag, str))
                      for keyword in keywords):
                    gesture_datasets.append(dataset_name)
            
            return gesture_datasets
            
        except Exception as e:
            print(f"Error searching datasets: {e}")
            return []
    
    def verify_datasets(datasets):
        """Verify which datasets actually exist and are accessible."""
        working_datasets = []
        
        for dataset_name in datasets:
            try:
                print(f"Checking {dataset_name}...")
                builder = load_dataset_builder(dataset_name)
                info = builder.info
                
                if hasattr(info, 'features') and info.features:
                    # Check if it looks like a video dataset
                    feature_names = list(info.features.keys())
                    if any(name in ['video', 'image', 'frames', 'clip'] for name in feature_names):
                        working_datasets.append({
                            'name': dataset_name,
                            'features': feature_names,
                            'description': getattr(info, 'description', 'No description')[:100],
                            'splits': list(builder.info.splits.keys()) if builder.info.splits else []
                        })
                        print(f"  ✓ {dataset_name} - looks like video dataset")
                    else:
                        print(f"  ✗ {dataset_name} - not a video dataset")
                else:
                    print(f"  ✗ {dataset_name} - no features info")
                    
            except Exception as e:
                print(f"  ✗ {dataset_name} - error: {e}")
        
        return working_datasets
    
    def main():
        print("=" * 60)
        print("Finding Real Gesture/Action Datasets on HuggingFace Hub")
        print("=" * 60)
        
        # Search for datasets
        candidate_datasets = search_gesture_datasets()
        
        if not candidate_datasets:
            # Fallback to known working datasets
            print("Search failed, trying known datasets...")
            candidate_datasets = [
                'microsoft/kinetics-400',
                'google/kinetics400', 
                'kinetics_400',
                'deepmind/kinetics400',
                'ucf101',
                'UCF-101',
                'jester',
                '20bn-jester',
                'something-something-v2',
                'chalearn',
                'sign-language'
            ]
        
        print(f"Found {len(candidate_datasets)} candidate datasets")
        
        # Verify datasets
        working_datasets = verify_datasets(candidate_datasets[:10])  # Test first 10
        
        print("\n" + "=" * 60)
        print("Working Video Datasets:")
        print("=" * 60)
        
        for dataset in working_datasets:
            print(f"\n{dataset['name']}:")
            print(f"  Features: {dataset['features']}")
            print(f"  Splits: {dataset['splits']}")
            print(f"  Description: {dataset['description']}")
        
        return working_datasets

    if __name__ == "__main__":
        working_datasets = main()
        
        if working_datasets:
            print(f"\n🎉 Found {len(working_datasets)} working datasets!")
            print("You can use any of these with:")
            for dataset in working_datasets[:3]:  # Show first 3
                print(f"  python setup_cloud_dataset.py --dataset {dataset['name']} --type huggingface")
        else:
            print("\n⚠ No working video datasets found on HuggingFace Hub")
            print("Falling back to manual dataset download...")

except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install: pip install datasets huggingface-hub")
    print("\nFor now, let's use manual dataset setup...")
    
    # Simple alternative without HF dependencies
    print("\nAlternative: Use synthetic dataset for testing")
    print("Run: python create_synthetic_dataset.py")