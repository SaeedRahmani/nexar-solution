"""
"""Predict and Aggregate from Validation Set

This script:
1. Loads your trained model
2. Predicts on VALIDATION set only (unbiased evaluation)
3. Tracks which segments belong to which videos
4. Aggregates to video level with metrics and visualizations

Outputs to aggregated_results_val/
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util

# Import from training script
def import_train_module():
    """Import training module."""
    spec = importlib.util.spec_from_file_location("train_module", "train-large.py")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train_module"] = train_module
    spec.loader.exec_module(train_module)
    return train_module

train_module = import_train_module()
VideoDataset = train_module.VideoDataset
val_transform = train_module.val_transform
FRAME_COUNT = train_module.FRAME_COUNT
device = train_module.device
VideoMAEClassifier = train_module.VideoMAEClassifier

from aggregate_predictions import PredictionAggregator

def load_model(model_path):
    """Load trained model."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = VideoMAEClassifier().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✅ Model loaded!")
    if 'val_acc' in checkpoint:
        print(f"   Segment-level validation accuracy: {checkpoint['val_acc']:.2f}%")
    return model

def predict_and_track(model, val_loader, val_dataset):
    """
    Make predictions and track which file each prediction belongs to.
    """
    print("\nMaking predictions on validation segments...")
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for i, (videos, labels) in enumerate(tqdm(val_loader, desc="Predicting")):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track file paths for this batch
            batch_start = i * val_loader.batch_size
            batch_end = min(batch_start + len(labels), len(val_dataset))
            for idx in range(batch_start, batch_end):
                file_path, _ = val_dataset.samples[idx]
                # Convert to relative path matching metadata format
                # From: balanced_dataset_2s/val/positive/00457_win122_l1.avi
                # To: val/positive/00457_win122_l1.avi
                rel_path = os.path.relpath(file_path, 'balanced_dataset_2s')
                all_paths.append(rel_path)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'path': all_paths,
        'pred': all_preds,
        'prob': all_probs,
        'true_label_segment': all_labels
    })
    
    # Calculate segment-level accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    print(f"\n✅ Segment-level accuracy: {accuracy:.2f}%")
    print(f"   Total segments predicted: {len(all_preds)}")
    
    return predictions_df

def main():
    print("="*70)
    print("PREDICT & AGGREGATE FROM VALIDATION SET")
    print("="*70)
    
    # Configuration
    MODEL_PATH = 'logs/archive/best_videomae_large_model.pth'
    VAL_ROOT = 'balanced_dataset_2s/val'
    BATCH_SIZE = 4
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        # Try alternative
        MODEL_PATH = 'best_videomae_large_model.pth'
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model not found at {MODEL_PATH} either")
            return
    
    if not os.path.exists(VAL_ROOT):
        print(f"❌ Error: Validation data not found at {VAL_ROOT}")
        return
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Load validation dataset (same as test_model-large.py)
    print(f"\nLoading validation dataset from {VAL_ROOT}...")
    val_dataset = VideoDataset(VAL_ROOT, val_transform, FRAME_COUNT)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"✅ Loaded {len(val_dataset)} validation segments")
    
    # Make predictions
    predictions_df = predict_and_track(model, val_loader, val_dataset)
    
    # Save segment predictions
    os.makedirs('aggregated_results_val', exist_ok=True)
    seg_path = 'aggregated_results_val/segment_predictions.csv'
    predictions_df.to_csv(seg_path, index=False)
    print(f"\n✅ Segment predictions saved to: {seg_path}")
    
    # Initialize aggregator
    print("\n" + "="*70)
    print("AGGREGATING TO VIDEO LEVEL")
    print("="*70)
    
    aggregator = PredictionAggregator(
        metadata_path='balanced_dataset_2s/metadata.csv',
        original_labels_path='dataset/train.csv',
        output_dir='aggregated_results_val'
    )
    
    # Load and merge with metadata
    segment_preds = aggregator.load_segment_predictions(predictions_df)
    
    # Aggregate using 'any' method (if ANY segment is positive, video is positive)
    print(f"\n{'='*70}")
    print("METHOD: ANY (Your Requirement)")
    print("If ANY segment is positive, the entire video is positive")
    print(f"{'='*70}")
    
    agg_df = aggregator.aggregate_to_original_videos(
        segment_preds,
        aggregation_method='any',
        threshold=0.5
    )
    
    # Calculate metrics
    metrics, cm = aggregator.calculate_metrics(agg_df)
    
    # Save results
    aggregator.save_results(agg_df, metrics, 'any')
    
    # Plot confusion matrix
    aggregator.plot_confusion_matrix(cm)
    
    # Plot comparison
    aggregator.plot_comparison(agg_df)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\nResults saved to 'aggregated_results_val/' directory:")
    print("  - aggregated_predictions_any.csv   ← Video-level predictions")
    print("  - metrics_any.json                 ← Performance metrics")
    print("  - confusion_matrix_original_videos.png")
    print("  - detailed_report_any.txt          ← Readable summary")
    print("\nTo view results:")
    print("  cat aggregated_results_val/detailed_report_any.txt")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
