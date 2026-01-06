"""
Predict and Aggregate from Train + Validation Sets (ALL)

This script:
1. Loads your trained model
2. Predicts on TRAIN + VALIDATION splits (for scenario analysis)
3. Aggregates predictions to original video level
4. Saves results for scenario analysis with large sample sizes

⚠️  WARNING: This includes training data - accuracy will be BIASED/INFLATED!
   Use this ONLY for per-scenario statistics, NOT for model evaluation.
   For unbiased evaluation, use: predict_agg_from_val.py
   
CHECKPOINTING: If interrupted, the script will resume from cached predictions.

Outputs to video_predictions_ALL/
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from training script
def import_train_module():
    """Import training module."""
    print("Loading training module...", flush=True)
    train_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "train-large.py")
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train_module"] = train_module
    spec.loader.exec_module(train_module)
    print("✓ Training module loaded", flush=True)
    return train_module

print("Starting predict_all_dataset.py...", flush=True)
train_module = import_train_module()
VideoDataset = train_module.VideoDataset
val_transform = train_module.val_transform
FRAME_COUNT = train_module.FRAME_COUNT
device = train_module.device
VideoMAEClassifier = train_module.VideoMAEClassifier

from utils.aggregate_predictions import PredictionAggregator

def load_model(model_path):
    """Load trained model."""
    print(f"Loading model from {model_path}...", flush=True)
    checkpoint = torch.load(model_path, map_location=device)
    print("Checkpoint loaded, initializing model...", flush=True)
    
    model = VideoMAEClassifier().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✅ Model loaded successfully!", flush=True)
    if 'val_acc' in checkpoint:
        print(f"   Original validation accuracy: {checkpoint['val_acc']:.2f}%", flush=True)
    return model

def predict_and_track(model, data_loader, dataset, split_name):
    """
    Make predictions and track which file each prediction belongs to.
    """
    print(f"\nMaking predictions on {split_name} segments...", flush=True)
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for i, (videos, labels) in enumerate(tqdm(data_loader, desc=f"Predicting {split_name}")):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track file paths for this batch
            batch_start = i * data_loader.batch_size
            batch_end = min(batch_start + len(labels), len(dataset))
            for idx in range(batch_start, batch_end):
                file_path, _ = dataset.samples[idx]
                # Convert to relative path (remove balanced_dataset_2s/ prefix)
                # Keep forward slashes - aggregator normalizes both paths
                PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                rel_path = os.path.relpath(file_path, os.path.join(PROJECT_ROOT, 'balanced_dataset_2s'))
                all_paths.append(rel_path)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'path': all_paths,
        'pred': all_preds,
        'prob': all_probs,
        'segment_label': all_labels
    })
    
    # Compute segment-level accuracy
    segment_acc = 100 * (df['pred'] == df['segment_label']).sum() / len(df)
    print(f"✅ {split_name} segment-level accuracy: {segment_acc:.2f}%", flush=True)
    
    return df

def main():
    """Main execution"""
    print("="*80, flush=True)
    print("PREDICTING ON ENTIRE DATASET (TRAIN + VALIDATION)", flush=True)
    print("="*80, flush=True)
    print("⚠️  WARNING: This includes training data!", flush=True)
    print("    Results will be biased/inflated - do NOT use for model evaluation.", flush=True)
    print("    Use this ONLY for scenario distribution analysis with larger samples.", flush=True)
    print("="*80, flush=True)
    
    # Get project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuration (relative to project root)
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'logs/archive/best_videomae_large_model.pth')
    TRAIN_ROOT = os.path.join(PROJECT_ROOT, 'balanced_dataset_2s/train')
    VAL_ROOT = os.path.join(PROJECT_ROOT, 'balanced_dataset_2s/val')
    METADATA_PATH = os.path.join(PROJECT_ROOT, 'balanced_dataset_2s/metadata.csv')
    GROUND_TRUTH_PATH = os.path.join(PROJECT_ROOT, 'dataset/train.csv')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'video_predictions_ALL')
    BATCH_SIZE = 8
    
    # Checkpoint paths
    TRAIN_PRED_CACHE = os.path.join(OUTPUT_DIR, '_train_predictions.csv')
    VAL_PRED_CACHE = os.path.join(OUTPUT_DIR, '_val_predictions.csv')
    SEGMENT_OUTPUT = os.path.join(OUTPUT_DIR, 'segment_predictions.csv')
    AGGREGATED_OUTPUT = os.path.join(OUTPUT_DIR, 'aggregated_predictions_any.csv')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    
    # Check if final output already exists
    if os.path.exists(AGGREGATED_OUTPUT):
        print(f"\n✅ Final output already exists: {AGGREGATED_OUTPUT}")
        print("Delete this file if you want to regenerate predictions.")
        return
    
    # Load model only if needed
    model = None
    if not os.path.exists(TRAIN_PRED_CACHE) or not os.path.exists(VAL_PRED_CACHE):
        model = load_model(MODEL_PATH)
    
    # ===== TRAIN SPLIT =====
    print("\n" + "="*80, flush=True)
    print("STEP 1: Predict on TRAINING split", flush=True)
    print("="*80, flush=True)
    
    if os.path.exists(TRAIN_PRED_CACHE):
        print(f"✅ Loading cached training predictions from {TRAIN_PRED_CACHE}", flush=True)
        train_predictions = pd.read_csv(TRAIN_PRED_CACHE)
        print(f"Loaded {len(train_predictions)} training predictions", flush=True)
    else:
        print(f"Initializing training dataset from {TRAIN_ROOT}...", flush=True)
        train_dataset = VideoDataset(
            root_dir=TRAIN_ROOT,
            transform=val_transform,
            frame_count=FRAME_COUNT
        )
        print(f"Training segments found: {len(train_dataset)}", flush=True)
        
        print("Creating data loader...", flush=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print("Starting prediction on training data...", flush=True)
        train_predictions = predict_and_track(model, train_loader, train_dataset, "TRAIN")
        train_predictions.to_csv(TRAIN_PRED_CACHE, index=False)
        print(f"✅ Cached training predictions to {TRAIN_PRED_CACHE}", flush=True)
    
    # ===== VALIDATION SPLIT =====
    print("\n" + "="*80, flush=True)
    print("STEP 2: Predict on VALIDATION split", flush=True)
    print("="*80, flush=True)
    
    if os.path.exists(VAL_PRED_CACHE):
        print(f"✅ Loading cached validation predictions from {VAL_PRED_CACHE}", flush=True)
        val_predictions = pd.read_csv(VAL_PRED_CACHE)
        print(f"Loaded {len(val_predictions)} validation predictions", flush=True)
    else:
        print(f"Initializing validation dataset from {VAL_ROOT}...", flush=True)
        val_dataset = VideoDataset(
            root_dir=VAL_ROOT,
            transform=val_transform,
            frame_count=FRAME_COUNT
        )
        print(f"Validation segments found: {len(val_dataset)}", flush=True)
        
        print("Creating data loader...", flush=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print("Starting prediction on validation data...", flush=True)
        val_predictions = predict_and_track(model, val_loader, val_dataset, "VAL")
        val_predictions.to_csv(VAL_PRED_CACHE, index=False)
        print(f"✅ Cached validation predictions to {VAL_PRED_CACHE}", flush=True)
    
    # ===== COMBINE =====
    print("\n" + "="*80, flush=True)
    print("STEP 3: Combine predictions from both splits", flush=True)
    print("="*80, flush=True)
    
    all_predictions = pd.concat([train_predictions, val_predictions], ignore_index=True)
    print(f"Total segments: {len(all_predictions)}", flush=True)
    print(f"Overall segment accuracy: {100 * (all_predictions['pred'] == all_predictions['segment_label']).sum() / len(all_predictions):.2f}%", flush=True)
    
    # Save segment predictions
    all_predictions.to_csv(SEGMENT_OUTPUT, index=False)
    print(f"✅ Saved segment predictions: {SEGMENT_OUTPUT}", flush=True)
    
    # ===== AGGREGATE TO VIDEO LEVEL =====
    print("\n" + "="*80, flush=True)
    print("STEP 4: Aggregate predictions to original video level", flush=True)
    print("="*80, flush=True)
    
    aggregator = PredictionAggregator(
        metadata_path=METADATA_PATH,
        original_labels_path=GROUND_TRUTH_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Load segment predictions with metadata merged
    merged_predictions = aggregator.load_segment_predictions(all_predictions)
    
    # Aggregate using 'any' method
    video_predictions = aggregator.aggregate_to_original_videos(
        merged_predictions,
        aggregation_method='any'
    )
    
    print(f"\n✅ Aggregated to {len(video_predictions)} original videos", flush=True)
    
    # Calculate video-level accuracy (only for videos with true labels)
    valid_videos = video_predictions.dropna(subset=['true_label'])
    if len(valid_videos) > 0:
        video_acc = 100 * (valid_videos['pred'] == valid_videos['true_label']).sum() / len(valid_videos)
        print(f"Video-level accuracy: {video_acc:.2f}% (on {len(valid_videos)} videos)", flush=True)
        print(f"  (NOTE: This is INFLATED because it includes training data!)", flush=True)
    else:
        print("⚠️  No videos with true labels found", flush=True)
    
    # Calculate metrics (same as VAL version for consistency)
    metrics, cm = aggregator.calculate_metrics(video_predictions)
    
    # Save results (includes metrics_any.json, detailed_report_any.txt)
    aggregator.save_results(video_predictions, metrics, 'any')
    
    # Plot confusion matrix
    aggregator.plot_confusion_matrix(cm)
    
    # Plot comparison
    aggregator.plot_comparison(video_predictions)
    
    print("\n" + "="*80, flush=True)
    print("COMPLETE (ALL DATA - BIASED)!", flush=True)
    print("="*80, flush=True)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/", flush=True)
    print("  - aggregated_predictions_any.csv   ← Video-level predictions", flush=True)
    print("  - segment_predictions.csv          ← Segment-level predictions", flush=True)
    print("  - metrics_any.json                 ← Performance metrics (BIASED)", flush=True)
    print("  - detailed_report_any.txt          ← Readable summary (BIASED)", flush=True)
    print("  - confusion_matrix_original_videos.png (BIASED)", flush=True)
    print("  - segment_vs_video_comparison.png", flush=True)
    print("\n⚠️  WARNING: These metrics are BIASED (includes training data)!", flush=True)
    print("    For unbiased evaluation, use: python predict_agg_from_val.py", flush=True)
    print("\nNext step:", flush=True)
    print("  Run: python analyze_scenarios_ALL.py", flush=True)

if __name__ == '__main__':
    main()
