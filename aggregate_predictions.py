"""
Aggregate predictions from 2-second video segments back to original videos.
If ANY segment from an original video is predicted as positive (1), 
the original video is labeled as positive (1).
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

class PredictionAggregator:
    def __init__(self, 
                 metadata_path='balanced_dataset_2s/metadata.csv',
                 original_labels_path='dataset/train.csv',
                 output_dir='aggregated_results'):
        """
        Initialize the aggregator.
        
        Args:
            metadata_path: Path to metadata.csv that maps segments to original videos
            original_labels_path: Path to train.csv with original video labels
            output_dir: Directory to save aggregated results
        """
        self.metadata_path = metadata_path
        self.original_labels_path = original_labels_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metadata and original labels
        self.metadata = pd.read_csv(metadata_path)
        self.original_df = pd.read_csv(original_labels_path)
        
        print(f"Loaded metadata: {len(self.metadata)} segments")
        print(f"Loaded original labels: {len(self.original_df)} videos")
        
    def load_segment_predictions(self, predictions_dict):
        """
        Load predictions for video segments.
        
        Args:
            predictions_dict: Dictionary mapping segment paths to predictions
                             Format: {segment_path: {'pred': 0/1, 'prob': 0.0-1.0}}
                             OR a DataFrame with columns ['path', 'pred', 'prob']
        
        Returns:
            DataFrame with segment-level predictions merged with metadata
        """
        if isinstance(predictions_dict, pd.DataFrame):
            pred_df = predictions_dict.copy()
        else:
            # Convert dictionary to DataFrame
            pred_data = []
            for path, pred_info in predictions_dict.items():
                pred_data.append({
                    'path': path,
                    'pred': pred_info['pred'],
                    'prob': pred_info.get('prob', pred_info['pred'])
                })
            pred_df = pd.DataFrame(pred_data)
        
        # Normalize paths for matching (handle both / and \)
        pred_df['path'] = pred_df['path'].str.replace('\\', '/').str.replace('//', '/')
        metadata_copy = self.metadata.copy()
        metadata_copy['path'] = metadata_copy['path'].str.replace('\\', '/').str.replace('//', '/')
        
        # Merge predictions with metadata using INNER join (only keep segments with predictions)
        merged = metadata_copy.merge(pred_df, on='path', how='inner')
        
        print(f"\n✅ Merged {len(merged)} segments with predictions")
        print(f"   Segments in predictions: {len(pred_df)}")
        print(f"   Segments in metadata: {len(metadata_copy)}")
        if len(merged) < len(pred_df):
            print(f"   ⚠️  Warning: {len(pred_df) - len(merged)} predictions couldn't be matched to metadata")
        
        return merged
    
    def aggregate_to_original_videos(self, segment_predictions_df, 
                                     aggregation_method='any',
                                     threshold=0.5):
        """
        Aggregate segment predictions to original video level.
        
        Args:
            segment_predictions_df: DataFrame with segment predictions and source video IDs
            aggregation_method: How to aggregate predictions
                - 'any': If ANY segment is positive, video is positive (default)
                - 'majority': If MAJORITY of segments are positive, video is positive
                - 'max_prob': Use maximum probability across segments
                - 'avg_prob': Use average probability across segments with threshold
            threshold: Probability threshold for classification (used with prob methods)
        
        Returns:
            DataFrame with aggregated predictions per original video
        """
        print(f"\nAggregating predictions using method: {aggregation_method}")
        
        # Group by source video ID
        aggregated = []
        
        for video_id, group in segment_predictions_df.groupby('source'):
            # Get predictions and probabilities for all segments from this video
            preds = group['pred'].values
            probs = group['prob'].values
            segment_label = group['label'].iloc[0]  # Ground truth for segments
            
            # Handle NaN in probabilities
            valid_probs = probs[~pd.isna(probs)]
            if len(valid_probs) == 0:
                # If no valid probabilities, use predictions only
                valid_probs = preds.astype(float)
            
            # Aggregate based on method
            if aggregation_method == 'any':
                # If ANY segment is positive (1), video is positive
                agg_pred = 1 if (preds == 1).any() else 0
                agg_prob = valid_probs.max() if len(valid_probs) > 0 else 0.0
                
            elif aggregation_method == 'majority':
                # If MAJORITY of segments are positive, video is positive
                agg_pred = 1 if (preds == 1).sum() > len(preds) / 2 else 0
                agg_prob = valid_probs.mean() if len(valid_probs) > 0 else 0.0
                
            elif aggregation_method == 'max_prob':
                # Use maximum probability
                agg_prob = valid_probs.max() if len(valid_probs) > 0 else 0.0
                agg_pred = 1 if agg_prob >= threshold else 0
                
            elif aggregation_method == 'avg_prob':
                # Use average probability
                agg_prob = valid_probs.mean() if len(valid_probs) > 0 else 0.0
                agg_pred = 1 if agg_prob >= threshold else 0
            
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            aggregated.append({
                'video_id': video_id,
                'pred': agg_pred,
                'prob': agg_prob,
                'num_segments': len(group),
                'num_positive_segments': (preds == 1).sum(),
                'segment_label': segment_label
            })
        
        agg_df = pd.DataFrame(aggregated)
        
        # Merge with original labels
        # Convert video_id format: "00128" -> "01924" or whatever format is in train.csv
        self.original_df['id'] = self.original_df['id'].astype(str).str.zfill(5)
        agg_df['video_id'] = agg_df['video_id'].astype(str).str.zfill(5)
        
        agg_df = agg_df.merge(
            self.original_df[['id', 'target']], 
            left_on='video_id', 
            right_on='id', 
            how='left'
        )
        
        # Rename for clarity
        agg_df['true_label'] = agg_df['target']
        agg_df = agg_df.drop(['target', 'id'], axis=1)
        
        print(f"Aggregated to {len(agg_df)} original videos")
        print(f"Missing true labels: {agg_df['true_label'].isna().sum()}")
        
        return agg_df
    
    def calculate_metrics(self, agg_df):
        """
        Calculate performance metrics on original video level.
        
        Args:
            agg_df: DataFrame with aggregated predictions and true labels
        
        Returns:
            Dictionary with various metrics
        """
        # Remove any rows with missing labels
        valid_df = agg_df.dropna(subset=['true_label'])
        
        y_true = valid_df['true_label'].values.astype(int)
        y_pred = valid_df['pred'].values.astype(int)
        y_prob = valid_df['prob'].values
        
        print("\n" + "="*70)
        print("ORIGINAL VIDEO LEVEL PERFORMANCE METRICS")
        print("="*70)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        
        print("\nPer-Class Metrics:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        print(f"{'Negative (0)':<15} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}")
        print(f"{'Positive (1)':<15} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}")
        
        # Weighted average
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        print(f"{'Weighted Avg':<15} {precision_w:<12.4f} {recall_w:<12.4f} {f1_w:<12.4f}")
        
        # ROC-AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"\nROC-AUC Score: {auc:.4f}")
        except Exception as e:
            print(f"\nCould not calculate ROC-AUC: {e}")
            auc = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("-" * 70)
        print(f"                    Predicted")
        print(f"                 Negative  Positive")
        print(f"Actual Negative    {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"       Positive    {cm[1][0]:6d}    {cm[1][1]:6d}")
        
        # Calculate per-class accuracy
        print("\nPer-Class Accuracy:")
        print("-" * 70)
        for i, class_name in enumerate(['Negative', 'Positive']):
            class_acc = (cm[i][i] / cm[i].sum()) * 100 if cm[i].sum() > 0 else 0
            print(f"{class_name}: {class_acc:.2f}%")
        
        # Additional statistics
        print("\nAdditional Statistics:")
        print("-" * 70)
        print(f"True Negatives:  {cm[0][0]} (correctly identified safe videos)")
        print(f"False Positives: {cm[0][1]} (safe videos classified as dangerous)")
        print(f"False Negatives: {cm[1][0]} (dangerous videos classified as safe) ⚠️")
        print(f"True Positives:  {cm[1][1]} (correctly identified dangerous videos)")
        
        metrics = {
            'accuracy': accuracy,
            'precision_neg': precision[0],
            'precision_pos': precision[1],
            'recall_neg': recall[0],
            'recall_pos': recall[1],
            'f1_neg': f1[0],
            'f1_pos': f1[1],
            'support_neg': int(support[0]),
            'support_pos': int(support[1]),
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'num_videos': len(valid_df)
        }
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix_original_videos.png')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative (Safe)', 'Positive (Dangerous)'],
                    yticklabels=['Negative (Safe)', 'Positive (Dangerous)'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Original Video Level', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_comparison(self, agg_df, save_path=None):
        """Plot comparison of segment-level vs video-level predictions."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'segment_vs_video_comparison.png')
        
        valid_df = agg_df.dropna(subset=['true_label'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Distribution of positive segments per video
        ax1 = axes[0]
        positive_videos = valid_df[valid_df['true_label'] == 1]
        negative_videos = valid_df[valid_df['true_label'] == 0]
        
        ax1.hist([positive_videos['num_positive_segments'], 
                  negative_videos['num_positive_segments']], 
                 label=['True Positive Videos', 'True Negative Videos'],
                 bins=20, alpha=0.7, color=['red', 'blue'])
        ax1.set_xlabel('Number of Positive Segments')
        ax1.set_ylabel('Number of Videos')
        ax1.set_title('Distribution of Positive Segments per Video')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Probability distribution
        ax2 = axes[1]
        ax2.hist([positive_videos['prob'], negative_videos['prob']], 
                 label=['True Positive Videos', 'True Negative Videos'],
                 bins=20, alpha=0.7, color=['red', 'blue'])
        ax2.set_xlabel('Aggregated Probability')
        ax2.set_ylabel('Number of Videos')
        ax2.set_title('Aggregated Probability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        plt.close()
    
    def save_results(self, agg_df, metrics, method_name):
        """Save aggregated results and metrics."""
        # Save aggregated predictions
        results_path = os.path.join(self.output_dir, f'aggregated_predictions_{method_name}.csv')
        agg_df.to_csv(results_path, index=False)
        print(f"\nAggregated predictions saved to: {results_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, f'metrics_{method_name}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save detailed report
        report_path = os.path.join(self.output_dir, f'detailed_report_{method_name}.txt')
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"AGGREGATED PREDICTION RESULTS - {method_name.upper()}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Aggregation Method: {method_name}\n")
            f.write(f"Total Videos: {metrics['num_videos']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-"*70 + "\n")
            f.write(f"Negative Class:\n")
            f.write(f"  Precision: {metrics['precision_neg']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_neg']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_neg']:.4f}\n")
            f.write(f"  Support:   {metrics['support_neg']}\n\n")
            
            f.write(f"Positive Class:\n")
            f.write(f"  Precision: {metrics['precision_pos']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_pos']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_pos']:.4f}\n")
            f.write(f"  Support:   {metrics['support_pos']}\n\n")
            
            if metrics['auc'] is not None:
                f.write(f"ROC-AUC: {metrics['auc']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-"*70 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"                    Predicted\n")
            f.write(f"                 Negative  Positive\n")
            f.write(f"Actual Negative    {cm[0][0]:6d}    {cm[0][1]:6d}\n")
            f.write(f"       Positive    {cm[1][0]:6d}    {cm[1][1]:6d}\n")
        
        print(f"Detailed report saved to: {report_path}")


def example_usage():
    """
    Example usage showing how to use the aggregator.
    """
    print("="*70)
    print("EXAMPLE: Aggregating Segment Predictions to Original Videos")
    print("="*70)
    
    # Initialize aggregator
    aggregator = PredictionAggregator(
        metadata_path='balanced_dataset_2s/metadata.csv',
        original_labels_path='dataset/train.csv',
        output_dir='aggregated_results'
    )
    
    # Example: Create dummy predictions (replace with your actual predictions)
    print("\n⚠️  This is an example with dummy data!")
    print("You need to provide actual predictions from your model.\n")
    
    # Load metadata to get segment paths
    metadata = pd.read_csv('balanced_dataset_2s/metadata.csv')
    
    # Create dummy predictions (replace this with your actual model predictions)
    dummy_predictions = pd.DataFrame({
        'path': metadata['path'].values,
        'pred': metadata['label'].values,  # Using ground truth as dummy predictions
        'prob': np.random.rand(len(metadata))  # Random probabilities
    })
    
    # Load segment predictions
    segment_preds = aggregator.load_segment_predictions(dummy_predictions)
    
    # Try different aggregation methods
    methods = ['any', 'majority', 'max_prob', 'avg_prob']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing aggregation method: {method.upper()}")
        print(f"{'='*70}")
        
        # Aggregate predictions
        agg_df = aggregator.aggregate_to_original_videos(
            segment_preds, 
            aggregation_method=method,
            threshold=0.5
        )
        
        # Calculate metrics
        metrics, cm = aggregator.calculate_metrics(agg_df)
        
        # Save results
        aggregator.save_results(agg_df, metrics, method)
        
        # Plot confusion matrix
        aggregator.plot_confusion_matrix(cm)
        
        # Plot comparison
        if method == 'any':  # Only plot once
            aggregator.plot_comparison(agg_df)
    
    print("\n" + "="*70)
    print("Done! Check the 'aggregated_results' directory for outputs.")
    print("="*70)


if __name__ == '__main__':
    example_usage()
