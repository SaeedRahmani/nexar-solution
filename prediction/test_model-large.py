"""
Test the trained VideoMAE model on validation set
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from train-large.py
def import_train_module():
    train_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "train-large.py")
    spec = importlib.util.spec_from_file_location("train_module", train_path)
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

# Load the trained model
def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = VideoMAEClassifier().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Model loaded! Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    return model, checkpoint

# Test the model
def test_model(model, test_loader):
    print("\nEvaluating model on validation set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(test_loader)} batches...")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_probs):
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n✅ Overall Accuracy: {acc:.2f}%")
    
    # Classification Report
    print("\nClassification Report:")
    print("-" * 60)
    class_names = ['Negative', 'Positive']
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC Score: {auc:.4f}")
    except:
        print("Could not calculate ROC-AUC")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("-" * 60)
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg   {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"       Pos   {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 60)
    for i, name in enumerate(class_names):
        class_acc = (cm[i][i] / cm[i].sum()) * 100
        print(f"{name}: {class_acc:.2f}%")
    
    return cm

# Plot confusion matrix
def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")

# Main testing function
def main():
    print("="*60)
    print("VideoMAE Model Testing")
    print("="*60)
    
    # Paths
    model_path = "best_videomae_large_model.pth"
    val_root = "balanced_dataset_2s/val"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Check if validation data exists
    if not os.path.exists(val_root):
        print(f"Error: Validation data not found at {val_root}")
        return
    
    # Load model
    model, checkpoint = load_model(model_path)
    
    # Load validation dataset
    print(f"\nLoading validation dataset from {val_root}...")
    val_dataset = VideoDataset(val_root, val_transform, FRAME_COUNT)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4,  # Smaller batch for testing
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    # Test model
    y_pred, y_true, y_probs = test_model(model, val_loader)
    
    # Calculate and display metrics
    cm = calculate_metrics(y_true, y_pred, y_probs)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)
    
    # Save results
    results = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'predictions': y_pred.tolist(),
        'labels': y_true.tolist(),
        'probabilities': y_probs.tolist()
    }
    
    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: test_results.json")

if __name__ == "__main__":
    main()