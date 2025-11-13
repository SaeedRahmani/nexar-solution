"""
Generate predictions on test videos using trained VideoMAE-large model
Updated to match the trained model architecture
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast

# Import from train.py to ensure consistency
from train import VideoMAEClassifier, device, CONFIG, FRAME_COUNT, IMG_SIZE

# Configuration
class PredictConfig:
    # Paths - UPDATE THESE FOR YOUR SETUP
    test_video_dir = "dataset/test"  # Directory with test videos
    model_path = "best_videomae_large_model.pth"  # Your trained model
    output_csv = "predictions.csv"  # Output file
    
    # Model parameters (must match training)
    batch_size = 2
    num_workers = 4
    frame_count = FRAME_COUNT
    img_size = IMG_SIZE
    temperature = CONFIG.get("temperature", 2.0)
    use_temperature_scaling = CONFIG.get("use_temperature_scaling", True)

# Test dataset
class TestVideoDataset(Dataset):
    def __init__(self, video_dir, transform, frame_count=16):
        self.video_paths = []
        self.video_ids = []
        self.transform = transform
        self.frame_count = frame_count
        
        # Find all videos in test directory
        if not os.path.exists(video_dir):
            raise ValueError(f"Test directory not found: {video_dir}")
        
        for fname in sorted(os.listdir(video_dir)):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                self.video_paths.append(os.path.join(video_dir, fname))
                self.video_ids.append(os.path.splitext(fname)[0])
        
        if not self.video_paths:
            raise ValueError(f"No videos found in {video_dir}")
        
        print(f"Found {len(self.video_paths)} test videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def _get_frame_indices(self, total):
        """Same logic as training dataset"""
        if total >= self.frame_count:
            step = total / self.frame_count
            return [int(i * step) for i in range(self.frame_count)]
        else:
            return list(range(total)) + [total-1]*(self.frame_count-total)
    
    def __getitem__(self, idx):
        vpath = self.video_paths[idx]
        video_id = self.video_ids[idx]
        
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {vpath}")
            # Return dummy data
            dummy = torch.zeros((self.frame_count, 3, self.img_size, self.img_size))
            return video_id, dummy
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._get_frame_indices(total)
        
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((self.img_size, self.img_size, 3), np.uint8)
            frames.append(frame)
        cap.release()
        
        # Pad if needed
        while len(frames) < self.frame_count:
            frames.append(frames[-1])
        
        # Apply transforms
        transformed = []
        for f in frames:
            transformed.append(self.transform(image=f)["image"])
        
        vid_tensor = torch.stack(transformed)  # (T, C, H, W)
        return video_id, vid_tensor

def apply_temperature_scaling(logits, temperature=2.0):
    """Apply temperature scaling (same as training)"""
    return logits / temperature

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = VideoMAEClassifier().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Trained for {checkpoint['epoch']+1} epochs")
    print(f"  Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model

def predict(model, dataloader, use_temp_scaling=True, temperature=2.0):
    """Generate predictions"""
    predictions = {}
    
    print("Generating predictions...")
    with torch.no_grad():
        for video_ids, videos in tqdm(dataloader):
            videos = videos.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(videos)
                
                # Apply temperature scaling if used during training
                if use_temp_scaling:
                    outputs = apply_temperature_scaling(outputs, temperature)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                # Get probability of positive class (class 1)
                scores = probs[:, 1].cpu().numpy()
            
            # Store results
            for vid_id, score in zip(video_ids, scores):
                predictions[vid_id] = float(score)
    
    return predictions

def save_predictions(predictions, output_path):
    """Save predictions to CSV"""
    df = pd.DataFrame([
        {"id": vid_id, "score": score}
        for vid_id, score in sorted(predictions.items())
    ])
    
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total videos: {len(predictions)}")
    
    # Print statistics
    scores = df['score'].values
    print(f"\nScore statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")
    
    # Distribution
    positive = (scores > 0.5).sum()
    negative = (scores <= 0.5).sum()
    print(f"\nPredicted distribution:")
    print(f"  Positive (>0.5): {positive} ({positive/len(scores)*100:.1f}%)")
    print(f"  Negative (≤0.5): {negative} ({negative/len(scores)*100:.1f}%)")

def main():
    print("="*60)
    print("VideoMAE Test Set Prediction")
    print("="*60)
    
    # Check paths
    if not os.path.exists(PredictConfig.test_video_dir):
        print(f"\nError: Test directory not found: {PredictConfig.test_video_dir}")
        print("Please update PredictConfig.test_video_dir in the script")
        return
    
    if not os.path.exists(PredictConfig.model_path):
        print(f"\nError: Model file not found: {PredictConfig.model_path}")
        print("Please update PredictConfig.model_path in the script")
        return
    
    # Load model
    model = load_model(PredictConfig.model_path)
    
    # Create test transform (same as validation)
    test_transform = A.Compose([
        A.Resize(PredictConfig.img_size, PredictConfig.img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    
    # Create dataset and dataloader
    test_dataset = TestVideoDataset(
        PredictConfig.test_video_dir,
        test_transform,
        PredictConfig.frame_count
    )
    
    def collate_fn(batch):
        video_ids = [item[0] for item in batch]
        videos = torch.stack([item[1] for item in batch])
        return video_ids, videos
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=PredictConfig.batch_size,
        shuffle=False,
        num_workers=PredictConfig.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Generate predictions
    predictions = predict(
        model,
        test_loader,
        use_temp_scaling=PredictConfig.use_temperature_scaling,
        temperature=PredictConfig.temperature
    )
    
    # Save results
    save_predictions(predictions, PredictConfig.output_csv)
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
