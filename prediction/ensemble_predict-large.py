import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import csv
from torch.cuda.amp import autocast
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import shutil
from transformers import AutoModel, AutoConfig

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    # Input/Output paths - UPDATE THESE FOR YOUR SETUP
    test_video_dir = "dataset/test"  # Test video directory
    output_video_dir = "processed_videos"  # Processed video save directory
    final_output_csv = "ensemble_submission.csv"  # Final results

    # Video processing parameters
    target_fps = 10  # Target frame rate
    img_size = (320, 320)  # Image size when processing videos
    num_threads = 8  # Video processing threads

    # Video window settings (all use 16 frames, but different time lengths)
    window_configs = [
        {"name": "2s", "duration": 2.0, "model_path": "best_videomae_large_model.pth"}
    ]
    
    # Model parameters
    frame_count = 16  # All models use 16 frames
    model_img_size = 224  # VideoMAE model input size
    batch_size = 2  # Reduced for VideoMAE-large
    temperature = 2.0  # Temperature scaling parameter, consistent with training
    video_format = "avi"  # Save format
    codec = "XVID"  # Codec

# Video processing class
class VideoProcessor:
    def __init__(self):
        self._init_directories()
    
    def _init_directories(self):
        # Create directories for each window config
        for config in Config.window_configs:
            window_dir = os.path.join(Config.output_video_dir, config["name"])
            os.makedirs(window_dir, exist_ok=True)
            # Clear old files in directory
            for f in os.listdir(window_dir):
                file_path = os.path.join(window_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _extract_frames(self, video_path):
        """Extract video frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(fps / Config.target_fps)))
        
        # Store all frames and timestamps
        all_frames = []
        all_timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, Config.img_size)
                all_frames.append(frame)
                all_timestamps.append(frame_count / fps)
            
            frame_count += 1
        
        cap.release()
        
        if not all_frames:
            print(f"No valid frames in video: {video_path}")
            return None, None
            
        return all_frames, all_timestamps
    
    def _save_video_window(self, frames, save_path):
        """Save frame sequence as video"""
        fourcc = cv2.VideoWriter_fourcc(*Config.codec)
        writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
        
        if not writer.isOpened():
            # Try backup codecs
            backup_codecs = ["MJPG", "DIVX", "MPEG", "X264"]
            for codec in backup_codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
                if writer.isOpened():
                    break
        
        if not writer.isOpened():
            print(f"Cannot create video writer: {save_path}")
            return False
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        return True
    
    def process_video(self, video_file):
        """Process single test video, extract different time windows"""
        try:
            video_path = os.path.join(Config.test_video_dir, video_file)
            video_id = os.path.splitext(video_file)[0]
            
            # Extract all frames
            all_frames, all_timestamps = self._extract_frames(video_path)
            if all_frames is None:
                return
            
            video_duration = all_timestamps[-1]
            
            # Extract different time windows for each config
            for config in Config.window_configs:
                window_duration = config["duration"]
                window_name = config["name"]
                
                # Calculate window start/end time, take specified seconds from video end
                end_time = video_duration
                start_time = max(0, end_time - window_duration)
                
                # Select frames within window
                window_frames = []
                for i, ts in enumerate(all_timestamps):
                    if start_time <= ts <= end_time:
                        window_frames.append(all_frames[i])
                
                # Ensure exactly 16 frames
                if len(window_frames) > Config.frame_count:
                    # Uniformly sample 16 frames
                    indices = np.linspace(0, len(window_frames)-1, Config.frame_count, dtype=int)
                    window_frames = [window_frames[i] for i in indices]
                elif len(window_frames) < Config.frame_count:
                    # If insufficient frames, duplicate last frame
                    while len(window_frames) < Config.frame_count:
                        window_frames.append(window_frames[-1] if window_frames else np.zeros_like(all_frames[0]))
                
                # Save as video file
                save_dir = os.path.join(Config.output_video_dir, window_name)
                save_path = os.path.join(save_dir, f"{video_id}.{Config.video_format}")
                if not self._save_video_window(window_frames, save_path):
                    print(f"Failed to save video: {save_path}")
            
            print(f"Processing complete: {video_id}")
            
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")
    
    def run(self):
        """Process all test videos"""
        video_files = [f for f in os.listdir(Config.test_video_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print("ERROR: No video files found in test directory!")
            return
        
        print(f"Found {len(video_files)} test videos, starting processing...")
        
        with ThreadPoolExecutor(max_workers=Config.num_threads) as executor:
            list(tqdm(executor.map(self.process_video, video_files), 
                     total=len(video_files), desc="Processing test videos"))
        
        print(f"Video processing complete, output directory: {Config.output_video_dir}")

# Define VideoMAE-large model (consistent with training code)
class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VideoMAEClassifier, self).__init__()
        
        # Load VideoMAE-large config
        config = AutoConfig.from_pretrained("MCG-NJU/videomae-large", trust_remote_code=True)
        
        # Set drop_path_rate (consistent with train.py)
        drop_path_value = 0.1
        if hasattr(config, 'model_config') and isinstance(config.model_config, dict):
            config.model_config['drop_path_rate'] = drop_path_value
        elif hasattr(config, 'drop_path_rate'):
            config.drop_path_rate = drop_path_value

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                "MCG-NJU/videomae-large",
                config=config,
                trust_remote_code=True
            ).to(device)
            print("Successfully loaded VideoMAE-large pretrained weights")
        else:
            self.backbone = AutoModel.from_config(config, trust_remote_code=True).to(device)
            print("Using randomly initialized VideoMAE-large model")

        # Dynamically get feature dimension
        with torch.no_grad():
            dummy_input = torch.rand(1, Config.frame_count, 3, Config.model_img_size, Config.model_img_size).to(device)
            dummy_outputs = self.backbone(pixel_values=dummy_input)
            feature_dim = dummy_outputs.last_hidden_state.shape[-1]
        
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: (B, T, C, H, W) tensor - VideoMAE expects this format
        x = x.to(device, dtype=torch.float32)
        outputs = self.backbone(pixel_values=x)
        # Get [CLS] token features
        features = outputs.last_hidden_state[:, 0]
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

# Video prediction dataset
class VideoPredictDataset(Dataset):
    def __init__(self, video_paths, transform=None, frame_count=16):
        self.video_paths = video_paths
        self.frame_count = frame_count
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        # Open and validate video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return video_id, torch.zeros((self.frame_count, 3, Config.model_img_size, Config.model_img_size))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If frame count mismatch, print warning and adjust
        if total_frames != self.frame_count:
            print(f"Warning: Video {video_path} has {total_frames} frames, expected {self.frame_count}, will adjust")
        
        # Read frames
        frames = []
        frame_idx = 0
        while len(frames) < self.frame_count:
            ret, frame = cap.read()
            if not ret:
                # If insufficient frames, pad with black frames
                frame = np.zeros((Config.model_img_size, Config.model_img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += 1
            
            # If video frames exceed, only take first 16 frames
            if frame_idx >= self.frame_count:
                break
        
        cap.release()
        
        # Ensure exactly 16 frames
        if len(frames) < self.frame_count:
            frames.extend([np.zeros((Config.model_img_size, Config.model_img_size, 3), dtype=np.uint8)] * (self.frame_count - len(frames)))
        frames = frames[:self.frame_count]
        
        # Preprocess frames
        processed = torch.stack([self.transform(image=f)["image"] for f in frames])  # (T, C, H, W)
        return video_id, processed

# Temperature scaling function
def apply_temperature_scaling(logits, temperature=2.0):
    """Temperature scaling, consistent with training"""
    return logits / temperature

# Predictor class
class Predictor:
    def __init__(self):
        # Data preprocessing (consistent with training val_transform)
        self.transform = A.Compose([
            A.Resize(Config.model_img_size, Config.model_img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Store predictions from each model
        self.all_predictions = {}
    
    def load_model(self, model_path):
        """Load model from specified path"""
        model = VideoMAEClassifier(num_classes=2, pretrained=True).to(device)
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f"Model weights loaded successfully: {model_path}")
            print(f"  Trained for {checkpoint['epoch']+1} epochs")
            print(f"  Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        except FileNotFoundError:
            print(f"Error: Cannot find model file '{model_path}', please ensure file exists!")
            return None
        except KeyError:
            print("Error: Weight file missing 'model' key, check save format!")
            return None
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            return None
        
        model.eval()
        return model
    
    def predict_videos(self, window_config):
        """Predict for videos in specified window configuration"""
        window_name = window_config["name"]
        model_path = window_config["model_path"]
        
        print(f"\nStarting processing for {window_name} window videos...")
        
        # Load model
        model = self.load_model(model_path)
        if model is None:
            return
        
        # Get video files
        video_dir = os.path.join(Config.output_video_dir, window_name)
        video_files = [
            os.path.join(video_dir, f) 
            for f in os.listdir(video_dir) 
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        
        if not video_files:
            print(f"Error: No video files found in {video_dir} directory!")
            return
        
        # Create data loader
        dataset = VideoPredictDataset(video_files, transform=self.transform, frame_count=Config.frame_count)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: (
                [item[0] for item in x],  # video_ids
                torch.stack([item[1] for item in x])  # frames
            )
        )
        
        # Execute prediction
        results = {}
        with torch.no_grad():
            for batch_ids, batch_frames in tqdm(dataloader, desc=f"Predicting {window_name} window"):
                # Move to device
                inputs = batch_frames.to(device)  # (B, T, C, H, W)
                
                # Forward pass, use mixed precision
                with autocast():
                    outputs = model(inputs)
                    outputs_scaled = apply_temperature_scaling(outputs, temperature=Config.temperature)
                
                # Calculate probabilities
                scores = torch.softmax(outputs_scaled, dim=1)[:, 1].cpu().numpy()
                
                # Store results
                for vid_id, score in zip(batch_ids, scores):
                    results[vid_id] = score
        
        # Save prediction results for this window config
        self.all_predictions[window_name] = results
        print(f"{window_name} window prediction complete, total {len(results)} videos")
        
        # Save individual model prediction results (optional)
        output_csv = f"prediction_{window_name}.csv"
        with open(output_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for vid_id, score in results.items():
                writer.writerow([vid_id, f"{score:.4f}"])
        print(f"Single model prediction results saved to {output_csv}")
    
    def ensemble_predictions(self):
        """Ensemble predictions from all models"""
        if not self.all_predictions:
            print("Error: No prediction results available for ensembling!")
            return
        
        # Get all video IDs
        all_video_ids = set()
        for predictions in self.all_predictions.values():
            all_video_ids.update(predictions.keys())
        
        # Calculate average prediction scores
        ensemble_results = {}
        for vid_id in all_video_ids:
            scores = []
            for window_name, predictions in self.all_predictions.items():
                if vid_id in predictions:
                    scores.append(predictions[vid_id])
            
            if scores:
                ensemble_results[vid_id] = np.mean(scores)
            else:
                print(f"Warning: Video {vid_id} has no valid prediction scores")
                ensemble_results[vid_id] = 0.5  # Default score
        
        # Save ensemble results
        with open(Config.final_output_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for vid_id, score in sorted(ensemble_results.items()):
                writer.writerow([vid_id, f"{score:.4f}"])
        
        print(f"\nAll models ensembled! Results saved to {Config.final_output_csv}")
        print(f"Total processed {len(ensemble_results)} videos")
        
        # Print statistics
        scores_array = np.array(list(ensemble_results.values()))
        print(f"\nScore statistics:")
        print(f"  Mean: {scores_array.mean():.4f}")
        print(f"  Std:  {scores_array.std():.4f}")
        print(f"  Min:  {scores_array.min():.4f}")
        print(f"  Max:  {scores_array.max():.4f}")
        
        positive = (scores_array > 0.5).sum()
        negative = (scores_array <= 0.5).sum()
        print(f"\nPredicted distribution:")
        print(f"  Positive (>0.5): {positive} ({positive/len(scores_array)*100:.1f}%)")
        print(f"  Negative (≤0.5): {negative} ({negative/len(scores_array)*100:.1f}%)")
    
    def run(self):
        """Run predictions for all window configurations and ensemble results"""
        for window_config in Config.window_configs:
            self.predict_videos(window_config)
        
        self.ensemble_predictions()

# Main program
def main():
    print("="*60)
    print("Ensemble Prediction Pipeline - VideoMAE-Large")
    print("="*60)
    
    # Check paths
    if not os.path.exists(Config.test_video_dir):
        print(f"\nError: Test directory not found: {Config.test_video_dir}")
        print("Please update Config.test_video_dir in the script")
        return
    
    print("\nStep 1: Processing test videos...")
    processor = VideoProcessor()
    processor.run()
    
    print("\nStep 2: Model prediction...")
    predictor = Predictor()
    predictor.run()
    
    print("\n" + "="*60)
    print("Ensemble prediction pipeline complete!")
    print("="*60)

if __name__ == "__main__":
    main()
