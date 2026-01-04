import os
import cv2
import numpy as np
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
import queue
import io

# Configuration parameters
class Config:
    video_dir = "dataset/train"
    output_dir = "balanced_dataset_2s"
    csv_path = "dataset/train.csv"
    target_fps = 8
    window_size = 16
    window_stride = 2
    img_size = (320, 320)
    test_size = 0.2
    seed = 42
    video_format = "avi"  # Changed back to the more widely supported avi format
    codec = "XVID"        # XVID encoder is available on most systems
    num_threads = min(os.cpu_count() * 2, 16)  # Dynamic thread count
    max_videos = None
    # Add memory buffer size configuration
    buffer_size = 48  # Number of windows to keep in memory

class VideoProcessor:
    def __init__(self):
        self.df = pd.read_csv(Config.csv_path)
        if Config.max_videos:
            self.df = self.df.head(Config.max_videos)
        self._init_directories()
        self.metadata_queue = queue.Queue()  # Asynchronous metadata queue
        self.lock = threading.Lock()
        # Add global video writer cache
        self.writer_cache = {}

    def _init_directories(self):
        for split in ["train", "val"]:
            for label in ["positive", "negative"]:
                path = os.path.join(Config.output_dir, split, label)
                os.makedirs(path, exist_ok=True)
                for f in os.listdir(path):
                    file_path = os.path.join(path, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

    def _extract_frames(self, video_path):
        """Changed to generator, generate frame by frame"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(fps / Config.target_fps)))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, Config.img_size)
                yield frame, frame_count / fps
            frame_count += 1
        cap.release()

    def _get_video_writer(self, save_path):
        """Get or create video writer"""
        cache_key = os.path.abspath(save_path)
        
        # Check if writer for this path already exists in cache
        if cache_key in self.writer_cache:
            return self.writer_cache[cache_key], True
        
        # Create new writer
        fourcc = cv2.VideoWriter_fourcc(*Config.codec)
        writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
        
        # Check if writer is successfully initialized
        if not writer.isOpened():
            # Try backup encoders
            backup_codecs = ["MJPG", "DIVX", "MPEG", "X264"]
            for codec in backup_codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
                if writer.isOpened():
                    break
        
        # If all encoders fail, use uncompressed format
        if not writer.isOpened():
            writer = cv2.VideoWriter(save_path, 0, Config.target_fps, Config.img_size)
            
        # Cache writer
        self.writer_cache[cache_key] = writer
        return writer, False

    def _release_writer(self, save_path):
        """Release writer for specified path"""
        cache_key = os.path.abspath(save_path)
        if cache_key in self.writer_cache:
            self.writer_cache[cache_key].release()
            del self.writer_cache[cache_key]

    def _save_video_window(self, frames, save_path):
        """Use cached VideoWriter"""
        try:
            writer, _ = self._get_video_writer(save_path)
            
            # Batch write frames to improve efficiency
            for frame in frames:
                writer.write(frame)
            
            # Release writer after video writing is complete
            self._release_writer(save_path)
            return True
        except Exception as e:
            print(f"Error occurred while saving video: {str(e)}")
            return False

    def _process_buffer(self, buffer):
        """Process window data in buffer"""
        processed_count = 0
        for item in buffer:
            # Save as video file
            save_success = self._save_video_window(item["frames"], item["save_path"])
            
            # Only record metadata if save is successful
            if save_success:
                self.metadata_queue.put(item["metadata"])
                processed_count += 1
        return processed_count

    def _process_video(self, row, train_ids):
        try:
            video_id = f"{int(row['id']):05d}" 
            video_path = os.path.join(Config.video_dir, video_id + ".mp4")
            base_name = os.path.splitext(video_id)[0]

            frame_gen = self._extract_frames(video_path)
            frames = []
            timestamps = []
            
            # Load initial frames first
            for _ in range(Config.window_size):
                try:
                    frame, ts = next(frame_gen)
                    frames.append(frame)
                    timestamps.append(ts)
                except StopIteration:
                    return

            if len(frames) < Config.window_size:
                return

            split = "train" if row['id'] in train_ids else "val"
            event_time = row['time_of_event']
            alert_time = row['time_of_alert']
            is_positive_video = (row['target'] == 1)

            # Use memory buffer to store window data
            buffer = []
            window_count = 0
            i = 0
            
            while True:
                # Get frames and times for current window
                window_frames = frames[i:i + Config.window_size]
                window_times = timestamps[i:i + Config.window_size]
                last_time = window_times[-1]

                # Label positive/negative samples based on time
                if is_positive_video:
                    if last_time >= event_time:
                        break
                    label = 1 if (event_time - 1.5 <= last_time < event_time) else 0
                else:
                    label = 0

                label_dir = "positive" if label == 1 else "negative"
                save_name = f"{base_name}_win{i:03d}_l{label}.{Config.video_format}"
                save_path = os.path.join(Config.output_dir, split, label_dir, save_name)
                
                # Add window data to buffer
                buffer.append({
                    "frames": window_frames.copy(),
                    "save_path": save_path,
                    "metadata": {
                        "path": os.path.join(split, label_dir, save_name),
                        "label": label,
                        "source": video_id,
                        "start_time": window_times[0],
                        "end_time": last_time
                    }
                })
                
                # When buffer reaches specified size, process and clear buffer
                if len(buffer) >= Config.buffer_size:
                    window_count += self._process_buffer(buffer)
                    buffer = []
                
                # Slide window and load more frames
                i += Config.window_stride
                while len(frames) < i + Config.window_size:
                    try:
                        frame, ts = next(frame_gen)
                        frames.append(frame)
                        timestamps.append(ts)
                    except StopIteration:
                        if len(frames) >= i + Config.window_size:
                            continue
                        else:
                            break
                if len(frames) < i + Config.window_size:
                    break

            # Process remaining buffer data
            if buffer:
                window_count += self._process_buffer(buffer)
                
            print(f"Processing complete: {video_id} generated {window_count} windows")
        except Exception as e:
            print(f"Processing failed {video_id}: {str(e)}")
            
    def _balance_dataset(self):
        print("\nStarting to balance dataset...")
        meta_df = pd.DataFrame(self.metadata)
        to_delete = []  # Delayed deletion
        for split in ["train", "val"]:
            pos_mask = (meta_df['label'] == 1) & (meta_df['path'].str.startswith(split))
            pos_samples = meta_df[pos_mask]
            neg_mask = (meta_df['label'] == 0) & (meta_df['path'].str.startswith(split))
            neg_samples = meta_df[neg_mask]
            sample_num = min(len(pos_samples), len(neg_samples))
            if sample_num == 0:
                continue
            if len(neg_samples) > sample_num:
                neg_selected = neg_samples.sample(sample_num, random_state=Config.seed)
                to_delete.extend(neg_samples[~neg_samples.index.isin(neg_selected.index)]['path'])
                meta_df = pd.concat([meta_df[~neg_mask], neg_selected])
        # Batch delete
        for path in to_delete:
            full_path = os.path.join(Config.output_dir, path)
            if os.path.exists(full_path):
                os.remove(full_path)
        
        meta_df.to_csv(os.path.join(Config.output_dir, "metadata.csv"), index=False)
        self.metadata = meta_df.to_dict('records')
        print("Dataset balancing complete")

    def _visualize_samples(self):
        plt.figure(figsize=(15, 5))
        samples = random.sample(self.metadata, min(3, len(self.metadata)))
        for idx, sample in enumerate(samples):
            full_path = os.path.join(Config.output_dir, sample['path'])
            
            # Load frames from video file
            cap = cv2.VideoCapture(full_path)
            ret, frame = cap.read()
            if ret:
                plt.subplot(1, 3, idx+1)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title(f"{'Positive sample' if sample['label'] else 'Negative sample'}\n{sample['source']}")
                plt.axis('off')
            cap.release()
                
        plt.tight_layout()
        plt.savefig(os.path.join(Config.output_dir, "sample_visualization.png"))
        plt.show()
        
    def _cleanup_resources(self):
        """Clean up all resources"""
        # Release all video writers
        for key in list(self.writer_cache.keys()):
            self.writer_cache[key].release()
        self.writer_cache.clear()

    def run(self):
        video_ids = self.df['id'].unique()
        train_ids, val_ids = train_test_split(video_ids, test_size=Config.test_size, random_state=Config.seed)
        
        with ThreadPoolExecutor(max_workers=Config.num_threads) as executor:
            futures = []
            for _, row in self.df.iterrows():
                futures.append(executor.submit(self._process_video, row=row, train_ids=train_ids))
            for future in tqdm(futures, desc="Processing videos", total=len(futures)):
                future.result()

        # Collect metadata from queue
        self.metadata = []
        while not self.metadata_queue.empty():
            self.metadata.append(self.metadata_queue.get())
        
        # Ensure all resources are released
        self._cleanup_resources()
        
        self._balance_dataset()
        if len(self.metadata) > 0:
            self._visualize_samples()
        print("All processing workflows complete")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()