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

# 配置参数
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
    video_format = "avi"  # 改回更广泛支持的avi格式
    codec = "XVID"        # XVID编码器在大多数系统上都可用
    num_threads = min(os.cpu_count() * 2, 16)  # 动态线程数
    max_videos = None
    # 添加内存缓冲区大小配置
    buffer_size = 48  # 在内存中保存的窗口数量

class VideoProcessor:
    def __init__(self):
        self.df = pd.read_csv(Config.csv_path)
        if Config.max_videos:
            self.df = self.df.head(Config.max_videos)
        self._init_directories()
        self.metadata_queue = queue.Queue()  # 异步元数据队列
        self.lock = threading.Lock()
        # 添加全局视频写入器缓存
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
        """改为生成器，逐帧生成"""
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
        """获取或创建视频写入器"""
        cache_key = os.path.abspath(save_path)
        
        # 检查缓存中是否已有该路径的writer
        if cache_key in self.writer_cache:
            return self.writer_cache[cache_key], True
        
        # 创建新的writer
        fourcc = cv2.VideoWriter_fourcc(*Config.codec)
        writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
        
        # 检查writer是否成功初始化
        if not writer.isOpened():
            # 尝试备用编码器
            backup_codecs = ["MJPG", "DIVX", "MPEG", "X264"]
            for codec in backup_codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
                if writer.isOpened():
                    break
        
        # 如果所有编码器都失败，使用未压缩格式
        if not writer.isOpened():
            writer = cv2.VideoWriter(save_path, 0, Config.target_fps, Config.img_size)
            
        # 缓存writer
        self.writer_cache[cache_key] = writer
        return writer, False

    def _release_writer(self, save_path):
        """释放指定路径的写入器"""
        cache_key = os.path.abspath(save_path)
        if cache_key in self.writer_cache:
            self.writer_cache[cache_key].release()
            del self.writer_cache[cache_key]

    def _save_video_window(self, frames, save_path):
        """使用缓存的 VideoWriter"""
        try:
            writer, _ = self._get_video_writer(save_path)
            
            # 批量写入帧以提高效率
            for frame in frames:
                writer.write(frame)
            
            # 视频写入完成后释放写入器
            self._release_writer(save_path)
            return True
        except Exception as e:
            print(f"保存视频时发生错误: {str(e)}")
            return False

    def _process_buffer(self, buffer):
        """处理缓冲区中的窗口数据"""
        processed_count = 0
        for item in buffer:
            # 保存为视频文件
            save_success = self._save_video_window(item["frames"], item["save_path"])
            
            # 只有保存成功才记录元数据
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
            
            # 先加载初始帧
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

            # 使用内存缓冲区存储窗口数据
            buffer = []
            window_count = 0
            i = 0
            
            while True:
                # 获取当前窗口的帧和时间
                window_frames = frames[i:i + Config.window_size]
                window_times = timestamps[i:i + Config.window_size]
                last_time = window_times[-1]

                # 根据时间标记正负样本
                if is_positive_video:
                    if last_time >= event_time:
                        break
                    label = 1 if (event_time - 1.5 <= last_time < event_time) else 0
                else:
                    label = 0

                label_dir = "positive" if label == 1 else "negative"
                save_name = f"{base_name}_win{i:03d}_l{label}.{Config.video_format}"
                save_path = os.path.join(Config.output_dir, split, label_dir, save_name)
                
                # 将窗口数据添加到缓冲区
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
                
                # 当缓冲区达到指定大小时，处理并清空缓冲区
                if len(buffer) >= Config.buffer_size:
                    window_count += self._process_buffer(buffer)
                    buffer = []
                
                # 滑动窗口并加载更多帧
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

            # 处理剩余的缓冲区数据
            if buffer:
                window_count += self._process_buffer(buffer)
                
            print(f"处理完成: {video_id} 生成 {window_count} 个窗口")
        except Exception as e:
            print(f"处理失败 {video_id}: {str(e)}")
            
    def _balance_dataset(self):
        print("\n开始平衡数据集...")
        meta_df = pd.DataFrame(self.metadata)
        to_delete = []  # 延迟删除
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
        # 批量删除
        for path in to_delete:
            full_path = os.path.join(Config.output_dir, path)
            if os.path.exists(full_path):
                os.remove(full_path)
        
        meta_df.to_csv(os.path.join(Config.output_dir, "metadata.csv"), index=False)
        self.metadata = meta_df.to_dict('records')
        print("数据集平衡完成")

    def _visualize_samples(self):
        plt.figure(figsize=(15, 5))
        samples = random.sample(self.metadata, min(3, len(self.metadata)))
        for idx, sample in enumerate(samples):
            full_path = os.path.join(Config.output_dir, sample['path'])
            
            # 从视频文件加载帧
            cap = cv2.VideoCapture(full_path)
            ret, frame = cap.read()
            if ret:
                plt.subplot(1, 3, idx+1)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title(f"{'正样本' if sample['label'] else '负样本'}\n{sample['source']}")
                plt.axis('off')
            cap.release()
                
        plt.tight_layout()
        plt.savefig(os.path.join(Config.output_dir, "sample_visualization.png"))
        plt.show()
        
    def _cleanup_resources(self):
        """清理所有资源"""
        # 释放所有视频写入器
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
            for future in tqdm(futures, desc="处理视频", total=len(futures)):
                future.result()

        # 从队列收集元数据
        self.metadata = []
        while not self.metadata_queue.empty():
            self.metadata.append(self.metadata_queue.get())
        
        # 确保所有资源被释放
        self._cleanup_resources()
        
        self._balance_dataset()
        if len(self.metadata) > 0:
            self._visualize_samples()
        print("处理流程全部完成")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()