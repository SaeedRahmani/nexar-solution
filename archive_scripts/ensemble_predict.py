import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import csv
from torch.cuda.amp import autocast
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import shutil

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置参数
class Config:
    # 输入输出路径
    test_video_dir = "/root/autodl-tmp/dataset/test"  # 测试视频目录
    output_video_dir = "processed_videos"  # 处理后的视频保存目录
    final_output_csv = "ensemble_submission.csv"  # 最终结果

    # 视频处理参数
    target_fps = 10  # 目标帧率
    img_size = (320, 320)  # 处理视频时的图像尺寸
    num_threads = 8  # 视频处理线程数

    # 视频窗口设置 (都是16帧，但时间长度不同)
    window_configs = [
        {"name": "2s", "duration": 2.0, "model_path": "/root/autodl-tmp/nexar-solution/best.pth"}
    ]
    
    # 模型参数
    frame_count = 16  # 所有模型都使用16帧
    model_img_size = 224  # VideoMAE模型输入尺寸
    batch_size = 3
    temperature = 2.0  # 温度缩放参数，与训练一致
    video_format = "avi"  # 保存格式
    codec = "XVID"  # 编码器

# 视频处理类
class VideoProcessor:
    def __init__(self):
        self._init_directories()
    
    def _init_directories(self):
        # 为每个窗口配置创建目录
        for config in Config.window_configs:
            window_dir = os.path.join(Config.output_video_dir, config["name"])
            os.makedirs(window_dir, exist_ok=True)
            # 清空目录中的旧文件
            for f in os.listdir(window_dir):
                file_path = os.path.join(window_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _extract_frames(self, video_path):
        """提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(fps / Config.target_fps)))
        
        # 存储所有帧和时间戳
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
            print(f"视频无有效帧: {video_path}")
            return None, None
            
        return all_frames, all_timestamps
    
    def _save_video_window(self, frames, save_path):
        """保存帧序列为视频"""
        fourcc = cv2.VideoWriter_fourcc(*Config.codec)
        writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
        
        if not writer.isOpened():
            # 尝试备用编码器
            backup_codecs = ["MJPG", "DIVX", "MPEG", "X264"]
            for codec in backup_codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(save_path, fourcc, Config.target_fps, Config.img_size)
                if writer.isOpened():
                    break
        
        if not writer.isOpened():
            print(f"无法创建视频写入器: {save_path}")
            return False
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        return True
    
    def process_video(self, video_file):
        """处理单个测试视频，提取不同时间窗口"""
        try:
            video_path = os.path.join(Config.test_video_dir, video_file)
            video_id = os.path.splitext(video_file)[0]
            
            # 提取所有帧
            all_frames, all_timestamps = self._extract_frames(video_path)
            if all_frames is None:
                return
            
            video_duration = all_timestamps[-1]
            
            # 为每个配置提取不同的时间窗口
            for config in Config.window_configs:
                window_duration = config["duration"]
                window_name = config["name"]
                
                # 计算窗口的起止时间，从视频末尾往前取指定秒数
                end_time = video_duration
                start_time = max(0, end_time - window_duration)
                
                # 选择窗口内的帧
                window_frames = []
                for i, ts in enumerate(all_timestamps):
                    if start_time <= ts <= end_time:
                        window_frames.append(all_frames[i])
                
                # 确保正好有16帧
                if len(window_frames) > Config.frame_count:
                    # 均匀抽取16帧
                    indices = np.linspace(0, len(window_frames)-1, Config.frame_count, dtype=int)
                    window_frames = [window_frames[i] for i in indices]
                elif len(window_frames) < Config.frame_count:
                    # 如果帧数不足，复制最后一帧
                    while len(window_frames) < Config.frame_count:
                        window_frames.append(window_frames[-1] if window_frames else np.zeros_like(all_frames[0]))
                
                # 保存为视频文件
                save_dir = os.path.join(Config.output_video_dir, window_name)
                save_path = os.path.join(save_dir, f"{video_id}.{Config.video_format}")
                if not self._save_video_window(window_frames, save_path):
                    print(f"保存视频失败: {save_path}")
            
            print(f"处理完成: {video_id}")
            
        except Exception as e:
            print(f"处理视频 {video_file} 时出错: {str(e)}")
    
    def run(self):
        """处理所有测试视频"""
        video_files = [f for f in os.listdir(Config.test_video_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print("错误：测试目录中没有找到任何视频文件！")
            return
        
        print(f"找到 {len(video_files)} 个测试视频，开始处理...")
        
        with ThreadPoolExecutor(max_workers=Config.num_threads) as executor:
            list(tqdm(executor.map(self.process_video, video_files), 
                     total=len(video_files), desc="处理测试视频"))
        
        print(f"视频处理完成，输出目录: {Config.output_video_dir}")

# 定义 VideoMAEv2 模型（与训练代码一致）
class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VideoMAEClassifier, self).__init__()
        
        # 加载 VideoMAEv2-giant 的配置和处理器
        config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-giant", trust_remote_code=True)
        self.processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-giant")
        
        # 尝试设置 drop_path_rate (与 train.py 一致)
        drop_path_value = 0.1 # 与 train.py 保持一致
        if hasattr(config, 'model_config') and isinstance(config.model_config, dict) and 'drop_path_rate' in config.model_config:
            config.model_config['drop_path_rate'] = drop_path_value
            print(f"成功设置 config.model_config['drop_path_rate'] = {config.model_config['drop_path_rate']}")
        elif hasattr(config, 'drop_path_rate'): # 备用检查
            config.drop_path_rate = drop_path_value
            print(f"成功设置 config.drop_path_rate = {config.drop_path_rate}")
        else:
            print("警告: 未能在 config 中找到或设置 drop_path_rate 参数。")

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                "OpenGVLab/VideoMAEv2-giant",
                config=config,
                trust_remote_code=True
            ).to(device)
            print("成功加载 VideoMAEv2-giant 预训练权重")
        else:
            self.backbone = AutoModel.from_config(config, trust_remote_code=True).to(device)
            print("使用随机初始化的 VideoMAEv2-giant 模型")

        # 动态获取特征维度
        with torch.no_grad():
            dummy_input = torch.rand(1, 3, Config.frame_count, Config.model_img_size, Config.model_img_size).to(device)
            dummy_outputs = self.backbone(pixel_values=dummy_input)
            feature_dim = dummy_outputs.shape[-1]
        
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: (B, T, C, H, W) tensor，转换为 (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).to(device, dtype=torch.float32)  # (B, C, T, H, W)
        inputs = {"pixel_values": x}
        outputs = self.backbone(**inputs)
        features = outputs  # 直接使用二维张量
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

# 视频预测数据集
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
        
        # 打开视频并验证
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频 {video_path}")
            return video_id, torch.zeros((self.frame_count, 3, Config.model_img_size, Config.model_img_size))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果视频帧数不足，打印警告并调整
        if total_frames != self.frame_count:
            print(f"警告：视频 {video_path} 的帧数为 {total_frames}，预期为 {self.frame_count}，将调整帧数")
        
        # 读取帧
        frames = []
        frame_idx = 0
        while len(frames) < self.frame_count:
            ret, frame = cap.read()
            if not ret:
                # 如果帧数不足，用黑帧填充
                frame = np.zeros((Config.model_img_size, Config.model_img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += 1
            
            # 如果视频帧数超过，只取前 16 帧
            if frame_idx >= self.frame_count:
                break
        
        cap.release()
        
        # 确保帧数正好为 16
        if len(frames) < self.frame_count:
            frames.extend([np.zeros((Config.model_img_size, Config.model_img_size, 3), dtype=np.uint8)] * (self.frame_count - len(frames)))
        frames = frames[:self.frame_count]
        
        # 预处理帧
        processed = torch.stack([self.transform(image=f)["image"] for f in frames])  # (T, C, H, W)
        return video_id, processed

# 温度缩放函数
def apply_temperature_scaling(logits, temperature=2.0):
    """温度缩放，与训练时保持一致"""
    return logits / temperature

# 预测类
class Predictor:
    def __init__(self):
        # 数据预处理（与训练时的 val_transform 一致）
        self.transform = A.Compose([
            A.Resize(Config.model_img_size, Config.model_img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 存储各个模型的预测结果
        self.all_predictions = {}
    
    def load_model(self, model_path):
        """加载指定路径的模型"""
        model = VideoMAEClassifier(num_classes=2, pretrained=True).to(device)
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f"模型权重加载成功：{model_path}")
        except FileNotFoundError:
            print(f"错误：无法找到模型文件 '{model_path}'，请确保文件存在！")
            return None
        except KeyError:
            print("错误：权重文件中缺少 'model' 键，检查保存格式！")
            return None
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return None
        
        model.eval()
        return model
    
    def predict_videos(self, window_config):
        """对指定窗口配置的视频进行预测"""
        window_name = window_config["name"]
        model_path = window_config["model_path"]
        
        print(f"\n开始处理 {window_name} 窗口的视频...")
        
        # 加载模型
        model = self.load_model(model_path)
        if model is None:
            return
        
        # 获取视频文件
        video_dir = os.path.join(Config.output_video_dir, window_name)
        video_files = [
            os.path.join(video_dir, f) 
            for f in os.listdir(video_dir) 
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        
        if not video_files:
            print(f"错误：{video_dir} 目录中没有找到任何视频文件！")
            return
        
        # 创建数据加载器
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
        
        # 执行预测
        results = {}
        with torch.no_grad():
            for batch_ids, batch_frames in tqdm(dataloader, desc=f"预测 {window_name} 窗口"):
                # 移至设备
                inputs = batch_frames.to(device)  # (B, T, C, H, W)
                
                # 前向传播，使用混合精度
                with autocast():
                    outputs = model(inputs)
                    outputs_scaled = apply_temperature_scaling(outputs, temperature=Config.temperature)
                
                # 计算概率
                scores = torch.softmax(outputs_scaled, dim=1)[:, 1].cpu().numpy()
                
                # 存储结果
                for vid_id, score in zip(batch_ids, scores):
                    results[vid_id] = score
        
        # 保存这个窗口配置的预测结果
        self.all_predictions[window_name] = results
        print(f"{window_name} 窗口预测完成，共 {len(results)} 个视频")
        
        # 单独保存这个模型的预测结果（可选）
        output_csv = f"prediction_{window_name}.csv"
        with open(output_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for vid_id, score in results.items():
                writer.writerow([vid_id, f"{score:.4f}"])
        print(f"单模型预测结果已保存至 {output_csv}")
    
    def ensemble_predictions(self):
        """集成所有模型的预测结果"""
        if not self.all_predictions:
            print("错误：没有可用的预测结果进行集成！")
            return
        
        # 获取所有视频ID
        all_video_ids = set()
        for predictions in self.all_predictions.values():
            all_video_ids.update(predictions.keys())
        
        # 计算平均预测分数
        ensemble_results = {}
        for vid_id in all_video_ids:
            scores = []
            for window_name, predictions in self.all_predictions.items():
                if vid_id in predictions:
                    scores.append(predictions[vid_id])
            
            if scores:
                ensemble_results[vid_id] = np.mean(scores)
            else:
                print(f"警告：视频 {vid_id} 没有有效的预测分数")
                ensemble_results[vid_id] = 0.5  # 默认分数
        
        # 保存集成结果
        with open(Config.final_output_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for vid_id, score in ensemble_results.items():
                writer.writerow([vid_id, f"{score:.4f}"])
        
        print(f"\n所有模型集成完成！结果已保存至 {Config.final_output_csv}")
        print(f"共处理 {len(ensemble_results)} 个视频")
    
    def run(self):
        """运行所有窗口配置的预测并集成结果"""
        for window_config in Config.window_configs:
            self.predict_videos(window_config)
        
        self.ensemble_predictions()

# 主程序
def main():
    #print("开始处理测试视频...")
    #processor = VideoProcessor()
    #processor.run()
    
    print("\n开始模型预测...")
    predictor = Predictor()
    predictor.run()
    
    print("\n集成预测流程全部完成!")

if __name__ == "__main__":
    main() 