# Nexar Safe Driving Video Analysis Competition Solution

[中文版](#中文版) | [English](#english)

## English

# VideoMAE-based Vehicle Collision Prediction Solution: 4th Place on Kaggle Public Leaderboard (Score 0.886)

## Solution Overview

Hey everyone,

I'd like to share my approach for the Nexar Safe Driving Video Analysis competition. This solution uses a deep learning method based on the **VideoMAEv2-giant** pre-trained model to predict collision and near-miss risks in driving videos.

The process was relatively straightforward, and it achieved **4th place on the public leaderboard** with a score of **0.886**.

## Why VideoMAEv2?

We framed the accident prediction task as a video classification problem: classifying videos into "imminent collision/near-miss" or "normal driving". After experimenting with several approaches, VideoMAEv2 delivered the best performance.

*   **Initial Attempts**: We first tried more traditional architectures like CNN+LSTM and CNN+Attention, using the same data processing pipeline. However, these models generally scored between 0.7 and 0.8, falling short of VideoMAEv2.

*   **VideoMAEv2 Advantages**: As a state-of-the-art pre-trained model for video understanding, VideoMAEv2 offers several key benefits:
    1.  **Strong Spatio-temporal Feature Extraction**: It excels at capturing the dynamic information crucial for collision prediction.
    2.  **Large-Scale Pre-training**: Pre-training on massive video datasets provides excellent generalization capabilities.
    3.  **Action Recognition Performance**: Its proven success in action recognition tasks suggests suitability for understanding dynamic events like near-misses and collisions.

Its strong performance on various video benchmarks further motivated its selection for this task.

## Technical Implementation

### Data Processing

Our data processing pipeline involved these steps:

1.  **Frame Extraction & Timestamps**: Extract frame sequences and corresponding timestamps from each video.
2.  **Sliding Window**: Apply a sliding window approach:
    *   Window Size: 16 frames.
    *   Stride: 2 frames (resulting in significant overlap).
3.  **Label Assignment**: Assign labels based on the following logic:
    *   For positive videos (collisions/near-misses), only process frames *before* the event time.
    *   A window is labeled **positive (1)** if its last frame's timestamp falls within 1.5 seconds *before* the event time.
    *   Otherwise (including all windows from normal driving videos), the window is labeled **negative (0)**.
4.  **Data Balancing**: Randomly undersample the negative class to match the number of positive samples in both training and validation sets.

**Important Considerations**:

*   **Ignoring `alert_time`**: The competition provided `alert_time` (human-annotated earliest prediction time). We didn't directly use it, hypothesizing that a deep learning model might capture longer-term dependencies better than human annotation. Instead, we opted for the fixed 1.5-second threshold before the event. Frankly, this lacks empirical rigor, as we didn't run experiments comparing different thresholds. This is a key area for future improvement.
*   **Data Balancing Details**: This process yielded 4,258 positive samples, and we matched this count with randomly selected negative samples. Our sliding window approach generated far more negative than positive windows initially. This aggressive undersampling, while simple, likely discarded many potentially informative or hard-negative examples, possibly hindering the model's ability to learn fine-grained distinctions. Exploring techniques like Focal Loss could be beneficial.

```python
# Core label assignment logic
if is_positive_video:
    if last_time >= event_time:
        break  # Stop processing if past the event time
    label = 1 if (event_time - 1.5 <= last_time < event_time) else 0
else:
    label = 0
```

### Model Architecture

The model architecture is straightforward:

*   **Backbone**: VideoMAEv2-giant pre-trained model.
*   **Classification Head**: A simple linear layer followed by Dropout (p=0.1).
*   **Training Technique**: Temperature scaling (T=2.0) applied to logits during training for smoother probability distributions.

### Inference Method

For the test set, we used a simple yet effective inference strategy:

*   **Extract only the last 2 seconds** of each test video.
*   Feed this 2-second clip into the trained model for prediction.

This approach relies on the intuition that the most critical information for predicting an imminent event is likely concentrated towards the end of the video clip (especially since test videos were trimmed). This simplification performed well, likely because:

1.  Positive samples (trimmed near events) have strong signals in the final moments.
2.  Negative samples (normal driving) shouldn't have strong signals regardless of the segment.

While testing various window lengths would be ideal, this focused approach reduced computational load and aligned with the intuition that later predictions tend to be more accurate for imminent events. It yielded good results despite its simplicity.

### Training Environment and Strategy

*   **Hardware**: Trained on an H20 server (96GB VRAM). Training is also feasible on an A100 (40GB). Smaller GPUs might require using smaller VideoMAE variants.
*   **Reproducibility**: Fixed random seed (`seed=42`) for Python, NumPy, and PyTorch (including CUDA) ensures deterministic runs.
*   **Key Hyperparameters**:
    *   Optimizer: AdamW
    *   Learning Rate: 1e-5 (with Cosine Annealing scheduler)
    *   Batch Size: 3 (effective batch size 24 due to gradient accumulation over 8 steps)
    *   Epochs: 10 (saving the best model based on validation accuracy)

## Limitations and Room for Improvement

Given academic and resource constraints, this solution has several limitations and areas for future work:

1.  **Fixed Sampling Strategy**: The 2-second window (16 frames) was chosen somewhat arbitrarily. Systematic experiments comparing different window lengths and frame rates are needed.
2.  **Validation Mismatch**: Our local validation setup didn't perfectly mirror the competition's evaluation metric (Mean Average Precision over time-to-accident thresholds). This might mean the submitted model wasn't the true optimal one based on the leaderboard metric.
3.  **Single Model Submission**: Although the prediction script includes ensemble capabilities, the final submission relied on a single model, missing potential gains from ensembling.
4.  **Limited Data Augmentation**: More sophisticated video-specific augmentations (e.g., temporal jittering, different cropping strategies) could be explored.
5.  **Crude Class Imbalance Handling**: As noted, simple undersampling might not be optimal. Exploring methods like Focal Loss or more intelligent negative mining could improve performance.

## Code and Model Weights

All code has been uploaded to this GitHub repository. 

### Model and Training Logs Access

The model weights and training logs are available in this repository and also from:

- **GitHub**: In this repository under the `models/` and `logs/` directories
- **Hugging Face**: Due to GitHub's file size limits, the full model weights are also available at [Hugging Face Space](https://huggingface.co/spaces/zhiyaowang/VideoMaev2-giant-nexar-solution)

**Note**: The model weights are quite large (~2.7GB). If you're interested in reproducing the results, you can either download the weights or follow the data processing and training scripts with the fixed random seed (`seed=42`).

## Repository Structure

- `train.py`: Main training script for the VideoMAEv2 model
- `process.py`: Data processing pipeline for preparing the training and validation datasets
- `ensemble_predict.py`: Inference script supporting single and ensemble model predictions
- `method.md`: Detailed description of the solution methodology
- `models/`: Directory containing the trained model weights
- `logs/`: Directory containing training logs
- `best.pth`: The best model checkpoint that achieved 0.886 on the public leaderboard

---

Achieving a decent rank with what feels like a somewhat unrefined and insufficiently experimented approach suggests either a good dose of luck, or perhaps, significant untapped potential in the VideoMAEv2 model for this specific task.

## 中文版

# 基于VideoMAE的车辆碰撞预测解决方案：Kaggle公开排行榜第4名（分数0.886）

## 解决方案概述

大家好，

我想分享一下我在Nexar安全驾驶视频分析比赛中的解决方案。这个方案使用了基于**VideoMAEv2-giant**预训练模型的深度学习方法来预测驾驶视频中的碰撞和险些碰撞风险。

整个过程相对简单，并在公开排行榜上获得了**第4名**，分数为**0.886**。

## 为什么选择VideoMAEv2？

我们将事故预测任务框架化为一个视频分类问题：将视频分为"即将发生碰撞/险些碰撞"或"正常驾驶"。在尝试了几种方法后，VideoMAEv2提供了最佳性能。

* **初始尝试**：我们首先尝试了更传统的架构，如CNN+LSTM和CNN+Attention，使用相同的数据处理流程。然而，这些模型的得分通常在0.7到0.8之间，低于VideoMAEv2。

* **VideoMAEv2优势**：作为视频理解领域最先进的预训练模型，VideoMAEv2提供了几个关键优势：
    1. **强大的时空特征提取**：它擅长捕捉对碰撞预测至关重要的动态信息。
    2. **大规模预训练**：在海量视频数据集上的预训练提供了出色的泛化能力。
    3. **动作识别性能**：它在动作识别任务上的成功表明适合理解险些碰撞和碰撞等动态事件。

它在各种视频基准测试上的强大表现进一步促使我们选择它用于这项任务。

## 技术实现

### 数据处理

我们的数据处理流程包括以下步骤：

1. **帧提取与时间戳**：从每个视频中提取帧序列和相应的时间戳。
2. **滑动窗口**：应用滑动窗口方法：
   * 窗口大小：16帧。
   * 步长：2帧（导致显著重叠）。
3. **标签分配**：基于以下逻辑分配标签：
   * 对于正样本视频（碰撞/险些碰撞），只处理事件时间*之前*的帧。
   * 如果窗口的最后一帧时间戳落在事件时间*之前*的1.5秒内，则标记为**正样本(1)**。
   * 否则（包括所有来自正常驾驶视频的窗口），该窗口被标记为**负样本(0)**。
4. **数据平衡**：随机对负类进行欠采样，使其与训练集和验证集中的正样本数量匹配。

**重要考虑因素**：

* **忽略`alert_time`**：比赛提供了`alert_time`（人工标注的最早预测时间）。我们没有直接使用它，假设深度学习模型可能比人工标注更好地捕捉长期依赖关系。相反，我们选择了事件前固定的1.5秒阈值。坦率地说，这缺乏实证严谨性，因为我们没有运行比较不同阈值的实验。这是未来改进的一个关键领域。
* **数据平衡细节**：这个过程产生了4,258个正样本，我们用随机选择的负样本匹配了这个数量。我们的滑动窗口方法最初生成了远多于正样本的负样本窗口。这种激进的欠采样虽然简单，但可能丢弃了许多潜在有信息或难负样本的例子，可能阻碍了模型学习细粒度区分的能力。探索像Focal Loss这样的技术可能会有所帮助。

```python
# 核心标签分配逻辑
if is_positive_video:
    if last_time >= event_time:
        break  # 如果超过事件时间则停止处理
    label = 1 if (event_time - 1.5 <= last_time < event_time) else 0
else:
    label = 0
```

### 模型架构

模型架构相当简单：

* **骨干网络**：VideoMAEv2-giant预训练模型。
* **分类头**：简单的线性层，后接Dropout（p=0.1）。
* **训练技术**：在训练期间对logits应用温度缩放（T=2.0），以获得更平滑的概率分布。

### 推理方法

对于测试集，我们使用了一种简单但有效的推理策略：

* **仅提取每个测试视频的最后2秒**。
* 将这个2秒的片段输入到训练好的模型中进行预测。

这种方法基于这样的直觉：预测即将发生的事件的最关键信息可能集中在视频片段的末尾（尤其是因为测试视频被修剪过）。这种简化表现良好，可能是因为：

1. 正样本（靠近事件修剪）在最后时刻有强烈的信号。
2. 负样本（正常驾驶）无论在哪个片段都不应该有强烈的信号。

虽然测试各种窗口长度会是理想的做法，但这种集中的方法减少了计算负担，并与后期预测往往对即将发生的事件更准确的直觉相符。尽管它很简单，但产生了良好的结果。

### 训练环境和策略

* **硬件**：在H20服务器（96GB VRAM）上训练。在A100（40GB）上训练也是可行的。较小的GPU可能需要使用更小的VideoMAE变体。
* **可重复性**：为Python、NumPy和PyTorch（包括CUDA）固定随机种子（`seed=42`）确保确定性运行。
* **关键超参数**：
   * 优化器：AdamW
   * 学习率：1e-5（使用余弦退火调度器）
   * 批量大小：3（由于在8个步骤上进行梯度累积，有效批量大小为24）
   * 轮次：10（基于验证准确率保存最佳模型）

## 限制和改进空间

鉴于学术和资源限制，这个解决方案有几个限制和未来工作的领域：

1. **固定采样策略**：2秒窗口（16帧）的选择有些随意。需要系统性地实验比较不同的窗口长度和帧率。
2. **验证不匹配**：我们的本地验证设置与比赛的评估指标（时间到事故阈值上的平均精度均值）不完全匹配。这可能意味着提交的模型基于排行榜指标并非真正的最优模型。
3. **单模型提交**：尽管预测脚本包含集成功能，但最终提交依赖于单个模型，错失了集成可能带来的潜在收益。
4. **有限的数据增强**：可以探索更复杂的视频特定增强（例如，时间抖动，不同的裁剪策略）。
5. **粗糙的类不平衡处理**：如前所述，简单的欠采样可能不是最优的。探索像Focal Loss或更智能的负样本挖掘等方法可能会提高性能。

## 代码和模型权重

所有代码已上传到这个GitHub仓库。

### 模型和训练日志访问

模型权重和训练日志可在此仓库中获取，也可从以下位置获取：

- **GitHub**：在本仓库的`models/`和`logs/`目录下
- **Hugging Face**：由于GitHub的文件大小限制，完整的模型权重也可在[Hugging Face Space](https://huggingface.co/spaces/zhiyaowang/VideoMaev2-giant-nexar-solution)获取

**注意**：模型权重相当大（约2.7GB）。如果您有兴趣重现结果，您可以下载权重或使用固定随机种子（`seed=42`）遵循数据处理和训练脚本。

## 仓库结构

- `train.py`：VideoMAEv2模型的主要训练脚本
- `process.py`：准备训练和验证数据集的数据处理流程
- `ensemble_predict.py`：支持单模型和集成模型预测的推理脚本
- `method.md`：解决方案方法的详细描述
- `models/`：包含训练好的模型权重的目录
- `logs/`：包含训练日志的目录
- `best.pth`：在公开排行榜上获得0.886分数的最佳模型检查点

---

以一种感觉有些不够精细和实验不足的方法获得体面的排名，要么是运气好，要么是VideoMAEv2模型在这个特定任务上有显著的未开发潜力。
