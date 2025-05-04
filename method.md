# 基于VideoMAEv2的车辆碰撞预测方案

## 解决方案概述

嘿，我想跟大家分享下我在这个比赛中的方案。我采用了基于VideoMAEv2-giant预训练模型的深度学习方法来预测驾驶视频中的碰撞和险情风险。整个流程比较简单直接，在公开排行榜上取得了第4名的成绩（分数0.886）。

## 为什么选择VideoMAEv2？

我们认为车祸预测本质上可以看作一个视频分类问题 - 将视频分为"即将发生碰撞/险情"和"正常驾驶"两类。在尝试了几种不同的方法后，VideoMAEv2表现最好。

我们最初尝试了一些更传统的方法，如CNN+LSTM和CNN+Attention结构，使用相同的数据处理流程，但这些模型的得分大多在0.7-0.8之间，明显低于VideoMAEv2的表现。

VideoMAEv2是目前视频理解领域最先进的预训练模型之一，有以下几个优势：

1. **强大的时空特征提取能力**：能够捕捉视频中的动态信息，这对于预测碰撞至关重要
2. **大规模预训练**：在海量视频数据上预训练，有很好的泛化能力
3. **对动作识别任务表现出色**：车祸预测与动作识别有相似性，都需要理解视频中的动态变化

当然，选择这个模型也是因为它在各种视频理解基准测试上都取得了很好的成绩，我们希望它同样能在这个任务上表现出色。

## 技术实现

### 数据处理

我的数据处理流程如下：

- 首先从每个视频中提取帧序列，同时记录每帧的时间戳
- 使用滑动窗口方法，每个窗口包含16帧，窗口步长为2，这样相邻窗口之间有较大的重叠
- 对于标签分配，关键逻辑是：
  - 对于碰撞/险情视频(正样本视频)，只处理事件发生前的帧
  - 当窗口的最后一帧时间在事件发生前1.5秒内，将窗口标记为正样本，否则为负样本
  - 对于正常驾驶视频，所有窗口都标记为负样本
- 最后通过随机抽样确保训练和验证集中正负样本数量平衡

值得注意的是，虽然比赛提供了alert_time（人类标注的最早可预测时间点），但我们没有直接利用这个信息。我们假设深度学习模型可能比人类更能捕捉长时间的依赖关系，因此简单地使用了距离碰撞前1.5秒作为划分标准。实话说，这种做法不够严谨，因为我们没有进行充分的对比实验来验证不同时间阈值的效果。这是未来工作中可以改进的重要方向之一。

在数据平衡方面，我们最终得到了4258个正样本，并从原本更多的负样本中随机抽取了相同数量来保持平衡。由于我们的采样策略和滑动窗口方法，实际上产生的负样本数量远多于正样本。这种粗暴的平衡方法虽然简单有效，但可能丢弃了许多有意义的或容易与正样本混淆的负样本，这对模型学习区分边界情况可能不利。未来的改进可以考虑采用Focal Loss等方法来处理类别不平衡问题，而不是直接丢弃样本。

```python
# 标签分配的实际代码
if is_positive_video:
    if last_time >= event_time:
        break  # 如果超过事件时间就停止处理
    label = 1 if (event_time - 1.5 <= last_time < event_time) else 0
else:
    label = 0
```

### 模型架构

模型结构很简单，就是在VideoMAEv2-giant基础上加了个分类头：

- **骨干网络**：VideoMAEv2-giant作为特征提取器
- **分类头**：一个简单的线性层 + Dropout(0.1)
- **训练技巧**：应用了温度缩放(T=2.0)让概率分布更平滑

### 训练环境与策略

我在H20服务器上进行训练（96GB显存），不过这个方案在A100 40GB上也是可以完成训练的。如果想用更小的GPU，可以考虑使用更小规格的VideoMAE模型。

训练策略包括：
- 学习率：1e-5
- 优化器：AdamW + 余弦退火学习率调度
- 批次大小：实际批次3，通过梯度累积扩大到24
- 训练10个epoch取最佳验证结果

## 局限性与改进空间

老实说，由于学业压力和计算资源成本限制，这个方案还有不少可以改进的地方：

1. **固定的2秒16帧采样**：这个参数设置比较随意，没有进行系统的对比实验来确定最佳的窗口大小和采样率。不同长度的窗口可能对预测效果有显著影响。

2. **验证方法与评估不一致**：我们的验证方法与比赛主办方的测试评估方法不完全一致，导致我们可能没有选择到真正最优的模型提交。

3. **单模型而非集成**：虽然代码中有ensemble_predict的部分，但最终提交只使用了单个模型的结果，没有充分利用集成学习的潜力。

4. **更多数据增强**：可以尝试更多视频特定的数据增强方法，如时间扭曲、视频剪切等。

5. **更优的类别不平衡处理**：如前所述，简单的随机欠采样可能不是处理类别不平衡的最佳方法，未来可以尝试Focal Loss等更先进的方法。

总的来说，这个方案提供了一个不错的基线，但还有很大的改进空间！

# VideoMAE-based Vehicle Collision Prediction Solution

## Solution Overview

Hey everyone, I'd like to share my approach to this competition. I used a deep learning method based on the VideoMAEv2-giant pre-trained model to predict collision and near-miss risks in driving videos. The whole process is relatively straightforward and achieved a rank of 4th on the public leaderboard (score: 0.886).

## Why VideoMAEv2?

We consider accident prediction fundamentally as a video classification problem - dividing videos into "imminent collision/near-miss" and "normal driving" categories. After experimenting with several approaches, VideoMAEv2 performed the best.

We initially tried more traditional approaches like CNN+LSTM and CNN+Attention structures with the same data processing pipeline, but these models mostly scored between 0.7-0.8, significantly lower than VideoMAEv2's performance.

VideoMAEv2 is one of the most advanced pre-trained models in the field of video understanding, with several advantages:

1. **Strong spatio-temporal feature extraction**: Capable of capturing dynamic information in videos, which is crucial for collision prediction
2. **Large-scale pre-training**: Pre-trained on massive video data, offering excellent generalization capability
3. **Outstanding performance on action recognition tasks**: Collision prediction shares similarities with action recognition, both requiring understanding of dynamic changes in videos

Of course, choosing this model was also influenced by its excellent performance on various video understanding benchmarks, and we hoped it would perform equally well on this task.

## Technical Implementation

### Data Processing

My data processing pipeline works as follows:

- First extract frame sequences from each video, recording timestamps for each frame
- Use a sliding window approach with 16 frames per window and a stride of 2, creating significant overlap between adjacent windows
- For label assignment, the key logic is:
  - For collision/near-miss videos (positive videos), only process frames before the event occurs
  - When the last frame of a window is within 1.5 seconds before the event, label the window as positive, otherwise negative
  - For normal driving videos, all windows are labeled as negative
- Finally, ensure balanced positive and negative samples in training and validation sets through random sampling

It's worth noting that although the competition provided alert_time (human-annotated earliest prediction time), we didn't directly utilize this information. We assumed that deep learning models might capture longer temporal dependencies than humans, so we simply used 1.5 seconds before collision as our threshold. To be honest, this approach lacks rigor, as we didn't conduct sufficient comparative experiments to validate the effect of different time thresholds. This is one important direction for improvement in future work.

Regarding data balancing, we ended up with 4258 positive samples and randomly selected the same number of negative samples to maintain balance. Due to our sampling strategy and sliding window approach, the original number of negative samples was much larger than positives. This crude balancing method, while simple and effective, likely discarded many meaningful or easily confusable negative samples, which might be detrimental to the model's ability to learn boundary cases. Future improvements could consider using methods like Focal Loss to handle class imbalance rather than simply discarding samples.

```python
# Actual label assignment code
if is_positive_video:
    if last_time >= event_time:
        break  # Stop processing if past the event time
    label = 1 if (event_time - 1.5 <= last_time < event_time) else 0
else:
    label = 0
```

### Model Architecture

The model structure is simple, adding a classification head on top of VideoMAEv2-giant:

- **Backbone**: VideoMAEv2-giant as the feature extractor
- **Classification Head**: A simple linear layer + Dropout(0.1)
- **Training Technique**: Applied temperature scaling (T=2.0) for smoother probability distributions

### Training Environment and Strategy

I trained on an H20 server (96GB VRAM), though this solution could also be trained on an A100 40GB. For smaller GPUs, consider using smaller variants of the VideoMAE model.

Training strategy includes:
- Learning rate: 1e-5
- Optimizer: AdamW + cosine annealing learning rate scheduling
- Batch size: actual batch 3, enlarged to 24 through gradient accumulation
- Trained for 10 epochs, keeping the best validation result

## Limitations and Room for Improvement

To be honest, due to academic pressure and computational resource constraints, there are several aspects of this solution that could be improved:

1. **Fixed 2-second 16-frame sampling**: This parameter setting was somewhat arbitrary, without systematic comparative experiments to determine the optimal window size and sampling rate. Different window lengths might significantly impact prediction performance.

2. **Validation method inconsistent with evaluation**: Our validation method isn't perfectly aligned with the competition organizer's testing evaluation method, which might have led us to not select the truly optimal model for submission.

3. **Single model rather than ensemble**: Although there's an ensemble_predict component in the code, the final submission only used results from a single model, not fully leveraging the potential of ensemble learning.

4. **More data augmentation**: More video-specific data augmentation methods could be explored, such as temporal warping, video clips, etc.

5. **Better class imbalance handling**: As mentioned earlier, simple random undersampling may not be the optimal approach for handling class imbalance. Future work could explore more advanced methods like Focal Loss.

Overall, this solution provides a decent baseline, but there's still substantial room for improvement!
