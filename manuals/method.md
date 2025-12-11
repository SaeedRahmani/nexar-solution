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

All code has been uploaded to the GitHub repository:
[https://github.com/wzyfromhust/nexer-solution](https://github.com/wzyfromhust/nexer-solution)

**Note**: The final model weights are quite large and will be uploaded later when feasible. However, the code includes the fixed random seed (`seed=42`), so reproducing the results by following the data processing and training scripts should be relatively straightforward.

---

Achieving a decent rank with what feels like a somewhat unrefined and insufficiently experimented approach suggests either a good dose of luck, or perhaps, significant untapped potential in the VideoMAEv2 model for this specific task.
