# Columbia HPC Cluster Guide

## Quick Setup

### 1. Connect to Cluster

```bash
ssh <uni>@ginsburg.rcs.columbia.edu
```

### 2. Load Environment

```bash
module load miniconda3/latest
source activate nexar
```

### 3. Navigate to Project

```bash
cd /home/<uni>/git/nexar-solution
```

---

## Running Jobs

### Interactive Session (Testing/Debugging)

```bash
# Start a tmux session (persists if disconnected)
tmux new -s work

# Request GPU node interactively
srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=4:00:00 --pty bash

# Activate environment and run
source activate nexar
python training/train-large.py
```

**Detach tmux:** `Ctrl+B`, then `D`  
**Reattach tmux:** `tmux attach -t work`

### Batch Job (Long Training)

Create/edit `submit_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=nexar_train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --partition=gpuloq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gpu[001-009]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<uni>@columbia.edu

module load miniconda3/latest
source activate nexar
cd /home/<uni>/git/nexar-solution

echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
nvidia-smi

python training/train-large.py
```

Submit:

```bash
mkdir -p logs
sbatch submit_job.sh
```

---

## Job Management

| Command | Description |
|---------|-------------|
| `sbatch script.sh` | Submit batch job |
| `squeue -u $USER` | View your jobs |
| `scancel <job_id>` | Cancel job |
| `scancel -u $USER` | Cancel all your jobs |
| `tail -f logs/train-<id>.err` | Monitor job output |
| `sacct -j <job_id>` | Job details after completion |

---

## GPU Node Selection

### Available Nodes

| Nodes | GPUs | RAM | Recommendation |
|-------|------|-----|----------------|
| gpu001-009 | 8 GPUs | 192GB | **Best choice** - High RAM |
| gpu010 | 8 GPUs | 128GB | Good |
| gpu011-013 | 6 GPUs | 128GB | Fewer GPUs |

### Request Specific Nodes

```bash
# High-RAM nodes only
#SBATCH --nodelist=gpu[001-009]

# Specific node
#SBATCH --nodelist=gpu001

# Exclude slow nodes
#SBATCH --exclude=gpu[010-013]
```

### Check GPU Availability

```bash
sinfo -p gpuloq -N -l
```

### Check GPU Type on Node

```bash
srun -w gpu001 --gres=gpu:1 -p gpuloq nvidia-smi --query-gpu=name,memory.total --format=csv
```

---

## Resource Settings

### For VideoMAE-Large (10-11GB GPU)

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
```

### For VideoMAE-Giant (>24GB GPU)

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=gpu008    # Node with large GPU
```

### Multi-GPU Training

```bash
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
```

---

## Common Tasks

### Training

```bash
# Interactive
srun --gres=gpu:1 --mem=64G --cpus-per-task=16 --time=4:00:00 --pty bash
source activate nexar
python training/train-large.py

# Batch
sbatch submit_job.sh
```

### Prediction

```bash
# Interactive (usually quick)
srun --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=1:00:00 --pty bash
source activate nexar
python prediction/predict_agg_from_val.py
```

### Analysis (No GPU needed)

```bash
# Interactive without GPU
srun --mem=16G --cpus-per-task=4 --time=1:00:00 --pty bash
source activate nexar
python prediction_analysis/scripts/run_all_analyses.py --dataset BOTH
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Job pending forever | Check `sinfo -p gpuloq` for availability |
| CUDA OOM | Reduce batch_size or request larger GPU |
| Job killed | Increase `--time` or `--mem` |
| Connection lost | Use `tmux` to preserve sessions |
| Can't find conda | Run `module load miniconda3/latest` |
| Package missing | `pip install -r requirements.txt` |

### Check Job Failure Reason

```bash
sacct -j <job_id> --format=JobID,State,ExitCode,MaxRSS,Elapsed
tail -100 logs/train-<job_id>.err
```

---

## File Transfer

### Upload to Cluster

```bash
# From local machine
scp -r local_folder/ <uni>@ginsburg.rcs.columbia.edu:/home/<uni>/git/nexar-solution/
```

### Download from Cluster

```bash
# From local machine
scp -r <uni>@ginsburg.rcs.columbia.edu:/home/<uni>/git/nexar-solution/results/ ./local_results/
```

---

## Quick Reference Card

```bash
# Connect
ssh <uni>@ginsburg.rcs.columbia.edu

# Setup
module load miniconda3/latest && source activate nexar
cd /home/<uni>/git/nexar-solution

# Interactive GPU
srun --gres=gpu:1 --mem=64G --cpus-per-task=16 --time=4:00:00 --pty bash

# Submit job
sbatch submit_job.sh

# Monitor
squeue -u $USER
tail -f logs/train-*.err

# Cancel
scancel <job_id>
```
