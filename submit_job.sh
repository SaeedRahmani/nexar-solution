#!/bin/bash
#SBATCH --job-name=nexar_train       # Job name
#SBATCH --output=logs/train-%j.out   # Output log file (%j = job ID)
#SBATCH --error=logs/train-%j.err    # Error log file
#SBATCH --partition=gpuloq           # GPU partition
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --gres=gpu:1                 # Request 1 GPU (change to gpu:8 for all GPUs)
#SBATCH --mem=64G                    # Memory per node
#SBATCH --time=24:00:00              # Time limit (24 hours)
#SBATCH --mail-type=ALL              # Email notifications for all events
#SBATCH --mail-user=your.email@example.com  # Your email

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Load modules and activate environment
module load miniconda3/latest
source activate nexar

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi
echo ""

# Navigate to project directory
cd /home/sra2157/git/nexar-solution

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training script
echo "Starting training..."
echo "================================"
python train.py

# Print completion time
echo ""
echo "================================"
echo "Training completed!"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
