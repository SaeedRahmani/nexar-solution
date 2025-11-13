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

# Prefer high-RAM GPU nodes (gpu001-009 have ~192GB RAM vs gpu010-013 with ~128GB)
# Uncomment the line below to only use high-RAM nodes:
#SBATCH --nodelist=gpu[001-009]

# Or request a specific node if you know it's fastest:
# #SBATCH --nodelist=gpu001

# Email notifications (update with your email!)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sra2157@columbia.edu

# ============================================
# Print Job Information
# ============================================
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================"
echo ""

# ============================================
# Load Environment
# ============================================
echo "Loading environment..."
module load miniconda3/latest
source activate nexar

# Verify Python environment
echo "Python: $(which python)"
python --version
echo ""

# ============================================
# GPU Information
# ============================================
echo "========================================"
echo "GPU Information"
echo "========================================"
nvidia-smi
echo ""
echo "Detailed GPU specs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version --format=csv
echo ""
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"
echo ""

# ============================================
# Navigate and Setup
# ============================================
cd /home/sra2157/git/nexar-solution
mkdir -p logs

# Verify critical files exist
if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found!"
    exit 1
fi

if [ ! -d "balanced_dataset_2s" ]; then
    echo "ERROR: balanced_dataset_2s directory not found!"
    exit 1
fi

echo "Dataset check:"
echo "  Train samples: $(find balanced_dataset_2s/train -name '*.mp4' -o -name '*.avi' -o -name '*.mov' | wc -l)"
echo "  Val samples: $(find balanced_dataset_2s/val -name '*.mp4' -o -name '*.avi' -o -name '*.mov' | wc -l)"
echo ""

# ============================================
# Run Training
# ============================================
echo "========================================"
echo "Starting Training"
echo "========================================"
echo "Command: python train.py"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Run the training script
python train.py

# Capture exit code
EXIT_CODE=$?

# ============================================
# Print Completion Information
# ============================================
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Completed Successfully!"
else
    echo "Training Failed with exit code: $EXIT_CODE"
fi
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "End Time: $(date)"
echo "========================================"

# Show saved model
if [ -f "best_videomae_large_model.pth" ]; then
    echo ""
    echo "Saved model:"
    ls -lh best_videomae_large_model.pth
fi

exit $EXIT_CODE
