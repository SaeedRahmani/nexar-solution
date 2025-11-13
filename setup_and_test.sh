#!/bin/bash
# Setup and Test Script for train.py on SLURM Cluster

echo "=== Nexar Training Setup Script ==="
echo ""

# Function to setup conda environment
setup_conda() {
    echo "Setting up conda environment..."
    module load miniconda3/latest
    
    # Check if environment exists
    if conda env list | grep -q "^nexar "; then
        echo "Environment 'nexar' already exists. Activating..."
        conda activate nexar
    else
        echo "Creating new environment 'nexar'..."
        conda config --remove channels defaults 2>/dev/null
        conda config --add channels conda-forge
        conda config --set channel_priority strict
        
        conda create -n nexar python=3.10 -y
        conda activate nexar
        
        echo "Installing packages..."
        conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
        pip install transformers opencv-python albumentations numpy pandas scikit-learn tqdm matplotlib psutil easydict timm
    fi
    
    echo "Conda environment ready!"
    conda list | grep -E "torch|transformers|opencv"
}

# Function to check cluster status
check_cluster() {
    echo ""
    echo "=== Cluster Status ==="
    echo "Available GPU nodes:"
    sinfo -p gpuloq
    echo ""
    echo "Current job queue:"
    ai-queue | head -20
    echo ""
    echo "Disk usage:"
    df -kh | grep -E "Filesystem|lustre|home"
}

# Function to request interactive session
request_interactive() {
    echo ""
    echo "Requesting interactive GPU session..."
    echo "Command: srun -N 1 -n 16 -p gpuloq --gres=gpu:1 --time 1:00:00 --pty bash"
    echo ""
    read -p "Press Enter to start interactive session (or Ctrl+C to cancel)..."
    srun -N 1 -n 16 -p gpuloq --gres=gpu:1 --time 1:00:00 --pty bash
}

# Function to test training
test_training() {
    echo ""
    echo "=== Testing train.py ==="
    module load miniconda3/latest
    conda activate nexar
    
    echo "GPU Status:"
    nvidia-smi
    echo ""
    
    echo "Starting training test..."
    cd /home/sra2157/git/nexar-solution
    python train.py
}

# Main menu
case "${1}" in
    setup)
        setup_conda
        ;;
    check)
        check_cluster
        ;;
    interactive)
        request_interactive
        ;;
    test)
        test_training
        ;;
    *)
        echo "Usage: $0 {setup|check|interactive|test}"
        echo ""
        echo "Commands:"
        echo "  setup       - Setup conda environment with required packages"
        echo "  check       - Check cluster status and available resources"
        echo "  interactive - Request interactive GPU session"
        echo "  test        - Run train.py (use inside interactive session)"
        echo ""
        echo "Example workflow:"
        echo "  1. bash setup_and_test.sh setup"
        echo "  2. bash setup_and_test.sh check"
        echo "  3. bash setup_and_test.sh interactive"
        echo "  4. (Once in interactive session) bash setup_and_test.sh test"
        ;;
esac
