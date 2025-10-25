#!/bin/bash

# Digital Ocean GPU Droplet Setup Script
# For LLaMA-2-7B Fine-tuning with LoRA
# Run this script on your DO droplet after initial login

set -e

echo "=========================================="
echo "Digital Ocean GPU Setup"
echo "LLaMA-2-7B Fine-tuning Environment"
echo "=========================================="
echo ""

# Step 1: System Update
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install Python 3.11
echo ""
echo "[2/8] Installing Python 3.11..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Step 3: Verify NVIDIA GPU and CUDA
echo ""
echo "[3/8] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ GPU detected"
else
    echo "⚠️  WARNING: nvidia-smi not found!"
    echo "   Installing NVIDIA drivers..."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "   Please reboot and run this script again: sudo reboot"
    exit 1
fi

# Step 4: Install CUDA Toolkit (if needed)
echo ""
echo "[4/8] Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✓ CUDA toolkit installed"
else
    echo "Installing CUDA toolkit 11.8..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-11-8
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# Step 5: Install Git
echo ""
echo "[5/8] Installing Git..."
sudo apt-get install -y git git-lfs

# Step 6: Clone Repository
echo ""
echo "[6/8] Setting up project..."
if [ ! -d "llama2-finetuning-assignment" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Pratyaksh99/llama2-finetuning-assignment.git
    cd llama2-finetuning-assignment
else
    echo "Repository already exists, updating..."
    cd llama2-finetuning-assignment
    git pull
fi

# Step 7: Create Virtual Environment
echo ""
echo "[7/8] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Step 8: Install PyTorch and Dependencies
echo ""
echo "[8/8] Installing dependencies..."
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing other requirements..."
pip install -r requirements.txt

# Verify Installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Authenticate HuggingFace: huggingface-cli login"
echo "3. Request LLaMA-2 access: https://huggingface.co/meta-llama/Llama-2-7b-hf"
echo "4. Run environment check: python codespace_check.py"
echo "5. Start training: python finetune_llama2_codespace.py"
echo ""
echo "To reconnect later:"
echo "  cd llama2-finetuning-assignment"
echo "  source venv/bin/activate"
echo ""
