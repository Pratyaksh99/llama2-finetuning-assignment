#!/bin/bash

echo "=========================================="
echo "Setting up LLaMA-2 Fine-tuning Environment"
echo "=========================================="

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: huggingface-cli login"
echo "2. Run: python codespace_check.py"
echo "3. Run: python finetune_llama2_codespace.py"
echo ""
