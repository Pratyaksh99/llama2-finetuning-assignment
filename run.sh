#!/bin/bash

# Quick Start Script for Assignment 7
# LLaMA-2-7B Instruction Fine-Tuning with LoRA

echo "=========================================="
echo "Assignment 7 - Quick Start Guide"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Check if requirements are installed
echo ""
echo "Installing/checking requirements..."
pip install -q -r requirements.txt
echo "✓ Requirements installed"

# Check HuggingFace login
echo ""
echo "=========================================="
echo "IMPORTANT: HuggingFace Authentication"
echo "=========================================="
echo ""
echo "You need access to meta-llama/Llama-2-7b-hf"
echo ""
echo "Steps:"
echo "1. Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf"
echo "2. Accept the license agreement"
echo "3. Get your access token from: https://huggingface.co/settings/tokens"
echo "4. Run: huggingface-cli login"
echo ""
read -p "Have you completed HuggingFace authentication? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please complete HuggingFace authentication first:"
    echo "  huggingface-cli login"
    exit 1
fi

# Menu
echo ""
echo "=========================================="
echo "What would you like to do?"
echo "=========================================="
echo ""
echo "1) Run training (fine-tune LLaMA-2)"
echo "2) Run evaluation (AlpacaEval & MT-Bench)"
echo "3) Run interactive inference"
echo "4) Run batch inference tests"
echo "5) Show system info"
echo "0) Exit"
echo ""
read -p "Enter your choice [0-5]: " choice

case $choice in
    1)
        echo ""
        echo "Starting training..."
        echo "This will take several hours depending on your GPU."
        echo ""
        python finetune_llama2.py
        ;;
    2)
        echo ""
        echo "Starting evaluation..."
        echo "This will compare base model vs fine-tuned model."
        echo ""
        python evaluate_model.py
        ;;
    3)
        echo ""
        echo "Starting interactive inference..."
        echo ""
        python inference.py --mode interactive
        ;;
    4)
        echo ""
        echo "Running batch inference tests..."
        echo ""
        python inference.py --mode batch
        ;;
    5)
        echo ""
        echo "=========================================="
        echo "System Information"
        echo "=========================================="
        echo ""
        echo "Python version:"
        python --version
        echo ""
        echo "PyTorch version:"
        python -c "import torch; print(torch.__version__)"
        echo ""
        echo "CUDA available:"
        python -c "import torch; print(torch.cuda.is_available())"
        echo ""
        echo "GPU info:"
        python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
        echo ""
        echo "CUDA version:"
        python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'No CUDA')"
        echo ""
        ;;
    0)
        echo ""
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
