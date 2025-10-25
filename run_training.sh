#!/bin/bash

# Quick Start Script for LLaMA-2 Fine-Tuning on GitHub Codespaces
# This script runs all necessary checks and starts training

set -e  # Exit on error

echo "========================================"
echo "LLaMA-2 Fine-Tuning - Quick Start"
echo "========================================"
echo ""

# Step 1: Verify environment
echo "[1/4] Verifying environment..."
python codespace_check.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment check failed!"
    echo "   Please fix the issues above before continuing."
    exit 1
fi

echo ""
echo "Press ENTER to continue to training, or Ctrl+C to cancel..."
read -r

# Step 2: Ask about subset
echo ""
echo "[2/4] Dataset configuration"
echo ""
echo "Do you want to run a quick test first? (Recommended)"
echo "  - Quick test: 1,000 samples, ~30-45 minutes"
echo "  - Full training: 15,011 samples, ~6-8 hours"
echo ""
read -p "Use quick test mode? (y/n): " use_subset

if [[ "$use_subset" == "y" || "$use_subset" == "Y" ]]; then
    export USE_SUBSET="True"
    echo "✓ Quick test mode selected (1,000 samples)"
else
    export USE_SUBSET="False"
    echo "✓ Full training mode selected (15,011 samples)"
fi

# Step 3: Check GPU
echo ""
echo "[3/4] Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo ""
    echo "⚠️  WARNING: No GPU detected!"
    echo "   Training will be EXTREMELY slow on CPU."
    echo ""
    read -p "Continue anyway? (yes/no): " continue_cpu
    if [[ "$continue_cpu" != "yes" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Step 4: Start training
echo ""
echo "[4/4] Starting training..."
echo "========================================"
echo ""

# Update script with subset preference
if [[ "$USE_SUBSET" == "True" ]]; then
    sed -i 's/USE_SUBSET = False/USE_SUBSET = True/' finetune_llama2_codespace.py 2>/dev/null || \
    sed -i '' 's/USE_SUBSET = False/USE_SUBSET = True/' finetune_llama2_codespace.py
else
    sed -i 's/USE_SUBSET = True/USE_SUBSET = False/' finetune_llama2_codespace.py 2>/dev/null || \
    sed -i '' 's/USE_SUBSET = True/USE_SUBSET = False/' finetune_llama2_codespace.py
fi

# Run training
python finetune_llama2_codespace.py

echo ""
echo "========================================"
echo "✅ Training completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python evaluate_model.py"
echo "  2. Test: python inference.py"
echo "  3. Download model: tar -czf llama2-dolly-lora.tar.gz llama2-dolly-lora/"
echo ""
