"""
Quick test - minimal imports to verify setup
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*80)
print("Quick Implementation Test")
print("="*80)

# Test 1: Basic imports
print("\n[1] Testing basic packages...")
try:
    import numpy as np
    print("✓ NumPy")
    import pandas as pd
    print("✓ Pandas")
    import matplotlib
    print("✓ Matplotlib")
except ImportError as e:
    print(f"✗ Error: {e}")

# Test 2: PyTorch
print("\n[2] Testing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")

# Test 3: Transformers & Datasets  
print("\n[3] Testing Transformers...")
try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError:
    print("✗ Transformers not installed")

try:
    import datasets
    print(f"✓ Datasets {datasets.__version__}")
except ImportError:
    print("✗ Datasets not installed")

# Test 4: PEFT
print("\n[4] Testing PEFT...")
try:
    import peft
    print(f"✓ PEFT {peft.__version__}")
except ImportError:
    print("✗ PEFT not installed")

# Test 5: Accelerate
print("\n[5] Testing Accelerate...")
try:
    import accelerate
    print(f"✓ Accelerate {accelerate.__version__}")
except ImportError:
    print("✗ Accelerate not installed")

# Test 6: Check files
print("\n[6] Checking implementation files...")
import os
files = [
    'finetune_llama2.py',
    'evaluate_model.py',
    'inference.py',
    'config.py',
    'requirements.txt',
    'README.md'
]

for f in files:
    if os.path.exists(f):
        print(f"✓ {f}")
    else:
        print(f"✗ {f} missing")

# Test 7: Config
print("\n[7] Testing config...")
try:
    import config
    print("✓ Config module loaded")
    print(f"  Model: {config.MODEL_CONFIG['base_model_name']}")
    print(f"  LoRA rank: {config.LORA_CONFIG['r']}")
    print(f"  Epochs: {config.TRAINING_CONFIG['num_train_epochs']}")
except Exception as e:
    print(f"✗ Config error: {e}")

print("\n" + "="*80)
print("✅ Basic validation complete!")
print("\nNext steps:")
print("  1. Ensure you have Hugging Face token: huggingface-cli login")
print("  2. Run training: python finetune_llama2.py")
print("="*80)
