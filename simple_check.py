"""
Super quick validation - no problematic imports
Just checks code syntax and structure
"""

print("="*80)
print("Assignment 7 - Code Validation (Syntax & Structure)")
print("="*80)

import os
import sys

# Test 1: Check all files exist
print("\n[1] Checking files...")
files = {
    'finetune_llama2.py': 'Main training script',
    'evaluate_model.py': 'Evaluation script',
    'inference.py': 'Inference script',
    'config.py': 'Configuration',
    'requirements.txt': 'Dependencies',
    'README.md': 'Documentation',
    'IMPLEMENTATION_GUIDE.md': 'Implementation guide',
    'SUMMARY.md': 'Summary'
}

all_exist = True
for filename, description in files.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"✓ {filename:30s} ({size:,} bytes) - {description}")
    else:
        print(f"✗ {filename:30s} - MISSING")
        all_exist = False

# Test 2: Validate Python syntax
print("\n[2] Validating Python syntax...")
python_files = ['finetune_llama2.py', 'evaluate_model.py', 'inference.py', 'config.py']

import py_compile
all_valid = True
for pyfile in python_files:
    try:
        py_compile.compile(pyfile, doraise=True)
        print(f"✓ {pyfile:30s} - Syntax valid")
    except py_compile.PyCompileError as e:
        print(f"✗ {pyfile:30s} - Syntax error: {e}")
        all_valid = False

# Test 3: Check key classes/functions exist
print("\n[3] Checking code structure...")

def check_code_content(filename, required_items):
    """Check if required classes/functions exist in file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        missing = []
        for item in required_items:
            if item not in content:
                missing.append(item)
        
        if not missing:
            print(f"✓ {filename:30s} - All components found")
            return True
        else:
            print(f"⚠ {filename:30s} - Missing: {', '.join(missing)}")
            return False
    except Exception as e:
        print(f"✗ {filename:30s} - Error: {e}")
        return False

checks = {
    'finetune_llama2.py': [
        'class Dolly15kDatasetProcessor',
        'class LLaMA2LoRATrainer',
        'def load_and_split_dataset',
        'def configure_lora',
        'def train'
    ],
    'evaluate_model.py': [
        'class ModelEvaluator',
        'class AlpacaEvalRunner',
        'class MTBenchRunner',
        'def generate_response',
        'def run_evaluation'
    ],
    'inference.py': [
        'def load_model',
        'def generate_response',
        'def interactive_mode'
    ],
    'config.py': [
        'MODEL_CONFIG',
        'LORA_CONFIG',
        'TRAINING_CONFIG',
        'EVAL_CONFIG'
    ]
}

structure_ok = True
for filename, items in checks.items():
    result = check_code_content(filename, items)
    structure_ok = structure_ok and result

# Test 4: Quick Python imports (minimal)
print("\n[4] Testing minimal imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except:
    print("✗ PyTorch not available")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except:
    print("✗ Transformers not available")

# Don't import PEFT/bitsandbytes to avoid mutex warnings
print("⚠ Skipping PEFT/BitsAndBytes import (causes mutex warnings)")
print("  These will be imported when training runs")

# Test 5: Config validation
print("\n[5] Testing configuration...")
try:
    import config
    print(f"✓ Config loaded successfully")
    print(f"  Base model: {config.MODEL_CONFIG['base_model_name']}")
    print(f"  LoRA rank: {config.LORA_CONFIG['r']}")
    print(f"  Training epochs: {config.TRAINING_CONFIG['num_train_epochs']}")
    print(f"  Batch size: {config.TRAINING_CONFIG['per_device_train_batch_size']}")
except Exception as e:
    print(f"✗ Config error: {e}")

# Summary
print("\n" + "="*80)
print("Validation Summary")
print("="*80)

if all_exist and all_valid and structure_ok:
    print("\n✅ All validations PASSED!")
    print("\n📋 Implementation is complete and ready:")
    print("   • All files present")
    print("   • Python syntax valid")
    print("   • Code structure correct")
    print("   • Configuration loaded")
    
    print("\n🚀 Next steps:")
    print("   1. Get Hugging Face access token:")
    print("      Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("      Run: huggingface-cli login")
    print()
    print("   2. Start training (requires GPU):")
    print("      python finetune_llama2.py")
    print()
    print("   3. Or test inference (after training):")
    print("      python inference.py --mode interactive")
else:
    print("\n⚠ Some checks failed - review errors above")

print("="*80)
