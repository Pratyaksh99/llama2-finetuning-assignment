"""
System Check Script
Verifies that all dependencies and requirements are properly installed
before starting the training or evaluation process.
"""

import os
import sys
import subprocess

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} (OK)")
        return True
    except ImportError:
        print(f"✗ {package_name} (Not installed)")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available (CPU-only mode, will be very slow)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_huggingface_auth():
    """Check if Hugging Face authentication is set up."""
    print("\nChecking Hugging Face authentication...")
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face token found")
            return True
        else:
            print("✗ No Hugging Face token found")
            print("  Run: huggingface-cli login")
            return False
    except Exception as e:
        print(f"✗ Error checking auth: {e}")
        return False

def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"  Available space: {free_gb:.1f} GB")
        if free_gb > 50:
            print("✓ Sufficient disk space")
            return True
        else:
            print("⚠ Low disk space (need at least 50 GB)")
            return False
    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return False

def main():
    print("="*80)
    print("System Check for Assignment 7")
    print("LLaMA-2-7B Instruction Fine-Tuning with LoRA")
    print("="*80)
    
    all_checks = []
    
    # Check Python version
    print("\n[1] Python Environment")
    print("-"*80)
    all_checks.append(check_python_version())
    
    # Check required packages
    print("\n[2] Required Packages")
    print("-"*80)
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('peft', 'peft'),
        ('bitsandbytes', 'bitsandbytes'),
        ('accelerate', 'accelerate'),
        ('sentencepiece', 'sentencepiece'),
        ('matplotlib', 'matplotlib'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('tqdm', 'tqdm'),
    ]
    
    for pkg_name, import_name in packages:
        all_checks.append(check_package(pkg_name, import_name))
    
    # Check CUDA
    print("\n[3] CUDA / GPU")
    print("-"*80)
    cuda_available = check_cuda()
    
    # Check Hugging Face auth
    print("\n[4] Hugging Face")
    print("-"*80)
    all_checks.append(check_huggingface_auth())
    
    # Check disk space
    print("\n[5] Disk Space")
    print("-"*80)
    all_checks.append(check_disk_space())
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if not cuda_available:
        print("\n⚠ WARNING: No CUDA detected. Training will be extremely slow on CPU.")
        print("   Recommended: Use a GPU with at least 16GB VRAM.")
    
    if all(all_checks):
        print("\n✓ All critical checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. Run training: python finetune_llama2.py")
        print("  2. Run evaluation: python evaluate_model.py")
        print("  3. Test inference: python inference.py --mode interactive")
    else:
        print("\n✗ Some checks failed. Please fix the issues above before proceeding.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nTo setup Hugging Face authentication:")
        print("  huggingface-cli login")
    
    print("\n" + "="*80)
    
    return all(all_checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
