"""
GitHub Codespaces Environment Validation
Run this before training to ensure everything is set up correctly.
"""

import sys
import os

print("="*80)
print("GitHub Codespaces Environment Check")
print("="*80)
print()

checks_passed = 0
checks_total = 0

# Check 1: Python version
checks_total += 1
print(f"[{checks_total}] Python Version")
print(f"    Version: {sys.version}")
if sys.version_info >= (3, 8):
    print("    ✅ PASS - Python 3.8+ detected")
    checks_passed += 1
else:
    print("    ❌ FAIL - Python 3.8+ required")
print()

# Check 2: CUDA availability
checks_total += 1
print(f"[{checks_total}] CUDA Availability")
try:
    import torch
    if torch.cuda.is_available():
        print(f"    ✅ PASS - CUDA is available")
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        checks_passed += 1
    else:
        print("    ❌ FAIL - CUDA not available")
        print("    Training will be extremely slow on CPU")
except Exception as e:
    print(f"    ❌ ERROR - {e}")
print()

# Check 3: PyTorch installation
checks_total += 1
print(f"[{checks_total}] PyTorch Installation")
try:
    import torch
    print(f"    Version: {torch.__version__}")
    print("    ✅ PASS - PyTorch installed")
    checks_passed += 1
except Exception as e:
    print(f"    ❌ FAIL - {e}")
print()

# Check 4: Transformers library
checks_total += 1
print(f"[{checks_total}] Transformers Library")
try:
    import transformers
    print(f"    Version: {transformers.__version__}")
    print("    ✅ PASS - Transformers installed")
    checks_passed += 1
except Exception as e:
    print(f"    ❌ FAIL - {e}")
print()

# Check 5: PEFT library
checks_total += 1
print(f"[{checks_total}] PEFT Library")
try:
    import peft
    print(f"    Version: {peft.__version__}")
    print("    ✅ PASS - PEFT installed")
    checks_passed += 1
except Exception as e:
    print(f"    ❌ FAIL - {e}")
print()

# Check 6: Datasets library
checks_total += 1
print(f"[{checks_total}] Datasets Library")
try:
    import datasets
    print(f"    Version: {datasets.__version__}")
    print("    ✅ PASS - Datasets installed")
    checks_passed += 1
except Exception as e:
    print(f"    ❌ FAIL - {e}")
print()

# Check 7: BitsAndBytes (for 4-bit quantization)
checks_total += 1
print(f"[{checks_total}] BitsAndBytes (4-bit Quantization)")
try:
    import bitsandbytes
    print(f"    Version: {bitsandbytes.__version__}")
    print("    ✅ PASS - BitsAndBytes installed")
    checks_passed += 1
except Exception as e:
    print(f"    ❌ FAIL - {e}")
    print("    Note: Required for 4-bit quantization")
print()

# Check 8: HuggingFace authentication
checks_total += 1
print(f"[{checks_total}] HuggingFace Authentication")
try:
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if token:
        print("    ✅ PASS - HuggingFace token found")
        checks_passed += 1
    else:
        print("    ❌ FAIL - No HuggingFace token")
        print("    Run: huggingface-cli login")
except Exception as e:
    print(f"    ⚠️  WARNING - {e}")
    print("    You may need to authenticate to access LLaMA-2")
print()

# Check 9: LLaMA-2 access
checks_total += 1
print(f"[{checks_total}] LLaMA-2 Model Access")
print("    Attempting to access meta-llama/Llama-2-7b-hf...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("    ✅ PASS - LLaMA-2 access granted")
    checks_passed += 1
except Exception as e:
    print(f"    ❌ FAIL - {str(e)[:100]}...")
    print()
    print("    To fix:")
    print("    1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("    2. Accept the license agreement")
    print("    3. Run: huggingface-cli login")
    print("    4. Enter your HuggingFace token")
print()

# Check 10: Disk space
checks_total += 1
print(f"[{checks_total}] Disk Space")
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"    Free space: {free_gb:.1f} GB")
    if free_gb > 30:
        print("    ✅ PASS - Sufficient disk space (30GB+ recommended)")
        checks_passed += 1
    else:
        print("    ⚠️  WARNING - Low disk space (30GB+ recommended)")
        print("    Model and checkpoints require ~20-30GB")
except Exception as e:
    print(f"    ⚠️  WARNING - {e}")
print()

# Summary
print("="*80)
print("Summary")
print("="*80)
print(f"Checks passed: {checks_passed}/{checks_total}")
print()

if checks_passed == checks_total:
    print("✅ ALL CHECKS PASSED!")
    print("   You're ready to start training.")
    print()
    print("Next steps:")
    print("   python finetune_llama2_codespace.py")
elif checks_passed >= checks_total - 2:
    print("⚠️  MOSTLY READY")
    print("   Fix the failed checks above, then start training.")
else:
    print("❌ NOT READY")
    print("   Please fix the issues above before training.")
    print()
    print("Quick fixes:")
    print("   - Install packages: pip install -r requirements.txt")
    print("   - Login to HuggingFace: huggingface-cli login")
    print("   - Request LLaMA-2 access: https://huggingface.co/meta-llama/Llama-2-7b-hf")

print("="*80)
