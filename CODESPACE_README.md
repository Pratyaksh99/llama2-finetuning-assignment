# GitHub Codespaces Setup Guide

Complete guide for running LLaMA-2-7B fine-tuning on GitHub Codespaces with GPU support.

## 🚀 Quick Start

### Step 1: Open in Codespaces

1. **Navigate to your GitHub repository**
2. **Click the green "Code" button**
3. **Select "Codespaces" tab**
4. **Click "Create codespace on main"** (or your branch)
5. **Select machine type**: Choose **4-core with GPU** (at minimum)
   - Recommended: 8-core or 16-core with GPU for faster training
   - GPU required: T4, A10, or better

### Step 2: Wait for Environment Setup

The devcontainer will automatically:
- ✅ Install Python 3.11
- ✅ Install CUDA toolkit and cuDNN
- ✅ Install PyTorch with CUDA support
- ✅ Install all dependencies from `requirements.txt`
- ✅ Verify GPU availability

**This takes 5-10 minutes on first launch.**

### Step 3: Authenticate with HuggingFace

```bash
# Login to HuggingFace
huggingface-cli login

# Paste your token when prompted
# Get your token from: https://huggingface.co/settings/tokens
```

### Step 4: Request LLaMA-2 Access

1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Agree and access repository"
3. Fill out the Meta AI license agreement
4. Wait for approval (usually instant to a few hours)

### Step 5: Verify Environment

```bash
# Run environment check
python codespace_check.py
```

You should see all checks passing ✅

### Step 6: Start Training

```bash
# For quick test (1000 samples, ~30 min on T4)
python finetune_llama2_codespace.py

# For full training (edit script: USE_SUBSET = False, ~6-8 hours on T4)
python finetune_llama2_codespace.py
```

---

## 📋 Detailed Instructions

### Machine Type Selection

When creating your Codespace, you'll see machine type options:

| Machine Type | vCPU | RAM | GPU | Training Time (full) | Cost/hour |
|--------------|------|-----|-----|---------------------|-----------|
| 4-core GPU | 4 | 16GB | T4 (16GB) | ~8 hours | ~$0.36 |
| 8-core GPU | 8 | 32GB | T4 (16GB) | ~6 hours | ~$0.72 |
| 16-core GPU | 16 | 64GB | A10 (24GB) | ~4 hours | ~$1.44 |

**Recommendation**: Start with 4-core GPU for testing, upgrade to 8-core for full training.

### What Gets Installed

The `.devcontainer/devcontainer.json` and `.devcontainer/setup.sh` handle:

1. **Base Image**: Python 3.11 on Ubuntu
2. **NVIDIA Tools**:
   - CUDA 11.8
   - cuDNN
   - GPU drivers
3. **Python Packages**:
   - PyTorch 2.8.0 with CUDA 11.8
   - Transformers 4.56.2
   - PEFT, BitsAndBytes
   - Datasets, Accelerate
4. **VSCode Extensions**:
   - Python
   - Pylance
   - Jupyter

### Training Configuration

The `finetune_llama2_codespace.py` script uses:

**Model Settings:**
- Model: meta-llama/Llama-2-7b-hf
- 4-bit quantization: Yes (NF4)
- Memory usage: ~12-14GB VRAM

**LoRA Settings:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: All attention + MLP layers

**Training Settings:**
- Epochs: 3
- Batch size: 4 per device
- Gradient accumulation: 4 steps (effective batch = 16)
- Learning rate: 2e-4
- Optimizer: paged_adamw_8bit
- Precision: bfloat16

**Dataset:**
- Full dataset: 15,011 samples
- Split: 80% train / 10% val / 10% test
- Subset mode: 1,000 samples (for testing)

### Expected Training Times

**With subset (1,000 samples):**
- T4 GPU: ~30-45 minutes
- A10 GPU: ~20-30 minutes

**Full dataset (15,011 samples):**
- T4 GPU: ~6-8 hours
- A10 GPU: ~4-5 hours

### Monitoring Training

While training, you'll see:

```
[1/5] Loading tokenizer... ✓
[2/5] Loading dataset... ✓
[3/5] Loading model... ✓
[4/5] Applying LoRA... ✓ 4,194,304 trainable params (0.06%)
[5/5] Starting training...

{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.9876, 'learning_rate': 0.00019, 'epoch': 0.2}
...
```

### Output Files

After training, you'll find in `./llama2-dolly-lora/`:

```
llama2-dolly-lora/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # Trained LoRA weights
├── training_metrics.json        # Training statistics
├── checkpoint-100/              # Intermediate checkpoints
├── checkpoint-200/
└── ...
```

### Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./llama2-dolly-lora")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate
prompt = "### Instruction:\nExplain quantum computing.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔧 Troubleshooting

### Issue: "CUDA not available"

**Cause**: You selected a Codespace without GPU.

**Fix**:
1. Delete current Codespace
2. Create new one with GPU machine type
3. Or run: `python codespace_check.py` to verify

### Issue: "401 Client Error: Unauthorized"

**Cause**: LLaMA-2 access not granted or not authenticated.

**Fix**:
1. Go to https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Accept license
3. Run: `huggingface-cli login`
4. Enter your token from https://huggingface.co/settings/tokens

### Issue: "Out of memory" during training

**Cause**: GPU VRAM insufficient (rare with 4-bit quantization).

**Fix**:
1. Reduce batch size: Change `BATCH_SIZE = 4` to `BATCH_SIZE = 2`
2. Increase gradient accumulation: `GRAD_ACCUM = 8`
3. Reduce max sequence length: `max_length=256`

### Issue: "Disk space full"

**Cause**: Model checkpoints filling disk.

**Fix**:
1. In `finetune_llama2_codespace.py`, change:
   ```python
   save_total_limit=2  # Keep only 2 checkpoints
   ```
2. Or increase Codespace disk size in settings

### Issue: Training is very slow

**Check**:
```bash
python codespace_check.py
```

Make sure:
- ✅ CUDA is available
- ✅ GPU detected (T4, A10, etc.)
- ❌ If CPU only, training will take days

---

## 💰 Cost Optimization

GitHub Codespaces GPU pricing:
- **Free tier**: 120 core-hours/month
- **GPU costs extra**: ~$0.36/hour for 4-core T4

**Tips to save costs:**

1. **Use subset for testing**
   - Set `USE_SUBSET = True` in script
   - Only use full training when ready

2. **Stop when not in use**
   - Codespaces auto-stop after 30 min idle
   - Manually stop: Click your codespace name → "Stop codespace"

3. **Download model after training**
   ```bash
   # Compress
   tar -czf llama2-dolly-lora.tar.gz llama2-dolly-lora/
   
   # Download via browser or:
   gh codespace cp llama2-dolly-lora.tar.gz ./local-path/
   ```

4. **Delete codespace when done**
   - Settings → Codespaces → Delete

---

## 📊 Evaluation

After training, evaluate your model:

```bash
# Quick evaluation (10 samples)
python evaluate_model.py

# Interactive testing
python inference.py
```

Example output:
```
Question: Explain the theory of relativity.

Base Model: The theory of relativity is a scientific theory...

Fine-tuned Model: The theory of relativity, developed by Albert 
Einstein, consists of two parts: special relativity and general 
relativity. Special relativity states that...
```

---

## 📝 Next Steps

After successful training:

1. **Evaluate performance** using AlpacaEval and MT-Bench
2. **Test interactively** with your own prompts
3. **Download the model** for local use
4. **Document results** in your assignment report
5. **Clean up** to stop billing

---

## 🆘 Getting Help

If you encounter issues:

1. **Check environment**: `python codespace_check.py`
2. **View logs**: Scroll up in terminal for error messages
3. **Check GPU usage**: `nvidia-smi`
4. **Verify disk space**: `df -h`

**Common commands:**

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Python packages
pip list | grep -E "torch|transformers|peft"

# View training output
tail -f nohup.out  # if running in background

# Check disk usage
du -sh llama2-dolly-lora/
```

---

## ✅ Checklist

Before you start:
- [ ] Codespace created with GPU
- [ ] HuggingFace authenticated (`huggingface-cli login`)
- [ ] LLaMA-2 access approved
- [ ] Environment verified (`python codespace_check.py`)
- [ ] All checks passing ✅

Ready to train:
- [ ] Run quick test first (`USE_SUBSET = True`)
- [ ] Monitor first few epochs for errors
- [ ] Switch to full training when confident
- [ ] Monitor GPU usage with `nvidia-smi`

After training:
- [ ] Model saved to `./llama2-dolly-lora/`
- [ ] Evaluation completed
- [ ] Results documented
- [ ] Model downloaded (if needed)
- [ ] Codespace stopped/deleted

---

## 📄 File Structure

```
Assignment7/
├── finetune_llama2_codespace.py    # Main training script
├── codespace_check.py              # Environment verification
├── evaluate_model.py               # Evaluation benchmarks
├── inference.py                    # Interactive testing
├── config.py                       # Configuration
├── requirements.txt                # Python dependencies
├── CODESPACE_README.md            # This file
└── .devcontainer/
    ├── devcontainer.json          # Codespace configuration
    └── setup.sh                   # Automated setup script
```

---

**Good luck with your fine-tuning! 🚀**
