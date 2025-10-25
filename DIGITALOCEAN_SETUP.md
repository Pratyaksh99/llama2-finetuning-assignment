# Digital Ocean GPU Droplet Setup Guide

Complete guide for running LLaMA-2-7B fine-tuning on Digital Ocean GPU droplet.

## 🚀 Quick Start

### Step 1: Create GPU Droplet

1. **Go to Digital Ocean Console**
2. **Create Droplet**:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: GPU Droplet
   - **Recommended**: Basic GPU ($0.89/hr) or Premium GPU ($2.38/hr)
     - Basic GPU: NVIDIA RTX A4000 (16GB VRAM) - Good for training
     - Premium GPU: NVIDIA A100 (40GB VRAM) - Faster but more expensive
   - **Region**: Choose closest to you with GPU availability
   - **Add SSH Key**: Your public SSH key
   - **Hostname**: llama2-training (or your choice)

3. **Wait for droplet to be created** (~2 minutes)

### Step 2: Connect to Droplet

```bash
# From your local machine
ssh root@YOUR_DROPLET_IP

# You'll see something like:
# Welcome to Ubuntu 22.04.x LTS
# root@llama2-training:~#
```

### Step 3: Run One-Command Setup

```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/Pratyaksh99/llama2-finetuning-assignment/main/setup_digitalocean.sh -o setup.sh
chmod +x setup.sh
./setup.sh
```

**This will take 10-15 minutes** and will:
- ✅ Update system packages
- ✅ Install Python 3.11
- ✅ Verify NVIDIA GPU drivers
- ✅ Install CUDA toolkit
- ✅ Clone your repository
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA
- ✅ Install all dependencies

### Step 4: Authenticate HuggingFace

```bash
cd llama2-finetuning-assignment
source venv/bin/activate
huggingface-cli login
```

Paste your token from: https://huggingface.co/settings/tokens

### Step 5: Request LLaMA-2 Access

**In your browser:**
1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Agree and access repository"
3. Wait for approval (usually instant)

### Step 6: Verify Environment

```bash
python codespace_check.py
```

All checks should pass ✅

### Step 7: Start Training!

```bash
# Quick test (1000 samples, ~30 min on A4000)
python finetune_llama2_codespace.py

# OR use interactive script
./run_training.sh
```

---

## 📋 Manual Setup (If Script Fails)

If the automated script fails, run these commands manually:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Check GPU
nvidia-smi

# Clone repo
git clone https://github.com/Pratyaksh99/llama2-finetuning-assignment.git
cd llama2-finetuning-assignment

# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 💰 Cost Estimates

### Digital Ocean GPU Pricing

| GPU Type | VRAM | Price/Hour | Quick Test (1000) | Full Training (15k) |
|----------|------|------------|-------------------|---------------------|
| RTX A4000 | 16GB | $0.89/hr | ~$0.45 (30 min) | ~$5.34 (6 hrs) |
| A100 | 40GB | $2.38/hr | ~$1.19 (30 min) | ~$9.52 (4 hrs) |

**Recommendation**: Use **Basic GPU (A4000)** - excellent performance for the price.

**Total cost for complete assignment (with testing):**
- Quick test run: ~$0.50
- Full training: ~$5-6
- Evaluation: ~$0.20
- **Total: ~$6-7**

---

## ⚙️ Training Configuration

The `finetune_llama2_codespace.py` is already optimized for GPU:

```python
# Current settings (in the script)
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
USE_4BIT = True  # 4-bit quantization
USE_SUBSET = True  # 1000 samples for testing

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4  # Effective batch = 16
LEARNING_RATE = 2e-4
```

**To switch to full training**: Edit `finetune_llama2_codespace.py` line ~172:
```python
USE_SUBSET = False  # Change True to False
```

---

## 📊 Expected Training Times

### RTX A4000 (16GB VRAM)
- Quick test (1,000 samples): ~30-40 minutes
- Full dataset (15,011 samples): ~5-7 hours

### A100 (40GB VRAM)
- Quick test (1,000 samples): ~15-20 minutes
- Full dataset (15,011 samples): ~3-4 hours

---

## 🔧 Troubleshooting

### Issue: "nvidia-smi command not found"

**Fix**: GPU drivers not installed or droplet doesn't have GPU
```bash
# Check if you created GPU droplet
lspci | grep -i nvidia

# If no output, you created wrong droplet type
# Delete and create GPU droplet instead
```

### Issue: "CUDA out of memory"

**Fix**: Reduce batch size
```bash
# Edit finetune_llama2_codespace.py
# Change line ~176:
BATCH_SIZE = 2  # Reduce from 4 to 2
GRAD_ACCUM = 8  # Increase from 4 to 8
```

### Issue: "Connection timed out" when SSH

**Fix**: Check firewall settings
```bash
# In DO Console, check droplet's Networking > Firewall
# Ensure SSH (port 22) is allowed from your IP
```

### Issue: Setup script fails at CUDA installation

**Fix**: DO GPU droplets usually have CUDA pre-installed
```bash
# Check existing CUDA
nvcc --version

# If installed, skip CUDA step in script
# Just install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔄 Running in Background (tmux)

For long training runs, use tmux to keep training running if you disconnect:

```bash
# Install tmux
sudo apt-get install -y tmux

# Start new session
tmux new -s training

# Inside tmux, start training
cd llama2-finetuning-assignment
source venv/bin/activate
python finetune_llama2_codespace.py

# Detach from tmux: Press Ctrl+B, then D

# Reconnect later
ssh root@YOUR_DROPLET_IP
tmux attach -t training

# View training progress
```

---

## 📊 Monitoring Training

### Option 1: Watch Loss in Real-time

Training will output:
```
{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.9876, 'learning_rate': 0.00019, 'epoch': 0.2}
...
```

Loss should decrease from ~2.5 to ~1.0-1.5

### Option 2: Monitor GPU Usage

```bash
# In another terminal/tmux pane
watch -n 1 nvidia-smi
```

You'll see:
- GPU utilization: Should be 80-100%
- Memory usage: ~12-14GB with 4-bit quantization
- Temperature: ~60-80°C is normal

### Option 3: Check Training Files

```bash
# View latest checkpoint
ls -lht llama2-dolly-lora/checkpoint-*/

# View metrics
cat llama2-dolly-lora/training_metrics.json
```

---

## 📥 Downloading Trained Model

After training completes:

```bash
# On the droplet, compress the model
cd llama2-finetuning-assignment
tar -czf llama2-dolly-lora.tar.gz llama2-dolly-lora/

# On your local machine, download
scp root@YOUR_DROPLET_IP:~/llama2-finetuning-assignment/llama2-dolly-lora.tar.gz .

# Extract locally
tar -xzf llama2-dolly-lora.tar.gz
```

---

## 🗑️ Cleanup After Training

**IMPORTANT**: Don't forget to destroy the droplet when done to stop charges!

```bash
# On droplet, download your model first!
# Then exit
exit

# From DO Console:
# 1. Go to Droplets
# 2. Click "..." on your droplet
# 3. Click "Destroy"
# 4. Confirm destruction
```

Or via CLI:
```bash
doctl compute droplet delete DROPLET_ID
```

---

## ✅ Complete Workflow Checklist

**Setup Phase:**
- [ ] Create DO GPU droplet (A4000 or A100)
- [ ] SSH into droplet
- [ ] Run setup_digitalocean.sh
- [ ] Verify GPU with nvidia-smi
- [ ] Authenticate HuggingFace
- [ ] Request LLaMA-2 access
- [ ] Run codespace_check.py (all pass ✅)

**Training Phase:**
- [ ] Start tmux session
- [ ] Run quick test first (1000 samples)
- [ ] Verify loss decreasing
- [ ] Switch to full training (edit USE_SUBSET=False)
- [ ] Monitor GPU usage
- [ ] Wait for completion (~6 hours on A4000)

**Completion Phase:**
- [ ] Run evaluation (python evaluate_model.py)
- [ ] Test inference (python inference.py)
- [ ] Compress model (tar -czf)
- [ ] Download to local machine (scp)
- [ ] **Destroy droplet** (stop billing!)

---

## 🆘 Getting Help

**Check logs:**
```bash
# Last 50 lines of output
tail -50 nohup.out

# Search for errors
grep -i error nohup.out
```

**Common commands:**
```bash
# GPU status
nvidia-smi

# Disk space
df -h

# Memory usage
free -h

# Python processes
ps aux | grep python

# Kill training if stuck
pkill -f finetune_llama2
```

---

## 🎯 Success Criteria

After training, you should have:
- ✅ `llama2-dolly-lora/adapter_model.safetensors` (~100-200MB)
- ✅ `llama2-dolly-lora/adapter_config.json`
- ✅ `llama2-dolly-lora/training_metrics.json`
- ✅ Final training loss < 1.5
- ✅ Evaluation shows improvement over base model
- ✅ Model responds correctly to test prompts

---

**Ready to start? SSH into your DO droplet and run the setup script!** 🚀
