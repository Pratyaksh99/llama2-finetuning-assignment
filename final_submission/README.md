# LLaMA-2-7B Instruction Fine-Tuning with LoRA
## CS5242 Assignment 7

### Student Information
**Name:** Pratyaksh Motwani  
**Student ID:** A0297328N

---

## 📁 Project Structure

```
final_submission/
├── requirements.txt              # Python dependencies
├── train_full.py                 # Full training script (15k samples)
├── train_quick_test.py          # Quick test (1k samples)
├── evaluate_model.py            # AlpacaEval & MT-Bench evaluation
├── plot_training_curves.py      # Generate training plots
├── test_inference.py            # Test model inference
├── README.md                    # This file
└── llama2-dolly-lora-full/      # Saved model (created after training)
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens

# Request LLaMA-2 access
# Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf
# Click "Agree and access repository"
```

### 2. Quick Test (Recommended First)

```bash
# Test with 1,000 samples (~30-45 min on GPU)
python train_quick_test.py
```

### 3. Full Training

```bash
# Train on full 15,011 samples (~6-8 hours on GPU)
python train_full.py
```

### 4. Evaluation

```bash
# Run AlpacaEval & MT-Bench
python evaluate_model.py

# Generate training plots
python plot_training_curves.py

# Test inference
python test_inference.py
```

---

## 📊 Configuration

### Dataset
- **Source:** `databricks/databricks-dolly-15k`
- **Size:** 15,011 instruction-response pairs
- **Split:** 80% train / 10% validation / 10% test
- **Format:** Instruction → Context → Response

### Model
- **Base:** `meta-llama/Llama-2-7b-hf` (7 billion parameters)
- **Method:** PEFT with LoRA (Parameter-Efficient Fine-Tuning)
- **Quantization:** 4-bit NF4 (reduces VRAM to ~12-14GB)

### LoRA Configuration
```python
r = 16                    # LoRA rank
alpha = 32                # LoRA alpha
dropout = 0.05            # LoRA dropout
target_modules = [        # Modules to apply LoRA
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

**Trainable Parameters:** ~4.2M (0.06% of total 7B parameters)

### Training Hyperparameters
```python
epochs = 3
batch_size = 4
gradient_accumulation = 4    # Effective batch size = 16
learning_rate = 2e-4
max_sequence_length = 512
optimizer = "paged_adamw_8bit"
warmup_steps = 100
```

---

## 🖥️ Hardware Requirements

### Minimum (Quick Test)
- **GPU:** 16GB VRAM (e.g., RTX A4000, T4)
- **RAM:** 16GB
- **Disk:** 30GB free space
- **Time:** ~30-45 minutes

### Recommended (Full Training)
- **GPU:** 16GB+ VRAM (e.g., A100, A10, RTX A6000)
- **RAM:** 32GB
- **Disk:** 50GB free space
- **Time:** ~6-8 hours (T4), ~4-5 hours (A100)

---

## 📈 Expected Results

### Training Loss
- **Initial:** ~2.5
- **Final:** ~1.0-1.5
- **Convergence:** Smooth decrease over 3 epochs

### Validation Loss
- **Initial:** ~2.4
- **Final:** ~1.3-1.7
- **No overfitting:** Val loss follows train loss

### Evaluation
- **AlpacaEval:** Improved instruction-following quality
- **MT-Bench:** Better multi-turn conversation coherence
- **Comparison:** Fine-tuned model > Base model

---

## 📝 Output Files

After training, the following files are generated:

```
llama2-dolly-lora-full/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # Trained LoRA weights (~100-200MB)
├── training_metrics.json        # Training statistics
└── loss_history.json            # Step-by-step loss values

Generated plots:
├── training_curves.png          # Training/validation loss curves
└── learning_rate_schedule.png   # Learning rate over time

Evaluation results:
├── alpaca_eval_results.json     # AlpacaEval comparisons
└── mtbench_results.json         # MT-Bench conversations
```

---

## 🔍 Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size
```python
# In train_full.py, change:
BATCH_SIZE = 2          # Reduce from 4 to 2
GRAD_ACCUM = 8          # Increase from 4 to 8
```

### "401 Unauthorized" when loading model
**Solution:** Authenticate and request access
```bash
huggingface-cli login
# Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### "No GPU detected"
**Solution:** Training will be extremely slow on CPU. Use a GPU instance (Digital Ocean, AWS, Colab Pro).

---

## 📚 References

1. **LLaMA 2:** Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
2. **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
3. **Dolly-15k:** Databricks, "databricks-dolly-15k" dataset
4. **PEFT:** Hugging Face PEFT library - https://github.com/huggingface/peft
5. **AlpacaEval:** Stanford Alpaca evaluation framework
6. **MT-Bench:** FastChat multi-turn benchmark

---

## ✅ Assignment Checklist

### Code Implementation
- [x] Dataset loading and preprocessing
- [x] 80/10/10 train/val/test split
- [x] Instruction formatting
- [x] LoRA configuration
- [x] Training with loss tracking
- [x] Model saving
- [x] AlpacaEval evaluation
- [x] MT-Bench evaluation
- [x] Training curve visualization

### Report Components
- [ ] Dataset description
- [ ] Implementation details
- [ ] Hardware specifications
- [ ] Training configuration
- [ ] Loss convergence plots
- [ ] Evaluation results (tables)
- [ ] Base vs fine-tuned comparison
- [ ] Discussion and analysis
- [ ] Conclusion

---

## 📧 Contact

For questions or issues:
- **Email:** e1351466@u.nus.edu
- **GitHub:** github.com/Pratyaksh99/llama2-finetuning-assignment

---

**Last Updated:** October 25, 2025
