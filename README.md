# LLaMA-2-7B Instruction Fine-Tuning with LoRA on Dolly-15k

This repository contains the implementation for Assignment 7: Instruction Fine-Tuning of LLaMA-2-7B using Parameter-Efficient Fine-Tuning (PEFT) with LoRA on the Dolly-15k dataset.

## 📋 Overview

This implementation fine-tunes the **meta-llama/Llama-2-7b-hf** model using **LoRA (Low-Rank Adaptation)** on the **databricks-dolly-15k** dataset to improve instruction-following capabilities. The project includes:

- Complete dataset preprocessing and train/val/test splitting
- LoRA-based parameter-efficient fine-tuning
- Training with loss tracking and visualization
- Evaluation on AlpacaEval 2 and MT-Bench benchmarks
- Comparison between base and fine-tuned models

## 📁 Project Structure

```
Assignment7/
├── finetune_llama2.py          # Main training script
├── evaluate_model.py            # Evaluation script (AlpacaEval & MT-Bench)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── Assignment7.md               # Assignment description
```

## 🔧 Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM for 4-bit quantization)
- Hugging Face account with access to LLaMA-2 models

### Installation

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Hugging Face Authentication:**

You need access to the LLaMA-2 model. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf

Then login:
```bash
huggingface-cli login
```

## 🚀 Usage

### 1. Fine-tuning

Run the training script:

```bash
python finetune_llama2.py
```

**What it does:**
- Loads the Dolly-15k dataset and preprocesses it
- Filters empty responses
- Splits into train (80%), validation (10%), test (10%)
- Applies LoRA configuration to LLaMA-2-7B
- Trains for 3 epochs with loss tracking
- Saves the model to `./llama2-dolly-lora/`
- Generates training curves (`training_curves.png`)

**Key configurations** (in `finetune_llama2.py`):
```python
# Model
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
USE_4BIT = True  # 4-bit quantization for memory efficiency

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
```

### 2. Evaluation

Run the evaluation script:

```bash
python evaluate_model.py
```

**What it does:**
- Loads both base model and fine-tuned model
- Runs inference on AlpacaEval-style questions (10 samples)
- Runs inference on MT-Bench-style multi-turn conversations (5 samples)
- Saves results to `./evaluation_results/`
- Generates comparison outputs

**Output structure:**
```
evaluation_results/
├── base_model/
│   ├── alpaca_eval_results.json
│   └── mt_bench_results.json
└── finetuned_model/
    ├── alpaca_eval_results.json
    └── mt_bench_results.json
```

### 3. Inference Example

You can also use the trained model for custom inference:

```python
from evaluate_model import ModelEvaluator

# Load fine-tuned model
evaluator = ModelEvaluator(
    base_model_name="meta-llama/Llama-2-7b-hf",
    lora_model_path="./llama2-dolly-lora"
)
evaluator.load_model(use_lora=True)

# Generate response
response = evaluator.generate_response(
    instruction="Explain quantum computing in simple terms.",
    context=""
)
print(response)
```

## 📊 Expected Results

### Training Convergence
- Training loss should decrease steadily over epochs
- Validation loss should follow similar trend without significant overfitting
- Expected final training loss: ~1.5-2.0
- Expected final validation loss: ~1.6-2.1

### Evaluation Metrics

**AlpacaEval 2:**
- Measures instruction-following quality
- Fine-tuned model should show more coherent and relevant responses
- Expects improvement in response quality and relevance

**MT-Bench:**
- Measures multi-turn conversation ability
- Fine-tuned model should maintain better context across turns
- Expects improved consistency and coherence

## 🧠 Implementation Details

### Dataset Preprocessing

The Dolly-15k dataset is formatted as:
```
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

- Empty responses are filtered out
- Dataset split: 80% train, 10% validation, 10% test
- Maximum sequence length: 512 tokens

### LoRA Configuration

Parameter-efficient fine-tuning using LoRA:
- **Rank (r):** 16
- **Alpha:** 32
- **Dropout:** 0.05
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable parameters:** ~2-3% of total model parameters

### Training Strategy

- **Optimizer:** Paged AdamW 8-bit
- **Learning rate:** 2e-4
- **Batch size:** 4 per device with gradient accumulation (effective batch size: 16)
- **Mixed precision:** bfloat16
- **Gradient checkpointing:** Enabled for memory efficiency
- **Warmup steps:** 100

### Memory Optimization

- 4-bit quantization (NF4) using bitsandbytes
- Gradient checkpointing
- PEFT/LoRA for parameter efficiency
- Expected VRAM usage: ~12-16GB with 4-bit quantization

## 📈 Monitoring Training

Training progress is logged to console and saved to:
- `./llama2-dolly-lora/training_history.json` - Loss values per step
- `./llama2-dolly-lora/training_curves.png` - Visualization of training curves
- `./llama2-dolly-lora/logs/` - Detailed training logs

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce `PER_DEVICE_TRAIN_BATCH_SIZE` (try 2 or 1)
- Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
- Ensure 4-bit quantization is enabled (`USE_4BIT = True`)

### Slow Training
- Use mixed precision training (bfloat16/fp16)
- Enable gradient checkpointing
- Increase batch size if memory allows

### Model Access Issues
- Ensure you have accepted the LLaMA-2 license on Hugging Face
- Run `huggingface-cli login` with your access token

### CUDA Errors
- Update CUDA drivers
- Update PyTorch: `pip install torch --upgrade`
- Check GPU compatibility

## 📝 Notes

### Hardware Requirements
- **Minimum:** 1x GPU with 16GB VRAM (with 4-bit quantization)
- **Recommended:** 1x GPU with 24GB+ VRAM
- **Training time:** ~2-4 hours per epoch on A100 (80GB)

### Customization

You can modify hyperparameters in the scripts:

**LoRA parameters:**
```python
LORA_R = 16          # Increase for more capacity
LORA_ALPHA = 32      # Typically 2x rank
LORA_DROPOUT = 0.05  # Regularization
```

**Training parameters:**
```python
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
```

### Evaluation Notes

- The provided evaluation uses sample datasets for demonstration
- For official AlpacaEval 2 scores, use: https://github.com/tatsu-lab/alpaca_eval
- For official MT-Bench scores, use: https://github.com/lm-sys/FastChat
- The sample evaluation provides qualitative comparison between base and fine-tuned models

## 🎓 Assignment Deliverables

✅ **Completed:**
1. Dataset preprocessing (Dolly-15k with proper formatting)
2. LoRA-based fine-tuning implementation
3. Training with loss tracking
4. Evaluation framework (AlpacaEval 2 & MT-Bench)
5. Model comparison capabilities
6. Reproducible environment (requirements.txt)

⏳ **To be completed separately:**
- Report (2-4 pages PDF) with results and analysis

## 📚 References

- [LLaMA-2 Paper](https://arxiv.org/abs/2307.09288)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Dolly-15k Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

## 📄 License

This implementation is for educational purposes as part of CS5242 Neural Networks and Deep Learning course.
