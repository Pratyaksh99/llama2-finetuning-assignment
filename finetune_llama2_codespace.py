"""
LLaMA-2-7B Fine-tuning for GitHub Codespaces
Optimized for cloud GPU environments with CUDA support

This script is designed to run in GitHub Codespaces with GPU access.
"""

import os
import sys

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


print("="*80)
print("LLaMA-2-7B Instruction Fine-Tuning - Codespaces Edition")
print("="*80)
print()

# Check CUDA
if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not available!")
    print("   This will run on CPU and be EXTREMELY slow.")
    print("   Please use a Codespace with GPU access.")
    response = input("\nContinue anyway? (yes/no): ")
    if response.lower() != 'yes':
        sys.exit(1)
else:
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()


class Dolly15kDatasetProcessor:
    """Handles Dolly-15k dataset loading and preprocessing."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_instruction(self, example: Dict) -> str:
        """Format example into instruction-response format."""
        instruction = example.get('instruction', '').strip()
        context = example.get('context', '').strip()
        response = example.get('response', '').strip()
        
        formatted_text = f"### Instruction:\n{instruction}\n\n"
        if context:
            formatted_text += f"### Context:\n{context}\n\n"
        formatted_text += f"### Response:\n{response}"
        
        return formatted_text
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize examples."""
        texts = []
        for i in range(len(examples['instruction'])):
            example = {
                'instruction': examples['instruction'][i],
                'context': examples['context'][i],
                'response': examples['response'][i]
            }
            texts.append(self.format_instruction(example))
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    def load_and_split_dataset(self, use_subset: bool = False, subset_size: int = 1000):
        """Load and split Dolly-15k dataset."""
        print("Loading databricks-dolly-15k dataset...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        print(f"Original dataset size: {len(dataset)}")
        
        # Filter empty responses
        dataset = dataset.filter(lambda x: x['response'] and len(x['response'].strip()) > 0)
        print(f"After filtering: {len(dataset)}")
        
        # Use subset for quick testing
        if use_subset:
            print(f"Using subset of {subset_size} samples for testing...")
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        # Split
        train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_split['train']
        val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Tokenize
        print("Tokenizing...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val"
        )
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test"
        )
        
        return train_dataset, val_dataset, test_dataset


def main():
    """Main training pipeline for Codespaces."""
    
    # Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    OUTPUT_DIR = "./llama2-dolly-lora"
    USE_4BIT = True  # Use 4-bit quantization for memory efficiency
    USE_SUBSET = True  # Set to False for full training
    SUBSET_SIZE = 1000  # Number of samples for quick test
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Training config
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    
    print("\n" + "="*80)
    print("Configuration")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"4-bit quantization: {USE_4BIT}")
    print(f"Use subset: {USE_SUBSET} ({SUBSET_SIZE} samples)" if USE_SUBSET else "Full dataset")
    print(f"LoRA rank: {LORA_R}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print("="*80)
    print()
    
    # Step 1: Load tokenizer
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✓ Tokenizer loaded")
    
    # Step 2: Load dataset
    print("\n[2/5] Loading dataset...")
    processor = Dolly15kDatasetProcessor(tokenizer)
    train_dataset, val_dataset, test_dataset = processor.load_and_split_dataset(
        use_subset=USE_SUBSET,
        subset_size=SUBSET_SIZE
    )
    print("✓ Dataset ready")
    
    # Step 3: Load model with quantization
    print("\n[3/5] Loading model...")
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    print("✓ Model loaded")
    
    # Step 4: Apply LoRA
    print("\n[4/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA applied: {trainable:,} trainable params ({100*trainable/total:.2f}%)")
    
    # Step 5: Train
    print("\n[5/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    model.config.use_cache = False
    
    print(f"Training on {len(train_dataset)} samples...")
    print(f"Validating on {len(val_dataset)} samples...")
    print("="*80)
    
    train_result = trainer.train()
    
    # Save
    print("\nSaving model...")
    trainer.save_model()
    
    # Save metrics
    with open(f"{OUTPUT_DIR}/training_metrics.json", 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Training time: {train_result.metrics.get('train_runtime', 0):.1f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()
