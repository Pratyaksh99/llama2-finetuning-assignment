"""
LLaMA-2-7B Fine-tuning with LoRA on Dolly-15k
Full training script for CS5242 Assignment 7
"""

import os
import sys
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
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


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
            padding='max_length',
            return_tensors=None
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    def load_and_split_dataset(self):
        """Load and split full Dolly-15k dataset (80/10/10)."""
        print("Loading databricks-dolly-15k dataset...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        print(f"Original dataset size: {len(dataset)}")
        
        # Filter empty responses
        dataset = dataset.filter(lambda x: x['response'] and len(x['response'].strip()) > 0)
        print(f"After filtering empty responses: {len(dataset)}")
        
        # Split: 80% train, 10% val, 10% test
        train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_split['train']
        val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        
        print(f"Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Tokenize
        print("Tokenizing datasets...")
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
            desc="Tokenizing validation"
        )
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test"
        )
        
        return train_dataset, val_dataset, test_dataset


def main():
    """Main training pipeline."""
    
    print("="*80)
    print("LLaMA-2-7B Instruction Fine-Tuning - Full Training")
    print("CS5242 Assignment 7")
    print("="*80)
    print()
    
    # Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    OUTPUT_DIR = "./llama2-dolly-lora-full"
    USE_4BIT = True
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Training config
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 512
    
    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  4-bit quantization: {USE_4BIT}")
    print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_LENGTH}")
    print("="*80)
    print()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Training will be extremely slow.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print()
    
    # Load tokenizer
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✓ Tokenizer loaded")
    
    # Load dataset
    print("\n[2/5] Loading and preprocessing dataset...")
    processor = Dolly15kDatasetProcessor(tokenizer, max_length=MAX_LENGTH)
    train_dataset, val_dataset, test_dataset = processor.load_and_split_dataset()
    print("✓ Dataset ready")
    
    # Save test dataset info for later evaluation
    with open(f"{OUTPUT_DIR}_test_size.json", 'w') as f:
        json.dump({"test_size": len(test_dataset)}, f)
    
    # Load model
    print("\n[3/5] Loading model with 4-bit quantization...")
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
    
    # Apply LoRA
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
    print(f"✓ LoRA applied")
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Total parameters: {total:,}")
    
    # Training arguments
    print("\n[5/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=3,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    model.config.use_cache = False
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples (for later evaluation)")
    print("="*80)
    
    # Train
    train_result = trainer.train()
    
    # Save model
    print("\nSaving final model...")
    trainer.save_model()
    
    # Save training metrics
    metrics = {
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        "total_flos": train_result.metrics.get("total_flos"),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    }
    
    with open(f"{OUTPUT_DIR}/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Extract and save loss history
    if hasattr(trainer.state, 'log_history'):
        with open(f"{OUTPUT_DIR}/loss_history.json", 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Training time: {train_result.metrics.get('train_runtime', 0):.1f} seconds ({train_result.metrics.get('train_runtime', 0)/3600:.2f} hours)")
    print(f"Samples/second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Run evaluation: python evaluate_model.py")
    print("2. Generate plots: python plot_training_curves.py")
    print("3. Test inference: python test_inference.py")


if __name__ == "__main__":
    main()
