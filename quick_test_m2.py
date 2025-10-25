"""
Quick Test Version - LLaMA-2-7B Fine-tuning on M2
Trains on only 100 samples to verify the implementation works

This is for TESTING ONLY - to verify:
1. Model loads correctly on M2
2. Dataset preprocessing works
3. LoRA training works
4. MPS acceleration works
5. No errors in the pipeline

Expected time: 10-20 minutes
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
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class QuickTester:
    """Quick test implementation for M2."""
    
    def __init__(self):
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.output_dir = "./llama2-quick-test"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_small_dataset(self, tokenizer, num_samples: int = 100):
        """Load a small subset of Dolly-15k for testing."""
        print(f"\nLoading {num_samples} samples from Dolly-15k...")
        print("⏳ Downloading dataset (this may take 1-2 minutes)...")
        
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        print(f"✓ Dataset downloaded: {len(dataset)} total examples")
        
        # Filter empty responses
        dataset = dataset.filter(lambda x: x['response'] and len(x['response'].strip()) > 0)
        
        # Take only num_samples
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Split: 80 train, 20 val
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split['train']
        val_dataset = split['test']
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Format and tokenize
        def format_and_tokenize(examples):
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                context = examples['context'][i]
                response = examples['response'][i]
                
                text = f"### Instruction:\n{instruction}\n\n"
                if context:
                    text += f"### Context:\n{context}\n\n"
                text += f"### Response:\n{response}"
                
                texts.append(text)
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=256,  # Shorter for quick test
                padding=False,
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        train_dataset = train_dataset.map(
            format_and_tokenize,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing"
        )
        
        val_dataset = val_dataset.map(
            format_and_tokenize,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing"
        )
        
        return train_dataset, val_dataset
    
    def run_quick_test(self):
        """Run a quick test of the training pipeline."""
        
        print("="*80)
        print("QUICK TEST - LLaMA-2-7B Fine-tuning on M2")
        print("Training on 100 samples to verify implementation")
        print("="*80)
        
        # Step 1: Load tokenizer
        print("\n[1/6] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✓ Tokenizer loaded")
        
        # Step 2: Load small dataset
        print("\n[2/6] Loading small dataset...")
        train_dataset, val_dataset = self.load_small_dataset(tokenizer, num_samples=100)
        print("✓ Dataset loaded")
        
        # Step 3: Load model
        print("\n[3/6] Loading LLaMA-2-7B model...")
        print("⏳ This will take 2-3 minutes...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for MPS stability
            device_map=None,  # Don't auto-assign devices
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to MPS if available
        if torch.backends.mps.is_available():
            print("Moving model to MPS device...")
            model = model.to("mps")
        
        print("✓ Model loaded")
        
        # Step 4: Apply LoRA
        print("\n[4/6] Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Just 2 modules for quick test
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        
        # Enable gradients for LoRA parameters
        model.enable_input_require_grads()
        model.train()
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✓ LoRA applied: {trainable:,} trainable params ({100*trainable/total:.2f}%)")
        
        # Step 5: Setup training
        print("\n[5/6] Setting up training...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=5,
            eval_steps=20,
            save_steps=50,
            eval_strategy="steps",
            fp16=False,  # Disable fp16 on MPS for stability
            gradient_checkpointing=False,  # Disable on MPS
            report_to="none",
            save_total_limit=1,
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
        print("✓ Trainer configured")
        
        # Step 6: Train!
        print("\n[6/6] Starting training...")
        print("⏳ This will take 10-20 minutes on M2 Pro")
        print("="*80)
        
        train_result = trainer.train()
        
        # Save results
        trainer.save_model()
        
        print("\n" + "="*80)
        print("✅ QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📊 Results:")
        print(f"   Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")
        print(f"   Training time: {train_result.metrics.get('train_runtime', 0):.1f} seconds")
        print(f"   Samples/second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
        
        print(f"\n💾 Model saved to: {self.output_dir}")
        print("\n✅ Your M2 can successfully run the training!")
        print("\n📝 Next steps:")
        print("   1. For full training, use: python finetune_llama2_m2.py")
        print("   2. Or use a GPU server for faster training")
        print("="*80)
        
        return train_result


def main():
    tester = QuickTester()
    tester.run_quick_test()


if __name__ == "__main__":
    main()
