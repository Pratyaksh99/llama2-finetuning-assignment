"""
Instruction Fine-Tuning of LLaMA-2-7B with PEFT (LoRA) on Dolly-15K
OPTIMIZED FOR APPLE M2 (MPS backend)

This script implements:
1. Dataset loading and preprocessing (Dolly-15k)
2. LoRA-based parameter-efficient fine-tuning
3. Training with loss tracking and validation
4. Model saving and checkpointing

Key differences from GPU version:
- No 4-bit quantization (not supported on macOS)
- Uses MPS (Metal Performance Shaders) backend
- Lower precision (float16) instead of bfloat16
- Smaller batch sizes for memory management
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
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class Dolly15kDatasetProcessor:
    """Handles Dolly-15k dataset loading, preprocessing, and splitting."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_instruction(self, example: Dict) -> str:
        """
        Format a single example into the instruction-response format.
        """
        instruction = example.get('instruction', '').strip()
        context = example.get('context', '').strip()
        response = example.get('response', '').strip()
        
        formatted_text = f"### Instruction:\n{instruction}\n\n"
        
        if context:
            formatted_text += f"### Context:\n{context}\n\n"
        
        formatted_text += f"### Response:\n{response}"
        
        return formatted_text
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize and prepare examples for training."""
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
    
    def load_and_split_dataset(self):
        """Load Dolly-15k dataset, filter, and split."""
        print("Loading databricks-dolly-15k dataset...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        print(f"Original dataset size: {len(dataset)}")
        
        # Filter out examples with empty responses
        dataset = dataset.filter(lambda x: x['response'] and len(x['response'].strip()) > 0)
        print(f"After filtering empty responses: {len(dataset)}")
        
        # Split into train (80%), validation (10%), test (10%)
        train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_split['train']
        
        val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset"
        )
        
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test dataset"
        )
        
        return train_dataset, val_dataset, test_dataset


class LLaMA2LoRATrainer:
    """Handles LLaMA-2-7B model loading, LoRA configuration, and training for M2."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "./llama2-dolly-lora-m2"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.training_history = {'train_loss': [], 'eval_loss': [], 'epochs': []}
        
        # Check if MPS is available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load LLaMA-2-7B model and tokenizer optimized for M2."""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model in float16 (no quantization on macOS)
        print("Loading model in float16 (this may take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=None,  # Don't use auto device mapping
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to MPS if available
        if self.device == "mps":
            print("Moving model to MPS device...")
            self.model = self.model.to("mps")
        
        print(f"Model loaded successfully!")
        return self.model, self.tokenizer
    
    def configure_lora(
        self,
        r: int = 8,  # Smaller rank for faster training on M2
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ):
        """Configure LoRA for parameter-efficient fine-tuning."""
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        print("Applying LoRA configuration...")
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Total parameters: {total_params:,}")
        
        return self.model
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        num_epochs: int = 1,  # Reduced for M2
        per_device_train_batch_size: int = 1,  # Smaller for M2
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,  # Higher to maintain effective batch size
        learning_rate: float = 2e-4,
        warmup_steps: int = 50,
        logging_steps: int = 10,
        save_steps: int = 200,
        eval_steps: int = 200
    ):
        """Train the model using Hugging Face Trainer."""
        
        # Training arguments optimized for M2
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,  # Use fp16 on M2
            bf16=False,  # bf16 not well supported on MPS
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=2,
            logging_dir=f"{self.output_dir}/logs",
            remove_unused_columns=False,
            # MPS is auto-detected, no need to specify
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Custom callback to track training history
        class TrainingHistoryCallback(Trainer):
            def __init__(self, *args, history_dict=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.history_dict = history_dict
            
            def log(self, logs: Dict[str, float]) -> None:
                super().log(logs)
                
                if self.history_dict is not None:
                    if 'loss' in logs:
                        self.history_dict['train_loss'].append(logs['loss'])
                    if 'eval_loss' in logs:
                        self.history_dict['eval_loss'].append(logs['eval_loss'])
                        self.history_dict['epochs'].append(self.state.epoch)
        
        # Initialize trainer
        trainer = TrainingHistoryCallback(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            history_dict=self.training_history
        )
        
        # Disable cache for gradient checkpointing
        self.model.config.use_cache = False
        
        print("Starting training...")
        print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
        print(f"This will be SLOW on M2 - consider running only a few steps for testing")
        
        train_result = trainer.train()
        
        # Save the final model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return trainer, self.training_history
    
    def plot_training_curves(self, save_path: str = None):
        """Plot and save training curves."""
        if not self.training_history['eval_loss']:
            print("No evaluation history available for plotting.")
            return
        
        plt.figure(figsize=(10, 6))
        
        epochs = self.training_history['epochs']
        eval_loss = self.training_history['eval_loss']
        
        plt.plot(epochs, eval_loss, 'b-', marker='o', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves (M2)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()
        
        # Save history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main training pipeline for M2."""
    
    # Configuration for M2
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    OUTPUT_DIR = "./llama2-dolly-lora-m2"
    MAX_LENGTH = 512
    
    # LoRA configuration (smaller for M2)
    LORA_R = 8  # Reduced from 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Training configuration (optimized for M2)
    NUM_EPOCHS = 1  # Start with 1 epoch for testing
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 50
    
    print("="*80)
    print("LLaMA-2-7B Instruction Fine-Tuning with LoRA on Dolly-15k")
    print("OPTIMIZED FOR APPLE M2")
    print("="*80)
    
    # Check device
    if torch.backends.mps.is_available():
        print("\n✅ MPS (Metal Performance Shaders) available - training will use GPU acceleration")
    else:
        print("\n⚠️  MPS not available - training will use CPU (VERY SLOW)")
    
    # Step 1: Load model and tokenizer
    print("\n[Step 1] Loading model and tokenizer...")
    trainer = LLaMA2LoRATrainer(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )
    model, tokenizer = trainer.load_model_and_tokenizer()
    
    # Step 2: Configure LoRA
    print("\n[Step 2] Configuring LoRA...")
    model = trainer.configure_lora(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    
    # Step 3: Load and preprocess dataset
    print("\n[Step 3] Loading and preprocessing Dolly-15k dataset...")
    dataset_processor = Dolly15kDatasetProcessor(tokenizer, max_length=MAX_LENGTH)
    train_dataset, val_dataset, test_dataset = dataset_processor.load_and_split_dataset()
    
    # Step 4: Train the model
    print("\n[Step 4] Starting training...")
    print("⚠️  NOTE: Training on M2 will be SLOW. Consider:")
    print("   - Using a smaller subset for testing")
    print("   - Running on a GPU server for full training")
    print("   - Or let it run overnight for full training")
    
    trained_model, history = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        num_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS
    )
    
    # Step 5: Plot training curves
    print("\n[Step 5] Plotting training curves...")
    trainer.plot_training_curves()
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
