"""
Quick training test script
Tests setup with small subset (1000 samples) before full training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
sys.path.append('.')

# Import main training script
from train_full import Dolly15kDatasetProcessor, main as train_main

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def quick_test():
    """Run quick training test with 1000 samples."""
    
    print("="*80)
    print("Quick Training Test (1000 samples)")
    print("="*80)
    print()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected. Use train_full.py on GPU droplet.")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Modify config for quick test
    import train_full
    
    # Monkey patch for quick test
    original_load = Dolly15kDatasetProcessor.load_and_split_dataset
    
    def quick_load(self):
        print("Loading subset of 1000 samples for quick test...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        dataset = dataset.filter(lambda x: x['response'] and len(x['response'].strip()) > 0)
        dataset = dataset.select(range(min(1000, len(dataset))))
        
        train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_split['train']
        val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        
        print(f"Quick test - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Tokenize
        train_dataset = train_dataset.map(self.preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(self.preprocess_function, batched=True, remove_columns=val_dataset.column_names)
        test_dataset = test_dataset.map(self.preprocess_function, batched=True, remove_columns=test_dataset.column_names)
        
        return train_dataset, val_dataset, test_dataset
    
    Dolly15kDatasetProcessor.load_and_split_dataset = quick_load
    
    # Run training
    train_main()


if __name__ == "__main__":
    quick_test()
