"""
Configuration file for LLaMA-2 Fine-tuning
Modify these parameters to customize your training and evaluation.
"""

# Model Configuration
MODEL_CONFIG = {
    'base_model_name': 'meta-llama/Llama-2-7b-hf',
    'use_4bit_quantization': True,  # Set to False if you have enough VRAM (>80GB)
    'torch_dtype': 'bfloat16',  # Options: 'float16', 'bfloat16', 'float32'
}

# LoRA Configuration
LORA_CONFIG = {
    'r': 16,  # LoRA rank (higher = more capacity, more memory)
    'lora_alpha': 32,  # LoRA scaling factor (typically 2x rank)
    'lora_dropout': 0.05,  # Dropout for regularization
    'target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ],  # Which layers to apply LoRA to
    'bias': 'none',  # Options: 'none', 'all', 'lora_only'
}

# Dataset Configuration
DATASET_CONFIG = {
    'dataset_name': 'databricks/databricks-dolly-15k',
    'max_length': 512,  # Maximum sequence length
    'train_split': 0.8,  # 80% for training
    'val_split': 0.1,  # 10% for validation
    'test_split': 0.1,  # 10% for testing
    'filter_empty_responses': True,
}

# Training Configuration
TRAINING_CONFIG = {
    'output_dir': './llama2-dolly-lora',
    'num_train_epochs': 3,
    'per_device_train_batch_size': 4,  # Reduce if OOM
    'per_device_eval_batch_size': 4,
    'gradient_accumulation_steps': 4,  # Effective batch size = batch_size * accumulation_steps
    'learning_rate': 2e-4,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'save_steps': 500,
    'eval_steps': 500,
    'save_total_limit': 2,  # Keep only 2 best checkpoints
    'gradient_checkpointing': True,
    'optim': 'paged_adamw_8bit',  # Memory-efficient optimizer
    'fp16': False,  # Don't use fp16 with bf16
    'bf16': True,  # Use bfloat16 for better stability
    'max_grad_norm': 1.0,
    'seed': 42,
}

# Evaluation Configuration
EVAL_CONFIG = {
    'evaluation_output_dir': './evaluation_results',
    'num_alpaca_samples': 10,  # Number of AlpacaEval samples to evaluate
    'num_mtbench_samples': 5,  # Number of MT-Bench conversations to evaluate
    'max_new_tokens': 256,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_cpu': False,  # Force CPU usage (very slow)
    'device_map': 'auto',  # Options: 'auto', 'balanced', 'sequential'
    'low_cpu_mem_usage': True,
}

# Paths
PATHS = {
    'model_save_path': './llama2-dolly-lora',
    'training_curves_path': './llama2-dolly-lora/training_curves.png',
    'training_history_path': './llama2-dolly-lora/training_history.json',
    'base_eval_dir': './evaluation_results/base_model',
    'finetuned_eval_dir': './evaluation_results/finetuned_model',
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'disable_tqdm': False,  # Set to True to disable progress bars
    'report_to': 'none',  # Options: 'none', 'tensorboard', 'wandb'
}

def get_effective_batch_size():
    """Calculate the effective batch size."""
    return (TRAINING_CONFIG['per_device_train_batch_size'] * 
            TRAINING_CONFIG['gradient_accumulation_steps'])

def get_total_training_steps(dataset_size):
    """Calculate total training steps."""
    steps_per_epoch = dataset_size // get_effective_batch_size()
    return steps_per_epoch * TRAINING_CONFIG['num_train_epochs']

def print_config():
    """Print current configuration."""
    print("="*80)
    print("Current Configuration")
    print("="*80)
    
    print("\n[Model]")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n[LoRA]")
    for key, value in LORA_CONFIG.items():
        if key != 'target_modules':
            print(f"  {key}: {value}")
    
    print("\n[Dataset]")
    for key, value in DATASET_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n[Training]")
    for key, value in TRAINING_CONFIG.items():
        if key not in ['output_dir']:
            print(f"  {key}: {value}")
    print(f"  effective_batch_size: {get_effective_batch_size()}")
    
    print("\n[Evaluation]")
    for key, value in EVAL_CONFIG.items():
        if key not in ['evaluation_output_dir']:
            print(f"  {key}: {value}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_config()
