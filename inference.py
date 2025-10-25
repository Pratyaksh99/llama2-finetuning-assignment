"""
Quick Inference Script for Testing the Fine-tuned Model

This script provides a simple interface to test the fine-tuned model
with custom instructions.
"""

import os
import sys

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_model(base_model_name: str, lora_path: str = None, device: str = None):
    """Load model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter if provided
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device,
    instruction: str,
    context: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Generate response for given instruction."""
    # Format prompt
    prompt = f"### Instruction:\n{instruction}\n\n"
    if context:
        prompt += f"### Context:\n{context}\n\n"
    prompt += "### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in full_output:
        response = full_output.split("### Response:")[-1].strip()
    else:
        response = full_output[len(prompt):].strip()
    
    return response


def interactive_mode(model, tokenizer, device):
    """Run interactive question-answering mode."""
    print("\n" + "="*80)
    print("Interactive Mode - Enter your instructions (type 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        instruction = input("\n📝 Instruction: ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not instruction:
            continue
        
        context = input("📄 Context (optional, press Enter to skip): ").strip()
        
        print("\n🤖 Generating response...")
        response = generate_response(model, tokenizer, device, instruction, context)
        
        print(f"\n💡 Response:\n{response}\n")
        print("-"*80)


def batch_mode(model, tokenizer, device, test_cases):
    """Run predefined test cases."""
    print("\n" + "="*80)
    print("Batch Mode - Running Test Cases")
    print("="*80 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'='*80}")
        print(f"\n📝 Instruction: {test['instruction']}")
        
        if test.get('context'):
            print(f"📄 Context: {test['context']}")
        
        response = generate_response(
            model, tokenizer, device,
            test['instruction'],
            test.get('context', '')
        )
        
        print(f"\n💡 Response:\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Quick inference with fine-tuned LLaMA-2")
    parser.add_argument(
        '--base_model',
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        '--lora_path',
        type=str,
        default="./llama2-dolly-lora",
        help="Path to LoRA adapter (omit to use base model)"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'batch'],
        default='interactive',
        help="Inference mode"
    )
    parser.add_argument(
        '--no_lora',
        action='store_true',
        help="Use base model without LoRA"
    )
    
    args = parser.parse_args()
    
    # Load model
    lora_path = None if args.no_lora else args.lora_path
    model, tokenizer, device = load_model(args.base_model, lora_path)
    
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer, device)
    else:
        # Predefined test cases
        test_cases = [
            {
                'instruction': 'What are the three primary colors?',
                'context': ''
            },
            {
                'instruction': 'Explain machine learning in simple terms.',
                'context': ''
            },
            {
                'instruction': 'Write a Python function to calculate factorial.',
                'context': 'Use recursion.'
            },
            {
                'instruction': 'What is the capital of France?',
                'context': ''
            },
            {
                'instruction': 'Describe the benefits of regular exercise.',
                'context': ''
            },
            {
                'instruction': 'Explain quantum computing to a 10-year-old.',
                'context': ''
            },
            {
                'instruction': 'Write a short poem about nature.',
                'context': ''
            },
            {
                'instruction': 'What is the difference between supervised and unsupervised learning?',
                'context': ''
            }
        ]
        batch_mode(model, tokenizer, device, test_cases)


if __name__ == "__main__":
    main()
