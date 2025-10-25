"""
Test inference with the fine-tuned model
CS5242 Assignment 7
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')


def load_finetuned_model(base_model_name: str = "meta-llama/Llama-2-7b-hf",
                         lora_path: str = "./llama2-dolly-lora-full"):
    """Load fine-tuned model."""
    
    print("Loading fine-tuned model...")
    print(f"  Base: {base_model_name}")
    print(f"  LoRA: {lora_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, context: str = "", 
                     max_new_tokens: int = 256, temperature: float = 0.7):
    """Generate response for given instruction."""
    
    # Format prompt
    prompt = f"### Instruction:\n{instruction}\n\n"
    if context:
        prompt += f"### Context:\n{context}\n\n"
    prompt += "### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def run_test_examples(model, tokenizer):
    """Run a few test examples."""
    
    test_cases = [
        {
            "instruction": "Explain what is machine learning in simple terms.",
            "context": ""
        },
        {
            "instruction": "Write a Python function to check if a number is prime.",
            "context": ""
        },
        {
            "instruction": "What are the benefits of regular exercise?",
            "context": ""
        },
        {
            "instruction": "Summarize the main idea.",
            "context": "Photosynthesis is a process used by plants to convert light energy into chemical energy. This energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water."
        },
    ]
    
    print("\n" + "="*80)
    print("Running Test Examples")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'='*80}")
        print(f"Instruction: {test['instruction']}")
        if test['context']:
            print(f"Context: {test['context']}")
        print(f"\nGenerating response...")
        
        response = generate_response(model, tokenizer, test['instruction'], test['context'])
        
        print(f"\nResponse:\n{response}")
        print(f"{'='*80}")
    
    print("\n✅ All test examples completed!")


def interactive_mode(model, tokenizer):
    """Interactive chat mode."""
    
    print("\n" + "="*80)
    print("Interactive Mode")
    print("="*80)
    print("Enter your instructions (type 'exit' to quit)")
    print()
    
    while True:
        instruction = input("Instruction: ").strip()
        
        if instruction.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode.")
            break
        
        if not instruction:
            continue
        
        context = input("Context (optional, press Enter to skip): ").strip()
        
        print("\nGenerating response...\n")
        response = generate_response(model, tokenizer, instruction, context)
        
        print(f"Response:\n{response}\n")
        print("-"*80 + "\n")


def main():
    """Main inference testing."""
    
    print("="*80)
    print("LLaMA-2-7B Fine-Tuned Model - Inference Testing")
    print("="*80)
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU (will be slow)")
    print()
    
    # Load model
    model, tokenizer = load_finetuned_model()
    
    # Run test examples
    run_test_examples(model, tokenizer)
    
    # Interactive mode
    print("\n" + "="*80)
    response = input("Enter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_mode(model, tokenizer)
    
    print("\n" + "="*80)
    print("✅ Inference testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
