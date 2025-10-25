"""
Evaluation Script for LLaMA-2-7B Fine-tuned Model
Supports AlpacaEval 2 and MT-Bench evaluations

This script:
1. Loads the fine-tuned model
2. Runs inference on AlpacaEval 2 dataset
3. Runs inference on MT-Bench dataset
4. Compares results with base model
"""

import os
import sys

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Handles model loading and evaluation on various benchmarks."""
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self, use_lora: bool = True):
        """
        Load the base model and optionally the LoRA adapter.
        
        Args:
            use_lora: Whether to load the LoRA adapter
        """
        print(f"Loading tokenizer from {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Loading base model from {self.base_model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if use_lora and self.lora_model_path:
            print(f"Loading LoRA adapter from {self.lora_model_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_model_path,
                torch_dtype=torch.float16
            )
            self.model = self.model.merge_and_unload()  # Merge LoRA weights for faster inference
        
        self.model.eval()
        print("Model loaded successfully!")
        
    def generate_response(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response for a given instruction.
        
        Args:
            instruction: The instruction/question
            context: Optional context
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated response string
        """
        # Format the prompt
        prompt = f"### Instruction:\n{instruction}\n\n"
        if context:
            prompt += f"### Context:\n{context}\n\n"
        prompt += "### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the response part
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after "### Response:"
        if "### Response:" in full_output:
            response = full_output.split("### Response:")[-1].strip()
        else:
            response = full_output[len(prompt):].strip()
        
        return response


class AlpacaEvalRunner:
    """Handles AlpacaEval 2 evaluation."""
    
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        
    def load_alpaca_eval_dataset(self, dataset_path: str = None) -> List[Dict]:
        """
        Load AlpacaEval dataset.
        
        If dataset_path is None, creates a sample dataset for demonstration.
        For full evaluation, download from: https://github.com/tatsu-lab/alpaca_eval
        """
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            # Sample dataset for demonstration
            print("Using sample AlpacaEval-style dataset for demonstration...")
            return [
                {
                    "instruction": "What are the three primary colors?",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "Explain the concept of machine learning in simple terms.",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "Write a short poem about nature.",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "How do you make a cup of tea?",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "What is the capital of France?",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "Explain the difference between supervised and unsupervised learning.",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "Write a function to calculate the factorial of a number.",
                    "input": "Language: Python",
                    "output": ""
                },
                {
                    "instruction": "Describe the water cycle.",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "What are the benefits of regular exercise?",
                    "input": "",
                    "output": ""
                },
                {
                    "instruction": "Explain quantum computing to a 10-year-old.",
                    "input": "",
                    "output": ""
                }
            ]
    
    def run_evaluation(
        self,
        dataset_path: str = None,
        output_path: str = "./alpaca_eval_results.json",
        num_samples: int = None
    ):
        """
        Run AlpacaEval evaluation.
        
        Args:
            dataset_path: Path to AlpacaEval dataset
            output_path: Where to save results
            num_samples: Number of samples to evaluate (None for all)
        """
        print("Loading AlpacaEval dataset...")
        dataset = self.load_alpaca_eval_dataset(dataset_path)
        
        if num_samples:
            dataset = dataset[:num_samples]
        
        print(f"Running inference on {len(dataset)} examples...")
        results = []
        
        for example in tqdm(dataset, desc="AlpacaEval"):
            instruction = example.get('instruction', '')
            context = example.get('input', '')
            
            response = self.evaluator.generate_response(
                instruction=instruction,
                context=context
            )
            
            results.append({
                'instruction': instruction,
                'input': context,
                'output': response,
                'reference': example.get('output', '')
            })
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return results


class MTBenchRunner:
    """Handles MT-Bench evaluation."""
    
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
    
    def load_mt_bench_dataset(self, dataset_path: str = None) -> List[Dict]:
        """
        Load MT-Bench dataset.
        
        If dataset_path is None, creates a sample multi-turn conversation dataset.
        For full evaluation, download from: https://github.com/lm-sys/FastChat
        """
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            # Sample multi-turn conversations for demonstration
            print("Using sample MT-Bench-style dataset for demonstration...")
            return [
                {
                    "question_id": 1,
                    "category": "writing",
                    "turns": [
                        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
                        "Rewrite your previous response. Start every sentence with the letter A."
                    ]
                },
                {
                    "question_id": 2,
                    "category": "reasoning",
                    "turns": [
                        "If a train leaves New York at 60 mph and another train leaves Chicago (800 miles away) at 80 mph, when will they meet?",
                        "What if the first train leaves 30 minutes earlier?"
                    ]
                },
                {
                    "question_id": 3,
                    "category": "coding",
                    "turns": [
                        "Write a Python function to find the longest common subsequence of two strings.",
                        "Now optimize it for space complexity."
                    ]
                },
                {
                    "question_id": 4,
                    "category": "math",
                    "turns": [
                        "What is the derivative of x^3 + 2x^2 - 5x + 7?",
                        "What is the second derivative?"
                    ]
                },
                {
                    "question_id": 5,
                    "category": "roleplay",
                    "turns": [
                        "You are a helpful assistant. Explain photosynthesis to a 5th grader.",
                        "Now explain it using an analogy with a kitchen."
                    ]
                }
            ]
    
    def run_evaluation(
        self,
        dataset_path: str = None,
        output_path: str = "./mt_bench_results.json",
        num_samples: int = None
    ):
        """
        Run MT-Bench evaluation.
        
        Args:
            dataset_path: Path to MT-Bench dataset
            output_path: Where to save results
            num_samples: Number of samples to evaluate (None for all)
        """
        print("Loading MT-Bench dataset...")
        dataset = self.load_mt_bench_dataset(dataset_path)
        
        if num_samples:
            dataset = dataset[:num_samples]
        
        print(f"Running inference on {len(dataset)} multi-turn conversations...")
        results = []
        
        for conversation in tqdm(dataset, desc="MT-Bench"):
            turns = conversation.get('turns', [])
            conversation_history = []
            
            for turn_idx, turn_question in enumerate(turns):
                # For multi-turn, we include previous context
                if turn_idx == 0:
                    instruction = turn_question
                    context = ""
                else:
                    # Include previous turn as context
                    instruction = turn_question
                    context = f"Previous question: {turns[turn_idx-1]}\nPrevious answer: {conversation_history[-1]['response']}"
                
                response = self.evaluator.generate_response(
                    instruction=instruction,
                    context=context
                )
                
                conversation_history.append({
                    'turn': turn_idx + 1,
                    'question': turn_question,
                    'response': response
                })
            
            results.append({
                'question_id': conversation.get('question_id'),
                'category': conversation.get('category', 'general'),
                'turns': conversation_history
            })
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return results


def compare_models(base_output_dir: str, finetuned_output_dir: str):
    """
    Compare base model and fine-tuned model outputs side-by-side.
    """
    print("\n" + "="*80)
    print("Model Comparison: Base vs Fine-tuned")
    print("="*80)
    
    # Load AlpacaEval results
    base_alpaca_path = os.path.join(base_output_dir, "alpaca_eval_results.json")
    ft_alpaca_path = os.path.join(finetuned_output_dir, "alpaca_eval_results.json")
    
    if os.path.exists(base_alpaca_path) and os.path.exists(ft_alpaca_path):
        with open(base_alpaca_path, 'r') as f:
            base_alpaca = json.load(f)
        with open(ft_alpaca_path, 'r') as f:
            ft_alpaca = json.load(f)
        
        print(f"\nAlpacaEval: Evaluated {len(ft_alpaca)} examples")
        print("Sample comparison (first 2 examples):")
        print("-"*80)
        
        for i in range(min(2, len(base_alpaca))):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {base_alpaca[i]['instruction']}")
            print(f"\nBase model: {base_alpaca[i]['output'][:200]}...")
            print(f"\nFine-tuned model: {ft_alpaca[i]['output'][:200]}...")
            print("-"*80)
    
    # Load MT-Bench results
    base_mt_path = os.path.join(base_output_dir, "mt_bench_results.json")
    ft_mt_path = os.path.join(finetuned_output_dir, "mt_bench_results.json")
    
    if os.path.exists(base_mt_path) and os.path.exists(ft_mt_path):
        with open(base_mt_path, 'r') as f:
            base_mt = json.load(f)
        with open(ft_mt_path, 'r') as f:
            ft_mt = json.load(f)
        
        print(f"\nMT-Bench: Evaluated {len(ft_mt)} conversations")
        print("Sample comparison (first conversation):")
        print("-"*80)
        
        if len(base_mt) > 0:
            print(f"\nCategory: {base_mt[0]['category']}")
            for turn in base_mt[0]['turns']:
                print(f"\nTurn {turn['turn']}: {turn['question']}")
                print(f"Base model: {turn['response'][:200]}...")
            
            for turn in ft_mt[0]['turns']:
                print(f"\nTurn {turn['turn']}: {turn['question']}")
                print(f"Fine-tuned model: {turn['response'][:200]}...")
            print("-"*80)


def main():
    """Main evaluation pipeline."""
    
    # Configuration
    BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    LORA_MODEL_PATH = "./llama2-dolly-lora"  # Path to saved LoRA adapter
    
    print("="*80)
    print("LLaMA-2-7B Model Evaluation")
    print("="*80)
    
    # Create output directories
    os.makedirs("./evaluation_results/base_model", exist_ok=True)
    os.makedirs("./evaluation_results/finetuned_model", exist_ok=True)
    
    # Evaluate base model
    print("\n[1] Evaluating BASE MODEL...")
    print("-"*80)
    base_evaluator = ModelEvaluator(
        base_model_name=BASE_MODEL_NAME,
        lora_model_path=None
    )
    base_evaluator.load_model(use_lora=False)
    
    # AlpacaEval for base model
    base_alpaca_runner = AlpacaEvalRunner(base_evaluator)
    base_alpaca_runner.run_evaluation(
        output_path="./evaluation_results/base_model/alpaca_eval_results.json",
        num_samples=10  # Adjust as needed
    )
    
    # MT-Bench for base model
    base_mt_runner = MTBenchRunner(base_evaluator)
    base_mt_runner.run_evaluation(
        output_path="./evaluation_results/base_model/mt_bench_results.json",
        num_samples=5  # Adjust as needed
    )
    
    # Clear memory
    del base_evaluator
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\n[2] Evaluating FINE-TUNED MODEL...")
    print("-"*80)
    ft_evaluator = ModelEvaluator(
        base_model_name=BASE_MODEL_NAME,
        lora_model_path=LORA_MODEL_PATH
    )
    ft_evaluator.load_model(use_lora=True)
    
    # AlpacaEval for fine-tuned model
    ft_alpaca_runner = AlpacaEvalRunner(ft_evaluator)
    ft_alpaca_runner.run_evaluation(
        output_path="./evaluation_results/finetuned_model/alpaca_eval_results.json",
        num_samples=10  # Adjust as needed
    )
    
    # MT-Bench for fine-tuned model
    ft_mt_runner = MTBenchRunner(ft_evaluator)
    ft_mt_runner.run_evaluation(
        output_path="./evaluation_results/finetuned_model/mt_bench_results.json",
        num_samples=5  # Adjust as needed
    )
    
    # Compare results
    print("\n[3] Comparing results...")
    print("-"*80)
    compare_models(
        "./evaluation_results/base_model",
        "./evaluation_results/finetuned_model"
    )
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("Results saved in ./evaluation_results/")
    print("="*80)


if __name__ == "__main__":
    main()
