"""
Evaluation script for AlpacaEval 2 and MT-Bench
CS5242 Assignment 7
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluates fine-tuned model vs base model."""
    
    def __init__(self, base_model_name: str, lora_model_path: str = None):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, use_lora: bool = False):
        """Load base model or LoRA-finetuned model."""
        print(f"Loading {'LoRA fine-tuned' if use_lora else 'base'} model...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if use_lora and self.lora_model_path:
            model = PeftModel.from_pretrained(model, self.lora_model_path)
            print("LoRA adapter loaded")
        
        model.eval()
        return model, tokenizer
    
    def generate_response(self, model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response for a given prompt."""
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response


class AlpacaEvalRunner:
    """Simplified AlpacaEval evaluation."""
    
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        
    def get_alpaca_eval_questions(self) -> List[str]:
        """Sample questions similar to AlpacaEval."""
        return [
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the main causes of climate change?",
            "How does a neural network learn?",
            "Explain the difference between supervised and unsupervised learning.",
            "What is the capital of France and what is it famous for?",
            "Write a short poem about nature.",
            "How do you make chocolate chip cookies?",
            "Explain quantum computing to a 10-year-old.",
            "What are the benefits of regular exercise?",
        ]
    
    def run_evaluation(self, output_file: str = "alpaca_eval_results.json"):
        """Run evaluation on both base and fine-tuned models."""
        print("\n" + "="*80)
        print("AlpacaEval Evaluation")
        print("="*80)
        
        questions = self.get_alpaca_eval_questions()
        results = []
        
        # Load both models
        base_model, base_tokenizer = self.evaluator.load_model(use_lora=False)
        ft_model, ft_tokenizer = self.evaluator.load_model(use_lora=True)
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question[:50]}...")
            
            # Base model response
            base_response = self.evaluator.generate_response(base_model, base_tokenizer, question)
            
            # Fine-tuned model response
            ft_response = self.evaluator.generate_response(ft_model, ft_tokenizer, question)
            
            results.append({
                "question": question,
                "base_response": base_response,
                "finetuned_response": ft_response
            })
            
            print(f"  Base: {base_response[:100]}...")
            print(f"  Fine-tuned: {ft_response[:100]}...")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
        return results


class MTBenchRunner:
    """Simplified MT-Bench evaluation (multi-turn conversations)."""
    
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        
    def get_mtbench_conversations(self) -> List[Dict]:
        """Sample multi-turn conversations."""
        return [
            {
                "turn1": "What is machine learning?",
                "turn2": "Can you give me a practical example?"
            },
            {
                "turn1": "Explain photosynthesis.",
                "turn2": "Why is it important for the ecosystem?"
            },
            {
                "turn1": "How do I bake a cake?",
                "turn2": "What if I don't have eggs?"
            },
            {
                "turn1": "What is blockchain technology?",
                "turn2": "How is it used in cryptocurrencies?"
            },
            {
                "turn1": "Explain the water cycle.",
                "turn2": "How does climate change affect it?"
            },
        ]
    
    def run_evaluation(self, output_file: str = "mtbench_results.json"):
        """Run multi-turn evaluation."""
        print("\n" + "="*80)
        print("MT-Bench Evaluation (Multi-Turn)")
        print("="*80)
        
        conversations = self.get_mtbench_conversations()
        results = []
        
        # Load both models
        base_model, base_tokenizer = self.evaluator.load_model(use_lora=False)
        ft_model, ft_tokenizer = self.evaluator.load_model(use_lora=True)
        
        for i, conv in enumerate(conversations, 1):
            print(f"\n[{i}/{len(conversations)}] Conversation")
            print(f"  Turn 1: {conv['turn1']}")
            print(f"  Turn 2: {conv['turn2']}")
            
            # Base model
            base_turn1 = self.evaluator.generate_response(base_model, base_tokenizer, conv['turn1'])
            base_turn2_prompt = f"{conv['turn1']}\n{base_turn1}\n\n{conv['turn2']}"
            base_turn2 = self.evaluator.generate_response(base_model, base_tokenizer, base_turn2_prompt)
            
            # Fine-tuned model
            ft_turn1 = self.evaluator.generate_response(ft_model, ft_tokenizer, conv['turn1'])
            ft_turn2_prompt = f"{conv['turn1']}\n{ft_turn1}\n\n{conv['turn2']}"
            ft_turn2 = self.evaluator.generate_response(ft_model, ft_tokenizer, ft_turn2_prompt)
            
            results.append({
                "conversation": i,
                "turn1_question": conv['turn1'],
                "turn2_question": conv['turn2'],
                "base_turn1": base_turn1,
                "base_turn2": base_turn2,
                "finetuned_turn1": ft_turn1,
                "finetuned_turn2": ft_turn2,
            })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
        return results


def main():
    """Main evaluation pipeline."""
    
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"
    LORA_MODEL = "./llama2-dolly-lora-full"
    
    print("="*80)
    print("Model Evaluation - AlpacaEval 2 & MT-Bench")
    print("="*80)
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA model: {LORA_MODEL}")
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected. Evaluation will be slow.")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(BASE_MODEL, LORA_MODEL)
    
    # Run AlpacaEval
    alpaca_runner = AlpacaEvalRunner(evaluator)
    alpaca_results = alpaca_runner.run_evaluation()
    
    # Run MT-Bench
    mtbench_runner = MTBenchRunner(evaluator)
    mtbench_results = mtbench_runner.run_evaluation()
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("Results saved:")
    print("  - alpaca_eval_results.json")
    print("  - mtbench_results.json")
    print()
    print("Review the results to compare base vs fine-tuned model performance.")
    print("="*80)


if __name__ == "__main__":
    main()
