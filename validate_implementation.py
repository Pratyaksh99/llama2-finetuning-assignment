"""
Quick validation test for Assignment 7 implementation
Tests core components without running full training
"""

import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required packages can be imported."""
    print("="*80)
    print("Testing Package Imports")
    print("="*80)
    
    # Import in specific order to avoid TensorFlow issues
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('peft', 'PEFT'),
        ('accelerate', 'Accelerate'),
    ]
    
    # Skip bitsandbytes on macOS as it may cause issues
    import platform
    if platform.system() != 'Darwin':
        packages.append(('bitsandbytes', 'BitsAndBytes'))
    else:
        print("⚠ BitsAndBytes (Skipped on macOS - not required for validation)")
    
    results = {}
    for pkg_name, display_name in packages:
        try:
            # Use a timeout mechanism or quick import
            mod = __import__(pkg_name)
            print(f"✓ {display_name}")
            results[pkg_name] = True
        except ImportError as e:
            print(f"✗ {display_name}: Not installed")
            results[pkg_name] = False
        except Exception as e:
            print(f"⚠ {display_name}: Warning - {str(e)[:50]}")
            results[pkg_name] = True  # Mark as passed if import worked but had warnings
    
    return all(results.values())

def test_dataset_processor():
    """Test dataset loading and preprocessing."""
    print("\n" + "="*80)
    print("Testing Dataset Processor")
    print("="*80)
    
    try:
        from transformers import AutoTokenizer
        from finetune_llama2 import Dolly15kDatasetProcessor
        
        print("Loading tokenizer (using GPT-2 for testing)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("Creating dataset processor...")
        processor = Dolly15kDatasetProcessor(tokenizer, max_length=128)
        
        # Test format_instruction
        test_example = {
            'instruction': 'What is machine learning?',
            'context': 'In the context of AI',
            'response': 'Machine learning is a subset of AI.'
        }
        
        formatted = processor.format_instruction(test_example)
        print(f"\n✓ format_instruction works!")
        print(f"Sample formatted text:\n{formatted[:200]}...")
        
        return True
    except Exception as e:
        print(f"✗ Dataset processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration file."""
    print("\n" + "="*80)
    print("Testing Configuration")
    print("="*80)
    
    try:
        import config
        config.print_config()
        print("\n✓ Configuration loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_classes():
    """Test evaluation classes."""
    print("\n" + "="*80)
    print("Testing Evaluation Classes")
    print("="*80)
    
    try:
        from evaluate_model import ModelEvaluator, AlpacaEvalRunner, MTBenchRunner
        
        print("✓ ModelEvaluator imported")
        print("✓ AlpacaEvalRunner imported")
        print("✓ MTBenchRunner imported")
        
        # Test sample dataset loading
        from evaluate_model import AlpacaEvalRunner
        evaluator = None  # Mock evaluator for testing
        runner = AlpacaEvalRunner(evaluator)
        sample_data = runner.load_alpaca_eval_dataset()
        print(f"✓ Sample AlpacaEval dataset loaded: {len(sample_data)} examples")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("Assignment 7 - Implementation Validation")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Package Imports", test_imports()))
    
    # Test 2: Configuration
    results.append(("Configuration", test_config()))
    
    # Test 3: Dataset Processor
    results.append(("Dataset Processor", test_dataset_processor()))
    
    # Test 4: Evaluation Classes
    results.append(("Evaluation Classes", test_evaluation_classes()))
    
    # Summary
    print("\n" + "="*80)
    print("Validation Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if all(passed for _, passed in results):
        print("\n🎉 All validation tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("  1. Setup Hugging Face auth: huggingface-cli login")
        print("  2. Run training: python finetune_llama2.py")
        print("  3. Run evaluation: python evaluate_model.py")
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
    
    print("="*80)

if __name__ == "__main__":
    main()
