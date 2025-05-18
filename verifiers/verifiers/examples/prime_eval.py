#!/usr/bin/env python
"""
Prime Number Verification Evaluation

This script evaluates the performance of a trained LLM on prime number verification
and factorization tasks without further training. It can be used to compare
performance across different models or to validate a model after training.
"""

import os
import argparse
import logging
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from datasets import Dataset

from transformers import AutoTokenizer
from verifiers.envs.prime_env import PrimeEnv
from verifiers.tools.prime_tools import is_prime, factorize, next_prime, prime_count, verify_factorization
from verifiers.inference.vllm_client import AsyncVLLMClient
from verifiers.parsers import XMLParser

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate LLM on prime number verification")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to evaluate")
parser.add_argument("--inference_url", type=str, default="http://localhost:8000", help="URL for the vLLM inference server")
parser.add_argument("--data_path", type=str, default=None, help="Path to prime dataset parquet file")
parser.add_argument("--output_file", type=str, default="prime_eval_results.json", help="Output file for evaluation results")
parser.add_argument("--difficulty", type=str, default="mixed", choices=["easy", "medium", "hard", "mixed"], 
                    help="Difficulty level for evaluation examples")
parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
parser.add_argument("--max_steps", type=int, default=5, help="Maximum steps in environment")
parser.add_argument("--benchmark_path", type=str, default=None, help="Optional path to benchmark JSON file")
parser.add_argument("--debug", action="store_true", help="Enable debug mode with fewer samples")
parser.add_argument("--use_cache", action="store_true", help="Enable vLLM cache for inference")
parser.add_argument("--compare_models", nargs="+", default=[], help="Additional models to compare against")
parser.add_argument("--detailed_output", action="store_true", help="Output detailed results for each sample")
parser.add_argument("--test_tier", type=int, default=None, help="Test only a specific difficulty tier")

args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_benchmark_data(benchmark_path: str, test_tier: Optional[int] = None) -> Dataset:
    """
    Load and format benchmark data from JSON file.
    
    Args:
        benchmark_path: Path to benchmark JSON file
        test_tier: Optional specific tier to test
        
    Returns:
        Formatted dataset for evaluation
    """
    logger.info(f"Loading benchmark data from {benchmark_path}")
    with open(benchmark_path, 'r') as f:
        benchmark = json.load(f)
        
    examples = []
    
    # Process tiers
    for tier_data in benchmark.get("tiers", []):
        tier_info = tier_data.get("tier_info", {})
        tier_id = tier_info.get("tier")
        
        # Skip if not the requested tier
        if test_tier is not None and tier_id != test_tier:
            continue
            
        # Process samples
        for sample in tier_data.get("samples", []):
            # Format each example for evaluation
            number = sample.get("n") or sample.get("number")
            if not number:
                continue
                
            # Get factors
            if "factors" in sample:
                if isinstance(sample["factors"], list):
                    factors = "×".join(map(str, sample["factors"]))
                else:
                    factors = sample["factors"]
            else:
                continue
            
            # Create different types of challenges based on tier
            challenge_types = []
            
            # Primality check for small numbers
            if int(str(number)) < 1000000:
                challenge_types.append({
                    "input": f"Is {number} a prime number?", 
                    "expected": "is_prime", 
                    "answer": "prime" if len(factors.split("×")) == 1 else "not prime",
                    "tier": tier_id
                })
            
            # Factorization for all numbers
            challenge_types.append({
                "input": f"Find the prime factorization of {number}.",
                "expected": "factorize",
                "answer": factors,
                "tier": tier_id,
                "bit_length": sample.get("bit_length")
            })
            
            # Verification challenge
            challenge_types.append({
                "input": f"Verify if these factors of {number} are correct: {factors.replace('×', ',')}",
                "expected": "verify",
                "answer": "correct",
                "tier": tier_id
            })
            
            examples.extend(challenge_types)
    
    # Convert to Dataset
    return Dataset.from_pandas(pd.DataFrame(examples))

async def evaluate_models():
    """
    Evaluate models on prime verification tasks.
    
    Evaluates the main model and any comparison models on prime verification tasks,
    measuring accuracy, reasoning quality, and efficiency.
    """
    # Create environment
    env = PrimeEnv(
        data_path=args.data_path,
        difficulty_level=args.difficulty,
        max_steps=args.max_steps
    )
    
    # Use benchmark data if provided
    if args.benchmark_path:
        eval_dataset = load_benchmark_data(args.benchmark_path, args.test_tier)
    else:
        eval_dataset = env.dataset
    
    # Limit dataset size
    if args.debug:
        eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))
    elif args.max_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))
    
    logger.info(f"Evaluating on {len(eval_dataset)} samples")
    
    # Models to evaluate
    models = [args.model] + args.compare_models
    
    all_results = {}
    
    # Set up the parser
    parser = XMLParser(fields=["reasoning", ("tool", "answer")])
    env_parser = XMLParser(fields=["result"])
    
    # Evaluate each model
    for model_name in models:
        logger.info(f"Evaluating model: {model_name}")
        
        # Set up tokenizer and client
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.chat_template:
            logger.warning(f"Model {model_name} doesn't have a chat template. Using default.")
            tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>system\n{{ message['content'] }}\n\n{% elif message['role'] == 'user' %}\nuser\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}\nassistant\n{{ message['content'] }}\n</s>\n{% endif %}\n{% endfor %}\n"

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create vLLM client
        client = AsyncVLLMClient(
            args.inference_url,
            tokenizer=tokenizer,
            enable_prefix_caching=args.use_cache,
            max_tokens=512,
        )
        
        # Prepare for evaluation
        env.inference_client = client
        
        # Evaluate on dataset
        results = await evaluate_model(env, eval_dataset, model_name, tokenizer)
        all_results[model_name] = results
    
    # Save all results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output_file}")
    
    # Print summary
    print_evaluation_summary(all_results)

async def evaluate_model(env, dataset, model_name, tokenizer):
    """
    Evaluate a specific model on the dataset.
    
    Args:
        env: Prime verification environment
        dataset: Evaluation dataset
        model_name: Name of the model being evaluated
        tokenizer: Tokenizer for the model
        
    Returns:
        Dictionary of evaluation results
    """
    results = {
        "model": model_name,
        "accuracy": [],
        "reasoning_quality": [],
        "tool_efficiency": [],
        "completion_time": [],
        "step_count": [],
        "samples": []
    }
    
    # Set up the parser
    parser = XMLParser(fields=["reasoning", ("tool", "answer")])
    env_parser = XMLParser(fields=["result"])
    
    # Evaluate in batches
    batch_size = args.batch_size
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {model_name}"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        for j, sample in enumerate(batch):
            try:
                # Run sample through environment
                sample_results = await env.run_sample(
                    sample,
                    max_steps=args.max_steps,
                    with_metrics=True
                )
                
                # Extract metrics
                metrics = sample_results.get("metrics", {})
                
                # Track performance
                accuracy = metrics.get("factorization_accuracy", 0.0)
                reasoning = metrics.get("reasoning_quality", 0.0)
                efficiency = metrics.get("tool_usage_efficiency", 0.0)
                steps = metrics.get("step_count", 0)
                
                results["accuracy"].append(accuracy)
                results["reasoning_quality"].append(reasoning)
                results["tool_efficiency"].append(efficiency)
                results["step_count"].append(steps)
                
                # Add detailed sample results if requested
                if args.detailed_output:
                    sample_detail = {
                        "input": sample["input"],
                        "expected": sample.get("expected"),
                        "expected_answer": sample.get("answer"),
                        "tier": sample.get("tier"),
                        "bit_length": sample.get("bit_length"),
                        "accuracy": accuracy,
                        "reasoning_quality": reasoning,
                        "tool_efficiency": efficiency,
                        "steps": steps,
                        "conversation": sample_results.get("messages", []),
                        "final_answer": sample_results.get("final_answer", "")
                    }
                    results["samples"].append(sample_detail)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i+j}: {e}")
                # Add failed sample with zero scores
                results["accuracy"].append(0.0)
                results["reasoning_quality"].append(0.0)
                results["tool_efficiency"].append(0.0)
                results["step_count"].append(0)
    
    # Calculate summary statistics
    summary = {
        "model": model_name,
        "average_accuracy": np.mean(results["accuracy"]),
        "median_accuracy": np.median(results["accuracy"]),
        "average_reasoning": np.mean(results["reasoning_quality"]),
        "average_efficiency": np.mean(results["tool_efficiency"]),
        "average_steps": np.mean(results["step_count"]),
        "perfect_score_rate": sum(1 for a in results["accuracy"] if a > 0.95) / len(results["accuracy"])
    }
    
    # Add summary to results
    results["summary"] = summary
    
    return results

def print_evaluation_summary(all_results):
    """Print a summary table of the evaluation results."""
    print("\n" + "="*80)
    print(f"PRIME VERIFICATION EVALUATION SUMMARY")
    print("="*80)
    
    # Print header
    header = ["Model", "Avg Accuracy", "Perfect Rate", "Avg Reasoning", "Avg Efficiency", "Avg Steps"]
    print(f"{header[0]:<30} {header[1]:<12} {header[2]:<12} {header[3]:<14} {header[4]:<14} {header[5]:<10}")
    print("-"*80)
    
    # Print each model's results
    for model_name, results in all_results.items():
        summary = results["summary"]
        print(f"{model_name:<30} "
              f"{summary['average_accuracy']:<12.4f} "
              f"{summary['perfect_score_rate']:<12.4f} "
              f"{summary['average_reasoning']:<14.4f} "
              f"{summary['average_efficiency']:<14.4f} "
              f"{summary['average_steps']:<10.2f}")
    
    print("="*80)
    print("NOTE: Perfect Rate = Proportion of samples with near-perfect accuracy (>0.95)")
    print("="*80)

async def main():
    """Main entry point."""
    # Print configuration
    logger.info("Configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Check if inference server is running
    import requests
    try:
        requests.get(args.inference_url, timeout=2)
        logger.info(f"Inference server found at {args.inference_url}")
    except:
        logger.error(f"Inference server not reachable at {args.inference_url}")
        logger.error("Please start the vLLM server using the following command:")
        logger.error(f"python -m verifiers.inference.vllm_serve --model {args.model} " +
                     "--tensor_parallel_size 4 --max_model_len 8192 --gpu_memory_utilization 0.9 " +
                     "--enable_prefix_caching True")
        return
    
    # Run evaluation
    await evaluate_models()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())