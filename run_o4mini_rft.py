#!/usr/bin/env python3
"""
run_o4mini_rft.py

Execute the complete reinforcement fine-tuning pipeline for o4-mini on prime factorization.
This script handles data generation, training, monitoring, and evaluation.
"""

import os
import sys
import json
import time
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from prime_rft_model import FactorizationRFTTrainer, OpenAIAPI
from generate_o4mini_rft_data import generate_balanced_dataset, optimize_tier_distribution

def setup_environment() -> bool:
    """
    Check if required environment variables are set.
    
    Returns:
        True if environment is properly set up, False otherwise
    """
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key as an environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return False
    
    return True

def run_rft_pipeline(
    num_samples: int = 500,
    max_value: int = 10**12,
    optimize_distribution: bool = True,
    model_suffix: str = None,
    monitor_training: bool = True,
    evaluate_model: bool = True,
    seed: int = 42,
    output_dir: str = "./rft_data"
) -> Dict[str, Any]:
    """
    Run the complete RFT pipeline for o4-mini on prime factorization.
    
    Args:
        num_samples: Number of samples to generate
        max_value: Maximum value for generated numbers
        optimize_distribution: Whether to use optimized tier distribution
        model_suffix: Suffix for the fine-tuned model name
        monitor_training: Whether to monitor training progress
        evaluate_model: Whether to evaluate the model after training
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with pipeline results
    """
    print("="*80)
    print(f"Starting o4-mini RFT Pipeline for Prime Factorization")
    print("="*80)
    
    # Step 1: Generate dataset
    print(f"\n📊 Generating dataset with {num_samples} samples...")
    tier_distribution = optimize_tier_distribution(max_value) if optimize_distribution else None
    
    dataset_result = generate_balanced_dataset(
        output_dir=output_dir,
        num_samples=num_samples,
        tier_distribution=tier_distribution,
        max_value=max_value,
        seed=seed
    )
    
    run_dir = dataset_result["run_dir"]
    data_dir = dataset_result["data_dir"]
    
    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")
    test_file = os.path.join(data_dir, "test.jsonl")
    
    # Step 2: Initialize trainer with optimized configuration
    print("\n🛠️ Initializing o4-mini RFT trainer...")
    trainer = FactorizationRFTTrainer(base_model="o4-mini-2025-04-16")
    
    # Step 3: Start training
    print("\n🚀 Starting reinforcement fine-tuning job...")
    model_suffix = model_suffix or f"factorization-expert-{int(time.time())}"
    
    # Configure hyperparameters
    hyperparameters = {
        "reasoning_effort": "high",
        "batch_size": 32,
        "learning_rate_multiplier": 1.0,
        "init_value": 0.0,
        "kl_coef": 0.1
    }
    
    # Start the training job
    job_id = trainer.train(
        train_file=train_file,
        valid_file=valid_file,
        suffix=model_suffix,
        hyperparameters=hyperparameters
    )
    
    # Save job configuration
    job_config = {
        "job_id": job_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_suffix": model_suffix,
        "hyperparameters": hyperparameters,
        "dataset": {
            "train_samples": dataset_result["train_samples"],
            "valid_samples": dataset_result["valid_samples"],
            "test_samples": dataset_result["test_samples"],
            "max_value": max_value,
            "optimize_distribution": optimize_distribution
        },
        "base_model": "o4-mini-2025-04-16"
    }
    
    with open(os.path.join(run_dir, "job_config.json"), 'w') as f:
        json.dump(job_config, f, indent=2)
    
    print(f"Training job started with ID: {job_id}")
    print(f"Job configuration saved to {os.path.join(run_dir, 'job_config.json')}")
    
    # Step 4: Monitor training (optional)
    final_job_status = None
    if monitor_training:
        print("\n📈 Monitoring training progress...")
        final_job_status = trainer.monitor_job(job_id, interval=120, max_time=7200)
        
        # Save final job status
        with open(os.path.join(dataset_result["results_dir"], "job_status.json"), 'w') as f:
            json.dump(final_job_status, f, indent=2)
    
    # Step 5: Evaluate model (optional)
    evaluation_results = None
    if evaluate_model and final_job_status and final_job_status.get("status") == "succeeded":
        print("\n🧪 Evaluating fine-tuned model...")
        fine_tuned_model_id = final_job_status.get("fine_tuned_model")
        
        if fine_tuned_model_id:
            # Load test examples
            with open(test_file, 'r') as f:
                test_examples = [json.loads(line) for line in f]
            
            # Evaluate the model
            evaluation_results = trainer.evaluate_model(fine_tuned_model_id, test_examples)
            
            # Save evaluation results
            with open(os.path.join(dataset_result["results_dir"], "evaluation_results.json"), 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            print(f"Model evaluation complete:")
            print(f"  Total examples: {evaluation_results['metrics']['total']}")
            print(f"  Correct: {evaluation_results['metrics']['correct']} " + 
                  f"({evaluation_results['metrics']['correct']/evaluation_results['metrics']['total']*100:.2f}%)")
            print(f"  Average score: {evaluation_results['metrics']['average_score']:.4f}")
            print(f"Evaluation results saved to {os.path.join(dataset_result['results_dir'], 'evaluation_results.json')}")
    
    # Final report
    print("\n✅ RFT Pipeline completed")
    print(f"  Run directory: {run_dir}")
    if final_job_status and final_job_status.get("status") == "succeeded":
        print(f"  Fine-tuned model: {final_job_status.get('fine_tuned_model')}")
    else:
        print(f"  Job ID: {job_id}")
    
    return {
        "job_id": job_id,
        "run_dir": run_dir,
        "job_status": final_job_status,
        "evaluation_results": evaluation_results
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run o4-mini RFT pipeline for prime factorization.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--max_value", type=int, default=10**12, help="Maximum value for generated numbers")
    parser.add_argument("--optimize", action="store_true", help="Use optimized tier distribution")
    parser.add_argument("--model_suffix", type=str, help="Suffix for the fine-tuned model name")
    parser.add_argument("--no_monitor", action="store_true", help="Do not monitor training progress")
    parser.add_argument("--no_evaluate", action="store_true", help="Do not evaluate the model after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./rft_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Check environment
    if not setup_environment():
        sys.exit(1)
    
    # Run pipeline
    try:
        result = run_rft_pipeline(
            num_samples=args.num_samples,
            max_value=args.max_value,
            optimize_distribution=args.optimize,
            model_suffix=args.model_suffix,
            monitor_training=not args.no_monitor,
            evaluate_model=not args.no_evaluate,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"Job ID: {result['job_id']}")
        print(f"Run directory: {result['run_dir']}")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()