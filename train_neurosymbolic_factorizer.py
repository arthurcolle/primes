#!/usr/bin/env python3
"""
train_neurosymbolic_factorizer.py

This script implements a training pipeline for neurosymbolic prime factorization using
OpenAI's reinforcement fine-tuning API. It prepares datasets, trains models, and
evaluates performance for factorizing extremely large numbers.
"""

import os
import json
import argparse
import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from neurosymbolic_factorizer import (
    DistributedFactorizationManager,
    OpenAIFactorizationTrainer,
    FactorizationResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("factorization_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("factorization_training")

def setup_working_dir(base_dir: str = "./rft_data") -> Dict[str, str]:
    """
    Set up the working directory structure for the training pipeline.
    
    Args:
        base_dir: Base directory for training data
        
    Returns:
        Dictionary with paths to different directories
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create directories
    dirs = {
        "base": base_dir,
        "run": run_dir,
        "data": os.path.join(run_dir, "data"),
        "models": os.path.join(run_dir, "models"),
        "results": os.path.join(run_dir, "results")
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return dirs

def generate_dataset(factorizer: DistributedFactorizationManager,
                    trainer: OpenAIFactorizationTrainer,
                    input_path: str,
                    output_path: str,
                    sample_count: Optional[int] = None,
                    max_tier: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a dataset for reinforcement fine-tuning.
    
    Args:
        factorizer: DistributedFactorizationManager to generate solutions
        trainer: OpenAIFactorizationTrainer for dataset formatting
        input_path: Path to input dataset with factorization challenges
        output_path: Path to save the JSONL training data
        sample_count: Maximum number of samples to include (None for all)
        max_tier: Maximum tier level to include (None for all)
        
    Returns:
        Dictionary with statistics about the generated dataset
    """
    logger.info(f"Generating dataset from {input_path}")
    
    # Load the dataset
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    # Get samples from all tiers
    all_samples = []
    if 'tiers' in dataset:
        for tier in dataset['tiers']:
            tier_id = tier.get('tier_info', {}).get('tier', 'unknown')
            tier_samples = tier.get('samples', [])
            for sample in tier_samples:
                sample['tier_id'] = tier_id
                all_samples.append(sample)
    else:
        all_samples = dataset.get('samples', [])
        
    logger.info(f"Loaded {len(all_samples)} samples from {input_path}")
    
    # Filter samples by tier if requested
    if max_tier is not None:
        all_samples = [s for s in all_samples if int(s.get('tier_id', 0)) <= max_tier]
        logger.info(f"Filtered to {len(all_samples)} samples with tier <= {max_tier}")
    
    # Limit sample count if requested
    if sample_count is not None and sample_count < len(all_samples):
        all_samples = random.sample(all_samples, sample_count)
        logger.info(f"Randomly sampled {sample_count} examples")
    
    # Generate factorization solutions
    training_data = []
    stats = {
        'total': len(all_samples),
        'successful': 0,
        'failed': 0,
        'tiers': {},
        'timing': {
            'start': time.time(),
            'end': None,
            'total_seconds': None
        }
    }
    
    for i, sample in enumerate(all_samples):
        logger.info(f"Processing sample {i+1}/{len(all_samples)} (tier {sample.get('tier_id', 'unknown')})")
        
        try:
            # Parse the number and reference factors
            number = int(sample.get('n', 0))
            factors = sample.get('factors', [])
            tier_id = sample.get('tier_id', 'unknown')
            tier_str = str(tier_id)
            
            # Update tier stats
            if tier_str not in stats['tiers']:
                stats['tiers'][tier_str] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0
                }
            stats['tiers'][tier_str]['total'] += 1
            
            # Generate factorization solution
            result = factorizer.factorize(number)
            
            # Create training example
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert factorization assistant. Given a number, find its prime factorization efficiently."
                    },
                    {
                        "role": "user",
                        "content": f"Find the prime factorization of {number}. Return your answer as JSON with the following fields: 'factors' (list of integers), 'algorithm' (string), 'reasoning' (list of steps), 'time_taken' (float), and 'confidence' (float)."
                    }
                ],
                "number": number,
                "factors": factors,
                "optimal_algorithm": result.algorithm,
                "expected_time": result.time_taken,
                "tier_id": tier_id
            }
            
            training_data.append(example)
            stats['successful'] += 1
            stats['tiers'][tier_str]['successful'] += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            stats['failed'] += 1
            if tier_str in stats['tiers']:
                stats['tiers'][tier_str]['failed'] += 1
    
    # Save the training data
    with open(output_path, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    # Update timing stats
    stats['timing']['end'] = time.time()
    stats['timing']['total_seconds'] = stats['timing']['end'] - stats['timing']['start']
    
    logger.info(f"Generated {len(training_data)} training examples, saved to {output_path}")
    logger.info(f"Total generation time: {stats['timing']['total_seconds']:.2f} seconds")
    
    # Save stats
    stats_path = os.path.join(os.path.dirname(output_path), "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def prepare_training_files(data_path: str, train_path: str, valid_path: str, test_path: str,
                          split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Tuple[str, str, str]:
    """
    Prepare training, validation, and test files from a JSONL dataset.
    
    Args:
        data_path: Path to the JSONL data file
        train_path: Path to save the training file
        valid_path: Path to save the validation file
        test_path: Path to save the test file
        split_ratio: Train/valid/test split ratio (must sum to 1.0)
        
    Returns:
        Tuple of (train_path, valid_path, test_path)
    """
    logger.info(f"Preparing training files from {data_path}")
    
    # Check split ratio
    if sum(split_ratio) != 1.0:
        raise ValueError(f"Split ratio must sum to 1.0, got {sum(split_ratio)}")
    
    # Load the data
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    total_examples = len(examples)
    
    # Ensure we have enough examples for all splits
    if total_examples < 3:
        # If we have very few examples, use them for all splits
        train_examples = examples.copy()
        valid_examples = examples.copy()
        test_examples = examples.copy()
    else:
        # Group by tier for stratified sampling
        tiers = {}
        for example in examples:
            tier = str(example.get('tier_id', 'unknown'))
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(example)
        
        # Perform stratified split
        train_examples = []
        valid_examples = []
        test_examples = []
        
        for tier, tier_examples in tiers.items():
            # Shuffle tier examples
            random.shuffle(tier_examples)
            
            # Calculate split points
            n = len(tier_examples)
            
            # Ensure at least one example in each split if possible
            if n >= 3:
                train_count = max(1, int(n * split_ratio[0]))
                valid_count = max(1, int(n * split_ratio[1]))
                test_count = n - train_count - valid_count
            elif n == 2:
                train_count = 1
                valid_count = 1
                test_count = 0
            else:  # n == 1
                train_count = 1
                valid_count = 1  # Use the same example for validation
                test_count = 1   # Use the same example for testing
            
            # Assign examples to splits
            train_examples.extend(tier_examples[:train_count])
            
            if valid_count <= len(tier_examples) - train_count:
                valid_examples.extend(tier_examples[train_count:train_count + valid_count])
            else:
                # Not enough examples, use duplicates
                valid_examples.extend(tier_examples[:1] * valid_count)
            
            if test_count > 0 and test_count <= len(tier_examples) - train_count - valid_count:
                test_examples.extend(tier_examples[train_count + valid_count:])
            elif test_count > 0:
                # Not enough examples, use duplicates
                test_examples.extend(tier_examples[:1] * test_count)
    
    # Shuffle again
    random.shuffle(train_examples)
    random.shuffle(valid_examples)
    random.shuffle(test_examples)
    
    # Ensure we have at least one example in each split
    if not train_examples and examples:
        train_examples = [examples[0]]
    if not valid_examples and examples:
        valid_examples = [examples[0]]
    if not test_examples and examples:
        test_examples = [examples[0]]
    
    # Save the files
    with open(train_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    with open(valid_path, 'w') as f:
        for example in valid_examples:
            f.write(json.dumps(example) + '\n')
    
    with open(test_path, 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Prepared {len(train_examples)} training, {len(valid_examples)} validation, and {len(test_examples)} test examples")
    return train_path, valid_path, test_path

def train_model(trainer: OpenAIFactorizationTrainer, 
               train_file: str, 
               valid_file: str,
               base_model: str,
               suffix: str,
               batch_size: int = 4,
               learning_rate_multiplier: float = 0.1,
               n_epochs: int = 3) -> Dict[str, Any]:
    """
    Train a model using OpenAI's reinforcement fine-tuning API.
    
    Args:
        trainer: OpenAIFactorizationTrainer instance
        train_file: Path to the training file
        valid_file: Path to the validation file
        base_model: Base model to fine-tune
        suffix: Suffix for the fine-tuned model name
        batch_size: Batch size for training
        learning_rate_multiplier: Learning rate multiplier
        n_epochs: Number of epochs for training
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Starting training with base model {base_model}")
    logger.info(f"Training parameters: batch_size={batch_size}, lr_multiplier={learning_rate_multiplier}, n_epochs={n_epochs}")
    
    # Create fine-tuning job
    job_id = trainer.train_model(
        train_file=train_file,
        valid_file=valid_file,
        base_model=base_model,
        suffix=suffix,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
        n_epochs=n_epochs
    )
    
    logger.info(f"Created fine-tuning job with ID: {job_id}")
    
    # Monitor training progress
    logger.info("Monitoring training progress...")
    job_result = trainer.monitor_job(job_id)
    
    # Extract results
    results = {
        'job_id': job_id,
        'status': job_result.get('status', 'unknown'),
        'model_id': job_result.get('fine_tuned_model', None),
        'elapsed_time': None,
        'train_loss': None,
        'val_loss': None
    }
    
    # Extract training metrics if available
    if hasattr(job_result, 'training_metrics'):
        results['train_loss'] = job_result.training_metrics.get('avg_train_reward')
        results['val_loss'] = job_result.training_metrics.get('avg_valid_reward')
    
    # Extract timing info
    if hasattr(job_result, 'created_at') and hasattr(job_result, 'finished_at'):
        # Convert timestamps to seconds
        created = job_result.created_at
        finished = job_result.finished_at
        if created and finished:
            results['elapsed_time'] = (finished - created) / 1000  # Convert ms to seconds
    
    logger.info(f"Training completed with status: {results['status']}")
    if results['model_id']:
        logger.info(f"Fine-tuned model ID: {results['model_id']}")
    
    # Save results
    results_dir = os.path.dirname(train_file)
    results_path = os.path.join(results_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def evaluate_model(trainer: OpenAIFactorizationTrainer,
                  model_id: str,
                  test_file: str,
                  results_file: str,
                  n_examples: int = 20) -> Dict[str, Any]:
    """
    Evaluate a fine-tuned model on factorization examples.
    
    Args:
        trainer: OpenAIFactorizationTrainer instance
        model_id: Model ID to evaluate
        test_file: Path to the test JSONL file
        results_file: Path to save the evaluation results
        n_examples: Number of examples to evaluate
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating model {model_id} on {n_examples} examples from {test_file}")
    
    # Evaluate the model
    eval_results = trainer.evaluate_model(
        model_id=model_id,
        test_file=test_file,
        n_examples=n_examples
    )
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Log summary
    metrics = eval_results['metrics']
    logger.info("\nEvaluation results:")
    logger.info(f"Total examples: {metrics['total']}")
    logger.info(f"Correct: {metrics['correct']} ({metrics['correct']/metrics['total']*100:.2f}%)")
    logger.info(f"Partially correct: {metrics['partially_correct']} ({metrics['partially_correct']/metrics['total']*100:.2f}%)")
    logger.info(f"Incorrect: {metrics['incorrect']} ({metrics['incorrect']/metrics['total']*100:.2f}%)")
    logger.info(f"Average score: {metrics['average_score']:.4f}")
    
    logger.info("\nPerformance by tier:")
    for tier, data in sorted(metrics["tier_performance"].items()):
        logger.info(f"Tier {tier}: {data['accuracy']*100:.2f}% accuracy, {data['score']:.4f} avg score ({data['count']} examples)")
    
    return eval_results

def main():
    """Main function for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train a neurosymbolic prime factorization model")
    
    # Dataset generation arguments
    parser.add_argument("--input_dataset", type=str, default="quantum_benchmark.json",
                       help="Path to input benchmark dataset")
    parser.add_argument("--sample_count", type=int, default=10,
                       help="Number of samples to use (None for all)")
    parser.add_argument("--max_tier", type=int, default=4,
                       help="Maximum tier level to include (None for all)")
    
    # Training arguments
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini-2024-07-18",
                       help="Base model for fine-tuning (options: gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, etc.)")
    parser.add_argument("--model_suffix", type=str, default="factorization-expert",
                       help="Suffix for the fine-tuned model name")
    parser.add_argument("--eval_examples", type=int, default=5,
                       help="Number of examples to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1,
                       help="Learning rate multiplier for fine-tuning")
    parser.add_argument("--n_epochs", type=int, default=3,
                       help="Number of epochs for training")
    
    # Advanced options
    parser.add_argument("--save_checkpoints", action="store_true",
                       help="Save intermediate checkpoints during training")
    parser.add_argument("--visualize_results", action="store_true",
                       help="Generate visualizations of factorization results")
    parser.add_argument("--export_format", type=str, choices=["json", "parquet", "csv"], default="json",
                       help="Format for exporting results")
    
    # Pipeline control arguments
    parser.add_argument("--skip_dataset", action="store_true",
                       help="Skip dataset generation step")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip model training step")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation step")
    parser.add_argument("--working_dir", type=str, default="./rft_data",
                       help="Working directory for training data")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up working directory
    dirs = setup_working_dir(args.working_dir)
    
    # Initialize components
    factorizer = DistributedFactorizationManager()
    trainer = OpenAIFactorizationTrainer()
    
    # Generate dataset
    dataset_path = os.path.join(dirs["data"], "factorization_dataset.jsonl")
    
    if not args.skip_dataset:
        logger.info("===== Generating Dataset =====")
        dataset_stats = generate_dataset(
            factorizer=factorizer,
            trainer=trainer,
            input_path=args.input_dataset,
            output_path=dataset_path,
            sample_count=args.sample_count,
            max_tier=args.max_tier
        )
    else:
        logger.info("Skipping dataset generation")
    
    # Prepare training files
    train_path = os.path.join(dirs["data"], "train.jsonl")
    valid_path = os.path.join(dirs["data"], "valid.jsonl")
    test_path = os.path.join(dirs["data"], "test.jsonl")
    
    if not args.skip_dataset:
        logger.info("===== Preparing Training Files =====")
        prepare_training_files(
            data_path=dataset_path,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path
        )
    
    # Train model
    if not args.skip_training:
        logger.info("===== Training Model =====")
        training_results = train_model(
            trainer=trainer,
            train_file=train_path,
            valid_file=valid_path,
            base_model=args.base_model,
            suffix=args.model_suffix,
            batch_size=args.batch_size,
            learning_rate_multiplier=args.learning_rate_multiplier,
            n_epochs=args.n_epochs
        )
        model_id = training_results.get('model_id')
    else:
        logger.info("Skipping model training")
        # Try to load model ID from previous run
        results_path = os.path.join(dirs["data"], "training_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                training_results = json.load(f)
                model_id = training_results.get('model_id')
        else:
            model_id = None
            logger.warning("No model ID found for evaluation")
    
    # Evaluate model
    if not args.skip_evaluation and model_id:
        logger.info("===== Evaluating Model =====")
        eval_results_path = os.path.join(dirs["results"], "evaluation_results.json")
        evaluate_model(
            trainer=trainer,
            model_id=model_id,
            test_file=test_path,
            results_file=eval_results_path,
            n_examples=args.eval_examples
        )
    elif not model_id:
        logger.warning("Skipping evaluation: No model ID available")
    else:
        logger.info("Skipping model evaluation")
    
    logger.info("===== Training Pipeline Completed =====")

if __name__ == "__main__":
    main()
