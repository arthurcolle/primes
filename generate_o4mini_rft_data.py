#!/usr/bin/env python3
"""
generate_o4mini_rft_data.py

Generate optimized training data for o4-mini reinforcement fine-tuning on prime factorization.
This script creates balanced datasets with diverse examples across different difficulty levels.
"""

import os
import json
import random
import numpy as np
import sympy
import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from prime_rft_dataset_generator import (
    generate_dataset_for_tier,
    convert_to_rft_format,
    save_to_jsonl
)

def generate_balanced_dataset(
    output_dir: str,
    num_samples: int = 1000,
    tier_distribution: Optional[Dict[int, float]] = None,
    max_value: int = 10**14,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a balanced dataset for o4-mini RFT with an emphasis on diverse examples.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Total number of samples to generate
        tier_distribution: Distribution of samples across tiers (as percentages)
        max_value: Maximum value for generated numbers
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with dataset statistics
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Default tier distribution if not provided
    if tier_distribution is None:
        tier_distribution = {
            0: 0.05,  # Very easy: 5%
            1: 0.10,  # Easy: 10%
            2: 0.15,  # Basic: 15%
            3: 0.15,  # Basic+: 15%
            4: 0.15,  # Intermediate: 15%
            5: 0.15,  # Challenging: 15%
            6: 0.10,  # Hard: 10%
            7: 0.05,  # Very Hard: 5%
            8: 0.05,  # Expert: 5%
            9: 0.05,  # Master: 5%
        }
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    data_dir = os.path.join(run_dir, "data")
    models_dir = os.path.join(run_dir, "models")
    results_dir = os.path.join(run_dir, "results")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Generating dataset in {run_dir}")
    
    # Calculate samples per tier
    samples_per_tier = {}
    for tier, percentage in tier_distribution.items():
        samples_per_tier[tier] = int(num_samples * percentage)
    
    # Adjust to ensure we get exactly num_samples
    total = sum(samples_per_tier.values())
    if total < num_samples:
        # Add remaining samples to mid-tier
        samples_per_tier[4] += (num_samples - total)
    elif total > num_samples:
        # Remove excess samples from highest tier with excess
        for tier in sorted(samples_per_tier.keys(), reverse=True):
            if samples_per_tier[tier] > (num_samples - (total - samples_per_tier[tier])):
                samples_per_tier[tier] -= (total - num_samples)
                break
    
    # Generate samples for each tier
    all_samples = []
    stats = {"total": 0, "successful": 0, "failed": 0, "tiers": {}}
    
    for tier, count in samples_per_tier.items():
        if count == 0:
            continue
            
        print(f"Generating {count} samples for tier {tier}")
        tier_samples = generate_dataset_for_tier(tier, count, max_value)
        all_samples.extend(tier_samples)
        
        # Update stats
        stats["total"] += len(tier_samples)
        stats["successful"] += len(tier_samples)
        stats["tiers"][tier] = {
            "total": len(tier_samples),
            "successful": len(tier_samples),
            "failed": 0
        }
    
    # Convert to RFT format
    rft_samples = convert_to_rft_format(all_samples)
    
    # Add enhanced system prompts to a percentage of examples (20%)
    enhanced_indices = random.sample(range(len(rft_samples)), int(0.2 * len(rft_samples)))
    for idx in enhanced_indices:
        enhanced_prompt = """You are an expert factorization assistant. Your task is to find the prime factorization of numbers efficiently.

Key algorithms to consider:
- Trial Division: Efficient for small numbers or when small factors are expected
- Wheel Factorization: More efficient version of trial division that skips multiples of small primes
- Pollard's Rho: Probabilistic algorithm good for finding small factors of large numbers
- Elliptic Curve Method (ECM): Effective for finding factors up to 50 digits
- Quadratic Sieve: Efficient for numbers up to 100 digits
- General Number Field Sieve (GNFS): Most efficient for very large numbers

Always verify your factorization by multiplying the factors to ensure they equal the original number. 
Provide detailed reasoning steps and estimate your solution's time complexity."""

        rft_samples[idx]["messages"][0]["content"] = enhanced_prompt
    
    # Split into train, validation, and test sets (70/15/15)
    random.shuffle(rft_samples)
    train_split = int(0.7 * len(rft_samples))
    valid_split = int(0.85 * len(rft_samples))
    
    train_samples = rft_samples[:train_split]
    valid_samples = rft_samples[train_split:valid_split]
    test_samples = rft_samples[valid_split:]
    
    # Save datasets
    save_to_jsonl(train_samples, os.path.join(data_dir, "train.jsonl"))
    save_to_jsonl(valid_samples, os.path.join(data_dir, "valid.jsonl"))
    save_to_jsonl(test_samples, os.path.join(data_dir, "test.jsonl"))
    
    # Save all samples in one file for reference
    save_to_jsonl(rft_samples, os.path.join(data_dir, "factorization_dataset.jsonl"))
    
    # Add timing information to stats
    stats["timing"] = {
        "start": datetime.datetime.now().timestamp(),
        "end": datetime.datetime.now().timestamp(),
        "total_seconds": 0
    }
    
    # Save stats
    with open(os.path.join(data_dir, "dataset_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generated {len(rft_samples)} total samples:")
    print(f"  - Training: {len(train_samples)}")
    print(f"  - Validation: {len(valid_samples)}")
    print(f"  - Testing: {len(test_samples)}")
    print(f"Data saved to {data_dir}")
    
    return {
        "run_dir": run_dir,
        "data_dir": data_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "stats": stats,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "test_samples": len(test_samples)
    }

def optimize_tier_distribution(max_value: int = 10**14) -> Dict[int, float]:
    """
    Optimize tier distribution based on o4-mini's capabilities.
    
    Args:
        max_value: Maximum value for generated numbers
        
    Returns:
        Dictionary with optimized tier distribution
    """
    # For o4-mini, we want to focus on tiers where it can reasonably succeed
    # but also include challenging examples for it to learn from
    return {
        0: 0.05,  # Very easy: 5%
        1: 0.10,  # Easy: 10%
        2: 0.15,  # Basic: 15%
        3: 0.20,  # Basic+: 20% (sweet spot for o4-mini)
        4: 0.20,  # Intermediate: 20% (sweet spot for o4-mini)
        5: 0.15,  # Challenging: 15% (stretching capabilities)
        6: 0.08,  # Hard: 8% (challenging but learnable)
        7: 0.05,  # Very Hard: 5% (challenging but learnable)
        8: 0.02,  # Expert: 2% (very challenging)
        9: 0.00,  # Master: 0% (too difficult for o4-mini)
    }

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate optimized o4-mini RFT datasets for factorization.")
    parser.add_argument("--output_dir", type=str, default="./rft_data", help="Directory to save datasets to")
    parser.add_argument("--num_samples", type=int, default=500, help="Total number of samples to generate")
    parser.add_argument("--max_value", type=int, default=10**12, help="Maximum value for generated numbers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optimize", action="store_true", help="Use optimized tier distribution for o4-mini")
    
    args = parser.parse_args()
    
    # Get tier distribution
    tier_distribution = optimize_tier_distribution(args.max_value) if args.optimize else None
    
    # Generate dataset
    result = generate_balanced_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        tier_distribution=tier_distribution,
        max_value=args.max_value,
        seed=args.seed
    )
    
    print(f"Dataset generation complete. Stats:")
    print(f"  Total samples: {result['stats']['total']}")
    print(f"  Training samples: {result['train_samples']}")
    print(f"  Validation samples: {result['valid_samples']}")
    print(f"  Test samples: {result['test_samples']}")
    print(f"  Output directory: {result['run_dir']}")

if __name__ == "__main__":
    main()