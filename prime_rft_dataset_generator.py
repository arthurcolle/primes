#!/usr/bin/env python3
"""
prime_rft_dataset_generator.py

Generate training datasets for reinforcement fine-tuning of factorization models.
This tool creates structured data with numbers, their prime factorizations, 
and optimal algorithm recommendations based on the S(N,K) framework.
"""

import os
import json
import random
import sympy
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm

def is_prime(n: int) -> bool:
    """Check if a number is prime using sympy."""
    return sympy.isprime(n)

def get_prime_factors(n: int) -> List[int]:
    """Get the prime factorization of a number using sympy."""
    return sorted([int(p) ** e for p, e in sympy.factorint(n).items() for _ in range(e)])

def digit_length(n: int) -> int:
    """Get the number of digits in an integer."""
    return len(str(n))

def bit_length(n: int) -> int:
    """Get the number of bits in the binary representation of an integer."""
    return n.bit_length()

def generate_prime_with_digits(digits: int) -> int:
    """Generate a random prime with a specific number of digits."""
    lower_bound = 10 ** (digits - 1)
    upper_bound = 10 ** digits - 1
    
    while True:
        candidate = random.randint(lower_bound, upper_bound)
        if is_prime(candidate):
            return candidate

def generate_prime_with_bits(bits: int) -> int:
    """Generate a random prime with a specific number of bits."""
    lower_bound = 2 ** (bits - 1)
    upper_bound = 2 ** bits - 1
    
    while True:
        candidate = random.randint(lower_bound, upper_bound)
        if is_prime(candidate):
            return candidate

def generate_snk_sample(N: int, K: List[int], max_value: int = 10**10) -> Dict[str, Any]:
    """
    Generate a sample from the S(N,K) space.
    
    Args:
        N: Maximum number of prime factors
        K: List of allowed digit lengths for prime factors
        max_value: Maximum value for the generated number
        
    Returns:
        Dictionary with the generated number, factors, and signature
    """
    # Determine number of factors to use (1 to N)
    num_factors = random.randint(1, N)
    
    # Generate random prime factors with digit lengths from K
    factors = []
    while len(factors) < num_factors:
        # Choose a random digit length from K
        digits = random.choice(K)
        
        # Generate a prime with that many digits
        p = generate_prime_with_digits(digits)
        
        # Add to factors if it doesn't make the product too large
        if factors:
            product = p
            for f in factors:
                product *= f
            if product > max_value:
                continue
                
        factors.append(p)
    
    # Calculate the product
    n = 1
    for f in factors:
        n *= f
    
    # Calculate signature (count of factors by digit length)
    signature = defaultdict(int)
    for f in factors:
        signature[digit_length(f)] += 1
    
    return {
        "n": n,
        "factors": factors,
        "signature": dict(signature),
        "digit_class": K,
        "max_factors": N
    }

def recommend_algorithm(n: int, factors: List[int]) -> str:
    """
    Recommend the optimal algorithm for factorizing a number.
    
    Args:
        n: Number to factorize
        factors: Known prime factors of n
        
    Returns:
        Recommended algorithm name
    """
    bits = bit_length(n)
    num_factors = len(factors)
    largest_factor_bits = max(bit_length(f) for f in factors) if factors else 0
    smallest_factor_bits = min(bit_length(f) for f in factors) if factors else 0
    
    # Simple decision tree based on number properties
    if bits <= 32:
        return "TrialDivision"
    elif bits <= 64:
        return "WheelFactorization"
    elif bits <= 128:
        if largest_factor_bits < bits / 3:
            return "PollardRho"  # Good for numbers with small factors
        else:
            return "ECM"
    elif bits <= 512:
        if num_factors <= 2:
            return "QuadraticSieve"
        else:
            return "ECM"
    else:
        return "GNFS"  # General Number Field Sieve for very large numbers

def estimate_factorization_time(n: int, algorithm: str) -> float:
    """
    Estimate the time needed to factorize a number with a given algorithm.
    This is a very rough estimate based on bit length and algorithm complexity.
    
    Args:
        n: Number to factorize
        algorithm: Algorithm to use
        
    Returns:
        Estimated time in seconds
    """
    bits = bit_length(n)
    
    # Simplified complexity estimates (very rough approximations)
    if algorithm == "TrialDivision":
        # O(sqrt(n))
        return 1e-6 * math.sqrt(n)
    elif algorithm == "WheelFactorization":
        # Better than trial division by a constant factor
        return 5e-7 * math.sqrt(n)
    elif algorithm == "PollardRho":
        # O(sqrt(p)) where p is the smallest prime factor
        # We estimate p as sqrt(n) as a heuristic
        return 1e-5 * math.sqrt(math.sqrt(n))
    elif algorithm == "ECM":
        # Very rough estimate for ECM
        return 1e-4 * math.exp(0.5 * math.sqrt(bits * math.log(bits)))
    elif algorithm == "QuadraticSieve":
        # O(exp(sqrt(log(n) * log(log(n))))
        return 1e-3 * math.exp(0.9 * math.sqrt(math.log(n) * math.log(math.log(n))))
    elif algorithm == "GNFS":
        # O(exp((log(n))^(1/3) * (log(log(n)))^(2/3)))
        return 1e-2 * math.exp(1.9 * (math.log(n))**(1/3) * (math.log(math.log(n)))**(2/3))
    else:
        return 1.0  # Default for unknown algorithms

def generate_dataset_for_tier(tier: int, samples_per_tier: int = 100, 
                             max_value: int = 10**12) -> List[Dict[str, Any]]:
    """
    Generate a dataset for a specific difficulty tier.
    
    Args:
        tier: Difficulty tier (0-15)
        samples_per_tier: Number of samples to generate per tier
        max_value: Maximum value for generated numbers
        
    Returns:
        List of samples for the tier
    """
    # Define tier configurations based on S(N,K) parameters
    tier_configs = {
        0: {"N": 2, "K": [1], "max_value": 10**3},                  # Very easy
        1: {"N": 2, "K": [1], "max_value": 10**4},                  # Easy
        2: {"N": 3, "K": [1, 2], "max_value": 10**6},               # Basic
        3: {"N": 3, "K": [1, 2], "max_value": 10**7},               # Basic+
        4: {"N": 3, "K": [1, 2], "max_value": 10**8},               # Intermediate
        5: {"N": 4, "K": [2, 3], "max_value": 10**9},               # Challenging
        6: {"N": 4, "K": [2, 3], "max_value": 10**10},              # Hard
        7: {"N": 4, "K": [2, 3], "max_value": 10**11},              # Very Hard
        8: {"N": 5, "K": [3, 4], "max_value": 10**12},              # Expert
        9: {"N": 5, "K": [3, 4], "max_value": 10**14},              # Master
        10: {"N": 2, "K": [30, 40], "max_value": 2**2048},          # RSA Easy
        11: {"N": 2, "K": [40, 50], "max_value": 2**3072},          # RSA Medium
        12: {"N": 2, "K": [50, 60], "max_value": 2**4096},          # RSA Hard
        13: {"N": 2, "K": [100], "max_value": 10**200},             # Extreme
        14: {"N": 2, "K": [150], "max_value": 10**300},             # Super Extreme
        15: {"N": 2, "K": [200], "max_value": 10**400}              # Ultra Extreme
    }
    
    # Get configuration for the specified tier
    config = tier_configs.get(tier, tier_configs[0])
    
    # Adjust max_value based on input parameter
    config["max_value"] = min(config["max_value"], max_value)
    
    # Generate samples
    samples = []
    for _ in tqdm(range(samples_per_tier), desc=f"Generating Tier {tier}"):
        sample = generate_snk_sample(config["N"], config["K"], config["max_value"])
        
        # Add tier information
        sample["tier"] = tier
        
        # Add algorithm recommendation
        algo = recommend_algorithm(sample["n"], sample["factors"])
        sample["recommended_algorithm"] = algo
        
        # Add estimated time
        sample["estimated_time"] = estimate_factorization_time(sample["n"], algo)
        
        samples.append(sample)
    
    return samples

def convert_to_rft_format(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert samples to the format needed for reinforcement fine-tuning.
    
    Args:
        samples: List of generated samples
        
    Returns:
        List of samples in RFT format
    """
    rft_samples = []
    
    for sample in samples:
        # Create the RFT format sample
        rft_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert factorization assistant. Your task is to find the prime factorization of numbers efficiently, using the most appropriate algorithm."
                },
                {
                    "role": "user",
                    "content": f"Find the prime factorization of {sample['n']}. Return your answer as JSON with the following fields: 'factors' (list of integers), 'algorithm' (string), 'reasoning' (list of steps), 'time_taken' (float), and 'confidence' (float)."
                }
            ],
            "number": sample["n"],
            "factors": sample["factors"],
            "optimal_algorithm": sample["recommended_algorithm"],
            "expected_time": sample["estimated_time"],
            "tier": sample["tier"]
        }
        
        rft_samples.append(rft_sample)
    
    return rft_samples

def save_to_jsonl(samples: List[Dict[str, Any]], filename: str) -> None:
    """
    Save samples to a JSONL file.
    
    Args:
        samples: List of samples to save
        filename: Name of the file to save to
    """
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(samples)} samples to {filename}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets for reinforcement fine-tuning of factorization models.")
    parser.add_argument("--output_dir", type=str, default="./rft_data", help="Directory to save datasets to")
    parser.add_argument("--tiers", type=int, nargs="+", default=list(range(10)), help="Tiers to generate datasets for")
    parser.add_argument("--samples_per_tier", type=int, default=100, help="Number of samples per tier")
    parser.add_argument("--max_value", type=int, default=10**12, help="Maximum value for generated numbers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate datasets for each tier
    all_samples = []
    for tier in args.tiers:
        samples = generate_dataset_for_tier(tier, args.samples_per_tier, args.max_value)
        all_samples.extend(samples)
        
        # Save tier-specific dataset
        tier_filename = os.path.join(args.output_dir, f"tier_{tier}.json")
        with open(tier_filename, 'w') as f:
            json.dump(samples, f, indent=2)
    
    # Convert to RFT format
    rft_samples = convert_to_rft_format(all_samples)
    
    # Split into training and validation (80/20)
    np.random.shuffle(rft_samples)
    split_idx = int(0.8 * len(rft_samples))
    train_samples = rft_samples[:split_idx]
    valid_samples = rft_samples[split_idx:]
    
    # Save training and validation datasets
    save_to_jsonl(train_samples, os.path.join(args.output_dir, "train.jsonl"))
    save_to_jsonl(valid_samples, os.path.join(args.output_dir, "valid.jsonl"))
    
    # Save metadata
    metadata = {
        "total_samples": len(rft_samples),
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "tiers": args.tiers,
        "samples_per_tier": args.samples_per_tier,
        "max_value": args.max_value,
        "seed": args.seed
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()