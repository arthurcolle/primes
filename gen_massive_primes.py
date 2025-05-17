#!/usr/bin/env python
import math, multiprocessing as mp, pyarrow as pa, pyarrow.parquet as pq
import random, sympy, tqdm, os, json, time
from datetime import datetime
from collections import defaultdict

# Constants for benchmark generation
MAX_NORMAL = 10**12          # Normal tier max
CRYPTO_SIZES = [2048, 3072, 4096]  # Cryptographic key sizes in bits
SAMPLES_PER_TIER = 1000      # Create 1000 samples per tier for statistical significance
MAX_WORKERS = mp.cpu_count()

def generate_large_prime(bits):
    """Generate a prime number with specified bit length"""
    return sympy.randprime(2**(bits-1), 2**bits-1)

def generate_near_prime_pair(digits):
    """Generate a pair of extremely close primes with given digit count"""
    base = random.randint(10**(digits-1), 10**digits-1)
    p = sympy.nextprime(base)
    q = sympy.nextprime(p)
    return p, q

def smallest_factor(n):
    """Find the smallest prime factor of n"""
    if n % 2 == 0: return 2
    r = int(math.isqrt(n))
    f = 3
    while f <= r and n % f: f += 2
    return f if f <= r else n

def factorize(n):
    """Factorize a number into its prime components"""
    factors = []
    while n > 1:
        f = smallest_factor(n)
        factors.append(f)
        n //= f
    return factors

def calc_difficulty(n, factors):
    """Calculate difficulty metrics for a factorization challenge"""
    bit_length = n.bit_length()
    factor_count = len(factors)
    
    # Calculate size variance
    if factor_count > 1:
        log_factors = [math.log(f) for f in factors]
        avg_log = sum(log_factors) / factor_count
        size_variance = sum((lf - avg_log)**2 for lf in log_factors) / factor_count
    else:
        size_variance = 0
    
    # Calculate minimum relative gap
    if factor_count > 1:
        sorted_factors = sorted(factors)
        gaps = [sorted_factors[i+1]/sorted_factors[i] for i in range(factor_count-1)]
        min_relative_gap = min(gaps) if gaps else 0
    else:
        min_relative_gap = 0
    
    # General Number Field Sieve difficulty estimate (simplified)
    gnfs_estimate = (1.92 + o(1)) * (bit_length**(1/3)) * (math.log(bit_length)**(2/3))
    
    # Combined score - weighted sum
    combined_score = (
        0.3 * bit_length + 
        0.2 * factor_count + 
        0.15 * size_variance + 
        0.15 * min_relative_gap + 
        0.2 * gnfs_estimate
    )
    
    return {
        "bit_length": bit_length,
        "factor_count": factor_count,
        "size_variance": size_variance,
        "min_relative_gap": min_relative_gap,
        "gnfs_estimate": gnfs_estimate,
        "combined_score": combined_score
    }

def o(n):
    """Helper function for asymptotic notation calculations"""
    return 1  # Simplified implementation

def create_signature(factors):
    """Create a factor signature (counts of each prime power)"""
    signature = defaultdict(int)
    for f in factors:
        signature[f] += 1
    # Convert to string representation
    return {str(k): v for k, v in signature.items()}

def generate_tier_samples(tier_config, num_samples=SAMPLES_PER_TIER):
    """Generate samples for a specific tier configuration"""
    samples = []
    
    # Extract tier configuration
    tier = tier_config["tier"]
    
    # Handle special tier types
    if "special_type" in tier_config:
        if tier_config["special_type"] == "quantum_resistant":
            bit_length = tier_config.get("bit_length", 2048)
            return generate_quantum_resistant_samples(tier, bit_length, num_samples)
        elif tier_config["special_type"] == "near_prime":
            digits = tier_config.get("digits", 40)
            return generate_near_prime_samples(tier, digits, num_samples)
        elif tier_config["special_type"] == "massive_prime":
            bits = tier_config.get("bits", 40)
            return generate_massive_prime_samples(tier, bits, num_samples)
    
    # Standard tier configuration
    N = tier_config.get("N", 2)  # Number of factors
    K = tier_config.get("K", [1])  # Distribution of factor sizes
    
    # Generate samples
    for i in range(num_samples):
        # Generate prime factors based on tier configuration
        factors = []
        for k in K:
            if k == 1:  # Small factors
                f = sympy.randprime(2, 2**16)
            elif k == 2:  # Medium factors
                f = sympy.randprime(2**16, 2**32)
            elif k == 3:  # Large factors
                f = sympy.randprime(2**32, 2**64)
            elif k == 4:  # Extra large factors
                f = sympy.randprime(2**64, 2**128)
            elif k == 5:  # Massive factors
                f = sympy.randprime(2**128, 2**256)
            factors.append(f)
        
        # Ensure we have exactly N factors
        while len(factors) < N:
            factors.append(sympy.randprime(2, 2**16))
        
        # Calculate the composite number and its properties
        n = math.prod(factors)
        signature = create_signature(factors)
        difficulty = calc_difficulty(n, factors)
        
        # Create sample
        sample = {
            "sample_id": i,
            "n": n,
            "factors": factors,
            "signature": signature,
            "factor_count": len(factors),
            "difficulty": difficulty
        }
        samples.append(sample)
    
    return samples

def generate_quantum_resistant_samples(tier, bit_length, num_samples):
    """Generate quantum-resistant samples with large bit length"""
    samples = []
    
    for i in range(num_samples):
        # For RSA-style, generate two large primes
        if random.random() < 0.6:  # 60% semiprime challenges
            p = generate_large_prime(bit_length // 2)
            q = generate_large_prime(bit_length // 2)
            factors = [p, q]
        else:  # 40% multi-factor challenges
            factor_count = random.randint(3, 5)
            total_bits = bit_length
            factors = []
            
            # Distribute bits among factors
            bit_shares = []
            remaining = total_bits
            for j in range(factor_count - 1):
                # Make sure each factor gets at least 128 bits
                share = random.randint(128, remaining - 128 * (factor_count - j - 1))
                bit_shares.append(share)
                remaining -= share
            bit_shares.append(remaining)
            
            # Generate primes with specified bit lengths
            for bits in bit_shares:
                factors.append(generate_large_prime(bits))
        
        n = math.prod(factors)
        signature = create_signature(factors)
        difficulty = calc_difficulty(n, factors)
        
        sample = {
            "sample_id": i,
            "n": n,
            "factors": factors,
            "signature": signature,
            "factor_count": len(factors),
            "difficulty": difficulty
        }
        samples.append(sample)
    
    return samples

def generate_near_prime_samples(tier, digits, num_samples):
    """Generate challenges with extremely close primes"""
    samples = []
    
    for i in range(num_samples):
        p, q = generate_near_prime_pair(digits)
        factors = [p, q]
        n = p * q
        signature = create_signature(factors)
        difficulty = calc_difficulty(n, factors)
        
        # Add additional distance metric for near-primes
        difficulty["prime_gap"] = q - p
        difficulty["relative_gap"] = (q - p) / p
        
        sample = {
            "sample_id": i,
            "n": n,
            "factors": factors,
            "signature": signature,
            "factor_count": len(factors),
            "difficulty": difficulty
        }
        samples.append(sample)
    
    return samples

def generate_massive_prime_samples(tier, bits, num_samples):
    """Generate challenges with extremely large primes (10^12+ range)"""
    samples = []
    
    for i in range(num_samples):
        # Generate 1-3 massive primes
        factor_count = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
        
        # Distribute bits among factors
        if factor_count == 1:
            # Single massive prime - just for verification
            factors = [generate_large_prime(bits)]
        else:
            # Multiple massive primes
            total_bits = bits
            bit_shares = []
            remaining = total_bits
            for j in range(factor_count - 1):
                share = random.randint(bits//factor_count//2, remaining - (bits//factor_count//2) * (factor_count - j - 1))
                bit_shares.append(share)
                remaining -= share
            bit_shares.append(remaining)
            
            factors = [generate_large_prime(bs) for bs in bit_shares]
        
        n = math.prod(factors)
        signature = create_signature(factors)
        difficulty = calc_difficulty(n, factors)
        
        sample = {
            "sample_id": i,
            "n": n,
            "factors": factors,
            "signature": signature,
            "factor_count": len(factors),
            "difficulty": difficulty
        }
        samples.append(sample)
    
    return samples

def create_prompts(n, factors):
    """Create factorization prompts for a number"""
    prompts = [
        f"Find the prime factorization of {n}.",
        f"What are the prime factors of {n}?",
        f"Factorize {n} into its prime components.",
        f"Decompose {n} into a product of primes."
    ]
    return random.choice(prompts)

def serialize_sample(sample, tier_info):
    """Convert a sample to a format suitable for the final dataset"""
    n = sample["n"]
    factors = sample["factors"]
    difficulty = sample["difficulty"]
    
    # Convert factors to strings for JSON serialization
    factors_str = "×".join(str(f) for f in factors)
    
    prompt = create_prompts(n, factors)
    answer = f"The prime factorization of {n} is {factors_str}."
    
    return {
        "tier_id": tier_info["tier"],
        "sample_id": sample["sample_id"],
        "prompt_id": f"tier{tier_info['tier']}_sample{sample['sample_id']}",
        "prompt": prompt,
        "number": str(n),
        "factors": factors_str,
        "factor_count": len(factors),
        "difficulty": difficulty["combined_score"],
        "bit_length": difficulty["bit_length"],
        "category": tier_info.get("category", "advanced"),
        "tier_description": tier_info.get("description", "Unknown tier"),
        "answer": answer
    }

def generate_dataset():
    """Generate the complete benchmark dataset"""
    # Define the tier configurations
    tiers = [
        # Basic tiers (similar to existing but with more samples)
        {"tier": 0, "N": 2, "K": [1], "description": "Basic factorization of small semiprimes", "category": "elementary"},
        {"tier": 1, "N": 2, "K": [1, 1], "description": "Mixed small semiprimes", "category": "elementary"},
        {"tier": 2, "N": 3, "K": [1, 1, 1], "description": "Multiple medium factors", "category": "intermediate"},
        {"tier": 3, "N": 3, "K": [1, 2], "description": "Mixed medium factorization", "category": "intermediate"},
        {"tier": 4, "N": 4, "K": [1, 1, 2], "description": "Complex mixed factorization", "category": "intermediate"},
        {"tier": 5, "N": 2, "K": [3], "description": "RSA-like semiprimes", "category": "advanced"},
        {"tier": 6, "N": 3, "K": [2, 3], "description": "Mixed with one large prime", "category": "advanced"},
        {"tier": 7, "N": 2, "K": [3, 3], "description": "Adversarial large semiprimes", "category": "advanced"},
        {"tier": 8, "N": 3, "K": [3, 3, 3], "description": "Multiple large prime factors", "category": "advanced"},
        {"tier": 9, "N": 6, "K": [2, 2, 2, 2, 2, 2], "description": "Many medium-sized factors", "category": "advanced"},
        
        # New cryptographic strength tiers
        {"tier": 10, "special_type": "quantum_resistant", "bit_length": 2048, 
         "description": "Quantum-resistant multi-factor (2048-bit)", "category": "cryptographic"},
        {"tier": 11, "special_type": "quantum_resistant", "bit_length": 3072, 
         "description": "Quantum-resistant multi-factor (3072-bit)", "category": "cryptographic"},
        {"tier": 12, "special_type": "quantum_resistant", "bit_length": 4096, 
         "description": "Quantum-resistant multi-factor (4096-bit)", "category": "cryptographic"},
        
        # Massive prime tiers
        {"tier": 13, "special_type": "massive_prime", "bits": 1024, 
         "description": "Massive prime factorization (1024-bit)", "category": "extreme"},
        {"tier": 14, "special_type": "massive_prime", "bits": 2048, 
         "description": "Massive prime factorization (2048-bit)", "category": "extreme"},
        
        # Near-prime challenges
        {"tier": 15, "special_type": "near_prime", "digits": 40, 
         "description": "Near-prime challenge - extremely close primes", "category": "extreme"},
        {"tier": 16, "special_type": "near_prime", "digits": 60, 
         "description": "Near-prime challenge - extremely close large primes", "category": "extreme"},
    ]
    
    # Generate samples for each tier
    all_samples = []
    
    # Use multiprocessing for faster generation
    with mp.Pool(MAX_WORKERS) as pool:
        results = []
        for tier_info in tiers:
            results.append(pool.apply_async(generate_tier_samples, (tier_info,)))
        
        # Collect results with progress bar
        for i, result in enumerate(tqdm.tqdm(results, desc="Generating tiers")):
            tier_samples = result.get()
            tier_info = tiers[i]
            
            # Serialize samples
            for sample in tier_samples:
                serialized = serialize_sample(sample, tier_info)
                all_samples.append(serialized)
    
    # Create the final dataset
    dataset = {
        "metadata": {
            "generation_date": datetime.now().isoformat(),
            "framework_version": "1.0.0",
            "samples_per_tier": SAMPLES_PER_TIER,
            "tier_count": len(tiers)
        },
        "samples": all_samples
    }
    
    # Save the dataset
    with open("quantum_benchmark_extreme.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    # Create Hugging Face format
    os.makedirs("quantum_benchmark_ml/huggingface_extreme", exist_ok=True)
    
    # Split into train/test/validation
    random.shuffle(all_samples)
    train_split = int(0.8 * len(all_samples))
    test_split = int(0.9 * len(all_samples))
    
    train_samples = all_samples[:train_split]
    test_samples = all_samples[train_split:test_split]
    validation_samples = all_samples[test_split:]
    
    # Write the splits
    for split_name, split_data in [
        ("train", train_samples),
        ("test", test_samples),
        ("validation", validation_samples)
    ]:
        with open(f"quantum_benchmark_ml/huggingface_extreme/{split_name}.jsonl", "w") as f:
            for sample in split_data:
                f.write(json.dumps(sample) + "\n")
    
    # Create README.md
    readme_content = f"""---
language:
- en
license: mit
---

# Extreme Prime Factorization Benchmark Dataset

This dataset contains extremely challenging prime factorization problems designed as a future-proof benchmark for evaluating advanced mathematical reasoning capabilities of AI models.

## Dataset Structure

- Number of samples: {len(all_samples)}
- Number of tiers: {len(tiers)}
- Framework version: 1.0.0
- Generation date: {datetime.now().isoformat()}

## Features

- `tier_id`: Difficulty tier identifier
- `sample_id`: Sample identifier
- `prompt_id`: Prompt identifier
- `prompt`: The challenge prompt
- `number`: The number to factorize
- `factors`: Ground truth factors (as string)
- `factor_count`: Number of prime factors
- `difficulty`: Calculated difficulty score
- `bit_length`: Bit length of the challenge number
- `category`: Category of the challenge
- `tier_description`: Description of the difficulty tier
- `answer`: Ground truth answer when available

## Tiers

This benchmark includes 17 tiers of increasing difficulty:

- Tiers 0-9: Standard prime factorization challenges
- Tiers 10-12: Cryptographic-strength challenges (2048, 3072, and 4096 bits)
- Tiers 13-14: Massive prime factorization challenges (1024 and 2048 bits)
- Tiers 15-16: Near-prime challenges with extremely close prime pairs

## Difficulty Categories

- elementary: Basic factorization challenges
- intermediate: Medium complexity challenges
- advanced: Challenging factorization problems
- cryptographic: Industry-standard cryptographic strength
- extreme: Beyond current computational limits, future-proof challenges
"""
    
    with open("quantum_benchmark_ml/huggingface_extreme/README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Generated {len(all_samples)} samples across {len(tiers)} tiers")
    print(f"Dataset saved to quantum_benchmark_extreme.json")
    print(f"Hugging Face format saved to quantum_benchmark_ml/huggingface_extreme/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate extreme prime factorization benchmark")
    parser.add_argument("--samples_per_tier", type=int, default=SAMPLES_PER_TIER, 
                      help="Number of samples per tier")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                      help="Number of worker processes")
    
    args = parser.parse_args()
    SAMPLES_PER_TIER = args.samples_per_tier
    MAX_WORKERS = args.workers
    
    start_time = time.time()
    generate_dataset()
    end_time = time.time()
    
    print(f"Generation completed in {end_time - start_time:.2f} seconds")