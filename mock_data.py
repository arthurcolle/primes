#!/usr/bin/env python
"""
Generate mock prime factorization data for testing the DSPy GRPO implementation.
"""

import os
import json
import random
import math
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def generate_prime(bits):
    """Generate a prime with specified bit length (simplified for testing)."""
    # For testing, we'll use a simpler approach - find a prime in a range
    lower = 2 ** (bits - 1)
    upper = 2 ** bits - 1
    
    # For small bit sizes, just search
    if bits <= 10:
        for n in range(lower, min(lower + 1000, upper)):
            if is_prime(n):
                return n
    
    # For larger sizes, use known primes for testing
    known_primes = {
        16: 65537,
        32: 4294967291,
        64: 18446744073709551557,
        128: 340282366920938463463374607431768211283,
    }
    
    if bits in known_primes:
        return known_primes[bits]
    
    # Default fallback for testing
    return 104729  # A random prime

def factorize(n):
    """Return the prime factorization of n."""
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def generate_mock_data():
    """Generate mock data for testing."""
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Create sample data for small primes
    records = []
    for n in range(2, 1000):
        factors = factorize(n)
        records.append({
            "n": n,
            "fact": "×".join(map(str, factors))
        })
    
    # Add some larger numbers
    for _ in range(20):
        # Generate a number with 2-3 prime factors
        factor_count = random.randint(2, 3)
        factors = []
        for _ in range(factor_count):
            bits = random.randint(4, 16)
            factors.append(generate_prime(bits))
        
        n = math.prod(factors)
        records.append({
            "n": n,
            "fact": "×".join(map(str, factors))
        })
    
    # Write to parquet
    tbl = pa.Table.from_pylist(records)
    pq.write_table(tbl, "data/primes_test.parquet", compression="zstd")
    print(f"Created mock data with {len(records)} records at data/primes_test.parquet")
    
    # Create mock benchmark file
    benchmark = {
        "metadata": {
            "generation_date": "2023-01-01T00:00:00",
            "prime_limit": 1000,
            "framework_version": "0.1.0",
            "samples_per_tier": 5
        },
        "samples": []
    }
    
    # Generate samples for different tiers
    tier_configs = [
        {"tier_id": 0, "bit_length": 8, "factor_count": 2},
        {"tier_id": 1, "bit_length": 16, "factor_count": 2},
        {"tier_id": 2, "bit_length": 32, "factor_count": 3}
    ]
    
    sample_id = 0
    for tier in tier_configs:
        for _ in range(5):  # 5 samples per tier
            # Generate factors
            factors = []
            for _ in range(tier["factor_count"]):
                bits = min(tier["bit_length"] // tier["factor_count"], 16)  # Keep bits reasonable for testing
                factors.append(generate_prime(bits))
            
            n = math.prod(factors)
            factors_str = "×".join(map(str, factors))
            
            # Create sample
            sample = {
                "tier_id": tier["tier_id"],
                "sample_id": sample_id,
                "prompt_id": f"tier{tier['tier_id']}_sample{sample_id}",
                "prompt": f"Find the prime factorization of {n}.",
                "number": str(n),
                "factors": factors_str,
                "factor_count": len(factors),
                "difficulty": tier["bit_length"] * 0.1,
                "bit_length": tier["bit_length"],
                "category": "testing",
                "tier_description": f"Tier {tier['tier_id']} test data",
                "answer": f"The prime factorization of {n} is {factors_str}."
            }
            
            benchmark["samples"].append(sample)
            sample_id += 1
    
    # Write benchmark
    with open("quantum_benchmark_test.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"Created mock benchmark with {len(benchmark['samples'])} samples at quantum_benchmark_test.json")

if __name__ == "__main__":
    generate_mock_data()