#!/usr/bin/env python
"""
Simple test script for Prime Number reasoning without full DSPy GRPO.
"""

import json
import os
import math
import random
import pandas as pd
import pyarrow.parquet as pq

# Test if we can load and process the generated data
def test_data_loading():
    """Test loading the generated data."""
    print("\n=== Testing Data Loading ===")
    
    if os.path.exists("data/primes_test.parquet"):
        table = pq.read_table("data/primes_test.parquet")
        df = table.to_pandas()
        print(f"Successfully loaded {len(df)} rows from parquet file")
        
        # Show a few examples
        for i, row in df.head(3).iterrows():
            n = row['n']
            factors = row['fact'].split('×')
            print(f"Number: {n}, Factors: {factors}")
            
            # Verify factors
            product = 1
            for f in factors:
                product *= int(f)
            if product == n:
                print(f"✓ Factors verified: product = {product}")
            else:
                print(f"✗ Factor mismatch: product = {product}, number = {n}")
    else:
        print("Parquet file not found")
    
    if os.path.exists("quantum_benchmark_test.json"):
        with open("quantum_benchmark_test.json", 'r') as f:
            benchmark = json.load(f)
        print(f"Successfully loaded benchmark with {len(benchmark['samples'])} samples")
        
        # Show a few examples
        for sample in benchmark['samples'][:3]:
            number = sample['number']
            factors = sample['factors'].split('×')
            tier = sample['tier_id']
            print(f"Tier: {tier}, Number: {number}, Factors: {factors}")
    else:
        print("Benchmark file not found")

# Test prime factorization functions
def test_factorization():
    """Test basic prime factorization logic."""
    print("\n=== Testing Factorization Logic ===")
    
    def is_prime(n):
        """Check if a number is prime."""
        n = int(n)
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
    
    def factorize(n):
        """Return the prime factorization of n."""
        n = int(n)
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
    
    # Test cases
    test_numbers = [12, 15, 28, 49, 97, 100, 1001, 3978]
    
    for n in test_numbers:
        factors = factorize(n)
        product = 1
        for f in factors:
            assert is_prime(f), f"{f} is not prime"
            product *= f
        
        print(f"Number: {n}, Factors: {factors}, Product: {product}")
        assert product == n, f"Product {product} does not match original number {n}"

# Simulate reasoning about prime factorization
def test_reasoning():
    """Test multi-hop reasoning for prime factorization."""
    print("\n=== Testing Multi-Hop Reasoning ===")
    
    def factorize_with_steps(n):
        """Factorize with reasoning steps to simulate multi-hop approach."""
        n = int(n)
        steps = []
        
        # Step 1: Initial approach
        steps.append(f"To factorize {n}, I'll start by checking if it's even.")
        
        factors = []
        # Step 2: Check for factor of 2
        if n % 2 == 0:
            steps.append(f"{n} is even, so 2 is a factor.")
            factors.append(2)
            n //= 2
            # Continue dividing by 2
            while n % 2 == 0:
                steps.append(f"{n} is still even, dividing by 2 again.")
                factors.append(2)
                n //= 2
            steps.append(f"After removing all factors of 2, we have {n} remaining.")
        else:
            steps.append(f"{n} is odd, so 2 is not a factor.")
        
        # Step 3: Check for other factors
        i = 3
        while i * i <= n:
            if n % i == 0:
                steps.append(f"Found that {i} divides {n} evenly.")
                factors.append(i)
                n //= i
                steps.append(f"After dividing by {i}, we have {n} remaining.")
                
                # Continue dividing by the same factor if applicable
                while n % i == 0:
                    steps.append(f"{n} is still divisible by {i}.")
                    factors.append(i)
                    n //= i
                    steps.append(f"After dividing by {i} again, we have {n} remaining.")
            else:
                steps.append(f"{i} is not a factor, trying the next odd number.")
                i += 2
        
        # Step 4: If n is still > 1, it's a prime factor
        if n > 1:
            steps.append(f"{n} is greater than 1 and has no factors, so it's a prime number.")
            factors.append(n)
        
        # Step 5: Verify the factorization
        product = 1
        for f in factors:
            product *= f
        steps.append(f"Verification: Product of all factors {' × '.join(map(str, factors))} = {product}")
        
        return factors, steps
    
    # Test on a few numbers
    test_numbers = [24, 105, 143]
    
    for n in test_numbers:
        factors, reasoning = factorize_with_steps(n)
        
        print(f"\nMulti-hop reasoning for factorizing {n}:")
        for i, step in enumerate(reasoning, 1):
            print(f"Step {i}: {step}")
        
        print(f"\nFinal factorization: {n} = {' × '.join(map(str, factors))}")
        print("-" * 50)

if __name__ == "__main__":
    # Run the tests
    test_data_loading()
    test_factorization()
    test_reasoning()