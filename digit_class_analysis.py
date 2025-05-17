#!/usr/bin/env python
"""
Advanced Digit-Class Prime Product Analysis Framework

This module provides sophisticated tools to analyze and generate datasets based on the
digit-class prime product problem S(N,K), where:
- N is the maximum number of prime factors
- K is the set of allowed digit-lengths for primes
- S(N,K) is the set of integers expressible as products of at most N primes,
  each with digit-length in K

The framework enables:
1. Exploration of advanced number-theoretic properties 
2. Creation of AI benchmarking datasets with precisely calibrated difficulty
3. Distributed computation of large prime sets
4. Generation of quantum-resistant factorization challenges
5. Specialized AI reasoning and mathematical capability evaluation

Key features:
- Parallelized prime generation and factorization
- Support for extremely large numbers beyond typical benchmarks
- Integration with ML benchmark formats
- Theoretical difficulty estimation based on computational complexity models
"""

import math
import numpy as np
import pandas as pd
import sympy as sp
import random
from tqdm import tqdm
from collections import defaultdict
import json
import multiprocessing as mp
from functools import partial
import os
import pyarrow as pa
import pyarrow.parquet as pq
import time
import hashlib
from pathlib import Path
import logging

# ======== PRIME GENERATION AND CLASSIFICATION ========

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PRIME_CACHE_DIR = Path("data")
PRIME_CACHE_DIR.mkdir(exist_ok=True)

def segmented_sieve(limit, segment_size=10**7):
    """
    Generate primes up to limit using a segmented Sieve of Eratosthenes.
    
    This implementation uses much less memory for very large limits by
    processing the sieve in segments.
    
    Args:
        limit: Upper bound for prime generation
        segment_size: Size of each segment to process
        
    Returns:
        List of primes up to limit
    """
    # Generate small primes first using basic sieve
    sqrt_limit = int(math.sqrt(limit))
    basic_primes = basic_sieve(sqrt_limit)
    
    # Use these small primes to sieve larger segments
    primes = basic_primes.copy()  # Start with the small primes
    
    # Process segments
    for segment_start in range(sqrt_limit + 1, limit + 1, segment_size):
        segment_end = min(segment_start + segment_size - 1, limit)
        segment = [True] * (segment_end - segment_start + 1)
        
        # Sieve the segment using the small primes
        for p in basic_primes:
            # Find the first multiple of p in the segment
            start = (segment_start // p) * p
            if start < segment_start:
                start += p
            
            # Mark all multiples of p in this segment as composite
            for j in range(start, segment_end + 1, p):
                segment[j - segment_start] = False
        
        # Collect the primes from this segment
        for i in range(segment_end - segment_start + 1):
            if segment[i]:
                primes.append(segment_start + i)
    
    return primes

def basic_sieve(limit):
    """Basic Sieve of Eratosthenes for smaller ranges."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def parallel_prime_chunks(limit, num_processes=None, chunk_size=10**7):
    """
    Generate primes in parallel using multiple processes.
    
    Args:
        limit: Upper bound for prime generation
        num_processes: Number of processes to use (defaults to CPU count)
        chunk_size: Size of each chunk to process
        
    Returns:
        List of primes up to limit
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Calculate chunk boundaries
    chunks = []
    for chunk_start in range(2, limit + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, limit)
        chunks.append((chunk_start, chunk_end))
    
    # Process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for chunk_start, chunk_end in chunks:
            results.append(pool.apply_async(basic_sieve, args=(chunk_end,)))
        
        # Wait for all results
        all_primes = []
        for result in results:
            chunk_primes = result.get()
            all_primes.extend(chunk_primes)
        
    # Remove duplicates and sort
    return sorted(list(set(all_primes)))

def generate_primes(limit, use_cache=True, parallel=True):
    """
    Generate all primes up to given limit with caching and parallel processing.
    
    Args:
        limit: Upper bound for prime generation
        use_cache: Whether to use cached primes (default True)
        parallel: Whether to use parallel processing for large limits
        
    Returns:
        List of primes up to limit
    """
    # Check for cached primes
    cache_file = PRIME_CACHE_DIR / f"primes_{limit}.parquet"
    
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached primes from {cache_file}")
        table = pq.read_table(cache_file)
        return table.column('prime').to_pylist()
    
    logger.info(f"Generating primes up to {limit}")
    start_time = time.time()
    
    # Choose appropriate algorithm based on limit size
    if limit < 10**6:
        primes = basic_sieve(limit)
    elif parallel and limit > 10**7:
        logger.info(f"Using parallel sieve with {mp.cpu_count()} processes")
        primes = parallel_prime_chunks(limit)
    else:
        logger.info("Using segmented sieve")
        primes = segmented_sieve(limit)
    
    elapsed = time.time() - start_time
    logger.info(f"Generated {len(primes)} primes in {elapsed:.2f} seconds")
    
    # Cache the results
    if use_cache:
        logger.info(f"Caching primes to {cache_file}")
        table = pa.Table.from_arrays([pa.array(primes)], ['prime'])
        pq.write_table(table, cache_file)
    
    return primes

def classify_primes_by_digits(primes):
    """Group primes by their digit length."""
    classified = defaultdict(list)
    for p in primes:
        digit_length = len(str(p))
        classified[digit_length].append(p)
    return classified

def get_prime_density_by_class(classified_primes, max_digits=10):
    """Calculate density of primes by digit class."""
    densities = {}
    for digit_length in range(1, max_digits + 1):
        if digit_length in classified_primes:
            count = len(classified_primes[digit_length])
            # Theoretical range for d-digit numbers: 10^(d-1) to 10^d - 1
            range_size = 9 * 10**(digit_length - 1)
            densities[digit_length] = count / range_size
    return densities

# ======== S(N,K) SET GENERATION ========

def parallel_sample_generation(params):
    """Helper function for parallel sample generation"""
    N, K, filtered_primes, batch_size, max_value = params
    samples = []
    valid_K = list(filtered_primes.keys())
    
    while len(samples) < batch_size:
        # Determine number of factors for this sample (1 to N)
        factor_count = random.randint(1, N)
        
        # Randomly select digit classes with replacement
        digit_classes = [random.choice(valid_K) for _ in range(factor_count)]
        
        # Select random primes from each chosen digit class
        factors = []
        for digit_class in digit_classes:
            prime = random.choice(filtered_primes[digit_class])
            factors.append(prime)
        
        # Calculate product
        product = math.prod(factors)
        
        # Skip if exceeds maximum value
        if product > max_value:
            continue
            
        # Calculate signature (omega vector)
        signature = {}
        for d in valid_K:
            signature[d] = sum(1 for f in factors if len(str(f)) == d)
        
        # Record sample
        samples.append({
            "n": product,
            "factors": sorted(factors),
            "signature": signature,
            "factor_count": factor_count
        })
    
    return samples

def generate_S_N_K_samples(N, K, classified_primes, sample_count=1000, max_value=10**12, parallel=True):
    """
    Generate random samples from S(N,K) set with parallel processing option.
    
    Args:
        N: Maximum number of prime factors
        K: Set of allowed digit lengths
        classified_primes: Dictionary of primes classified by digit length
        sample_count: Number of samples to generate
        max_value: Maximum allowed value for generated numbers
        parallel: Whether to use parallel processing
        
    Returns:
        List of dictionaries with number, factors, and signature
    """
    # Validate input
    available_digit_classes = set(classified_primes.keys())
    valid_K = set(K) & available_digit_classes
    
    if not valid_K:
        raise ValueError(f"None of the requested digit classes {K} are available in the classified primes.")
    
    # Filter primes to only use requested digit classes
    filtered_primes = {d: primes for d, primes in classified_primes.items() if d in valid_K}
    
    # Option 1: Parallel processing for larger sample counts
    if parallel and sample_count >= 100 and mp.cpu_count() > 1:
        num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max
        batch_size = math.ceil(sample_count / num_processes)
        
        params = [(N, K, filtered_primes, batch_size, max_value) for _ in range(num_processes)]
        
        with mp.Pool(processes=num_processes) as pool:
            logger.info(f"Generating samples using {num_processes} parallel processes")
            results = pool.map(parallel_sample_generation, params)
            
        # Combine results
        samples = []
        for batch in results:
            samples.extend(batch)
            
        # Trim to exact sample count
        samples = samples[:sample_count]
        
    # Option 2: Sequential processing
    else:
        samples = []
        for _ in tqdm(range(sample_count), desc=f"Generating S({N},{K}) samples"):
            # Determine number of factors for this sample (1 to N)
            factor_count = random.randint(1, N)
            
            # Randomly select digit classes with replacement
            digit_classes = [random.choice(list(valid_K)) for _ in range(factor_count)]
            
            # Select random primes from each chosen digit class
            factors = []
            for digit_class in digit_classes:
                prime = random.choice(filtered_primes[digit_class])
                factors.append(prime)
            
            # Calculate product
            product = math.prod(factors)
            
            # Skip if exceeds maximum value
            if product > max_value:
                continue
                
            # Calculate signature (omega vector)
            signature = {}
            for d in valid_K:
                signature[d] = sum(1 for f in factors if len(str(f)) == d)
            
            # Record sample
            samples.append({
                "n": product,
                "factors": sorted(factors),
                "signature": signature,
                "factor_count": factor_count
            })
            
            if len(samples) >= sample_count:
                break
    
    # Generate difficulty metadata for each sample
    for sample in samples:
        sample["difficulty"] = calculate_factorization_difficulty(sample["n"], sample["factors"])
    
    return samples

def calculate_factorization_difficulty(n, factors):
    """
    Calculate the computational difficulty of factoring n.
    
    This uses heuristics based on:
    1. Bit length of n
    2. Number of factors
    3. Size distribution of factors
    4. Proximity of factors to each other
    """
    # Bit complexity
    bit_length = n.bit_length()
    
    # Number of prime factors
    factor_count = len(factors)
    
    # Size distribution - variance in digit lengths
    digit_lengths = [len(str(f)) for f in factors]
    size_variance = np.var(digit_lengths) if factor_count > 1 else 0
    
    # Minimum gap between consecutive factors
    if factor_count > 1:
        sorted_factors = sorted(factors)
        relative_gaps = [(sorted_factors[i+1] - sorted_factors[i]) / sorted_factors[i] 
                         for i in range(factor_count-1)]
        min_gap = min(relative_gaps) if relative_gaps else 1.0
    else:
        min_gap = 1.0
    
    # GNFS difficulty estimate (simplified)
    gnfs_difficulty = 1.923 * (bit_length**(1/3)) * (math.log(bit_length)**(2/3))
    
    # Combined measure with weights
    base_score = bit_length * math.sqrt(factor_count)
    distribution_factor = 1 + (0.5 * size_variance)
    gap_factor = 1 + (1 / (1 + 10 * min_gap))  # Closer gaps are harder
    
    final_score = base_score * distribution_factor * gap_factor
    
    return {
        "bit_length": bit_length,
        "factor_count": factor_count,
        "size_variance": float(size_variance),
        "min_relative_gap": float(min_gap),
        "gnfs_estimate": float(gnfs_difficulty),
        "combined_score": float(final_score)
    }

# ======== ANALYTIC FUNCTIONS ========

def estimate_S_N_K_density(N, K, max_exponent=10):
    """
    Estimate the density of S(N,K) using analytic approximations.
    
    This implements a simplified version of the zeta-function approach.
    """
    # Constants
    ln10 = math.log(10)
    
    # Prime density approximation by digit class
    prime_density_approx = {}
    for d in K:
        # Approximate count of primes in range 10^(d-1) to 10^d
        # Using prime number theorem: π(x) ≈ x/ln(x)
        lower_bound = 10**(d-1)
        upper_bound = 10**d - 1
        avg_x = (upper_bound + lower_bound) / 2
        prime_density_approx[d] = 1 / (d * ln10)
    
    # Approximate growth rate for log-uniform N factors
    total_density = sum(prime_density_approx.values())
    
    # Return growth coefficient for power of log
    return (total_density**N) / math.factorial(N)

def complexity_measure(n, factors, signature):
    """
    Calculate a complexity measure for factoring n based on:
    1. Bit length
    2. Distribution of factor sizes
    3. Entropy of signature
    """
    # Bit complexity
    bit_length = n.bit_length()
    
    # Size distribution measure
    size_variance = np.var([len(str(f)) for f in factors]) if factors else 0
    
    # Entropy of signature distribution
    total_factors = sum(signature.values())
    if total_factors > 0:
        probabilities = [count/total_factors for count in signature.values() if count > 0]
        entropy = -sum(p * math.log2(p) for p in probabilities)
    else:
        entropy = 0
        
    # Combined measure (can be adjusted based on empirical testing)
    return {
        "bit_length": bit_length,
        "size_variance": size_variance,
        "signature_entropy": entropy,
        "combined_score": bit_length * (1 + entropy) * (1 + size_variance)
    }

# ======== AI BENCHMARK UTILITIES ========

def generate_benchmark_prompts(samples, formats=None, capabilities=None):
    """
    Generate benchmark prompts in various formats for AI evaluation.
    
    Args:
        samples: List of sample dictionaries with n, factors, signature
        formats: List of format strings to use; defaults to basic formats
        capabilities: List of capabilities to target with specialized prompts
        
    Returns:
        Dictionary mapping sample IDs to prompts in different formats
    """
    # Basic formats for factorization tasks
    basic_formats = [
        "Factor the number {n} into its prime factors.",
        "Is {n} a product of exactly {factor_count} primes? If so, what are they?",
        "Find the signature ω_K({n}) for K = {K}.",
        "Determine whether {n} belongs to S({N},{K})."
    ]
    
    # Advanced formats by capability type
    capability_formats = {
        "basic_arithmetic": [
            "What is the product of all prime factors of {n}?",
            "Is {n} divisible by any single-digit prime number? If so, which ones?"
        ],
        
        "number_theory_concepts": [
            "Calculate φ(n) for n = {n}, where φ is Euler's totient function.",
            "What is the sum of all divisors of {n}?",
            "Determine whether {n} is a perfect number, abundant number, or deficient number."
        ],
        
        "factorization_algorithms": [
            "Use the quadratic sieve algorithm to factor {n}.",
            "Apply Pollard's rho algorithm to find a non-trivial factor of {n}.",
            "Use trial division to identify the smallest prime factor of {n}."
        ],
        
        "number_patterns": [
            "Identify any patterns in the prime factorization of {n}.",
            "Are the prime factors of {n} all in an arithmetic progression? If not, find the closest arithmetic progression."
        ],
        
        "algorithm_application": [
            "If RSA encryption used {n} as the modulus, would it be secure? Explain why or why not.",
            "How many modular multiplications would be required to factor {n} using the General Number Field Sieve?"
        ],
        
        "mathematical_insight": [
            "Without directly computing the factors, explain the most efficient strategy to factorize {n}.",
            "Based on the digit pattern of {n}, what properties can you infer about its prime factorization?"
        ],
        
        "advanced_factorization": [
            "Use algebraic techniques to find non-trivial factors of {n}.",
            "Does {n} have any factors that are Mersenne primes or Fermat primes?"
        ],
        
        "mathematical_creativity": [
            "Design a probabilistic algorithm that would be especially efficient for factoring numbers like {n}.",
            "Explore three different factorization approaches for {n} and compare their theoretical complexity."
        ],
        
        "cryptographic_reasoning": [
            "If {n} were used as an RSA modulus, estimate how long it would take to break with current technology.",
            "Analyze the security implications if {n} were used in a cryptographic protocol."
        ],
        
        "computational_complexity": [
            "Calculate the bit operations required to factor {n} using the best known classical algorithm.",
            "Compare the computational complexity of factoring {n} using GNFS versus Shor's algorithm."
        ],
        
        "post_quantum_cryptography": [
            "Why would Shor's algorithm struggle with factoring {n} compared to a traditional RSA modulus?",
            "Explain why having multiple large prime factors makes {n} more resistant to quantum factorization."
        ],
        
        "algorithm_limitations": [
            "Explain why the structure of {n} poses challenges for commonly used factorization algorithms.",
            "What mathematical properties of {n} would cause ECM factorization to perform poorly?"
        ],
        
        "adversarial_reasoning": [
            "The prime factors of {n} have a special property making them difficult to find. Identify this property.",
            "Design a strategy to find the prime factors of {n} when they are known to be unusually close to each other."
        ]
    }
    
    # Use provided formats or defaults
    if formats is None:
        formats = basic_formats
        
    benchmark_prompts = {}
    
    for idx, sample in enumerate(samples):
        n = sample["n"]
        factors = sample["factors"]
        signature = sample.get("signature", {})
        factor_count = sample.get("factor_count", len(factors))
        sample_type = sample.get("type", "standard")
        
        sample_id = f"sample_{idx}"
        benchmark_prompts[sample_id] = {
            "sample": sample,
            "prompts": [],
            "capability_prompts": {}
        }
        
        K_str = str(list(signature.keys())) if signature else "[]"
        N = max(factor_count, 1)  # Ensure N is at least the actual factor count
        
        # Base prompts
        for fmt in formats:
            prompt = fmt.format(
                n=n, 
                factors=factors,
                signature=signature,
                factor_count=factor_count,
                K=K_str,
                N=N
            )
            benchmark_prompts[sample_id]["prompts"].append(prompt)
        
        # Add capability-specific prompts if requested
        if capabilities:
            for capability in capabilities:
                if capability in capability_formats:
                    cap_prompts = []
                    for fmt in capability_formats[capability]:
                        prompt = fmt.format(
                            n=n, 
                            factors=factors,
                            signature=signature,
                            factor_count=factor_count,
                            K=K_str,
                            N=N
                        )
                        cap_prompts.append(prompt)
                    benchmark_prompts[sample_id]["capability_prompts"][capability] = cap_prompts
    
    return benchmark_prompts

def generate_specialized_benchmark(tier_configs, capabilities, sample_count=25):
    """
    Generate a benchmark dataset specialized for testing specific AI capabilities.
    
    Args:
        tier_configs: List of tier configuration dictionaries
        capabilities: List of capabilities to test
        sample_count: Number of samples per tier
        
    Returns:
        A benchmark dataset with specialized prompts
    """
    dataset = {
        "metadata": {
            "generation_date": pd.Timestamp.now().isoformat(),
            "framework_version": "0.2.0",
            "capabilities": capabilities,
            "sample_count_per_tier": sample_count
        },
        "tiers": []
    }
    
    # Generate primes first
    limit = 10**7  # Default limit for prime generation
    for tier in tier_configs:
        if "K" in tier and tier["K"]:
            max_digits = max(tier["K"])
            needed_limit = 10**(max_digits + 1)
            limit = max(limit, needed_limit)
    
    logger.info(f"Generating primes up to {limit} for specialized benchmark")
    primes = generate_primes(limit)
    classified = classify_primes_by_digits(primes)
    
    # Generate samples for each tier
    for tier_config in tier_configs:
        tier_data = {"tier_info": tier_config, "samples": []}
        
        # Handle different types of tier configurations
        if "special_type" in tier_config:
            special_type = tier_config["special_type"]
            
            if special_type == "quantum_resistant":
                bit_length = tier_config.get("bit_length", 256)
                min_factor_bits = tier_config.get("min_factor_bits", 64)
                
                for _ in range(sample_count):
                    sample = generate_quantum_resistant_challenge(bit_length, min_factor_bits)
                    tier_data["samples"].append(sample)
                    
            elif special_type == "near_prime":
                digits = tier_config.get("digits", 40)
                gap_factor = tier_config.get("gap_factor", 0.0001)
                
                for _ in range(sample_count):
                    sample = generate_near_prime_challenge(digits, gap_factor)
                    tier_data["samples"].append(sample)
        
        # Standard S(N,K) tier
        elif "N" in tier_config and "K" in tier_config:
            N, K = tier_config["N"], tier_config["K"]
            max_value = 10**(N*max(K)+1)
            
            try:
                samples = generate_S_N_K_samples(
                    N, K, classified, 
                    sample_count=sample_count,
                    max_value=max_value
                )
                tier_data["samples"] = samples
            except ValueError as e:
                logger.error(f"Error generating samples for tier {tier_config['tier']}: {e}")
                continue
        
        # Generate capability-specific prompts
        tier_data["prompts"] = generate_benchmark_prompts(
            tier_data["samples"], 
            capabilities=capabilities
        )
        
        dataset["tiers"].append(tier_data)
        logger.info(f"Generated specialized benchmark samples for tier {tier_config.get('tier')}")
    
    return dataset

def generate_quantum_resistant_challenge(bit_length=256, min_factor_bits=128):
    """
    Generate a quantum-resistant factorization challenge.
    
    These challenges are designed to be resistant to Shor's algorithm by using
    multiple large prime factors rather than just two.
    
    Args:
        bit_length: Target bit length for the full number
        min_factor_bits: Minimum bit length for each prime factor
        
    Returns:
        Dictionary with challenge number and its factors
    """
    # Use sympy to generate large random primes
    logger.info(f"Generating quantum-resistant challenge of {bit_length} bits")
    
    factors = []
    current_bits = 0
    max_factor_bits = bit_length // 2  # No single factor should be too large
    
    while current_bits < bit_length * 0.9:  # Aim for at least 90% of desired bits
        # Generate random factor size between min and max
        factor_bits = random.randint(min_factor_bits, max_factor_bits)
        
        # Generate prime of this bit size
        # We add a random offset to ensure primes are well-separated
        lower_bound = 2**(factor_bits-1)
        upper_bound = 2**factor_bits - 1
        
        # Find a random probable prime in range
        p = sp.randprime(lower_bound, upper_bound)
        
        factors.append(int(p))
        current_bits = sum(f.bit_length() for f in factors)
    
    # Calculate product
    n = math.prod(factors)
    
    return {
        "n": n,
        "factors": factors,
        "bit_length": n.bit_length(),
        "factor_count": len(factors),
        "type": "quantum_resistant"
    }

def generate_near_prime_challenge(digits=100, gap_factor=0.001):
    """
    Generate a challenge with two very closely spaced primes.
    
    These are particularly difficult for most factorization algorithms.
    
    Args:
        digits: Number of digits for each prime
        gap_factor: Maximum relative gap between primes
        
    Returns:
        Dictionary with challenge number and its factors
    """
    # Use sympy to find a large prime
    lower_bound = 10**(digits-1)
    upper_bound = 10**digits - 1
    
    p1 = sp.randprime(lower_bound, upper_bound)
    
    # Find another prime close to p1
    max_gap = int(p1 * gap_factor)
    if max_gap < 2:
        max_gap = 2
    
    # Check a narrow range for another prime
    for offset in range(2, max_gap, 2):  # Step by 2 to check only odd numbers
        if sp.isprime(p1 + offset):
            p2 = p1 + offset
            break
    else:
        # If no close prime found, generate a new one and try again
        logger.warning(f"No close prime found within gap {max_gap}, using next prime")
        p2 = sp.nextprime(p1)
    
    # Calculate product and gap
    n = p1 * p2
    gap = (p2 - p1) / p1
    
    return {
        "n": n,
        "factors": [int(p1), int(p2)],
        "bit_length": n.bit_length(),
        "gap": float(gap),
        "type": "near_prime"
    }

def generate_difficulty_tiers(max_digits=10, max_factors=7, include_quantum=True):
    """
    Generate a progression of difficulty tiers for benchmarking.
    
    Args:
        max_digits: Maximum digit length for primes
        max_factors: Maximum number of factors
        include_quantum: Whether to include quantum-resistant challenges
    
    Returns:
        List of tier dictionaries with configurations
    """
    tiers = []
    
    # === STANDARD TIERS ===
    
    # Tier 0: Single-digit primes, up to 2 factors
    tiers.append({"tier": 0, "N": 2, "K": [1], "description": "Basic factorization of small semiprimes",
                 "category": "elementary"})
    
    # Tier 1: Single and double-digit primes, up to 2 factors
    tiers.append({"tier": 1, "N": 2, "K": [1, 2], "description": "Mixed small semiprimes",
                 "category": "elementary"})
    
    # Tier 2: Double-digit primes only, up to 3 factors
    tiers.append({"tier": 2, "N": 3, "K": [2], "description": "Multiple medium factors",
                 "category": "elementary"})
    
    # Tier 3: Mixed primes up to 3 digits, up to 3 factors
    tiers.append({"tier": 3, "N": 3, "K": [1, 2, 3], "description": "Mixed medium factorization",
                 "category": "intermediate"})
    
    # Tier 4: Primes of 2-4 digits, up to 4 factors
    tiers.append({"tier": 4, "N": 4, "K": [2, 3, 4], "description": "Complex mixed factorization",
                 "category": "intermediate"})
    
    # Tier 5: RSA-like with exactly 2 large factors
    tiers.append({"tier": 5, "N": 2, "K": [4, 5], "description": "RSA-like semiprimes",
                 "category": "advanced"})
    
    # Tier 6: Mixture with one large prime
    tiers.append({"tier": 6, "N": 3, "K": [1, 2, 6], "description": "Mixed with one large prime",
                 "category": "advanced"})
    
    # Tier 7: Adversarial - closely spaced large primes
    tiers.append({"tier": 7, "N": 2, "K": [5], "description": "Adversarial large semiprimes",
                 "category": "advanced"})
    
    # === ADVANCED TIERS ===
    
    # Tier 8: Multiple large factors (5+ digits)
    tiers.append({"tier": 8, "N": 3, "K": [5, 6], "description": "Multiple large prime factors",
                 "category": "research"})
    
    # Tier 9: Complex factorization with many factors
    tiers.append({"tier": 9, "N": 6, "K": [2, 3, 4], "description": "Many medium-sized factors",
                 "category": "research"})
    
    # Tier 10: RSA-challenge like very large semiprimes
    tiers.append({"tier": 10, "N": 2, "K": [7, 8], "description": "Very large semiprimes (RSA challenge)",
                 "category": "cryptographic"})
    
    # Tier 11: Extreme - mixture of many factors including very large ones
    tiers.append({"tier": 11, "N": 5, "K": [1, 3, 5, 7], "description": "Mixed extreme factorization",
                 "category": "cryptographic"})
    
    # Tier 12: Cryptographic strength
    tiers.append({"tier": 12, "N": 2, "K": [9, 10], "description": "Cryptographic-grade semiprimes",
                 "category": "cryptographic"})
    
    # === QUANTUM-RESISTANT CHALLENGES ===
    if include_quantum:
        # Tier 13: Quantum resistant - multiple large factors
        tiers.append({
            "tier": 13, 
            "special_type": "quantum_resistant",
            "bit_length": 256,
            "min_factor_bits": 64,
            "description": "Quantum-resistant multi-factor (256-bit)",
            "category": "quantum_resistant"
        })
        
        # Tier 14: Extreme quantum resistant 
        tiers.append({
            "tier": 14, 
            "special_type": "quantum_resistant",
            "bit_length": 512,
            "min_factor_bits": 128,
            "description": "Quantum-resistant multi-factor (512-bit)",
            "category": "quantum_resistant"
        })
        
        # Tier 15: Near-primes - extremely close primes
        tiers.append({
            "tier": 15, 
            "special_type": "near_prime",
            "digits": 40,
            "gap_factor": 0.0001,
            "description": "Near-prime challenge - extremely close primes",
            "category": "adversarial"
        })
    
    # Add capability focus for each tier
    capability_mapping = {
        "elementary": ["basic_arithmetic", "number_theory_concepts"],
        "intermediate": ["factorization_algorithms", "number_patterns"],
        "advanced": ["algorithm_application", "mathematical_insight"],
        "research": ["advanced_factorization", "mathematical_creativity"],
        "cryptographic": ["cryptographic_reasoning", "computational_complexity"],
        "quantum_resistant": ["post_quantum_cryptography", "algorithm_limitations"],
        "adversarial": ["adversarial_reasoning", "algorithm_robustness"]
    }
    
    for tier in tiers:
        category = tier.get("category")
        if category in capability_mapping:
            tier["capabilities"] = capability_mapping[category]
    
    return tiers

# ======== LATTICE-THEORETIC FUNCTIONS ========

def signature_to_point(signature, K):
    """Convert a signature to a point in |K|-dimensional space."""
    return [signature.get(d, 0) for d in sorted(K)]

def distance_in_signature_space(sig1, sig2, K):
    """Calculate Euclidean distance between two signatures in signature space."""
    point1 = signature_to_point(sig1, K)
    point2 = signature_to_point(sig2, K)
    return math.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))

def find_similar_signatures(samples, target_signature, K, n=5):
    """Find n samples with signatures most similar to target_signature."""
    distances = []
    for sample in samples:
        dist = distance_in_signature_space(sample["signature"], target_signature, K)
        distances.append((dist, sample))
    
    # Return n closest samples
    return [s for _, s in sorted(distances)[:n]]

# ======== MAIN DEMO FUNCTIONS ========

def demo_basic_analysis(limit=10000):
    """Run a basic analysis demonstrating the framework capabilities."""
    print("Generating primes up to", limit)
    primes = generate_primes(limit)
    print(f"Found {len(primes)} primes up to {limit}")
    
    classified = classify_primes_by_digits(primes)
    print("\nPrime counts by digit length:")
    for digits, prime_list in sorted(classified.items()):
        print(f"{digits} digits: {len(prime_list)} primes")
    
    print("\nPrime densities by digit class:")
    densities = get_prime_density_by_class(classified)
    for digits, density in sorted(densities.items()):
        print(f"{digits} digits: {density:.6f} ({density*100:.4f}%)")
    
    # Generate samples for a few S(N,K) configurations
    configurations = [
        (2, [1]),      # Semiprimes with single-digit primes
        (2, [1, 2]),   # Semiprimes with 1-2 digit primes
        (3, [1, 2])    # Products of up to 3 primes with 1-2 digits
    ]
    
    all_samples = []
    for N, K in configurations:
        print(f"\nGenerating 10 samples from S({N}, {K}):")
        samples = generate_S_N_K_samples(N, K, classified, sample_count=10, max_value=10**9)
        all_samples.extend(samples)
        
        for sample in samples:
            factor_str = " × ".join(map(str, sample["factors"]))
            sig_str = ", ".join(f"{d}:{c}" for d, c in sample["signature"].items() if c > 0)
            print(f"{sample['n']} = {factor_str} (signature: {sig_str})")
    
    # Analyze complexity of a few samples
    print("\nComplexity analysis of samples:")
    for idx, sample in enumerate(all_samples[:5]):
        complexity = sample.get("difficulty", complexity_measure(sample["n"], sample["factors"], sample["signature"]))
        print(f"Sample {idx+1}: {sample['n']}")
        print(f"  Bit length: {complexity['bit_length']}")
        print(f"  Size variance: {complexity.get('size_variance', 0):.4f}")
        print(f"  Signature entropy: {complexity.get('signature_entropy', 0):.4f}")
        print(f"  Combined score: {complexity.get('combined_score', complexity['bit_length']):.4f}")
    
    # Generate benchmark prompts
    print("\nSample benchmark prompts:")
    prompts = generate_benchmark_prompts(all_samples[:2])
    for sample_id, data in prompts.items():
        print(f"\n{sample_id} (n={data['sample']['n']}):")
        for i, prompt in enumerate(data["prompts"]):
            print(f"  Prompt {i+1}: {prompt}")
    
    # Show difficulty tiers
    print("\nDifficulty progression tiers:")
    tiers = generate_difficulty_tiers()
    for tier in tiers:
        # Only print standard tiers with N and K
        if "N" in tier and "K" in tier:
            print(f"Tier {tier['tier']}: N={tier['N']}, K={tier['K']} - {tier['description']}")
        elif "special_type" in tier:
            print(f"Tier {tier['tier']}: Special type: {tier['special_type']} - {tier['description']}")
            
    # Show example of quantum-resistant challenge
    print("\nQuantum-resistant challenge example:")
    qr_challenge = generate_quantum_resistant_challenge(bit_length=128, min_factor_bits=32)
    print(f"Number: {qr_challenge['n']}")
    print(f"Factors: {' × '.join(map(str, qr_challenge['factors']))}")
    print(f"Bit length: {qr_challenge['bit_length']}")
    print(f"Factor count: {qr_challenge['factor_count']}")

def export_dataset_to_ml_format(dataset, output_dir="ml_benchmarks", formats=None):
    """
    Export benchmark dataset to various ML-friendly formats.
    
    Args:
        dataset: The benchmark dataset to export
        output_dir: Directory to save exported files
        formats: List of formats to export (parquet, csv, jsonl)
    
    Returns:
        Dictionary of paths to exported files
    """
    if formats is None:
        formats = ["parquet", "csv", "jsonl", "huggingface"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Flatten the dataset for easier ML consumption
    flattened_data = []
    
    for tier_idx, tier_data in enumerate(dataset["tiers"]):
        tier_info = tier_data["tier_info"]
        tier_id = tier_info.get("tier", tier_idx)
        
        for sample_id, prompt_data in tier_data["prompts"].items():
            sample = prompt_data["sample"]
            
            # Get all prompts
            all_prompts = prompt_data["prompts"]
            
            # Add capability-specific prompts if available
            capability_prompts = prompt_data.get("capability_prompts", {})
            for capability, prompts in capability_prompts.items():
                for prompt in prompts:
                    all_prompts.append(f"[{capability}] {prompt}")
            
            # Create an entry for each prompt
            for prompt_idx, prompt in enumerate(all_prompts):
                # Extract factors as string for easier handling in ML contexts
                factors_str = " × ".join(map(str, sample["factors"]))
                
                entry = {
                    "tier_id": tier_id,
                    "sample_id": sample_id,
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "number": str(sample["n"]),  # Convert to string to avoid integer overflow
                    "factors": factors_str,
                    "factor_count": sample.get("factor_count", len(sample["factors"])),
                    "difficulty": float(sample.get("difficulty", {}).get("combined_score", 0)),
                    "bit_length": int(sample.get("bit_length", sample["n"].bit_length())),
                    "category": tier_info.get("category", "standard"),
                    "tier_description": tier_info.get("description", ""),
                }
                
                # Add ground truth answer
                if "Factor" in prompt or "factor" in prompt:
                    entry["answer"] = factors_str
                elif "product" in prompt.lower():
                    entry["answer"] = str(sample["n"])
                elif "exactly" in prompt and "prime" in prompt:
                    entry["answer"] = f"Yes, the prime factors are {factors_str}"
                    
                flattened_data.append(entry)
    
    export_paths = {}
    
    # Export to various formats
    if "parquet" in formats:
        parquet_path = output_path / "benchmark.parquet"
        df = pd.DataFrame(flattened_data)
        pq.write_table(pa.Table.from_pandas(df), str(parquet_path))
        export_paths["parquet"] = str(parquet_path)
        logger.info(f"Exported dataset to Parquet: {parquet_path}")
    
    if "csv" in formats:
        csv_path = output_path / "benchmark.csv"
        pd.DataFrame(flattened_data).to_csv(csv_path, index=False)
        export_paths["csv"] = str(csv_path)
        logger.info(f"Exported dataset to CSV: {csv_path}")
    
    if "jsonl" in formats:
        jsonl_path = output_path / "benchmark.jsonl"
        with open(jsonl_path, 'w') as f:
            for entry in flattened_data:
                f.write(json.dumps(entry) + '\n')
        export_paths["jsonl"] = str(jsonl_path)
        logger.info(f"Exported dataset to JSONL: {jsonl_path}")
    
    if "huggingface" in formats:
        # Create a HuggingFace-compatible dataset structure
        hf_data = {
            "train": [],
            "validation": [],
            "test": []
        }
        
        # Split data into train/val/test (80/10/10)
        random.shuffle(flattened_data)
        n = len(flattened_data)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        hf_data["train"] = flattened_data[:train_end]
        hf_data["validation"] = flattened_data[train_end:val_end]
        hf_data["test"] = flattened_data[val_end:]
        
        # Save as separate jsonl files
        hf_dir = output_path / "huggingface"
        hf_dir.mkdir(exist_ok=True)
        
        for split, data in hf_data.items():
            split_path = hf_dir / f"{split}.jsonl"
            with open(split_path, 'w') as f:
                for entry in data:
                    f.write(json.dumps(entry) + '\n')
        
        # Create dataset card
        dataset_card = f"""---
language:
- en
license: mit
---

# Prime Factorization Benchmark Dataset

This dataset contains prime factorization challenges of varying difficulty levels designed to evaluate mathematical reasoning capabilities of AI models.

## Dataset Structure

- Number of samples: {len(flattened_data)}
- Number of tiers: {len(dataset["tiers"])}
- Framework version: {dataset["metadata"]["framework_version"]}
- Generation date: {dataset["metadata"]["generation_date"]}

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
"""
        
        with open(hf_dir / "README.md", 'w') as f:
            f.write(dataset_card)
            
        export_paths["huggingface"] = str(hf_dir)
        logger.info(f"Exported dataset in HuggingFace format: {hf_dir}")
    
    return export_paths

def save_benchmark_dataset(filename="benchmark_dataset.json", limit=1000000, samples_per_tier=100, include_quantum=True, export_ml=False):
    """
    Generate and save a complete benchmark dataset.
    
    Args:
        filename: Output JSON filename
        limit: Prime generation limit
        samples_per_tier: Number of samples per tier
        include_quantum: Whether to include quantum-resistant challenges
        export_ml: Whether to export dataset in ML-friendly formats
    
    Returns:
        Path to saved dataset
    """
    # Generate prime base
    logger.info(f"Generating primes up to {limit}...")
    primes = generate_primes(limit)
    classified = classify_primes_by_digits(primes)
    logger.info(f"Found {len(primes)} primes up to {limit}")
    logger.info(f"Max digit length prime: {len(str(primes[-1]))} digits")
    
    # Get difficulty tiers
    tiers = generate_difficulty_tiers(include_quantum=include_quantum)
    
    # Generate samples for each tier
    dataset = {
        "metadata": {
            "generation_date": pd.Timestamp.now().isoformat(),
            "prime_limit": limit,
            "framework_version": "0.2.0",
            "samples_per_tier": samples_per_tier
        },
        "tiers": []
    }
    
    for tier in tiers:
        # Handle special tier types
        if "special_type" in tier:
            special_type = tier["special_type"]
            tier_data = {"tier_info": tier, "samples": []}
            
            try:
                if special_type == "quantum_resistant":
                    bit_length = tier.get("bit_length", 256)
                    min_factor_bits = tier.get("min_factor_bits", 64)
                    
                    for _ in range(samples_per_tier):
                        sample = generate_quantum_resistant_challenge(bit_length, min_factor_bits)
                        tier_data["samples"].append(sample)
                        
                elif special_type == "near_prime":
                    digits = tier.get("digits", 40)
                    gap_factor = tier.get("gap_factor", 0.0001)
                    
                    for _ in range(samples_per_tier):
                        sample = generate_near_prime_challenge(digits, gap_factor)
                        tier_data["samples"].append(sample)
                
                # Add prompts
                tier_data["prompts"] = generate_benchmark_prompts(tier_data["samples"])
                dataset["tiers"].append(tier_data)
                logger.info(f"Generated {len(tier_data['samples'])} samples for Tier {tier['tier']} - {tier['description']}")
                
            except Exception as e:
                logger.error(f"Error generating samples for special tier {tier['tier']}: {e}")
                continue
        
        # Handle standard tiers
        elif "N" in tier and "K" in tier:
            N, K = tier["N"], tier["K"]
            try:
                max_digit_length = max(K)
                if max_digit_length > len(str(primes[-1])):
                    logger.warning(f"Tier {tier['tier']} requires {max_digit_length}-digit primes, but largest prime is only {len(str(primes[-1]))}-digits")
                    logger.warning(f"Skipping tier {tier['tier']}")
                    continue
                    
                tier_samples = generate_S_N_K_samples(
                    N, K, classified, 
                    sample_count=samples_per_tier,
                    max_value=10**(N*max(K)+1),
                    parallel=True
                )
                
                # Add prompts to each sample
                tier_data = {
                    "tier_info": tier,
                    "samples": tier_samples,
                    "prompts": generate_benchmark_prompts(tier_samples)
                }
                
                dataset["tiers"].append(tier_data)
                logger.info(f"Generated {len(tier_samples)} samples for Tier {tier['tier']} - {tier['description']}")
                
            except ValueError as e:
                logger.error(f"Error generating samples for Tier {tier['tier']}: {e}")
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Benchmark dataset saved to {filename}")
    
    # Export to ML formats if requested
    if export_ml:
        output_dir = Path(filename).stem + "_ml"
        export_paths = export_dataset_to_ml_format(dataset, output_dir)
        logger.info(f"Dataset exported to ML formats in {output_dir}")
    
    return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Digit-Class Prime Product Analysis Framework")
    parser.add_argument("--demo", action="store_true", help="Run basic demo")
    parser.add_argument("--benchmark", action="store_true", help="Generate benchmark dataset")
    parser.add_argument("--specialized", type=str, help="Generate specialized benchmark for specific capabilities (comma-separated)")
    parser.add_argument("--limit", type=int, default=1000000, help="Prime generation limit")
    parser.add_argument("--samples", type=int, default=100, help="Samples per tier")
    parser.add_argument("--output", type=str, default="benchmark_dataset.json", help="Output file for benchmark dataset")
    parser.add_argument("--quantum", action="store_true", help="Include quantum-resistant challenges")
    parser.add_argument("--ml-export", action="store_true", help="Export dataset in ML-friendly formats")
    parser.add_argument("--parallel", action="store_true", default=True, help="Use parallel processing")
    parser.add_argument("--tiers", type=str, help="Specific tiers to generate (comma-separated)")
    parser.add_argument("--cache", action="store_true", default=True, help="Use cached primes when available")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "INFO"
    logging.basicConfig(level=getattr(logging, log_level),
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.demo:
        demo_basic_analysis(args.limit)
    
    elif args.benchmark:
        # Regular benchmark generation
        save_benchmark_dataset(
            filename=args.output,
            limit=args.limit,
            samples_per_tier=args.samples,
            include_quantum=args.quantum,
            export_ml=args.ml_export
        )
    
    elif args.specialized:
        # Specialized benchmark for specific capabilities
        capabilities = [cap.strip() for cap in args.specialized.split(",")]
        logger.info(f"Generating specialized benchmark for capabilities: {capabilities}")
        
        # Get selected tiers
        all_tiers = generate_difficulty_tiers(include_quantum=args.quantum)
        
        if args.tiers:
            # Filter to specified tiers
            tier_ids = [int(t.strip()) for t in args.tiers.split(",")]
            selected_tiers = [t for t in all_tiers if t.get("tier") in tier_ids]
        else:
            # Use all tiers
            selected_tiers = all_tiers
        
        # Generate specialized benchmark
        specialized_dataset = generate_specialized_benchmark(
            tier_configs=selected_tiers,
            capabilities=capabilities,
            sample_count=args.samples
        )
        
        # Save dataset
        with open(args.output, 'w') as f:
            json.dump(specialized_dataset, f, indent=2)
        logger.info(f"Specialized benchmark saved to {args.output}")
        
        # Export ML formats if requested
        if args.ml_export:
            output_dir = Path(args.output).stem + "_ml"
            export_dataset_to_ml_format(specialized_dataset, output_dir)
    
    else:
        print("""
Advanced Digit-Class Prime Product Analysis Framework

Available commands:
  --demo            Run a basic demo of the framework
  --benchmark       Generate a complete benchmark dataset
  --specialized     Generate specialized benchmark for specific capabilities
  
Advanced options:
  --limit           Prime generation limit (default: 1000000)
  --samples         Samples per tier (default: 100)
  --output          Output filename (default: benchmark_dataset.json)
  --quantum         Include quantum-resistant challenges
  --ml-export       Export dataset in ML-friendly formats
  --tiers           Specific tiers to generate (comma-separated)
  
Examples:
  python digit_class_analysis.py --demo
  python digit_class_analysis.py --benchmark --limit 10000000 --quantum --ml-export
  python digit_class_analysis.py --specialized "mathematical_insight,factorization_algorithms" --tiers "5,6,7,13"
        """)