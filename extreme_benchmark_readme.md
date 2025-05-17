# Extreme Prime Factorization Benchmark

This repository contains an extremely challenging prime factorization benchmark designed to be future-proof against advances in computational capability and AI reasoning.

## Key Features

- **Massive Prime Numbers**: Includes factorization challenges with numbers up to 10^12+ range
- **Cryptographic-Strength Problems**: Incorporates 2048, 3072, and 4096-bit challenges (industry standard cryptographic key sizes)
- **Statistical Significance**: 1000+ samples per difficulty tier for rigorous evaluation
- **Specialized Resistance Tiers**: Includes factorization-resistant number configurations
- **Future-Proof Design**: Problems intentionally beyond current computational limits

## Benchmark Structure

The benchmark consists of 17 tiers of increasing difficulty:

### Basic Tiers (0-4)
- Tier 0: Basic factorization of small semiprimes
- Tier 1: Mixed small semiprimes
- Tier 2: Multiple medium factors
- Tier 3: Mixed medium factorization
- Tier 4: Complex mixed factorization

### Advanced Tiers (5-9)
- Tier 5: RSA-like semiprimes
- Tier 6: Mixed with one large prime
- Tier 7: Adversarial large semiprimes
- Tier 8: Multiple large prime factors
- Tier 9: Many medium-sized factors

### Cryptographic Tiers (10-12)
- Tier 10: Quantum-resistant multi-factor (2048-bit)
- Tier 11: Quantum-resistant multi-factor (3072-bit)
- Tier 12: Quantum-resistant multi-factor (4096-bit)

### Extreme Tiers (13-16)
- Tier 13: Massive prime factorization (1024-bit)
- Tier 14: Massive prime factorization (2048-bit)
- Tier 15: Near-prime challenge - extremely close primes
- Tier 16: Near-prime challenge - extremely close large primes

## Difficulty Categories

- **elementary**: Basic factorization challenges
- **intermediate**: Medium complexity challenges
- **advanced**: Challenging factorization problems
- **cryptographic**: Industry-standard cryptographic strength
- **extreme**: Beyond current computational limits, future-proof challenges

## Dataset Format

Each challenge includes:
- Unique identifier
- The number to factorize
- Ground truth factors
- Difficulty metrics
- Bit length information
- Category classification
- Textual prompts for evaluating AI responses

## Intended Use

This benchmark is designed to:

1. Evaluate the mathematical reasoning capabilities of advanced AI models
2. Provide a future-proof benchmark that will remain challenging for years to come
3. Test factorization capabilities across multiple orders of magnitude
4. Serve as a reference dataset for cryptographic security evaluations

## Generation

The benchmark is generated using the `gen_massive_primes.py` script, which:
- Employs parallel processing for efficiency
- Utilizes sympy for prime number generation
- Creates carefully calibrated difficulty tiers
- Structures data for machine learning evaluation

## Machine Learning Format

The benchmark is provided in both JSON and Hugging Face formats:
- Complete benchmark in `quantum_benchmark_extreme.json`
- Train/test/validation splits in `quantum_benchmark_ml/huggingface_extreme/`

## Usage

To generate the benchmark:

```bash
python gen_massive_primes.py --samples_per_tier 1000 --workers 8
```

To use a smaller sample size for testing:

```bash
python gen_massive_primes.py --samples_per_tier 10 --workers 4
```