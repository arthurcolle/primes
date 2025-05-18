# Digit-Class Prime Product Framework (Prime Factorization Evals for LMs) 

This repository contains analysis tools and benchmarking utilities for exploring the S(N,K) digit-class prime product problem and its applications to AI evaluation.

## Core Concept

The S(N,K) framework classifies integers based on their prime factorization patterns, where:

- **N**: Maximum number of prime factors
- **K**: Set of allowed digit-lengths for primes
- **S(N,K)**: Set of integers expressible as products of at most N primes, each with digit-length in K

This framework provides a controllable "difficulty dial" for creating reasoning problems with precise complexity characteristics.

## Repository Contents

- `gen_dataset.py`: Original script to generate prime factorization dataset
- `digit_class_analysis.py`: Implementation of S(N,K) framework with analysis tools
- `analytic_framework.md`: Detailed mathematical analysis of S(N,K) properties
- `ai_reasoning_evaluation.md`: Framework for using S(N,K) in AI benchmarking
- `novel_extensions.md`: Advanced extensions to the basic framework
- `data/`: Pre-generated datasets

## Analysis Tools

The `digit_class_analysis.py` module provides:

1. Prime generation and classification utilities
2. S(N,K) sample generation with parameterized difficulty
3. Analytic approximations of S(N,K) properties
4. Difficulty measurement and calibration
5. Benchmark prompt generation
6. Lattice-theoretic analysis functions

## Usage

Basic usage example:

```python
from digit_class_analysis import generate_primes, classify_primes_by_digits, generate_S_N_K_samples

# Generate prime base
primes = generate_primes(100000)
classified = classify_primes_by_digits(primes)

# Generate samples from S(3, [1, 2])
samples = generate_S_N_K_samples(
    N=3,               # Maximum 3 prime factors
    K=[1, 2],          # Using 1 and 2-digit primes
    classified_primes=classified,
    sample_count=100,  # Generate 100 samples
    max_value=10**9    # Up to 1 billion
)

# Analyze samples
for sample in samples[:5]:
    print(f"Number: {sample['n']}")
    print(f"Factors: {' × '.join(map(str, sample['factors']))}")
    print(f"Signature: {sample['signature']}")
    print()
```

## Benchmark Generation

To create a complete benchmark dataset:

```bash
python digit_class_analysis.py --benchmark --limit 1000000 --output benchmark.json
```

## Mathematical Framework

See `analytic_framework.md` for in-depth mathematical analysis of:

- Algebraic structure (multigraded monoid perspective)
- Analytic properties (colored zeta functions)
- Probabilistic aspects (digit-conditioned Erdős-Kac phenomenon)
- Complexity dimensions

## AI Evaluation Framework

See `ai_reasoning_evaluation.md` for detailed exploration of:

- Cognitive skills assessment dimensions
- Evaluation protocols
- Chain-of-thought analysis
- Progressive difficulty curriculum
- Implementation frameworks

## Advanced Extensions

See `novel_extensions.md` for cutting-edge extensions including:

- Multivariate classification metrics
- Dynamic constraint systems
- Topological perspectives
- Information-theoretic frameworks
- Quantum-inspired approaches
- Game-theoretic models

## Development

To contribute or extend this framework:

1. Fork the repository
2. Create a feature branch
3. Implement your extension
4. Submit a pull request with detailed description

## References

- Erdős–Kac theorem
- Analytic number theory
- Lattice point enumeration
- Multigraded monoids
- AI reasoning evaluation

## License

MIT License - See LICENSE file for details
