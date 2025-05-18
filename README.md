# Prime Factorization Framework for AI Evaluation

A comprehensive toolkit for generating, analyzing, and benchmarking prime factorization challenges with precise complexity control, designed for evaluating mathematical reasoning in AI systems.

## 🔍 Core Concept: The S(N,K) Framework

The S(N,K) digit-class prime product framework provides a rigorous mathematical foundation for creating factorization problems with controlled difficulty:

- **N**: Maximum number of prime factors allowed
- **K**: Set of allowed digit-lengths for prime factors
- **S(N,K)**: Set of integers expressible as products of at most N primes, each with digit-length in K

This enables creation of problems with precise complexity characteristics across multiple dimensions:
- Computational complexity (bit-length)
- Working memory requirements (factor count)
- Search space magnitude (digit classes)
- Distribution variance (signature entropy)

## 🛠️ Key Components

- **Advanced Prime Generation**: Parallel, segmented sieving with caching for primes up to 10^12+
- **S(N,K) Sample Generation**: Generate integers with specified factorization properties
- **Quantum-Resistant Challenges**: Multi-factor challenges resistant to Shor's algorithm
- **Tiered Benchmarks**: 17 difficulty tiers from elementary to cryptographic-strength
- **Comprehensive Evaluation**: Metrics spanning basic arithmetic to advanced factorization algorithms
- **Machine Learning Integration**: Export in multiple ML-friendly formats (JSONL, Parquet, HuggingFace)

## 🚀 Usage Examples

### Basic Usage

```python
from digit_class_analysis import generate_primes, classify_primes_by_digits, generate_S_N_K_samples

# Generate and classify primes
primes = generate_primes(100000)
classified = classify_primes_by_digits(primes)

# Generate samples from S(3, [1, 2])
samples = generate_S_N_K_samples(
    N=3,                # Maximum 3 prime factors
    K=[1, 2],           # Using 1 and 2-digit primes
    classified_primes=classified,
    sample_count=100,   # Generate 100 samples
    max_value=10**9     # Up to 1 billion
)

# Analyze samples
for sample in samples[:5]:
    print(f"Number: {sample['n']}")
    print(f"Factors: {' × '.join(map(str, sample['factors']))}")
    print(f"Signature: {sample['signature']}")
    print()
```

### Generate a Benchmark Dataset

```bash
python digit_class_analysis.py --benchmark --limit 1000000 --output benchmark.json --ml-export
```

### Generate Extreme Quantum-Resistant Challenge Set

```bash
python gen_massive_primes.py --samples_per_tier 1000 --workers 8
```

### Using DSPy GRPO for Prime Factorization

```bash
python prime_dspy_grpo.py --model "Qwen/Qwen2.5-7B-Instruct" --train_steps 100 --hops 3
```

## 📊 Benchmark Structure

The benchmark includes diverse tiers of increasing difficulty:

| Tier | Configuration | Description | Category |
|------|--------------|-------------|----------|
| 0-4  | Basic S(N,K) with small N,K | Elementary factorization | Elementary |
| 5-9  | Complex S(N,K) combinations | Medium-to-advanced factorization | Intermediate/Advanced |
| 10-12 | Quantum-resistant challenges | Cryptographic-strength factorization (2048-4096 bit) | Cryptographic |
| 13-16 | Massive primes & near-primes | Extreme factorization challenges | Extreme |

## 🧠 Mathematical Framework

The framework connects to multiple mathematical disciplines:

- **Analytic Number Theory**: Colored zeta functions, asymptotic density formulas
- **Algebraic Geometry**: S(N,K) as rational points on algebraic schemes 
- **Category Theory**: Monoidal categories of factorizations, topos-theoretic structures
- **Information Theory**: Signature entropy, factorization complexity
- **Quantum Computing**: Enhanced Shor's algorithm with digit-class constraints
- **Lattice Theory**: Signature vector spaces, distance metrics between factorizations

## 🤖 AI Evaluation Capabilities

Systematically evaluates multiple reasoning dimensions:

- **Decomposition Skills**: Breaking problems into manageable parts
- **Number-Theoretic Reasoning**: Understanding divisibility and number patterns
- **Metacognitive Awareness**: Strategy switching and progress monitoring
- **Tool-Use Sophistication**: Appropriate application of mathematical tools
- **Algorithmic Reasoning**: Selecting and applying efficient factorization approaches

## 📚 Resources

The repository includes extensive documentation:

- **analytic_framework.md**: Deep mathematical analysis of S(N,K) properties
- **ai_reasoning_evaluation.md**: Framework for AI benchmarking with S(N,K)
- **novel_extensions.md**: Advanced extensions to the basic framework
- **quantum_algorithms.md**: Quantum approaches to S(N,K) problems
- **category_theory.md**: Category-theoretic foundations of factorization
- **extreme_benchmark_readme.md**: Guide to extreme challenge benchmarks

## 🔬 Advanced Research Applications

- **Cryptography**: Quantum-resistant factorization challenges
- **AI Training**: Progressive difficulty curriculum for mathematical reasoning
- **Number Theory**: Exploring distributions of factorization patterns
- **Educational Technology**: Cognitive scaffolding with controlled complexity
- **Quantum Computing**: Testing factorization approaches on quantum simulators

## ⚙️ Development

To contribute:

1. Fork the repository
2. Create a feature branch
3. Implement your extension or enhancement
4. Submit a pull request with detailed description

## 📋 References

- Erdős–Kac theorem
- Analytic number theory
- Lattice point enumeration
- Multigraded monoids
- AI reasoning evaluation
- GNFS (General Number Field Sieve)
- GRPO (Group Relative Policy Optimization)

## 📄 License

MIT License - See LICENSE file for details