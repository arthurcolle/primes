# Prime Number Toolkit

A comprehensive toolkit for prime factorization, algorithmic challenges, and AI reasoning assessment with precise complexity control.

## 📋 Overview

This project provides a mathematical framework and implementation for generating, testing, and analyzing prime factorization problems across multiple complexity dimensions. It's designed for:

- **Researchers**: Evaluate AI mathematical reasoning abilities
- **Educators**: Create curriculum with progressive complexity
- **Cryptography**: Generate quantum-resistant factorization challenges
- **Algorithm Development**: Benchmark factorization algorithms
- **Machine Learning**: Train and evaluate models on mathematical reasoning

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/primes.git
cd primes

# Install dependencies
pip install -r requirements.txt

# Generate a basic benchmark dataset
python digit_class_analysis.py --benchmark --limit 1000000 --output benchmark.json

# Visualize factorization
python factorization_visualizer.py --number 30030
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- NumPy, SciPy, SymPy
- pandas, pyarrow (for data processing)
- Matplotlib, NetworkX, Plotly (for visualizations)
- DSPy and Arbor (for AI reasoning components)

### Standard Setup

```bash
# Base dependencies
pip install -U numpy scipy sympy pandas pyarrow matplotlib networkx plotly seaborn

# Optional performance enhancers
pip install numba torch  # For GPU acceleration and neural guidance
```

### AI Reasoning Components (Optional)

```bash
# Install DSPy with GRPO support
pip install -U dspy arbor-ai "jax[cpu]" 
pip install git+https://github.com/stanfordnlp/dspy.git@refs/pull/8171/head

# Configure Arbor (for distributed training)
python -m arbor.cli serve --arbor-config arbor.yaml
```

## 🔍 Core Concept: The S(N,K) Framework

The S(N,K) digit-class prime product framework provides a rigorous mathematical foundation for creating factorization problems with controlled difficulty.

### Mathematical Definition

Let $\mathbb{P}$ be the set of all primes and define a size metric $\kappa: \mathbb{P} \rightarrow \mathbb{N}$ which maps each prime to its digit length. For parameters $N \in \mathbb{N}$ and $K \subseteq \mathbb{N}$, we define:

$$S(N,K) = \left\{ n \in \mathbb{N} \mid n = \prod_{j=1}^{m} p_j, 1 \leq m \leq N, p_j \in \mathbb{P}, \kappa(p_j) \in K \right\}$$

And for each $n \in S(N,K)$, we define its signature (or $\omega$-vector):

$$\omega_K(n) = (c_d)_{d \in K}, \text{ where } c_d = |\{j \mid \kappa(p_j) = d\}|$$

This enables creation of problems with precise complexity characteristics across multiple dimensions:
- **Computational complexity**: Controlled by bit-length
- **Working memory requirements**: Controlled by factor count (N)
- **Search space magnitude**: Controlled by allowed digit classes (K)
- **Distribution variance**: Controlled by signature entropy

### Asymptotic Properties

Using the prime number theorem with exponential error term, for $\text{Re}(s) > 1$:

$$P_d(s) = \frac{10^{-d(s-1)}}{(s-1)\ln 10} + O\left(\frac{10^{-d\sigma}}{\sigma^2}\right), \quad \sigma = \text{Re}(s)-1$$

The asymptotic density follows:

$$|S(N,K) \cap [1,X]| \approx \frac{\left(\sum_{d \in K} d^{-1}\right)^N}{N!} \cdot (\log X)^N$$

## 🧩 Core Components

### Prime Generation and Analysis

```python
from digit_class_analysis import generate_primes, classify_primes_by_digits

# Generate prime numbers up to a limit
primes = generate_primes(1000000)
print(f"Generated {len(primes)} primes")

# Classify primes by digit length
classified = classify_primes_by_digits(primes)
for digits, primes_list in classified.items():
    print(f"{digits}-digit primes: {len(primes_list)}")
```

### Factorization Challenge Creation

```python
from digit_class_analysis import generate_S_N_K_samples

# Create challenges with specific complexity
samples = generate_S_N_K_samples(
    N=3,                # Maximum 3 prime factors
    K=[1, 2],           # Using 1 and 2-digit primes
    classified_primes=classified,
    sample_count=100,
    max_value=10**9     # Up to 1 billion
)

# Display challenges
for sample in samples[:5]:
    print(f"Number: {sample['n']}")
    print(f"Factors: {' × '.join(map(str, sample['factors']))}")
    print(f"Signature: {sample['signature']}")
    print()
```

### Advanced Factorization Algorithms

```python
from novel_factorization_algorithms import AdvancedFactorizationEngine

# Create a factorization engine with multiple algorithms
engine = AdvancedFactorizationEngine()

# Compare algorithm performance on a challenge
number = 104729873
results = engine.benchmark_algorithms(number)
for algo, result in results.items():
    print(f"{algo}: {result['time']:.6f}s - Factors: {result['factors']}")

# Use concurrent execution for fastest result
factors = engine.factorize_concurrent(number)
print(f"Factorization: {' × '.join(map(str, factors))}")
```

### Visualization Tools

```python
from factorization_visualizer import DecompositionTreeVisualizer

# Create an interactive visualization 
visualizer = DecompositionTreeVisualizer(dark_mode=True)
visualizer.visualize_decomposition_tree(30030, show_steps=True)
```

### Hierarchical Reasoning System

```python
from hierarchical_reasoning_system import HierarchicalReasoningSystem

# Create a multi-expert reasoning system
system = HierarchicalReasoningSystem()

# Generate detailed reasoning trace
result = system.factorize_with_trace(104729873)
print(f"Factors: {result['factors']}")
print(f"Reasoning trace:\n{result['trace']}")
```

## 📊 Benchmark Structure

The benchmark includes diverse tiers of increasing difficulty:

| Tier | Configuration | Description | Example |
|------|--------------|-------------|---------|
| 0-1  | S(2, [1]) | Single-digit prime factors | 15 = 3 × 5 |
| 2-4  | S(3, [1,2]) | Small prime factors | 1001 = 7 × 11 × 13 |
| 5-7  | S(4, [2,3]) | Medium complexity | 100447 = 17 × 31 × 191 |
| 8-9  | S(5, [3,4]) | Advanced factorization | 98716243 = 991 × 1993 × 4999 |
| 10-12 | S(2, [30, 40]) | Cryptographic-strength | 2048-4096 bit RSA-like numbers |
| 13-16 | S(2, [100+]) | Extreme challenges | Massive prime products (10^200+) |

### Tier Calibration Framework

The intrinsic difficulty of factoring an element of $S(N,K)$ can be approximated by:

$$D(n) \approx \alpha \cdot \text{bits}(n) + \beta \cdot H(\omega_K(n)) + \gamma \cdot \text{Var}(\kappa(p_j))$$

Where:
- $\text{bits}(n)$ is the bit-length
- $H(\omega_K(n))$ is the signature entropy
- $\text{Var}(\kappa(p_j))$ is the variance in prime factor sizes
- $\alpha, \beta, \gamma$ are empirically calibrated weights

### Generating Benchmark Datasets

```bash
# Generate standard benchmark
python digit_class_analysis.py --benchmark --limit 1000000 --output benchmark.json --ml-export

# Generate quantum-resistant challenges
python gen_massive_primes.py --samples_per_tier 1000 --workers 8 --output quantum_benchmark.json
```

## 🤖 AI Integration

### DSPy GRPO for Prime Factorization

Train language models to perform factorization reasoning:

```bash
# Start Arbor server (requires GPUs)
python -m arbor.cli serve --arbor-config arbor.yaml

# Run training
python prime_dspy_grpo.py --model "Qwen/Qwen2.5-7B-Instruct" --train_steps 100 --hops 3 
```

### Multi-Hop Reasoning Process

The factorization approach uses these steps:

1. Generate initial approach for breaking down the number
2. Iteratively refine the approach based on discoveries
3. Verify results to ensure all factors have been found
4. Validate that the product of factors equals the original number

```python
from prime_dspy_grpo import PrimeFactorizationHop
import dspy

# Load your optimized model
optimized_program = PrimeFactorizationHop(num_hops=3)
optimized_program.load("path_to_saved_model")

# Factorize a number
result = optimized_program(prompt="Find the prime factorization of 2023.")
print(f"Factors: {result.factors}")
print(f"Reasoning: {result.approach_history}")
print(f"Verification: {result.verification}")
```

### Verifiers Framework Integration

The project integrates with the Verifiers framework for reinforcement learning with LLMs:

```bash
# Run training with the PrimeEnv
cd verifiers
python verifiers/examples/prime_train.py --model "Qwen/Qwen2.5-7B-Instruct" --data_path "../data/primes_1000000.parquet"
```

Environment setup example:

```python
from verifiers.envs.prime_env import PrimeEnv
from verifiers.tools.prime_tools import is_prime, factorize, verify_factorization

# Create a prime factorization environment
env = PrimeEnv(
    data_path="./data/primes_1000000.parquet",
    difficulty_level="medium",  # "easy", "medium", "hard", "mixed"
    max_steps=10
)

# The environment provides tasks like:
# - "Is 589 a prime number?"
# - "Find the prime factorization of 589."
# - "Verify if these factors of 589 are correct: 19,31"
```

### Metrics and Evaluation

The system evaluates multiple reasoning dimensions through a structured Chain-of-Thought protocol:

```json
{
  "prompt": "Factor 589 and explain your approach.",
  "exemplar_response": {
    "final_answer": [19, 31],
    "reasoning_trace": [
      "I need to find the prime factors of 589.",
      "First, I check if it's divisible by small primes: 2, 3, 5, 7, 11, 13, 17, 19...",
      "589 ÷ 19 = 31 with remainder 0.",
      "So 589 = 19 × 31. Let me verify if 31 is prime...",
      "31 is prime since it's not divisible by any smaller primes.",
      "Therefore, 589 = 19 × 31 is the complete prime factorization."
    ],
    "verification": "19 × 31 = 589 ✓"
  }
}
```

Each response is scored along multiple dimensions:
- **Correctness**: Is the factorization correct? (0-1)
- **Completeness**: Are all prime factors identified? (0-1) 
- **Efficiency**: How direct is the solution path? (0-1)
- **Logical Soundness**: Are all inference steps mathematically valid? (0-1)
- **Explanation Quality**: How clearly is the process articulated? (0-1)

## 🔬 Advanced Mathematical Framework

### Algebraic Structure

S(N,K) can be viewed within the free abelian monoid $M = \bigoplus_{p \in \mathbb{P}} \mathbb{N} \cdot e_p$ where:

- Each element represents a unique factorization via exponent vector $e = (e_p)_{p \in \mathbb{P}}$
- The $\kappa$-grading induces a decomposition $M = \bigoplus_{d \in \mathbb{N}} M_d$ where $M_d = \bigoplus_{p:\kappa(p)=d} \mathbb{N} \cdot e_p$
- $S(N,K)$ corresponds to the order-ideal $I_{N,K} = \{e \in M \mid |e| := \sum e_p \leq N, \text{supp}(e) \subseteq \bigcup_{d \in K} \mathbb{P}_d\}$

The lattice structure (under meet=gcd, join=lcm) provides rich combinatorial properties.

### Category-Theoretic Foundations

The S(N,K) framework connects to advanced category theory through the prime factorization category $\mathcal{F}$ where:
- Objects are natural numbers $n \in \mathbb{N}$
- Morphisms $f: m \to n$ exist when $m | n$
- Composition is given by divisibility transitivity
- Monoidal product is multiplication: $m \otimes n = m \cdot n$
- Monoidal unit is $1$

This forms a symmetric monoidal category that can be extended to topos-theoretic structures, ∞-categories, and categorical dynamics.

### Hilbert Series and Generating Functions

The $\kappa$-Hilbert series captures the distribution of elements by signature:

$$H_{N,K}(z) = \sum_{e \in I_{N,K}} z^{\text{deg}_\kappa(e)} = \sum_{\omega} |\Delta(\omega)| \cdot z^{\langle\omega,d\rangle}$$

where $\langle\omega,d\rangle = \sum_{d \in K} c_d \cdot d$.

### Information-Theoretic Aspects

For a given signature $\omega = (c_d)_{d \in K}$, the entropy is:

$$H(\omega) = -\sum_{d \in K} \frac{c_d}{|\omega|} \log_2 \frac{c_d}{|\omega|}$$

This quantifies the diversity of prime factor sizes, with balanced signatures having maximum entropy.

The size of the factorization search space for a product with signature $(c_d)_{d \in K}$ is approximately:

$$\prod_{d \in K} \binom{|\mathbb{P}_d|}{c_d} \approx \prod_{d \in K} \frac{(9 \cdot 10^{d-2})^{c_d}}{c_d!}$$

### Quantum Algorithms

The framework extends to quantum computational approaches through:

1. **Enhanced Shor's Algorithm with Digit-Class Constraints**

   Shor's algorithm can be modified to exploit S(N,K) structure through constraint-based amplitude amplification to enhance states corresponding to factors in $\bigcup_{d \in K} \mathbb{P}_d$.

2. **Quantum Walks for Signature Detection**

   Construct quantum walks on factor graphs:
   
   $$U = S \cdot (2|\psi_0\rangle\langle\psi_0| - I)$$
   
   Where $S$ is the flip-flop shift operator on a graph of partial factorizations.

3. **Quantum-Resistant Extensions**

   Define quantum-resistant factorization problems like:
   
   $$S_{\text{lat}}(N,K) = \{n \in S(N,K) | \exists \text{ lattice } L \text{ s.t. factors of } n \text{ determine shortest vectors in } L\}$$

## 🔍 Intermediate Verification System

The intermediate_verification.py module implements real-time verification:

```python
from intermediate_verification import IntermediateVerificationSystem

# Create verification system
verifier = IntermediateVerificationSystem()

# Verify a factorization step
step = {
    "step_id": 1,
    "type": "divisibility_check",
    "divisibility_claims": [
        {"dividend": 210, "divisor": 2, "result": True},
        {"dividend": 210, "divisor": 3, "result": True}
    ]
}

result = verifier.verify_step(step, context={"number": 210})
print(f"Step verification: {result['overall_status']}")

# Verify a complete solution
solution = {"factors": [2, 3, 5, 7]}
verification_result = verifier.verify_solution(solution, 210)
print(f"Solution verification: {verification_result['status']}")
```

The system can verify multiple types of intermediate steps:
- Divisibility claims
- Primality assessments
- Partial factorizations
- Algorithm selection decisions
- Reasoning heuristics

## 📈 Project Structure

```
primes/
├── digit_class_analysis.py     # Core S(N,K) implementation
├── novel_factorization_algorithms.py  # Advanced algorithms
├── hierarchical_reasoning_system.py   # Multi-expert system
├── factorization_visualizer.py        # Visualization tools
├── intermediate_verification.py       # Step-by-step verification
├── gen_massive_primes.py              # Quantum-resistant challenges
├── prime_dspy_grpo.py                 # DSPy integration
├── data/                              # Generated datasets
│   ├── primes_1000000.parquet         # Prime number database
│   └── primes_10000000.parquet        # Larger prime database
├── verifiers/                         # RL verification framework
│   ├── verifiers/                     # Core verification code
│   │   ├── envs/                      # Environment definitions
│   │   ├── tools/                     # Mathematical tools
│   │   └── examples/                  # Training examples
├── quantum_benchmark_ml/              # ML-ready benchmarks
└── visualizations/                    # Generated visualizations
```

## 📚 Detailed Mathematical Documentation

The repository includes extensive documentation:

- [Advanced README](README_advanced.md): Details on advanced algorithms and features
- [GRPO Integration](README_GRPO.md): Guide to using DSPy GRPO with prime factorization
- [Analytic Framework](analytic_framework.md): Mathematical foundations of S(N,K)
- [AI Reasoning Evaluation](ai_reasoning_evaluation.md): Framework for AI benchmarking
- [Novel Extensions](novel_extensions.md): Advanced extensions to the framework
- [Quantum Algorithms](quantum_algorithms.md): Quantum approaches to factorization
- [Category Theory](category_theory.md): Category-theoretic foundations
- [Extreme Benchmark Guide](extreme_benchmark_readme.md): Guide to extreme challenges

### Analytic Framework Highlights

The analytic_framework.md file provides deep mathematical analysis, including:

- Multigraded monoid perspective of S(N,K)
- Lattice structure and combinatorial properties
- Digit-colored zeta functions
- Probabilistic aspects including digit-conditioned Erdős-Kac phenomenon
- Algorithmic complexity analysis
- Cryptographic connections

### Category Theory Highlights

The category_theory.md file explores advanced mathematical structures:

- Monoidal categories of factorizations
- Topos-theoretic structure
- Higher categorical structures
- Categorical logic and type theory
- Enriched category theory
- 2-categorical structure
- Operadic and multicategory perspectives
- Categorical dynamics
- Grothendieck fibrations
- Sheaf-theoretic interpretation

### Quantum Algorithms Highlights

The quantum_algorithms.md file details quantum computing approaches:

- Quantum complexity classification
- Enhanced Shor's algorithm with digit-class constraints
- Quantum walks for signature detection
- Quantum tensor networks
- Quantum annealing approach
- Topological quantum computing
- Quantum error correction
- Variational quantum factorization
- Quantum machine learning
- Quantum random walk factorization
- Quantum rejection sampling
- Quantum-resistant factorization

## 💡 Research and Development

The prime factorization toolkit opens up research possibilities in:

- **Cryptography**: Development of quantum-resistant schemes
- **AI Training**: Progressive curriculum for mathematical reasoning
- **Number Theory**: Exploration of prime distribution patterns
- **Educational Technology**: Complexity-controlled learning materials
- **Quantum Computing**: Testing factorization approaches on quantum simulators

### Novel Extensions

Alternative $\kappa$ metrics include:
- Bit-length: $\kappa_{\text{bit}}(p) = \lceil \log_2 p \rceil$
- Prime index: $\kappa_{\pi}(p) = \pi(p)$ (position in sequence of primes)
- Log-log scale: $\kappa_{\text{ll}}(p) = \lceil \log_{10} \log_{10} p \rceil$

The framework extends to:
- Polynomial rings: $\mathbb{F}_q[x]$ with irreducibles as "primes"
- Number fields: $\mathcal{O}_K$ with prime ideals
- Function fields with divisor theory

### Open Research Questions

1. Exact formula for counting $|S(N,K) \cap [1,X]|$ via Ehrhart theory
2. Threshold phenomena in $N$ where enumeration transitions from sparse to dense
3. Correlation between theoretical hardness measures and empirical AI performance
4. Extension to quantum-resistant cryptographic primitives
5. Optimal curriculum design for maximizing learning rate in AI systems

## 🔧 Development and Contributing

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Implement your extension or enhancement
4. Add tests for your feature
5. Submit a pull request with detailed description

## 📄 License

MIT License - See LICENSE file for details

## 📋 References

- Erdős–Kac theorem and prime factor distributions
- Analytic number theory and zeta functions
- Lattice point enumeration techniques
- Multigraded monoids and factorization theory
- AI reasoning evaluation frameworks
- GNFS (General Number Field Sieve)
- GRPO (Group Relative Policy Optimization)
- Quantum computing and Shor's algorithm

---

For questions, discussions, and contributions, please open an issue or submit a pull request.