# Prime Number Factorization with DSPy GRPO

This project demonstrates how to use DSPy's Group Relative Policy Optimization (GRPO) to train language models for multi-hop reasoning on prime factorization tasks.

## Overview

The system uses multi-hop reasoning to tackle prime factorization problems of varying complexity. By applying GRPO, we optimize the language model's ability to:

1. Break down complex factorization problems into logical steps
2. Apply appropriate mathematical techniques at each step
3. Verify results for correctness

This approach combines the strengths of:
- Structured reasoning (DSPy modules)
- Reinforcement learning optimization with GRPO
- Mathematical knowledge (prime factorization)

## Setup Instructions

### Prerequisites

1. Install Arbor AI and required dependencies:

```bash
pip install -U dspy arbor-ai "jax[cpu]" pyarrow pandas sympy
```

2. Clone DSPy with GRPO support:

```bash
pip install git+https://github.com/stanfordnlp/dspy.git@refs/pull/8171/head
```

3. Start Arbor server (requires GPUs):

```bash
python -m arbor.cli serve --arbor-config arbor.yaml
```

### Running the Training

```bash
python prime_dspy_grpo.py --model "Qwen/Qwen2.5-7B-Instruct" --train_steps 100 --hops 3
```

Options:
- `--port`: Port for Arbor server (default: 7453)
- `--model`: Local model to use (default: "Qwen/Qwen2.5-7B-Instruct")
- `--train_steps`: Number of GRPO training steps (default: 500)
- `--hops`: Number of reasoning hops (default: 3)
- `--batch_size`: Per-device batch size (default: 2)
- `--data_path`: Path to prime data (default: "./data/primes_10000000.parquet")
- `--benchmark_path`: Path to benchmark (default: "./quantum_benchmark_extreme.json")

## Dataset

The training uses extreme prime factorization benchmarks with problems ranging from:
- Basic factorization challenges
- Medium complexity multi-factor problems
- Cryptographic-strength challenges (2048-4096 bit)
- Massive prime factorization problems (10^12+)

## Performance Metrics

The system is evaluated using three metrics:

1. **Factor Accuracy**: F1 score between predicted and actual prime factors
2. **Product Correctness**: Whether the product of predicted factors equals the original number
3. **Combined Score**: Weighted combination of factor accuracy and product correctness

## Multi-Hop Reasoning Process

The factorization approach uses these steps:

1. Generate initial approach for breaking down the number
2. Iteratively refine the approach based on discoveries
3. Verify results to ensure all factors have been found
4. Validate that the product of factors equals the original number

## Implementation Details

Key components:
- `PrimeFactorizationHop`: Multi-hop reasoning module
- `GRPO`: Group Relative Policy Optimization reinforcement learning framework
- Mathematical helper functions for validation
- Metrics for evaluating factorization correctness

## Example Usage

After training, you can use the optimized model to factorize numbers:

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