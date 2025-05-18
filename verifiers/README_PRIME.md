# Prime Number Verification with Reinforcement Learning

This extension to the verifiers library implements a reinforcement learning system for prime number verification and factorization tasks. It uses Group Relative Policy Optimization (GRPO) to train language models to improve their mathematical reasoning abilities.

## Overview

The Prime Verification environment focuses on training LLMs to perform three key tasks:

1. **Primality Testing**: Determining whether a number is prime or not
2. **Prime Factorization**: Finding the prime factorization of a composite number 
3. **Verification**: Validating whether a given factorization is correct

These tasks provide a controllable mathematical environment with clear success criteria, making them ideal for reinforcement learning.

## Installation

First, ensure you have the base verifiers repository installed:

```bash
git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
source .venv/bin/activate
```

The prime verification components are already integrated and ready to use.

## Components

### Prime Tools

The system provides several mathematical tools for the LLM to use:

- `is_prime`: Check if a number is prime
- `factorize`: Find the prime factorization of a number
- `next_prime`: Find the next prime number after a given number
- `prime_count`: Count the number of primes up to a given limit
- `verify_factorization`: Verify if a list of factors is correct

File: `verifiers/tools/prime_tools.py`

### Prime Environment

The custom environment for prime verification tasks:

- Provides a structured tool-based environment for prime verification tasks
- Supports different difficulty levels (easy, medium, hard, mixed)
- Can load datasets from parquet files or generate synthetic examples
- Implements proper reward functions for evaluating the model's performance

File: `verifiers/envs/prime_env.py`

### Reward Functions

Special rubrics for evaluating prime verification tasks:

- `factorization_accuracy`: Measures correctness of prime factorization or primality checks
- `reasoning_quality`: Evaluates the quality of mathematical reasoning
- `tool_usage_efficiency`: Measures how effectively the model uses available tools

File: `verifiers/rubrics/prime_rubric.py`

## Usage

### Training

To train a model on prime verification tasks:

```bash
# Start the vLLM inference server
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.9 \
  --enable_prefix_caching True

# Run the training script
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --num-processes 4 \
  --config-file configs/zero3.yaml \
  verifiers/examples/prime_train.py \
  --data_path "/Users/agent/primes/data/primes_10000000.parquet" \
  --difficulty "mixed" \
  --output_dir "./outputs/prime_model" \
  --batch_size 8
```

Key parameters:
- `--data_path`: Path to parquet file with prime numbers data
- `--difficulty`: Difficulty level (easy, medium, hard, mixed)
- `--output_dir`: Directory to save the trained model
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--max_steps`: Maximum steps in environment per example
- `--debug`: Enable debug mode with fewer samples

### Evaluation

To evaluate a model on prime verification tasks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py \
  --model "./outputs/prime_model" \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.9

# Run evaluation script
python verifiers/examples/prime_eval.py \
  --model "./outputs/prime_model" \
  --data_path "/Users/agent/primes/data/primes_test.parquet" \
  --benchmark_path "/Users/agent/primes/quantum_benchmark_test.json" \
  --output_file "prime_eval_results.json" \
  --compare_models "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-2-7b-chat-hf"
```

Key parameters:
- `--model`: Path to the model to evaluate
- `--data_path`: Path to parquet file with test data
- `--benchmark_path`: Optional path to benchmark JSON file
- `--output_file`: File to save evaluation results
- `--compare_models`: Additional models to compare against
- `--detailed_output`: Include detailed per-sample results
- `--test_tier`: Test only a specific difficulty tier

## Working with Benchmark Data

The prime verification system can use the specialized benchmark datasets provided in the primes repository:

- `primes_1000000.parquet`: Basic prime factorization dataset with numbers up to 10^6
- `primes_10000000.parquet`: Extended dataset with numbers up to 10^7
- `quantum_benchmark_test.json`: Specialized benchmark with difficulty tiers

The benchmark tiers range from simple to complex:
- Tiers 0-2: Basic factorization tasks with small numbers
- Tiers 3-5: Intermediate tasks with medium-sized numbers
- Tiers 6-8: Advanced tasks with larger numbers and more factors
- Tiers 9-12: Research-level tasks with very large semiprimes
- Tiers 13-15: Quantum-resistant challenges

## Creating Custom Datasets

You can create custom prime verification datasets using the tools in the primes repository:

```python
from digit_class_analysis import generate_primes, classify_primes_by_digits, generate_S_N_K_samples, save_benchmark_dataset

# Generate a benchmark dataset with specified parameters
save_benchmark_dataset(
    filename="my_benchmark.json",
    limit=1000000,           # Prime generation limit
    samples_per_tier=100,    # Samples per difficulty tier
    include_quantum=True,    # Include quantum-resistant challenges
    export_ml=True           # Export in ML-friendly formats
)
```

## Extending the System

The prime verification system can be extended in several ways:

1. Add new mathematical tools in `prime_tools.py`
2. Create more complex factorization challenges
3. Implement specialized training regimes for different mathematical skills
4. Add new evaluation metrics in `prime_rubric.py`

## Performance Tips

- For best training results, use models of at least 7 billion parameters
- Use 4-8 GPUs for efficient training
- Start with easier difficulty levels and gradually increase
- Evaluate on multiple difficulty tiers to measure progress comprehensively

## References

- Verifiers: Reinforcement Learning with LLMs in Verifiable Environments
- Digit-Class Prime Product Framework
- Group Relative Policy Optimization (GRPO)