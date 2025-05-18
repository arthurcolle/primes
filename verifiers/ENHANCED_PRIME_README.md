# Enhanced Prime Verification with Reinforcement Learning

This document describes the enhanced version of the prime number verification environment, which adds several advanced learning techniques to improve model training.

## Key Enhancements

The enhanced prime verification system builds on the base implementation with these key features:

1. **Adaptive Difficulty**: Automatically adjusts task difficulty based on model performance
2. **Curriculum Learning**: Implements progressive difficulty increases through a curriculum
3. **Scaffolded Learning**: Provides step-by-step guidance that gradually decreases as the model improves
4. **Varied Challenge Types**: Expands beyond basic primality tests to include multiple challenge types
5. **Hint System**: Can provide mathematical hints with configurable probability
6. **Quantum-Resistant Challenges**: Special challenges designed to be resistant to quantum factorization

## Challenge Types

The enhanced environment supports multiple types of mathematical challenges:

1. **Primality**: Determining whether a number is prime ("Is 17 a prime number?")
2. **Factorization**: Finding the prime factorization of a number ("Find the prime factorization of 60")
3. **Verification**: Validating a given factorization ("Verify if these factors of 60 are correct: 2,2,3,5")
4. **Properties**: Finding properties of the factors ("Find the sum and product of the prime factors of 60")
5. **Pattern**: Identifying patterns in factorizations ("Identify patterns in the prime factorization of 60")
6. **Quantum-Resistant**: Special challenges with multiple large prime factors

## Usage

To use the enhanced environment:

```bash
# Launch the vLLM inference server
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.9 \
  --enable_prefix_caching True

# Train with enhanced features
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --num-processes 4 \
  --config-file configs/zero3.yaml \
  verifiers/examples/prime_train_enhanced.py \
  --data_path "/Users/agent/primes/data/primes_10000000.parquet" \
  --difficulty "progressive" \
  --challenge_modes "primality,factorization,verification,properties" \
  --curriculum_learning \
  --scaffolding \
  --hint_probability 0.3 \
  --output_dir "./outputs/enhanced_prime_model"
```

## Enhanced Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--difficulty` | Difficulty level (easy, medium, hard, extreme, mixed, progressive) | mixed |
| `--challenge_modes` | Comma-separated list of challenge types | primality,factorization,verification |
| `--adaptive_difficulty` | Enable adaptive difficulty adjustment | False |
| `--scaffolding` | Enable step-by-step learning scaffolding | False |
| `--curriculum_learning` | Enable curriculum learning | False |
| `--quantum_resistant` | Include quantum-resistant challenges | False |
| `--hint_probability` | Probability of providing hints (0-1) | 0.2 |

## Curriculum Learning

When curriculum learning is enabled, the model progresses through increasingly difficult stages:

1. **Level 0 (Easy)**: Basic primality and factorization with small numbers
2. **Level 1 (Medium)**: More complex factorization with medium-sized numbers
3. **Level 2 (Hard)**: Advanced challenges with larger numbers and more factors
4. **Level 3 (Extreme)**: Research-level challenges including quantum-resistant problems

The model advances to the next level when it achieves a recent average performance above 80%.

## Adaptive Difficulty

With adaptive difficulty enabled, the environment:

1. Monitors model performance over recent episodes
2. Increases difficulty when performance exceeds 90%:
   - Adds more challenging problem types
   - Reduces hint probability
   - Introduces more complex numbers
3. Decreases difficulty when performance falls below 30%:
   - Simplifies to basic problem types
   - Increases hint probability
   - Uses smaller, simpler numbers

## Scaffolded Learning

Scaffolding provides a structured approach to mathematical reasoning:

1. Detailed system prompt with step-by-step guidance for solving problems
2. Explicit hints for approaching different types of prime challenges
3. Incremental hint system that can provide assistance when needed
4. Progressive reduction of assistance as the model improves

## Example Challenges

```
# Primality challenge
Is 8191 a prime number?

# Factorization challenge
Find the prime factorization of 60.

# Verification challenge
Verify if these factors of 60 are correct: 2,2,3,5

# Properties challenge
Find the sum and product of the prime factors of 60.

# Pattern challenge
Identify patterns in the prime factorization of 2310.

# Quantum-resistant challenge with hint
Find the prime factorization of 1763903164323213. This is a quantum-resistant challenge with multiple prime factors.

Hint: The number is divisible by 41.
```

## Implementation Details

The enhanced environment is implemented in:

- `verifiers/envs/enhanced_prime_env.py`: Main environment implementation
- `verifiers/examples/prime_train_enhanced.py`: Training script with enhanced features

The code builds on the base prime verification system but adds significant new capabilities for advanced training techniques.

## Performance Tips

1. Start with curriculum learning and scaffolding for new models
2. Use a hint probability of 0.3-0.5 initially, then reduce as training progresses
3. Enable adaptive difficulty for longer training runs to optimize learning efficiency
4. Combine multiple challenge types for more comprehensive mathematical reasoning
5. For pretrained mathematical models, start with higher difficulty levels

## References

- Curriculum Learning for Neural Networks (Bengio et al.)
- Scaffolded Learning in AI Education
- Adaptive Difficulty in Reinforcement Learning Environments
- Group Relative Policy Optimization (GRPO)
- Quantum-Resistant Cryptographic Challenge Design