---
language:
- en
license: mit
---

# Prime Factorization Benchmark Dataset

This dataset contains prime factorization challenges of varying difficulty levels designed to evaluate mathematical reasoning capabilities of AI models.

## Dataset Structure

- Number of samples: 260
- Number of tiers: 13
- Framework version: 0.2.0
- Generation date: 2025-05-17T18:21:15.026763

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
