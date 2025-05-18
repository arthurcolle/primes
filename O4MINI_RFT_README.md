# o4-mini Reinforcement Fine-Tuning for Prime Factorization

This directory contains tools for fine-tuning OpenAI's o4-mini model with reinforcement learning for expert-level prime factorization.

## Overview

The reinforcement fine-tuning (RFT) pipeline trains the o4-mini model to:

1. Correctly factorize numbers into their prime components
2. Select appropriate algorithms based on number size
3. Provide detailed mathematical reasoning steps
4. Return results in consistent JSON format

## What is Reinforcement Fine-Tuning?

Reinforcement Fine-Tuning (RFT) is a training technique that optimizes models using a reward signal rather than fixed correct outputs. For o4-mini, this means:

- The model generates multiple candidate responses to each prompt
- Each response is scored by custom grading functions
- The model weights are updated to increase the probability of high-scoring outputs
- Over time, the model learns to generate responses that maximize the reward

Unlike supervised fine-tuning which attempts to match specific answers, RFT allows us to optimize for complex, multi-dimensional metrics like:
- Mathematical correctness
- Reasoning quality
- Algorithm selection
- Computational efficiency
- Explanation clarity

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key with access to o4-mini and RFT capabilities
- Required Python packages: `numpy`, `sympy`, `tqdm`, `requests`

### Environment Setup

1. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

2. **Install required packages**:
   ```bash
   pip install numpy sympy tqdm requests
   ```

3. **Clone repository** (if applicable):
   ```bash
   git clone https://your-repository-url.git
   cd primes
   ```

### Cost Considerations

Reinforcement fine-tuning o4-mini requires:
- API usage costs for data generation and evaluation
- Training compute costs (billed per hour)
- Inference costs when using the fine-tuned model

For cost-effective development:
- Start with small datasets (100-200 examples)
- Use fewer training epochs (2-3)
- Limit evaluation to a small test set

For production quality:
- Scale up to 500-1000 examples
- Use 3-5 training epochs
- Consider multiple training runs with different hyperparameters

### Available Scripts

1. **Data Generation**
   ```bash
   python generate_o4mini_rft_data.py --num_samples 500 --optimize
   ```

2. **Running the Full Pipeline**
   ```bash
   python run_o4mini_rft.py --num_samples 500 --optimize --model_suffix "my-factorization-expert"
   ```

3. **Using the Fine-Tuned Model**
   ```bash
   python use_factorization_model.py 12345 --model "ft:o4-mini-2025-04-16:your-org:my-factorization-expert:123abc"
   ```

## Pipeline Components

### 1. Data Generation

The `generate_o4mini_rft_data.py` script:
- Creates balanced datasets across difficulty tiers
- Produces examples with optimized distributions for o4-mini
- Generates training, validation, and test sets
- Adds enhanced prompts to a subset of examples

#### Data Tier Structure

| Tier | Difficulty   | Number Range       | Characteristics                    | % of Dataset |
|------|--------------|--------------------|------------------------------------|--------------|
| 0    | Very Easy    | 10-99              | Two-digit numbers, simple factors  | 5%           |
| 1    | Easy         | 100-999            | Three-digit numbers                | 10%          |
| 2    | Basic        | 1,000-9,999        | Four-digit numbers                 | 15%          |
| 3    | Basic+       | 10,000-99,999      | Five-digit numbers                 | 20%          |
| 4    | Intermediate | 100,000-999,999    | Six-digit numbers                  | 20%          |
| 5    | Challenging  | 1M-9.9M            | Complex factorization patterns     | 15%          |
| 6    | Hard         | 10M-99M            | Large numbers with multiple factors| 8%           |
| 7    | Very Hard    | 100M-999M          | Very large numbers                 | 5%           |
| 8    | Expert       | 1B+                | Numbers requiring advanced methods | 2%           |

### 2. RFT Configuration

The `prime_rft_model.py` module includes:
- JSON schema for structured factorization output
- Multi-component grader configuration
- Fine-tuning parameters optimized for o4-mini
- API interfaces for OpenAI's RFT endpoints

#### Structured Output Format

```json
{
  "factors": [2, 3, 5, 7],
  "algorithm": "WheelFactorization",
  "reasoning": [
    "First, I'll check if the number is divisible by small primes.",
    "210 is divisible by 2: 210 ÷ 2 = 105",
    "105 is divisible by 3: 105 ÷ 3 = 35",
    "35 is divisible by 5: 35 ÷ 5 = 7",
    "7 is a prime number.",
    "Therefore, the prime factorization of 210 is 2 × 3 × 5 × 7."
  ],
  "time_taken": 0.023,
  "confidence": 0.99
}
```

### 3. Training Pipeline

The `run_o4mini_rft.py` script:
- Runs the complete RFT pipeline
- Monitors training progress
- Evaluates the fine-tuned model
- Saves results and artifacts

#### Pipeline Workflow
1. **Data Generation**: Create balanced training data
2. **File Upload**: Upload files to OpenAI API
3. **Training Job**: Configure and start RFT job
4. **Monitoring**: Track training metrics and progress
5. **Evaluation**: Test fine-tuned model performance
6. **Results Storage**: Save metrics and examples

### 4. Evaluation

The evaluation metrics include:
- **Factorization correctness (60% weight)**: Verifies the mathematical accuracy of the factorization
- **Algorithm selection quality (20% weight)**: Assesses if the model chose an appropriate algorithm for the number size
- **Reasoning quality (15% weight)**: Evaluates the logical progression and clarity of reasoning steps
- **Efficiency (5% weight)**: Measures computational performance and time complexity awareness

#### Grading Components
1. **Python Graders**: Use code-based evaluation for objective metrics
2. **GPT-4o Grader**: Assesses reasoning quality and mathematical explanations
3. **Weighted Average**: Combines all metrics into a final score

## Advanced Usage

### Customizing Graders

The grader configuration uses multiple components:
- `correctness`: Verifies mathematical correctness of factorization
- `algorithm`: Evaluates appropriateness of algorithm selection
- `reasoning`: Uses GPT-4o to assess reasoning quality
- `efficiency`: Measures computational efficiency

To modify these components, edit the `_load_grader_config` method in `prime_rft_model.py`.

#### Custom Grader Example

Here's how to modify the reasoning quality grader to emphasize different aspects:

```python
"reasoning": {
    "name": "Reasoning Quality",
    "type": "score_model",
    "input": [
        {
            "role": "user",
            "type": "message",
            "content": """
Evaluate the quality of reasoning in this prime factorization attempt. Focus on:
1. Mathematical rigor and correctness
2. Step-by-step clarity 
3. Completeness of explanation
4. Efficiency of the approach

Score on a scale from 0.0 to 1.0, where:
- 0.0: Missing or completely incorrect reasoning
- 0.25: Poor reasoning with significant errors
- 0.5: Basic reasoning with minor errors
- 0.75: Good reasoning with clear steps
- 1.0: Excellent, mathematically rigorous reasoning

Number to factorize: {{item.number}}
True factors: {{item.factors}}
Algorithm used: {{sample.output_json.algorithm}}
Reasoning steps:
{{sample.output_json.reasoning}}

Provide ONLY a single floating point score between 0.0 and 1.0.
            """
        }
    ],
    "model": "gpt-4o-2024-08-06"
}
```

### Optimizing Training

For best results with o4-mini:
1. Focus on numbers with 1-10 digits (tiers 0-5)
2. Include some challenging examples (tiers 6-8)
3. Use the `--optimize` flag to balance the distribution
4. Set `reasoning_effort` to "high" in hyperparameters

#### Hyperparameter Recommendations

| Parameter               | Value     | Description                                         |
|-------------------------|-----------|-----------------------------------------------------|
| `reasoning_effort`      | "high"    | Controls thoroughness of reasoning (use "high")     |
| `batch_size`            | 32        | Number of examples processed together               |
| `learning_rate_multiplier` | 1.0    | Controls step size during training                  |
| `init_value`            | 0.0       | Initial reward estimate for new outputs             |
| `kl_coef`               | 0.1       | KL penalty to prevent divergence from base model    |
| `n_epochs`              | 3         | Number of training passes through the dataset       |

### Working with Large Datasets

For datasets with thousands of examples:

1. **Create Multiple Tiers**:
   ```bash
   # Generate each tier separately
   python generate_o4mini_rft_data.py --output_dir ./rft_data/tier0 --tiers 0 --samples_per_tier 100
   python generate_o4mini_rft_data.py --output_dir ./rft_data/tier1 --tiers 1 --samples_per_tier 200
   
   # Combine datasets
   cat ./rft_data/tier0/train.jsonl ./rft_data/tier1/train.jsonl > ./rft_data/combined/train.jsonl
   cat ./rft_data/tier0/valid.jsonl ./rft_data/tier1/valid.jsonl > ./rft_data/combined/valid.jsonl
   ```

2. **Parallel Processing**:
   ```bash
   # Use the Batch tool to run data generation in parallel
   python -m concurrent.batch_processing generate_o4mini_rft_data.py --output_dir ./rft_data --num_workers 4
   ```

## Examples

### Running a Small-Scale Test

```bash
python run_o4mini_rft.py --num_samples 100 --model_suffix "test-factorization" --optimize
```

### Training Multiple Models with Different Configurations

```bash
# Model 1: Focus on easy to medium difficulty
python generate_o4mini_rft_data.py --output_dir ./rft_data/easy_medium --num_samples 300 --max_value 1000000
python run_o4mini_rft.py --model_suffix "factorization-easy-medium" --output_dir ./rft_data/easy_medium

# Model 2: Focus on medium to hard difficulty
python generate_o4mini_rft_data.py --output_dir ./rft_data/medium_hard --num_samples 300 --max_value 1000000000 
python run_o4mini_rft.py --model_suffix "factorization-medium-hard" --output_dir ./rft_data/medium_hard
```

### Using a Fine-Tuned Model

```bash
python use_factorization_model.py 123456789 --model "ft:o4-mini-2025-04-16:your-org:factorization-expert:123abc" --verbose
```

### Command Line Interface Examples

#### Basic Factorization

```bash
python use_factorization_model.py 210 --model "ft:o4-mini-2025-04-16:your-org:factorization-expert:123abc"
```

Output:
```
============================================================
📊 Factorization of 210
============================================================
🔢 Factors: [2, 3, 5, 7]
✅ Correct: True
⚙️ Algorithm: WheelFactorization
⏱️ Reported time: 0.0230 seconds
🔄 API call time: 1.2534 seconds
🎯 Confidence: 0.9900

📝 Reasoning:
  1. First, I'll check if the number is divisible by small primes.
  2. 210 is divisible by 2: 210 ÷ 2 = 105
  3. 105 is divisible by 3: 105 ÷ 3 = 35
  4. 35 is divisible by 5: 35 ÷ 5 = 7
  5. 7 is a prime number.
  6. Therefore, the prime factorization of 210 is 2 × 3 × 5 × 7.

🔍 Verification:
  ✅ 2 × 3 × 5 × 7 = 210
============================================================
```

#### Factorizing a Large Number

```bash
python use_factorization_model.py 1234567890 --model "ft:o4-mini-2025-04-16:your-org:factorization-expert:123abc"
```

#### Raw JSON Output

```bash
python use_factorization_model.py 210 --model "ft:o4-mini-2025-04-16:your-org:factorization-expert:123abc" --raw
```

Output:
```json
{
  "factors": [2, 3, 5, 7],
  "algorithm": "WheelFactorization",
  "reasoning": [
    "First, I'll check if the number is divisible by small primes.",
    "210 is divisible by 2: 210 ÷ 2 = 105",
    "105 is divisible by 3: 105 ÷ 3 = 35",
    "35 is divisible by 5: 35 ÷ 5 = 7",
    "7 is a prime number.",
    "Therefore, the prime factorization of 210 is 2 × 3 × 5 × 7."
  ],
  "time_taken": 0.023,
  "confidence": 0.99,
  "api_call_time": 1.2534
}

## File Descriptions

### Core Modules
- `prime_rft_model.py`: Core RFT components and API interfaces
  - Contains the `FactorizationRFTTrainer` class
  - Handles API communication with OpenAI
  - Implements JSON schema and grader configuration
  - Manages training and evaluation processes

- `prime_rft_dataset_generator.py`: Base dataset generation utilities
  - Implements the S(N,K) framework for factorization problems
  - Generates diverse examples across difficulty tiers
  - Provides utilities for file creation and management

- `prime_reinforcement_grader.py`: Factorization grading components
  - Contains the `FactorizationGrader` class for comprehensive evaluation
  - Implements factorization correctness validation
  - Measures algorithm selection quality
  - Evaluates reasoning steps and efficiency

### Pipeline Scripts
- `generate_o4mini_rft_data.py`: Optimized data generation for o4-mini
  - Creates balanced datasets with appropriate difficulty distribution
  - Generates training, validation, and test splits
  - Enhances examples with rich system prompts
  - Configures metadata and statistics for tracking

- `run_o4mini_rft.py`: End-to-end RFT pipeline
  - Orchestrates the complete fine-tuning workflow
  - Manages file uploads and job creation
  - Monitors training progress and metrics
  - Evaluates and records model performance

- `use_factorization_model.py`: Interface for using fine-tuned models
  - Provides a clean CLI for factorization tasks
  - Handles API communication with fine-tuned models
  - Formats and verifies factorization results
  - Supports both user-friendly and raw output formats

## Directory Structure After Running the Pipeline

```
rft_data/
  ├── run_YYYYMMDD_HHMMSS/        # Timestamped run directory
  │   ├── data/                   # Dataset files
  │   │   ├── train.jsonl         # Training examples
  │   │   ├── valid.jsonl         # Validation examples
  │   │   ├── test.jsonl          # Testing examples
  │   │   ├── factorization_dataset.jsonl  # All examples combined
  │   │   └── dataset_stats.json  # Statistics about the dataset
  │   │
  │   ├── models/                 # Model checkpoints and artifacts
  │   │   └── ...
  │   │
  │   └── results/                # Evaluation results
  │       ├── job_status.json     # Final status of the training job
  │       └── evaluation_results.json  # Performance metrics
  │
  └── ...                        # Other run directories
```

## Monitoring and Metrics

During training, the pipeline tracks key metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| `train_reward_mean` | Average reward on training batches | Increasing trend |
| `valid_reward_mean` | Average reward on validation set | > 0.8 |
| `correctness` | % of correct factorizations | > 90% |
| `algorithm_selection` | % of optimal algorithm choices | > 80% |
| `reasoning_quality` | Average score for explanation quality | > 0.75 |

## Tips for Best Results

1. **Data Quality**:
   - Generate at least 500 samples for meaningful training
   - Use a diverse tier distribution with `--optimize`
   - Include both easy and challenging examples

2. **Training Configuration**:
   - Set `reasoning_effort` to "high"
   - Use 3-5 training epochs
   - Monitor `valid_reward_mean` for convergence

3. **Model Selection**:
   - Run multiple fine-tuning jobs with different configurations
   - Compare models based on validation metrics
   - Select the model with the best balance of correctness and reasoning

4. **Deployment**:
   - Use a temperature of 0.2-0.4 for reliable factorization 
   - For larger numbers, increase the token limit
   - Verify factorizations by multiplying the factors

## Troubleshooting

| Problem | Possible Causes | Solutions |
|---------|-----------------|-----------|
| API Errors | Insufficient permissions | Ensure your API key has access to o4-mini and RFT capabilities |
| Grading Issues | Errors in grader functions | Check the Python code in grader components |
| Poor Performance | Inadequate training data | Increase dataset size and diversity |
| Training Failures | Resource limitations | Reduce batch size or dataset size |
| Factorization Errors | Model limitations | Verify model output and consider additional training |

### Common Error Messages

```
Error: {"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota"}}
```
Solution: Check your OpenAI API usage and billing status.

```
Error: {"error":{"message":"The fine-tuning job failed due to an internal error.","type":"invalid_request_error"}}
```
Solution: Check your grader code for errors and try again with a smaller dataset.

```
Error: The `response_format` parameter must be set to `{ type: "json_schema"... }` when using a fine-tuned model
```
Solution: Ensure you're setting the correct response format in API calls.