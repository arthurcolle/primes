#!/usr/bin/env python
"""
Prime Number Verification with Reinforcement Learning

This script demonstrates how to train an LLM to become better at prime number 
verification and factorization using the verifiers framework with GRPO.
"""

import os
import argparse
import logging
from typing import List, Dict, Any

from datasets import Dataset
import pandas as pd
import pyarrow.parquet as pq

import torch
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from trl import GRPOConfig
from datasets import load_dataset

# Import verifiers tools
from verifiers.envs.prime_env import PrimeEnv
from verifiers.tools.prime_tools import is_prime, factorize, next_prime, prime_count, verify_factorization
from verifiers.rubrics.prime_rubric import PrimeRubric
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer
from verifiers.inference.vllm_client import AsyncVLLMClient
from verifiers.envs.multiturn_env import MultiTurnEnv 
from verifiers.utils.config_utils import TrainingMode


# Set up argument parser
parser = argparse.ArgumentParser(description="Train LLM on prime number verification with RL")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use for training")
parser.add_argument("--inference_url", type=str, default="http://localhost:8000", help="URL for the vLLM inference server")
parser.add_argument("--data_path", type=str, default=None, help="Path to prime dataset parquet file")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
parser.add_argument("--difficulty", type=str, default="mixed", choices=["easy", "medium", "hard", "mixed"], 
                    help="Difficulty level for training examples")
parser.add_argument("--max_steps", type=int, default=5, help="Maximum steps in environment")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save model outputs")
parser.add_argument("--report_to", type=str, default="wandb", help="Where to report training results")
parser.add_argument("--debug", action="store_true", help="Enable debug mode with fewer samples")
parser.add_argument("--use_cache", action="store_true", help="Enable vLLM cache for inference")
parser.add_argument("--no_eval", action="store_true", help="Skip evaluation during training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")

args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train_model():
    """Train model on prime verification using GRPO."""
    logger.info(f"Setting up training for model: {args.model}")
    
    # Create prime environment
    env = PrimeEnv(
        data_path=args.data_path,
        difficulty_level=args.difficulty,
        max_steps=args.max_steps
    )
    
    # Get dataset
    train_dataset = env.dataset
    eval_dataset = None
    
    # Limit dataset size for debug mode or based on max_samples
    if args.debug:
        logger.info("Debug mode enabled, using smaller dataset")
        train_dataset = train_dataset.select(range(min(50, len(train_dataset))))
    elif args.max_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
    # Split dataset for training/eval if no separate eval dataset
    if not args.no_eval and eval_dataset is None:
        split_dataset = train_dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Set up model and tokenizer for training
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Configure tokenizer for chat format if needed
    if not tokenizer.chat_template:
        logger.warning("Tokenizer does not have a chat template. Using default template.")
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n<s>system\n{{ message['content'] }}\n\n{% elif message['role'] == 'user' %}\nuser\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}\nassistant\n{{ message['content'] }}\n</s>\n{% endif %}\n{% endfor %}\n"

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create vLLM client for inference
    inference_client = AsyncVLLMClient(
        args.inference_url,
        tokenizer=tokenizer,
        enable_prefix_caching=args.use_cache,
        max_tokens=512,
    )
    
    # Configure GRPO parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        logging_steps=10,
        report_to=args.report_to.split(",") if args.report_to else None,
        save_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
    )
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        ppo_epochs=1,
        reward_penalty=-0.5,  # Penalty for invalid outputs
        epsilon=0.2,           # PPO clipping range
        beta=0.01,             # Entropy coefficient
        vf_coef=0.5,           # Value function coefficient
        device_map="auto",     # Automatically map model to devices
        max_length=2048,       # Maximum sequence length
        max_prompt_length=1024, # Maximum prompt length
        is_reward_based=True,  # Use rewards as optimization target
        reward_type="completion", # Reward based on completions
        normalize_rewards=True  # Normalize rewards
    )
    
    # Create trainer
    trainer = GRPOEnvTrainer(
        env=env,
        model=args.model,
        tokenizer=tokenizer,
        training_args=training_args,
        grpo_config=grpo_config,
        inference_client=inference_client,
        training_mode=TrainingMode.LOCAL if not hasattr(args, "deepspeed") else TrainingMode.DS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not args.no_eval else None
    )
    
    # Start training
    logger.info("Starting GRPO training...")
    result = trainer.train()
    
    # Log results
    logger.info(f"Training complete. Results: {result}")
    
    # Save model
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")
    
    return trainer, result

def evaluate_model(trainer, result=None):
    """Evaluate the trained model on prime verification tasks."""
    if args.no_eval:
        logger.info("Evaluation skipped as requested")
        return
    
    # Use additional test data if available, otherwise use the existing eval set
    eval_dataset = trainer.eval_dataset
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("No evaluation dataset available, skipping evaluation")
        return
    
    logger.info(f"Evaluating model on {len(eval_dataset)} samples...")
    
    # Run evaluation
    eval_result = trainer.evaluate()
    
    # Log evaluation results
    logger.info(f"Evaluation results: {eval_result}")
    
    return eval_result

def main():
    """Main entry point."""
    # Print configuration
    logger.info("Configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training might be slow.")
    else:
        logger.info(f"Using CUDA with {torch.cuda.device_count()} device(s)")
    
    # Launch inference server if not already running and reachable
    import requests
    try:
        requests.get(args.inference_url, timeout=2)
        logger.info(f"Inference server found at {args.inference_url}")
    except:
        logger.error(f"Inference server not reachable at {args.inference_url}")
        logger.error("Please start the vLLM server using the following command:")
        logger.error(f"python -m verifiers.inference.vllm_serve --model {args.model} " +
                     "--tensor_parallel_size 4 --max_model_len 8192 --gpu_memory_utilization 0.9 " +
                     "--enable_prefix_caching True")
        return
    
    # Train the model
    trainer, result = train_model()
    
    # Evaluate the model
    eval_result = evaluate_model(trainer, result)
    
    logger.info("Training and evaluation complete")

if __name__ == "__main__":
    main()