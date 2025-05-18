#!/usr/bin/env python3
"""
prime_rft_model.py

This module implements a reinforcement fine-tuning pipeline for factorization models.
It integrates with the OpenAI API to fine-tune models for prime factorization tasks
using reinforcement learning with expert feedback.
"""

import os
import json
import time
import requests
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

# Check if OpenAI API key is available
try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    OPENAI_API_KEY = "not_set"

# API constants
API_BASE = "https://api.openai.com/v1"
TIMEOUT = 60  # seconds

@dataclass
class RFTConfig:
    """Configuration for RFT fine-tuning job."""
    model_id: str
    training_file_id: str
    validation_file_id: str
    suffix: Optional[str] = None
    grader_config: Optional[Dict[str, Any]] = None
    json_schema: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    n_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    

class OpenAIAPI:
    """Helper class for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with API key.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """
        Upload a file to OpenAI.
        
        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file (e.g., "fine-tune")
            
        Returns:
            File ID
        """
        with open(file_path, 'rb') as f:
            files = {"file": f}
            response = requests.post(
                f"{API_BASE}/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"purpose": purpose},
                files=files,
                timeout=TIMEOUT
            )
        
        response.raise_for_status()
        file_id = response.json()["id"]
        print(f"Uploaded file {file_path} with ID: {file_id}")
        return file_id
    
    def create_fine_tuning_job(self, config: RFTConfig) -> str:
        """
        Create a fine-tuning job.
        
        Args:
            config: RFT configuration
            
        Returns:
            Job ID
        """
        # Prepare the payload
        payload = {
            "training_file": config.training_file_id,
            "validation_file": config.validation_file_id,
            "model": config.model_id,
            "method": {
                "type": "reinforcement",
                "reinforcement": {
                    "grader": config.grader_config,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": config.json_schema
                    }
                }
            }
        }
        
        # Add optional parameters
        if config.suffix:
            payload["suffix"] = config.suffix
            
        if config.hyperparameters:
            payload["method"]["reinforcement"]["hyperparameters"] = config.hyperparameters
            
        if config.n_epochs:
            payload["hyperparameters"] = payload.get("hyperparameters", {})
            payload["hyperparameters"]["n_epochs"] = config.n_epochs
            
        if config.batch_size:
            payload["hyperparameters"] = payload.get("hyperparameters", {})
            payload["hyperparameters"]["batch_size"] = config.batch_size
            
        if config.learning_rate_multiplier:
            payload["hyperparameters"] = payload.get("hyperparameters", {})
            payload["hyperparameters"]["learning_rate_multiplier"] = config.learning_rate_multiplier
        
        # Create the job
        response = requests.post(
            f"{API_BASE}/fine_tuning/jobs",
            headers=self.headers,
            json=payload,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Created fine-tuning job with ID: {job_id}")
        return job_id
    
    def get_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job details
        """
        response = requests.get(
            f"{API_BASE}/fine_tuning/jobs/{job_id}",
            headers=self.headers,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_fine_tuning_events(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get events for a fine-tuning job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of events
        """
        response = requests.get(
            f"{API_BASE}/fine_tuning/jobs/{job_id}/events",
            headers=self.headers,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        return response.json()["data"]
    
    def cancel_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a fine-tuning job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job details
        """
        response = requests.post(
            f"{API_BASE}/fine_tuning/jobs/{job_id}/cancel",
            headers=self.headers,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get details of a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model details
        """
        response = requests.get(
            f"{API_BASE}/models/{model_id}",
            headers=self.headers,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        return response.json()
    
    def create_completion(self, model_id: str, prompt: str, 
                          temperature: float = 0.7, max_tokens: int = 256) -> Dict[str, Any]:
        """
        Create a completion using a model.
        
        Args:
            model_id: Model ID
            prompt: Prompt to complete
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            Completion response
        """
        payload = {
            "model": model_id,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{API_BASE}/completions",
            headers=self.headers,
            json=payload,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        return response.json()
    
    def chat_completion(self, model_id: str, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, max_tokens: int = 256) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            model_id: Model ID
            messages: List of message objects
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            Chat completion response
        """
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=TIMEOUT
        )
        
        response.raise_for_status()
        return response.json()

class FactorizationRFTTrainer:
    """
    Reinforcement fine-tuning trainer for factorization models.
    """
    
    def __init__(self, base_model: str = "o4-mini"):
        """
        Initialize the trainer.
        
        Args:
            base_model: Base model to fine-tune
        """
        self.base_model = base_model
        self.api = OpenAIAPI()
        
        # Load necessary configurations
        self.json_schema = self._load_json_schema()
        self.grader_config = self._load_grader_config()
    
    def _load_json_schema(self) -> Dict[str, Any]:
        """
        Load the JSON schema for factorization output.
        
        Returns:
            JSON schema configuration
        """
        # Structured output schema for factorization
        return {
            "name": "prime_factorization",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "factors": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of prime factors"
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "Algorithm used for factorization"
                    },
                    "reasoning": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Step-by-step reasoning process"
                    },
                    "time_taken": {
                        "type": "number",
                        "description": "Time taken for factorization in seconds"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in the factorization (0-1)"
                    }
                },
                "required": ["factors", "algorithm", "reasoning"],
                "additionalProperties": False
            }
        }
    
    def _load_grader_config(self) -> Dict[str, Any]:
        """
        Load the grader configuration for RFT.
        
        Returns:
            Grader configuration
        """
        # Simplified version of the Python grader from prime_reinforcement_grader.py
        return {
            "type": "python",
            "python": {
                "fn": """
def grade(item, sample):
    # Extract model output and reference data
    try:
        model_output = sample.get("output_json", {})
        
        # Extract reference data
        reference_data = {
            "number": item["number"],
            "factors": item["factors"],
            "optimal_algorithm": item.get("optimal_algorithm"),
            "expected_time": item.get("expected_time")
        }
        
        # Grade the output
        overall_score = grade_factorization(model_output, reference_data)
        return overall_score
    except Exception as e:
        print(f"Grading error: {e}")
        return 0.0

def grade_factorization(output, reference):
    # Check correctness
    correctness = grade_correctness(output.get("factors", []), reference["number"], reference["factors"])
    
    # Check efficiency
    efficiency = grade_efficiency(
        output.get("time_taken", 0.0),
        reference.get("expected_time"),
        reference["number"]
    )
    
    # Check algorithm selection
    algorithm_score = grade_algorithm(
        output.get("algorithm", "Unknown"),
        reference.get("optimal_algorithm"),
        reference["number"]
    )
    
    # Check reasoning quality
    reasoning_score = grade_reasoning(
        output.get("reasoning", []),
        output.get("algorithm", "Unknown"),
        reference["number"]
    )
    
    # Combine scores with weights
    weights = {
        "correctness": 3.0,
        "efficiency": 1.0,
        "algorithm": 1.0,
        "reasoning": 1.0
    }
    
    total_weight = sum(weights.values())
    weighted_sum = (
        correctness * weights["correctness"] +
        efficiency * weights["efficiency"] +
        algorithm_score * weights["algorithm"] +
        reasoning_score * weights["reasoning"]
    )
    
    return weighted_sum / total_weight

def grade_correctness(predicted_factors, number, true_factors):
    if not predicted_factors:
        return 0.0
    
    # Check if product equals the original number
    product = 1
    for factor in predicted_factors:
        product *= factor
    
    if product != number:
        return 0.0
    
    # Check if all predicted factors match reference
    predicted_sorted = sorted(predicted_factors)
    reference_sorted = sorted(true_factors)
    
    if predicted_sorted == reference_sorted:
        return 1.0
    
    # Partial credit for some correct factors
    correct_factors = [f for f in predicted_sorted if f in reference_sorted]
    return len(correct_factors) / len(reference_sorted)

def grade_efficiency(time_taken, expected_time, number):
    if not time_taken or time_taken <= 0:
        return 0.0
    
    if expected_time and expected_time > 0:
        return min(expected_time / time_taken, 1.0)
    
    # Normalize by bit length
    bit_length = len(bin(number)) - 2
    time_threshold = 10.0 * (1 + 0.1 * bit_length)
    return max(0, 1 - (time_taken / time_threshold))

def grade_algorithm(algorithm_used, optimal_algorithm, number):
    if optimal_algorithm:
        return 1.0 if algorithm_used == optimal_algorithm else 0.0
    
    # Simple algorithm preferences based on bit length
    bit_length = len(bin(number)) - 2
    
    algorithm_preferences = {
        "TrialDivision": {"max_bits": 32, "score": 1.0},
        "WheelFactorization": {"max_bits": 64, "score": 1.0},
        "PollardRho": {"max_bits": 128, "score": 1.0},
        "ECM": {"max_bits": 256, "score": 1.0},
        "QuadraticSieve": {"max_bits": 512, "score": 1.0},
        "GNFS": {"max_bits": float('inf'), "score": 1.0}
    }
    
    for algo, prefs in algorithm_preferences.items():
        if algorithm_used == algo and bit_length <= prefs["max_bits"]:
            return prefs["score"]
    
    return 0.1

def grade_reasoning(reasoning_steps, algorithm, number):
    if not reasoning_steps:
        return 0.0
    
    # Expected step count based on algorithm or bit length
    bit_length = len(bin(number)) - 2
    expected_steps = max(3, min(10, bit_length // 4))
    
    # Score based on step count
    step_count_score = min(
        1.0,
        len(reasoning_steps) / expected_steps if len(reasoning_steps) < expected_steps
        else expected_steps / len(reasoning_steps)
    )
    
    # Simple progression score
    progression_score = 0.5
    if len(reasoning_steps) >= 2:
        progression_score = 0.8  # Simplified - assume reasonable progression
    
    return 0.5 * step_count_score + 0.5 * progression_score
                """
            }
        }
    
    def train(self, train_file: str, valid_file: str, suffix: str = None,
              hyperparameters: Dict[str, Any] = None) -> str:
        """
        Train a model using reinforcement fine-tuning.
        
        Args:
            train_file: Path to training data file
            valid_file: Path to validation data file
            suffix: Suffix for the fine-tuned model name
            hyperparameters: Hyperparameters for training
            
        Returns:
            Job ID
        """
        # Upload files
        train_file_id = self.api.upload_file(train_file)
        valid_file_id = self.api.upload_file(valid_file)
        
        # Set default hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = {
                "reasoning_effort": "medium"
            }
        
        # Create config
        config = RFTConfig(
            model_id=self.base_model,
            training_file_id=train_file_id,
            validation_file_id=valid_file_id,
            suffix=suffix,
            grader_config=self.grader_config,
            json_schema=self.json_schema,
            hyperparameters=hyperparameters
        )
        
        # Create fine-tuning job
        job_id = self.api.create_fine_tuning_job(config)
        
        return job_id
    
    def monitor_job(self, job_id: str, interval: int = 60, max_time: int = 3600) -> Dict[str, Any]:
        """
        Monitor a fine-tuning job until it completes or times out.
        
        Args:
            job_id: Job ID
            interval: Polling interval in seconds
            max_time: Maximum monitoring time in seconds
            
        Returns:
            Final job status
        """
        start_time = time.time()
        elapsed = 0
        
        while elapsed < max_time:
            # Get job status
            job = self.api.get_fine_tuning_job(job_id)
            status = job.get("status")
            
            print(f"Job {job_id} status: {status}")
            
            # Check if job completed or failed
            if status in ["succeeded", "failed", "cancelled"]:
                return job
            
            # Get events for detailed progress
            events = self.api.get_fine_tuning_events(job_id)
            for event in events[-5:]:  # Show last 5 events
                print(f"Event: {event.get('message', 'No message')}")
            
            # Wait before polling again
            time.sleep(interval)
            elapsed = time.time() - start_time
        
        print(f"Monitoring timed out after {max_time} seconds")
        return self.api.get_fine_tuning_job(job_id)
    
    def evaluate_model(self, model_id: str, examples: List[Dict[str, Any]], 
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model on factorization examples.
        
        Args:
            model_id: Model ID
            examples: List of test examples
            verbose: Whether to print detailed results
            
        Returns:
            Evaluation metrics
        """
        results = []
        metrics = {
            "total": len(examples),
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "average_score": 0.0
        }
        
        for i, example in enumerate(examples):
            if verbose:
                print(f"\nEvaluating example {i+1}/{len(examples)}")
            
            number = example.get("number")
            true_factors = example.get("factors", [])
            
            # Create messages
            messages = [
                {"role": "system", "content": "You are an expert factorization assistant. Find the prime factorization efficiently."},
                {"role": "user", "content": f"Find the prime factorization of {number}. Return your answer as JSON with the following fields: 'factors' (list of integers), 'algorithm' (string), 'reasoning' (list of steps), 'time_taken' (float), and 'confidence' (float)."}
            ]
            
            try:
                # Get model response
                response = self.api.chat_completion(
                    model_id=model_id,
                    messages=messages
                )
                
                # Extract content
                content = response["choices"][0]["message"]["content"]
                
                # Parse JSON response
                model_output = json.loads(content)
                
                # Create reference data for grading
                reference_data = {
                    "number": number,
                    "factors": true_factors,
                    "optimal_algorithm": example.get("optimal_algorithm"),
                    "expected_time": example.get("expected_time")
                }
                
                # Grade the response
                from prime_reinforcement_grader import RFTGrader
                grader = RFTGrader()
                score = grader.grade_json_output(content, reference_data)
                
                # Update metrics
                if score > 0.8:
                    metrics["correct"] += 1
                else:
                    metrics["incorrect"] += 1
                    
                metrics["average_score"] += score
                
                # Store result
                result = {
                    "number": number,
                    "true_factors": true_factors,
                    "predicted_factors": model_output.get("factors", []),
                    "algorithm": model_output.get("algorithm", "Unknown"),
                    "score": score
                }
                
                results.append(result)
                
                if verbose:
                    print(f"Number: {number}")
                    print(f"True factors: {true_factors}")
                    print(f"Predicted factors: {model_output.get('factors', [])}")
                    print(f"Algorithm: {model_output.get('algorithm', 'Unknown')}")
                    print(f"Score: {score:.4f}")
                
            except Exception as e:
                metrics["errors"] += 1
                print(f"Error processing example {i+1}: {e}")
        
        # Calculate average score
        if metrics["total"] - metrics["errors"] > 0:
            metrics["average_score"] /= (metrics["total"] - metrics["errors"])
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total examples: {metrics['total']}")
        print(f"Correct: {metrics['correct']} ({metrics['correct']/metrics['total']*100:.2f}%)")
        print(f"Incorrect: {metrics['incorrect']} ({metrics['incorrect']/metrics['total']*100:.2f}%)")
        print(f"Errors: {metrics['errors']} ({metrics['errors']/metrics['total']*100:.2f}%)")
        print(f"Average score: {metrics['average_score']:.4f}")
        
        return {
            "metrics": metrics,
            "results": results
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Reinforcement fine-tuning for factorization models.")
    parser.add_argument("--action", choices=["train", "monitor", "evaluate"], required=True,
                        help="Action to perform")
    parser.add_argument("--train_file", type=str, help="Path to training data file")
    parser.add_argument("--valid_file", type=str, help="Path to validation data file")
    parser.add_argument("--test_file", type=str, help="Path to test data file")
    parser.add_argument("--job_id", type=str, help="Job ID to monitor")
    parser.add_argument("--model_id", type=str, help="Model ID to evaluate")
    parser.add_argument("--base_model", type=str, default="o4-mini",
                        help="Base model to fine-tune")
    parser.add_argument("--suffix", type=str, help="Suffix for the fine-tuned model name")
    
    args = parser.parse_args()
    
    # Create the trainer
    trainer = FactorizationRFTTrainer(base_model=args.base_model)
    
    if args.action == "train":
        if not args.train_file or not args.valid_file:
            print("Error: train_file and valid_file are required for training.")
            return
        
        # Train the model
        job_id = trainer.train(args.train_file, args.valid_file, args.suffix)
        print(f"Started training job: {job_id}")
        
        # Start monitoring
        print("Monitoring job...")
        trainer.monitor_job(job_id)
        
    elif args.action == "monitor":
        if not args.job_id:
            print("Error: job_id is required for monitoring.")
            return
        
        # Monitor the job
        trainer.monitor_job(args.job_id)
        
    elif args.action == "evaluate":
        if not args.model_id or not args.test_file:
            print("Error: model_id and test_file are required for evaluation.")
            return
        
        # Load test examples
        with open(args.test_file, 'r') as f:
            examples = [json.loads(line) for line in f]
        
        # Evaluate the model
        trainer.evaluate_model(args.model_id, examples)
    
if __name__ == "__main__":
    main()