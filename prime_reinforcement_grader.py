#!/usr/bin/env python3
"""
prime_reinforcement_grader.py

This module implements a grading system for reinforcement fine-tuning of factorization models.
The graders evaluate factorization performance along multiple dimensions and provide
reward signals that can be used to optimize model behavior through RFT.
"""

import json
import time
import math
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class FactorizationAttempt:
    """Represents a single factorization attempt by a model."""
    number: int
    predicted_factors: List[int]
    algorithm_used: str
    reasoning_steps: List[str]
    time_taken: float
    confidence: float

@dataclass
class FactorizationReference:
    """Reference data for factorization grading."""
    number: int
    true_factors: List[int]
    optimal_algorithm: Optional[str] = None
    expected_time: Optional[float] = None

class BaseGrader:
    """Base class for factorization graders."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    def grade(self, attempt: FactorizationAttempt, reference: FactorizationReference) -> float:
        """
        Grade a factorization attempt.
        
        Args:
            attempt: The factorization attempt to grade
            reference: Reference data for grading
            
        Returns:
            A score between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement grade")

class CorrectnessGrader(BaseGrader):
    """Grades the correctness of factorization."""
    
    def grade(self, attempt: FactorizationAttempt, reference: FactorizationReference) -> float:
        """
        Check if the predicted factors are correct.
        
        A perfect score (1.0) is given if the product of predicted factors equals
        the original number AND all factors are prime.
        
        Args:
            attempt: The factorization attempt to grade
            reference: Reference data for grading
            
        Returns:
            A score between 0 and 1
        """
        # Check if product of factors equals the original number
        product = 1
        for factor in attempt.predicted_factors:
            product *= factor
        
        if product != reference.number:
            return 0.0
        
        # Check if all predicted factors are in the reference factors
        # (this also implicitly checks primality if reference factors are all prime)
        predicted_sorted = sorted(attempt.predicted_factors)
        reference_sorted = sorted(reference.true_factors)
        
        if predicted_sorted == reference_sorted:
            return 1.0
        
        # Partial credit for finding some correct factors
        correct_factors = [f for f in predicted_sorted if f in reference_sorted]
        return len(correct_factors) / len(reference_sorted)

class EfficiencyGrader(BaseGrader):
    """Grades the efficiency of factorization."""
    
    def __init__(self, name: str = "Efficiency", weight: float = 1.0, 
                 time_threshold: float = 10.0, bit_length_factor: float = 0.1):
        super().__init__(name, weight)
        self.time_threshold = time_threshold
        self.bit_length_factor = bit_length_factor
    
    def grade(self, attempt: FactorizationAttempt, reference: FactorizationReference) -> float:
        """
        Grade the efficiency of the factorization.
        
        The score is inversely proportional to the time taken, normalized
        by the expected time or by the bit length of the number.
        
        Args:
            attempt: The factorization attempt to grade
            reference: Reference data for grading
            
        Returns:
            A score between 0 and 1
        """
        if attempt.time_taken <= 0:
            return 0.0
        
        # If we have an expected time, use it for normalization
        if reference.expected_time is not None and reference.expected_time > 0:
            efficiency = min(reference.expected_time / attempt.time_taken, 1.0)
        else:
            # Otherwise normalize by bit length
            bit_length = len(bin(reference.number)) - 2
            adjusted_threshold = self.time_threshold * (1 + self.bit_length_factor * bit_length)
            efficiency = max(0, 1 - (attempt.time_taken / adjusted_threshold))
        
        return efficiency

class AlgorithmSelectionGrader(BaseGrader):
    """Grades the appropriateness of the algorithm selection."""
    
    def __init__(self, name: str = "Algorithm Selection", weight: float = 1.0,
                 algorithm_preferences: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__(name, weight)
        self.algorithm_preferences = algorithm_preferences or {}
    
    def grade(self, attempt: FactorizationAttempt, reference: FactorizationReference) -> float:
        """
        Grade the algorithm selection based on the properties of the number.
        
        Args:
            attempt: The factorization attempt to grade
            reference: Reference data for grading
            
        Returns:
            A score between 0 and 1
        """
        # If optimal algorithm is provided in reference, use it
        if reference.optimal_algorithm is not None:
            return 1.0 if attempt.algorithm_used == reference.optimal_algorithm else 0.0
        
        # Otherwise use our algorithm preferences based on number properties
        bit_length = len(bin(reference.number)) - 2
        
        # Default preferences if none provided
        if not self.algorithm_preferences:
            self.algorithm_preferences = {
                "TrialDivision": {"max_bits": 32, "score": 1.0},
                "WheelFactorization": {"max_bits": 64, "score": 1.0},
                "PollardRho": {"max_bits": 128, "score": 1.0},
                "ECM": {"max_bits": 256, "score": 1.0},
                "QuadraticSieve": {"max_bits": 512, "score": 1.0},
                "GNFS": {"max_bits": float('inf'), "score": 1.0}
            }
        
        # Find all appropriate algorithms for this bit length
        appropriate_algorithms = []
        for algo, prefs in self.algorithm_preferences.items():
            if bit_length <= prefs.get("max_bits", float('inf')):
                appropriate_algorithms.append((algo, prefs.get("score", 1.0)))
        
        # If the selected algorithm is among the appropriate ones, score based on its preference
        for algo, score in appropriate_algorithms:
            if attempt.algorithm_used == algo:
                return score
        
        # If no appropriate algorithm was selected, give a low score
        return 0.1

class ReasoningGrader(BaseGrader):
    """Grades the quality of reasoning steps."""
    
    def __init__(self, name: str = "Reasoning", weight: float = 1.0,
                expected_step_count: Optional[Dict[str, int]] = None):
        super().__init__(name, weight)
        self.expected_step_count = expected_step_count or {}
    
    def grade(self, attempt: FactorizationAttempt, reference: FactorizationReference) -> float:
        """
        Grade the quality of reasoning steps.
        
        Args:
            attempt: The factorization attempt to grade
            reference: Reference data for grading
            
        Returns:
            A score between 0 and 1
        """
        # If no reasoning steps, return minimum score
        if not attempt.reasoning_steps:
            return 0.0
        
        # Determine expected steps based on algorithm or default values
        expected_steps = self.expected_step_count.get(
            attempt.algorithm_used,
            max(3, min(10, len(bin(reference.number)) - 2))  # Default based on bit length
        )
        
        # Score based on step count (neither too few nor too many)
        step_count_score = min(
            1.0,
            len(attempt.reasoning_steps) / expected_steps if len(attempt.reasoning_steps) < expected_steps
            else expected_steps / len(attempt.reasoning_steps)
        )
        
        # Check step progression logic (simplified)
        progression_score = 0.0
        if len(attempt.reasoning_steps) >= 2:
            logical_steps = 0
            for i in range(1, len(attempt.reasoning_steps)):
                # Simplified logic check
                if self._steps_are_logical(attempt.reasoning_steps[i-1], attempt.reasoning_steps[i]):
                    logical_steps += 1
            
            progression_score = logical_steps / (len(attempt.reasoning_steps) - 1)
        
        # Combine scores
        return 0.5 * step_count_score + 0.5 * progression_score
    
    def _steps_are_logical(self, prev_step: str, curr_step: str) -> bool:
        """
        Simplified check for logical progression between steps.
        
        In a real implementation, this would use more sophisticated 
        NLP techniques to analyze the reasoning.
        """
        # Simplified logic check (placeholder)
        if prev_step and curr_step:
            return True
        return False

class FactorizationGrader:
    """
    Main grader for factorization attempts that combines multiple sub-graders.
    This is the top-level grader used for reinforcement fine-tuning.
    """
    
    def __init__(self, graders: Optional[List[BaseGrader]] = None):
        """
        Initialize with a list of sub-graders.
        
        Args:
            graders: List of graders to use for evaluation
        """
        self.graders = graders or [
            CorrectnessGrader("Correctness", weight=3.0),
            EfficiencyGrader("Efficiency", weight=1.0),
            AlgorithmSelectionGrader("Algorithm Selection", weight=1.0),
            ReasoningGrader("Reasoning", weight=1.0)
        ]
    
    def grade(self, attempt: FactorizationAttempt, reference: FactorizationReference) -> Dict[str, float]:
        """
        Grade a factorization attempt using all sub-graders.
        
        Args:
            attempt: The factorization attempt to grade
            reference: Reference data for grading
            
        Returns:
            Dictionary with individual and overall scores
        """
        results = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        for grader in self.graders:
            score = grader.grade(attempt, reference)
            results[grader.name] = score
            weighted_sum += score * grader.weight
            total_weight += grader.weight
        
        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        results["overall"] = overall_score
        
        return results

class RFTGrader:
    """
    Grader for Reinforcement Fine-Tuning (RFT) that converts model outputs
    to factorization attempts and applies the factorization grader.
    
    This class is designed to be used with OpenAI's reinforcement fine-tuning API.
    """
    
    def __init__(self):
        """Initialize the RFT grader with the factorization grader."""
        self.factorization_grader = FactorizationGrader()
    
    def grade_json_output(self, json_str: str, reference_data: Dict[str, Any]) -> float:
        """
        Grade a JSON output from a model.
        
        Args:
            json_str: JSON string from the model
            reference_data: Reference data containing true factors, etc.
            
        Returns:
            A score between 0 and 1
        """
        try:
            output = json.loads(json_str)
            
            # Extract factorization attempt data
            attempt = FactorizationAttempt(
                number=reference_data["number"],
                predicted_factors=output.get("factors", []),
                algorithm_used=output.get("algorithm", "Unknown"),
                reasoning_steps=output.get("reasoning", []),
                time_taken=output.get("time_taken", 0.0),
                confidence=output.get("confidence", 0.0)
            )
            
            # Create reference from data
            reference = FactorizationReference(
                number=reference_data["number"],
                true_factors=reference_data["factors"],
                optimal_algorithm=reference_data.get("optimal_algorithm"),
                expected_time=reference_data.get("expected_time")
            )
            
            # Apply the factorization grader
            scores = self.factorization_grader.grade(attempt, reference)
            
            return scores["overall"]
            
        except Exception as e:
            # If there's an error parsing or grading, return a low score
            print(f"Error grading output: {e}")
            return 0.1
    
    def generate_grader_config(self) -> Dict[str, Any]:
        """
        Generate the grader configuration for OpenAI's RFT API.
        
        Returns:
            Grader configuration dictionary
        """
        return {
            "type": "python",
            "python": {
                "fn": """
def grade(item, sample):
    import json
    
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

def generate_rft_dataset(numbers: List[int], true_factors: List[List[int]], 
                         optimal_algorithms: Optional[List[str]] = None,
                         expected_times: Optional[List[float]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate a dataset for reinforcement fine-tuning.
    
    Args:
        numbers: List of numbers to factorize
        true_factors: List of true factors for each number
        optimal_algorithms: Optional list of optimal algorithms for each number
        expected_times: Optional list of expected times for each number
        
    Returns:
        Dictionary with training and validation datasets
    """
    if optimal_algorithms is None:
        optimal_algorithms = [None] * len(numbers)
    
    if expected_times is None:
        expected_times = [None] * len(numbers)
    
    dataset = []
    
    for i, number in enumerate(numbers):
        item = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert factorization assistant. Given a number, find its prime factorization efficiently."
                },
                {
                    "role": "user",
                    "content": f"Find the prime factorization of {number}. Return your answer as JSON with the following fields: 'factors' (list of integers), 'algorithm' (string), 'reasoning' (list of steps), 'time_taken' (float), and 'confidence' (float)."
                }
            ],
            "number": number,
            "factors": true_factors[i]
        }
        
        if optimal_algorithms[i]:
            item["optimal_algorithm"] = optimal_algorithms[i]
        
        if expected_times[i]:
            item["expected_time"] = expected_times[i]
        
        dataset.append(item)
    
    # Split into training and validation (80/20)
    split_idx = int(0.8 * len(dataset))
    np.random.shuffle(dataset)
    
    return {
        "training": dataset[:split_idx],
        "validation": dataset[split_idx:]
    }

def generate_rft_json_schema() -> Dict[str, Any]:
    """
    Generate the JSON schema for the factorization model output.
    
    Returns:
        JSON schema dictionary
    """
    return {
        "type": "json_schema",
        "json_schema": {
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
    }

def main():
    """Main function for demonstration."""
    # Example usage
    grader = RFTGrader()
    
    # Sample reference data
    reference_data = {
        "number": 15,
        "factors": [3, 5],
        "optimal_algorithm": "TrialDivision",
        "expected_time": 0.001
    }
    
    # Sample model output
    model_output = json.dumps({
        "factors": [3, 5],
        "algorithm": "TrialDivision",
        "reasoning": [
            "First, I'll check if the number is divisible by small primes.",
            "15 is not divisible by 2.",
            "15 is divisible by 3: 15 ÷ 3 = 5.",
            "5 is a prime number.",
            "Therefore, the prime factorization of 15 is 3 × 5."
        ],
        "time_taken": 0.002,
        "confidence": 0.99
    })
    
    # Grade the output
    score = grader.grade_json_output(model_output, reference_data)
    print(f"Overall score: {score:.4f}")
    
    # Print grader configuration
    config = grader.generate_grader_config()
    print("\nGrader configuration:")
    print(json.dumps(config, indent=2))
    
    # Generate a sample dataset
    numbers = [15, 21, 35, 91, 143]
    true_factors = [
        [3, 5],
        [3, 7],
        [5, 7],
        [7, 13],
        [11, 13]
    ]
    
    dataset = generate_rft_dataset(numbers, true_factors)
    print(f"\nGenerated dataset with {len(dataset['training'])} training and {len(dataset['validation'])} validation examples.")
    
    # Generate JSON schema
    schema = generate_rft_json_schema()
    print("\nJSON Schema:")
    print(json.dumps(schema, indent=2))

if __name__ == "__main__":
    main()