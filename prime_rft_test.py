#!/usr/bin/env python3
"""
prime_rft_test.py

Test the reinforcement fine-tuning pipeline with a small set of examples.
This script simulates the RFT process locally without requiring API calls.
"""

import json
import random
import math
import numpy as np
import sympy
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Simulated model behavior
class SimpleFactorizationModel:
    """Simplified model to simulate factorization behavior."""
    
    def __init__(self, ability_level: float = 0.5, learning_rate: float = 0.1):
        """
        Initialize model with a given ability level.
        
        Args:
            ability_level: Initial ability (0.0 to 1.0)
            learning_rate: How quickly the model improves with feedback
        """
        self.ability = ability_level
        self.learning_rate = learning_rate
        self.algorithm_knowledge = {
            "TrialDivision": 0.9,
            "WheelFactorization": 0.7,
            "PollardRho": 0.5,
            "ECM": 0.3,
            "QuadraticSieve": 0.2,
            "GNFS": 0.1
        }
        self.iterations = 0
        self.correct_count = 0
        self.rewards = []
        
    def factorize(self, number: int) -> Dict[str, Any]:
        """
        Attempt to factorize a number based on current ability.
        
        Args:
            number: Number to factorize
            
        Returns:
            Factorization attempt as a dictionary
        """
        # Determine bit length for algorithm selection
        bit_length = number.bit_length()
        
        # Select algorithm based on bit length and current knowledge
        algorithm = self._select_algorithm(bit_length)
        
        # Simulate factorization attempt
        success_prob = self._compute_success_probability(number, algorithm)
        success = random.random() < success_prob
        
        # Generate result
        if success:
            # Use sympy to get the correct factorization
            true_factors = sorted([int(p) ** e for p, e in sympy.factorint(number).items() for _ in range(e)])
            factors = true_factors
            self.correct_count += 1
        else:
            # Generate incorrect factorization
            if random.random() < 0.5 and number > 4:
                # Miss a factor or add an incorrect one
                true_factors = sorted([int(p) ** e for p, e in sympy.factorint(number).items() for _ in range(e)])
                if len(true_factors) > 1 and random.random() < 0.5:
                    # Miss a factor
                    factors = [f for i, f in enumerate(true_factors) if i != random.randrange(len(true_factors))]
                else:
                    # Add incorrect factor
                    incorrect = random.randint(2, 100)
                    while sympy.isprime(incorrect) and incorrect not in true_factors:
                        incorrect = random.randint(2, 100)
                    factors = true_factors + [incorrect]
            else:
                # Completely wrong factorization
                factors = [number]  # Just return the number itself
        
        # Generate reasoning steps
        reasoning_length = max(3, min(10, int(self.ability * 10)))
        reasoning = self._generate_reasoning(number, algorithm, success, reasoning_length)
        
        # Simulate time taken
        time_factor = 1.0 if success else random.uniform(1.5, 3.0)
        time_taken = self._simulate_time(number, algorithm) * time_factor
        
        # Generate confidence
        confidence = min(0.99, max(0.1, self.ability * (1.2 if success else 0.5)))
        
        return {
            "factors": factors,
            "algorithm": algorithm,
            "reasoning": reasoning,
            "time_taken": time_taken,
            "confidence": confidence
        }
    
    def _select_algorithm(self, bit_length: int) -> str:
        """Select an algorithm based on bit length and knowledge."""
        eligible_algorithms = []
        
        if bit_length <= 32:
            eligible_algorithms = ["TrialDivision", "WheelFactorization"]
        elif bit_length <= 64:
            eligible_algorithms = ["WheelFactorization", "PollardRho"]
        elif bit_length <= 128:
            eligible_algorithms = ["PollardRho", "ECM"]
        elif bit_length <= 512:
            eligible_algorithms = ["ECM", "QuadraticSieve"]
        else:
            eligible_algorithms = ["QuadraticSieve", "GNFS"]
        
        # Select based on knowledge, with some exploration
        if random.random() < 0.2:  # 20% exploration
            return random.choice(eligible_algorithms)
        else:
            return max(eligible_algorithms, key=lambda a: self.algorithm_knowledge[a])
    
    def _compute_success_probability(self, number: int, algorithm: str) -> float:
        """Compute probability of successful factorization."""
        base_prob = self.ability * self.algorithm_knowledge[algorithm]
        
        # Adjust based on number complexity
        complexity_factor = min(1.0, 30 / number.bit_length())
        
        return base_prob * complexity_factor
    
    def _simulate_time(self, number: int, algorithm: str) -> float:
        """Simulate time taken for factorization."""
        bit_length = number.bit_length()
        
        # Base time depending on algorithm and bit length
        if algorithm == "TrialDivision":
            base_time = 0.001 * bit_length * (1 - 0.5 * self.ability)
        elif algorithm == "WheelFactorization":
            base_time = 0.0008 * bit_length * (1 - 0.5 * self.ability)
        elif algorithm == "PollardRho":
            base_time = 0.005 * math.sqrt(bit_length) * (1 - 0.7 * self.ability)
        elif algorithm == "ECM":
            base_time = 0.01 * bit_length * (1 - 0.7 * self.ability)
        elif algorithm == "QuadraticSieve":
            base_time = 0.02 * bit_length * math.log(bit_length) * (1 - 0.8 * self.ability)
        elif algorithm == "GNFS":
            base_time = 0.05 * bit_length * math.log(bit_length) * (1 - 0.8 * self.ability)
        else:
            base_time = 0.1 * bit_length
            
        # Add some randomness
        return base_time * random.uniform(0.8, 1.2)
    
    def _generate_reasoning(self, number: int, algorithm: str, success: bool, length: int) -> List[str]:
        """Generate simulated reasoning steps."""
        reasoning = []
        
        # Initial approach
        reasoning.append(f"I'll use {algorithm} to factorize {number}.")
        
        if algorithm == "TrialDivision":
            reasoning.append(f"I'll check divisibility by small primes starting with 2, 3, 5, 7, 11, ...")
        elif algorithm == "WheelFactorization":
            reasoning.append("Using wheel factorization to skip multiples of 2, 3, 5.")
        elif algorithm == "PollardRho":
            reasoning.append("Using Pollard's Rho algorithm with starting value x₀=2, c=1.")
        elif algorithm == "ECM":
            reasoning.append("Using Elliptic Curve Method with B1=10000.")
        elif algorithm == "QuadraticSieve":
            reasoning.append("Using Quadratic Sieve with a factor base of small primes.")
        elif algorithm == "GNFS":
            reasoning.append("Using General Number Field Sieve for this large number.")
            
        # Add some specific steps
        if success:
            # Correct factorization steps
            factors = sorted([int(p) ** e for p, e in sympy.factorint(number).items() for _ in range(e)])
            
            if len(factors) == 1:
                reasoning.append(f"After checking divisibility by small primes, I've verified {number} is prime.")
            else:
                for i, factor in enumerate(factors):
                    if i < min(3, len(factors)):  # Show first few factors explicitly
                        reasoning.append(f"Found factor: {factor}")
                        if i == 0:
                            reasoning.append(f"{number} ÷ {factor} = {number // factor}")
                        
                reasoning.append(f"The complete prime factorization is: {' × '.join(map(str, factors))}")
        else:
            # Incorrect reasoning
            if random.random() < 0.3:
                # Computational error
                reasoning.append("Computing polynomial selection...")
                reasoning.append("Error in sieving step, retrying with different parameters.")
            else:
                # Logical error
                reasoning.append("Testing divisibility by small primes.")
                incorrect_factor = random.randint(2, 100)
                reasoning.append(f"Found potential factor: {incorrect_factor}")
                reasoning.append(f"Verifying: {number} ÷ {incorrect_factor} = {number / incorrect_factor}")
        
        # Add verification step
        if len(reasoning) < length:
            if success:
                product = " × ".join(map(str, factors))
                reasoning.append(f"Verification: {product} = {number} ✓")
            else:
                reasoning.append("Need to verify the factorization.")
        
        # Trim or pad reasoning to desired length
        if len(reasoning) > length:
            reasoning = reasoning[:length]
        while len(reasoning) < length:
            reasoning.append("Continuing computation...")
            
        return reasoning
    
    def update_from_reward(self, reward: float) -> None:
        """
        Update model ability based on reward.
        
        Args:
            reward: Reward value (0 to 1)
        """
        # Update general ability
        delta = self.learning_rate * (reward - self.ability)
        self.ability = min(0.95, max(0.1, self.ability + delta))
        
        # Record reward
        self.rewards.append(reward)
        self.iterations += 1

# Simplified version of the factorization grader
class SimpleFactorizationGrader:
    """Simplified grader for factorization attempts."""
    
    def grade(self, attempt: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """
        Grade a factorization attempt.
        
        Args:
            attempt: The factorization attempt
            reference: Reference data with true factors
            
        Returns:
            Overall score between 0 and 1
        """
        # Check correctness (highest weight)
        correctness = self._grade_correctness(
            attempt.get("factors", []), 
            reference["number"], 
            reference["factors"]
        )
        
        # Check algorithm selection
        algorithm_score = self._grade_algorithm(
            attempt.get("algorithm", "Unknown"),
            reference.get("optimal_algorithm"),
            reference["number"]
        )
        
        # Check reasoning
        reasoning_score = self._grade_reasoning(
            attempt.get("reasoning", []),
            attempt.get("algorithm", "Unknown")
        )
        
        # Check efficiency
        efficiency = self._grade_efficiency(
            attempt.get("time_taken", 0.0),
            reference.get("expected_time"),
            reference["number"]
        )
        
        # Weighted average
        weights = {"correctness": 0.5, "algorithm": 0.2, "reasoning": 0.2, "efficiency": 0.1}
        score = (
            weights["correctness"] * correctness +
            weights["algorithm"] * algorithm_score +
            weights["reasoning"] * reasoning_score +
            weights["efficiency"] * efficiency
        )
        
        return score
    
    def _grade_correctness(self, predicted_factors: List[int], number: int, true_factors: List[int]) -> float:
        """Grade the correctness of factorization."""
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
        if not correct_factors:
            return 0.1  # At least the product is correct
        
        return len(correct_factors) / len(reference_sorted)
    
    def _grade_algorithm(self, algorithm: str, optimal_algorithm: str, number: int) -> float:
        """Grade the algorithm selection."""
        if optimal_algorithm and algorithm == optimal_algorithm:
            return 1.0
        
        # Simple scoring based on bit length
        bit_length = number.bit_length()
        
        if bit_length <= 32:
            if algorithm == "TrialDivision":
                return 1.0
            elif algorithm == "WheelFactorization":
                return 0.9
        elif bit_length <= 64:
            if algorithm == "WheelFactorization":
                return 1.0
            elif algorithm in ["TrialDivision", "PollardRho"]:
                return 0.8
        elif bit_length <= 128:
            if algorithm == "PollardRho":
                return 1.0
            elif algorithm in ["WheelFactorization", "ECM"]:
                return 0.7
        elif bit_length <= 512:
            if algorithm == "ECM":
                return 1.0
            elif algorithm in ["PollardRho", "QuadraticSieve"]:
                return 0.7
        else:
            if algorithm in ["QuadraticSieve", "GNFS"]:
                return 1.0
            elif algorithm == "ECM":
                return 0.6
        
        return 0.3  # Default for unsuitable algorithms
    
    def _grade_reasoning(self, reasoning: List[str], algorithm: str) -> float:
        """Grade the reasoning steps."""
        if not reasoning:
            return 0.0
        
        # Check length (not too short, not too verbose)
        if len(reasoning) < 3:
            length_score = 0.3
        elif len(reasoning) > 15:
            length_score = 0.6
        else:
            length_score = 1.0
            
        # Check for algorithm mention
        has_algorithm = any(algorithm in step for step in reasoning)
        
        # Check for verification step
        has_verification = any("verif" in step.lower() for step in reasoning)
        
        # Combined score
        reasoning_score = 0.5 * length_score
        reasoning_score += 0.25 if has_algorithm else 0
        reasoning_score += 0.25 if has_verification else 0
        
        return reasoning_score
    
    def _grade_efficiency(self, time_taken: float, expected_time: float, number: int) -> float:
        """Grade the efficiency of factorization."""
        if not time_taken or time_taken <= 0:
            return 0.0
        
        if expected_time and expected_time > 0:
            return min(expected_time / time_taken, 1.0)
        
        # Normalize by bit length
        bit_length = number.bit_length()
        time_threshold = 0.01 * bit_length * (1 + 0.1 * bit_length)
        
        return max(0, 1 - (time_taken / time_threshold))

# Function to generate a small dataset for testing
def generate_test_dataset(n_samples: int = 40) -> List[Dict[str, Any]]:
    """
    Generate a small test dataset with varying difficulty.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        List of test examples
    """
    dataset = []
    
    # Generate samples with increasing difficulty
    for i in range(n_samples):
        # Scale difficulty from easy to moderate
        tier = min(7, i // 6)  # Tiers 0-7 for 40 samples
        
        if tier <= 1:  # Very easy: 2-digit numbers
            n = random.randint(10, 99)
        elif tier <= 3:  # Easy: 3-4 digit numbers
            n = random.randint(100, 9999)
        elif tier <= 5:  # Moderate: 5-7 digit numbers
            n = random.randint(10000, 9999999)
        else:  # Harder: 8-10 digit numbers
            n = random.randint(10000000, 9999999999)
            
        # Get true factorization
        factors = sorted([int(p) ** e for p, e in sympy.factorint(n).items() for _ in range(e)])
        
        # Determine optimal algorithm
        bit_length = n.bit_length()
        if bit_length <= 32:
            optimal_algorithm = "TrialDivision"
        elif bit_length <= 64:
            optimal_algorithm = "WheelFactorization"
        elif bit_length <= 128:
            optimal_algorithm = "PollardRho"
        elif bit_length <= 512:
            optimal_algorithm = "ECM"
        else:
            optimal_algorithm = "QuadraticSieve"
            
        # Estimate time
        if bit_length <= 32:
            expected_time = 0.001 * bit_length
        elif bit_length <= 64:
            expected_time = 0.005 * bit_length
        elif bit_length <= 128:
            expected_time = 0.01 * bit_length
        else:
            expected_time = 0.05 * bit_length
            
        dataset.append({
            "number": n,
            "factors": factors,
            "optimal_algorithm": optimal_algorithm,
            "expected_time": expected_time,
            "tier": tier
        })
    
    return dataset

# Function to simulate RFT training
def simulate_rft_training(model: SimpleFactorizationModel, dataset: List[Dict[str, Any]], 
                          n_epochs: int = 5, verbose: bool = True) -> Dict[str, List[float]]:
    """
    Simulate RFT training on a dataset.
    
    Args:
        model: The model to train
        dataset: Training dataset
        n_epochs: Number of training epochs
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with training metrics
    """
    grader = SimpleFactorizationGrader()
    metrics = {
        "epoch_rewards": [],
        "correctness": [],
        "algorithm_scores": [],
        "reasoning_scores": [],
        "efficiency_scores": []
    }
    
    for epoch in range(n_epochs):
        epoch_rewards = []
        epoch_correctness = []
        epoch_algorithm = []
        epoch_reasoning = []
        epoch_efficiency = []
        
        if verbose:
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            
        random.shuffle(dataset)  # Shuffle data each epoch
        
        for i, example in enumerate(tqdm(dataset, disable=not verbose)):
            # Get factorization attempt from model
            attempt = model.factorize(example["number"])
            
            # Grade the attempt
            reward = grader.grade(attempt, example)
            
            # Update model based on reward
            model.update_from_reward(reward)
            
            # Record detailed metrics
            correctness = grader._grade_correctness(
                attempt["factors"], example["number"], example["factors"]
            )
            algorithm_score = grader._grade_algorithm(
                attempt["algorithm"], example["optimal_algorithm"], example["number"]
            )
            reasoning_score = grader._grade_reasoning(
                attempt["reasoning"], attempt["algorithm"]
            )
            efficiency_score = grader._grade_efficiency(
                attempt["time_taken"], example["expected_time"], example["number"]
            )
            
            epoch_rewards.append(reward)
            epoch_correctness.append(correctness)
            epoch_algorithm.append(algorithm_score)
            epoch_reasoning.append(reasoning_score)
            epoch_efficiency.append(efficiency_score)
            
        # Record epoch metrics
        metrics["epoch_rewards"].append(np.mean(epoch_rewards))
        metrics["correctness"].append(np.mean(epoch_correctness))
        metrics["algorithm_scores"].append(np.mean(epoch_algorithm))
        metrics["reasoning_scores"].append(np.mean(epoch_reasoning))
        metrics["efficiency_scores"].append(np.mean(epoch_efficiency))
        
        if verbose:
            print(f"  Average reward: {metrics['epoch_rewards'][-1]:.4f}")
            print(f"  Correctness: {metrics['correctness'][-1]:.4f}")
            print(f"  Algorithm selection: {metrics['algorithm_scores'][-1]:.4f}")
            print(f"  Reasoning: {metrics['reasoning_scores'][-1]:.4f}")
            print(f"  Efficiency: {metrics['efficiency_scores'][-1]:.4f}")
            print(f"  Model ability: {model.ability:.4f}")
    
    return metrics

# Function to evaluate model
def evaluate_model(model: SimpleFactorizationModel, test_set: List[Dict[str, Any]], 
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate model on a test set.
    
    Args:
        model: The model to evaluate
        test_set: Test dataset
        verbose: Whether to print verbose output
        
    Returns:
        Evaluation metrics
    """
    grader = SimpleFactorizationGrader()
    results = []
    metrics = {
        "total": len(test_set),
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
        "average_score": 0.0,
        "by_tier": defaultdict(lambda: {"count": 0, "correct": 0, "score": 0.0})
    }
    
    for i, example in enumerate(tqdm(test_set, disable=not verbose)):
        # Get factorization attempt from model
        attempt = model.factorize(example["number"])
        
        # Grade the attempt
        score = grader.grade(attempt, example)
        correctness = grader._grade_correctness(
            attempt["factors"], example["number"], example["factors"]
        )
        
        # Update metrics
        metrics["average_score"] += score
        
        if correctness >= 0.99:  # Fully correct
            metrics["correct"] += 1
        elif correctness >= 0.5:  # Partially correct
            metrics["partially_correct"] += 1
        else:
            metrics["incorrect"] += 1
        
        # Update tier-specific metrics
        tier = example.get("tier", 0)
        metrics["by_tier"][tier]["count"] += 1
        metrics["by_tier"][tier]["score"] += score
        if correctness >= 0.99:
            metrics["by_tier"][tier]["correct"] += 1
        
        # Record result
        result = {
            "number": example["number"],
            "true_factors": example["factors"],
            "predicted_factors": attempt["factors"],
            "algorithm": attempt["algorithm"],
            "score": score,
            "correctness": correctness
        }
        results.append(result)
        
        if verbose and i < 5:  # Show first few examples
            print(f"\nExample {i+1}:")
            print(f"  Number: {example['number']}")
            print(f"  True factors: {example['factors']}")
            print(f"  Predicted: {attempt['factors']}")
            print(f"  Algorithm: {attempt['algorithm']}")
            print(f"  Score: {score:.4f}")
            print(f"  Reasoning:")
            for step in attempt["reasoning"]:
                print(f"    - {step}")
    
    # Calculate averages
    metrics["average_score"] /= metrics["total"]
    
    # Finalize tier metrics
    for tier, data in metrics["by_tier"].items():
        if data["count"] > 0:
            data["score"] /= data["count"]
            data["correct_rate"] = data["correct"] / data["count"]
    
    # Print summary
    if verbose:
        print("\nEvaluation Summary:")
        print(f"Total examples: {metrics['total']}")
        print(f"Correct: {metrics['correct']} ({metrics['correct']/metrics['total']*100:.2f}%)")
        print(f"Partially correct: {metrics['partially_correct']} ({metrics['partially_correct']/metrics['total']*100:.2f}%)")
        print(f"Incorrect: {metrics['incorrect']} ({metrics['incorrect']/metrics['total']*100:.2f}%)")
        print(f"Average score: {metrics['average_score']:.4f}")
        
        print("\nResults by tier:")
        for tier in sorted(metrics["by_tier"].keys()):
            data = metrics["by_tier"][tier]
            print(f"  Tier {tier}: {data['correct_rate']*100:.2f}% correct, avg score: {data['score']:.4f} ({data['count']} examples)")
    
    return {
        "metrics": metrics,
        "results": results
    }

def main():
    """Main function to run the test."""
    print("Simulating Reinforcement Fine-Tuning for Prime Factorization")
    
    # Generate test dataset
    print("\nGenerating dataset...")
    dataset = generate_test_dataset(n_samples=40)
    
    # Split into train/test (30/10)
    random.shuffle(dataset)
    train_dataset = dataset[:30]
    test_dataset = dataset[30:]
    
    print(f"Generated {len(train_dataset)} training and {len(test_dataset)} test examples")
    
    # Create model with medium initial ability
    model = SimpleFactorizationModel(ability_level=0.4, learning_rate=0.1)
    
    # Evaluate before training
    print("\nInitial evaluation:")
    initial_eval = evaluate_model(model, test_dataset, verbose=True)
    
    # Train the model
    print("\nTraining model...")
    metrics = simulate_rft_training(model, train_dataset, n_epochs=5, verbose=True)
    
    # Evaluate after training
    print("\nFinal evaluation:")
    final_eval = evaluate_model(model, test_dataset, verbose=True)
    
    # Show improvement
    improvement = final_eval["metrics"]["average_score"] - initial_eval["metrics"]["average_score"]
    correct_improvement = final_eval["metrics"]["correct"] - initial_eval["metrics"]["correct"]
    
    print("\nImprovement Summary:")
    print(f"Score improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    print(f"Correctness improvement: {correct_improvement} more correct examples")
    
    # Show some detailed results
    print("\nDetailed training metrics:")
    for i, epoch_reward in enumerate(metrics["epoch_rewards"]):
        print(f"Epoch {i+1}: Reward={epoch_reward:.4f}, "
              f"Correctness={metrics['correctness'][i]:.4f}, "
              f"Algorithm={metrics['algorithm_scores'][i]:.4f}, "
              f"Reasoning={metrics['reasoning_scores'][i]:.4f}, "
              f"Efficiency={metrics['efficiency_scores'][i]:.4f}")
    
    print("\nFinal model ability level:", model.ability)
    print("Training iterations:", model.iterations)
    print("Correct factorizations:", model.correct_count)
    
    # Example of final model behavior
    print("\nFinal model behavior examples:")
    for _ in range(3):
        n = random.randint(1000, 999999)
        attempt = model.factorize(n)
        true_factors = sorted([int(p) ** e for p, e in sympy.factorint(n).items() for _ in range(e)])
        
        print(f"\nFactorizing {n}:")
        print(f"True factors: {true_factors}")
        print(f"Model factors: {attempt['factors']}")
        print(f"Algorithm: {attempt['algorithm']}")
        print(f"Confidence: {attempt['confidence']:.4f}")
        print("Reasoning:")
        for step in attempt["reasoning"]:
            print(f"  - {step}")

if __name__ == "__main__":
    main()