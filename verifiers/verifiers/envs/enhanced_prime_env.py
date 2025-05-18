from typing import List, Dict, Any, Optional, Tuple
import random
import json
import math
import sympy
import numpy as np

from datasets import Dataset
import pandas as pd
import pyarrow.parquet as pq

from verifiers import RewardFunc
from verifiers.envs.tool_env import ToolEnv
from verifiers.tools.prime_tools import is_prime, factorize, next_prime, prime_count, verify_factorization
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

class EnhancedPrimeEnv(ToolEnv):
    """Enhanced environment for prime number verification and factorization tasks."""
    
    def __init__(self,
                 data_path: str = None,
                 difficulty_level: str = "mixed",
                 max_steps: int = 10,
                 challenge_modes: List[str] = None,
                 adaptive_difficulty: bool = False,
                 scaffolding: bool = False,
                 quantum_resistant: bool = False,
                 curriculum_learning: bool = False,
                 hint_probability: float = 0.0,
                 **kwargs):
        """
        Initialize the enhanced prime verification environment.
        
        Args:
            data_path: Path to parquet file containing prime number challenges
            difficulty_level: One of "easy", "medium", "hard", "mixed", or "progressive"
            max_steps: Maximum steps allowed for solving a task
            challenge_modes: Specific types of challenges to include
            adaptive_difficulty: Whether to adapt difficulty based on performance
            scaffolding: Whether to provide incremental hints or scaffolding
            quantum_resistant: Whether to include quantum-resistant challenges
            curriculum_learning: Whether to implement curriculum learning
            hint_probability: Probability of providing a hint (0.0-1.0)
        """
        # Default challenge modes if not specified
        self.challenge_modes = challenge_modes or ["primality", "factorization", "verification"]
        
        # Learning features
        self.adaptive_difficulty = adaptive_difficulty
        self.scaffolding = scaffolding
        self.quantum_resistant = quantum_resistant
        self.curriculum_learning = curriculum_learning
        self.hint_probability = hint_probability
        self.performance_history = []
        self.current_difficulty = 0 if curriculum_learning else None
        
        # Tool setup - include additional tools for enhanced version
        tools = [is_prime, factorize, next_prime, prime_count, verify_factorization]
        
        # Create an appropriate system prompt
        if scaffolding:
            prompt_template = self._create_scaffolded_prompt()
        else:
            prompt_template = DEFAULT_TOOL_PROMPT_TEMPLATE
        
        # Load dataset
        dataset = self._load_dataset(data_path, difficulty_level)
        
        # Initialize parent ToolEnv
        super().__init__(
            dataset=dataset,
            tools=tools,
            system_prompt=prompt_template,
            max_steps=max_steps,
            **kwargs
        )
        
        # Store extra settings
        self.difficulty_level = difficulty_level
        self.data_path = data_path
    
    def _create_scaffolded_prompt(self) -> str:
        """Create a custom prompt template with mathematical scaffolding."""
        return """\
You have access to the following tools to help solve mathematical problems:

{tool_descriptions}

For each step of working with prime numbers:

1. <reasoning>
   Break down the problem into clear steps:
   - For primality testing: Consider divisibility rules and trial division
   - For factorization: Start with small primes (2, 3, 5, 7...) and work upward
   - For verification: Check if the product equals the original number
   
   Explain your mathematical thinking carefully.
</reasoning>

2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool

3. You will see the tool's output inside <r> tags

4. Continue until you can give the final answer inside <answer> tags, making sure to:
   - For primality: State clearly if the number is prime or composite
   - For factorization: List all prime factors, with exponents if needed
   - For verification: Confirm if the factorization is correct

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""
    
    def _load_dataset(self, data_path: str, difficulty_level: str) -> Dataset:
        """
        Load and prepare the prime number dataset with enhanced options.
        
        Args:
            data_path: Path to parquet file with prime data
            difficulty_level: Difficulty level to filter by
            
        Returns:
            HuggingFace dataset for the environment
        """
        if data_path is None:
            # Generate a synthetic dataset
            return self._create_enhanced_synthetic_dataset(difficulty_level)
        
        try:
            # Load from parquet file
            table = pq.read_table(data_path)
            df = table.to_pandas()
            
            # Format the dataset based on difficulty
            return self._format_enhanced_dataset(df, difficulty_level)
        except Exception as e:
            print(f"Error loading dataset from {data_path}: {e}")
            print("Falling back to synthetic dataset")
            return self._create_enhanced_synthetic_dataset(difficulty_level)
    
    def _format_enhanced_dataset(self, df, difficulty_level: str) -> Dataset:
        """Format the loaded data with enhanced challenge types."""
        # Process dataframe into proper dictionary format
        examples = []
        
        # Apply difficulty filtering if needed
        if difficulty_level != "mixed" and difficulty_level != "progressive":
            if "tier_id" in df.columns:
                # Filter based on tier if available
                tier_mapping = {
                    "easy": [0, 1, 2], 
                    "medium": [3, 4, 5], 
                    "hard": [6, 7, 8, 9, 10, 11, 12],
                    "extreme": [13, 14, 15]
                }
                tiers = tier_mapping.get(difficulty_level, [])
                if tiers:
                    df = df[df["tier_id"].isin(tiers)]
            elif "bit_length" in df.columns:
                # Filter based on bit length if tiers not available
                bit_mapping = {
                    "easy": (0, 16), 
                    "medium": (17, 32), 
                    "hard": (33, 64),
                    "extreme": (65, 1000)
                }
                bit_range = bit_mapping.get(difficulty_level, (0, 1000))
                if bit_range:
                    df = df[(df["bit_length"] >= bit_range[0]) & (df["bit_length"] <= bit_range[1])]
        
        # Format each row into various challenges
        for _, row in df.iterrows():
            # Extract number and factors
            if "n" in df.columns:
                number = str(row["n"])
            elif "number" in df.columns:
                number = str(row["number"])
            else:
                continue
                
            # Get factors as a string
            if "factors" in df.columns:
                if isinstance(row["factors"], str):
                    factors = row["factors"]
                else:
                    factors = "×".join(map(str, row["factors"]))
            elif "fact" in df.columns:
                factors = row["fact"]
            else:
                continue
            
            # Get difficulty metadata
            tier_id = row.get("tier_id", 0)
            bit_length = row.get("bit_length", len(bin(int(number))[2:]))
            
            # Create challenges based on enabled modes
            challenges = []
            
            # Basic primality check
            if "primality" in self.challenge_modes:
                # Only include primality checks for reasonably sized numbers
                if int(number) < 10**10:  # Limit size for primality checks
                    challenges.append({
                        "input": f"Is {number} a prime number?", 
                        "expected": "is_prime", 
                        "answer": "prime" if len(factors.split("×")) == 1 else "not prime",
                        "tier_id": tier_id,
                        "bit_length": bit_length,
                        "challenge_type": "primality"
                    })
            
            # Basic factorization
            if "factorization" in self.challenge_modes:
                challenges.append({
                    "input": f"Find the prime factorization of {number}.",
                    "expected": "factorize",
                    "answer": factors,
                    "tier_id": tier_id,
                    "bit_length": bit_length,
                    "challenge_type": "factorization"
                })
            
            # Verification challenge
            if "verification" in self.challenge_modes:
                challenges.append({
                    "input": f"Verify if these factors of {number} are correct: {factors.replace('×', ',')}",
                    "expected": "verify",
                    "answer": "correct" if self._verify_product(number, factors) else "incorrect",
                    "tier_id": tier_id,
                    "bit_length": bit_length,
                    "challenge_type": "verification"
                })
            
            # Properties of factors
            if "properties" in self.challenge_modes:
                # Only add for numbers with multiple factors
                if len(factors.split("×")) > 1:
                    challenges.append({
                        "input": f"Find the sum and product of the prime factors of {number}.",
                        "expected": "properties",
                        "answer": self._factor_properties(factors),
                        "tier_id": tier_id, 
                        "bit_length": bit_length,
                        "challenge_type": "properties"
                    })
            
            # Pattern recognition
            if "pattern" in self.challenge_modes:
                if len(factors.split("×")) > 2:
                    challenges.append({
                        "input": f"Identify patterns in the prime factorization of {number}.",
                        "expected": "pattern",
                        "answer": self._factor_pattern(factors),
                        "tier_id": tier_id,
                        "bit_length": bit_length,
                        "challenge_type": "pattern"
                    })
            
            # Add scaffolding if enabled
            if self.scaffolding:
                for challenge in challenges:
                    hints = self._generate_hints(number, factors, challenge["challenge_type"])
                    challenge["hints"] = hints
            
            # Add hints
            if self.hint_probability > 0:
                for challenge in challenges:
                    if random.random() < self.hint_probability:
                        hint = self._generate_single_hint(number, factors, challenge["challenge_type"])
                        challenge["input"] = f"{challenge['input']}\n\nHint: {hint}"
            
            # Add all challenges
            examples.extend(challenges)
        
        # Convert to HuggingFace Dataset
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    def _create_enhanced_synthetic_dataset(self, difficulty_level: str) -> Dataset:
        """Create an enhanced synthetic dataset with various challenge types."""
        examples = []
        
        # Define difficulty parameters
        if difficulty_level == "easy":
            max_bits = 8  # Numbers up to 255
            max_factors = 2
        elif difficulty_level == "medium":
            max_bits = 16  # Numbers up to 65535
            max_factors = 3
        elif difficulty_level == "hard":
            max_bits = 32  # Numbers up to 4 billion
            max_factors = 5
        elif difficulty_level == "extreme":
            max_bits = 64  # Very large numbers
            max_factors = 8
        elif difficulty_level == "progressive":
            # For curriculum learning
            difficulty_levels = ["easy", "medium", "hard"]
            datasets = [self._create_enhanced_synthetic_dataset(level) for level in difficulty_levels]
            # Concatenate datasets
            combined_data = []
            for i, ds in enumerate(datasets):
                df = ds.to_pandas()
                df["curriculum_level"] = i
                combined_data.append(df)
            return Dataset.from_pandas(pd.concat(combined_data))
        else:  # mixed
            difficulties = ["easy", "medium", "hard"]
            return self._create_enhanced_synthetic_dataset(random.choice(difficulties))
        
        # Sample counts per challenge type
        challenge_counts = {
            "primality": {"simple": 10, "complex": 5},
            "factorization": {"simple": 15, "complex": 10},
            "verification": {"correct": 10, "incorrect": 5},
            "properties": 10,
            "pattern": 5
        }
        
        # Quantum-resistant challenges
        if self.quantum_resistant:
            quantum_samples = self._generate_quantum_resistant_samples(5, max_bits)
            examples.extend(quantum_samples)
        
        # Generate prime challenges
        if "primality" in self.challenge_modes:
            # Simple primality checks
            for _ in range(challenge_counts["primality"]["simple"]):
                is_prime_example = random.choice([True, False])
                if is_prime_example:
                    # Generate a prime number
                    prime = self._generate_prime(random.randint(2, max_bits))
                    examples.append({
                        "input": f"Is {prime} a prime number?",
                        "expected": "is_prime",
                        "answer": "prime",
                        "bit_length": prime.bit_length(),
                        "challenge_type": "primality"
                    })
                else:
                    # Generate a composite number
                    num_factors = random.randint(2, max_factors)
                    factors = [self._generate_prime(max_bits // num_factors) for _ in range(num_factors)]
                    composite = math.prod(factors)
                    examples.append({
                        "input": f"Is {composite} a prime number?",
                        "expected": "is_prime",
                        "answer": "not prime",
                        "bit_length": composite.bit_length(),
                        "challenge_type": "primality"
                    })
        
        # Generate factorization challenges
        if "factorization" in self.challenge_modes:
            for _ in range(challenge_counts["factorization"]["simple"] + challenge_counts["factorization"]["complex"]):
                # Determine complexity
                is_complex = len(examples) < challenge_counts["factorization"]["simple"]
                
                # Determine number of factors
                if is_complex:
                    num_factors = random.randint(3, max_factors)
                else:
                    num_factors = random.randint(2, 3)
                
                # Generate factors and product
                factors = []
                for _ in range(num_factors):
                    factor_bits = random.randint(2, max_bits // num_factors)
                    factor = self._generate_prime(factor_bits)
                    factors.append(factor)
                
                product = math.prod(factors)
                
                # Create factorization challenge
                examples.append({
                    "input": f"Find the prime factorization of {product}.",
                    "expected": "factorize",
                    "answer": "×".join(map(str, factors)),
                    "bit_length": product.bit_length(),
                    "challenge_type": "factorization"
                })
        
        # Generate verification challenges
        if "verification" in self.challenge_modes:
            # Correct verifications
            for _ in range(challenge_counts["verification"]["correct"]):
                num_factors = random.randint(2, max_factors)
                factors = []
                for _ in range(num_factors):
                    factor_bits = random.randint(2, max_bits // num_factors)
                    factor = self._generate_prime(factor_bits)
                    factors.append(factor)
                
                product = math.prod(factors)
                
                examples.append({
                    "input": f"Verify if these factors of {product} are correct: {','.join(map(str, factors))}",
                    "expected": "verify",
                    "answer": "correct",
                    "bit_length": product.bit_length(),
                    "challenge_type": "verification"
                })
            
            # Incorrect verifications
            for _ in range(challenge_counts["verification"]["incorrect"]):
                num_factors = random.randint(2, max_factors)
                factors = []
                for _ in range(num_factors):
                    factor_bits = random.randint(2, max_bits // num_factors)
                    factor = self._generate_prime(factor_bits)
                    factors.append(factor)
                
                product = math.prod(factors)
                
                # Create incorrect factors
                incorrect_factors = factors.copy()
                mod_index = random.randint(0, len(incorrect_factors) - 1)
                
                # Either change a factor or add/remove one
                mod_type = random.choice(["change", "add", "remove"])
                if mod_type == "change":
                    incorrect_factors[mod_index] += 2
                elif mod_type == "add":
                    incorrect_factors.append(self._generate_prime(max_bits // num_factors))
                elif mod_type == "remove" and len(incorrect_factors) > 1:
                    incorrect_factors.pop(mod_index)
                
                examples.append({
                    "input": f"Verify if these factors of {product} are correct: {','.join(map(str, incorrect_factors))}",
                    "expected": "verify",
                    "answer": "incorrect",
                    "bit_length": product.bit_length(),
                    "challenge_type": "verification"
                })
        
        # Generate properties challenges
        if "properties" in self.challenge_modes and challenge_counts["properties"] > 0:
            for _ in range(challenge_counts["properties"]):
                num_factors = random.randint(2, max_factors)
                factors = []
                for _ in range(num_factors):
                    factor_bits = random.randint(2, max_bits // num_factors)
                    factor = self._generate_prime(factor_bits)
                    factors.append(factor)
                
                product = math.prod(factors)
                
                examples.append({
                    "input": f"Find the sum and product of the prime factors of {product}.",
                    "expected": "properties",
                    "answer": self._factor_properties("×".join(map(str, factors))),
                    "bit_length": product.bit_length(),
                    "challenge_type": "properties"
                })
        
        # Add scaffolding if enabled
        if self.scaffolding:
            for example in examples:
                number = example["input"].split()[example["input"].split().index("of") + 1].strip(".").strip(",").strip()
                if example["challenge_type"] in ["factorization", "verification", "properties"]:
                    factors = example.get("answer", "").replace("×", ",")
                    hints = self._generate_hints(number, factors, example["challenge_type"])
                    example["hints"] = hints
        
        # Add hints with specified probability
        if self.hint_probability > 0:
            for example in examples:
                if random.random() < self.hint_probability:
                    number = example["input"].split()[example["input"].split().index("of") + 1].strip(".").strip(",").strip()
                    if ":" in number:
                        number = number.split(":")[0]
                    factors = example.get("answer", "").replace("×", ",")
                    hint = self._generate_single_hint(number, factors, example["challenge_type"])
                    example["input"] = f"{example['input']}\n\nHint: {hint}"
        
        # Convert to HuggingFace Dataset
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    def _generate_quantum_resistant_samples(self, count: int, max_bits: int) -> List[Dict]:
        """Generate quantum-resistant factorization challenges."""
        samples = []
        
        for _ in range(count):
            # Generate multiple factors instead of just two
            num_factors = random.randint(3, 5)
            factors = []
            
            # Each factor should be large but not too large
            factor_bits = max(16, max_bits // num_factors)
            
            for _ in range(num_factors):
                factor = self._generate_prime(factor_bits)
                factors.append(factor)
            
            product = math.prod(factors)
            
            # Create a factorization challenge
            samples.append({
                "input": f"Find the prime factorization of {product}. This is a quantum-resistant challenge with multiple prime factors.",
                "expected": "factorize",
                "answer": "×".join(map(str, factors)),
                "bit_length": product.bit_length(),
                "challenge_type": "quantum_resistant"
            })
        
        return samples
    
    def _generate_hints(self, number: str, factors: str, challenge_type: str) -> List[str]:
        """Generate a sequence of hints for a given challenge."""
        hints = []
        
        try:
            num = int(number)
            
            if challenge_type == "primality":
                # Primality hints
                if num % 2 == 0:
                    hints.append("The number is even.")
                else:
                    hints.append("The number is odd.")
                
                digit_sum = sum(int(d) for d in str(num))
                if digit_sum % 3 == 0:
                    hints.append(f"The sum of digits ({digit_sum}) is divisible by 3.")
                
                if str(num)[-1] in "05":
                    hints.append("The number ends in 0 or 5, making it divisible by 5.")
            
            elif challenge_type == "factorization":
                # Factorization hints
                if num % 2 == 0:
                    hints.append("The number is even, so 2 is a factor.")
                
                if num % 3 == 0:
                    hints.append("The number is divisible by 3.")
                
                sqrt_n = int(math.sqrt(num))
                hints.append(f"When checking for factors, you only need to check up to √{num} ≈ {sqrt_n}.")
                
                # Find a small factor to suggest
                for i in range(2, min(100, sqrt_n + 1)):
                    if num % i == 0:
                        hints.append(f"The number is divisible by {i}.")
                        hints.append(f"After dividing by {i}, you get {num // i}.")
                        break
            
            elif challenge_type == "verification":
                # Verification hints
                factor_list = [int(f.strip()) for f in factors.split(',')]
                
                if len(factor_list) > 0:
                    product = math.prod(factor_list)
                    hints.append(f"To verify the factorization, multiply all factors together and check if it equals {num}.")
                    
                    if product != num:
                        hints.append(f"The product of the given factors is {product}, which is different from {num}.")
                        
                        if product < num:
                            hints.append("The product is too small, suggesting a missing factor.")
                        else:
                            hints.append("The product is too large, suggesting an incorrect factor.")
            
            elif challenge_type == "properties":
                # Properties hints
                factor_list = [int(f.strip()) for f in factors.split(',')]
                
                if len(factor_list) > 0:
                    factor_sum = sum(factor_list)
                    hints.append(f"Calculate the sum of all prime factors.")
                    hints.append(f"Calculate the product of all prime factors, which should equal the original number.")
            
        except:
            # Default hint if parsing fails
            hints.append("Start by determining if the number has any small prime factors like 2, 3, 5, or 7.")
        
        return hints
    
    def _generate_single_hint(self, number: str, factors: str, challenge_type: str) -> str:
        """Generate a single hint for a challenge."""
        hints = self._generate_hints(number, factors, challenge_type)
        if hints:
            return random.choice(hints)
        return "Start by checking divisibility by small primes."
    
    def _is_prime(self, n: int) -> bool:
        """Simple primality test."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a prime number of approximately the given bit length."""
        # For small primes, use sympy to generate primes quickly
        if bits <= 32:
            lower = 2**(bits-1)
            upper = 2**bits - 1
            return sympy.randprime(lower, upper)
        
        # For small bit sizes, use predefined primes
        if bits <= 8:
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                           101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
            return random.choice(small_primes)
        
        # For larger primes, use a simple generation approach
        attempts = 0
        while attempts < 100:  # Limit attempts
            # Generate random odd number with approximately the right bit length
            n = random.randint(2**(bits-1), 2**bits - 1) | 1  # Make sure it's odd
            if self._is_prime(n):
                return n
            attempts += 1
        
        # Fallback if we fail to find a prime
        return sympy.randprime(2**(bits-1), 2**bits - 1)
    
    def _verify_product(self, n, factors_str):
        """Verify if the product of factors equals n."""
        try:
            n = int(n)
            factors = [int(f.strip()) for f in factors_str.replace('×', ',').split(',')]
            return math.prod(factors) == n
        except:
            return False
    
    def _factor_properties(self, factors_str):
        """Calculate properties of a set of factors."""
        try:
            factors = [int(f.strip()) for f in factors_str.replace('×', ',').split(',')]
            factor_sum = sum(factors)
            factor_product = math.prod(factors)
            unique_factors = sorted(set(factors))
            return f"Sum: {factor_sum}, Product: {factor_product}, Unique factors: {len(unique_factors)}"
        except:
            return "Unable to calculate properties"
    
    def _factor_pattern(self, factors_str):
        """Identify patterns in the factorization."""
        try:
            factors = [int(f.strip()) for f in factors_str.replace('×', ',').split(',')]
            
            # Count occurrences of each factor
            counts = {}
            for f in factors:
                counts[f] = counts.get(f, 0) + 1
            
            # Check for patterns
            patterns = []
            
            # Check for repeated factors
            repeated = [f"{p}^{c}" for p, c in counts.items() if c > 1]
            if repeated:
                patterns.append(f"Repeated factors: {', '.join(repeated)}")
            
            # Check for consecutive primes
            sorted_unique = sorted(counts.keys())
            consecutive = []
            for i in range(len(sorted_unique) - 1):
                if sorted_unique[i+1] - sorted_unique[i] == 2:
                    consecutive.append(f"{sorted_unique[i]} and {sorted_unique[i+1]}")
            
            if consecutive:
                patterns.append(f"Twin primes: {', '.join(consecutive)}")
            
            # Check for arithmetic progression
            if len(sorted_unique) >= 3:
                diffs = [sorted_unique[i+1] - sorted_unique[i] for i in range(len(sorted_unique)-1)]
                if len(set(diffs)) == 1:
                    patterns.append(f"Arithmetic progression with common difference {diffs[0]}")
            
            if patterns:
                return "; ".join(patterns)
            else:
                return "No specific patterns found in the factorization"
        except:
            return "Unable to analyze patterns"
    
    def update_difficulty(self, reward: float) -> None:
        """
        Update the current difficulty level based on performance.
        
        Args:
            reward: The reward from the last episode
        """
        if not self.adaptive_difficulty and not self.curriculum_learning:
            return
        
        # Store the reward
        self.performance_history.append(reward)
        
        # Only update after collecting enough data
        if len(self.performance_history) < 5:
            return
        
        # Calculate recent average performance
        recent_avg = sum(self.performance_history[-5:]) / 5
        
        if self.curriculum_learning:
            # Update difficulty based on curriculum progression
            if recent_avg > 0.8 and self.current_difficulty < 3:  # Progress to next level
                self.current_difficulty += 1
                print(f"Curriculum advanced to level {self.current_difficulty}")
                
                # Refresh the dataset with the new difficulty
                tiers = {0: "easy", 1: "medium", 2: "hard", 3: "extreme"}
                level = tiers.get(self.current_difficulty, "mixed")
                
                self.dataset = self._load_dataset(self.data_path, level)
                self.performance_history = []  # Reset history
                
        elif self.adaptive_difficulty:
            # Adaptive difficulty adjustment
            if recent_avg > 0.9:  # Too easy
                self._increase_difficulty()
            elif recent_avg < 0.3:  # Too hard
                self._decrease_difficulty()
    
    def _increase_difficulty(self) -> None:
        """Increase the difficulty of the environment."""
        print("Increasing task difficulty")
        
        # Add more challenging modes
        if "properties" not in self.challenge_modes:
            self.challenge_modes.append("properties")
        if "pattern" not in self.challenge_modes:
            self.challenge_modes.append("pattern")
        
        # Reduce scaffolding and hints
        self.hint_probability = max(0.0, self.hint_probability - 0.2)
        
        # Refresh dataset with more challenging examples
        self.dataset = self._create_enhanced_synthetic_dataset("hard")
    
    def _decrease_difficulty(self) -> None:
        """Decrease the difficulty of the environment."""
        print("Decreasing task difficulty")
        
        # Remove more challenging modes
        self.challenge_modes = [mode for mode in self.challenge_modes 
                               if mode in ["primality", "factorization", "verification"]]
        
        # Increase scaffolding and hints
        self.hint_probability = min(0.8, self.hint_probability + 0.2)
        
        # Refresh dataset with easier examples
        self.dataset = self._create_enhanced_synthetic_dataset("easy")
    
    async def run_sample(self, sample, max_steps=None, with_metrics=False):
        """
        Run a sample through the environment with enhanced features.
        
        Args:
            sample: The sample to run
            max_steps: Maximum number of steps to run
            with_metrics: Whether to return metrics
            
        Returns:
            Dictionary with results and metrics
        """
        # Add hints if applicable
        if self.scaffolding and "hints" in sample:
            input_text = sample["input"]
            
            # Choose one hint to add
            if self.hint_probability > 0 and random.random() < self.hint_probability:
                hint = random.choice(sample["hints"]) if sample["hints"] else None
                if hint:
                    input_text = f"{input_text}\n\nHint: {hint}"
            
            sample["input"] = input_text
        
        # Run the sample using the parent class method
        result = await super().run_sample(sample, max_steps, with_metrics)
        
        # Update difficulty based on performance
        if with_metrics and "metrics" in result:
            reward = result["metrics"].get("factorization_accuracy", 0.0)
            self.update_difficulty(reward)
        
        return result