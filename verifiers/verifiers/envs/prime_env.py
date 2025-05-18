from typing import List, Dict, Any
import random
import json
import math

from datasets import Dataset
import pandas as pd
import pyarrow.parquet as pq

from verifiers import RewardFunc
from verifiers.envs.tool_env import ToolEnv
from verifiers.tools.prime_tools import is_prime, factorize, next_prime, prime_count, verify_factorization
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

class PrimeEnv(ToolEnv):
    """Environment for prime number verification and factorization tasks."""
    
    def __init__(self,
                 data_path: str = None,
                 difficulty_level: str = "mixed",
                 max_steps: int = 10,
                 **kwargs):
        """
        Initialize the prime verification environment.
        
        Args:
            data_path: Path to parquet file containing prime number challenges
            difficulty_level: One of "easy", "medium", "hard", or "mixed"
            max_steps: Maximum steps allowed for solving a task
        """
        # Tool setup
        tools = [is_prime, factorize, next_prime, prime_count, verify_factorization]
        
        # Load dataset
        dataset = self._load_dataset(data_path, difficulty_level)
        
        # Initialize parent ToolEnv
        super().__init__(
            dataset=dataset,
            tools=tools,
            system_prompt=DEFAULT_TOOL_PROMPT_TEMPLATE,
            max_steps=max_steps,
            **kwargs
        )
    
    def _load_dataset(self, data_path: str, difficulty_level: str) -> Dataset:
        """
        Load and prepare the prime number dataset.
        
        Args:
            data_path: Path to parquet file with prime data
            difficulty_level: Difficulty level to filter by
            
        Returns:
            HuggingFace dataset for the environment
        """
        if data_path is None:
            # Generate a simple synthetic dataset if no data path is provided
            return self._create_synthetic_dataset(difficulty_level)
        
        try:
            # Load from parquet file
            table = pq.read_table(data_path)
            df = table.to_pandas()
            
            # Format the dataset based on difficulty
            return self._format_dataset(df, difficulty_level)
        except Exception as e:
            print(f"Error loading dataset from {data_path}: {e}")
            print("Falling back to synthetic dataset")
            return self._create_synthetic_dataset(difficulty_level)
    
    def _format_dataset(self, df, difficulty_level: str) -> Dataset:
        """Format the loaded data into the expected dataset structure."""
        # Process dataframe into proper dictionary format
        examples = []
        
        # Apply difficulty filtering if needed
        if difficulty_level != "mixed":
            if "tier_id" in df.columns:
                # Filter based on tier if available
                tier_mapping = {"easy": [0, 1, 2], "medium": [3, 4, 5], "hard": [6, 7, 8, 9, 10, 11, 12]}
                tiers = tier_mapping.get(difficulty_level, [])
                if tiers:
                    df = df[df["tier_id"].isin(tiers)]
            elif "bit_length" in df.columns:
                # Filter based on bit length if tiers not available
                bit_mapping = {"easy": (0, 16), "medium": (17, 32), "hard": (33, 1000)}
                bit_range = bit_mapping.get(difficulty_level, (0, 1000))
                if bit_range:
                    df = df[(df["bit_length"] >= bit_range[0]) & (df["bit_length"] <= bit_range[1])]
        
        # Format each row into a challenge
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
            
            # Create different types of challenges
            challenge_types = [
                {"input": f"Is {number} a prime number?", "expected": "is_prime", "answer": "prime" if len(factors.split("×")) == 1 else "not prime"},
                {"input": f"Find the prime factorization of {number}.", "expected": "factorize", "answer": factors},
                {"input": f"Verify if these factors of {number} are correct: {factors.replace('×', ',')}", "expected": "verify", "answer": "correct" if self._verify_product(number, factors) else "incorrect"}
            ]
            
            # Add each challenge type to examples
            examples.append(random.choice(challenge_types))
        
        # Convert to HuggingFace Dataset
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    def _create_synthetic_dataset(self, difficulty_level: str) -> Dataset:
        """Create a synthetic dataset for testing."""
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
        else:  # mixed
            difficulties = ["easy", "medium", "hard"]
            return self._create_synthetic_dataset(random.choice(difficulties))
        
        # Generate prime challenges
        prime_counts = {"small": 10, "large": 5}
        for size, count in prime_counts.items():
            # Generate small and large primes
            if size == "small":
                primes = [random.randint(2, 100) for _ in range(count)]
                primes = [p for p in primes if self._is_prime(p)]
            else:
                bit_size = random.randint(max_bits // 2, max_bits)
                primes = [self._generate_prime(bit_size) for _ in range(count)]
            
            for prime in primes:
                examples.append({
                    "input": f"Is {prime} a prime number?",
                    "expected": "is_prime",
                    "answer": "prime"
                })
        
        # Generate factorization challenges
        for _ in range(20):
            # Random number of factors between 2 and max_factors
            num_factors = random.randint(2, max_factors)
            
            # Generate random prime factors
            factors = []
            for _ in range(num_factors):
                bit_size = random.randint(2, max_bits // num_factors)
                factor = self._generate_prime(bit_size)
                factors.append(factor)
            
            # Calculate the product
            product = math.prod(factors)
            
            # Create factorization challenge
            examples.append({
                "input": f"Find the prime factorization of {product}.",
                "expected": "factorize",
                "answer": "×".join(map(str, factors))
            })
            
            # Create verification challenge
            examples.append({
                "input": f"Verify if these factors of {product} are correct: {','.join(map(str, factors))}",
                "expected": "verify",
                "answer": "correct"
            })
        
        # Add some incorrect verification challenges
        for _ in range(5):
            # Generate a random number and its factorization
            num_factors = random.randint(2, max_factors)
            factors = []
            for _ in range(num_factors):
                bit_size = random.randint(2, max_bits // num_factors)
                factor = self._generate_prime(bit_size)
                factors.append(factor)
            
            product = math.prod(factors)
            
            # Modify one factor to make it incorrect
            incorrect_factors = factors.copy()
            idx = random.randint(0, len(incorrect_factors) - 1)
            incorrect_factors[idx] = incorrect_factors[idx] + 2  # Make it non-prime
            
            examples.append({
                "input": f"Verify if these factors of {product} are correct: {','.join(map(str, incorrect_factors))}",
                "expected": "verify",
                "answer": "incorrect"
            })
        
        # Convert to HuggingFace Dataset
        return Dataset.from_pandas(pd.DataFrame(examples))
    
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
        # Simple implementation - in production you'd use a better algorithm
        # For small primes, just use a predefined list
        if bits <= 8:
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                           101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
            return random.choice(small_primes)
        
        # For larger primes, use a simple generation approach
        while True:
            # Generate random odd number with approximately the right bit length
            n = random.randint(2**(bits-1), 2**bits - 1) | 1  # Make sure it's odd
            if self._is_prime(n):
                return n
    
    def _verify_product(self, n, factors_str):
        """Verify if the product of factors equals n."""
        try:
            n = int(n)
            factors = [int(f.strip()) for f in factors_str.replace('×', ',').split(',')]
            return math.prod(factors) == n
        except:
            return False