from typing import List, Dict, Any, Optional, Tuple, Union
import random
import json
import math
import sympy
import numpy as np
import re
from collections import defaultdict

from datasets import Dataset
import pandas as pd
import pyarrow.parquet as pq

from verifiers import RewardFunc
from verifiers.envs.tool_env import ToolEnv
from verifiers.tools.advanced_prime_tools import (
    advanced_is_prime, advanced_factorize, prime_gaps, prime_density,
    twin_primes, prime_factorization_tree, number_theory_analysis, carmichael_check
)
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

class QuantumPrimeEnv(ToolEnv):
    """Advanced environment for quantum-resistant prime number verification and deep number theory."""
    
    def __init__(self,
                 data_path: str = None,
                 difficulty_level: str = "mixed",
                 max_steps: int = 15,
                 challenge_modes: List[str] = None,
                 advanced_features: List[str] = None,
                 cognitive_architecture: str = "none",
                 adaptive_system: bool = False,
                 quantum_resistant_mode: bool = True,
                 math_exploration: bool = False,
                 metacognitive: bool = False,
                 theoretical_challenges: bool = False,
                 step_based_rewards: bool = False,
                 **kwargs):
        """
        Initialize the advanced quantum-resistant prime environment.
        
        Args:
            data_path: Path to parquet file containing prime number challenges
            difficulty_level: Difficulty tier (foundational, intermediate, advanced, 
                             theoretical, quantum, or mixed)
            max_steps: Maximum steps allowed for solving a task
            challenge_modes: Types of challenges to include
            advanced_features: Additional features to enable
            cognitive_architecture: Type of cognitive structure to use
                - "none": Standard interaction
                - "hierarchical": Two-level reasoning system
                - "metacognitive": Includes reflection stages
                - "multi-agent": Simulates multiple expert agents
            adaptive_system: Whether to adapt dynamically to model capability
            quantum_resistant_mode: Include quantum-resistant challenges
            math_exploration: Enable open-ended mathematical exploration
            metacognitive: Enable metacognitive prompting
            theoretical_challenges: Include theoretical number theory challenges
            step_based_rewards: Provide rewards based on step quality
        """
        # Default challenge modes if not specified
        self.challenge_modes = challenge_modes or [
            "primality", "factorization", "verification", 
            "number_theory", "conjectures", "proof_sketching"
        ]
        
        # Default advanced features if not specified
        self.advanced_features = advanced_features or [
            "step_by_step", "algorithm_selection", "complexity_analysis"
        ]
        
        # Learning features
        self.cognitive_architecture = cognitive_architecture
        self.adaptive_system = adaptive_system
        self.quantum_resistant_mode = quantum_resistant_mode
        self.math_exploration = math_exploration
        self.metacognitive = metacognitive
        self.theoretical_challenges = theoretical_challenges
        self.step_based_rewards = step_based_rewards
        
        # Performance tracking
        self.performance_history = []
        self.strategy_effectiveness = defaultdict(list)
        self.current_difficulty_level = 0
        self.exploration_depth = 0
        
        # Tool setup - include all advanced tools
        tools = [
            advanced_is_prime, advanced_factorize, prime_gaps, prime_density,
            twin_primes, prime_factorization_tree, number_theory_analysis, carmichael_check
        ]
        
        # Create an appropriate system prompt based on cognitive architecture
        if cognitive_architecture == "hierarchical":
            prompt_template = self._create_hierarchical_prompt()
        elif cognitive_architecture == "metacognitive":
            prompt_template = self._create_metacognitive_prompt()
        elif cognitive_architecture == "multi-agent":
            prompt_template = self._create_multi_agent_prompt()
        else:
            prompt_template = self._create_advanced_prompt()
        
        # Load dataset
        dataset = self._load_advanced_dataset(data_path, difficulty_level)
        
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
        
        # Advanced metadata
        self.current_cognitive_state = {
            "exploration_phase": "initial",
            "depth": 0,
            "tools_used": [],
            "key_insights": [],
            "strategies_attempted": [],
            "verification_level": 0
        }
    
    def _create_advanced_prompt(self) -> str:
        """Create a prompt template for advanced mathematical reasoning."""
        return """\
You are an advanced mathematical reasoning system with expertise in number theory and cryptography. 
You have access to the following tools to solve complex mathematical problems:

{tool_descriptions}

Your task is to solve advanced number theory problems, particularly focusing on prime numbers and factorization.
For quantum-resistant challenges, remember that large numbers with multiple prime factors present special difficulties.

Approach problems with these techniques:
1. Analyze the mathematical structure of the problem
2. Consider multiple algorithms and select the most appropriate one
3. Think step-by-step through the solution process
4. Track your progress and verify results
5. When approaching very large numbers, look for mathematical properties that could simplify the problem

For each step:
1. <reasoning>
   Document your mathematical reasoning, algorithm selection, complexity analysis, and verification steps.
   For primality testing, consider the appropriate algorithm based on number size.
   For factorization, apply number theory insights before brute force approaches.
   For quantum-resistant problems, analyze the problem structure carefully.
</reasoning>

2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool

3. You will see the tool's output inside <r> tags

4. Continue until you can give the final answer inside <answer> tags, making sure to:
   - Include the complete reasoning chain
   - Verify your answer
   - Provide complexity analysis and confidence level

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""
    
    def _create_hierarchical_prompt(self) -> str:
        """Create a prompt template for hierarchical mathematical reasoning."""
        return """\
You are a hierarchical mathematical reasoning system with both high-level strategic thinking and detailed computational capabilities.
You have access to the following tools to solve complex number theory problems:

{tool_descriptions}

Your reasoning will proceed in two distinct phases for each problem:

PHASE 1 - Strategic Level:
- Analyze the mathematical structure of the problem
- Consider the theoretical complexity of different approaches
- Select appropriate algorithms based on problem characteristics
- Decompose complex problems into simpler subproblems
- Make high-level strategic decisions about solution paths

PHASE 2 - Tactical Level:
- Implement the strategies identified in Phase 1
- Perform detailed calculations and algorithm execution
- Track progress and intermediate results
- Validate results and check for errors
- Handle edge cases and exceptional conditions

For each step:
1. <reasoning>
   Begin with "STRATEGIC:" to outline your high-level approach, algorithm selection, and problem decomposition.
   Follow with "TACTICAL:" to document detailed calculations, tool usage, and verification.
</reasoning>

2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool

3. You will see the tool's output inside <r> tags

4. Continue until you can give the final answer inside <answer> tags, structured as:
   - "Approach": Strategic summary of the approach used
   - "Solution": The detailed answer with justification
   - "Verification": How you verified the correctness of your answer
   - "Complexity": Analysis of computational complexity

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""
    
    def _create_metacognitive_prompt(self) -> str:
        """Create a prompt template with metacognitive capabilities."""
        return """\
You are a metacognitive mathematical system with expertise in number theory, cryptography, and computational complexity.
You have access to the following tools to solve complex mathematical problems:

{tool_descriptions}

Your reasoning includes both object-level mathematical thinking and meta-level reflection on your own problem-solving process.
This metacognitive approach allows you to monitor, evaluate, and regulate your mathematical reasoning.

For each step of your reasoning:

1. <reasoning>
   Use these explicitly labeled reasoning components:
   
   [MATHEMATICAL ANALYSIS]
   Analyze the mathematical structure and properties of the problem.
   Consider relevant theorems, algorithms, and number theory principles.
   
   [STRATEGY SELECTION]
   Identify and evaluate possible solution approaches.
   Select the most promising strategy based on problem characteristics.
   
   [EXECUTION]
   Apply the selected strategy with careful calculations.
   
   [MONITORING]
   Track your progress and evaluate whether your approach is working.
   Identify any errors, inefficiencies, or roadblocks.
   
   [REFLECTION]
   Assess the effectiveness of your current approach.
   Consider alternative strategies if needed.
   Identify what you've learned that can inform your next steps.
</reasoning>

2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool

3. You will see the tool's output inside <r> tags

4. Continue until you can give the final answer inside <answer> tags, with these sections:
   - "Solution": The mathematical answer
   - "Confidence": Your assessment of solution reliability (high/medium/low) with justification
   - "Process Reflection": What worked well and what could be improved in your approach
   - "Knowledge Gaps": Any areas where additional information would have helped

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""
    
    def _create_multi_agent_prompt(self) -> str:
        """Create a prompt template that simulates multiple expert agents."""
        return """\
You are a collaborative mathematical reasoning system that simulates multiple specialized experts working together to solve complex number theory problems.
You have access to the following tools:

{tool_descriptions}

Your reasoning team consists of these specialized experts:

THEORIST: Expert in abstract mathematical concepts, number theory principles, and theoretical foundations
ALGORITHMIST: Expert in selecting and applying efficient computational algorithms and complexity analysis
IMPLEMENTER: Expert in precise calculations, tool usage, and detailed execution
CRITIC: Expert in verification, edge cases, testing assumptions, and finding counterexamples

For each step:
1. <reasoning>
   [THEORIST]: Analyze mathematical structure, relevant theorems, and theoretical implications
   
   [ALGORITHMIST]: Evaluate algorithmic approaches, computational complexity, and efficiency considerations
   
   [IMPLEMENTER]: Execute calculations, use tools, track intermediate results
   
   [CRITIC]: Verify results, check for errors, test edge cases, question assumptions
   
   [CONSENSUS]: Synthesize insights from all experts to determine next steps
</reasoning>

2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool

3. You will see the tool's output inside <r> tags

4. Continue until your team reaches a consensus for the final answer inside <answer> tags, including:
   - Theoretical foundation
   - Algorithm selection justification
   - Implementation details
   - Verification and robustness analysis

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""
    
    def _load_advanced_dataset(self, data_path: str, difficulty_level: str) -> Dataset:
        """
        Load and prepare an advanced prime number dataset.
        
        Args:
            data_path: Path to parquet file with prime data
            difficulty_level: Difficulty level to filter by
            
        Returns:
            HuggingFace dataset for the environment
        """
        if data_path is None:
            # Generate a synthetic dataset
            return self._create_advanced_synthetic_dataset(difficulty_level)
        
        try:
            # Load from parquet file
            table = pq.read_table(data_path)
            df = table.to_pandas()
            
            # Format the dataset based on difficulty
            return self._format_advanced_dataset(df, difficulty_level)
        except Exception as e:
            print(f"Error loading dataset from {data_path}: {e}")
            print("Falling back to synthetic dataset")
            return self._create_advanced_synthetic_dataset(difficulty_level)
    
    def _format_advanced_dataset(self, df, difficulty_level: str) -> Dataset:
        """Format the loaded data with advanced challenge types."""
        # Process dataframe into proper dictionary format
        examples = []
        
        # Apply difficulty filtering
        if difficulty_level != "mixed":
            # Map difficulty levels to tier ranges
            tier_mapping = {
                "foundational": [0, 1, 2], 
                "intermediate": [3, 4, 5, 6],
                "advanced": [7, 8, 9, 10, 11],
                "theoretical": [12, 13],
                "quantum": [14, 15]
            }
            
            if difficulty_level in tier_mapping and "tier_id" in df.columns:
                tiers = tier_mapping[difficulty_level]
                df = df[df["tier_id"].isin(tiers)]
        
        # Format each row into various challenge types
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
            
            # Create advanced challenges based on enabled modes
            advanced_challenges = []
            
            # Basic challenges first
            if "primality" in self.challenge_modes:
                advanced_challenges.append({
                    "input": f"Determine if {number} is prime using the most appropriate algorithm. Explain your approach.",
                    "expected": "primality",
                    "answer": "prime" if len(factors.split("×")) == 1 else "composite",
                    "tier_id": tier_id,
                    "bit_length": bit_length,
                    "challenge_type": "primality",
                    "cognitive_complexity": 2,
                    "mathematical_depth": 1
                })
            
            if "factorization" in self.challenge_modes:
                # Only add factorization for composite numbers
                if len(factors.split("×")) > 1:
                    advanced_challenges.append({
                        "input": f"Find the prime factorization of {number}. Analyze algorithm choices and computational complexity.",
                        "expected": "factorization",
                        "answer": factors,
                        "tier_id": tier_id,
                        "bit_length": bit_length,
                        "challenge_type": "factorization",
                        "cognitive_complexity": 3,
                        "mathematical_depth": 2
                    })
            
            # Number theory analysis
            if "number_theory" in self.challenge_modes:
                advanced_challenges.append({
                    "input": f"Provide a comprehensive number theory analysis of {number}. Include its prime factorization, divisor properties, and special number classifications.",
                    "expected": "analysis",
                    "answer": factors,  # The actual analysis will be more comprehensive
                    "tier_id": tier_id,
                    "bit_length": bit_length,
                    "challenge_type": "number_theory",
                    "cognitive_complexity": 4,
                    "mathematical_depth": 3
                })
            
            # Quantum-resistant challenges
            if "quantum" in self.challenge_modes and self.quantum_resistant_mode and bit_length >= 64:
                advanced_challenges.append({
                    "input": f"Analyze {number} from a quantum computing perspective. How resistant would this number be to factorization by Shor's algorithm? What properties make it challenging to factor?",
                    "expected": "quantum",
                    "answer": factors,
                    "tier_id": tier_id,
                    "bit_length": bit_length,
                    "challenge_type": "quantum",
                    "cognitive_complexity": 5,
                    "mathematical_depth": 4
                })
            
            # Conjectures and patterns
            if "conjectures" in self.challenge_modes:
                # Only for numbers with interesting properties
                if (int(number) % 2 == 1 and int(number) > 3) or tier_id >= 5:
                    advanced_challenges.append({
                        "input": f"Explore mathematical conjectures related to {number}. Consider Goldbach's conjecture if it's even, or the twin prime conjecture if it's odd.",
                        "expected": "conjectures",
                        "answer": "exploration",
                        "tier_id": tier_id,
                        "bit_length": bit_length,
                        "challenge_type": "conjectures",
                        "cognitive_complexity": 5,
                        "mathematical_depth": 5
                    })
            
            # Proof sketching
            if "proof_sketching" in self.challenge_modes and self.theoretical_challenges:
                if tier_id >= 8:  # Only for advanced tiers
                    factor_list = factors.split("×")
                    if len(factor_list) >= 2:
                        advanced_challenges.append({
                            "input": f"Sketch a mathematical proof that {number} can be expressed as the product of exactly {len(factor_list)} distinct primes, and explain why this is the minimal factorization.",
                            "expected": "proof",
                            "answer": "proof_sketch",
                            "tier_id": tier_id,
                            "bit_length": bit_length,
                            "challenge_type": "proof_sketching",
                            "cognitive_complexity": 5,
                            "mathematical_depth": 5
                        })
            
            # Add metacognitive elements if enabled
            if self.metacognitive:
                for challenge in advanced_challenges:
                    challenge["input"] += "\n\nAs you solve this problem, explicitly track your metacognitive process—how you select approaches, monitor progress, identify errors, and reflect on your reasoning."
            
            # Add all challenges
            examples.extend(advanced_challenges)
        
        # Add theoretical challenges if enabled
        if self.theoretical_challenges:
            theoretical_examples = self._generate_theoretical_challenges()
            examples.extend(theoretical_examples)
        
        # Convert to HuggingFace Dataset
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    def _create_advanced_synthetic_dataset(self, difficulty_level: str) -> Dataset:
        """Create an advanced synthetic dataset with various challenge types."""
        examples = []
        
        # Define difficulty parameters with quantum-resistant focus
        if difficulty_level == "foundational":
            max_bits = 16
            max_factors = 2
            special_challenges = ["primality", "factorization"]
        elif difficulty_level == "intermediate":
            max_bits = 32
            max_factors = 3
            special_challenges = ["primality", "factorization", "number_theory"]
        elif difficulty_level == "advanced":
            max_bits = 64
            max_factors = 4
            special_challenges = ["primality", "factorization", "number_theory", "conjectures"]
        elif difficulty_level == "theoretical":
            max_bits = 128
            max_factors = 5
            special_challenges = ["factorization", "number_theory", "conjectures", "proof_sketching"]
        elif difficulty_level == "quantum":
            max_bits = 256
            max_factors = 4
            special_challenges = ["factorization", "quantum"]
        else:  # mixed
            difficulty_levels = ["foundational", "intermediate", "advanced"]
            return self._create_advanced_synthetic_dataset(random.choice(difficulty_levels))
        
        # Generate a diverse set of challenges
        challenge_count = 30  # Total challenges to generate
        
        # Prime numbers for testing
        primes = [self._generate_prime(random.randint(4, max_bits)) for _ in range(5)]
        
        # Carmichael numbers - special composite numbers that pass some primality tests
        carmichael_numbers = [561, 1105, 1729, 2465, 2821, 6601, 8911]
        if max_bits >= 32:
            carmichael_numbers.extend([41041, 62745, 63973, 75361, 101101])
        random.shuffle(carmichael_numbers)
        carmichael_numbers = carmichael_numbers[:3]  # Take a few
        
        # Generate interesting composites with specific structures
        composites = []
        for _ in range(5):
            # Create composites with different structures
            structure = random.choice(["balanced", "unbalanced", "power", "mixed"])
            composite = self._generate_structured_composite(structure, max_bits, max_factors)
            composites.append(composite)
        
        # Create primality challenges
        if "primality" in special_challenges:
            # Mix of primes, composites, and Carmichael numbers
            for number in primes + composites + carmichael_numbers:
                if len(examples) >= challenge_count:
                    break
                    
                is_prime = sympy.isprime(number)
                is_carmichael = number in carmichael_numbers
                
                # Create appropriate prompt based on the number
                if is_carmichael:
                    prompt = f"Determine if {number} is prime. This is a challenging number that may require multiple primality tests."
                else:
                    prompt = f"Determine if {number} is prime using the most appropriate algorithm. Explain your reasoning."
                
                examples.append({
                    "input": prompt,
                    "expected": "primality",
                    "answer": "prime" if is_prime else "composite",
                    "bit_length": number.bit_length(),
                    "challenge_type": "primality",
                    "is_carmichael": is_carmichael,
                    "cognitive_complexity": 3 if is_carmichael else 2,
                    "mathematical_depth": 3 if is_carmichael else 1
                })
        
        # Create factorization challenges
        if "factorization" in special_challenges:
            for number in composites:
                if len(examples) >= challenge_count:
                    break
                
                # Get factorization
                factors = list(sympy.factorint(number).items())
                factor_str = " × ".join([f"{p}^{e}" if e > 1 else str(p) for p, e in factors])
                
                bit_length = number.bit_length()
                complexity = 2 if bit_length < 32 else (3 if bit_length < 64 else 4)
                
                examples.append({
                    "input": f"Find the prime factorization of {number}. Analyze the computational complexity of your approach.",
                    "expected": "factorization",
                    "answer": factor_str,
                    "bit_length": bit_length,
                    "challenge_type": "factorization",
                    "cognitive_complexity": complexity,
                    "mathematical_depth": complexity
                })
        
        # Create number theory analysis challenges
        if "number_theory" in special_challenges:
            interesting_numbers = [28, 120, 496, 1001, 8128]  # Some interesting numbers
            interesting_numbers.extend(composites[:2])  # Plus some of our generated composites
            
            for number in interesting_numbers:
                if len(examples) >= challenge_count:
                    break
                
                examples.append({
                    "input": f"Provide a comprehensive number theory analysis of {number}. Include its prime factorization, divisor properties, and special classifications.",
                    "expected": "analysis",
                    "answer": "analysis",
                    "bit_length": number.bit_length() if isinstance(number, int) else len(bin(number)[2:]),
                    "challenge_type": "number_theory",
                    "cognitive_complexity": 4,
                    "mathematical_depth": 3
                })
        
        # Create quantum-resistant challenges
        if "quantum" in special_challenges:
            quantum_resistant = self._generate_quantum_resistant_challenges(3, max_bits)
            
            for challenge in quantum_resistant:
                if len(examples) >= challenge_count:
                    break
                
                examples.append(challenge)
        
        # Create theoretical challenges
        if "proof_sketching" in special_challenges:
            theoretical = self._generate_theoretical_challenges(3)
            
            for challenge in theoretical:
                if len(examples) >= challenge_count:
                    break
                
                examples.append(challenge)
        
        # Create conjecture exploration challenges
        if "conjectures" in special_challenges:
            conjecture_challenges = [
                {"input": "Explore the Collatz conjecture for the starting number 27. Track the sequence and analyze patterns.", 
                 "expected": "conjectures", "answer": "exploration", "challenge_type": "conjectures"},
                {"input": "Investigate Goldbach's conjecture for the even number 100. Find all ways to express it as the sum of two primes.", 
                 "expected": "conjectures", "answer": "exploration", "challenge_type": "conjectures"},
                {"input": "Explore twin primes up to 200. How many twin prime pairs are there, and do they follow any patterns?", 
                 "expected": "conjectures", "answer": "exploration", "challenge_type": "conjectures"}
            ]
            
            for challenge in conjecture_challenges:
                if len(examples) >= challenge_count:
                    break
                
                challenge["cognitive_complexity"] = 4
                challenge["mathematical_depth"] = 5
                challenge["bit_length"] = 32  # Approximate
                examples.append(challenge)
        
        # Add metacognitive elements if enabled
        if self.metacognitive:
            for example in examples:
                example["input"] += "\n\nAs you solve this problem, explicitly track your metacognitive process—how you select approaches, monitor progress, and reflect on your reasoning."
        
        # Ensure we don't exceed the desired count
        if len(examples) > challenge_count:
            random.shuffle(examples)
            examples = examples[:challenge_count]
        
        # Convert to HuggingFace Dataset
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    def _generate_quantum_resistant_challenges(self, count: int, max_bits: int) -> List[Dict]:
        """Generate quantum-resistant factorization challenges."""
        challenges = []
        
        for i in range(count):
            # Generate multiple factors instead of just two (to resist Shor's algorithm)
            num_factors = random.randint(3, 5)
            factors = []
            
            # Each factor should be large
            factor_bits = max(32, max_bits // num_factors)
            
            for _ in range(num_factors):
                factor = self._generate_prime(factor_bits)
                factors.append(factor)
            
            product = math.prod(factors)
            
            # Get factorization string
            factor_str = " × ".join(map(str, factors))
            
            # Create different types of quantum challenges
            challenge_types = [
                f"Analyze the factorization of {product} from a quantum computing perspective. How would Shor's algorithm approach this number compared to classical methods?",
                f"The number {product} has been designed to resist quantum factorization. Find its prime factors and explain why certain structures make factorization more difficult.",
                f"Quantum computers using Shor's algorithm can factor many RSA keys. Analyze {product} and determine if its structure would make it more or less resistant to quantum attacks."
            ]
            
            challenges.append({
                "input": challenge_types[i % len(challenge_types)],
                "expected": "quantum",
                "answer": factor_str,
                "bit_length": product.bit_length(),
                "challenge_type": "quantum",
                "cognitive_complexity": 5,
                "mathematical_depth": 4
            })
        
        return challenges
    
    def _generate_theoretical_challenges(self, count: int = 3) -> List[Dict]:
        """Generate theoretical number theory challenges."""
        theoretical_challenges = [
            {
                "input": "Sketch a proof of why there are infinitely many primes. Use proof by contradiction and explain the key insights.",
                "expected": "proof",
                "answer": "proof_sketch",
                "challenge_type": "proof_sketching",
                "cognitive_complexity": 5,
                "mathematical_depth": 5,
                "bit_length": 0  # Not applicable
            },
            {
                "input": "Provide a theoretical analysis of the density of prime numbers as numbers get larger. Reference the Prime Number Theorem in your explanation.",
                "expected": "proof",
                "answer": "analysis",
                "challenge_type": "proof_sketching",
                "cognitive_complexity": 5,
                "mathematical_depth": 5,
                "bit_length": 0  # Not applicable
            },
            {
                "input": "Explain why Fermat's Last Theorem is more difficult to prove than the Fundamental Theorem of Arithmetic. Sketch the key insights from both theorems.",
                "expected": "proof",
                "answer": "comparison",
                "challenge_type": "proof_sketching",
                "cognitive_complexity": 5,
                "mathematical_depth": 5,
                "bit_length": 0  # Not applicable
            },
            {
                "input": "Sketch a proof for why the product of two Mersenne primes plus 1 is never a perfect square. Use properties of modular arithmetic.",
                "expected": "proof",
                "answer": "proof_sketch",
                "challenge_type": "proof_sketching",
                "cognitive_complexity": 5,
                "mathematical_depth": 5,
                "bit_length": 0  # Not applicable
            },
            {
                "input": "Prove that there exist infinitely many primes of the form 4n+3. Outline a rigorous mathematical proof.",
                "expected": "proof",
                "answer": "proof_sketch",
                "challenge_type": "proof_sketching",
                "cognitive_complexity": 5,
                "mathematical_depth": 5,
                "bit_length": 0  # Not applicable
            }
        ]
        
        # Select a subset if count is provided
        if count and count < len(theoretical_challenges):
            return random.sample(theoretical_challenges, count)
        
        return theoretical_challenges
    
    def _generate_structured_composite(self, structure: str, max_bits: int, max_factors: int) -> int:
        """Generate a composite number with specific structural properties."""
        if structure == "balanced":
            # Roughly equal-sized prime factors
            num_factors = random.randint(2, max_factors)
            factor_bits = max(8, max_bits // num_factors)
            factors = [self._generate_prime(factor_bits) for _ in range(num_factors)]
            return math.prod(factors)
            
        elif structure == "unbalanced":
            # One large prime and several small ones
            large_bits = max(16, int(max_bits * 0.7))
            small_bits = max(4, int(max_bits * 0.1))
            
            large_prime = self._generate_prime(large_bits)
            small_primes = []
            
            num_small = random.randint(1, max_factors - 1)
            for _ in range(num_small):
                small_primes.append(self._generate_prime(small_bits))
            
            factors = [large_prime] + small_primes
            return math.prod(factors)
            
        elif structure == "power":
            # Prime power (p^n)
            prime_bits = max(8, max_bits // 3)
            prime = self._generate_prime(prime_bits)
            power = random.randint(2, min(5, max_factors))
            return prime ** power
            
        elif structure == "mixed":
            # Mix of different structures
            num_factors = random.randint(2, max_factors)
            factors = []
            
            for _ in range(num_factors):
                factor_bits = random.randint(4, max(8, max_bits // num_factors))
                prime = self._generate_prime(factor_bits)
                
                # Maybe use a small power
                if random.random() < 0.3:
                    power = random.randint(1, 3)
                    factors.append(prime ** power)
                else:
                    factors.append(prime)
            
            return math.prod(factors)
        
        # Default - simple product of random primes
        num_factors = random.randint(2, max_factors)
        factor_bits = max(4, max_bits // num_factors)
        factors = [self._generate_prime(factor_bits) for _ in range(num_factors)]
        return math.prod(factors)
    
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
        
        # For larger primes, use sympy's randprime
        lower = 2**(bits-1)
        upper = 2**bits - 1
        return sympy.randprime(lower, upper)
    
    async def run_sample(self, sample, max_steps=None, with_metrics=False):
        """Run a sample through the environment with advanced processing."""
        # Reset cognitive state for new problem
        self.current_cognitive_state = {
            "exploration_phase": "initial",
            "depth": 0,
            "tools_used": [],
            "key_insights": [],
            "strategies_attempted": [],
            "verification_level": 0
        }
        
        # Add metacognitive guidance if enabled
        if self.metacognitive and "input" in sample and not "As you solve this problem" in sample["input"]:
            sample = dict(sample)  # Create a copy to avoid modifying the original
            sample["input"] += "\n\nAs you solve this problem, explicitly track your metacognitive process—how you select approaches, monitor progress, and reflect on your reasoning."
        
        # Run the sample using the parent class method
        result = await super().run_sample(sample, max_steps, with_metrics)
        
        # Advanced post-processing
        if with_metrics and "metrics" in result:
            # Track performance for adaptive learning
            reward = result["metrics"].get("factorization_accuracy", 0.0)
            self.performance_history.append(reward)
            
            # Track cognitive patterns
            messages = result.get("messages", [])
            self._analyze_reasoning_patterns(messages, sample)
            
            # Update metrics with advanced insights
            result["metrics"]["cognitive_depth"] = self.current_cognitive_state["depth"]
            result["metrics"]["tools_used"] = len(self.current_cognitive_state["tools_used"])
            result["metrics"]["metacognitive_score"] = self._calculate_metacognitive_score(messages)
        
        return result
    
    def _analyze_reasoning_patterns(self, messages, sample):
        """Analyze reasoning patterns in the conversation."""
        for message in messages:
            if message.get("role") != "assistant":
                continue
                
            content = message.get("content", "")
            
            # Track tools used
            tool_matches = re.findall(r'<tool>.*?"name":\s*"([^"]+)"', content, re.DOTALL)
            self.current_cognitive_state["tools_used"].extend(tool_matches)
            
            # Track cognitive depth
            self.current_cognitive_state["depth"] += 1
            
            # Look for metacognitive patterns
            if re.search(r'\b(reflect|monitoring|evaluate|assessment|reasoning about|thinking about)\b', content, re.IGNORECASE):
                self.current_cognitive_state["verification_level"] += 1
            
            # Look for strategy shifts
            strategy_patterns = {
                "brute_force": r'\b(brute force|try all|exhaustive|check each)\b',
                "mathematical_property": r'\b(property|theorem|identity|principle)\b',
                "algorithm_selection": r'\b(algorithm|method|approach|technique)\b',
                "divide_conquer": r'\b(divide|break down|split|smaller problems)\b',
                "heuristic": r'\b(heuristic|approximation|estimation|shortcut)\b'
            }
            
            for strategy, pattern in strategy_patterns.items():
                if re.search(pattern, content, re.IGNORECASE) and strategy not in self.current_cognitive_state["strategies_attempted"]:
                    self.current_cognitive_state["strategies_attempted"].append(strategy)
    
    def _calculate_metacognitive_score(self, messages):
        """Calculate a metacognitive score based on reasoning quality."""
        if not self.metacognitive:
            return 0.0
            
        score = 0.0
        max_score = 5.0
        
        # Check for explicit metacognitive elements
        patterns = {
            "strategy_selection": r'\b(strategy|approach|method|algorithm)\s+(selection|choice|choosing)\b',
            "progress_monitoring": r'\b(monitoring|tracking|checking|evaluating)\s+(progress|advancement|development)\b',
            "error_detection": r'\b(error|mistake|issue|problem)\s+(detection|identification|finding|spotting)\b',
            "reflection": r'\b(reflect|reflection|thinking about|considering|evaluating)\s+(approach|strategy|process|thinking)\b',
            "adjustment": r'\b(adjust|change|modify|revise|update)\s+(approach|strategy|method|plan)\b'
        }
        
        for message in messages:
            if message.get("role") != "assistant":
                continue
                
            content = message.get("content", "")
            
            # Check for each metacognitive pattern
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    score += 0.5  # Add points for each metacognitive element
            
            # Check for structured reasoning sections
            if re.search(r'\[(MATHEMATICAL ANALYSIS|STRATEGY SELECTION|EXECUTION|MONITORING|REFLECTION)\]', content):
                score += 1.0
                
            # Look for explicit metacognitive language
            metacognitive_terms = [
                r'\bI notice\b', r'\bI realize\b', r'\bI am thinking\b', 
                r'\bmy approach\b', r'\bmy strategy\b', r'\bI need to reconsider\b',
                r'\bmonitoring my progress\b', r'\breflecting on\b'
            ]
            
            for term in metacognitive_terms:
                if re.search(term, content, re.IGNORECASE):
                    score += 0.2
        
        # Cap the score at the maximum
        return min(score, max_score) / max_score
    
    def update_adaptive_system(self, reward: float) -> None:
        """
        Update the adaptive learning system based on performance.
        
        Args:
            reward: The reward from the last episode
        """
        if not self.adaptive_system:
            return
        
        # Store the reward
        self.performance_history.append(reward)
        
        # Only update after collecting enough data
        if len(self.performance_history) < 5:
            return
        
        # Calculate recent average performance
        recent_avg = sum(self.performance_history[-5:]) / 5
        
        # Update the system based on performance
        if recent_avg > 0.8:
            # Doing well - increase complexity
            self._increase_system_complexity()
        elif recent_avg < 0.3:
            # Struggling - decrease complexity
            self._decrease_system_complexity()
    
    def _increase_system_complexity(self) -> None:
        """Increase the complexity of the learning system."""
        print("Increasing system complexity")
        
        # Increase depth and reduce scaffolding
        if self.metacognitive:
            # Shift to more advanced metacognitive approaches
            self.cognitive_architecture = "multi-agent" if self.cognitive_architecture != "multi-agent" else "metacognitive"
            
        # Add more challenging modes
        if "proof_sketching" not in self.challenge_modes and self.theoretical_challenges:
            self.challenge_modes.append("proof_sketching")
        
        # Add quantum challenges if not already present
        if "quantum" not in self.challenge_modes and self.quantum_resistant_mode:
            self.challenge_modes.append("quantum")
    
    def _decrease_system_complexity(self) -> None:
        """Decrease the complexity of the learning system."""
        print("Decreasing system complexity")
        
        # Simplify cognitive architecture
        self.cognitive_architecture = "hierarchical" if self.cognitive_architecture == "multi-agent" else "none"
        
        # Remove more challenging modes
        if "proof_sketching" in self.challenge_modes:
            self.challenge_modes.remove("proof_sketching")
        
        if "quantum" in self.challenge_modes and len(self.challenge_modes) > 2:
            self.challenge_modes.remove("quantum")
            
        # Focus on core challenges
        core_challenges = ["primality", "factorization", "number_theory"]
        self.challenge_modes = [mode for mode in self.challenge_modes if mode in core_challenges]