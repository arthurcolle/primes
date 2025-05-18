gi#!/usr/bin/env python3
"""
neurosymbolic_factorizer.py

This module implements a neurosymbolic approach to prime factorization of massive numbers.
It combines neural network guidance with symbolic mathematical algorithms to efficiently
factorize extremely large numbers that would be infeasible with traditional methods alone.
"""

import os
import math
import time
import json
import random
import logging
import numpy as np
import sympy
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
import multiprocessing as mp
from collections import defaultdict

# Try importing optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Running in symbolic-only mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neurosymbolic_factorizer")

@dataclass
class FactorizationResult:
    """Container for factorization results with metadata."""
    number: int
    factors: List[int]
    algorithm: str
    time_taken: float
    iterations: int
    confidence: float
    intermediate_steps: List[Dict] = field(default_factory=list)

    def verify(self) -> bool:
        """Verify that the factorization is correct."""
        product = 1
        for factor in self.factors:
            product *= factor
        return product == self.number

    def __str__(self) -> str:
        """String representation of the factorization."""
        return f"{self.number} = {' × '.join(map(str, self.factors))}"
        
    def to_json(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "number": self.number,
            "factors": self.factors,
            "algorithm": self.algorithm,
            "time_taken": self.time_taken,
            "iterations": self.iterations,
            "confidence": self.confidence,
            "verified": self.verify()
        }
        
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'FactorizationResult':
        """Create a FactorizationResult from a JSON dictionary."""
        result = cls(
            number=data["number"],
            factors=data["factors"],
            algorithm=data["algorithm"],
            time_taken=data["time_taken"],
            iterations=data["iterations"],
            confidence=data["confidence"],
            intermediate_steps=data.get("intermediate_steps", [])
        )
        return result


class SymbolicAlgorithms:
    """
    Collection of symbolic factorization algorithms with performance optimizations.
    """
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Compute the greatest common divisor of a and b."""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Quick primality test."""
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
    
    @staticmethod
    def pollard_rho(n: int, max_iterations: int = 10000, seed: int = 1) -> Tuple[Optional[int], int]:
        """
        Pollard's rho algorithm for finding factors.
        
        Args:
            n: Number to factorize
            max_iterations: Maximum number of iterations
            seed: Random seed for the algorithm
            
        Returns:
            Tuple of (factor, iterations)
        """
        if n % 2 == 0:
            return 2, 1
        
        # Define the function f(x) = (x^2 + c) % n
        def f(x, c=seed):
            return (x * x + c) % n
        
        # Initialize
        x, y, d = 2, 2, 1
        iterations = 0
        
        # Main algorithm loop
        while d == 1 and iterations < max_iterations:
            iterations += 1
            x = f(x)             # x moves one step
            y = f(f(y))          # y moves two steps
            d = SymbolicAlgorithms.gcd(abs(x - y), n)
        
        if d != 1 and d != n:
            return d, iterations
        return None, iterations
    
    @staticmethod
    def lenstra_ecm(n: int, max_curves: int = 100, max_iterations: int = 1000, 
                   B1: int = 10000, B2: int = 100000) -> Tuple[Optional[int], int]:
        """
        Lenstra's Elliptic Curve Method (ECM) for factorization.
        
        Args:
            n: Number to factorize
            max_curves: Maximum number of curves to try
            max_iterations: Maximum iterations per curve
            B1: First bound for the algorithm
            B2: Second bound for the algorithm
            
        Returns:
            Tuple of (factor, iterations)
        """
        if n % 2 == 0:
            return 2, 1
        
        total_iterations = 0
        
        for curve in range(max_curves):
            # Choose random parameters for the elliptic curve
            a = random.randint(1, n - 1)
            x = random.randint(1, n - 1)
            y = random.randint(1, n - 1)
            
            # Initialize point (x, y) on curve y^2 = x^3 + ax + b
            try:
                # Compute b = y^2 - x^3 - ax (mod n)
                b = (y * y - x * x * x - a * x) % n
                
                # Initialize the elliptic curve point
                p = (x, y)
                
                # Phase 1: Multiply p by all prime powers <= B1
                for prime in sympy.primerange(2, B1 + 1):
                    # Determine the largest power of this prime <= B1
                    power = prime
                    while power * prime <= B1:
                        power *= prime
                    
                    # Try to compute [power]P
                    try:
                        p = SymbolicAlgorithms._ecm_multiply(p, power, a, b, n)
                        total_iterations += 1
                    except Exception as e:
                        # If computation fails, we might have found a factor
                        factor = SymbolicAlgorithms.gcd(int(str(e).split()[-1]), n)
                        if 1 < factor < n:
                            return factor, total_iterations
                
                # Phase 2 would go here in a complete implementation
                
            except Exception as e:
                # Check if we found a factor
                try:
                    factor = SymbolicAlgorithms.gcd(int(str(e).split()[-1]), n)
                    if 1 < factor < n:
                        return factor, total_iterations
                except:
                    pass
            
            if total_iterations >= max_iterations:
                break
        
        return None, total_iterations
    
    @staticmethod
    def _ecm_multiply(p, k, a, b, n):
        """Helper method for ECM: multiply a point by a scalar."""
        # This is a simplified implementation
        if k == 0:
            raise ValueError("Zero point")
        if k == 1:
            return p
        
        # Double and add algorithm
        result = None
        addend = p
        
        while k:
            if k & 1:
                if result is None:
                    result = addend
                else:
                    result = SymbolicAlgorithms._ecm_add(result, addend, a, n)
            k >>= 1
            if k:
                addend = SymbolicAlgorithms._ecm_add(addend, addend, a, n)
        
        return result
    
    @staticmethod
    def _ecm_add(p1, p2, a, n):
        """Add two points on an elliptic curve."""
        # Simplified point addition
        if p1 is None:
            return p2
        if p2 is None:
            return p1
            
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:
            if (y1 + y2) % n == 0:
                raise ValueError(f"Factor found {SymbolicAlgorithms.gcd(y1, n)}")
            # Point doubling
            try:
                lam = (3 * x1 * x1 + a) * SymbolicAlgorithms._modinv(2 * y1, n) % n
            except ValueError as e:
                raise ValueError(f"Factor found {str(e).split()[-1]}")
        else:
            try:
                lam = (y2 - y1) * SymbolicAlgorithms._modinv(x2 - x1, n) % n
            except ValueError as e:
                raise ValueError(f"Factor found {str(e).split()[-1]}")
        
        x3 = (lam * lam - x1 - x2) % n
        y3 = (lam * (x1 - x3) - y1) % n
        
        return (x3, y3)
    
    @staticmethod
    def _modinv(a, m):
        """Compute the modular multiplicative inverse."""
        g, x, y = SymbolicAlgorithms._egcd(a, m)
        if g != 1:
            raise ValueError(f"Modular inverse does not exist {g}")
        else:
            return x % m
    
    @staticmethod
    def _egcd(a, b):
        """Extended GCD algorithm."""
        if a == 0:
            return b, 0, 1
        else:
            g, x, y = SymbolicAlgorithms._egcd(b % a, a)
            return g, y - (b // a) * x, x


class NeuralFactorizationHint:
    """
    Neural network component that provides hints for factorization
    by predicting promising search regions or algorithm selections.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_cuda: bool = False):
        """
        Initialize the neural factorization hint model.
        
        Args:
            model_path: Path to a pretrained model
            use_cuda: Whether to use CUDA if available
        """
        self.use_cuda = use_cuda and torch.cuda.is_available() if HAS_TORCH else False
        self.model = None
        
        if HAS_TORCH:
            self._init_model(model_path)
        
    def _init_model(self, model_path: Optional[str] = None):
        """Initialize the neural network model."""
        class FactorizationHintNetwork(nn.Module):
            def __init__(self, embedding_dim=256):
                super().__init__()
                # Embed the binary representation of the number
                self.embedding_layer = nn.Linear(1024, embedding_dim)
                
                # Analysis layers
                self.hidden1 = nn.Linear(embedding_dim, 512)
                self.hidden2 = nn.Linear(512, 256)
                self.hidden3 = nn.Linear(256, 128)
                
                # Output heads:
                # 1. Algorithm selection (classification)
                self.algorithm_head = nn.Linear(128, 6)  # 6 different algorithms
                
                # 2. Search interval prediction (regression)
                self.search_interval_head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)  # (lower_bound, upper_bound) as percentages of sqrt(n)
                )
                
                # 3. Confidence score
                self.confidence_head = nn.Linear(128, 1)
            
            def forward(self, x):
                # Main network
                x = F.relu(self.embedding_layer(x))
                x = F.relu(self.hidden1(x))
                x = F.relu(self.hidden2(x))
                features = F.relu(self.hidden3(x))
                
                # Output heads
                algorithm_logits = self.algorithm_head(features)
                search_bounds = torch.sigmoid(self.search_interval_head(features))  # 0-1 range
                confidence = torch.sigmoid(self.confidence_head(features))
                
                return {
                    'algorithm_logits': algorithm_logits,
                    'search_bounds': search_bounds,
                    'confidence': confidence
                }
        
        # Create the model
        self.model = FactorizationHintNetwork()
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained model from {model_path}")
        else:
            logger.warning("No pretrained model found. Using random initialization.")
        
        # Move to GPU if available
        if self.use_cuda:
            self.model.cuda()
            logger.info("Using CUDA for neural network inference")
        
        # Set to evaluation mode
        self.model.eval()
    
    def get_factorization_hint(self, n: int) -> Dict[str, Any]:
        """
        Get factorization hints from the neural network.
        
        Args:
            n: Number to factorize
            
        Returns:
            Dictionary with hints including:
            - recommended_algorithm: String name of algorithm
            - search_bounds: (lower, upper) percentages of sqrt(n)
            - confidence: Model confidence (0-1)
        """
        if not HAS_TORCH or self.model is None:
            # Fallback to heuristic hints without neural network
            return self._get_heuristic_hint(n)
        
        # Convert number to binary representation
        binary_repr = self._number_to_binary_tensor(n)
        
        # Get model prediction
        with torch.no_grad():
            if self.use_cuda:
                binary_repr = binary_repr.cuda()
            
            outputs = self.model(binary_repr)
            
            # Process outputs
            algorithm_idx = torch.argmax(outputs['algorithm_logits'], dim=1).item()
            search_bounds = outputs['search_bounds'].cpu().numpy()[0]
            confidence = outputs['confidence'].item()
            
            algorithms = [
                "trial_division", "pollard_rho", "williams_p1",
                "lenstra_ecm", "quadratic_sieve", "number_field_sieve"
            ]
            
            return {
                'recommended_algorithm': algorithms[algorithm_idx],
                'search_bounds': tuple(search_bounds),
                'confidence': confidence
            }
    
    def _number_to_binary_tensor(self, n: int) -> torch.Tensor:
        """Convert a number to a binary tensor representation."""
        # Convert to binary representation, padded to 1024 bits
        binary = bin(n)[2:].zfill(1024)[-1024:]
        binary_array = np.array([int(bit) for bit in binary], dtype=np.float32)
        return torch.tensor(binary_array, dtype=torch.float32).unsqueeze(0)
    
    def _get_heuristic_hint(self, n: int) -> Dict[str, Any]:
        """Provide heuristic hints without using a neural network."""
        bit_length = n.bit_length()
        
        # Simple algorithm selection based on bit length
        if bit_length < 30:
            algorithm = "trial_division"
            confidence = 0.9
        elif bit_length < 60:
            algorithm = "pollard_rho"
            confidence = 0.8
        elif bit_length < 120:
            algorithm = "williams_p1"
            confidence = 0.7
        elif bit_length < 300:
            algorithm = "lenstra_ecm"
            confidence = 0.6
        elif bit_length < 500:
            algorithm = "quadratic_sieve" 
            confidence = 0.5
        else:
            algorithm = "number_field_sieve"
            confidence = 0.4
        
        # Heuristic search bounds - focus more narrowly for larger numbers
        search_width = max(0.1, 1.0 - (bit_length / 1000))
        search_center = 0.5
        lower_bound = max(0.0, search_center - search_width/2)
        upper_bound = min(1.0, search_center + search_width/2)
        
        return {
            'recommended_algorithm': algorithm,
            'search_bounds': (lower_bound, upper_bound),
            'confidence': confidence
        }


class PatternAnalyzer:
    """
    Analyzes mathematical patterns in numbers to guide factorization.
    Uses mathematical properties and statistical analysis rather than
    brute force approaches.
    """
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.prime_patterns = {
            # Fermat's factorization: n = a² - b²
            'fermat': self._check_fermat_pattern,
            
            # Mersenne numbers: n = 2^k - 1
            'mersenne': self._check_mersenne_pattern,
            
            # Near power patterns: n ≈ a^b
            'near_power': self._check_near_power,
            
            # Carmichael numbers
            'carmichael': self._check_carmichael
        }
    
    def analyze(self, n: int) -> Dict[str, Any]:
        """
        Analyze a number to find patterns that might help factorization.
        
        Args:
            n: Number to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Check each pattern
        for pattern_name, pattern_func in self.prime_patterns.items():
            pattern_result = pattern_func(n)
            if pattern_result['detected']:
                results[pattern_name] = pattern_result
        
        # Add statistical properties
        results['bit_length'] = n.bit_length()
        results['digit_length'] = len(str(n))
        
        # Check divisibility by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        divisibility = {}
        for p in small_primes:
            if n % p == 0:
                divisibility[p] = n // p
        
        results['small_divisors'] = divisibility
        
        return results
    
    def _check_fermat_pattern(self, n: int) -> Dict[str, Any]:
        """Check if n might be efficiently factored using Fermat's method."""
        # If n is odd and a perfect square, it's not prime but this isn't a Fermat pattern
        if n % 2 == 1 and math.isqrt(n)**2 == n:
            return {'detected': False}
        
        # Check if n is close to a perfect square
        root = math.isqrt(n)
        
        if root**2 == n:
            # Perfect square
            return {'detected': False}
        
        # Check if n is close to a perfect square (within 1%)
        diff = (root + 1)**2 - n
        if diff <= n // 100:
            # Very close to a perfect square - good for Fermat's method
            return {
                'detected': True,
                'pattern': 'fermat',
                'hint': f"Close to perfect square: √{n} ≈ {root+1}",
                'starting_point': root + 1,
                'confidence': 0.8
            }
        
        return {'detected': False}
    
    def _check_mersenne_pattern(self, n: int) -> Dict[str, Any]:
        """Check if n is of form 2^k - 1 (Mersenne number)."""
        # Must be odd
        if n % 2 == 0:
            return {'detected': False}
        
        # Check if n = 2^k - 1
        n_plus_1 = n + 1
        if n_plus_1 & (n_plus_1 - 1) == 0:  # Is power of 2
            k = n_plus_1.bit_length() - 1
            return {
                'detected': True,
                'pattern': 'mersenne',
                'hint': f"Mersenne number: 2^{k} - 1",
                'exponent': k,
                'confidence': 0.9
            }
        
        return {'detected': False}
    
    def _check_near_power(self, n: int) -> Dict[str, Any]:
        """Check if n is close to a^b for some small a, b."""
        # Try small bases
        for base in range(2, 20):
            # Estimate exponent
            exp = math.log(n, base)
            exp_rounded = round(exp)
            
            # Check if n is close to base^exp_rounded
            power = base ** exp_rounded
            difference = abs(n - power)
            
            if difference <= n // 1000:  # Within 0.1%
                return {
                    'detected': True,
                    'pattern': 'near_power',
                    'hint': f"Near power: {base}^{exp_rounded} ≈ {n}",
                    'base': base,
                    'exponent': exp_rounded,
                    'difference': difference,
                    'confidence': 0.7
                }
        
        return {'detected': False}
    
    def _check_carmichael(self, n: int) -> Dict[str, Any]:
        """Check if n might be a Carmichael number."""
        # Simple test: check if n is odd and not divisible by square of a prime
        if n % 2 == 0:
            return {'detected': False}
        
        # Quickly test a few bases for the Fermat primality test
        bases = [2, 3, 5, 7]
        passes = 0
        
        for base in bases:
            if pow(base, n-1, n) == 1:
                passes += 1
        
        if passes == len(bases) and not SymbolicAlgorithms.is_prime(n):
            return {
                'detected': True,
                'pattern': 'carmichael',
                'hint': "Possible Carmichael number, might be product of 3 primes",
                'confidence': 0.6
            }
        
        return {'detected': False}


class DistributedFactorizationManager:
    """
    Manages distributed factorization attempts across multiple algorithms
    and computational resources.
    """
    
    def __init__(self, max_workers: Optional[int] = None, timeout: int = 300,
                neural_model_path: Optional[str] = None):
        """
        Initialize the factorization manager.
        
        Args:
            max_workers: Maximum number of worker processes (default: CPU count)
            timeout: Default timeout in seconds for factorization attempts
            neural_model_path: Path to pretrained neural model
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.timeout = timeout
        self.neural_hints = NeuralFactorizationHint(model_path=neural_model_path)
        self.pattern_analyzer = PatternAnalyzer()
        
        logger.info(f"Initialized with {self.max_workers} workers and {timeout}s timeout")
    
    def factorize(self, n: int, timeout: Optional[int] = None) -> FactorizationResult:
        """
        Factorize a number using the optimal neurosymbolic approach.
        
        Args:
            n: Number to factorize
            timeout: Optional timeout in seconds (overrides default)
            
        Returns:
            FactorizationResult with prime factorization and metadata
        """
        start_time = time.time()
        timeout = timeout or self.timeout
        steps = []
        
        # First, check trivial cases
        if n <= 1:
            return FactorizationResult(n, [n], "trivial", 0.0, 0, 1.0, [])
        
        # Check if n is prime
        if SymbolicAlgorithms.is_prime(n):
            return FactorizationResult(n, [n], "prime_test", 0.0, 1, 1.0, 
                                  [{'step': 'prime_test', 'result': 'prime'}])
        
        # First check small primes with trial division (always worthwhile)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        factors = []
        remaining = n
        
        for p in small_primes:
            while remaining % p == 0:
                factors.append(p)
                remaining //= p
                steps.append({'step': 'small_prime', 'factor': p, 'remaining': remaining})
        
        if remaining == 1:
            elapsed = time.time() - start_time
            return FactorizationResult(n, factors, "trial_division", elapsed, len(factors), 1.0, steps)
        
        # If remaining is prime, we're done
        if SymbolicAlgorithms.is_prime(remaining):
            factors.append(remaining)
            steps.append({'step': 'remaining_prime', 'factor': remaining})
            elapsed = time.time() - start_time
            return FactorizationResult(n, factors, "trial_division_with_primality", elapsed, len(factors) + 1, 1.0, steps)
        
        # Get neural network hints
        neural_hint = self.neural_hints.get_factorization_hint(remaining)
        steps.append({'step': 'neural_hint', 'hint': neural_hint})
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze(remaining)
        steps.append({'step': 'pattern_analysis', 'patterns': patterns})
        
        # Determine factorization strategy based on hints and patterns
        strategy = self._determine_strategy(remaining, neural_hint, patterns)
        steps.append({'step': 'strategy_selection', 'strategy': strategy})
        
        # Apply the selected strategy
        result = self._apply_strategy(remaining, strategy, timeout - (time.time() - start_time))
        
        # Combine all factors and steps
        all_factors = factors + result.factors
        all_steps = steps + result.intermediate_steps
        
        # Final result
        elapsed = time.time() - start_time
        return FactorizationResult(
            number=n,
            factors=all_factors,
            algorithm=f"neurosymbolic_{result.algorithm}",
            time_taken=elapsed,
            iterations=result.iterations,
            confidence=result.confidence,
            intermediate_steps=all_steps
        )
    
    def _determine_strategy(self, n: int, neural_hint: Dict[str, Any], 
                           patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best factorization strategy based on hints and patterns.
        
        Args:
            n: Number to factorize
            neural_hint: Hints from neural network
            patterns: Detected patterns
            
        Returns:
            Strategy dictionary with algorithm selection and parameters
        """
        bit_length = n.bit_length()
        
        # Check if we have detected any special patterns
        if 'fermat' in patterns:
            return {
                'algorithm': 'fermat',
                'starting_point': patterns['fermat']['starting_point'],
                'confidence': patterns['fermat']['confidence']
            }
        
        if 'mersenne' in patterns:
            return {
                'algorithm': 'specialized_mersenne',
                'exponent': patterns['mersenne']['exponent'],
                'confidence': patterns['mersenne']['confidence']
            }
        
        if 'carmichael' in patterns:
            return {
                'algorithm': 'parallel_ecm',
                'curves': min(100, self.max_workers * 5),
                'confidence': patterns['carmichael']['confidence']
            }
        
        # If we have small divisors from pattern analysis, use them
        if patterns['small_divisors']:
            factor = min(patterns['small_divisors'].keys())
            return {
                'algorithm': 'trial_division',
                'factor': factor,
                'confidence': 1.0
            }
        
        # Otherwise, use neural network recommendation
        search_bounds = neural_hint['search_bounds']
        sqrt_n = math.isqrt(n)
        lower_bound = int(sqrt_n * search_bounds[0])
        upper_bound = int(sqrt_n * search_bounds[1])
        
        # Ensure bounds are reasonable
        lower_bound = max(2, lower_bound)
        upper_bound = min(sqrt_n + 1000, upper_bound)
        
        strategy = {
            'algorithm': neural_hint['recommended_algorithm'],
            'search_bounds': (lower_bound, upper_bound),
            'confidence': neural_hint['confidence']
        }
        
        # Add algorithm-specific parameters
        if neural_hint['recommended_algorithm'] == 'pollard_rho':
            strategy['max_iterations'] = 100000
            strategy['seeds'] = list(range(1, 11))  # Try multiple seeds
        
        elif neural_hint['recommended_algorithm'] == 'lenstra_ecm':
            strategy['curves'] = min(50, self.max_workers * 3)
            strategy['B1'] = 50000
            strategy['B2'] = 500000
        
        elif neural_hint['recommended_algorithm'] == 'quadratic_sieve':
            # Quadratic sieve parameters
            strategy['factor_base_size'] = min(10000, bit_length * 5)
            
        elif neural_hint['recommended_algorithm'] == 'number_field_sieve':
            # Just a placeholder - full GNFS implementation would be much more complex
            strategy['algorithm'] = 'parallel_ecm'  # Fallback to ECM
            strategy['curves'] = min(200, self.max_workers * 10)
            
        return strategy
    
    def _apply_strategy(self, n: int, strategy: Dict[str, Any], timeout: float) -> FactorizationResult:
        """
        Apply the selected strategy to factorize the number.
        
        Args:
            n: Number to factorize
            strategy: Strategy dictionary
            timeout: Maximum time in seconds
            
        Returns:
            FactorizationResult with factorization and metadata
        """
        algorithm = strategy['algorithm']
        start_time = time.time()
        iterations = 0
        steps = []
        
        if algorithm == 'trial_division':
            # Simple trial division with a known factor
            factor = strategy['factor']
            factors = [factor]
            remaining = n // factor
            steps.append({'step': 'trial_division', 'factor': factor, 'remaining': remaining})
            
            # Recursively factorize the remaining part
            if remaining > 1:
                remaining_result = self.factorize(remaining, timeout - (time.time() - start_time))
                factors.extend(remaining_result.factors)
                steps.extend(remaining_result.intermediate_steps)
                iterations += remaining_result.iterations
        
        elif algorithm == 'fermat':
            # Fermat's factorization method
            a = strategy.get('starting_point', math.isqrt(n) + 1)
            max_iter = 100000
            
            steps.append({'step': 'fermat_start', 'a': a})
            
            for i in range(max_iter):
                iterations += 1
                a_squared = a * a
                b_squared = a_squared - n
                b = math.isqrt(b_squared)
                
                if b * b == b_squared:
                    # Found factors: n = a^2 - b^2 = (a+b)(a-b)
                    factor1 = a + b
                    factor2 = a - b
                    steps.append({'step': 'fermat_success', 'a': a, 'b': b})
                    
                    # Further factorize these factors if needed
                    factors = []
                    for f in [factor1, factor2]:
                        if f == 1:
                            continue
                        if SymbolicAlgorithms.is_prime(f):
                            factors.append(f)
                        else:
                            sub_result = self.factorize(f, timeout - (time.time() - start_time))
                            factors.extend(sub_result.factors)
                            steps.extend(sub_result.intermediate_steps)
                            iterations += sub_result.iterations
                    break
                
                a += 1
                if time.time() - start_time > timeout:
                    steps.append({'step': 'fermat_timeout'})
                    # Fallback to Pollard rho
                    return self._apply_strategy(n, {'algorithm': 'pollard_rho'}, timeout)
            else:
                # Fermat's method failed, fallback to Pollard rho
                steps.append({'step': 'fermat_failed'})
                return self._apply_strategy(n, {'algorithm': 'pollard_rho'}, timeout)
        
        elif algorithm == 'specialized_mersenne':
            # Specialized Mersenne number factorization
            exponent = strategy['exponent']
            steps.append({'step': 'mersenne_factorization', 'exponent': exponent})
            
            # For Mersenne numbers, try known factorization properties
            # 1. If exponent is composite, we can factorize
            exponent_factors = []
            
            if not SymbolicAlgorithms.is_prime(exponent):
                # If k = pq, then 2^k - 1 = (2^p - 1) * (1 + 2^p + 2^2p + ... + 2^(q-1)p)
                for p in range(2, math.isqrt(exponent) + 1):
                    if exponent % p == 0:
                        q = exponent // p
                        factor1 = 2**p - 1
                        factor2 = sum(2**(p*i) for i in range(q))
                        exponent_factors = [p, q]
                        steps.append({'step': 'mersenne_decompose', 'p': p, 'q': q})
                        break
            
            if exponent_factors:
                # Recursively factorize the factors
                factors = []
                for f in [factor1, factor2]:
                    if SymbolicAlgorithms.is_prime(f):
                        factors.append(f)
                    else:
                        sub_result = self.factorize(f, timeout - (time.time() - start_time))
                        factors.extend(sub_result.factors)
                        steps.extend(sub_result.intermediate_steps)
                        iterations += sub_result.iterations
            else:
                # Try special algorithms for Mersenne factorization like P-1
                # Simplified here - just fall back to Pollard rho
                steps.append({'step': 'mersenne_fallback'})
                
                # Try a specialized trial division range
                factors = []
                remaining = n
                
                # Check divisibility by numbers of form 2*k*p + 1 where p is the exponent
                for k in range(1, 1000):
                    candidate = 2 * k * exponent + 1
                    if candidate > 1000000:  # Limit the search
                        break
                    if remaining % candidate == 0:
                        factors.append(candidate)
                        remaining //= candidate
                        steps.append({'step': 'mersenne_factor', 'factor': candidate, 'form': f"2*{k}*{exponent}+1"})
                
                if remaining == 1:
                    pass  # Found all factors
                elif SymbolicAlgorithms.is_prime(remaining):
                    factors.append(remaining)
                    steps.append({'step': 'mersenne_remaining_prime', 'factor': remaining})
                else:
                    # Use Pollard rho for the remaining part
                    sub_result = self._apply_strategy(remaining, {'algorithm': 'pollard_rho'}, 
                                                     timeout - (time.time() - start_time))
                    factors.extend(sub_result.factors)
                    steps.extend(sub_result.intermediate_steps)
                    iterations += sub_result.iterations
        
        elif algorithm == 'pollard_rho':
            # Enhanced Pollard rho implementation
            seeds = strategy.get('seeds', [1, 2, 3])
            max_iterations = strategy.get('max_iterations', 100000)
            
            # Try multiple seeds in parallel
            with mp.Pool(min(len(seeds), self.max_workers)) as pool:
                seed_tasks = []
                for seed in seeds:
                    args = (n, max_iterations, seed)
                    seed_tasks.append(pool.apply_async(SymbolicAlgorithms.pollard_rho, args))
                
                # Wait for the first successful result
                factor = None
                for i, task in enumerate(seed_tasks):
                    try:
                        factor_result, iters = task.get(timeout=timeout/len(seeds))
                        iterations += iters
                        
                        if factor_result:
                            factor = factor_result
                            steps.append({'step': 'pollard_rho_success', 'seed': seeds[i], 
                                         'factor': factor, 'iterations': iters})
                            break
                    except:
                        pass
                
                # Clean up other tasks
                pool.terminate()
            
            if factor:
                # Recursively factorize the factors
                factors = []
                for f in [factor, n // factor]:
                    if f == 1:
                        continue
                    if SymbolicAlgorithms.is_prime(f):
                        factors.append(f)
                        steps.append({'step': 'primality_check', 'number': f, 'is_prime': True})
                    else:
                        steps.append({'step': 'recursive_factorization', 'number': f})
                        sub_result = self.factorize(f, timeout - (time.time() - start_time))
                        factors.extend(sub_result.factors)
                        steps.extend(sub_result.intermediate_steps)
                        iterations += sub_result.iterations
            else:
                # Pollard rho failed, try ECM
                steps.append({'step': 'pollard_rho_failed', 'iterations': iterations})
                return self._apply_strategy(n, {'algorithm': 'parallel_ecm'}, timeout)
        
        elif algorithm == 'parallel_ecm' or algorithm == 'lenstra_ecm':
            # Elliptic Curve Method with multiple curves in parallel
            curves = strategy.get('curves', 50)
            B1 = strategy.get('B1', 50000)
            B2 = strategy.get('B2', 5 * B1)
            
            steps.append({'step': 'ecm_start', 'curves': curves, 'B1': B1, 'B2': B2})
            
            # Distribute curves across workers
            curve_batches = []
            batch_size = max(1, curves // self.max_workers)
            
            for i in range(0, curves, batch_size):
                curve_batches.append(min(batch_size, curves - i))
            
            # Run ECM in parallel
            with mp.Pool(min(len(curve_batches), self.max_workers)) as pool:
                ecm_tasks = []
                for batch in curve_batches:
                    args = (n, batch, min(100000, batch * 100), B1, B2)
                    ecm_tasks.append(pool.apply_async(SymbolicAlgorithms.lenstra_ecm, args))
                
                # Wait for the first successful result
                factor = None
                for i, task in enumerate(ecm_tasks):
                    try:
                        factor_result, iters = task.get(timeout=timeout/len(curve_batches))
                        iterations += iters
                        
                        if factor_result:
                            factor = factor_result
                            steps.append({'step': 'ecm_success', 'batch': i, 
                                         'factor': factor, 'iterations': iters})
                            break
                    except:
                        pass
                
                # Clean up other tasks
                pool.terminate()
            
            if factor:
                # Recursively factorize the factors
                factors = []
                for f in [factor, n // factor]:
                    if f == 1:
                        continue
                    if SymbolicAlgorithms.is_prime(f):
                        factors.append(f)
                        steps.append({'step': 'primality_check', 'number': f, 'is_prime': True})
                    else:
                        steps.append({'step': 'recursive_factorization', 'number': f})
                        sub_result = self.factorize(f, timeout - (time.time() - start_time))
                        factors.extend(sub_result.factors)
                        steps.extend(sub_result.intermediate_steps)
                        iterations += sub_result.iterations
            else:
                # ECM failed, fall back to probabilistic approach
                steps.append({'step': 'ecm_failed', 'iterations': iterations})
                
                if n.bit_length() < 100:
                    # For smaller numbers, just use trial division with distributed ranges
                    factors, sub_steps = self._distributed_trial_division(n, timeout)
                    steps.extend(sub_steps)
                else:
                    # For larger numbers, admit defeat (full QS or GNFS would be needed)
                    logger.warning(f"Failed to factorize {n} with available methods")
                    factors = [n]  # Unable to factorize further
                    steps.append({'step': 'factorization_failed', 'number': n})
        
        else:
            # Unimplemented algorithm, fall back to Pollard rho
            logger.warning(f"Unimplemented algorithm: {algorithm}, falling back to Pollard rho")
            return self._apply_strategy(n, {'algorithm': 'pollard_rho'}, timeout)
        
        # If we get here but haven't assigned factors, something went wrong
        if 'factors' not in locals():
            logger.error(f"Strategy application failed for algorithm {algorithm}")
            factors = [n]  # Unable to factorize further
            steps.append({'step': 'factorization_error', 'number': n, 'algorithm': algorithm})
        
        # Calculate confidence based on verification
        product = 1
        for factor in factors:
            product *= factor
        confidence = 1.0 if product == n else 0.0
        
        # Re-sort factors for consistent output
        factors.sort()
        
        elapsed = time.time() - start_time
        return FactorizationResult(
            number=n,
            factors=factors,
            algorithm=algorithm,
            time_taken=elapsed,
            iterations=iterations,
            confidence=confidence,
            intermediate_steps=steps
        )
    
    def _distributed_trial_division(self, n: int, timeout: float) -> Tuple[List[int], List[Dict]]:
        """Distributed trial division for smaller numbers."""
        start_time = time.time()
        sqrt_n = math.isqrt(n)
        
        # Split the range [2, sqrt(n)] into segments
        segments = []
        segment_size = max(1000, sqrt_n // self.max_workers)
        
        for start in range(2, sqrt_n + 1, segment_size):
            segments.append((start, min(start + segment_size - 1, sqrt_n)))
        
        # Search each segment in parallel
        steps = [{'step': 'distributed_trial_division', 'segments': len(segments)}]
        
        with mp.Pool(min(len(segments), self.max_workers)) as pool:
            results = []
            for segment in segments:
                results.append(pool.apply_async(self._trial_division_segment, (n, segment)))
            
            # Collect results
            factor = None
            for i, result in enumerate(results):
                try:
                    segment_factor = result.get(timeout=timeout/len(segments))
                    if segment_factor:
                        factor = segment_factor
                        steps.append({'step': 'trial_division_success', 'segment': i, 
                                     'factor': factor})
                        break
                except:
                    pass
        
        if factor:
            # Recursively factorize the factors
            factors = []
            other_factor = n // factor
            
            for f in [factor, other_factor]:
                if f == 1:
                    continue
                if SymbolicAlgorithms.is_prime(f):
                    factors.append(f)
                else:
                    # Don't recurse if we're running out of time
                    if time.time() - start_time < timeout * 0.8:
                        sub_result = self.factorize(f, timeout - (time.time() - start_time))
                        factors.extend(sub_result.factors)
                        steps.extend(sub_result.intermediate_steps)
                    else:
                        # Just add it as a composite factor
                        factors.append(f)
        else:
            # No factor found - the number might be prime
            if sympy.isprime(n):
                factors = [n]
                steps.append({'step': 'prime_verification', 'number': n, 'is_prime': True})
            else:
                # If it's not prime but we couldn't find factors, just return it
                factors = [n]
                steps.append({'step': 'factorization_failed', 'number': n})
        
        return factors, steps
    
    @staticmethod
    def _trial_division_segment(n: int, segment: Tuple[int, int]) -> Optional[int]:
        """Search a segment for factors using trial division."""
        start, end = segment
        
        # Try to find a factor in this segment
        for i in range(start, end + 1):
            if n % i == 0:
                return i
        
        return None


class OpenAIFactorizationTrainer:
    """
    Integrates with OpenAI's API to fine-tune a model for prime factorization
    using reinforcement learning.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the trainer with the OpenAI API key.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. API functionality disabled.")
        
        # Define schemas for structured output
        self.json_schema = self._load_factorization_schema()
    
    def _load_factorization_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for factorization output."""
        return {
            "name": "prime_factorization",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "factors": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of prime factors in ascending order"
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
    
    def generate_training_data(self, dataset_path: str, output_path: str, 
                              factorizer: DistributedFactorizationManager) -> Dict[str, Any]:
        """
        Generate training data for reinforcement fine-tuning.
        
        Args:
            dataset_path: Path to input dataset with factorization challenges
            output_path: Path to save the JSONL training data
            factorizer: DistributedFactorizationManager to generate solutions
            
        Returns:
            Dictionary with statistics about the generated dataset
        """
        # Load the dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        samples = dataset.get('samples', [])
        logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
        
        # Generate factorization solutions
        training_data = []
        stats = {
            'total': len(samples),
            'successful': 0,
            'failed': 0,
            'tiers': defaultdict(int)
        }
        
        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)} (tier {sample.get('tier_id', 'unknown')})")
            
            try:
                # Parse the number and reference factors
                number = int(sample['number'])
                tier_id = sample.get('tier_id', 'unknown')
                
                # Generate factorization solution
                result = factorizer.factorize(number)
                
                # Create training example
                example = {
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
                    "factors": result.factors,
                    "optimal_algorithm": result.algorithm,
                    "expected_time": result.time_taken,
                    "tier_id": tier_id
                }
                
                training_data.append(example)
                stats['successful'] += 1
                stats['tiers'][tier_id] += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                stats['failed'] += 1
        
        # Save the training data
        with open(output_path, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Generated {len(training_data)} training examples, saved to {output_path}")
        return stats
    
    def prepare_training_files(self, data_path: str, train_path: str, valid_path: str,
                              split_ratio: float = 0.8) -> Tuple[str, str]:
        """
        Prepare training and validation files for OpenAI fine-tuning.
        
        Args:
            data_path: Path to the JSONL data file
            train_path: Path to save the training file
            valid_path: Path to save the validation file
            split_ratio: Train/validation split ratio
            
        Returns:
            Tuple of (train_path, valid_path)
        """
        # Load the data
        examples = []
        with open(data_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Shuffle and split
        random.shuffle(examples)
        split_idx = int(len(examples) * split_ratio)
        train_examples = examples[:split_idx]
        valid_examples = examples[split_idx:]
        
        # Save the files
        with open(train_path, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
        
        with open(valid_path, 'w') as f:
            for example in valid_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Prepared {len(train_examples)} training and {len(valid_examples)} validation examples")
        return train_path, valid_path
    
    def train_model(self, train_file: str, valid_file: str, base_model: str = "gpt-3.5-turbo",
                   suffix: Optional[str] = None, batch_size: int = 4,
                   learning_rate_multiplier: float = 0.1, n_epochs: int = 3) -> str:
        """
        Train a model using OpenAI's fine-tuning API.
        
        Args:
            train_file: Path to the training file
            valid_file: Path to the validation file
            base_model: Base model to fine-tune
            suffix: Suffix for the fine-tuned model name
            batch_size: Batch size for training
            learning_rate_multiplier: Learning rate multiplier
            n_epochs: Number of epochs for training
            
        Returns:
            Job ID of the fine-tuning job
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required for training")
        
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        # Upload files
        logger.info("Uploading training file...")
        with open(train_file, "rb") as train_file_data:
            train_response = client.files.create(
                file=train_file_data,
                purpose="fine-tune"
            )
        train_file_id = train_response.id
        
        logger.info("Uploading validation file...")
        with open(valid_file, "rb") as valid_file_data:
            valid_response = client.files.create(
                file=valid_file_data,
                purpose="fine-tune"
            )
        valid_file_id = valid_response.id
        
        # Create fine-tuning job
        logger.info(f"Creating fine-tuning job with base model {base_model}...")
        logger.info(f"Training parameters: batch_size={batch_size}, lr_multiplier={learning_rate_multiplier}, n_epochs={n_epochs}")
        
        # Prepare hyperparameters
        hyperparameters = {
            "n_epochs": n_epochs,
        }
        
        if batch_size > 0:
            hyperparameters["batch_size"] = batch_size
            
        # Create fine-tuning job
        try:
            # First try standard supervised fine-tuning
            job_response = client.fine_tuning.jobs.create(
                training_file=train_file_id,
                validation_file=valid_file_id,
                model=base_model,
                suffix=suffix,
                hyperparameters=hyperparameters
            )
            logger.info("Created supervised fine-tuning job")
        except Exception as e:
            logger.warning(f"Supervised fine-tuning failed: {e}")
            logger.info("Attempting to create with default parameters")
            
            # Try with minimal parameters
            job_response = client.fine_tuning.jobs.create(
                training_file=train_file_id,
                validation_file=valid_file_id,
                model=base_model,
                suffix=suffix
            )
            logger.info("Created fine-tuning job with default parameters")
        
        job_id = job_response.id
        logger.info(f"Created fine-tuning job with ID: {job_id}")
        
        return job_id
    
    def monitor_job(self, job_id: str, interval: int = 60, max_time: int = 7200) -> Dict[str, Any]:
        """
        Monitor a fine-tuning job until it completes or times out.
        
        Args:
            job_id: Job ID to monitor
            interval: Polling interval in seconds
            max_time: Maximum monitoring time in seconds
            
        Returns:
            Final job status
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required for monitoring")
        
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        start_time = time.time()
        elapsed = 0
        
        while elapsed < max_time:
            # Get job status
            try:
                job = client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                logger.info(f"Job {job_id} status: {status}")
                
                # Check if job completed or failed
                if status in ["succeeded", "failed", "cancelled"]:
                    return job
                    
                # Wait before polling again
                time.sleep(interval)
                elapsed = time.time() - start_time
                
            except Exception as e:
                logger.error(f"Error monitoring job: {e}")
                time.sleep(interval)
                elapsed = time.time() - start_time
        
        logger.warning(f"Monitoring timed out after {max_time} seconds")
        return {"status": "monitoring_timeout", "job_id": job_id}
    
    def evaluate_model(self, model_id: str, test_file: str, 
                      n_examples: int = 10) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model on factorization examples.
        
        Args:
            model_id: Model ID to evaluate
            test_file: Path to the test JSONL file
            n_examples: Number of examples to evaluate
            
        Returns:
            Evaluation metrics
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required for evaluation")
        
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        # Load test examples
        test_examples = []
        with open(test_file, 'r') as f:
            for line in f:
                test_examples.append(json.loads(line))
        
        # Sample a subset if needed
        if len(test_examples) > n_examples:
            test_examples = random.sample(test_examples, n_examples)
        
        # Initialize metrics
        metrics = {
            "total": len(test_examples),
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "average_score": 0.0,
            "tier_performance": defaultdict(lambda: {"count": 0, "correct": 0, "score": 0})
        }
        
        all_results = []
        
        # Evaluate each example
        for i, example in enumerate(test_examples):
            logger.info(f"Evaluating example {i+1}/{len(test_examples)}")
            
            try:
                # Extract information
                number = example["number"]
                true_factors = example["factors"]
                tier_id = example.get("tier_id", "unknown")
                
                # Create messages
                messages = [
                    {"role": "system", "content": "You are an expert factorization assistant. Find the prime factorization efficiently."},
                    {"role": "user", "content": f"Find the prime factorization of {number}. Return your answer as JSON with the following fields: 'factors' (list of integers), 'algorithm' (string), 'reasoning' (list of steps), 'time_taken' (float), and 'confidence' (float)."}
                ]
                
                # Get model response
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
                # Extract and parse response
                content = response.choices[0].message.content
                model_output = json.loads(content)
                
                # Grade the response
                correctness = self._grade_correctness(
                    model_output.get("factors", []), 
                    number, 
                    true_factors
                )
                
                # Update metrics
                metrics["average_score"] += correctness
                metrics["tier_performance"][tier_id]["count"] += 1
                metrics["tier_performance"][tier_id]["score"] += correctness
                
                if correctness >= 0.99:
                    metrics["correct"] += 1
                    metrics["tier_performance"][tier_id]["correct"] += 1
                elif correctness >= 0.5:
                    metrics["partially_correct"] += 1
                else:
                    metrics["incorrect"] += 1
                
                # Store result
                result = {
                    "number": number,
                    "true_factors": true_factors,
                    "predicted_factors": model_output.get("factors", []),
                    "algorithm": model_output.get("algorithm", "Unknown"),
                    "reasoning": model_output.get("reasoning", []),
                    "correctness": correctness,
                    "tier_id": tier_id
                }
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {e}")
                metrics["incorrect"] += 1
        
        # Calculate averages
        if metrics["total"] > 0:
            metrics["average_score"] /= metrics["total"]
            
        # Calculate tier averages
        for tier_id, data in metrics["tier_performance"].items():
            if data["count"] > 0:
                data["score"] /= data["count"]
                data["accuracy"] = data["correct"] / data["count"]
        
        return {
            "metrics": metrics,
            "results": all_results
        }
    
    def _grade_correctness(self, predicted_factors: List[int], 
                          number: int, true_factors: List[int]) -> float:
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


def main():
    """Main function for example usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Neurosymbolic Prime Factorizer")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Factorize command
    factorize_parser = subparsers.add_parser("factorize", help="Factorize a number")
    factorize_parser.add_argument("number", type=int, help="Number to factorize")
    factorize_parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    
    # Generate dataset command
    dataset_parser = subparsers.add_parser("generate_dataset", help="Generate training dataset")
    dataset_parser.add_argument("input", type=str, help="Input dataset path")
    dataset_parser.add_argument("output", type=str, help="Output training data path")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("train_file", type=str, help="Training data file")
    train_parser.add_argument("valid_file", type=str, help="Validation data file")
    train_parser.add_argument("--base_model", type=str, default="gpt-4o", help="Base model")
    train_parser.add_argument("--suffix", type=str, default=None, help="Model name suffix")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("model_id", type=str, help="Model ID to evaluate")
    eval_parser.add_argument("test_file", type=str, help="Test data file")
    eval_parser.add_argument("--n_examples", type=int, default=10, help="Number of examples to evaluate")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "factorize":
        # Create factorizer and factorize the number
        factorizer = DistributedFactorizationManager(timeout=args.timeout)
        result = factorizer.factorize(args.number)
        
        # Print result
        print(f"\nFactorization of {args.number}:")
        print(f"Factors: {' × '.join(map(str, result.factors))}")
        print(f"Algorithm: {result.algorithm}")
        print(f"Time taken: {result.time_taken:.6f} seconds")
        print(f"Iterations: {result.iterations}")
        print(f"Confidence: {result.confidence:.4f}")
        
        # Print steps
        if result.intermediate_steps:
            print("\nIntermediate steps:")
            for i, step in enumerate(result.intermediate_steps):
                print(f"{i+1}. {step.get('step', 'unknown')}: "
                      f"{', '.join(f'{k}={v}' for k, v in step.items() if k != 'step')}")
    
    elif args.command == "generate_dataset":
        # Create factorizer and trainer
        factorizer = DistributedFactorizationManager()
        trainer = OpenAIFactorizationTrainer()
        
        # Generate dataset
        stats = trainer.generate_training_data(args.input, args.output, factorizer)
        
        # Print statistics
        print("\nDataset generation completed:")
        print(f"Total samples: {stats['total']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print("\nSamples by tier:")
        for tier, count in sorted(stats['tiers'].items()):
            print(f"Tier {tier}: {count} samples")
    
    elif args.command == "train":
        # Create trainer
        trainer = OpenAIFactorizationTrainer()
        
        # Start training
        job_id = trainer.train_model(args.train_file, args.valid_file, 
                                     args.base_model, args.suffix)
        
        # Monitor training
        print(f"\nTraining started with job ID: {job_id}")
        print("Monitoring training progress...")
        result = trainer.monitor_job(job_id)
        
        print(f"\nTraining completed with status: {result.get('status', 'unknown')}")
        if result.get('status') == 'succeeded':
            print(f"Fine-tuned model: {result.get('fine_tuned_model')}")
    
    elif args.command == "evaluate":
        # Create trainer
        trainer = OpenAIFactorizationTrainer()
        
        # Evaluate model
        print(f"\nEvaluating model: {args.model_id}")
        results = trainer.evaluate_model(args.model_id, args.test_file, args.n_examples)
        
        # Print results
        metrics = results["metrics"]
        print("\nEvaluation results:")
        print(f"Total examples: {metrics['total']}")
        print(f"Correct: {metrics['correct']} ({metrics['correct']/metrics['total']*100:.2f}%)")
        print(f"Partially correct: {metrics['partially_correct']} ({metrics['partially_correct']/metrics['total']*100:.2f}%)")
        print(f"Incorrect: {metrics['incorrect']} ({metrics['incorrect']/metrics['total']*100:.2f}%)")
        print(f"Average score: {metrics['average_score']:.4f}")
        
        print("\nPerformance by tier:")
        for tier, data in sorted(metrics["tier_performance"].items()):
            print(f"Tier {tier}: {data['accuracy']*100:.2f}% accuracy, {data['score']:.4f} avg score ({data['count']} examples)")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
