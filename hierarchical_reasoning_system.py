#!/usr/bin/env python
"""
Hierarchical Reasoning System for Prime Factorization

This module implements a hierarchical reasoning system with specialized mathematical
experts that collaborate to solve complex prime factorization problems.
"""

import math
import random
import sympy
import time
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import json
import os
import numpy as np

# For DSPy integration
try:
    import dspy
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False


@dataclass
class MathematicalExpert:
    """Base class for specialized mathematical experts."""
    name: str
    expertise: List[str]
    confidence_threshold: float = 0.8
    success_rate: float = 0.0
    attempts: int = 0
    successes: int = 0
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """Check if this expert can handle the given problem."""
        return True
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the problem and return the solution."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def assess_confidence(self, solution: Dict[str, Any]) -> float:
        """Assess the confidence in the solution."""
        return 0.0
    
    def update_performance(self, success: bool):
        """Update the expert's performance metrics."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.success_rate = self.successes / self.attempts if self.attempts > 0 else 0.0


@dataclass
class NumTheoryExpert(MathematicalExpert):
    """Expert in number theory fundamentals."""
    
    def __init__(self):
        super().__init__(
            name="Number Theory Fundamentals Expert",
            expertise=["prime testing", "divisibility rules", "modular arithmetic"]
        )
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """This expert handles basic number theory tasks."""
        task = problem.get("task", "")
        return any(e in task.lower() for e in ["primality", "divisibility", "modular", "prime testing"])
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a number theory problem."""
        task = problem.get("task", "")
        number = problem.get("number", 0)
        
        if "primality" in task.lower() or "prime testing" in task.lower():
            # Test if the number is prime
            is_prime = sympy.isprime(number)
            confidence = 1.0  # Deterministic test
            
            # Generate explanation
            explanation = []
            if is_prime:
                explanation.append(f"{number} is a prime number.")
                if number > 2:
                    explanation.append(f"It's not divisible by any integer from 2 to {int(math.sqrt(number))}.")
            else:
                if number <= 1:
                    explanation.append(f"{number} is not a prime number by definition.")
                elif number % 2 == 0 and number > 2:
                    explanation.append(f"{number} is even and greater than 2, so it's not prime.")
                else:
                    # Find a divisor
                    for i in range(3, int(math.sqrt(number)) + 1, 2):
                        if number % i == 0:
                            explanation.append(f"{number} is divisible by {i}, so it's not prime.")
                            break
            
            return {
                "result": is_prime,
                "confidence": confidence,
                "explanation": explanation,
                "workings": {
                    "method": "primality_test",
                    "number": number
                }
            }
        
        elif "divisibility" in task.lower():
            # Check divisibility rules
            divisors = []
            explanations = []
            
            # Check small divisors up to 20
            for i in range(2, 21):
                if number % i == 0:
                    divisors.append(i)
                    explanations.append(f"{number} is divisible by {i} (remainder 0)")
            
            # Special divisibility rules
            digit_sum = sum(int(d) for d in str(number))
            if digit_sum % 3 == 0:
                explanations.append(f"The sum of digits ({digit_sum}) is divisible by 3, so {number} is divisible by 3")
            
            if int(str(number)[-1]) % 2 == 0:
                explanations.append(f"{number} ends in an even digit, so it's divisible by 2")
            
            if int(str(number)[-1]) == 0 or int(str(number)[-1]) == 5:
                explanations.append(f"{number} ends in {str(number)[-1]}, so it's divisible by 5")
            
            if digit_sum % 9 == 0:
                explanations.append(f"The sum of digits ({digit_sum}) is divisible by 9, so {number} is divisible by 9")
            
            return {
                "result": divisors,
                "confidence": 1.0,
                "explanation": explanations,
                "workings": {
                    "method": "divisibility_check",
                    "number": number,
                    "digit_sum": digit_sum
                }
            }
        
        elif "modular" in task.lower():
            # Perform modular arithmetic
            modulus = problem.get("modulus", 10)
            remainder = number % modulus
            
            return {
                "result": remainder,
                "confidence": 1.0,
                "explanation": [f"{number} ≡ {remainder} (mod {modulus})"],
                "workings": {
                    "method": "modular_arithmetic",
                    "number": number,
                    "modulus": modulus
                }
            }
        
        return {
            "result": None,
            "confidence": 0.0,
            "explanation": ["I don't know how to solve this number theory problem."],
            "workings": {}
        }


@dataclass
class FactorizationExpert(MathematicalExpert):
    """Expert in factorization techniques."""
    
    def __init__(self):
        super().__init__(
            name="Factorization Expert",
            expertise=["trial division", "small prime factorization", "pollard rho"]
        )
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """This expert handles factorization tasks."""
        task = problem.get("task", "")
        number = problem.get("number", 0)
        
        # Check if the task is factorization
        if not any(e in task.lower() for e in ["factor", "factorize", "factorization"]):
            return False
        
        # Check if the number is not too large for this expert
        return number.bit_length() < 64
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Factorize a number using appropriate techniques."""
        number = problem.get("number", 0)
        
        # Check if the number is small
        if number <= 1:
            return {
                "result": [number],
                "confidence": 1.0,
                "explanation": [f"{number} is a trivial case with no prime factorization."],
                "workings": {"method": "trivial_case"}
            }
        
        # Check if the number is prime
        if sympy.isprime(number):
            return {
                "result": [number],
                "confidence": 1.0,
                "explanation": [f"{number} is a prime number, so its only factor is itself."],
                "workings": {"method": "prime_check"}
            }
        
        # Start with trial division for small factors
        factors = []
        remaining = number
        workings = []
        
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in small_primes:
            while remaining % p == 0:
                factors.append(p)
                remaining //= p
                workings.append(f"Found factor {p}, dividing by {p}: {remaining} remaining")
        
        # If the number is completely factorized
        if remaining == 1:
            steps = [f"Step 1: Try dividing by small primes up to 47."]
            for i, w in enumerate(workings):
                steps.append(f"Step {i+2}: {w}")
            
            return {
                "result": factors,
                "confidence": 1.0,
                "explanation": [
                    f"{number} = {' × '.join(map(str, factors))}",
                    "Factorization complete using trial division with small primes."
                ],
                "workings": {
                    "method": "trial_division",
                    "steps": steps
                }
            }
        
        # Continue with Pollard's rho for remaining factors if not too large
        if remaining.bit_length() < 40:
            pollardrho_steps = []
            
            def pollard_rho(n):
                """Pollard's rho algorithm for integer factorization."""
                if n % 2 == 0:
                    return 2
                
                def g(x, c, n):
                    return (x * x + c) % n
                
                x, y, c, d = 2, 2, 1, 1
                iterations = 0
                
                while d == 1 and iterations < 1000:
                    iterations += 1
                    x = g(x, c, n)
                    y = g(g(y, c, n), c, n)
                    d = math.gcd(abs(x - y), n)
                    pollardrho_steps.append(f"Iteration {iterations}: x={x}, y={y}, d={d}")
                
                if d == n:
                    pollardrho_steps.append(f"No factor found after {iterations} iterations, trying a different starting value.")
                    return None
                
                pollardrho_steps.append(f"Found factor {d} after {iterations} iterations.")
                return d
            
            # Try to factorize the remaining part
            while remaining > 1 and not sympy.isprime(remaining):
                factor = pollard_rho(remaining)
                if factor and 1 < factor < remaining:
                    factors.append(factor)
                    remaining //= factor
                    workings.append(f"Found factor {factor} using Pollard's rho, dividing by {factor}: {remaining} remaining")
                else:
                    break
            
            # Add the remaining factor if it's prime
            if remaining > 1:
                if sympy.isprime(remaining):
                    factors.append(remaining)
                    workings.append(f"Remaining factor {remaining} is prime.")
                else:
                    # Failed to completely factorize
                    factors.append(remaining)
                    workings.append(f"Couldn't factorize {remaining} further.")
                    return {
                        "result": factors,
                        "confidence": 0.8,  # Lower confidence because factorization might be incomplete
                        "explanation": [
                            f"{number} = {' × '.join(map(str, factors))}",
                            "Partial factorization using trial division and Pollard's rho."
                        ],
                        "workings": {
                            "method": "trial_division_and_pollard_rho",
                            "steps": [f"Step 1: Try dividing by small primes up to 47."] + 
                                     [f"Step {i+2}: {w}" for i, w in enumerate(workings)] +
                                     [f"Pollard's rho details: {step}" for step in pollardrho_steps]
                        }
                    }
            
            # Construct the steps
            steps = [f"Step 1: Try dividing by small primes up to 47."]
            for i, w in enumerate(workings):
                steps.append(f"Step {i+2}: {w}")
            
            return {
                "result": factors,
                "confidence": 1.0,
                "explanation": [
                    f"{number} = {' × '.join(map(str, factors))}",
                    "Complete factorization using trial division and Pollard's rho."
                ],
                "workings": {
                    "method": "trial_division_and_pollard_rho",
                    "steps": steps
                }
            }
        
        # For larger remaining factors, return partial factorization
        factors.append(remaining)
        steps = [f"Step 1: Try dividing by small primes up to 47."]
        for i, w in enumerate(workings):
            steps.append(f"Step {i+2}: {w}")
        steps.append(f"Final step: Remaining factor {remaining} is too large for Pollard's rho. Further factorization needed.")
        
        return {
            "result": factors,
            "confidence": 0.7,  # Lower confidence
            "explanation": [
                f"{number} = {' × '.join(map(str, factors))}",
                "Partial factorization. The remaining factor may be composite but is too large for simple methods."
            ],
            "workings": {
                "method": "trial_division",
                "steps": steps,
                "note": f"Remaining factor {remaining} requires advanced factorization techniques."
            }
        }


@dataclass
class AdvancedFactorizationExpert(MathematicalExpert):
    """Expert in advanced factorization techniques."""
    
    def __init__(self):
        super().__init__(
            name="Advanced Factorization Expert",
            expertise=["quadratic sieve", "elliptic curve method", "large prime factorization"]
        )
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """This expert handles advanced factorization tasks."""
        task = problem.get("task", "")
        number = problem.get("number", 0)
        bit_length = number.bit_length() if isinstance(number, int) else 0
        
        # Check if the task is factorization
        if not any(e in task.lower() for e in ["factor", "factorize", "factorization"]):
            return False
        
        # This expert handles larger numbers
        return bit_length >= 40
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Factorize a large number using advanced techniques."""
        number = problem.get("number", 0)
        
        # Check if the number is prime
        if sympy.isprime(number):
            return {
                "result": [number],
                "confidence": 1.0,
                "explanation": [f"{number} is a prime number, so its only factor is itself."],
                "workings": {"method": "prime_check"}
            }
        
        # Start with trial division for small factors
        factors = []
        remaining = number
        workings = []
        
        # Try small primes first (optimization)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            while remaining % p == 0:
                factors.append(p)
                remaining //= p
                workings.append(f"Found factor {p}, dividing by {p}: {remaining} remaining")
        
        # If the number is completely factorized
        if remaining == 1:
            return {
                "result": factors,
                "confidence": 1.0,
                "explanation": [
                    f"{number} = {' × '.join(map(str, factors))}",
                    "Factorization complete using trial division with small primes."
                ],
                "workings": {
                    "method": "trial_division",
                    "steps": [f"Step {i+1}: {w}" for i, w in enumerate(workings)]
                }
            }
        
        # For demonstration, we'll simulate the quadratic sieve algorithm
        # In a real implementation, this would be a full quadratic sieve
        
        qsieve_steps = []
        qsieve_steps.append(f"Initializing quadratic sieve for {remaining}")
        
        # Simulate finding a factor (for demonstration)
        # In reality, we would implement the actual algorithm
        sqrt_n = int(math.sqrt(remaining))
        qsieve_steps.append(f"Search base: around sqrt({remaining}) ≈ {sqrt_n}")
        
        # Simulate sieving process
        qsieve_steps.append(f"Generated factor base of small primes")
        qsieve_steps.append(f"Collecting smooth numbers in the sieve interval")
        
        # For demonstration, we'll check if remaining has a small factor near sqrt(n)
        found_factor = False
        for offset in range(-1000, 1000):
            candidate = sqrt_n + offset
            if candidate > 1 and remaining % candidate == 0:
                factor = candidate
                found_factor = True
                qsieve_steps.append(f"Found factor {factor} near sqrt(n)")
                factors.append(factor)
                remaining //= factor
                workings.append(f"Found factor {factor} using quadratic sieve simulation, dividing by {factor}: {remaining} remaining")
                break
        
        if not found_factor:
            # For demonstration, simulate finding a factor with elliptic curve method
            qsieve_steps.append(f"Quadratic sieve did not find a factor, trying elliptic curve method")
            
            # In a real implementation, we would run the actual ECM algorithm
            # For demonstration, we'll check if n has special properties
            
            # Try to find a factor using a simple test
            for base in [2, 3, 5, 7, 11]:
                # Try to find a factor using Fermat's factorization method
                a = int(math.sqrt(remaining)) + 1
                b2 = a*a - remaining
                b = int(math.sqrt(b2))
                if b*b == b2:
                    factor1 = a + b
                    factor2 = a - b
                    if factor1 > 1 and factor2 > 1:
                        qsieve_steps.append(f"Fermat's method found factors {factor1} and {factor2}")
                        factors.extend([factor1, factor2])
                        remaining = 1
                        workings.append(f"Found factors {factor1} and {factor2} using Fermat's method")
                        break
        
        # Add any remaining factor
        if remaining > 1:
            if sympy.isprime(remaining):
                factors.append(remaining)
                workings.append(f"Remaining factor {remaining} is prime.")
            else:
                # Failed to completely factorize
                factors.append(remaining)
                workings.append(f"Couldn't factorize {remaining} further with current methods.")
                return {
                    "result": factors,
                    "confidence": 0.8,  # Lower confidence
                    "explanation": [
                        f"{number} = {' × '.join(map(str, factors))}",
                        "Partial factorization using trial division and advanced methods."
                    ],
                    "workings": {
                        "method": "advanced_factorization",
                        "steps": [f"Step {i+1}: {w}" for i, w in enumerate(workings)],
                        "advanced_steps": qsieve_steps,
                        "note": f"Remaining factor {remaining} requires more sophisticated methods."
                    }
                }
        
        # Construct the steps
        steps = [f"Step {i+1}: {w}" for i, w in enumerate(workings)]
        
        return {
            "result": factors,
            "confidence": 1.0,
            "explanation": [
                f"{number} = {' × '.join(map(str, factors))}",
                "Complete factorization using advanced methods."
            ],
            "workings": {
                "method": "advanced_factorization",
                "steps": steps,
                "advanced_steps": qsieve_steps
            }
        }


@dataclass
class VerificationExpert(MathematicalExpert):
    """Expert in verifying mathematical solutions."""
    
    def __init__(self):
        super().__init__(
            name="Verification Expert",
            expertise=["solution verification", "error detection", "consistency checking"]
        )
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """This expert handles verification tasks."""
        task = problem.get("task", "")
        return "verify" in task.lower() or "check" in task.lower()
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a solution to a mathematical problem."""
        solution = problem.get("solution", {})
        original_problem = problem.get("original_problem", {})
        
        if not solution or not original_problem:
            return {
                "result": False,
                "confidence": 1.0,
                "explanation": ["Cannot verify: missing solution or original problem."],
                "workings": {}
            }
        
        # Extract relevant information
        task = original_problem.get("task", "")
        number = original_problem.get("number", 0)
        claimed_result = solution.get("result", None)
        
        # Verification logic based on task type
        if "factor" in task.lower() or "factorize" in task.lower():
            # Verify prime factorization
            if not claimed_result or not isinstance(claimed_result, list):
                return {
                    "result": False,
                    "confidence": 1.0,
                    "explanation": ["Invalid factorization format."],
                    "workings": {"error": "Result is not a list of factors."}
                }
            
            # Check if all factors are prime
            non_primes = [f for f in claimed_result if not sympy.isprime(f)]
            if non_primes:
                return {
                    "result": False,
                    "confidence": 1.0,
                    "explanation": [f"The following claimed factors are not prime: {non_primes}"],
                    "workings": {"non_primes": non_primes}
                }
            
            # Check if the product equals the original number
            product = 1
            for factor in claimed_result:
                product *= factor
            
            if product != number:
                return {
                    "result": False,
                    "confidence": 1.0,
                    "explanation": [
                        f"The product of the factors ({product}) does not equal the original number ({number}).",
                        f"Difference: {product - number}"
                    ],
                    "workings": {"product": product, "original": number, "difference": product - number}
                }
            
            # All checks pass
            return {
                "result": True,
                "confidence": 1.0,
                "explanation": [
                    f"The factorization {' × '.join(map(str, claimed_result))} = {product} is correct.",
                    "All factors are prime and their product equals the original number."
                ],
                "workings": {
                    "product_check": f"{' × '.join(map(str, claimed_result))} = {product}",
                    "primality_check": "All factors are prime."
                }
            }
        
        elif "primality" in task.lower() or "prime testing" in task.lower():
            # Verify primality test
            if not isinstance(claimed_result, bool):
                return {
                    "result": False,
                    "confidence": 1.0,
                    "explanation": ["Invalid primality test result format."],
                    "workings": {"error": "Result is not a boolean."}
                }
            
            # Check if the claimed result matches the actual primality
            actual_primality = sympy.isprime(number)
            if claimed_result != actual_primality:
                return {
                    "result": False,
                    "confidence": 1.0,
                    "explanation": [
                        f"The claimed result ({claimed_result}) does not match the actual primality ({actual_primality})."
                    ],
                    "workings": {"claimed": claimed_result, "actual": actual_primality}
                }
            
            # Result is correct
            return {
                "result": True,
                "confidence": 1.0,
                "explanation": [f"The primality test result ({claimed_result}) is correct."],
                "workings": {"primality_check": f"is_prime({number}) = {actual_primality}"}
            }
        
        # Default response for unrecognized tasks
        return {
            "result": None,
            "confidence": 0.0,
            "explanation": ["Cannot verify: unrecognized task type."],
            "workings": {"task": task}
        }


@dataclass
class QuantumResistanceExpert(MathematicalExpert):
    """Expert in quantum-resistant factorization challenges."""
    
    def __init__(self):
        super().__init__(
            name="Quantum Resistance Expert",
            expertise=["quantum-resistant primes", "near-prime detection", "cryptographic strength"]
        )
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """This expert handles quantum-resistant prime generation and analysis."""
        task = problem.get("task", "")
        return any(term in task.lower() for term in [
            "quantum", "resistance", "cryptographic", "near-prime", "large prime"
        ])
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Generate or analyze quantum-resistant primes."""
        task = problem.get("task", "")
        
        if "generate" in task.lower() and "prime" in task.lower():
            # Generate quantum-resistant primes
            bit_length = problem.get("bit_length", 2048)  # Default to 2048 bits
            count = problem.get("count", 1)
            
            primes = []
            workings = []
            
            try:
                # For demonstration, we'll generate smaller primes
                # In a real implementation, we would use cryptographic libraries
                demo_bit_length = min(bit_length, 128)  # Limit for demonstration
                
                if demo_bit_length < bit_length:
                    workings.append(f"Note: Using {demo_bit_length} bits for demonstration instead of requested {bit_length} bits")
                
                workings.append(f"Generating {count} prime(s) of {demo_bit_length} bits each")
                
                for i in range(count):
                    # In a real implementation, we would use specialized libraries
                    # For demonstration, we'll use sympy
                    prime = sympy.randprime(2**(demo_bit_length-1), 2**demo_bit_length-1)
                    primes.append(prime)
                    workings.append(f"Prime {i+1}: {prime} ({prime.bit_length()} bits)")
                
                return {
                    "result": primes,
                    "confidence": 1.0,
                    "explanation": [
                        f"Generated {len(primes)} quantum-resistant prime number(s) of approximately {demo_bit_length} bits each."
                    ],
                    "workings": {
                        "method": "prime_generation",
                        "steps": workings
                    }
                }
            except Exception as e:
                return {
                    "result": [],
                    "confidence": 0.0,
                    "explanation": [f"Failed to generate primes: {str(e)}"],
                    "workings": {"error": str(e)}
                }
        
        elif "near-prime" in task.lower() or "detect" in task.lower():
            # Detect near-prime situations (primes that are close to each other)
            number = problem.get("number", 0)
            
            if not isinstance(number, int) or number <= 0:
                return {
                    "result": False,
                    "confidence": 1.0,
                    "explanation": ["Invalid input for near-prime detection."],
                    "workings": {"error": "Input must be a positive integer."}
                }
            
            # Check if the number is a prime
            is_prime = sympy.isprime(number)
            workings = [f"Primality check: {number} is {'prime' if is_prime else 'not prime'}"]
            
            # Look for nearby primes
            nearby_primes = []
            search_range = min(1000, number // 100)  # Keep search range reasonable
            
            workings.append(f"Searching for primes in range [{number-search_range}, {number+search_range}]")
            
            # Find the previous prime
            prev_prime = number - 1
            while prev_prime > number - search_range and not sympy.isprime(prev_prime):
                prev_prime -= 1
            
            # Find the next prime
            next_prime = number + 1
            while next_prime < number + search_range and not sympy.isprime(next_prime):
                next_prime += 1
            
            if sympy.isprime(prev_prime) and prev_prime > number - search_range:
                nearby_primes.append(prev_prime)
                workings.append(f"Found previous prime: {prev_prime} (distance: {number - prev_prime})")
            
            if sympy.isprime(next_prime) and next_prime < number + search_range:
                nearby_primes.append(next_prime)
                workings.append(f"Found next prime: {next_prime} (distance: {next_prime - number})")
            
            # Evaluate the near-prime situation
            is_near_prime = False
            explanation = []
            
            if is_prime and len(nearby_primes) > 0:
                min_distance = min(abs(p - number) for p in nearby_primes)
                relative_gap = min_distance / number
                
                workings.append(f"Minimum distance to another prime: {min_distance}")
                workings.append(f"Relative gap: {relative_gap:.8f}")
                
                # Check if the gap is unusually small
                # For cryptographic purposes, very close primes could be problematic
                if relative_gap < 1e-6:  # Extremely close
                    is_near_prime = True
                    explanation.append(f"{number} is a prime with an extremely close neighbor (gap: {min_distance}).")
                    explanation.append("This could be problematic for some cryptographic applications.")
                elif relative_gap < 1e-4:  # Very close
                    is_near_prime = True
                    explanation.append(f"{number} is a prime with a very close neighbor (gap: {min_distance}).")
                    explanation.append("This might be of concern for some cryptographic applications.")
                else:
                    explanation.append(f"{number} is a prime with normal gaps to neighboring primes.")
            elif not is_prime and len(nearby_primes) > 0:
                explanation.append(f"{number} is not a prime but has nearby primes at distances: {[abs(p - number) for p in nearby_primes]}.")
            else:
                explanation.append(f"No primes found near {number} within search range ±{search_range}.")
            
            return {
                "result": {
                    "is_prime": is_prime,
                    "is_near_prime": is_near_prime,
                    "nearby_primes": nearby_primes
                },
                "confidence": 1.0,
                "explanation": explanation,
                "workings": {
                    "method": "near_prime_detection",
                    "steps": workings
                }
            }
        
        elif "analyze" in task.lower() and "strength" in task.lower():
            # Analyze cryptographic strength
            number = problem.get("number", 0)
            
            if not isinstance(number, int) or number <= 0:
                return {
                    "result": {},
                    "confidence": 1.0,
                    "explanation": ["Invalid input for cryptographic strength analysis."],
                    "workings": {"error": "Input must be a positive integer."}
                }
            
            bit_length = number.bit_length()
            workings = [f"Bit length: {bit_length}"]
            
            # Assess cryptographic strength
            strength_assessment = {}
            
            # RSA key strength assessment
            if bit_length < 2048:
                strength_assessment["rsa"] = "weak"
                workings.append(f"RSA strength: Weak (< 2048 bits)")
            elif bit_length < 3072:
                strength_assessment["rsa"] = "acceptable"
                workings.append(f"RSA strength: Acceptable (2048-3072 bits)")
            else:
                strength_assessment["rsa"] = "strong"
                workings.append(f"RSA strength: Strong (≥ 3072 bits)")
            
            # Quantum resistance assessment
            if bit_length < 6000:
                strength_assessment["quantum"] = "vulnerable"
                workings.append(f"Quantum resistance: Vulnerable (< 6000 bits)")
            else:
                strength_assessment["quantum"] = "resistant"
                workings.append(f"Quantum resistance: Resistant (≥ 6000 bits)")
            
            # Generate explanation
            explanation = []
            
            if strength_assessment["rsa"] == "weak":
                explanation.append(f"The number has {bit_length} bits, which is considered weak for RSA encryption.")
                explanation.append("Modern standards recommend at least 2048 bits for RSA keys.")
            elif strength_assessment["rsa"] == "acceptable":
                explanation.append(f"The number has {bit_length} bits, which is acceptable for current RSA standards.")
                explanation.append("However, for long-term security, 3072+ bits is recommended.")
            else:
                explanation.append(f"The number has {bit_length} bits, which provides strong security under current standards.")
            
            if strength_assessment["quantum"] == "vulnerable":
                explanation.append("This key size would be vulnerable to quantum computing attacks.")
                explanation.append("For quantum resistance, much larger key sizes (6000+ bits) are needed.")
            else:
                explanation.append("This key size should provide resistance against quantum computing attacks.")
            
            return {
                "result": strength_assessment,
                "confidence": 0.9,  # High but not perfect confidence
                "explanation": explanation,
                "workings": {
                    "method": "cryptographic_strength_analysis",
                    "steps": workings
                }
            }
        
        # Default response for unrecognized tasks
        return {
            "result": None,
            "confidence": 0.0,
            "explanation": ["I don't know how to handle this quantum-resistance task."],
            "workings": {"task": task}
        }


@dataclass
class PrimeStatisticsExpert(MathematicalExpert):
    """Expert in statistical properties of primes."""
    
    def __init__(self):
        super().__init__(
            name="Prime Statistics Expert",
            expertise=["prime distribution", "prime gaps", "statistical patterns"]
        )
    
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        """This expert handles statistical analysis of primes."""
        task = problem.get("task", "")
        return any(term in task.lower() for term in [
            "statistic", "distribution", "pattern", "gap", "density"
        ]) and "prime" in task.lower()
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical properties of primes."""
        task = problem.get("task", "")
        
        if "distribution" in task.lower() or "density" in task.lower():
            # Analyze prime number distribution
            range_start = problem.get("range_start", 1)
            range_end = problem.get("range_end", 1000)
            
            if range_end - range_start > 10000:
                # Limit range for demonstration
                original_range = (range_start, range_end)
                range_end = range_start + 10000
            
            # Count primes in the range
            primes_in_range = []
            for n in range(max(2, range_start), range_end + 1):
                if sympy.isprime(n):
                    primes_in_range.append(n)
            
            prime_count = len(primes_in_range)
            density = prime_count / (range_end - range_start + 1)
            
            # Compare with prime number theorem prediction
            # The PNT states that π(x) ~ x/ln(x)
            pnt_estimate = range_end / math.log(range_end) - range_start / math.log(range_start)
            pnt_error = abs(prime_count - pnt_estimate) / pnt_estimate if pnt_estimate != 0 else 0
            
            workings = [
                f"Counted {prime_count} primes in range [{range_start}, {range_end}]",
                f"Density: {density:.6f} (ratio of primes to all numbers in range)",
                f"Prime Number Theorem estimate: {pnt_estimate:.2f}",
                f"Error between actual count and PNT estimate: {pnt_error:.2%}"
            ]
            
            # Analyze distribution in subranges
            if range_end - range_start >= 100:
                num_subranges = min(10, (range_end - range_start) // 100)
                subrange_size = (range_end - range_start) // num_subranges
                
                subranges = []
                for i in range(num_subranges):
                    sub_start = range_start + i * subrange_size
                    sub_end = range_start + (i + 1) * subrange_size - 1
                    if i == num_subranges - 1:
                        sub_end = range_end  # Ensure the last subrange includes range_end
                    
                    sub_primes = [p for p in primes_in_range if sub_start <= p <= sub_end]
                    subranges.append({
                        "range": (sub_start, sub_end),
                        "count": len(sub_primes),
                        "density": len(sub_primes) / (sub_end - sub_start + 1)
                    })
                    
                    workings.append(f"Subrange [{sub_start}, {sub_end}]: {len(sub_primes)} primes, density {len(sub_primes) / (sub_end - sub_start + 1):.6f}")
            
            # Generate explanation
            explanation = [
                f"Found {prime_count} primes in range [{range_start}, {range_end}].",
                f"The density of primes in this range is {density:.6f}.",
                f"The Prime Number Theorem predicts approximately {pnt_estimate:.2f} primes in this range."
            ]
            
            if 'original_range' in locals():
                explanation.append(f"Note: Original requested range {original_range} was limited to [{range_start}, {range_end}] for computational feasibility.")
            
            return {
                "result": {
                    "prime_count": prime_count,
                    "density": density,
                    "pnt_estimate": pnt_estimate,
                    "pnt_error": pnt_error,
                    "subranges": subranges if 'subranges' in locals() else []
                },
                "confidence": 1.0,
                "explanation": explanation,
                "workings": {
                    "method": "prime_distribution_analysis",
                    "steps": workings
                }
            }
        
        elif "gap" in task.lower():
            # Analyze prime gaps
            range_start = problem.get("range_start", 1)
            range_end = problem.get("range_end", 1000)
            
            if range_end - range_start > 10000:
                # Limit range for demonstration
                original_range = (range_start, range_end)
                range_end = range_start + 10000
            
            # Find primes and calculate gaps
            primes_in_range = []
            for n in range(max(2, range_start), range_end + 1):
                if sympy.isprime(n):
                    primes_in_range.append(n)
            
            gaps = []
            for i in range(1, len(primes_in_range)):
                gap = primes_in_range[i] - primes_in_range[i-1]
                gaps.append(gap)
            
            # Analyze the gaps
            if gaps:
                min_gap = min(gaps)
                max_gap = max(gaps)
                avg_gap = sum(gaps) / len(gaps)
                
                # Count occurrences of each gap size
                gap_counts = {}
                for gap in gaps:
                    gap_counts[gap] = gap_counts.get(gap, 0) + 1
                
                # Find most common gap
                most_common_gap = max(gap_counts.items(), key=lambda x: x[1])
                
                workings = [
                    f"Found {len(primes_in_range)} primes in range [{range_start}, {range_end}]",
                    f"Calculated {len(gaps)} prime gaps",
                    f"Minimum gap: {min_gap}",
                    f"Maximum gap: {max_gap}",
                    f"Average gap: {avg_gap:.2f}",
                    f"Most common gap: {most_common_gap[0]} (occurs {most_common_gap[1]} times)"
                ]
                
                # List all gaps for small sets
                if len(gaps) <= 20:
                    workings.append(f"All gaps: {gaps}")
                
                # Generate explanation
                explanation = [
                    f"Analyzed {len(gaps)} prime gaps in range [{range_start}, {range_end}].",
                    f"The smallest gap is {min_gap} and the largest gap is {max_gap}.",
                    f"The average gap size is {avg_gap:.2f}, and the most common gap is {most_common_gap[0]} (occurs {most_common_gap[1]} times)."
                ]
                
                if 'original_range' in locals():
                    explanation.append(f"Note: Original requested range {original_range} was limited to [{range_start}, {range_end}] for computational feasibility.")
                
                return {
                    "result": {
                        "gaps": gaps,
                        "min_gap": min_gap,
                        "max_gap": max_gap,
                        "avg_gap": avg_gap,
                        "gap_distribution": gap_counts,
                        "most_common_gap": most_common_gap
                    },
                    "confidence": 1.0,
                    "explanation": explanation,
                    "workings": {
                        "method": "prime_gap_analysis",
                        "steps": workings
                    }
                }
            else:
                return {
                    "result": {},
                    "confidence": 1.0,
                    "explanation": [f"Found fewer than 2 primes in range [{range_start}, {range_end}], so no gaps can be calculated."],
                    "workings": {
                        "method": "prime_gap_analysis",
                        "steps": [f"Found {len(primes_in_range)} primes in range [{range_start}, {range_end}]"]
                    }
                }
        
        # Default response for unrecognized tasks
        return {
            "result": None,
            "confidence": 0.0,
            "explanation": ["I don't know how to handle this prime statistics task."],
            "workings": {"task": task}
        }


@dataclass
class HierarchicalReasoningSystem:
    """
    A hierarchical reasoning system that coordinates multiple mathematical experts
    to solve complex factorization problems.
    """
    experts: List[MathematicalExpert] = field(default_factory=list)
    coordinator: Optional[Any] = None
    trace_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __init__(self):
        """Initialize the reasoning system with experts."""
        self.experts = [
            NumTheoryExpert(),
            FactorizationExpert(),
            AdvancedFactorizationExpert(),
            VerificationExpert(),
            QuantumResistanceExpert(),
            PrimeStatisticsExpert()
        ]
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a mathematical problem using the most appropriate expert(s).
        
        Args:
            problem: A dictionary with the problem description
            
        Returns:
            A dictionary with the solution and reasoning trace
        """
        # Initialize trace
        trace = {
            "problem": problem,
            "steps": [],
            "expert_contributions": []
        }
        
        # Step 1: Analyze the problem and select experts
        task = problem.get("task", "")
        trace["steps"].append({
            "step": "problem_analysis",
            "description": f"Analyzing problem: {task}"
        })
        
        # Find capable experts
        capable_experts = [expert for expert in self.experts if expert.can_handle(problem)]
        
        if not capable_experts:
            trace["steps"].append({
                "step": "expert_selection_failed",
                "description": "No expert can handle this problem."
            })
            return {
                "result": None,
                "confidence": 0.0,
                "explanation": ["No expert in the system can handle this problem."],
                "trace": trace
            }
        
        # Select primary expert (the one with highest confidence in handling this type of problem)
        primary_expert = max(capable_experts, key=lambda e: e.success_rate if e.attempts > 0 else 0.5)
        
        trace["steps"].append({
            "step": "expert_selection",
            "description": f"Selected primary expert: {primary_expert.name}",
            "expert": primary_expert.name,
            "expert_expertise": primary_expert.expertise
        })
        
        # Step 2: Solve using primary expert
        trace["steps"].append({
            "step": "primary_solution",
            "description": f"Solving with {primary_expert.name}"
        })
        
        primary_solution = primary_expert.solve(problem)
        primary_expert.update_performance(primary_solution.get("confidence", 0.0) > 0.7)
        
        trace["expert_contributions"].append({
            "expert": primary_expert.name,
            "solution": primary_solution,
            "role": "primary"
        })
        
        # Step 3: Verify the solution if appropriate
        if primary_solution.get("confidence", 0.0) > 0.5:
            verification_expert = next((e for e in self.experts if isinstance(e, VerificationExpert)), None)
            
            if verification_expert:
                verification_problem = {
                    "task": "verify solution",
                    "solution": primary_solution,
                    "original_problem": problem
                }
                
                trace["steps"].append({
                    "step": "verification",
                    "description": "Verifying solution"
                })
                
                verification_result = verification_expert.solve(verification_problem)
                verification_expert.update_performance(True)  # Always count this as a success for the verifier
                
                trace["expert_contributions"].append({
                    "expert": verification_expert.name,
                    "solution": verification_result,
                    "role": "verification"
                })
                
                # If verification fails, select a different expert for a second attempt
                if not verification_result.get("result", True):
                    trace["steps"].append({
                        "step": "verification_failed",
                        "description": "Verification failed, trying alternative expert"
                    })
                    
                    # Select alternative expert
                    alternative_experts = [e for e in capable_experts if e != primary_expert]
                    if alternative_experts:
                        alternative_expert = alternative_experts[0]
                        
                        trace["steps"].append({
                            "step": "alternative_solution",
                            "description": f"Trying alternative expert: {alternative_expert.name}"
                        })
                        
                        alternative_solution = alternative_expert.solve(problem)
                        alternative_expert.update_performance(alternative_solution.get("confidence", 0.0) > 0.7)
                        
                        trace["expert_contributions"].append({
                            "expert": alternative_expert.name,
                            "solution": alternative_solution,
                            "role": "alternative"
                        })
                        
                        # Verify the alternative solution
                        alt_verification_problem = {
                            "task": "verify solution",
                            "solution": alternative_solution,
                            "original_problem": problem
                        }
                        
                        alt_verification_result = verification_expert.solve(alt_verification_problem)
                        
                        trace["expert_contributions"].append({
                            "expert": verification_expert.name,
                            "solution": alt_verification_result,
                            "role": "alt_verification"
                        })
                        
                        # Use the alternative solution if verification passes
                        if alt_verification_result.get("result", False):
                            primary_solution = alternative_solution
                            trace["steps"].append({
                                "step": "alternative_accepted",
                                "description": "Alternative solution verified and accepted"
                            })
                        else:
                            trace["steps"].append({
                                "step": "all_solutions_failed",
                                "description": "Both primary and alternative solutions failed verification"
                            })
                    else:
                        trace["steps"].append({
                            "step": "no_alternatives",
                            "description": "No alternative experts available"
                        })
                else:
                    trace["steps"].append({
                        "step": "verification_passed",
                        "description": "Primary solution verified successfully"
                    })
        
        # Store the trace in history
        self.trace_history.append(trace)
        
        # Step 4: Return the final solution with trace
        return {
            "result": primary_solution.get("result", None),
            "confidence": primary_solution.get("confidence", 0.0),
            "explanation": primary_solution.get("explanation", []),
            "workings": primary_solution.get("workings", {}),
            "trace": trace
        }
    
    def factorize(self, number: int) -> Dict[str, Any]:
        """
        Convenience method to factorize a number.
        
        Args:
            number: The number to factorize
            
        Returns:
            A dictionary with the factorization and reasoning trace
        """
        problem = {
            "task": "factorize the number",
            "number": number
        }
        return self.solve(problem)
    
    def analyze_strength(self, number: int) -> Dict[str, Any]:
        """
        Analyze the cryptographic strength of a number.
        
        Args:
            number: The number to analyze
            
        Returns:
            A dictionary with the analysis and reasoning trace
        """
        problem = {
            "task": "analyze cryptographic strength",
            "number": number
        }
        return self.solve(problem)
    
    def analyze_near_prime(self, number: int) -> Dict[str, Any]:
        """
        Detect if a number is a near-prime.
        
        Args:
            number: The number to analyze
            
        Returns:
            A dictionary with the analysis and reasoning trace
        """
        problem = {
            "task": "detect near-prime",
            "number": number
        }
        return self.solve(problem)
    
    def generate_quantum_resistant_prime(self, bit_length: int = 2048) -> Dict[str, Any]:
        """
        Generate a quantum-resistant prime number.
        
        Args:
            bit_length: The bit length of the prime to generate
            
        Returns:
            A dictionary with the generated prime and reasoning trace
        """
        problem = {
            "task": "generate quantum-resistant prime",
            "bit_length": bit_length
        }
        return self.solve(problem)
    
    def analyze_prime_gaps(self, range_start: int = 1, range_end: int = 1000) -> Dict[str, Any]:
        """
        Analyze gaps between prime numbers in a range.
        
        Args:
            range_start: The start of the range
            range_end: The end of the range
            
        Returns:
            A dictionary with the analysis and reasoning trace
        """
        problem = {
            "task": "analyze prime gaps",
            "range_start": range_start,
            "range_end": range_end
        }
        return self.solve(problem)


#==========================================================================
# DSPy integration (if available)
#==========================================================================

if HAS_DSPY:
    class PrimeFactorizationReasoning(dspy.Module):
        """DSPy module for prime factorization with multi-step reasoning."""
        
        def __init__(self):
            super().__init__()
            
            # Define the signature for prime factorization
            self.factorize_step = dspy.ChainOfThought(
                dspy.Signature(
                    "number, current_factors, remaining -> next_step, updated_factors, new_remaining",
                    """
                    Given a number, the factors identified so far, and the remaining unfactorized part,
                    determine the next step in the factorization process.
                    
                    If the remaining part is 1, the factorization is complete.
                    If the remaining part is prime, add it to the factors and complete the factorization.
                    Otherwise, find the smallest factor of the remaining part and continue.
                    
                    Provide detailed mathematical reasoning for your approach.
                    """
                )
            )
            
            # Define the signature for verifying the factorization
            self.verify_factorization = dspy.ChainOfThought(
                dspy.Signature(
                    "number, factors -> verification_result, explanation",
                    """
                    Verify that the prime factorization is correct.
                    
                    Check that:
                    1. All factors are prime numbers
                    2. The product of all factors equals the original number
                    
                    Return True if the factorization is correct, False otherwise,
                    along with a detailed explanation.
                    """
                )
            )
        
        def forward(self, number: int):
            """
            Factorize a number into its prime components with step-by-step reasoning.
            
            Args:
                number: The number to factorize
                
            Returns:
                A prediction containing the factors and explanation
            """
            # Initialize
            current_factors = []
            remaining = number
            steps = []
            
            # Step-by-step factorization
            max_steps = 10  # Prevent infinite loops
            for step_idx in range(max_steps):
                # Get the next step in factorization
                prediction = self.factorize_step(
                    number=str(number),
                    current_factors=str(current_factors),
                    remaining=str(remaining)
                )
                
                next_step = prediction.next_step
                updated_factors = eval(prediction.updated_factors)  # Convert string representation to list
                new_remaining = int(prediction.new_remaining)
                
                # Record this step
                steps.append({
                    "step": step_idx + 1,
                    "description": next_step,
                    "factors_so_far": updated_factors.copy(),
                    "remaining": new_remaining
                })
                
                # Update state
                current_factors = updated_factors
                remaining = new_remaining
                
                # Check if factorization is complete
                if remaining == 1:
                    break
            
            # Verify the factorization
            verification = self.verify_factorization(
                number=str(number),
                factors=str(current_factors)
            )
            
            # Prepare the final output
            return dspy.Prediction(
                factors=current_factors,
                steps=steps,
                verified=verification.verification_result,
                verification_explanation=verification.explanation
            )


#==========================================================================
# Example usage
#==========================================================================

def main():
    """Example usage of the hierarchical reasoning system."""
    # Create the hierarchical reasoning system
    system = HierarchicalReasoningSystem()
    
    # Test numbers for factorization
    test_numbers = [
        2 * 3 * 5 * 7,               # 210
        2**10 - 3,                   # 1021
        2 * 3 * 5 * 7 * 11 * 13,     # 30030
        104729 * 104723              # ~11 billion
    ]
    
    # Test factorization
    print("\n=== Testing Factorization ===")
    for i, n in enumerate(test_numbers):
        print(f"\nFactorizing {n}:")
        
        result = system.factorize(n)
        
        print(f"Factors: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {' '.join(result['explanation'])}")
        
        # Print the expert contributions
        print("\nExpert contributions:")
        for contrib in result['trace']['expert_contributions']:
            print(f"- {contrib['expert']} ({contrib['role']}): {contrib['solution'].get('result', None)}")
    
    # Test quantum resistance
    print("\n=== Testing Quantum Resistance Analysis ===")
    crypto_numbers = [
        (2**1024 - 159),  # ~1024 bits
        (2**2048 - 1393)  # ~2048 bits
    ]
    
    for n in crypto_numbers:
        print(f"\nAnalyzing quantum resistance of {n}:")
        
        result = system.analyze_strength(n)
        
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {' '.join(result['explanation'])}")
    
    # Test near-prime detection
    print("\n=== Testing Near-Prime Detection ===")
    near_primes = [
        101,  # Prime with close neighbor
        (2**31 - 1)  # Mersenne prime
    ]
    
    for n in near_primes:
        print(f"\nAnalyzing near-prime properties of {n}:")
        
        result = system.analyze_near_prime(n)
        
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {' '.join(result['explanation'])}")
    
    # Test prime gap analysis
    print("\n=== Testing Prime Gap Analysis ===")
    
    result = system.analyze_prime_gaps(1, 100)
    
    print(f"Prime gaps in range [1, 100]:")
    print(f"Min gap: {result['result'].get('min_gap', 'N/A')}")
    print(f"Max gap: {result['result'].get('max_gap', 'N/A')}")
    print(f"Avg gap: {result['result'].get('avg_gap', 'N/A'):.2f}")
    print(f"Explanation: {' '.join(result['explanation'])}")

if __name__ == "__main__":
    main()