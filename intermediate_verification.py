#!/usr/bin/env python
"""
Intermediate Step Verification System for Prime Factorization

This module implements a verification system that checks each step in the factorization
process to detect errors early and provide confidence metrics for the overall solution.
"""

import math
import sympy
import time
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field
import json
import random

# Verification status constants
VERIFIED = "verified"
UNCERTAIN = "uncertain"
ERROR = "error"


@dataclass
class VerificationResult:
    """Result of a verification check."""
    status: str
    confidence: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_verified(self) -> bool:
        """Check if the status is verified."""
        return self.status == VERIFIED


@dataclass
class StepVerifier:
    """Base class for step verifiers."""
    name: str
    description: str
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify a specific step in the factorization process."""
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class PrimalityVerifier(StepVerifier):
    """Verifies claims about primality of numbers."""
    
    def __init__(self):
        super().__init__(
            name="Primality Verifier",
            description="Verifies claims about whether a number is prime"
        )
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify primality claims in a step."""
        # Extract the claimed prime/composite numbers
        claimed_primes = step.get("claimed_primes", [])
        claimed_composites = step.get("claimed_composites", [])
        
        # Verify each claimed prime
        errors = []
        details = {"verified_primes": [], "verified_composites": [], "errors": []}
        
        for number in claimed_primes:
            try:
                is_prime = sympy.isprime(number)
                if is_prime:
                    details["verified_primes"].append(number)
                else:
                    error = f"{number} is claimed to be prime, but it is composite"
                    errors.append(error)
                    details["errors"].append({"number": number, "claim": "prime", "actual": "composite"})
            except Exception as e:
                errors.append(f"Error checking primality of {number}: {str(e)}")
                details["errors"].append({"number": number, "error": str(e)})
        
        # Verify each claimed composite
        for number in claimed_composites:
            try:
                is_prime = sympy.isprime(number)
                if not is_prime:
                    details["verified_composites"].append(number)
                else:
                    error = f"{number} is claimed to be composite, but it is prime"
                    errors.append(error)
                    details["errors"].append({"number": number, "claim": "composite", "actual": "prime"})
            except Exception as e:
                errors.append(f"Error checking primality of {number}: {str(e)}")
                details["errors"].append({"number": number, "error": str(e)})
        
        # Determine overall status and confidence
        if errors:
            return VerificationResult(
                status=ERROR,
                confidence=0.0,
                error_message="; ".join(errors),
                details=details
            )
        
        return VerificationResult(
            status=VERIFIED,
            confidence=1.0,
            details=details
        )


@dataclass
class DivisibilityVerifier(StepVerifier):
    """Verifies divisibility claims in factorization steps."""
    
    def __init__(self):
        super().__init__(
            name="Divisibility Verifier",
            description="Verifies claims about divisibility relationships"
        )
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify divisibility claims in a step."""
        # Extract the divisibility claims
        divisibility_claims = step.get("divisibility_claims", [])
        
        if not divisibility_claims:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No divisibility claims to verify",
                details={"message": "No divisibility claims provided in the step"}
            )
        
        # Verify each claim
        errors = []
        details = {"verified_claims": [], "errors": []}
        
        for claim in divisibility_claims:
            dividend = claim.get("dividend")
            divisor = claim.get("divisor")
            claimed_result = claim.get("result", True)  # Default to claiming divisibility
            
            if not dividend or not divisor:
                errors.append(f"Invalid claim format: {claim}")
                details["errors"].append({"claim": claim, "error": "Missing dividend or divisor"})
                continue
            
            try:
                # Check if divisor divides dividend
                is_divisible = (dividend % divisor == 0)
                
                if is_divisible == claimed_result:
                    details["verified_claims"].append({
                        "dividend": dividend,
                        "divisor": divisor,
                        "is_divisible": is_divisible
                    })
                else:
                    error_msg = f"{dividend} {'is' if claimed_result else 'is not'} claimed to be divisible by {divisor}, but this is incorrect"
                    errors.append(error_msg)
                    details["errors"].append({
                        "dividend": dividend,
                        "divisor": divisor,
                        "claim": claimed_result,
                        "actual": is_divisible
                    })
            except Exception as e:
                errors.append(f"Error checking divisibility: {dividend} / {divisor}: {str(e)}")
                details["errors"].append({
                    "dividend": dividend,
                    "divisor": divisor,
                    "error": str(e)
                })
        
        # Determine overall status and confidence
        if errors:
            return VerificationResult(
                status=ERROR,
                confidence=0.0,
                error_message="; ".join(errors),
                details=details
            )
        
        return VerificationResult(
            status=VERIFIED,
            confidence=1.0,
            details=details
        )


@dataclass
class FactorProductVerifier(StepVerifier):
    """Verifies that the product of claimed factors equals the target number."""
    
    def __init__(self):
        super().__init__(
            name="Factor Product Verifier",
            description="Verifies that the product of factors equals the original number"
        )
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify that the product of factors equals the target number."""
        # Extract the claimed factors and the target number
        factors = step.get("factors", [])
        target_number = context.get("number")
        
        if not factors:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No factors to verify",
                details={"message": "No factors provided in the step"}
            )
        
        if not target_number:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No target number available",
                details={"message": "Target number not found in context"}
            )
        
        try:
            # Calculate the product of factors
            product = 1
            for factor in factors:
                product *= factor
            
            # Check if the product equals the target number
            if product == target_number:
                return VerificationResult(
                    status=VERIFIED,
                    confidence=1.0,
                    details={"target": target_number, "product": product, "factors": factors}
                )
            else:
                error_msg = f"Product of factors ({product}) does not equal target number ({target_number})"
                return VerificationResult(
                    status=ERROR,
                    confidence=0.0,
                    error_message=error_msg,
                    details={
                        "target": target_number,
                        "product": product,
                        "factors": factors,
                        "difference": product - target_number
                    }
                )
        except Exception as e:
            return VerificationResult(
                status=ERROR,
                confidence=0.0,
                error_message=f"Error calculating product: {str(e)}",
                details={"error": str(e), "factors": factors}
            )


@dataclass
class PrimeFactorVerifier(StepVerifier):
    """Verifies that all claimed factors are prime."""
    
    def __init__(self):
        super().__init__(
            name="Prime Factor Verifier",
            description="Verifies that all claimed factors are prime numbers"
        )
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify that all claimed factors are prime."""
        # Extract the claimed factors
        factors = step.get("factors", [])
        
        if not factors:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No factors to verify",
                details={"message": "No factors provided in the step"}
            )
        
        # Verify each factor is prime
        non_prime_factors = []
        verified_factors = []
        
        for factor in factors:
            try:
                is_prime = sympy.isprime(factor)
                if is_prime:
                    verified_factors.append(factor)
                else:
                    non_prime_factors.append(factor)
            except Exception as e:
                return VerificationResult(
                    status=ERROR,
                    confidence=0.0,
                    error_message=f"Error checking primality of {factor}: {str(e)}",
                    details={"error": str(e), "factor": factor}
                )
        
        # Determine overall status and confidence
        if non_prime_factors:
            error_msg = f"The following factors are not prime: {non_prime_factors}"
            return VerificationResult(
                status=ERROR,
                confidence=0.0,
                error_message=error_msg,
                details={"non_prime_factors": non_prime_factors, "verified_factors": verified_factors}
            )
        
        return VerificationResult(
            status=VERIFIED,
            confidence=1.0,
            details={"verified_factors": verified_factors}
        )


@dataclass
class RemainderVerifier(StepVerifier):
    """Verifies remainder calculations in the factorization process."""
    
    def __init__(self):
        super().__init__(
            name="Remainder Verifier",
            description="Verifies remainder calculations during division"
        )
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify remainder calculations in a step."""
        # Extract the remainder calculations
        remainder_calculations = step.get("remainder_calculations", [])
        
        if not remainder_calculations:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No remainder calculations to verify",
                details={"message": "No remainder calculations provided in the step"}
            )
        
        # Verify each calculation
        errors = []
        details = {"verified_calculations": [], "errors": []}
        
        for calc in remainder_calculations:
            dividend = calc.get("dividend")
            divisor = calc.get("divisor")
            claimed_remainder = calc.get("remainder")
            
            if not dividend or not divisor or claimed_remainder is None:
                errors.append(f"Invalid calculation format: {calc}")
                details["errors"].append({"calculation": calc, "error": "Missing values"})
                continue
            
            try:
                # Calculate the actual remainder
                actual_remainder = dividend % divisor
                
                if actual_remainder == claimed_remainder:
                    details["verified_calculations"].append({
                        "dividend": dividend,
                        "divisor": divisor,
                        "remainder": actual_remainder
                    })
                else:
                    error_msg = f"Remainder of {dividend} / {divisor} is claimed to be {claimed_remainder}, but it's actually {actual_remainder}"
                    errors.append(error_msg)
                    details["errors"].append({
                        "dividend": dividend,
                        "divisor": divisor,
                        "claimed_remainder": claimed_remainder,
                        "actual_remainder": actual_remainder
                    })
            except Exception as e:
                errors.append(f"Error calculating remainder: {dividend} % {divisor}: {str(e)}")
                details["errors"].append({
                    "dividend": dividend,
                    "divisor": divisor,
                    "error": str(e)
                })
        
        # Determine overall status and confidence
        if errors:
            return VerificationResult(
                status=ERROR,
                confidence=0.0,
                error_message="; ".join(errors),
                details=details
            )
        
        return VerificationResult(
            status=VERIFIED,
            confidence=1.0,
            details=details
        )


@dataclass
class AlgorithmicConsistencyVerifier(StepVerifier):
    """Verifies that the factorization algorithm is being applied correctly."""
    
    def __init__(self):
        super().__init__(
            name="Algorithmic Consistency Verifier",
            description="Verifies that the factorization algorithm is being applied correctly"
        )
    
    def verify(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify algorithmic consistency in a step."""
        # Extract the algorithm being used
        algorithm = step.get("algorithm", "")
        
        if not algorithm:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No algorithm specified",
                details={"message": "No algorithm specified in the step"}
            )
        
        # Verification logic depends on the algorithm
        if algorithm == "trial_division":
            return self._verify_trial_division(step, context)
        elif algorithm == "pollard_rho":
            return self._verify_pollard_rho(step, context)
        elif algorithm == "quadratic_sieve":
            return self._verify_quadratic_sieve(step, context)
        else:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message=f"Unsupported algorithm: {algorithm}",
                details={"algorithm": algorithm, "message": "Verification not implemented for this algorithm"}
            )
    
    def _verify_trial_division(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify trial division algorithm."""
        # Check if the step tries all divisors in order
        current_divisor = step.get("current_divisor")
        previous_divisor = context.get("previous_divisor", 1)
        
        if not current_divisor:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No current divisor specified",
                details={"message": "Current divisor not specified in the step"}
            )
        
        # In trial division, we should check divisors in ascending order
        if current_divisor <= previous_divisor and previous_divisor > 1:
            return VerificationResult(
                status=ERROR,
                confidence=0.0,
                error_message=f"Trial division is not checking divisors in ascending order: {previous_divisor} -> {current_divisor}",
                details={"previous_divisor": previous_divisor, "current_divisor": current_divisor}
            )
        
        # Check if we're only testing prime divisors (optimization)
        if current_divisor > 3 and current_divisor % 2 == 0:
            return VerificationResult(
                status=ERROR,
                confidence=0.3,  # Minor error (inefficient but not incorrect)
                error_message=f"Testing composite divisor ({current_divisor}) in trial division is inefficient",
                details={"divisor": current_divisor, "message": "Should only test prime divisors"}
            )
        
        # Check if we're going beyond sqrt(n) (inefficient)
        number = context.get("remaining_number", context.get("number"))
        if number and current_divisor > math.sqrt(number):
            return VerificationResult(
                status=ERROR,
                confidence=0.3,  # Minor error (inefficient but not incorrect)
                error_message=f"Testing divisors beyond sqrt(n) in trial division is inefficient",
                details={
                    "divisor": current_divisor,
                    "sqrt_n": math.sqrt(number),
                    "message": "Should stop at sqrt(n)"
                }
            )
        
        return VerificationResult(
            status=VERIFIED,
            confidence=1.0,
            details={"algorithm": "trial_division", "current_divisor": current_divisor}
        )
    
    def _verify_pollard_rho(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify Pollard's rho algorithm."""
        # Extract key values
        x = step.get("x")
        y = step.get("y")
        gcd_value = step.get("gcd_value")
        
        if x is None or y is None or gcd_value is None:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="Missing key values for Pollard's rho verification",
                details={"message": "x, y, or gcd_value not specified in the step"}
            )
        
        # Verify function application (x' = f(x), y' = f(f(y)))
        f_function = step.get("f_function", "default")
        c_value = step.get("c_value", 1)
        number = context.get("remaining_number", context.get("number"))
        
        if not number:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="No number available for verification",
                details={"message": "Number not found in context"}
            )
        
        # Verify GCD calculation
        if gcd_value:
            actual_gcd = math.gcd(abs(x - y), number)
            
            if actual_gcd != gcd_value:
                return VerificationResult(
                    status=ERROR,
                    confidence=0.0,
                    error_message=f"GCD calculation is incorrect: gcd({abs(x - y)}, {number}) should be {actual_gcd}, not {gcd_value}",
                    details={
                        "x": x,
                        "y": y,
                        "number": number,
                        "claimed_gcd": gcd_value,
                        "actual_gcd": actual_gcd
                    }
                )
        
        return VerificationResult(
            status=VERIFIED,
            confidence=0.9,  # Slight uncertainty due to randomized nature
            details={"algorithm": "pollard_rho", "x": x, "y": y, "gcd": gcd_value}
        )
    
    def _verify_quadratic_sieve(self, step: Dict[str, Any], context: Dict[str, Any]) -> VerificationResult:
        """Verify quadratic sieve algorithm."""
        # This is a complex algorithm to verify fully
        # We'll check some basic properties
        
        # Verify the sieving interval is reasonable
        interval_start = step.get("interval_start")
        interval_end = step.get("interval_end")
        number = context.get("remaining_number", context.get("number"))
        
        if not interval_start or not interval_end or not number:
            return VerificationResult(
                status=UNCERTAIN,
                confidence=0.5,
                error_message="Missing key values for quadratic sieve verification",
                details={"message": "interval_start, interval_end, or number not available"}
            )
        
        # The interval should be around sqrt(n)
        sqrt_n = int(math.sqrt(number))
        
        if abs(interval_start - sqrt_n) > sqrt_n / 2 or abs(interval_end - sqrt_n) > sqrt_n / 2:
            return VerificationResult(
                status=ERROR,
                confidence=0.4,  # Not necessarily wrong, but unusual
                error_message="Sieving interval is not centered around sqrt(n)",
                details={
                    "interval": (interval_start, interval_end),
                    "sqrt_n": sqrt_n,
                    "message": "Interval should be centered around sqrt(n)"
                }
            )
        
        # For full verification, we'd need to check more details of the sieving process
        
        return VerificationResult(
            status=VERIFIED,
            confidence=0.7,  # Some uncertainty due to complexity
            details={"algorithm": "quadratic_sieve", "interval": (interval_start, interval_end)}
        )


@dataclass
class IntermediateVerificationSystem:
    """
    A comprehensive system for verifying each step in the factorization process.
    """
    verifiers: List[StepVerifier] = field(default_factory=list)
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __init__(self):
        """Initialize the verification system with standard verifiers."""
        self.verifiers = [
            PrimalityVerifier(),
            DivisibilityVerifier(),
            FactorProductVerifier(),
            PrimeFactorVerifier(),
            RemainderVerifier(),
            AlgorithmicConsistencyVerifier()
        ]
    
    def verify_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a single step in the factorization process.
        
        Args:
            step: The step to verify
            context: Contextual information needed for verification
            
        Returns:
            Dictionary with verification results for each relevant verifier
        """
        step_id = step.get("step_id", len(self.verification_history) + 1)
        step_type = step.get("type", "unknown")
        
        verification_results = {}
        applicable_verifiers = []
        
        # Determine applicable verifiers based on step type
        for verifier in self.verifiers:
            # Apply specific verifiers based on step type
            if step_type == "primality_check" and isinstance(verifier, PrimalityVerifier):
                applicable_verifiers.append(verifier)
            elif step_type == "divisibility_check" and isinstance(verifier, DivisibilityVerifier):
                applicable_verifiers.append(verifier)
            elif step_type == "factor_product" and isinstance(verifier, FactorProductVerifier):
                applicable_verifiers.append(verifier)
            elif step_type == "prime_factorization" and isinstance(verifier, PrimeFactorVerifier):
                applicable_verifiers.append(verifier)
            elif step_type == "remainder_calculation" and isinstance(verifier, RemainderVerifier):
                applicable_verifiers.append(verifier)
            elif step_type == "algorithm_application" and isinstance(verifier, AlgorithmicConsistencyVerifier):
                applicable_verifiers.append(verifier)
            
            # If no specific type is given, try all verifiers
            if step_type == "unknown":
                applicable_verifiers = self.verifiers
                break
        
        # Apply each applicable verifier
        for verifier in applicable_verifiers:
            try:
                result = verifier.verify(step, context)
                verification_results[verifier.name] = {
                    "status": result.status,
                    "confidence": result.confidence,
                    "error_message": result.error_message,
                    "details": result.details
                }
            except Exception as e:
                verification_results[verifier.name] = {
                    "status": ERROR,
                    "confidence": 0.0,
                    "error_message": f"Verifier error: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        # Calculate overall verification status
        overall_status = VERIFIED
        overall_confidence = 1.0
        error_messages = []
        
        for verifier_name, result in verification_results.items():
            if result["status"] == ERROR:
                overall_status = ERROR
                overall_confidence = min(overall_confidence, result["confidence"])
                if result.get("error_message"):
                    error_messages.append(f"{verifier_name}: {result['error_message']}")
            elif result["status"] == UNCERTAIN:
                if overall_status != ERROR:
                    overall_status = UNCERTAIN
                overall_confidence = min(overall_confidence, result["confidence"])
        
        # Create verification record
        verification_record = {
            "step_id": step_id,
            "step_type": step_type,
            "timestamp": time.time(),
            "overall_status": overall_status,
            "overall_confidence": overall_confidence,
            "error_messages": error_messages,
            "verifier_results": verification_results
        }
        
        # Add to history
        self.verification_history.append(verification_record)
        
        return verification_record
    
    def verify_solution(self, solution: Dict[str, Any], number: int) -> Dict[str, Any]:
        """
        Verify a complete factorization solution.
        
        Args:
            solution: The complete factorization solution
            number: The original number to factorize
            
        Returns:
            Dictionary with verification results
        """
        # Extract the claimed factors
        factors = solution.get("factors", [])
        
        # Create a step for verification
        step = {
            "type": "prime_factorization",
            "factors": factors
        }
        
        # Create context with the original number
        context = {"number": number}
        
        # Verify the step
        return self.verify_step(step, context)
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all verifications performed.
        
        Returns:
            Dictionary with verification summary
        """
        if not self.verification_history:
            return {
                "status": UNCERTAIN,
                "confidence": 0.0,
                "error_messages": ["No verification steps recorded"],
                "step_count": 0
            }
        
        # Count status types
        verified_count = sum(1 for record in self.verification_history if record["overall_status"] == VERIFIED)
        error_count = sum(1 for record in self.verification_history if record["overall_status"] == ERROR)
        uncertain_count = sum(1 for record in self.verification_history if record["overall_status"] == UNCERTAIN)
        
        # Calculate overall confidence
        overall_confidence = sum(record["overall_confidence"] for record in self.verification_history) / len(self.verification_history)
        
        # Determine overall status
        if error_count > 0:
            overall_status = ERROR
        elif uncertain_count > 0:
            overall_status = UNCERTAIN
        else:
            overall_status = VERIFIED
        
        # Collect error messages
        all_error_messages = []
        for record in self.verification_history:
            all_error_messages.extend(record["error_messages"])
        
        return {
            "status": overall_status,
            "confidence": overall_confidence,
            "verified_steps": verified_count,
            "uncertain_steps": uncertain_count,
            "error_steps": error_count,
            "total_steps": len(self.verification_history),
            "error_messages": all_error_messages
        }


def test_verification_system():
    """Test the verification system with sample factorization steps."""
    # Create the verification system
    system = IntermediateVerificationSystem()
    
    # Test number
    number = 210  # 2 × 3 × 5 × 7
    context = {"number": number}
    
    # Test primality check step
    primality_step = {
        "step_id": 1,
        "type": "primality_check",
        "claimed_primes": [2, 3, 5, 7],
        "claimed_composites": [210, 15, 35]
    }
    
    print("=== Verifying Primality Check ===")
    primality_result = system.verify_step(primality_step, context)
    print(f"Status: {primality_result['overall_status']}")
    print(f"Confidence: {primality_result['overall_confidence']}")
    if primality_result["error_messages"]:
        print(f"Errors: {', '.join(primality_result['error_messages'])}")
    
    # Test divisibility check step
    divisibility_step = {
        "step_id": 2,
        "type": "divisibility_check",
        "divisibility_claims": [
            {"dividend": 210, "divisor": 2, "result": True},
            {"dividend": 210, "divisor": 3, "result": True},
            {"dividend": 210, "divisor": 5, "result": True},
            {"dividend": 210, "divisor": 7, "result": True},
            {"dividend": 210, "divisor": 11, "result": False}
        ]
    }
    
    print("\n=== Verifying Divisibility Check ===")
    divisibility_result = system.verify_step(divisibility_step, context)
    print(f"Status: {divisibility_result['overall_status']}")
    print(f"Confidence: {divisibility_result['overall_confidence']}")
    if divisibility_result["error_messages"]:
        print(f"Errors: {', '.join(divisibility_result['error_messages'])}")
    
    # Test factor product step
    factor_product_step = {
        "step_id": 3,
        "type": "factor_product",
        "factors": [2, 3, 5, 7]
    }
    
    print("\n=== Verifying Factor Product ===")
    factor_product_result = system.verify_step(factor_product_step, context)
    print(f"Status: {factor_product_result['overall_status']}")
    print(f"Confidence: {factor_product_result['overall_confidence']}")
    if factor_product_result["error_messages"]:
        print(f"Errors: {', '.join(factor_product_result['error_messages'])}")
    
    # Test remainder calculation step
    remainder_step = {
        "step_id": 4,
        "type": "remainder_calculation",
        "remainder_calculations": [
            {"dividend": 210, "divisor": 2, "remainder": 0},
            {"dividend": 105, "divisor": 3, "remainder": 0},
            {"dividend": 35, "divisor": 5, "remainder": 0},
            {"dividend": 7, "divisor": 7, "remainder": 0}
        ]
    }
    
    print("\n=== Verifying Remainder Calculations ===")
    remainder_result = system.verify_step(remainder_step, context)
    print(f"Status: {remainder_result['overall_status']}")
    print(f"Confidence: {remainder_result['overall_confidence']}")
    if remainder_result["error_messages"]:
        print(f"Errors: {', '.join(remainder_result['error_messages'])}")
    
    # Test algorithm application step
    algorithm_step = {
        "step_id": 5,
        "type": "algorithm_application",
        "algorithm": "trial_division",
        "current_divisor": 7,
        "previous_divisor": 5
    }
    
    algorithm_context = {
        "number": 210,
        "remaining_number": 35,
        "previous_divisor": 5
    }
    
    print("\n=== Verifying Algorithm Application ===")
    algorithm_result = system.verify_step(algorithm_step, algorithm_context)
    print(f"Status: {algorithm_result['overall_status']}")
    print(f"Confidence: {algorithm_result['overall_confidence']}")
    if algorithm_result["error_messages"]:
        print(f"Errors: {', '.join(algorithm_result['error_messages'])}")
    
    # Test complete solution verification
    solution = {
        "factors": [2, 3, 5, 7]
    }
    
    print("\n=== Verifying Complete Solution ===")
    solution_result = system.verify_solution(solution, number)
    print(f"Status: {solution_result['overall_status']}")
    print(f"Confidence: {solution_result['overall_confidence']}")
    if solution_result["error_messages"]:
        print(f"Errors: {', '.join(solution_result['error_messages'])}")
    
    # Get summary
    print("\n=== Verification Summary ===")
    summary = system.get_verification_summary()
    print(f"Overall Status: {summary['status']}")
    print(f"Overall Confidence: {summary['confidence']:.2f}")
    print(f"Steps: {summary['verified_steps']} verified, {summary['uncertain_steps']} uncertain, {summary['error_steps']} errors")
    print(f"Total Steps: {summary['total_steps']}")


if __name__ == "__main__":
    test_verification_system()