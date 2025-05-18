from typing import List, Dict, Any, Union, Tuple
import re
import math

from verifiers import RewardFunc
from verifiers.rubrics.rubric import Rubric
from verifiers.parsers import XMLParser

class PrimeRubric(Rubric):
    """Rubric for evaluating prime number factorization and verification tasks."""
    
    def __init__(self, parser: XMLParser, env_parser: XMLParser = None, **kwargs):
        super().__init__(parser, env_parser, **kwargs)
    
    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return a list of reward functions for prime verification tasks."""
        return [
            self.factorization_accuracy,
            self.reasoning_quality,
            self.tool_usage_efficiency
        ]
    
    def get_reward_weights(self) -> List[float]:
        """Return weights for reward functions."""
        return [0.6, 0.3, 0.1]
    
    def factorization_accuracy(self, messages: List[Dict[str, str]], **kwargs: Any) -> float:
        """
        Evaluate the accuracy of the prime factorization or verification.
        
        Args:
            messages: List of conversation messages
            kwargs: Additional arguments including the expected answer
            
        Returns:
            Reward score between 0 and 1
        """
        # Extract user query, model answer and expected answer
        user_message = None
        for msg in messages:
            if msg.get("role") == "user" and "input" in kwargs and kwargs["input"] in msg.get("content", ""):
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            return 0.0
        
        # Parse expected answer type from user query
        expected_type = self._get_expected_type(user_message)
        if not expected_type:
            return 0.0
        
        # Get model's answer
        try:
            answer_message = messages[-1]
            if answer_message.get("role") != "assistant":
                return 0.0
                
            parsed = self.parser.parse(answer_message.get("content", ""))
            answer = getattr(parsed, "answer", None)
            if not answer:
                return 0.0
        except Exception:
            return 0.0
        
        # Expected answer
        expected_answer = kwargs.get("answer", "")
        
        # Evaluate based on task type
        if expected_type == "is_prime":
            return self._evaluate_primality(answer, expected_answer)
        elif expected_type == "factorize":
            return self._evaluate_factorization(answer, expected_answer)
        elif expected_type == "verify":
            return self._evaluate_verification(answer, expected_answer)
        else:
            return 0.0
    
    def reasoning_quality(self, messages: List[Dict[str, str]], **kwargs: Any) -> float:
        """
        Evaluate the quality of reasoning in the solution.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Reward score between 0 and 1
        """
        try:
            answer_message = messages[-1]
            if answer_message.get("role") != "assistant":
                return 0.0
                
            parsed = self.parser.parse(answer_message.get("content", ""))
            reasoning = getattr(parsed, "reasoning", "")
            
            # Count reasoning steps
            reasoning_steps = len(re.findall(r"[.!?]\s+|[-—]\s+|\n\d+\.\s+|\n[-*]\s+", reasoning))
            
            # Basic quality check - note in a real implementation you'd want more sophisticated metrics
            quality_indicators = 0
            
            # Check for mathematical terms
            math_terms = [
                "prime", "factor", "divisible", "remainder", "quotient", 
                "multiple", "divisor", "modulo", "algorithm", "check"
            ]
            for term in math_terms:
                if term in reasoning.lower():
                    quality_indicators += 1
            
            # Check for numbers and calculations
            calculations = len(re.findall(r"\d+\s*[+\-*/÷×%]\s*\d+", reasoning))
            quality_indicators += min(3, calculations)
            
            # Calculate score
            if reasoning_steps < 1:
                return 0.0
            elif reasoning_steps < 3:
                return 0.3 + min(0.3, quality_indicators * 0.05)
            else:
                return min(1.0, 0.6 + quality_indicators * 0.05)
                
        except Exception:
            return 0.0
    
    def tool_usage_efficiency(self, messages: List[Dict[str, str]], **kwargs: Any) -> float:
        """
        Evaluate how efficiently tools were used.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Reward score between 0 and 1
        """
        # Extract only assistant and tool messages
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        # No messages or just one means we can't evaluate tool usage fully
        if len(assistant_messages) <= 1:
            # If there's just one message and no tool use, give a small penalty
            if len(assistant_messages) == 1:
                parsed = self.parser.parse(assistant_messages[0].get("content", ""))
                if not hasattr(parsed, "tool") or not parsed.tool:
                    return 0.5  # Penalty for not using tools at all
            return 0.7  # Default
        
        # Check for tool use patterns
        tool_uses = []
        for msg in assistant_messages:
            try:
                parsed = self.parser.parse(msg.get("content", ""))
                if hasattr(parsed, "tool") and parsed.tool:
                    # Extract tool name from JSON
                    import json
                    try:
                        tool_data = json.loads(parsed.tool)
                        tool_name = tool_data.get("name", "")
                        if tool_name:
                            tool_uses.append(tool_name)
                    except:
                        # Failed to parse tool JSON
                        pass
            except:
                continue
        
        # No tool uses
        if not tool_uses:
            return 0.5  # Penalty for not using tools
        
        # Check for appropriate tool use based on task type
        expected_type = self._get_expected_type(kwargs.get("input", ""))
        
        # Appropriate tool mappings
        appropriate_tools = {
            "is_prime": ["is_prime"],
            "factorize": ["factorize"],
            "verify": ["verify_factorization", "factorize"]
        }
        
        # Count appropriate tool uses
        if expected_type in appropriate_tools:
            best_tools = appropriate_tools[expected_type]
            appropriate_count = sum(1 for tool in tool_uses if tool in best_tools)
            
            # Calculate efficiency score
            if appropriate_count > 0:
                # If they used appropriate tools
                tool_efficiency = min(1.0, appropriate_count / len(tool_uses))
                return 0.7 + 0.3 * tool_efficiency
            else:
                # Used tools but not the most appropriate ones
                return 0.6
        
        # If we can't determine expected type, just evaluate general tool efficiency
        unique_tools = len(set(tool_uses))
        if len(tool_uses) <= 3 and unique_tools >= 1:
            return 0.7  # Reasonable tool use
        elif len(tool_uses) > 5:
            return 0.5  # Too many tool uses
        else:
            return 0.6  # Moderate efficiency
    
    def _get_expected_type(self, user_message: str) -> str:
        """Determine the expected type of task from the user query."""
        if not user_message:
            return ""
            
        user_message = user_message.lower()
        
        if "is prime" in user_message or "prime number" in user_message:
            return "is_prime"
        elif "factorization" in user_message or "find the prime factors" in user_message:
            return "factorize"
        elif "verify" in user_message or "check if" in user_message or "correct" in user_message:
            return "verify"
        else:
            return ""
    
    def _evaluate_primality(self, answer: str, expected: str) -> float:
        """Evaluate the accuracy of a primality check."""
        answer = answer.lower()
        expected = expected.lower()
        
        # Direct match cases
        if ("prime" in answer and "prime" in expected) or ("not prime" in answer and "not prime" in expected):
            return 1.0
        
        # Yes/no responses
        if expected == "prime" and ("yes" in answer or "is prime" in answer):
            return 1.0
        if expected == "not prime" and ("no" in answer or "not prime" in answer or "is not prime" in answer):
            return 1.0
            
        # Partial matches
        if expected == "prime" and "prime" in answer and "not" not in answer:
            return 0.8
        if expected == "not prime" and ("not" in answer or "composite" in answer):
            return 0.8
            
        # Complete mismatch
        return 0.0
    
    def _evaluate_factorization(self, answer: str, expected: str) -> float:
        """Evaluate the accuracy of a factorization."""
        # Clean and normalize answers
        def normalize_factors(factors_str):
            # Replace various multiplication symbols
            factors_str = factors_str.replace('×', '*').replace('⋅', '*').replace('·', '*')
            # Extract numbers
            factors = []
            for part in re.split(r'[^0-9^]+', factors_str):
                if part:
                    # Handle exponents like 2^3
                    if '^' in part:
                        base, exp = part.split('^')
                        factors.extend([int(base)] * int(exp))
                    else:
                        try:
                            factors.append(int(part))
                        except:
                            pass
            return sorted(factors)
        
        try:
            # Extract expected factors
            expected_factors = normalize_factors(expected)
            
            # Extract answer factors
            answer_factors = normalize_factors(answer)
            
            # Empty factors case
            if not expected_factors or not answer_factors:
                return 0.0
            
            # Calculate precision and recall
            correct = sum(1 for f in answer_factors if f in expected_factors)
            precision = correct / len(answer_factors)
            recall = correct / len(expected_factors)
            
            # F1 score
            if precision + recall == 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            
            # Product check
            expected_product = math.prod(expected_factors)
            answer_product = math.prod(answer_factors)
            
            # If products match but factorization differs (e.g., missed some factors)
            if expected_product == answer_product:
                return max(f1, 0.7)  # At least 0.7 if products match
            else:
                return f1 * 0.8  # Penalty if products don't match
                
        except:
            # If there's any error in parsing, give a minimal score
            # Simple keyword match fallback
            if any(term in answer.lower() for term in expected.lower().split('×')):
                return 0.3
            return 0.0
    
    def _evaluate_verification(self, answer: str, expected: str) -> float:
        """Evaluate the accuracy of a verification."""
        answer = answer.lower()
        expected = expected.lower()
        
        # Direct matches
        if (expected == "correct" and ("correct" in answer or "valid" in answer or "true" in answer or "yes" in answer)) or \
           (expected == "incorrect" and ("incorrect" in answer or "invalid" in answer or "false" in answer or "no" in answer)):
            return 1.0
        
        # Wrong verifications
        if (expected == "correct" and ("incorrect" in answer or "invalid" in answer or "false" in answer or "no" in answer)) or \
           (expected == "incorrect" and ("correct" in answer or "valid" in answer or "true" in answer or "yes" in answer)):
            return 0.0
            
        # Partial matches for "correct"
        if expected == "correct" and "verification successful" in answer:
            return 0.9
            
        # Partial matches for "incorrect"
        if expected == "incorrect" and "verification failed" in answer:
            return 0.9
            
        # Default
        return 0.5