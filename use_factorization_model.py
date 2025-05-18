#!/usr/bin/env python3
"""
use_factorization_model.py

Use a fine-tuned o4-mini model to factorize numbers.
This script provides a command-line interface and also demonstrates how to use the model programmatically.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Union

from prime_rft_model import OpenAIAPI

def factorize_number(
    api: OpenAIAPI, 
    model_id: str, 
    number: int,
    verbose: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Factorize a number using a fine-tuned model.
    
    Args:
        api: OpenAI API instance
        model_id: Model ID to use
        number: Number to factorize
        verbose: Whether to print verbose output
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with factorization results
    """
    # Create system and user messages
    messages = [
        {"role": "system", "content": "You are an expert factorization assistant. Your task is to find the prime factorization of numbers efficiently."},
        {"role": "user", "content": f"Find the prime factorization of {number}. Return your answer as JSON with the following fields: 'factors' (list of integers), 'algorithm' (string), 'reasoning' (list of steps), 'time_taken' (float), and 'confidence' (float)."}
    ]
    
    # Start timer
    start_time = time.time()
    
    # Get model response
    response = api.chat_completion(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # End timer
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"API call time: {elapsed_time:.4f} seconds")
    
    # Extract and parse result
    try:
        content = response["choices"][0]["message"]["content"]
        result = json.loads(content)
        
        # Add API call time for reference
        result["api_call_time"] = elapsed_time
        
        return result
    except Exception as e:
        error_msg = f"Error parsing model output: {e}"
        if verbose:
            print(error_msg)
            print("Raw output:", content)
        return {
            "error": error_msg,
            "raw_output": response["choices"][0]["message"]["content"] if "choices" in response else None,
            "api_call_time": elapsed_time
        }

def verify_factorization(number: int, factors: List[int]) -> bool:
    """
    Verify that the factorization is correct.
    
    Args:
        number: The original number
        factors: List of factors
        
    Returns:
        True if the factorization is correct, False otherwise
    """
    if not factors:
        return False
    
    # Multiply factors
    product = 1
    for factor in factors:
        product *= factor
    
    return product == number

def pretty_print_result(result: Dict[str, Any], number: int, verify: bool = True) -> None:
    """
    Print factorization result in a nice format.
    
    Args:
        result: Factorization result
        number: Original number
        verify: Whether to verify the factorization
    """
    print("\n" + "="*60)
    print(f"📊 Factorization of {number}")
    print("="*60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "raw_output" in result and result["raw_output"]:
            print("\nRaw output:")
            print(result["raw_output"])
        return
    
    # Extract fields
    factors = result.get("factors", [])
    algorithm = result.get("algorithm", "Unknown")
    reasoning = result.get("reasoning", [])
    time_taken = result.get("time_taken", 0.0)
    confidence = result.get("confidence", 0.0)
    api_call_time = result.get("api_call_time", 0.0)
    
    # Verify factorization
    is_correct = verify_factorization(number, factors) if verify else "Not verified"
    
    # Print results
    print(f"🔢 Factors: {factors}")
    if verify:
        print(f"✅ Correct: {is_correct}")
    print(f"⚙️ Algorithm: {algorithm}")
    print(f"⏱️ Reported time: {time_taken:.4f} seconds")
    print(f"🔄 API call time: {api_call_time:.4f} seconds")
    print(f"🎯 Confidence: {confidence:.4f}")
    
    print("\n📝 Reasoning:")
    for i, step in enumerate(reasoning):
        print(f"  {i+1}. {step}")
    
    # Print verification
    if verify:
        print("\n🔍 Verification:")
        if is_correct:
            print(f"  ✅ {' × '.join(map(str, factors))} = {number}")
        else:
            print(f"  ❌ {' × '.join(map(str, factors))} ≠ {number}")
    
    print("="*60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Use fine-tuned o4-mini model for prime factorization.")
    parser.add_argument("number", type=int, help="Number to factorize")
    parser.add_argument("--model", type=str, required=True, help="Fine-tuned model ID")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--no-verify", action="store_true", help="Do not verify factorization")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON output")
    
    args = parser.parse_args()
    
    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key as an environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Initialize API
    api = OpenAIAPI()
    
    try:
        # Factorize number
        result = factorize_number(
            api=api,
            model_id=args.model,
            number=args.number,
            verbose=args.verbose,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Print result
        if args.raw:
            print(json.dumps(result, indent=2))
        else:
            pretty_print_result(result, args.number, verify=not args.no_verify)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()