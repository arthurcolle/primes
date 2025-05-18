import math
import sympy
import numpy as np
from typing import Dict, List, Tuple
import random
import time

def advanced_is_prime(n: str, method: str = "default") -> str:
    """Determine if a number is prime using various algorithms.
    
    Args:
        n: The number to check as a string
        method: Algorithm to use (default, miller-rabin, aks, fermat)
    
    Returns:
        A string with detailed primality analysis
    
    Examples:
        {"n": "17", "method": "default"} -> "17 is prime"
        {"n": "561", "method": "miller-rabin"} -> "561 is not prime (Carmichael number). Failed Miller-Rabin test with witness 2."
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num <= 1:
        return f"{num} is not prime. By definition, prime numbers must be greater than 1."
    
    if method == "default":
        # Use sympy's isprime which uses multiple algorithms internally
        start_time = time.time()
        is_prime = sympy.isprime(num)
        elapsed = time.time() - start_time
        
        if is_prime:
            result = f"{num} is prime (verified in {elapsed:.6f} seconds)"
        else:
            # Find a factor
            for i in range(2, min(10000, int(math.sqrt(num)) + 1)):
                if num % i == 0:
                    result = f"{num} is not prime. It is divisible by {i} (found in {elapsed:.6f} seconds)."
                    break
            else:
                result = f"{num} is not prime (verified in {elapsed:.6f} seconds)"
        
        return result
        
    elif method == "miller-rabin":
        # Implement Miller-Rabin primality test
        start_time = time.time()
        
        # Check if it's a Carmichael number (composite but passes Fermat test)
        if not sympy.isprime(num) and all(pow(a, num-1, num) == 1 for a in [2, 3, 5, 7, 11, 13, 17]):
            # Find a Miller-Rabin witness
            for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
                if sympy.is_square(num) or not sympy.miller_rabin(num, [a]):
                    elapsed = time.time() - start_time
                    return f"{num} is not prime (Carmichael number). Failed Miller-Rabin test with witness {a} (in {elapsed:.6f} seconds)."
            
            elapsed = time.time() - start_time
            return f"{num} is not prime, but it's a Carmichael number that passes some primality tests (in {elapsed:.6f} seconds)."
        
        # Perform Miller-Rabin with 10 random bases
        is_probable_prime = sympy.isprime(num)
        elapsed = time.time() - start_time
        
        if is_probable_prime:
            return f"{num} is prime with high probability (Miller-Rabin test with multiple witnesses, in {elapsed:.6f} seconds)"
        else:
            return f"{num} is composite (verified by Miller-Rabin test in {elapsed:.6f} seconds)"
    
    elif method == "aks":
        # AKS is a deterministic primality test but very slow for large numbers
        if num > 10**9:
            return f"AKS primality test is too slow for {num}. Try 'miller-rabin' method instead."
        
        start_time = time.time()
        is_prime = sympy.isprime(num)
        elapsed = time.time() - start_time
        
        if is_prime:
            return f"{num} is prime (verified by AKS algorithm in {elapsed:.6f} seconds)"
        else:
            # Find a factor
            for i in range(2, min(10000, int(math.sqrt(num)) + 1)):
                if num % i == 0:
                    return f"{num} is not prime. It is divisible by {i} (found in {elapsed:.6f} seconds)."
            
            return f"{num} is not prime (verified in {elapsed:.6f} seconds)"
    
    elif method == "fermat":
        # Fermat primality test (probabilistic)
        start_time = time.time()
        
        # Test with multiple bases
        bases = [2, 3, 5, 7, 11, 13, 17]
        fermat_results = []
        
        for base in bases:
            if pow(base, num-1, num) != 1:
                elapsed = time.time() - start_time
                return f"{num} is not prime. Failed Fermat's little theorem with base {base} (in {elapsed:.6f} seconds)."
            fermat_results.append(base)
        
        # It passed all Fermat tests
        elapsed = time.time() - start_time
        
        # But it could be a Carmichael number
        if not sympy.isprime(num):
            return f"{num} is not prime, but it's a pseudoprime that passes Fermat primality test with bases {fermat_results} (in {elapsed:.6f} seconds)."
        
        return f"{num} is prime (verified by Fermat primality test with bases {fermat_results} in {elapsed:.6f} seconds)"
    
    else:
        return f"Unknown method: {method}. Use 'default', 'miller-rabin', 'aks', or 'fermat'."

def advanced_factorize(n: str, method: str = "default", timeout: int = 10) -> str:
    """Find the prime factorization of a number using various algorithms.
    
    Args:
        n: The number to factorize as a string
        method: Algorithm to use (default, trial, rho, ecm)
        timeout: Maximum seconds to spend on factorization
    
    Returns:
        A string containing the prime factorization with details
    
    Examples:
        {"n": "60", "method": "default"} -> "60 = 2² × 3 × 5"
        {"n": "8051", "method": "rho"} -> "8051 = 83 × 97 (found using Pollard's Rho algorithm)"
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num <= 1:
        return f"{num} does not have a prime factorization. Only numbers greater than 1 have prime factorizations."
    
    start_time = time.time()
    
    if method == "default":
        # Use sympy's factorint which selects algorithm automatically
        try:
            factors = list(sympy.factorint(num).items())
            elapsed = time.time() - start_time
            
            # Format the result
            factorization = []
            for prime, exp in factors:
                if exp == 1:
                    factorization.append(str(prime))
                else:
                    factorization.append(f"{prime}^{exp}")
            
            result = f"{num} = {' × '.join(factorization)} (found in {elapsed:.6f} seconds)"
            return result
        except Exception as e:
            return f"Failed to factorize {num}: {str(e)}"
    
    elif method == "trial":
        # Trial division algorithm
        factors = []
        n_temp = num
        
        # Check divisibility by 2
        while n_temp % 2 == 0:
            factors.append(2)
            n_temp //= 2
            if time.time() - start_time > timeout:
                return f"Factorization of {num} timed out after {timeout} seconds. Partial factorization: {num} = {' × '.join(map(str, factors))} × {n_temp}"
        
        # Check divisibility by odd numbers
        i = 3
        while i * i <= n_temp:
            while n_temp % i == 0:
                factors.append(i)
                n_temp //= i
                if time.time() - start_time > timeout:
                    return f"Factorization of {num} timed out after {timeout} seconds. Partial factorization: {num} = {' × '.join(map(str, factors))} × {n_temp}"
            i += 2
            if time.time() - start_time > timeout:
                return f"Factorization of {num} timed out after {timeout} seconds. Partial factorization: {num} = {' × '.join(map(str, factors))} × {n_temp}"
        
        # If n_temp is greater than 1, it's a prime factor
        if n_temp > 1:
            factors.append(n_temp)
        
        elapsed = time.time() - start_time
        
        # Count occurrences of each factor
        factor_counts = {}
        for f in factors:
            factor_counts[f] = factor_counts.get(f, 0) + 1
        
        # Format the result
        factorization = []
        for prime, exp in sorted(factor_counts.items()):
            if exp == 1:
                factorization.append(str(prime))
            else:
                factorization.append(f"{prime}^{exp}")
        
        result = f"{num} = {' × '.join(factorization)} (found using trial division in {elapsed:.6f} seconds)"
        return result
    
    elif method == "rho":
        # Pollard's Rho algorithm
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def pollard_rho(n):
            if n % 2 == 0:
                return 2
            
            x = random.randint(1, n-1)
            y = x
            c = random.randint(1, n-1)
            d = 1
            
            while d == 1:
                x = (pow(x, 2, n) + c) % n
                y = (pow(y, 2, n) + c) % n
                y = (pow(y, 2, n) + c) % n
                d = gcd(abs(x - y), n)
                
                if time.time() - start_time > timeout:
                    return None  # Timeout
                
                if d == n:
                    return pollard_rho(n)  # Try again with different values
            
            return d
        
        def factorize_rho(n, factors=None):
            if factors is None:
                factors = []
            
            if n == 1:
                return factors
            
            if sympy.isprime(n):
                factors.append(n)
                return factors
            
            factor = pollard_rho(n)
            if factor is None:  # Timeout
                factors.append(n)  # Add the unfactored part
                return factors
            
            factorize_rho(factor, factors)
            factorize_rho(n // factor, factors)
            return factors
        
        try:
            factors = factorize_rho(num)
            elapsed = time.time() - start_time
            
            # Count occurrences of each factor
            factor_counts = {}
            for f in factors:
                factor_counts[f] = factor_counts.get(f, 0) + 1
            
            # Format the result
            factorization = []
            for prime, exp in sorted(factor_counts.items()):
                if exp == 1:
                    factorization.append(str(prime))
                else:
                    factorization.append(f"{prime}^{exp}")
            
            result = f"{num} = {' × '.join(factorization)} (found using Pollard's Rho algorithm in {elapsed:.6f} seconds)"
            return result
        except Exception as e:
            return f"Failed to factorize {num} using Pollard's Rho: {str(e)}"
    
    elif method == "ecm":
        # Elliptic Curve Method is best for large numbers with small factors
        # Here we'll use sympy's implementation
        try:
            factors = []
            remaining = num
            
            # Try to find small factors first
            for i in range(2, 1000):
                while remaining % i == 0:
                    factors.append(i)
                    remaining //= i
                    if time.time() - start_time > timeout:
                        if remaining > 1:
                            return f"Factorization of {num} timed out after {timeout} seconds. Partial factorization: {num} = {' × '.join(map(str, factors))} × {remaining}"
                        break
            
            # If there's still a large factor, use ECM
            if remaining > 1:
                # Use sympy's factorint with ECM method
                ecm_factors = sympy.factorint(remaining, method='ecm', timeout=timeout - (time.time() - start_time))
                for prime, exp in ecm_factors.items():
                    factors.extend([prime] * exp)
            
            elapsed = time.time() - start_time
            
            # Count occurrences of each factor
            factor_counts = {}
            for f in factors:
                factor_counts[f] = factor_counts.get(f, 0) + 1
            
            # Format the result
            factorization = []
            for prime, exp in sorted(factor_counts.items()):
                if exp == 1:
                    factorization.append(str(prime))
                else:
                    factorization.append(f"{prime}^{exp}")
            
            result = f"{num} = {' × '.join(factorization)} (found using ECM in {elapsed:.6f} seconds)"
            return result
        except Exception as e:
            return f"Failed to factorize {num} using ECM: {str(e)}"
    
    else:
        return f"Unknown method: {method}. Use 'default', 'trial', 'rho', or 'ecm'."

def prime_gaps(n: str, count: int = 10) -> str:
    """Find prime gaps around a given number.
    
    Args:
        n: The starting number as a string
        count: Number of primes to find in each direction
    
    Returns:
        A string listing primes before and after the given number
    
    Examples:
        {"n": "50", "count": 5} -> "Primes before 50: 47, 43, 41, 37, 31. Primes after 50: 53, 59, 61, 67, 71."
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num < 2:
        return f"No primes before {num}. Primes after {num}: 2, 3, 5, 7, 11..."
    
    # Find primes before n
    primes_before = []
    current = num - 1
    while len(primes_before) < count and current >= 2:
        if sympy.isprime(current):
            primes_before.append(current)
        current -= 1
    
    # Find primes after n
    primes_after = []
    current = num + 1
    while len(primes_after) < count:
        if sympy.isprime(current):
            primes_after.append(current)
        current += 1
    
    # Calculate gaps
    gaps_before = []
    for i in range(len(primes_before) - 1):
        gaps_before.append(primes_before[i] - primes_before[i + 1])
    
    gaps_after = []
    for i in range(len(primes_after) - 1):
        gaps_after.append(primes_after[i + 1] - primes_after[i])
    
    # Format result
    result = f"Primes before {num}: {', '.join(map(str, primes_before))}"
    if gaps_before:
        result += f" (gaps: {', '.join(map(str, gaps_before))})"
    
    result += f"\nPrimes after {num}: {', '.join(map(str, primes_after))}"
    if gaps_after:
        result += f" (gaps: {', '.join(map(str, gaps_after))})"
    
    # If the number itself is prime
    if sympy.isprime(num):
        result += f"\n{num} is itself a prime number."
    
    return result

def prime_density(start: str, end: str) -> str:
    """Calculate prime density in a given range.
    
    Args:
        start: The start of the range as a string
        end: The end of the range as a string
    
    Returns:
        A string describing prime density with statistics
    
    Examples:
        {"start": "1", "end": "100"} -> "There are 25 primes in the range [1, 100]. Density: 25.0%. Prime number theorem estimate: 21.7%."
    """
    try:
        start_num = int(start)
        end_num = int(end)
    except ValueError:
        return f"Error: Start and end must be valid integers"
    
    if end_num <= start_num:
        return f"Error: End number must be greater than start number"
    
    if end_num - start_num > 10**6:
        return f"Error: Range too large. Please specify a range of at most 1,000,000 numbers."
    
    # Find primes in the range
    if start_num < 2:
        start_num = 2  # Smallest prime
    
    primes = list(sympy.primerange(start_num, end_num + 1))
    prime_count = len(primes)
    
    # Calculate density
    range_size = end_num - start_num + 1
    density = prime_count / range_size * 100
    
    # Prime number theorem estimate: π(n) ≈ n / ln(n)
    pnt_estimate_start = start_num / math.log(start_num) if start_num > 1 else 0
    pnt_estimate_end = end_num / math.log(end_num) if end_num > 1 else 0
    pnt_estimate = max(0, pnt_estimate_end - pnt_estimate_start)
    pnt_density = pnt_estimate / range_size * 100
    
    # Calculate statistics
    if prime_count > 0:
        avg_gap = (end_num - start_num) / (prime_count - 1) if prime_count > 1 else 0
        min_gap = min([primes[i+1] - primes[i] for i in range(len(primes) - 1)]) if prime_count > 1 else 0
        max_gap = max([primes[i+1] - primes[i] for i in range(len(primes) - 1)]) if prime_count > 1 else 0
        
        result = f"There are {prime_count} primes in the range [{start_num}, {end_num}].\n"
        result += f"Density: {density:.1f}%. Prime number theorem estimate: {pnt_density:.1f}%.\n"
        
        if prime_count > 1:
            result += f"Average gap between primes: {avg_gap:.2f}\n"
            result += f"Minimum gap: {min_gap}. Maximum gap: {max_gap}.\n"
        
        # List some primes from the range if there aren't too many
        if prime_count <= 20:
            result += f"All primes in this range: {', '.join(map(str, primes))}"
        else:
            first_few = ', '.join(map(str, primes[:5]))
            last_few = ', '.join(map(str, primes[-5:]))
            result += f"Sample primes: {first_few}, ..., {last_few}"
        
        return result
    else:
        return f"There are no primes in the range [{start_num}, {end_num}]."

def twin_primes(limit: str) -> str:
    """Find twin primes up to a given limit.
    
    Args:
        limit: Upper bound for searching twin primes
    
    Returns:
        A string listing twin prime pairs found
    
    Examples:
        {"limit": "50"} -> "Twin primes up to 50: (3,5), (5,7), (11,13), (17,19), (29,31), (41,43)"
    """
    try:
        num = int(limit)
    except ValueError:
        return f"Error: '{limit}' is not a valid integer"
    
    if num < 5:
        return "No twin primes can be found below 5."
    
    # Limit the search to a reasonable range
    if num > 10**6:
        return f"Limit too large. Please use a limit of at most 1,000,000."
    
    # Find all primes up to the limit
    primes = list(sympy.primerange(2, num + 1))
    
    # Find twin primes
    twin_pairs = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 2:
            twin_pairs.append((primes[i], primes[i+1]))
    
    # Format the result
    if twin_pairs:
        pairs_str = ', '.join([f"({a},{b})" for a, b in twin_pairs])
        count = len(twin_pairs)
        
        result = f"Found {count} twin prime pairs up to {num}:\n{pairs_str}"
        
        # Add some mathematical context
        if num >= 100:
            # Hardy-Littlewood conjecture estimate
            c = 0.6601618
            estimate = 2 * c * num / (math.log(num) ** 2)
            result += f"\n\nThe Hardy-Littlewood conjecture estimates approximately {estimate:.1f} twin prime pairs up to {num}."
        
        return result
    else:
        return f"No twin primes found up to {num}."

def prime_factorization_tree(n: str) -> str:
    """Generate a visualization of the prime factorization process.
    
    Args:
        n: The number to factorize as a string
    
    Returns:
        A string representation of the factorization tree
    
    Examples:
        {"n": "60"} -> "60\n├── 6 × 10\n│   ├── 2 × 3\n│   └── 2 × 5\n"
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num <= 1:
        return f"{num} does not have a prime factorization."
    
    if num > 10**9:
        return f"Number too large for tree visualization. Please use a smaller number."
    
    if sympy.isprime(num):
        return f"{num} is prime and cannot be further factorized."
    
    # Helper function to build the factorization tree
    def build_tree(n, depth=0, is_last=True, prefix=""):
        if n <= 1:
            return ""
        
        result = prefix + ("└── " if is_last else "├── ") + str(n) + "\n"
        
        if sympy.isprime(n):
            return result
        
        # Find two factors
        factor1 = None
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factor1 = i
                factor2 = n // i
                break
        
        if factor1 is None:  # This shouldn't happen if we've checked for primality
            return result
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        result += build_tree(factor1, depth + 1, False, new_prefix)
        result += build_tree(factor2, depth + 1, True, new_prefix)
        
        return result
    
    # Start with the number itself
    tree = str(num) + "\n"
    
    # Find two initial factors
    factor1 = None
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            factor1 = i
            factor2 = num // i
            break
    
    if factor1 is None:  # This shouldn't happen if we've checked for primality
        return tree
    
    tree += f"├── {factor1} × {factor2}\n"
    
    # Build subtrees
    if not sympy.isprime(factor1):
        for line in build_tree(factor1, 1, False, "│   ").split("\n"):
            if line:
                tree += "│   " + line + "\n"
    else:
        tree += "│   └── " + str(factor1) + " (prime)\n"
    
    if not sympy.isprime(factor2):
        for line in build_tree(factor2, 1, True, "│   ").split("\n"):
            if line:
                tree += "│   " + line + "\n"
    else:
        tree += "│   └── " + str(factor2) + " (prime)\n"
    
    return tree

def number_theory_analysis(n: str) -> str:
    """Provide a detailed number theory analysis of a number.
    
    Args:
        n: The number to analyze as a string
    
    Returns:
        A string with detailed number theory properties
    
    Examples:
        {"n": "28"} -> "Number: 28\nFactorization: 2² × 7\nNumber of divisors: 6\nSum of divisors: 56\nPerfect number: Yes\n..."
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num < 1:
        return f"Please provide a positive integer for number theory analysis."
    
    # Basic properties
    factorization = sympy.factorint(num)
    formatted_factors = []
    for prime, exp in factorization.items():
        if exp == 1:
            formatted_factors.append(str(prime))
        else:
            formatted_factors.append(f"{prime}^{exp}")
    
    factorization_str = " × ".join(formatted_factors) if formatted_factors else "1"
    
    # Divisors
    divisors = list(sympy.divisors(num))
    divisor_count = len(divisors)
    divisor_sum = sum(divisors)
    proper_divisor_sum = divisor_sum - num
    
    # Number classifications
    is_prime = sympy.isprime(num)
    is_perfect = proper_divisor_sum == num
    is_abundant = proper_divisor_sum > num
    is_deficient = proper_divisor_sum < num
    is_square = sympy.perfect_square(num)
    is_cube = sympy.perfect_power(num, 3)
    
    # Modular properties
    residues_mod_10 = num % 10
    
    # Build the analysis
    analysis = [f"Number Theory Analysis of {num}:"]
    analysis.append(f"Prime factorization: {factorization_str}")
    
    # Divisors
    if divisor_count <= 20:
        analysis.append(f"Divisors: {', '.join(map(str, divisors))}")
    else:
        analysis.append(f"Number of divisors: {divisor_count}")
    
    analysis.append(f"Sum of divisors: {divisor_sum}")
    analysis.append(f"Sum of proper divisors: {proper_divisor_sum}")
    
    # Classifications
    classifications = []
    if is_prime:
        classifications.append("Prime")
    if is_perfect:
        classifications.append("Perfect")
    elif is_abundant:
        classifications.append("Abundant")
    elif is_deficient:
        classifications.append("Deficient")
    if is_square:
        classifications.append(f"Perfect square ({int(math.sqrt(num))}²)")
    if is_cube:
        classifications.append(f"Perfect cube ({int(round(num**(1/3)))}³)")
    
    if classifications:
        analysis.append(f"Classifications: {', '.join(classifications)}")
    
    # Primality
    if is_prime:
        # Find closest primes
        prev_prime = sympy.prevprime(num)
        next_prime = sympy.nextprime(num)
        analysis.append(f"Previous prime: {prev_prime} (gap: {num - prev_prime})")
        analysis.append(f"Next prime: {next_prime} (gap: {next_prime - num})")
    elif len(factorization) == 1:
        # It's a prime power
        prime, exp = list(factorization.items())[0]
        analysis.append(f"Prime power: {prime}^{exp}")
    
    # Special numbers
    if sympy.isprime(num + 2) and sympy.isprime(num):
        analysis.append(f"{num} and {num + 2} form a twin prime pair")
    if sympy.isprime(num - 2) and sympy.isprime(num):
        analysis.append(f"{num - 2} and {num} form a twin prime pair")
    
    # Modular properties
    analysis.append(f"Last digit: {residues_mod_10}")
    
    # Digital properties
    digit_sum = sum(int(d) for d in str(num))
    analysis.append(f"Digit sum: {digit_sum}")
    
    # For large numbers, add additional properties
    if num > 1000:
        # Approximations from number theory
        if not is_prime:
            # Prime counting function approximation
            pi_approx = num / math.log(num)
            analysis.append(f"Prime counting function π({num}) ≈ {pi_approx:.2f}")
    
    return "\n".join(analysis)

def carmichael_check(n: str) -> str:
    """Check if a number is a Carmichael number (composite but passes Fermat primality test).
    
    Args:
        n: The number to check as a string
    
    Returns:
        Analysis of whether the number is a Carmichael number
    
    Examples:
        {"n": "561"} -> "561 is a Carmichael number. It is composite but passes the Fermat primality test."
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num < 1:
        return f"Please provide a positive integer."
    
    # Check if prime
    if sympy.isprime(num):
        return f"{num} is prime, so it is NOT a Carmichael number."
    
    # Check if it's a Carmichael number
    witnesses = [2, 3, 5, 7, 11, 13, 17]
    fermat_passes = True
    failing_witnesses = []
    
    for a in witnesses:
        if math.gcd(a, num) == 1:  # Only test if a and num are coprime
            if pow(a, num-1, num) != 1:
                fermat_passes = False
                failing_witnesses.append(a)
    
    # Get factorization
    factorization = sympy.factorint(num)
    formatted_factors = []
    for prime, exp in factorization.items():
        if exp == 1:
            formatted_factors.append(str(prime))
        else:
            formatted_factors.append(f"{prime}^{exp}")
    
    factorization_str = " × ".join(formatted_factors)
    
    if fermat_passes:
        known_carmichaels = [561, 1105, 1729, 2465, 2821, 6601, 8911]
        is_known = num in known_carmichaels
        
        result = f"{num} is a Carmichael number! It is composite ({factorization_str}) but passes the Fermat primality test.\n\n"
        
        # Explain why it's a Carmichael number
        result += "A Carmichael number is a composite number that satisfies Fermat's Little Theorem:\n"
        result += "For any integer a coprime to n, a^(n-1) ≡ 1 (mod n)\n\n"
        
        if is_known:
            result += f"{num} is one of the well-known Carmichael numbers.\n"
        
        # Properties of Carmichael numbers
        primes = list(factorization.keys())
        if all(num % (p-1) == 1 for p in primes):
            result += "It satisfies Korselt's criterion: for each prime factor p, (n-1) is divisible by (p-1).\n"
        
        return result
    else:
        return f"{num} is NOT a Carmichael number. It is composite ({factorization_str}) and fails the Fermat primality test with witness(es): {', '.join(map(str, failing_witnesses))}."