import math
import sympy

def is_prime(n: str) -> str:
    """Check if a number is prime.
    
    Args:
        n: The number to check as a string
    
    Returns:
        A string indicating whether the number is prime, with explanation
    
    Examples:
        {"n": "17"} -> "17 is prime"
        {"n": "20"} -> "20 is not prime. It is divisible by 2 and 5."
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num <= 1:
        return f"{num} is not prime. By definition, prime numbers must be greater than 1."
    
    if num == 2 or num == 3:
        return f"{num} is prime"
    
    if num % 2 == 0:
        return f"{num} is not prime. It is divisible by 2."
    
    # Check for divisibility using 6k±1 optimization
    for i in range(3, int(math.sqrt(num)) + 1, 2):
        if num % i == 0:
            return f"{num} is not prime. It is divisible by {i}."
    
    return f"{num} is prime"

def factorize(n: str) -> str:
    """Find the prime factorization of a number.
    
    Args:
        n: The number to factorize as a string
    
    Returns:
        A string containing the prime factorization
    
    Examples:
        {"n": "20"} -> "20 = 2² × 5"
        {"n": "60"} -> "60 = 2² × 3 × 5"
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num <= 1:
        return f"{num} does not have a prime factorization. Only numbers greater than 1 have prime factorizations."
    
    # Use sympy for efficient factorization
    factors = list(sympy.factorint(num).items())
    
    # Format the result
    factorization = []
    for prime, exp in factors:
        if exp == 1:
            factorization.append(str(prime))
        else:
            factorization.append(f"{prime}^{exp}")
    
    result = f"{num} = {' × '.join(factorization)}"
    return result

def next_prime(n: str) -> str:
    """Find the next prime number after a given number.
    
    Args:
        n: The starting number as a string
    
    Returns:
        A string indicating the next prime number
    
    Examples:
        {"n": "10"} -> "The next prime after 10 is 11"
        {"n": "17"} -> "The next prime after 17 is 19"
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num < 1:
        return "The next prime after a negative number or 0 is 2"
    
    next_p = sympy.nextprime(num)
    return f"The next prime after {num} is {next_p}"

def prime_count(n: str) -> str:
    """Count the number of primes less than or equal to n.
    
    Args:
        n: The upper bound as a string
    
    Returns:
        A string indicating the count of primes
    
    Examples:
        {"n": "10"} -> "There are 4 primes less than or equal to 10: 2, 3, 5, 7"
        {"n": "20"} -> "There are 8 primes less than or equal to 20: 2, 3, 5, 7, 11, 13, 17, 19"
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    if num < 2:
        return f"There are 0 primes less than or equal to {num}"
    
    # For small numbers, list all primes
    if num <= 10000:
        primes = list(sympy.primerange(2, num + 1))
        count = len(primes)
        if count <= 20:  # Only list them all if there aren't too many
            return f"There are {count} primes less than or equal to {num}: {', '.join(map(str, primes))}"
        else:
            return f"There are {count} primes less than or equal to {num}"
    else:
        # For larger numbers, use approximation
        count = sympy.ntheory.primepi(num)
        return f"There are approximately {count} primes less than or equal to {num}"

def verify_factorization(n: str, factors: str) -> str:
    """Verify if a given factorization is correct.
    
    Args:
        n: The number to verify as a string
        factors: Comma-separated list of factors
    
    Returns:
        A string indicating whether the factorization is correct
    
    Examples:
        {"n": "60", "factors": "2,2,3,5"} -> "Verification successful: 2 × 2 × 3 × 5 = 60"
        {"n": "20", "factors": "2,3,3"} -> "Verification failed: 2 × 3 × 3 = 18, not 20"
    """
    try:
        num = int(n)
    except ValueError:
        return f"Error: '{n}' is not a valid integer"
    
    try:
        factor_list = [int(f.strip()) for f in factors.split(',')]
    except ValueError:
        return f"Error: Factors must be comma-separated integers"
    
    product = 1
    for f in factor_list:
        product *= f
    
    if product == num:
        return f"Verification successful: {' × '.join(map(str, factor_list))} = {num}"
    else:
        return f"Verification failed: {' × '.join(map(str, factor_list))} = {product}, not {num}"