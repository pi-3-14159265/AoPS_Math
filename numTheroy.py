import math
import multiprocessing

def helper(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
    return inner

# Eulers totient function
def phi(n):
    '''
    Eulers totient function aka ϕ(n) is the number of positive integers less than n that are coprime to n.

    Args:
        n: the argument
    
    Returns:
        the number of positive integers less than n that are coprime to n
    '''
    if n == 1:
        return 1
    count = 0
    for k in range(1, n):
        if math.gcd(n, k) == 1:
            count += 1
    return count


def reverse_crt(n):
    """
    collects order pair (p,q) such that n=pq such that gcd(p,q)=1

    Args:
        n: the argument
    
    Returns:
        a list of tuples of order pairs (p,q) such that n=pq and gcd(p,q)=1 not including (1,n) or (p,q) and (q,p)
    
    """

    pairs = []
    for p in range(2, n):
        if n % p == 0:
            q = n // p
            if math.gcd(p, q) == 1:
                pairs.append((p, q))
    for pair in pairs:
        if pair[::-1] in pairs:
            pairs.remove(pair[::-1])
    return pairs

def is_prime(n):
    """
    checks if a number is prime

    Args:
        n: the argument
    
    Returns:
        True if n is prime, False otherwise
    
    """
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def inverse(a, n):
    """
    finds the modular inverse of a number

    Args:
        a: the number
        n: the modulus
    
    Returns:
        the modular inverse of a number
    
    """
    for i in range(1, n):
        if (a * i) % n == 1:
            return i
    return None

@helper
def ppt(m, n):
    '''
    Genrates a Primitive Pythagorean triples

    Args:
        m: the first argument
        n: the second argument
    
    Returns:
        a tuple of the Pythagorean triple (a,b,c) where a^2 + b^2 = c^2
    
    '''
    if m <= n:
        raise ValueError('m must be greater than n')
    a = m**2 - n**2
    b = 2 * m * n
    c = m**2 + n**2
    return a, b, c

def pythagorean_triples(n):
    '''
    Generates a list of Pythagorean triples

    Args:
        n: the argument
    
    Returns:
        a list of Pythagorean triples (a,b,c) where a^2 + b^2 = c^2
    
    '''
    triples = []
    for m in range(1, n):
        for n in range(1, m):
            if math.gcd(m, n) == 1 and (m - n) % 2 == 1:
                triples.append(ppt(m, n))
    return triples
    
def tetration(n, m):
    '''
    Tetration is the next hyperoperation after exponentiation. It is the number of times you exponentiate a number by itself.

    Args:
        n: the base
        m: the height
    
    Returns:
        the result of tetration
    
    '''
    if m == 0:
        return 1
    if m == 1:
        return n
    return n ** tetration(n, m - 1)

def order(a, n):
    '''
    The order of a number is the smallest positive integer k such that a^k ≡ 1 (mod n)

    Args:
        a: the number
        n: the modulus

    Returns:
        the order of a number

    '''
    for k in range(1, n):
        if a ** k % n == 1:
            return k
    return None

def gammma(n, perc=9999):
    import calculus as calc
    import Prob
    '''
    The gamma function is a generalization of the factorial function to complex and real numbers.

    Args:
        n: the argument
    Returns:
        the value of the gamma function at n
    '''
    if n <= 0 and n == int(n):
        raise ValueError('n Not Valid for negative integers')

    if n == int(n):
        return Prob.factorial(n - 1)
    else:
        return calc.Function(f"x^({n-1})*e^(-x)").bonded_integral(0,perc)