import math


def is_random(nums):
    """
    determines how random a list of numbers is

    Args:
        nums: the list of numbers
    Returns:
        a percentage of how random the list is
    Note:
        the higher the percentage, the more random the list
        the more elements in the list, the more accurate the percentage
    """
    points_inside_circle = 0
    
    high = float('-inf')
    
    for num in nums:
        if num > high:
            high = num
    
    for i in range(0, len(nums), 2):
        if i + 1 >= len(nums):
            break
        
        # Normalize the points to be between -1 and 1
        x = nums[i] / high
        y = nums[i + 1] / high
        
        if x * x + y * y <= 1:
            points_inside_circle += 1
    
    pi_estimate = 4.0 * points_inside_circle / (len(nums) / 2)
    return (abs((pi_estimate) / math.pi * 100 - 100)-100)*-1

def nCr(n,r):
    """
    finds the number of combinations of n things taken r at a time

    Args:
        n: the total number of things
        r: the number of things to choose
    Returns:
        the number of combinations
    """
    return math.comb(n, r)

def nPr(n,r):
    """
    finds the number of permutations of n things taken r at a time

    Args:
        n: the total number of things
        r: the number of things to choose
    Returns:
        the number of permutations
    """
    return math.perm(n, r)

def factorial(n):
    """
    finds the factorial of n

    Args:
        n: the number
    Returns:
        the factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
