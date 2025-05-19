import math

def convert(decimal):
    if decimal == 0:
        return 0
    elif decimal == 1:
        return 1
    
    sign = -1 if decimal < 0 else 1
    decimal = abs(decimal)
    
    # Use continued fraction to find the best rational approximation
    n = 1
    while True:
        numerator = round(decimal * n)
        denominator = n
        if abs(decimal - numerator / denominator) < 0.000001:
            gcd = math.gcd(numerator, denominator)
            numerator = (numerator // gcd) * sign
            denominator = denominator // gcd
            return (numerator, denominator) if denominator != 1 else numerator
        n += 1

def convert_(decimal):
    if decimal == 0:
        return 0, 1
    elif decimal == 1:
        return 1, 1
    
    sign = -1 if decimal < 0 else 1
    decimal = abs(decimal)
    
    # Use continued fraction to find the best rational approximation
    n = 1
    while True:
        numerator = round(decimal * n)
        denominator = n
        if abs(decimal - numerator / denominator) < 0.000001:
            gcd = math.gcd(numerator, denominator)
            numerator = (numerator // gcd) * sign
            denominator = denominator // gcd
            return numerator, denominator
        n += 1

def convert_frac(decimal) -> str:
    numerator, denominator = convert_(decimal)
    if denominator == 1:
        return str(numerator)
    return f'({numerator}/{denominator})'