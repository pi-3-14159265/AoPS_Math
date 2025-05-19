import math
import fraction
# ax+b=c
class LinearEquation:
    def __init__(self, a=1, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c
    
    def solve(self):
        if self.a == 0:
            if self.b == self.c:
                raise 'Error: Infinite solutions'
            else:
                raise 'Error: No solutions'
        else:
            return (self.c - self.b) / self.a
        
def __str__(self):
    a_ = self.a if self.a != 1 else ''
    b_ = self.b if self.b != 0 else ''
    return f'{fraction.convert(a_)}x + {fraction.convert(b_)} = {fraction.convert(self.c)}'
    

# ax + by = c
class BiLinearEquation:
    def __init__(self, a=1, b=1, c=0):
        self.a = a
        self.b = b
        self.c = c

    def solve(self, other):
        det = self.a * other.b - self.b * other.a
        if det == 0:
            raise ValueError('The system has no solutions')
        x = (self.c * other.b - self.b * other.c) / det
        y = (self.a * other.c - self.c * other.a) / det
        return fraction.convert(x), fraction.convert(y)
    
    def __str__(self):
        return f'{fraction.convert(self.a)}x + {fraction.convert(self.b)}y = {fraction.convert(self.c)}'
