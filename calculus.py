from math import sin, cos, tan, asin, acos, atan, log, exp
import re
from fraction import convert_frac
import matplotlib.pyplot as plt

def helper(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
    return inner

class Node:
    """
    Base class for all nodes in the expression tree.
    """
    def evaluate(self, x):
        """Needed for subclasses not here"""
        raise NotImplementedError

    def simplify(self):
        """Needed for subclasses not here"""
        return self
    
    def differentiate(self):
        """Needed for subclasses not here"""
        raise NotImplementedError
    
    def integrate(self):
        """Needed for subclasses not here"""
        raise NotImplementedError


class Constant(Node):
    """
    Represents a constant value in the expression tree.
    
    Attributes:
        value (float): The constant value.
    
    """
    def __init__(self, value):
        self.value = value

    def evaluate(self, x):
        """ Reurns self.value """
        return self.value

    def simplify(self):
        """ Returns self and ensures that the value is not zero """
        if self.value == 0:
            return Constant(0)  # Zero stays zero
        return self
    
    def differentiate(self):
        """" d/dy c = 0 """
        return Constant(0)
    
    def integrate(self):
        """ Integrate a constant: ∫a dx = a*x """
        # ∫a dx = a*x
        return BinaryOp(Constant(self.value), Variable(), '*')
    
    def __str__(self):
        return str(self.value)

class Variable(Node):
    """
    Variables in the expression tree.

    Attributes:
        None
    """
    def evaluate(self, x):
        """ Returns the value of x """
        return x
    
    def simplify(self):
        """ Returns self """
        return self  # Variable does not simplify
    
    def differentiate(self):
        """ d/dx x = 1 """
        return Constant(1)
    
    def integrate(self):
        """ Integrate x: ∫x dx = x^2/2 """
        # ∫x dx = x^2/2
        return BinaryOp(BinaryOp(self, Constant(2), '^'), Constant(1/2), '*')

    def __str__(self):
        return "x"

class BinaryOp(Node):
    """
    Represents a binary operation in the expression tree.

    Supported operations:
    - +: Addition
    - -: Subtraction
    - *: Multiplication
    - /: Division
    - ^: Exponentiation
    """
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, x):
        """ Evaluates the binary operation at a given value of x """
        a = self.left.evaluate(x)
        b = self.right.evaluate(x)
        return {
            '+': a + b,
            '-': a - b,
            '*': a * b,
            '/': a / b if b != 0 else float('inf'),  # Avoid division by zero
            '^': a ** b
        }[self.op]
    
    def differentiate(self):
        """ Differentiates the binary operation """
        u = self.left
        v = self.right
        du = u.differentiate()
        dv = v.differentiate()

        if self.op == '+':
            return BinaryOp(du, dv, '+')
        elif self.op == '-':
            return BinaryOp(du, dv, '-')
        elif self.op == '*':
            return BinaryOp(BinaryOp(du, v, '*'), BinaryOp(u, dv, '*'), '+')  # Product rule
        elif self.op == '/':
            num = BinaryOp(BinaryOp(du, v, '*'), BinaryOp(u, dv, '*'), '-')
            denom = BinaryOp(v, Constant(2), '^')
            return BinaryOp(num, denom, '/')
        elif self.op == '^':
            # d/dx[u^v] = u^v * (v' * ln(u) + v * u'/u)
            if isinstance(v, Constant):
                return BinaryOp(BinaryOp(Constant(v.value), BinaryOp(u, Constant(v.value - 1), '^'), '*'), du, '*')  # Power rule for constant exponent
            ln_u = UnaryOp(u, 'ln')
            v_prime = dv
            u_prime = du
            left = BinaryOp(v_prime, ln_u, '*')
            right = BinaryOp(v, BinaryOp(u_prime, u, '/'), '*')
            return BinaryOp(BinaryOp(u, v, '^'), BinaryOp(left, right, '+'), '*')
        else:
            raise ValueError(f"Unknown operator {self.op}")
        
    def integrate(self):
        """ Integrates the binary operation """
        u = self.left
        v = self.right

        if self.op == '+':
            return BinaryOp(u.integrate(), v.integrate(), '+')
        elif self.op == '-':
            return BinaryOp(u.integrate(), v.integrate(), '-')
        elif self.op == '*':
            if isinstance(u, Constant):
                return BinaryOp(u, v.integrate(), '*')
            elif isinstance(v, Constant):
                return BinaryOp(v, u.integrate(), '*')
            else:
                raise NotImplementedError("∫(f(x) * g(x)) not supported")
        elif self.op == '^':
            if isinstance(u, Variable) and isinstance(v, Constant):
                n = v.value
                if n == -1:
                    return UnaryOp(Variable(), 'ln')  # ∫x^-1 dx = ln|x|
                else:
                    new_exp = Constant(n + 1)
                    return BinaryOp(BinaryOp(Variable(), new_exp, '^'), Constant(1 / (n + 1)), '*')
            else:
                raise NotImplementedError("Only ∫x^n dx supported")
        else:
            raise NotImplementedError(f"Integration for operator {self.op} not implemented")

    def simplify(self):
        """ Simplifies the binary operation 
            NOTE: Needs imporvement ans sucks at the moment enter your functions like a normal person
        """
        # Simplify the left and right parts of the operation
        left_simplified = self.left.simplify()
        right_simplified = self.right.simplify()

        # Check for simplifications in binary operations
        if self.op == '+':
            if isinstance(left_simplified, Constant) and left_simplified.value == 0:
                return right_simplified  # x + 0 -> x
            elif isinstance(right_simplified, Constant) and right_simplified.value == 0:
                return left_simplified  # 0 + x -> x
        elif self.op == '*':
            if isinstance(left_simplified, Constant) and left_simplified.value == 1:
                return right_simplified  # x * 1 -> x
            elif isinstance(right_simplified, Constant) and right_simplified.value == 1:
                return left_simplified  # 1 * x -> x
            elif isinstance(left_simplified, Constant) and left_simplified.value == 0:
                return Constant(0)  # x * 0 -> 0
            elif isinstance(right_simplified, Constant) and right_simplified.value == 0:
                return Constant(0)  # 0 * x -> 0
        elif self.op == '-':
            if isinstance(right_simplified, Constant) and right_simplified.value == 0:
                return left_simplified  # x - 0 -> x
        elif self.op == '/':
            if isinstance(right_simplified, Constant) and right_simplified.value == 1:
                return left_simplified  # x / 1 -> x
            elif isinstance(left_simplified, Constant) and left_simplified.value == 0:
                return Constant(0)  # 0 / x -> 0

        # Return the operation with simplified parts
        return BinaryOp(left_simplified, right_simplified, self.op)

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        op = self.op
        return f"({left} {op} {right})"

class UnaryOp(Node):
    """
    Represents a unary operation in the expression tree.

    Supported operations:
    - sin
    - cos
    - tan
    - sec
    - cot
    - csc
    - arcsin
    - arccos
    - arctan
    - ln
    - e (exponential function)
    """
    def __init__(self, operand, func):
        self.operand = operand
        self.func = func

    def evaluate(self, x):
        """ Evaluates the unary operation at a given value of x """
        val = self.operand.evaluate(x)
        try:
            return {
                'sin': sin(val),
                'cos': cos(val),
                'tan': tan(val),
                'sec': 1 / cos(val) if cos(val) != 0 else float('inf'),
                'cot': 1 / tan(val) if tan(val) != 0 else float('inf'),
                'csc': 1 / sin(val) if sin(val) != 0 else float('inf'),
                'arcsin': asin(val) if -1 <= val <= 1 else float('nan'),
                'arccos': acos(val) if -1 <= val <= 1 else float('nan'),
                'arctan': atan(val),
                'ln': log(val) if val > 0 else float('-inf'),
                'e': exp(val)
            }[self.func]
        except (ValueError, ZeroDivisionError):
            return float('nan')  # You could also raise an error if you prefer

    def differentiate(self):
        """ Differentiates the unary operation """
        u = self.operand
        du = u.differentiate()

        f = self.func
        if f == 'sin':
            return BinaryOp(UnaryOp(u, 'cos'), du, '*')
        elif f == 'cos':
            return BinaryOp(Constant(-1), BinaryOp(UnaryOp(u, 'sin'), du, '*'), '*')
        elif f == 'tan':
            return BinaryOp(BinaryOp(UnaryOp(u, 'sec'), Constant(2), '^'), du, '*')
        elif f == 'sec':
            return BinaryOp(UnaryOp(u, 'sec'), BinaryOp(UnaryOp(u, 'tan'), du, '*'), '*')
        elif f == 'csc':
            neg = Constant(-1)
            return BinaryOp(neg, BinaryOp(UnaryOp(u, 'csc'), BinaryOp(UnaryOp(u, 'cot'), du, '*'), '*'), '*')
        elif f == 'cot':
            return BinaryOp(Constant(-1), BinaryOp(BinaryOp(UnaryOp(u, 'csc'), Constant(2), '^'), du, '*'), '*')
        elif f == 'arcsin':
            return BinaryOp(du, BinaryOp(Constant(1), BinaryOp(u, Constant(2), '^'), '-'), '/')
        elif f == 'arccos':
            return BinaryOp(Constant(-1), BinaryOp(du, BinaryOp(Constant(1), BinaryOp(u, Constant(2), '^'), '-'), '/'), '*')
        elif f == 'arctan':
            return BinaryOp(du, BinaryOp(Constant(1), BinaryOp(u, Constant(2), '^'), '+'), '/')
        elif f == 'ln':
            return BinaryOp(du, u, '/')
        elif f == 'e':
            return BinaryOp(UnaryOp(u, 'e'), du, '*')
        else:
            raise ValueError(f"Unknown function {f}")
        
    def integrate(self):
        """ Integrates the unary operation """
        u = self.operand
        if isinstance(u, Variable):
            if self.func == 'sin':
                return BinaryOp(Constant(-1), UnaryOp(u, 'cos'), '*')  # ∫sin(x) dx = -cos(x)
            elif self.func == 'cos':
                return UnaryOp(u, 'sin')  # ∫cos(x) dx = sin(x)
            elif self.func == 'e':
                return UnaryOp(u, 'e')  # ∫e^x dx = e^x
        raise NotImplementedError(f"Integration for {self.func} not implemented or requires substitution")

    def __str__(self):
        return f"{self.func}({str(self.operand)})"

@helper
def tokenize(expr):
    token_spec = [
        ('NUMBER',   r'\d+(\.\d*)?'),       # Integer or decimal number
        ('FUNC',     r'sin|cos|tan|sec|cot|csc|arcsin|arccos|arctan|ln|e'),  # Functions/constants
        ('ID',       r'x'),                # Variable
        ('OP',       r'[\+\-\*/\^]'),      # Arithmetic operators
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('SKIP',     r'\s+'),              # Skip spaces
        ('MISMATCH', r'.')                 # Any other character
    ]

    token_regex = '|'.join(f'(?P<{name}>{regex})' for name, regex in token_spec)
    scanner = re.compile(token_regex).scanner(expr)

    tokens = []
    for match in iter(scanner.match, None):
        kind = match.lastgroup
        value = match.group()
        if kind == 'NUMBER':
            tokens.append(('NUMBER', float(value)))
        elif kind in {'FUNC', 'ID', 'OP', 'LPAREN', 'RPAREN'}:
            tokens.append((kind, value))
        elif kind == 'SKIP':
            continue
        else:
            raise SyntaxError(f'Unexpected character {value}')
    return tokens

class Parser:
    """
    A simple recursive descent parser for mathematical expressions.

    It uses the Shunting Yard algorithm to convert infix expressions to an expression tree.
    The parser supports the following operations:
    - Addition (+)
    - Subtraction (-)
    - Multiplication (*)
    - Division (/)
    - Exponentiation (^)
    - Functions (sin, cos, tan, sec, cot, csc, arcsin, arccos, arctan, ln, e)
    - Parentheses for grouping expressions
    - Variables (x)
    - Constants (numbers)
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        """ Returns the current token without consuming it """
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def consume(self, expected_type=None):
        """ Consumes the current token and returns it """
        current = self.peek()
        if expected_type and current[0] != expected_type:
            raise SyntaxError(f'Expected {expected_type} but got {current[0]}')
        self.pos += 1
        return current

    def parse(self):
        """ Parses the entire expression and returns the root of the expression tree """
        return self.parse_expression()

    def parse_expression(self):
        """ Parses an expression and returns the root of the expression tree """
        node = self.parse_term()
        while self.peek()[1] in ('+', '-'):
            op = self.consume()[1]
            right = self.parse_term()
            node = BinaryOp(node, right, op)
        return node

    def parse_term(self):
        """ Parses a term and returns the root of the expression tree """
        node = self.parse_factor()
        while self.peek()[1] in ('*', '/'):
            op = self.consume()[1]
            right = self.parse_factor()
            node = BinaryOp(node, right, op)
        return node

    def parse_factor(self):
        """ Parses a factor and returns the root of the expression tree """
        node = self.parse_unary()
        while self.peek()[1] == '^':
            op = self.consume()[1]
            right = self.parse_unary()
            node = BinaryOp(node, right, op)
        return node

    def parse_unary(self):
        """ Parses a unary operation and returns the root of the expression tree """
        token_type, token_value = self.peek()
        if token_type == 'FUNC':
            func = self.consume()[1]
            self.consume('LPAREN')
            arg = self.parse_expression()
            self.consume('RPAREN')
            return UnaryOp(arg, func)
        else:
            return self.parse_primary()

    def parse_primary(self):
        """ Parses a primary expression and returns the root of the expression tree """
        token_type, token_value = self.peek()
        if token_type == 'NUMBER':
            return Constant(self.consume()[1])
        elif token_type == 'ID':
            self.consume()
            return Variable()
        elif token_type == 'LPAREN':
            self.consume()
            node = self.parse_expression()
            self.consume('RPAREN')
            return node
        else:
            raise SyntaxError(f"Unexpected token: {token_value}")

class Function:
    """
    Class to represent a mathematical function.

    Input:
        sin
        cos
        tan
        sec
        cot
        csc
        arcsin
        arccos
        arctan
        ln
        e
        x (variable)
        +, -, *, /, ^ (operators)
        ( ) (parentheses for grouping)
        Constants (numbers)
    
    Note:
        cofficents NEED mutiplication
    
    Supported operations:
        - Evaluate the function at a given point
        - Differentiate the function
        - Integrate the function (not fully implemented)
        - Simplify the function
        - Bounded integral (definite integral between two points)
        - Convert constants to fractions in the string representation
    """
    def __init__(self, expr: str):
        self.expr = expr
        tokens = tokenize(expr)
        self.tree = Parser(tokens).parse()

    def evaluate(self, x: float) -> float:
        """ Evaluates the function at a given point x """
        return self.tree.evaluate(x)
    
    def derivative(self):
        """ Returns the derivative of the function """
        derived_tree = self.tree.differentiate()
        return Function(str(derived_tree))
    
    def integral(self):
        """ Returns the integral of the function """
        # NOTE: This is a placeholder for the integration logic
        try:
            integrated_tree = self.tree.integrate()
            return Function(str(integrated_tree))
        except NotImplementedError as e:
            print(f"Integration not implemented for this form: {e}")
            return None

    def simplify(self):
        """ Simplifies the function """
        simplified_tree = self.tree.simplify()
        return Function(str(simplified_tree))
    
    def bonded_integral(self, a: float, b: float) -> float:
        """ Returns the bounded integral of the function between a and b """
        F = self.integral()
        return F.evaluate(b) - F.evaluate(a) if F else None
    
    def plot(self, x_range: tuple, num_points: int = 1000):
        """ Plots the function over a specified range """
        x_values = [x_range[0] + i * (x_range[1] - x_range[0]) / num_points for i in range(num_points)]
        y_values = [self.evaluate(x) for x in x_values]

        plt.plot(x_values, y_values)
        plt.title(f"Plot of {self.expr}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid()
        plt.show()

    def __call__(self, x: float) -> float:
        return self.evaluate(x)
    
    def __str__(self):
        # Simplify the expression tree before printing it
        simplified_tree = self.tree.simplify()

        # Helper function to traverse the tree and convert decimals to fractions
        def convert_constants_to_fractions(node):
            if isinstance(node, Constant):
                # Convert the constant value to a fraction
                return Constant(convert_frac(node.value))
            elif isinstance(node, BinaryOp):
                # Recursively process left and right children
                left = convert_constants_to_fractions(node.left)
                right = convert_constants_to_fractions(node.right)
                return BinaryOp(left, right, node.op)
            elif isinstance(node, UnaryOp):
                # Recursively process the operand
                operand = convert_constants_to_fractions(node.operand)
                return UnaryOp(operand, node.func)
            else:
                return node  # Return the node as is for Variable or other types

        # Apply the conversion to the simplified tree
        converted_tree = convert_constants_to_fractions(simplified_tree)
        return str(converted_tree)
