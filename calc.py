import fraction
import Prob

print("WARNING!\n\n\nTHIS IS DEPRICATED CODE! \n\n\n USE calculus.py INSTEAD!\n\n\n")

class Function:
    '''
    Requires:
        - polynomial in the form of ax^n + bx^(n-1) + ... + c

    Supports:
        - Derrivative
        - Integral
        - Evaluation at a point
        - String representation
    
    '''
    def __init__(self, function, isIntegral=False):
        self.isIntegral = isIntegral
        if isinstance(function, str):
            if '=' in function:
                raise ValueError("Function should not contain an equal sign")
            function = function.replace(' ', '')
            function = function.replace("-", "+-")
            
            # if "+" not in parthesis split by "+"
            if "(" in function:
                for i in range(len(function)):
                    if function[i] == "(":
                        start = i
                    if function[i] == ")":
                        end = i
                        function = function[:start] + function[start:end+1].replace("+", "p") + function[end+1:]
                function = function.split("+")
                for i in range(len(function)):
                    function[i] = function[i].replace("p", "+")
            else:
                function = function.split("+")

            function = [i for i in function if i != '']

            for i in range(len(function)):
                if "^" not in function[i] and "x" in function[i]:
                    function[i] = function[i].replace("x", "x^1")

            self.coefficients = []
            for i in function:
                if "x" in i:
                    if "^" in i:
                        start = i.index("^") + 1
                        if i[start] == "(":
                            start += 1
                            end = i.index(")")
                            exponent = eval(i[start:end])
                        else:
                            exponent = eval(i[start:])
                    else:
                        exponent = 1
                    coefficient_part = i[:i.index("x")]
                    coefficient = int(coefficient_part) if coefficient_part else 1
                else:
                    exponent = 0
                    coefficient = int(i)
                self.coefficients.append((exponent, coefficient))
        else:
            self.coefficients = function

    def getCoefficients(self):
        """
        Returns:
            the coefficients of the function (deg, coefficient) as a list
        """
        return self.coefficients
    
    def derrivative(self):
        """
        Returns:
            the derrivative of the function
        """
        derrivative = []
        for i in self.coefficients:
            if i[0] == 0:
                continue
            derrivative.append((i[0]-1, i[0]*i[1]))
        return Function(derrivative)
    
    def of(self, x):
        """
        Args:
            x: the value to evaluate the function at
        Returns:
            the value of the function at x
        """
        value = 0
        for i in self.coefficients:
            value += i[1] * (x ** i[0])
        return value
    
    def integral(self):
        """
        Returns:
            the integral of the function
        """
        integral = []
        for i in self.coefficients:
            integral.append((i[0]+1, i[1]/(i[0]+1)))
        return Function(integral, isIntegral=True)
    
    def taylor(self, nDegree, around=0):
        """
        Args:
            nDegree: the degree of the taylor polynomial
            around: the point to evaluate the taylor polynomial at
        Returns:
            the taylor polynomial of the function at the given point
        """
        taylor = []
        current_function = self  # Start with the original function
        for i in range(nDegree + 1):
            # Evaluate the current derivative at the point 'around'
            coefficient = current_function.of(around) / Prob.factorial(i)
            taylor.append((i, coefficient))
            # Compute the next derivative
            current_function = current_function.derrivative()
        return Function(taylor)

    def __str__(self):
        terms = []
        for i in self.coefficients:
            coefficient = fraction.convert_frac(i[1])
            exponent = fraction.convert_frac(i[0])
            
            if i[0] == 0:
                terms.append(f"{coefficient}")
            elif i[0] == 1:
                if i[1] == 1:
                    terms.append(f"x")
                elif i[1] == -1:
                    terms.append(f"-x")
                else:
                    terms.append(f"{coefficient}x")
            else:
                if i[1] == 1:
                    terms.append(f"x^{exponent}")
                elif i[1] == -1:
                    terms.append(f"-x^{exponent}")
                else:
                    terms.append(f"{coefficient}x^{exponent}")
        return " + ".join(terms).replace("+ -", "- ") + (" + C" if self.isIntegral else "") 

