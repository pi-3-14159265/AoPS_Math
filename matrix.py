import math
import fraction

class Vector_2:
    """
    Represents a 2D vector

    Supports:
    - Magnitude
    - Norm
    - Normalization
    - Dot product
    - Angle between vectors
    - Addition
    - Subtraction
    - Vector multiplication
    - String representation
    - Exponentiation

    """
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        """
        Returns the magnitude of the vector (the length of the vector)

        Returns:
            float: the magnitude of the vector
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def norm(self) -> float:
        """
        Returns the norm of the vector (the length of the vector)

        Returns:
            float: the magnitude of the vector
        """
        return self.magnitude()
    
    def normalize(self) -> 'Vector_2':
        """
        Returns the normalized vector (the vector with the same direction but with magnitude 1)

        Returns:
            Vector_2: the normalized vector
        """
        mag = self.magnitude()
        return Vector_2(self.x / mag, self.y / mag)
    
    def dot(self, other):
        """
        Returns the dot product of the vector and another vector

        Args:
            other (Vector_2): the other vector

        Returns:
            float: the dot product of the two vectors
        """
        return self.x * other.x + self.y * other.y
    
    def angle(self, other: 'Vector_2') -> float:
        """
        Returns the angle between the vector and another vector in radians

        Args:
            other (Vector_2): the other vector
        
        Returns:
            float: the angle between the two vectors in radians
        """
        return math.acos(self.dot(other) / (self.magnitude() * other.magnitude()))
    
    def __add__(self, other: 'Vector_2') -> 'Vector_2':
        return Vector_2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector_2') -> 'Vector_2':
        return Vector_2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector_2':
        return Vector_2(self.x * scalar, self.y * scalar)
    
    def __str__(self) -> str:
        return f'({fraction.convert(self.x)}; {fraction.convert(self.y)})'
    
    def __pow__(self, power: float) -> 'Vector_2':
        return Vector_2(self.x ** power, self.y ** power)
    

class Matrix_2x2:
    """
    2x2 matrix class
    [a , b
     c , d]

    Supports:
    - Determinant
    - Inverse
    - Multiplication by vector
    - Addition
    - Subtraction
    - Multiplication by matrix NOT commutative
    - Exponentiation
    """
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def determinant(self) -> float:
        """
        Returns the determinant of the matrix
        """
        return self.a * self.d - self.b * self.c
    
    def inverse(self) -> 'Matrix_2x2':
        """
        Returns the inverse of the matrix

        Raises:
            ValueError: if the matrix is not invertible
        """
        det = self.determinant()
        if det == 0:
            raise ValueError('The matrix is not invertible')
        return Matrix_2x2(self.d / det, -self.b / det, -self.c / det, self.a / det)
    
    def multiply_vector(self, vector: 'Vector_2') -> 'Vector_2':
        """
        Returns the product of the matrix and the vector

        Args:
            vector (Vector_2): the vector to multiply by

        Returns:
            Vector_2: the resulting vector
        """
        return Vector_2(self.a * vector.x + self.b * vector.y, self.c * vector.x + self.d * vector.y)
    
    def __mul__(self, other: 'Matrix_2x2') -> 'Matrix_2x2':
        return Matrix_2x2(self.a * other.a + self.b * other.c, self.a * other.b + self.b * other.d,
                          self.c * other.a + self.d * other.c, self.c * other.b + self.d * other.d)
    
    def __add__(self, other: 'Matrix_2x2') -> 'Matrix_2x2':
        return Matrix_2x2(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
    
    def __sub__(self, other: 'Matrix_2x2') -> 'Matrix_2x2':
        return Matrix_2x2(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)
    
    def __str__(self) -> str:
        return f'[{fraction.convert(self.a)}, {fraction.convert(self.b)}]\n[{fraction.convert(self.c)}, {fraction.convert(self.d)}]'
    
    def __pow__(self, power: int) -> 'Matrix_2x2':
        result = Matrix_2x2(1, 0, 0, 1)  # Identity matrix
        for _ in range(power):
            result = result * self
        return result
        

class Vector_3:
    """
    Represents a 3D vector

    Supports:
    - Magnitude
    - Norm
    - Normalization
    - Dot product
    - Cross product
    - Angle between vectors
    - Addition
    - Subtraction
    - Vector multiplication
    - String representation
    - Exponentiation

    """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self) -> float:
        """
        Returns the magnitude of the vector (the length of the vector)

        Returns:
            float: the magnitude of the vector
        """
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def norm(self) -> float:
        """
        Returns the norm of the vector (the length of the vector)

        Returns:
            float: the magnitude of the vector
        """
        return self.magnitude()
    
    def normalize(self) -> 'Vector_3':
        """
        Returns the normalized vector (the vector with the same direction but with magnitude 1)

        Returns:
            Vector_3: the normalized vector
        """
        mag = self.magnitude()
        return Vector_3(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector_3') -> float:
        """
        Returns the dot product of the vector and another vector

        Args:
            other (Vector_3): the other vector

        Returns:
            float: the dot product of the two vectors
        """
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector_3') -> 'Vector_3':
        """
        Returns the cross product of the vector and another vector

        Args:
            other (Vector_3): the other vector
        
        Returns:
            Vector_3: the cross product of the two vectors
        """
        return Vector_3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    
    def angle(self, other: 'Vector_3') -> float:
        """
        Returns the angle between the vector and another vector in radians

        Args:
            other (Vector_3): the other vector
        
        Returns:
            float: the angle between the two vectors in radians
        
        """
        return math.acos(self.dot(other) / (self.magnitude() * other.magnitude()))
    
    def __add__(self, other: 'Vector_3') -> 'Vector_3':
        return Vector_3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector_3') -> 'Vector_3':
        return Vector_3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, vector: 'Vector_3') -> 'Vector_3':
        return Vector_3(self.x * vector.x, self.y * vector.y, self.z * vector.z)
    
    def __str__(self) -> str:
        return f'({fraction.convert(self.x)}; {fraction.convert(self.y)}; {fraction.convert(self.z)})'
    
    def __pow__(self, power: float) -> 'Vector_3':
        return Vector_3(self.x ** power, self.y ** power, self.z ** power)
    

class Matrix_3x3():
    """
    3x3 matrix class
    [a , b , c
     d , e , f
     g , h , i]

    Supports:
    - Determinant
    - Inverse
    - Multiplication by vector
    - Addition
    - Subtraction
    - Multiplication by matrix NOT commutative
    - Exponentiation
    """
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, i: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.i = i

    def determinant(self) -> float:
        """
        Returns the determinant of the matrix
        """
        return self.a * (self.e * self.i - self.f * self.h) - self.b * (self.d * self.i - self.f * self.g) + self.c * (self.d * self.h - self.e * self.g)
    
    def inverse(self) -> 'Matrix_3x3':
        """
        Returns the inverse of the matrix

        Raises:
            ValueError: if the matrix is not invertible
        
        Returns:
            Matrix_3x3: the inverse of the matrix
        """
        det = self.determinant()
        if det == 0:
            raise ValueError('The matrix is not invertible')
        return Matrix_3x3((self.e * self.i - self.f * self.h) / det, -(self.b * self.i - self.c * self.h) / det, (self.b * self.f - self.c * self.e) / det,
                          -(self.d * self.i - self.f * self.g) / det, (self.a * self.i - self.c * self.g) / det, -(self.a * self.f - self.c * self.d) / det,
                          (self.d * self.h - self.e * self.g) / det, -(self.a * self.h - self.b * self.g) / det, (self.a * self.e - self.b * self.d) / det)
    
    def multiply_vector(self, vector: 'Vector_3') -> 'Vector_3':
        """
        Returns the product of the matrix and the vector

        Args:
            vector (Vector_3): the vector to multiply by

        Returns:
            Vector_3: the resulting vector
        """
        return Vector_3(self.a * vector.x + self.b * vector.y + self.c * vector.z, self.d * vector.x + self.e * vector.y + self.f * vector.z, self.g * vector.x + self.h * vector.y + self.i * vector.z)
    
    def __mul__(self, other: 'Matrix_3x3') -> 'Matrix_3x3':
        return Matrix_3x3(self.a * other.a + self.b * other.d + self.c * other.g, self.a * other.b + self.b * other.e + self.c * other.h, self.a * other.c + self.b * other.f + self.c * other.i,
                          self.d * other.a + self.e * other.d + self.f * other.g, self.d * other.b + self.e * other.e + self.f * other.h, self.d * other.c + self.e * other.f + self.f * other.i,
                          self.g * other.a + self.h * other.d + self.i * other.g, self.g * other.b + self.h * other.e + self.i * other.h, self.g * other.c + self.h * other.f + self.i * other.i)
    
    def __add__(self, other: 'Matrix_3x3') -> 'Matrix_3x3':
        return Matrix_3x3(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d, self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h, self.i + other.i)
    
    def __sub__(self, other: 'Matrix_3x3') -> 'Matrix_3x3':
        return Matrix_3x3(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d, self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h, self.i - other.i)
    
    def __str__(self) -> str:
        return f'[{fraction.convert(self.a)}, {fraction.convert(self.b)}, {fraction.convert(self.c)}]\n[{fraction.convert(self.d)}, {fraction.convert(self.e)}, {fraction.convert(self.f)}]\n[{fraction.convert(self.g)}, {fraction.convert(self.h)}, {fraction.convert(self.i)}]'
    
    def __pow__(self, power: int) -> 'Matrix_3x3':
        result = Matrix_2x2(1, 0, 0, 1)  # Identity matrix
        for _ in range(power):
            result = result * self
        return result
    
def projectionMatrix(to_this_vector: 'Vector_2') -> 'Matrix_2x2':
    """
    Returns the projection matrix for the given vector

    Args:
        to_this_vector (Vector_2): the vector to project onto

    Returns:
        Matrix_2x2: the projection matrix
    """
    to_this_vector = to_this_vector.normalize()
    return Matrix_2x2(to_this_vector.x ** 2, to_this_vector.x * to_this_vector.y, 
                      to_this_vector.x * to_this_vector.y, to_this_vector.y ** 2)

def reflectionMatrix(over_this_vector: 'Vector_2') -> 'Matrix_2x2':
    """
    Returns the reflection matrix for the given vector

    Args:
        over_this_vector (Vector_2): the vector to reflect over

    Returns:
        Matrix_2x2: the reflection matrix
    """
    over_this_vector = over_this_vector.normalize()
    return Matrix_2x2(2 * over_this_vector.x ** 2 - 1, 2 * over_this_vector.x * over_this_vector.y, 
                      2 * over_this_vector.x * over_this_vector.y, 2 * over_this_vector.y ** 2 - 1)

# angle in degrees
def rotationMatrix_degree(angle: float) -> 'Matrix_2x2':
    """
    Returns the rotation matrix for the given angle in degrees GOING COUNTERCLOCKWISE
    for clockwise rotation, use -angle

    Args:
        angle (float): the angle in degrees

    Returns:
        Matrix_2x2: the rotation matrix
    """
    angle = math.radians(angle)
    return Matrix_2x2(math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle))

# angle in radians
def rotationMatrix_radian(angle: float) -> 'Matrix_2x2':
    """
    Returns the rotation matrix for the given angle in radians GOING COUNTERCLOCKWISE
    for clockwise rotation, use -angle

    Args:
        angle (float): the angle in radians

    Returns:
        Matrix_2x2: the rotation matrix
    """
    return Matrix_2x2(math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle))

def dilatationMatrix(k: float) -> 'Matrix_2x2':
    """
    Returns the dilatation matrix for the given factor k

    Args:
        k (float): the factor

    Returns:
        Matrix_2x2: the dilatation matrix
    """
    return Matrix_2x2(k, 0, 0, k)
