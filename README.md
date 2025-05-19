Its kinda a pet of mine to write coded sloutions to AoPS problems and started making a library of methouds theres alot here feel free to use.

**System.py**
*Linear Equation*
Takes `a`, `b`, and `c` such that equation is in the form ax+b=c methods:
  solve() 

*BiLinear Equation*
Takes `a`, `b`, and `c` such that ax+by=c methods:
  solve(other)

  --------------------------------------
  **Matrix.py**
  *Vector_2*
  Takes `a` and `b` represents <a,b> methods
    magnitude()
    norm()
    normalize() returns a normal vector
    dot(other) dot product
    angel(other) the angel between two vectors
    Sum of vectors with +
    Diffrence of vectors with -
    Scaling vectors with *
    Scaling with exponets with ^

*Matrix_2x2*
Takes `a`, `b`, `c` and `d` such that
[a,b
 c,d]

Supports:
  determinant()
  inverse()
  multiply_vector(Vector_2)
  Sum of Matrices with +
  Diffrence of Marices with -
  Product of Matrices with * (NOT COMMUNATIVE)
  Scaling with exponets with ^


  
