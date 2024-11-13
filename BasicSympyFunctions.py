import re
import sympy

def DefineMatrix(name, m, n):
    """
    This method defines a symbolic matrix.

    Keyword arguments:
    - name -- Name of variables.
    - m -- Number of rows.
    - n -- Number of columns.
    """
    return sympy.Matrix(m, n, lambda i, j: sympy.var("{name}_{i}_{j}".format(name=name, i=i, j=j)))

def DefineVector( name, m):
    """
    This method defines a symbolic vector.

    Keyword arguments:
    - name -- Name of variables.
    - m -- Number of components.
    """
    return sympy.Matrix(m, 1, lambda i,_: sympy.var("{name}_{i}".format(name=name, i=i)))

def DefineShapeFunctions(nnodes, dim, impose_partion_of_unity=False):
    """
    This method defines shape functions and derivatives.
    Note that partition of unity is imposed
    the name HAS TO BE --> N and DN

    Keyword arguments:
    - nnodes -- Number of nodes
    - dim -- Dimension of the space
    - impose_partion_of_unity -- Impose the partition of unity
    """
    DN = DefineMatrix('DN', nnodes, dim)
    N = DefineVector('N', nnodes)

    #impose partition of unity
    if impose_partion_of_unity:
        N[nnodes-1] = 1
        for i in range(nnodes-1):
            N[nnodes-1] -= N[i]

        DN[nnodes-1,:] = -DN[0,:]
        for i in range(1,nnodes-1):
            DN[nnodes-1,:] -= DN[i,:]

    return N, DN