import numpy as np
import numpy.linalg as la
from numpy.random import normal

def bits(n, N):
    """ Yields the N bit sequence of an integer n. """
    spot = 1 << (N-1)
    for i in range(N):
        yield (n & spot) >> (N-1)
        n = n << 1

def mpa(psi):
    """ Calculates the matrix product ansatz
    for a spin chain state psi. """

    DIM = psi.size
    N = DIM.bit_length() - 1

    # Create containers
    coefficients = []
    matrices = []
    
    # Do the MPA
    M = 1
    for j in range(1, N):
    
        # Make SVD
        psi = psi.reshape((2*M), 2**(N-j))
        u, d, v = la.svd(psi, full_matrices=False)
    
        # Store results
        coefficients.append(d)
        matrices.append([u[::2], u[1::2]])
    
        # Prepare for next iteration
        psi = np.matmul(np.diag(d), v)
        M = d.size
    # Final MPA step
    matrices.append([psi[:, 0], psi[:, 1]])

    return matrices, coefficients,
