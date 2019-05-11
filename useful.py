def bits(n, N):
    """ Yields the N bit sequence of an integer n. """
    spot = 1 << (N-1)
    for i in range(N):
        yield (n & spot) >> (N-1)
        n = n << 1

import numpy as np
import numpy.linalg as la
from numpy.random import normal

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
        print('%d/%d' % (j, N), end='\r')

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
    print('%d/%d' % (N, N))

    return matrices, coefficients,

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

def gstate(N, periodic):
    """ Creates a spin chain ground state, saves it to a file. """
    
    # Problem size
    DIM = 2**N
    
    # Hamiltonian matrix
    H = lil_matrix((DIM, DIM))
    
    # Hamiltonian construction
    for row in range(DIM):
        print('Row %d/%d' % (row, DIM), end='\r')
        for n, bit in enumerate(bits(row, N)): 
            if n == 0:
                if periodic: first = bit
                b1 = bit
            else:
                if b1 == bit:
                    H[row, row] += 1
                else:
                    H[row, row] -= 1
                    col = row ^ (3 << (n-1))
                    H[row, col] += 2
                b1 = bit
        if periodic:
            if b1 == first:
                H[row, row] += 1
            else:
                H[row, row] -= 1
                col = row ^ (1 << (N-2) + 1)
                H[row, col] += 2
    print()
    
    # Diagonalize
    print('Diagonalizing...', end=' ', flush=True)
    w, v = eigsh(H, k=1, which='SA')
    print('Done')
    
    # Save
    fname = 'data/groundState' + str(N)
    if periodic: fname += 'p'
    fname += '.npy'
    np.save(fname, v)
    print('Output saved to ' + fname)

