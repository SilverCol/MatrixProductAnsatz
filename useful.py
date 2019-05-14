

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
    """ Calculates the matrix product ansatz for a spin chain state psi. """

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


import scipy.sparse as sp

def kronH(N, periodic):
    """ Constructs heisenberg spin chain Hamiltonian using Kronecker products. """

    # Initialize constants
    DIM = 2**N
    S2 = sp.csc_matrix([
        [1,  0,  0, 0],
        [0, -1,  2, 0],
        [0,  2, -1, 0],
        [0,  0,  0, 1]
        ])

    # Hamiltonian matrix
    H = sp.csc_matrix((DIM, DIM))
    for j in range(1, N):
        print('Term %d/%d' % (j, N), end='\r')
        H += sp.kron( sp.kron(sp.identity(2**(j-1)), S2), sp.identity(2**(N - j - 1)) )
    print()

    if periodic:
        Sx = sp.csc_matrix([
            [0, 1],
            [1, 0]
            ])
        H += sp.kron( sp.kron(Sx, sp.identity(2**(N-2))), Sx )

        Sy = sp.csc_matrix([
            [0, -1j],
            [1j,  0]
            ])
        H += np.real( sp.kron( sp.kron(Sy,sp.identity(2**(N-2))), Sy ) )

        Sz = sp.csc_matrix([
            [1,  0],
            [0, -1]
            ])
        H += sp.kron( sp.kron(Sz, sp.identity(2**(N-2))), Sz )
    return H


def idxH(N, periodic):
    """ Constructs Hamiltonian matrix row by row. Do not use it - extremely inefficient. """
    # TODO: This function doesnt work properly!
    
    # Problem size
    DIM = 2**N
    
    # Hamiltonian matrix
    print('Constructing Hamiltonian matrix...')
    H = sp.csc_matrix((DIM, DIM))
    
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
 

from scipy.sparse.linalg import eigsh

def gstate(N, periodic):
    """ Creates a spin chain ground state, saves it to a file. """

    # Create Hamiltonian matrix
    H = kronH(N, periodic)
   
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


from numpy.random import normal

def mpaEntropy(N, ground, periodic):

    DIM = 2**N
    # Initialize quantum state
    if not ground:
        psi = np.array([ [normal()] for n in range(DIM) ])
        psi /= la.norm(psi)
    elif periodic: psi = np.load('data/groundState' + str(N) + 'p.npy')
    else: psi = np.load('data/groundState' + str(N) + '.npy')
    initial = psi
    
    # Cacluate the MPA
    print('Calculating MPA...')
    A, lambdas = mpa(psi)
    
    # Reconstruct initial state for error estimation
    #print('Reconstructing initial state...', end=' ', flush=True)
    #psi = np.array([
    #    la.multi_dot([A[n][b] for n, b in enumerate(bits(s, N))])
    #    for s in range(DIM) ])
    #print('Done')
    #print('Max. error: %.2e' % np.amax(psi - initial))
    
    # Calculate entropy
    return [-2 * np.sum(a**2 * np.log(a)) for a in lambdas]


def svdEntropy(N, frag, periodic):

    # Initialize quantum state and the bipartition
    if periodic: psi = np.load('data/groundState' + str(N) + 'p.npy')
    else: psi = np.load('data/groundState' + str(N) + '.npy')
    bpart = np.array([int(n/frag) % 2 == 0 for n in range(N)])
    
    # Measure partition sizes
    A = 0
    for spot in bpart:
        if spot: A += 1
    B = N - A
    print('Chain divided: (%d) -> (%d, %d)' % (N, A, B))
    print(''.join(['A' if spot else 'B' for spot in bpart]))
    
    # Set partition Hilbert dimensions
    dimA = 2**A
    dimB = 2**B
    
    # Build work matrix
    print('Constructing work matrix...')
    wmatrix = np.empty((dimA, dimB))
    for i in range(dimA):
        print('Row %d/%d' % (i, dimA), end='\r')
        for j in range(dimB):
            a = bits(i, A)
            b = bits(j, B)
            k = 0
    
            for part in bpart:
                k = k << 1
                if part: k += a.__next__()
                else: k += b.__next__()
    
            wmatrix[i, j] = psi[k]
    print()
    
    # Make SVD
    print('Creating SVD...', end=' ', flush=True)
    u, d, v = la.svd(wmatrix, False)
    print('Done')
    
    return -2 * np.sum(d**2 * np.log(d))
